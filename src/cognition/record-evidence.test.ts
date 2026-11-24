import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import { identityMigrations } from "../memory/identity/migrations.js";
import { IdentityEventRepository } from "../memory/identity/repository.js";
import {
  recordTestAnnotation,
  recordTestAnnotationWithMalformedHistory,
} from "../offline/evidence-test-support.js";
import {
  createRecordEvidenceFixture,
  createWarmRecordEvidenceFixture,
} from "./record-evidence-test-support.js";
import { withRetrievedRecordEvidence } from "./record-evidence.js";
import {
  summarizeRetrievedEvidence,
  renderPlanRequestedVerificationRetrieval,
  summarizeSemanticContext,
} from "./deliberation/prompt/retrieval.js";
import { RECORD_ANNOTATION_GUIDANCE } from "../memory/common/record-evidence.js";
import { SELF_CORRECTION_GUIDANCE } from "../memory/common/evidence-prompt.js";

describe("live retrieval record annotations", () => {
  let db: SqliteDatabase | undefined;
  afterEach(() => {
    db?.close();
    db = undefined;
  });
  it.each(["episode", "semantic_node", "semantic_edge"] as const)(
    "carries the warm %s's own correction through unified and verification retrieval",
    async (recordType) => {
      db = openDatabase(":memory:", { migrations: identityMigrations });
      const identityEventRepository = new IdentityEventRepository({ db });
      const fixture = createWarmRecordEvidenceFixture();
      for (const record of fixture.records)
        recordTestAnnotation(identityEventRepository, record.target.record_id, record.correction, {
          recordType: record.target.record_type,
        });
      const projected = await withRetrievedRecordEvidence(
        { identityEventRepository },
        { evidence: fixture.evidence },
      );
      const record = fixture.records.find((row) => row.target.record_type === recordType)!;
      const row = projected.evidence.find((item) => item.id === record.evidence.id)!;
      expect(row.evidence_context?.annotations?.map((annotation) => annotation.text)).toEqual([
        record.correction,
      ]);
      expect(row.text).toBe(record.evidence.text);
      for (const untouched of fixture.unrelatedEvidence)
        expect(projected.evidence.find((item) => item.id === untouched.id)).not.toHaveProperty(
          "evidence_context",
        );
      for (const prompt of [
        summarizeRetrievedEvidence("Recall", { evidence: [row] }),
        renderPlanRequestedVerificationRetrieval(
          {
            evidence: [row],
            episodes: [],
            semantic: fixture.input.retrievedSemantic,
            open_questions: [],
          },
          10_000,
          10_000,
          { rowsTotalReadAtMs: 1_000 },
        ),
      ]) {
        expect(prompt).toContain(record.correction);
        for (const other of fixture.records.filter((other) => other !== record))
          expect(prompt).not.toContain(other.correction);
      }
    },
  );
  it("keeps valid retrieval corrections and other targets when stored annotation history is malformed", async () => {
    db = openDatabase(":memory:", { migrations: identityMigrations });
    const emit = vi.fn();
    const identityEventRepository = new IdentityEventRepository({
      db,
      tracer: { enabled: true, includePayloads: false, emit },
    });
    const fixture = createRecordEvidenceFixture();
    const correction = recordTestAnnotationWithMalformedHistory(
      { db, identityEventRepository },
      fixture.episodes[0]!.episode.id,
    );
    recordTestAnnotation(identityEventRepository, fixture.node.id, "Other target correction", {
      recordType: "semantic_node",
    });
    const projected = await withRetrievedRecordEvidence(
      { identityEventRepository },
      { ...fixture, open_questions: [] },
    );
    expect(projected.evidence[0]!.evidence_context?.annotations).toEqual([
      { text: correction, authority: "Synthetic operator" },
    ]);
    expect(projected.evidence[1]!.evidence_context?.annotations?.[0]?.text).toBe(
      "Other target correction",
    );
    expect(projected.evidence.map((item) => item.text)).toEqual(
      fixture.evidence.map((item) => item.text),
    );
    for (const prompt of [
      summarizeRetrievedEvidence("Recall", projected),
      renderPlanRequestedVerificationRetrieval(projected, 10_000, 10_000, {
        rowsTotalReadAtMs: 1_000,
      }),
    ]) {
      expect(prompt).toContain(correction);
      expect(prompt).toContain("Other target correction");
    }
    expect(emit).toHaveBeenCalledExactlyOnceWith("memory.annotations.skipped", {
      turnId: "memory-annotation-read",
      counts_by_reason: {
        invalid_timestamp: 1,
        invalid_json: 1,
        missing_text: 1,
        invalid_payload: 1,
      },
    });
  });
  it("carries exact annotations through legacy, additional and verification projections including legacy semantic buckets", async () => {
    db = openDatabase(":memory:", { migrations: identityMigrations });
    const identityEventRepository = new IdentityEventRepository({ db });
    const original = createRecordEvidenceFixture();
    const episodeCorrection = "Episode 更正. تصحيح. 🧭";
    const nodeCorrection = "Node-only correction. Zażółć.";
    recordTestAnnotation(
      identityEventRepository,
      original.episodes[0]!.episode.id,
      episodeCorrection,
    );
    recordTestAnnotation(identityEventRepository, original.node.id, nodeCorrection, {
      recordType: "semantic_node",
    });
    const projected = await withRetrievedRecordEvidence(
      { identityEventRepository },
      {
        episodes: original.episodes,
        evidence: original.evidence,
        semantic: { ...original.semantic, supports: [original.node] },
        open_questions: [],
      },
    );
    expect(projected.episodes[1]!.episode).not.toHaveProperty("evidence_context");
    expect(original.episodes[0]!.episode).not.toHaveProperty("evidence_context");
    for (const prompt of [
      summarizeRetrievedEvidence("Recall", projected),
      summarizeRetrievedEvidence("Additional", { ...projected, evidence: [] }),
      renderPlanRequestedVerificationRetrieval(projected, 10_000, 10_000, {
        rowsTotalReadAtMs: 1000,
      }),
    ]) {
      expect(prompt).toContain(episodeCorrection);
      expect(prompt).toContain(nodeCorrection);
      expect(prompt!.split(RECORD_ANNOTATION_GUIDANCE)).toHaveLength(2);
    }
    expect(
      summarizeSemanticContext(
        { ...projected.semantic, matched_nodes: [], matched_node_ids: [] },
        3_000,
      ),
    ).toContain(nodeCorrection);
    const omitted = renderPlanRequestedVerificationRetrieval(projected, 1, 10_000, {
      rowsTotalReadAtMs: 1000,
    });
    expect(omitted).toContain('annotations_omitted="1"');
    expect(omitted).not.toContain(episodeCorrection);
    expect(omitted).not.toContain(nodeCorrection);
  });

  it("keeps unified retrieval bytes when the only self-written correction is on an omitted row", () => {
    const { evidence } = createRecordEvidenceFixture();
    const operator = {
      ...evidence[0]!,
      evidence_context: { annotations: [{ text: "Audited note.", authority: null }] },
    };
    const candidate = { ...evidence[1]!, text: "界".repeat(600) };
    const corrected = {
      ...candidate,
      evidence_context: {
        self_correction_count: 1,
        annotations: [{ text: "My provisional note.", authority: null, author: "borg" as const }],
      },
    };
    const tight = summarizeRetrievedEvidence("Recall", { evidence: [operator, candidate] }, 100);
    expect(tight).toContain(RECORD_ANNOTATION_GUIDANCE);
    expect(tight).toContain("Audited note.");
    expect(tight).toContain("- ... truncated");
    expect(summarizeRetrievedEvidence("Recall", { evidence: [operator, corrected] }, 100)).toBe(
      tight,
    );
    const roomy = summarizeRetrievedEvidence("Recall", { evidence: [operator, corrected] }, 1_000);
    expect(roomy).toContain(SELF_CORRECTION_GUIDANCE);
    expect(roomy).toContain("My provisional note.");
    expect(roomy).not.toContain(RECORD_ANNOTATION_GUIDANCE);
  });

  it("queries typed handles and leaves exact text unresolved when recall has only operator annotations", async () => {
    db = openDatabase(":memory:", { migrations: identityMigrations });
    const identityEventRepository = new IdentityEventRepository({ db });
    const fixture = createRecordEvidenceFixture();
    recordTestAnnotation(identityEventRepository, fixture.episodes[0]!.episode.id);
    const targets = vi.spyOn(identityEventRepository, "listAnnotationsForTargets");
    const getEpisodes = vi.fn(async () => []);
    const getNodes = vi.fn(async () => []);
    const untouchedNode = fixture.node;
    const readDescription = vi.fn(() => "Unchanged original description.");
    Object.defineProperty(untouchedNode, "description", { get: readDescription });
    const projected = await withRetrievedRecordEvidence(
      {
        identityEventRepository,
        episodicRepository: { getLifecycleMany: () => new Map(), getMany: getEpisodes },
        semanticNodeRepository: { getMany: getNodes },
      },
      fixture,
    );
    expect(targets.mock.calls[0]?.[0]).toEqual([
      ...fixture.episodes.map(({ episode }) => ({ record_type: "episode", record_id: episode.id })),
      { record_type: "semantic_node", record_id: untouchedNode.id },
      { record_type: "episode", record_id: fixture.episodes[0]!.episode.id },
      { record_type: "semantic_node", record_id: untouchedNode.id },
    ]);
    expect(projected.evidence[0]!.evidence_context?.annotations).toHaveLength(1);
    expect(readDescription).not.toHaveBeenCalled();
    expect(getEpisodes).not.toHaveBeenCalled();
    expect(getNodes).not.toHaveBeenCalled();
  });
});
