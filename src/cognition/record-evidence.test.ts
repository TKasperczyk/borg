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

describe("live retrieval record annotations", () => {
  let db: SqliteDatabase | undefined;
  afterEach(() => {
    db?.close();
    db = undefined;
  });
  it.each(["episode", "semantic_node", "semantic_edge"] as const)(
    "carries the warm %s's own correction through unified and verification retrieval",
    (recordType) => {
      db = openDatabase(":memory:", { migrations: identityMigrations });
      const identityEventRepository = new IdentityEventRepository({ db });
      const fixture = createWarmRecordEvidenceFixture();
      for (const record of fixture.records)
        recordTestAnnotation(identityEventRepository, record.target.record_id, record.correction, {
          recordType: record.target.record_type,
        });
      const projected = withRetrievedRecordEvidence(
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
  it("keeps valid retrieval corrections and other targets when stored annotation history is malformed", () => {
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
    const projected = withRetrievedRecordEvidence(
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
  it("carries exact annotations through legacy, additional and verification projections including legacy semantic buckets", () => {
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
    const projected = withRetrievedRecordEvidence(
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
});
