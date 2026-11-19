import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase, type SqliteDatabase } from "../../storage/sqlite/index.js";
import { identityMigrations } from "../../memory/identity/migrations.js";
import { IdentityEventRepository } from "../../memory/identity/repository.js";
import {
  recordTestAnnotation,
  recordTestAnnotationWithMalformedHistory,
} from "../../offline/evidence-test-support.js";
import {
  createRecordEvidenceFixture,
  createWarmRecordEvidenceFixture,
  buildRecordEvidenceLedger,
} from "../record-evidence-test-support.js";
import {
  readLedgerRecordEvidence,
  MAX_LEDGER_ANNOTATIONS_PER_RECORD,
  MAX_LEDGER_ANNOTATION_CHARS,
  ledgerAnnotationCount,
  ledgerAnnotations,
} from "./record-annotations.js";
import {
  readRecordEvidenceContexts,
  RECORD_ANNOTATION_GUIDANCE,
  type RecordEvidenceReader,
} from "../../memory/common/record-evidence.js";
import { compactEvidenceLedger } from "./compaction.js";
import { renderEvidenceLedger, estimateEvidenceLedgerPromptTokens } from "./finalizer-ledger.js";
import { renderCompactPlannerLedger } from "./compact-planner.js";
import { renderEntry, renderSection } from "./section-rendering.js";
import { omissionCounts } from "./omission-accounting.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "./types.js";
import { createEpisodeId } from "../../util/ids.js";

const MULTILINGUAL = "Zażółć. 中文更正。 تصحيح. 🧭 e\u0301\nSecond line.";
function ledger(entries: EvidenceLedgerEntry[]): EvidenceLedger {
  return {
    sections: [{ id: "episodes", label: "Episodes", entries }],
    transcriptIncluded: false,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
  };
}
describe("live ledger record annotations", () => {
  let db: SqliteDatabase | undefined;
  afterEach(() => {
    db?.close();
    db = undefined;
  });
  function setup() {
    db = openDatabase(":memory:", { migrations: identityMigrations });
    const identityEventRepository = new IdentityEventRepository({ db });
    const reader: RecordEvidenceReader = {
      identityEventRepository,
      episodicRepository: {
        getLifecycleMany: (ids) => new Map(ids.map((id) => [id, "active" as const])),
      },
    };
    return { identityEventRepository, reader };
  }
  it.each(["episode", "semantic_node", "semantic_edge"] as const)(
    "attaches the warm %s's exact annotation even without fresh recall rows",
    async (recordType) => {
      const { reader, identityEventRepository } = setup();
      const fixture = createWarmRecordEvidenceFixture();
      for (const record of fixture.records)
        recordTestAnnotation(identityEventRepository, record.target.record_id, record.correction, {
          recordType: record.target.record_type,
        });
      const built = await buildRecordEvidenceLedger(fixture.input, reader);
      const entries = built.sections.flatMap((section) => section.entries);
      const record = fixture.records.find((row) => row.target.record_type === recordType)!;
      const entry = entries.find((row) => row.id === `retrieved_evidence:${record.evidence.id}`)!;
      expect(entry.evidence_context?.annotations?.map((annotation) => annotation.text)).toEqual([
        record.correction,
      ]);
      expect(entry.text).toBe(record.evidence.text);
      expect(renderEntry(entry)).toContain(record.correction);
      for (const other of fixture.records.filter((other) => other !== record))
        expect(renderEntry(entry)).not.toContain(other.correction);
      for (const untouched of fixture.unrelatedEvidence)
        expect(
          entries.find((row) => row.id === `retrieved_evidence:${untouched.id}`),
        ).not.toHaveProperty("evidence_context");
    },
  );
  it("builds and renders valid corrections despite malformed history, with one counts-only trace", async () => {
    setup();
    const emit = vi.fn();
    const identityEventRepository = new IdentityEventRepository({
      db: db!,
      tracer: { enabled: true, includePayloads: false, emit },
    });
    const fixture = createRecordEvidenceFixture();
    const correction = recordTestAnnotationWithMalformedHistory(
      { db: db!, identityEventRepository },
      fixture.episodes[0]!.episode.id,
    );
    recordTestAnnotation(identityEventRepository, fixture.node.id, "Other target correction", {
      recordType: "semantic_node",
    });
    const built = await buildRecordEvidenceLedger(fixture.input, { identityEventRepository });
    const entries = built.sections.flatMap((section) => section.entries);
    const episode = entries.find((row) => row.id === `episode:${fixture.episodes[0]!.episode.id}`)!;
    expect(episode.text).toBe(fixture.episodes[0]!.episode.narrative);
    expect(episode.evidence_context?.annotations).toEqual([
      { text: correction, authority: "Synthetic operator" },
    ]);
    expect(
      entries.find((row) => row.id === `semantic_node:${fixture.node.id}`)?.evidence_context
        ?.annotations?.[0]?.text,
    ).toBe("Other target correction");
    expect(renderEvidenceLedger(built)).toContain(correction);
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
  it("places exact episode and node corrections beside every live page projection without inheritance", async () => {
    const { reader, identityEventRepository } = setup();
    const { input, episodes, node } = createRecordEvidenceFixture();
    recordTestAnnotation(identityEventRepository, episodes[0]!.episode.id, MULTILINGUAL);
    recordTestAnnotation(identityEventRepository, node.id, "Node-only 更正", {
      recordType: "semantic_node",
    });
    const built = await buildRecordEvidenceLedger(input, reader);
    const all = built.sections.flatMap((section) => section.entries);
    for (const id of [
      `episode:${episodes[0]!.episode.id}`,
      "retrieved_evidence:synthetic-episode-evidence",
      "autobiographical_recall:synthetic-autobiographical-episode",
    ])
      expect(all.find((entry) => entry.id === id)?.evidence_context?.annotations?.[0]?.text).toBe(
        MULTILINGUAL,
      );
    for (const id of [`semantic_node:${node.id}`, "retrieved_evidence:synthetic-node-evidence"])
      expect(all.find((entry) => entry.id === id)?.evidence_context).toMatchObject({
        annotations: [{ text: "Node-only 更正" }],
      });
    expect(
      all.find((entry) => entry.id === `episode:${episodes[1]!.episode.id}`),
    ).not.toHaveProperty("evidence_context");
    expect(all.find((entry) => entry.id === `episode:${episodes[0]!.episode.id}`)?.text).toBe(
      episodes[0]!.episode.narrative,
    );
    expect(renderEvidenceLedger(built)?.split(RECORD_ANNOTATION_GUIDANCE)).toHaveLength(2);
    expect(
      Buffer.from(
        all.find((entry) => entry.id === `episode:${episodes[0]!.episode.id}`)!.evidence_context!
          .annotations![0]!.text,
      ),
    ).toEqual(Buffer.from(MULTILINGUAL));
  });
  it("selects at most two complete annotations of at most 600 characters, newest first, and counts the rest", () => {
    const { reader, identityEventRepository } = setup();
    const { input, episodes } = createRecordEvidenceFixture();
    for (const [index, text] of ["old", MULTILINGUAL, "界".repeat(600), "x".repeat(601)].entries())
      recordTestAnnotation(identityEventRepository, episodes[0]!.episode.id, text, { ts: index });
    const contexts = readLedgerRecordEvidence(reader, input);
    const context = contexts.get(`episode:${episodes[0]!.episode.id}`)!;
    expect(context.annotations).toHaveLength(MAX_LEDGER_ANNOTATIONS_PER_RECORD);
    expect(context.annotations!.map((row) => row.text)).toEqual([
      "界".repeat(MAX_LEDGER_ANNOTATION_CHARS),
      MULTILINGUAL,
    ]);
    expect(context.annotations_omitted).toBe(2);
    expect(
      readRecordEvidenceContexts(reader, [
        { record_type: "semantic_node", record_id: episodes[0]!.episode.id },
      ]),
    ).toEqual(new Map());
  });
  it.each([500, 800, 1_400, 2_500])(
    "accounts whole annotations and removed entries under a %i-token section cap, including a second compaction",
    (sectionCap) => {
      const entries: EvidenceLedgerEntry[] = Array.from({ length: 5 }, (_, index) => ({
        id: `episode:${createEpisodeId()}`,
        source_type: "episode",
        session_scope: "global",
        actor: "memory",
        trust_rank: 52,
        text: `Original ${index}: ` + "n".repeat(500),
        evidence_context: {
          annotations: [
            { text: "甲".repeat(600), authority: null },
            { text: MULTILINGUAL, authority: "Synthetic" },
          ],
          annotations_omitted: 1,
        },
      }));
      const input = ledger(entries);
      const options = {
        targetTokens: 4_000,
        hardCapTokens: 4_000,
        sectionOptions: { episodes: { maxEntries: 3, maxTokens: sectionCap } },
      };
      for (const compacted of [
        compactEvidenceLedger(input, options),
        compactEvidenceLedger(compactEvidenceLedger(input, options).ledger, options),
      ]) {
        const section = compacted.ledger.sections[0]!;
        const kept = section.entries.filter((entry) => entry.omission_accounting === undefined);
        for (const row of kept) {
          expect(row.text).toBe(entries.find((original) => original.id === row.id)!.text);
          for (const annotation of ledgerAnnotations(row))
            expect(["甲".repeat(600), MULTILINGUAL]).toContain(annotation.text);
        }
        const accounts = section.entries.filter((entry) => entry.omission_accounting !== undefined);
        expect(accounts.length).toBeLessThanOrEqual(1);
        const counts =
          accounts[0] === undefined
            ? compacted.ledger.compaction_accounting?.folded_sections.episodes
            : omissionCounts(accounts[0]);
        expect(
          (counts?.annotations_omitted ?? 0) +
            kept.reduce((sum, entry) => sum + ledgerAnnotations(entry).length, 0),
        ).toBe(15);
        expect(estimatePromptTokens(renderSection(section))).toBeLessThanOrEqual(sectionCap);
        expect(estimateEvidenceLedgerPromptTokens(compacted.ledger)).toBeLessThanOrEqual(4_000);
        expect(compacted.traceSummary.budgetOutcome.status).toBe("within_budget");
        expect(renderCompactPlannerLedger(compacted.ledger)).not.toContain(
          "__borg_evidence_context",
        );
      }
      expect(entries.every((entry) => ledgerAnnotationCount(entry) === 3)).toBe(true);
    },
  );
  it("carries folded annotation omissions into the section accounting without attaching them to another record", async () => {
    const { reader, identityEventRepository } = setup();
    const { input, node } = createRecordEvidenceFixture();
    recordTestAnnotation(identityEventRepository, node.id, MULTILINGUAL, {
      recordType: "semantic_node",
    });
    const built = await buildRecordEvidenceLedger(input, reader);
    const compacted = compactEvidenceLedger(built);
    const all = compacted.ledger.sections.flatMap((section) => section.entries);
    for (const entry of all.filter((entry) => entry.source_type === "episode"))
      expect(entry.evidence_context).toBeUndefined();
    const source = built.sections.find((section) => section.id === "semantic_graph")!;
    const section = compacted.ledger.sections.find((section) => section.id === "semantic_graph")!;
    const total = source.entries.reduce((sum, entry) => sum + ledgerAnnotationCount(entry), 0);
    const omitted = section.entries.reduce(
      (sum, entry) => sum + (omissionCounts(entry).annotations_omitted ?? 0),
      0,
    );
    expect(
      section.entries.reduce((sum, entry) => sum + ledgerAnnotations(entry).length, 0) + omitted,
    ).toBe(total);
    expect(
      all
        .filter((entry) => entry.omission_accounting !== undefined)
        .every((entry) => !renderEntry(entry).includes(MULTILINGUAL)),
    ).toBe(true);
  });
});
