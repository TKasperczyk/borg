import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase, type SqliteDatabase } from "../../storage/sqlite/index.js";
import { identityMigrations } from "../../memory/identity/migrations.js";
import { IdentityEventRepository } from "../../memory/identity/repository.js";
import { RecordCorrectionService } from "../../memory/lifecycle-ops/correct-record.js";
import { createOfflineTestHarness, type OfflineTestHarness } from "../../offline/test-support.js";
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
  let harness: OfflineTestHarness | undefined;
  afterEach(async () => {
    db?.close();
    db = undefined;
    await harness?.cleanup();
    harness = undefined;
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
  it("selects at most two complete annotations of at most 600 characters, newest first, and counts the rest", async () => {
    const { reader, identityEventRepository } = setup();
    const { input, episodes } = createRecordEvidenceFixture();
    for (const [index, text] of ["old", MULTILINGUAL, "界".repeat(600), "x".repeat(601)].entries())
      recordTestAnnotation(identityEventRepository, episodes[0]!.episode.id, text, { ts: index });
    const contexts = await readLedgerRecordEvidence(reader, input);
    const context = contexts.get(`episode:${episodes[0]!.episode.id}`)!;
    expect(context.annotations).toHaveLength(MAX_LEDGER_ANNOTATIONS_PER_RECORD);
    expect(context.annotations!.map((row) => row.text)).toEqual([
      "界".repeat(MAX_LEDGER_ANNOTATION_CHARS),
      MULTILINGUAL,
    ]);
    expect(context.annotations_omitted).toBe(2);
    expect(
      await readRecordEvidenceContexts(reader, [
        { record_type: "semantic_node", record_id: episodes[0]!.episode.id },
      ]),
    ).toEqual(new Map());
  });

  it.each(["neither", "episode", "semantic_node", "both"] as const)(
    "keeps retained annotations and uses independent body batches when %s fails",
    async (failure) => {
      harness = await createOfflineTestHarness();
      const { input, episodes, semantic, node, evidence } = createRecordEvidenceFixture();
      const warmEpisode = episodes[0]!.episode;
      const autobiographicalEpisode = episodes[1]!.episode;
      for (const { episode } of episodes) await harness.episodicRepository.createEpisode(episode);
      await harness.semanticNodeRepository.insert(node);
      const service = new RecordCorrectionService({
        ...harness,
        entryIndex: harness.createContext().entryIndex!,
      });
      const quote = "原文。";
      for (const id of [warmEpisode.id, autobiographicalEpisode.id, node.id]) {
        await service.correct(
          {
            target_id: id,
            corrects_span: quote,
            correction_note: "Synthetic correction.",
            evidence: [{ id: warmEpisode.id, basis: "own_read" }],
          },
          {
            turnId: "synthetic-turn",
            turnOrigin: "user",
            path: "reflector",
            streamWriter: harness.streamWriter,
          },
        );
      }
      const narrative = "Rewritten episode.";
      const description = `Rewritten node. ${quote}`;
      await harness.episodicRepository.update(warmEpisode.id, { narrative });
      await harness.semanticNodeRepository.update(node.id, { description });
      const queryEpisodes = vi.spyOn(harness.episodicRepository, "getMany");
      const queryNodes = vi.spyOn(harness.semanticNodeRepository, "getMany");
      const episodeFailed = failure === "episode" || failure === "both";
      const nodeFailed = failure === "semantic_node" || failure === "both";
      if (episodeFailed)
        queryEpisodes.mockRejectedValueOnce(new Error("Synthetic episode read failure"));
      if (nodeFailed) queryNodes.mockRejectedValueOnce(new Error("Synthetic node read failure"));
      const read = vi.spyOn(harness.identityEventRepository, "listAnnotationsForTargets");
      const emit = vi.fn();
      const uncorrectedEpisode = createEpisodeId();
      const autobiographical = input.autobiographicalRecall!;
      const built = await buildRecordEvidenceLedger(
        {
          ...input,
          retrievedEpisodes: [],
          retrievedSemantic: { ...semantic, matched_node_ids: [], matched_nodes: [] },
          retrievedEvidence: [
            {
              ...evidence[0]!,
              source: "warm_recall",
              text: narrative,
            },
            {
              ...evidence[1]!,
              source: "warm_recall",
              text: description,
            },
            {
              ...evidence[0]!,
              id: "synthetic-other-episode",
              provenance: { episodeId: uncorrectedEpisode },
            },
          ],
          autobiographicalRecall: {
            ...autobiographical,
            evidence: [
              {
                ...autobiographical.evidence[0]!,
                text: "Second original text.",
                sourceEpisodeIds: [autobiographicalEpisode.id],
                metadata: { episode_id: autobiographicalEpisode.id },
              },
            ],
          },
        },
        { ...harness, tracer: { enabled: true, includePayloads: false, emit } },
      );
      expect(read).toHaveBeenCalledBefore(queryEpisodes);
      expect(read).toHaveBeenCalledBefore(queryNodes);
      expect(queryEpisodes).toHaveBeenCalledExactlyOnceWith([
        warmEpisode.id,
        autobiographicalEpisode.id,
      ]);
      expect(queryNodes).toHaveBeenCalledExactlyOnceWith([node.id], { includeArchived: true });
      const entries = built.sections.flatMap((section) => section.entries);
      const unknown = { target_changed: null, quoted_span_present: null };
      for (const [id, status] of [
        [
          `retrieved_evidence:${evidence[0]!.id}`,
          episodeFailed ? unknown : { target_changed: true, quoted_span_present: false },
        ],
        [
          `retrieved_evidence:${evidence[1]!.id}`,
          nodeFailed ? unknown : { target_changed: true, quoted_span_present: true },
        ],
        [
          `autobiographical_recall:${autobiographical.evidence[0]!.id}`,
          episodeFailed ? unknown : { target_changed: false },
        ],
      ] as const) {
        expect(entries.find((entry) => entry.id === id)?.evidence_context).toMatchObject({
          self_correction_count: 1,
          annotations: [{ text: "Synthetic correction.", ...status }],
        });
      }
      if (!episodeFailed && !nodeFailed) expect(emit).not.toHaveBeenCalled();
      else
        expect(emit).toHaveBeenCalledExactlyOnceWith("memory.annotations.body_lookup_failed", {
          turnId: "memory-annotation-read",
          counts_by_reason: {
            ...(episodeFailed ? { episode: 2 } : {}),
            ...(nodeFailed ? { semantic_node: 1 } : {}),
          },
        });

      queryEpisodes.mockClear();
      queryNodes.mockClear();
      const readLoadedText = vi.fn();
      const targets = [
        { record_type: "episode", record_id: warmEpisode.id },
        { record_type: "semantic_node", record_id: node.id },
      ] as const;
      const countOnly = await readRecordEvidenceContexts(harness, targets, {
        maxAnnotations: 0,
        readLoadedText,
      });
      expect([...countOnly.values()]).toEqual(
        Array(2).fill({ annotations_omitted: 1, self_correction_count: 1 }),
      );
      expect(readLoadedText).not.toHaveBeenCalled();
      expect(queryEpisodes).not.toHaveBeenCalled();
      expect(queryNodes).not.toHaveBeenCalled();
    },
  );
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

  it("admits a short operator annotation after the reserved self-written correction exceeds the section budget", () => {
    const short = { text: "Audited observation.", authority: "Synthetic operator" };
    const entry: EvidenceLedgerEntry = {
      id: `episode:${createEpisodeId()}`,
      source_type: "episode",
      session_scope: "global",
      actor: "memory",
      trust_rank: 52,
      text: "Original observation.",
      evidence_context: {
        self_correction_count: 1,
        annotations: [{ author: "borg", text: "界".repeat(600), authority: null }, short],
      },
    };
    const maxTokens = 200;
    const source = ledger([entry]);
    expect(estimatePromptTokens(renderSection(source.sections[0]!))).toBeGreaterThan(maxTokens);
    const result = compactEvidenceLedger(source, {
      sectionOptions: { episodes: { maxTokens } },
    }).ledger.sections[0]!;
    expect(result.entries[0]!.evidence_context).toEqual({
      self_correction_count: 1,
      annotations: [short],
      annotations_omitted: 1,
    });
    expect(estimatePromptTokens(renderSection(result))).toBeLessThanOrEqual(maxTokens);
  });
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
