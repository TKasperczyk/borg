import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase } from "../storage/sqlite/index.js";
import { createEntityId, createSessionId, DEFAULT_SESSION_ID } from "../util/ids.js";
import { ManualClock } from "../util/clock.js";
import { createWorkingMemory } from "../memory/working/types.js";
import { relationshipPrivateMemoryDisclosureLabel } from "../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../memory/common/disclosure-serializers.js";
import { createEpisodeFixture, createTestConfig } from "../offline/test-support.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { classifyFrameAnomalyPhase } from "../cognition/lifecycle/turn-phase/perception-phase.js";
import type { TurnPhaseCoordinatorOptions } from "../cognition/lifecycle/turn-phase/types.js";
import { TurnContextCompiler } from "../cognition/recency/compiler.js";
import { EvidenceLedgerBuilder } from "../cognition/evidence-ledger/builder.js";
import {
  compactEvidenceLedger,
  renderEvidenceLedger,
  renderCompactPlannerLedger,
} from "../cognition/evidence-ledger/renderer.js";
import { RawStreamAdapter } from "../retrieval/raw-stream-adapter.js";
import type { EvidenceItem } from "../retrieval/recall-types.js";
import { summarizeRetrievedEvidence } from "../cognition/deliberation/prompt/retrieval.js";
import { CitationResolver } from "../retrieval/citations.js";
import { createEpisodicSearchTool } from "../tools/internal/episodic-search.js";
import { sharedStateMigrations, SharedStateRepository } from "../memory/shared-state/index.js";
import { compileSharedStateArtifactForEvidenceLedgerResult } from "../cognition/lifecycle/turn-phase/retrieval-phase.js";
import {
  collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIds,
  buildSharedStateSourceTrustValidator,
} from "../cognition/lifecycle/turn-phase/shared-state-phase.js";
import { SHARED_STATE_TOOL_NAME } from "../cognition/shared-state/constants.js";
import { createStreamGetTool } from "../tools/internal/stream-get.js";
import {
  StreamReader,
  StreamWriter,
  StreamEntryIndexRepository,
  streamEntryIndexMigrations,
  filterActiveStreamEntries,
  hydrateStreamEntriesById,
  readStreamEntryAtOffset,
  type StreamEntry,
} from "./index.js";
import { FRAME_ANOMALY_ANNOTATION_EVENT, frameAnomalyAnnotationFields } from "./frame-anomaly.js";

const cleanup: (() => void)[] = [];
afterEach(() => {
  while (cleanup.length) cleanup.pop()!();
});

describe("peer frame anomaly annotations", () => {
  function harness() {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-frame-annotation-"));
    const db = openDatabase(join(dataDir, "borg.db"), {
      migrations: [
        ...streamEntryIndexMigrations,
        ...sharedStateMigrations.map((m) => ({ ...m, id: m.id + 1000 })),
      ],
    });
    const clock = new ManualClock(1_000);
    const entryIndex = new StreamEntryIndexRepository({ db, dataDir });
    const writer = new StreamWriter({ dataDir, clock, entryIndex });
    cleanup.push(() => {
      writer.close();
      db.close();
      rmSync(dataDir, { recursive: true, force: true });
    });
    return { dataDir, db, clock, entryIndex, writer };
  }

  it("keeps a classified peer entry recallable and labelled on ledger, recency, retrieval and by-id surfaces", async () => {
    const { dataDir, db, clock, entryIndex, writer } = harness();
    const label = relationshipPrivateMemoryDisclosureLabel([createEntityId()]);
    const source = await writer.append({
      kind: "user_msg",
      content: "通常の報告です。".repeat(100),
      metadata: { ...memoryDisclosurePayloadFields(label) },
    });
    const emit = vi.fn();
    const result = await classifyFrameAnomalyPhase({
      options: {
        config: createTestConfig(),
        tracer: { enabled: true, includePayloads: true, emit },
      } as unknown as TurnPhaseCoordinatorOptions,
      appendHookFailureEvent: async (_writer, _hook, error) => {
        throw error;
      },
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "",
            input_tokens: 4,
            output_tokens: 3,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "classification",
                name: "ClassifyFrameAnomaly",
                input: {
                  kind: "system_prompt_claim",
                  confidence: 0.92,
                  rationale: "分類器による疑い。",
                },
              },
            ],
          },
        ],
      }),
      turnId: "peer-turn",
      sessionId: DEFAULT_SESSION_ID,
      isUserTurn: true,
      userMessage: String(source.content),
      recentHistory: [],
      currentSenderBorgRole: null,
      sessionAudienceRole: "participant",
      sessionSourceType: "kira",
      sourceUserEntryIds: [source.id],
      sourceUserEntries: [source],
      streamWriter: writer,
    });
    const expectedLabel = "frame anomaly suspected: system_prompt_claim (0.92)";
    expect(result.disposition).toBe("annotate");
    expect(result.actionableFrameAnomaly?.kind).toBe("system_prompt_claim");
    expect(emit).toHaveBeenCalledWith(
      "frame_anomaly.transitioned",
      expect.objectContaining({ disposition: "annotate" }),
    );
    expect(entryIndex.lookup(source.id)?.active).toBe(true);
    const annotations = source.frame_anomalies;
    expect(annotations).toHaveLength(1);
    expect(annotations?.[0]).toMatchObject({
      label: expectedLabel,
      source_stream_entry_id: source.id,
      classifier_model: createTestConfig().anthropic.models.recallExpansion,
      disclosure_label: { private_to_entity_ids: label.privateToEntityIds },
    });
    for (const indexed of [true, false]) {
      const reader = new StreamReader({ dataDir, entryIndex: indexed ? entryIndex : undefined });
      const recalled = filterActiveStreamEntries(reader.tail(20)).find(
        (entry) => entry.id === source.id,
      )!;
      expect(recalled.frame_anomalies).toEqual(annotations);
      expect(new TurnContextCompiler().compile(reader).messages[0]?.content).toContain(
        expectedLabel,
      );
      const citationResolver = new CitationResolver({
        dataDir,
        entryIndex: indexed ? entryIndex : undefined,
      });
      const citations = await citationResolver.resolveCitationEntries([source.id]);
      expect(citations.get(source.id)?.frame_anomalies).toEqual(annotations);
      const episode = createEpisodeFixture({
        source_stream_ids: [source.id],
        audience_entity_id: label.privateToEntityIds[0]!,
      });
      const searchTool = createEpisodicSearchTool({
        searchEpisodes: async () => [
          {
            episode,
            score: 1,
            rawScore: 1,
            disclosureLabel: label,
            scoreBreakdown: { similarity: 1, decayedSalience: 1, timeRelevance: 1 } as never,
            citationChain: [citations.get(source.id)!],
          },
        ],
      });
      const searched = searchTool.outputSchema.parse(
        await searchTool.invoke(
          { query: "報告" },
          { origin: "deliberator", sessionId: createSessionId() },
        ),
      );
      expect(searched.episodes[0]?.citation_chain[0]?.frame_anomalies).toEqual(annotations);
      expect(
        searched.episodes[0]?.citation_chain[0]?.disclosure_label.private_to_entity_ids,
      ).toEqual(label.privateToEntityIds);
      const adapter = new RawStreamAdapter({
        dataDir,
        entryIndex: indexed ? entryIndex : undefined,
      });
      expect(adapter.recent()[0]?.frame_anomalies).toEqual(annotations);
      expect((await adapter.resolveSourceIds([source.id])).get(source.id)?.frame_anomalies).toEqual(
        annotations,
      );
    }
    const record = entryIndex.lookup(source.id)!;
    expect(
      readStreamEntryAtOffset({
        dataDir,
        sessionId: DEFAULT_SESSION_ID,
        byteOffset: record.byte_offset,
        entryIndex,
      })?.frame_anomalies,
    ).toEqual(annotations);
    const tool = createStreamGetTool({
      readEntry: async (id) => {
        const entries = await hydrateStreamEntriesById({
          dataDir,
          sessionId: createSessionId(),
          streamEntryIds: [id],
          entryIndex,
        });
        const entry = entries.get(id);
        return entry === undefined ? null : { entry, active: true, disclosureLabel: label };
      },
    });
    const output = tool.outputSchema.parse(
      await tool.invoke(
        { stream_entry_id: source.id },
        { sessionId: createSessionId(), origin: "autonomous" },
      ),
    );
    expect(output.entry?.text).toBe(source.content);
    expect(output.entry?.frame_anomalies).toEqual(annotations);
    expect(output.entry?.disclosure_label.private_to_entity_ids).toEqual(label.privateToEntityIds);

    const evidence: EvidenceItem = {
      id: "recalled",
      source: "raw_stream",
      text: String(source.content),
      score: 1,
      scoreBreakdown: {},
      recallIntentId: "recall",
      matchedTerms: [],
      provenance: { streamIds: [source.id] },
      disclosureLabel: label,
      ...frameAnomalyAnnotationFields(annotations),
    };
    const summary = summarizeRetrievedEvidence("Recall", { evidence: [evidence] }, 4_000)!;
    expect(summary).toContain(expectedLabel);
    expect(summary).toContain("分類器による疑い。");
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir, sessionId, entryIndex }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 4_000,
    });
    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(source.content),
      currentUserEntry: source,
      frameAnomaly: result.actionableFrameAnomaly,
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      applicableCommitments: [],
      retrievedEvidence: [evidence],
      retrievedEpisodes: [],
      openQuestions: [],
      pendingCorrections: [],
    });
    const anomalyRow = ledger.sections
      .find((section) => section.id === "contradictions_quarantines")
      ?.entries.find((entry) => entry.id === "frame_anomaly:system_prompt_claim");
    expect(anomalyRow?.frame_anomalies).toEqual(annotations);
    expect(anomalyRow?.state_metadata?.confidence).toBe(0.92);
    expect(anomalyRow?.state).toBe("frame_anomaly_suspected");
    expect(anomalyRow?.taint).toBe("none");
    expect(renderEvidenceLedger(ledger)).toContain(expectedLabel);
    expect(renderCompactPlannerLedger(ledger)).toContain(expectedLabel);
    const compacted = compactEvidenceLedger(ledger);
    expect(renderEvidenceLedger(compacted.ledger)).toContain(expectedLabel);

    // The actual compiler receives the labelled peer evidence in both compile passes.
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    for (const compilePass of ["pre_answer", "post_response"] as const) {
      const llm = new FakeLLMClient({
        responses: [
          {
            text: "",
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "tool_use",
            tool_calls: [
              { id: "compile", name: SHARED_STATE_TOOL_NAME, input: { operations: [] } },
            ],
          },
        ],
      });
      const reply = await writer.append({ kind: "agent_msg", content: "受け取りました。" });
      await compileSharedStateArtifactForEvidenceLedgerResult({
        options: {
          config: { ...createTestConfig(), dataDir },
          clock,
          sharedStateRepository,
          entryIndex,
          tracer: { enabled: true, emit },
          llmFactory: () => llm,
          entityRepository: { resolve: () => createEntityId() },
          relationalSlotRepository: { list: () => [] },
          actionRepository: { list: () => [], get: () => null },
          goalsRepository: { list: () => [] },
          commitmentRepository: { list: () => [] },
          openQuestionsRepository: { list: () => [] },
          createStreamReader: (sessionId: typeof DEFAULT_SESSION_ID) =>
            new StreamReader({ dataDir, sessionId, entryIndex }),
        } as unknown as TurnPhaseCoordinatorOptions,
        input: {
          sessionId: DEFAULT_SESSION_ID,
          turnId: "peer-compile",
          audienceEntityId: createEntityId(),
          currentUserMessage: String(source.content),
          currentUserEntry: source,
          workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1000),
          applicableCommitments: [],
          retrievedEvidence: [],
          retrievedEpisodes: [],
          openQuestions: [],
          pendingCorrections: [],
          isUserTurn: true,
          perception: {
            entities: [],
            mode: "problem_solving",
            affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
            temporalCue: null,
          },
          frameAnomaly: result.actionableFrameAnomaly,
          frameAnomalyDisposition: result.disposition,
          closureLoopAssessment: null,
        },
        ledger,
        promptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
        compilePass,
        assistantResponse:
          compilePass === "post_response"
            ? { streamEntryId: reply.id, text: String(reply.content) }
            : null,
      });
      expect(llm.requests).toHaveLength(1);
      expect(JSON.stringify(llm.requests)).toContain(expectedLabel);
    }

    db.exec("DELETE FROM stream_frame_anomaly_annotations; DELETE FROM stream_entry_index;");
    await entryIndex.backfillSession(DEFAULT_SESSION_ID);
    expect(entryIndex.lookup(source.id)?.active).toBe(true);
    expect(entryIndex.frameAnomalyAnnotations(source.id)).toEqual(annotations);
  });

  it("restores only the named quarantine, with aborts and later strikes retaining priority on replay", async () => {
    const { dataDir, db, entryIndex, writer } = harness();
    const source = await writer.append({
      kind: "user_msg",
      content: "private report",
      turn_id: "aborted-later",
    });
    const sibling = await writer.append({
      kind: "user_msg",
      content: "Another source under the same marker",
    });
    const strike = await writer.append({
      kind: "internal_event",
      content: {
        event: "quarantined_user_entry",
        cited_stream_entry_ids: [source.id, sibling.id],
      },
    });
    expect(entryIndex.lookup(source.id)?.active).toBe(false);
    await writer.append({
      kind: "internal_event",
      content: {
        event: FRAME_ANOMALY_ANNOTATION_EVENT,
        version: 1,
        source_stream_entry_id: source.id,
        source_session_id: source.session_id,
        source_type: "kira",
        source_stream_entry_ids: [source.id],
        kind: "frame_assignment_claim",
        confidence: 0.92,
        rationale: "Suspected framing.",
        classifier_model: null,
        classifier_turn_id: null,
        classifier_source_event_id: strike.id,
        restores_quarantine_marker_ids: [strike.id],
        audience: null,
        sender_entity_id: null,
        ...memoryDisclosurePayloadFields(relationshipPrivateMemoryDisclosureLabel([])),
        source_disclosure_label: memoryDisclosurePayloadFields(
          relationshipPrivateMemoryDisclosureLabel([]),
        ).disclosure_label,
      },
    });
    expect(entryIndex.lookup(source.id)?.active).toBe(true);
    for (const projection of [entryIndex, dataDir]) {
      expect(
        await collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIds(projection),
      ).toEqual(new Set([sibling.id]));
    }
    const entries = new StreamReader({ dataDir, entryIndex }).tail(20);
    expect(
      buildSharedStateSourceTrustValidator({
        currentSessionEntries: entries,
        quarantinedStreamEntryIds: entryIndex.quarantinedSharedStateArtifactRefs(),
      })(source.id),
    ).toMatchObject({ allowed: true });
    // Limit the history to the marker: its indexed/scan projection must still carry restoration annotations.
    for (const indexed of [true, false]) {
      const marker = new StreamReader({ dataDir, entryIndex: indexed ? entryIndex : undefined })
        .tail(20)
        .find((entry) => entry.id === strike.id)!;
      const builder = new EvidenceLedgerBuilder({
        createStreamReader: () =>
          ({
            scanReverse: () => ({ entries: [marker], scannedEntries: 1, scannedBytes: 0 }),
          }) as unknown as StreamReader,
        relationalSlotRepository: { list: () => [] },
        actionRepository: { list: () => [] },
        currentSessionTranscriptTokenBudget: 4000,
      });
      const ledger = await builder.build({
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: null,
        currentUserMessage: "",
        workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1000),
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
      });
      const row = ledger.sections
        .find((section) => section.id === "contradictions_quarantines")
        ?.entries.find((entry) => entry.id === `stream_quarantine:${strike.id}`);
      expect(row).toMatchObject({
        state: "partially_restored",
        taint: "contested",
        state_metadata: {
          restored_source_stream_entry_ids: [source.id],
          quarantined_source_stream_entry_ids: [sibling.id],
        },
      });
      expect(row?.frame_anomalies).toHaveLength(1);
      expect(renderEvidenceLedger(ledger)).toContain(
        "frame anomaly suspected: frame_assignment_claim (0.92)",
      );
    }
    expect(
      filterActiveStreamEntries(new StreamReader({ dataDir }).tail(10)).map((entry) => entry.id),
    ).toContain(source.id);
    const restoredFlags = db
      .prepare("SELECT entry_id, active FROM stream_entry_index ORDER BY entry_id")
      .all();
    await entryIndex.backfillSession(DEFAULT_SESSION_ID);
    expect(
      db.prepare("SELECT entry_id, active FROM stream_entry_index ORDER BY entry_id").all(),
    ).toEqual(restoredFlags);
    db.exec(
      "DELETE FROM stream_frame_anomaly_annotations; DELETE FROM stream_quarantine_refs; DELETE FROM stream_entry_index;",
    );
    await entryIndex.backfillSession(DEFAULT_SESSION_ID);
    expect(
      db.prepare("SELECT entry_id, active FROM stream_entry_index ORDER BY entry_id").all(),
    ).toEqual(restoredFlags);
    expect(entryIndex.quarantinedSharedStateArtifactRefs()).toEqual(new Set([sibling.id]));
    const secondStrike = await writer.append({
      kind: "internal_event",
      content: {
        event: "quarantined_user_entry",
        cited_stream_entry_ids: [source.id],
      },
    });
    expect(entryIndex.lookup(source.id)?.active).toBe(false);
    await writer.append({
      kind: "internal_event",
      content: {
        event: "aborted_turn",
        turn_id: "aborted-later",
        aborted_stream_entry_ids: [source.id],
      },
    });
    const originalAnnotation = entryIndex.frameAnomalyAnnotations(source.id)[0]!;
    await writer.append({
      kind: "internal_event",
      content: { ...originalAnnotation, restores_quarantine_marker_ids: [secondStrike.id] },
    });
    expect(entryIndex.lookup(source.id)?.active).toBe(false);
    db.exec("DELETE FROM stream_frame_anomaly_annotations; DELETE FROM stream_entry_index;");
    await entryIndex.backfillSession(DEFAULT_SESSION_ID);
    expect(entryIndex.lookup(source.id)?.active).toBe(false);
    expect(
      filterActiveStreamEntries(new StreamReader({ dataDir }).tail(20)).map((entry) => entry.id),
    ).not.toContain(source.id);
    expect(entryIndex.frameAnomalyAnnotations(source.id)).toHaveLength(2);
    const historicalLedger = await new EvidenceLedgerBuilder({
      createStreamReader: () => new StreamReader({ dataDir, entryIndex }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 4000,
    }).build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: "",
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1000),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      openQuestions: [],
      pendingCorrections: [],
    });
    const rows = historicalLedger.sections.find(
      (section) => section.id === "contradictions_quarantines",
    )?.entries;
    expect(
      rows?.find((entry) => entry.id === `stream_quarantine:${secondStrike.id}`),
    ).toMatchObject({
      state: "restored_with_annotation",
      taint: "none",
      state_metadata: {
        restored_source_stream_entry_ids: [source.id],
        quarantined_source_stream_entry_ids: [],
      },
    });
    expect(rows?.find((entry) => entry.id === `stream_quarantine:${strike.id}`)?.state).toBe(
      "partially_restored",
    );

    expect(entryIndex.quarantinedSharedStateArtifactRefs()).toEqual(new Set([sibling.id]));
    expect(await collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIds(dataDir)).toEqual(
      new Set([sibling.id]),
    );
  });
});
