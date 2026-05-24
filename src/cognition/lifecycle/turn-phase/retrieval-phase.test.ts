import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "../../../config/index.js";
import { FakeLLMClient } from "../../../llm/test-support/fake-client.js";
import { sharedStateMigrations } from "../../../memory/decision-artifacts/index.js";
import { SharedStateRepository } from "../../../memory/decision-artifacts/repository.js";
import { openDatabase } from "../../../storage/sqlite/index.js";
import { FixedClock } from "../../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createActionId,
  createCommitmentId,
  createEntityId,
  createGoalId,
  createOpenQuestionId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
} from "../../../util/ids.js";
import type { ActionRecord } from "../../../memory/actions/index.js";
import {
  QUARANTINED_USER_ENTRY_EVENT,
  StreamWriter,
  type StreamEntry,
  type StreamReader,
} from "../../../stream/index.js";
import {
  makeLockedSharedStateEntry,
  makeSharedStateArtifact,
} from "../../../test-support/factories/shared-state.js";
import type { PerceptionResult } from "../../types.js";
import { SHARED_STATE_TOOL_NAME } from "../../shared-state/schema.js";
import { SESSION_REENTRY_CONTINUITY_TAG } from "../../session-reentry-continuity.js";
import {
  compileSharedStateArtifactForEvidenceLedger,
  compileSharedStateArtifactForEvidenceLedgerResult,
  runRetrievalPhase,
} from "./retrieval-phase.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";

describe("compileSharedStateArtifactForEvidenceLedger", () => {
  const cleanup: Array<() => void> = [];

  afterEach(() => {
    while (cleanup.length > 0) {
      cleanup.pop()?.();
    }
  });

  it("uses the global turn counter for shared-state action canonicalization", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(10_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const actionId = createActionId();
    const streamEntryId = createStreamEntryId();
    const currentUserContent = "The clinic callback follow-up is locked.";
    const currentUserEntry = {
      id: streamEntryId,
      kind: "user_msg",
      content: currentUserContent,
      timestamp: 10_000,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const action = {
      id: actionId,
      description: "Follow up with the clinic",
      actor: "user",
      audience_entity_id: audienceEntityId,
      state: "committed_to_do",
      updated_at: 9_000,
      session_scope: null,
      scheduled_at: null,
      last_referenced_turn_counter: 2,
      last_referenced_turn_global: null,
    } as ActionRecord;
    const update = vi.fn();
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [
                  {
                    type: "add",
                    state_key: "decision.route",
                    kind: "locked",
                    text: "The clinic callback follow-up is locked.",
                    owner_entity_id: audienceEntityId,
                    source_stream_entry_ids: [streamEntryId],
                    canonicalizes: {
                      action_ids: [actionId],
                    },
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: false,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [action],
        get: () => action,
        update,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    await compileSharedStateArtifactForEvidenceLedger({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-global-canonicalization",
        audienceEntityId,
        currentUserMessage: currentUserContent,
        currentUserEntry,
        globalTurnCounter: 42,
        workingMemory: {
          turn_counter: 3,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "current_user_message",
            label: "1. Current User Message",
            entries: [
              {
                id: `current_user_message:${streamEntryId}`,
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: currentUserContent,
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "Action candidate: Follow up with the clinic.",
    });

    expect(update).toHaveBeenCalledWith(
      actionId,
      expect.objectContaining({
        last_referenced_turn_counter: 42,
        last_referenced_turn_global: 42,
      }),
      { skipSideEffects: true },
    );
  });

  it("uses the same structural render salience signals when compile is skipped", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-skip-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(20_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const actionId = createActionId();
    const goalId = createGoalId();
    const openQuestionId = createOpenQuestionId();
    const commitmentId = createCommitmentId();
    const operationalCommitmentId = createCommitmentId();
    const streamEntryId = createStreamEntryId();
    const currentUserEntry = {
      id: streamEntryId,
      kind: "user_msg",
      content: "Thanks, that closes it.",
      timestamp: 20_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-skipped-render-signals",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const action = {
      id: actionId,
      description: "Send the project note",
      actor: "user",
      audience_entity_id: audienceEntityId,
      state: "committed_to_do",
      updated_at: 19_000,
      session_scope: null,
      scheduled_at: null,
      last_referenced_turn_counter: null,
      last_referenced_turn_global: null,
    } as ActionRecord;
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: true,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => new FakeLLMClient({ responses: [] }),
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [action],
        get: () => action,
      },
      goalsRepository: {
        list: () => [
          {
            id: goalId,
            description: "Keep project notes current",
          },
        ],
      },
      commitmentRepository: {
        list: () => [
          {
            id: commitmentId,
            directive: "Do not reveal private project notes.",
            kind: "boundary",
            type: "rule",
            directive_family: "privacy",
            enforcement_class: "critical",
            critical_domain: "privacy",
          },
          {
            id: operationalCommitmentId,
            directive: "Prefer concise project-note summaries.",
            kind: "process_norm",
            type: "rule",
            directive_family: "brevity",
            enforcement_class: "advisory",
            critical_domain: null,
          },
        ],
      },
      openQuestionsRepository: {
        list: () => [
          {
            id: openQuestionId,
            question: "Which project note is current?",
          },
        ],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-skipped-render-signals",
        audienceEntityId,
        currentUserMessage: "Thanks, that closes it.",
        currentUserEntry,
        globalTurnCounter: 12,
        workingMemory: {
          turn_counter: 12,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "idle",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "",
    });

    expect(result.appliedOperationCount).toBe(0);
    expect(result.renderOptions?.activeOpenQuestionIds).toEqual([openQuestionId]);
    expect(result.renderOptions?.activeActionIds).toEqual([actionId]);
    expect(result.renderOptions?.activeGoalIds).toEqual([goalId]);
    expect(result.renderOptions?.activeCriticalCommitmentIds).toEqual([commitmentId]);
    expect(result.renderOptions?.activeOperationalCommitmentIds).toEqual([operationalCommitmentId]);
    expect(result.renderOptions?.activeOperationalCommitmentIds).not.toContain(commitmentId);
  });

  it("uses indexed source-trust facts instead of loading the full session stream", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-indexed-trust-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(21_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const inactiveSourceEntryId = createStreamEntryId();
    const currentSourceEntryId = createStreamEntryId();
    const missingIndexedSourceEntryId = createStreamEntryId();
    const currentUserEntry = {
      id: currentSourceEntryId,
      kind: "user_msg",
      content: "Thanks, that closes it.",
      timestamp: 21_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-indexed-source-trust",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const lookupEntriesById = vi.fn((entryIds: readonly string[]) => {
      const facts = new Map();

      if (entryIds.includes(inactiveSourceEntryId)) {
        facts.set(inactiveSourceEntryId, {
          entry_id: inactiveSourceEntryId,
          session_id: DEFAULT_SESSION_ID,
          timestamp: 19_000,
          kind: "user_msg",
          turn_id: "turn-aborted",
          turn_status: "active",
          active: false,
        });
      }

      return facts;
    });
    const iterate = vi.fn(async function* () {
      throw new Error("session stream should not be loaded for indexed source trust");
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: true,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => new FakeLLMClient({ responses: [] }),
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      entryIndex: {
        countSessionEntriesByKind: () => 0,
        lookupEntriesById,
        quarantinedSharedStateArtifactRefs: () => new Set(),
      },
      createStreamReader: () =>
        ({
          iterate,
        }) as unknown as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-indexed-source-trust",
        audienceEntityId,
        currentUserMessage: "Thanks, that closes it.",
        currentUserEntry,
        globalTurnCounter: 13,
        workingMemory: {
          turn_counter: 13,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "idle",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "current_user_message",
            label: "1. Current User Message",
            entries: [
              {
                id: `current_session_stream:${inactiveSourceEntryId}`,
                source_type: "current_session_stream",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: "Inactive evidence.",
              },
              {
                id: `current_user_message:${currentSourceEntryId}`,
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 0,
                text: "Current evidence.",
              },
              {
                id: "retrieved_evidence:missing-index-source",
                source_type: "prior_session_stream",
                session_scope: "prior_session",
                actor: "user",
                trust_rank: 1,
                citations: [missingIndexedSourceEntryId],
                text: "Evidence missing from the index.",
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "",
    });

    expect(iterate).not.toHaveBeenCalled();
    expect(lookupEntriesById).toHaveBeenCalled();
    expect(warn).toHaveBeenCalledWith(
      `Stream entry ${missingIndexedSourceEntryId} was not found in the stream entry index during shared-state source trust validation`,
    );
    expect(result.renderOptions?.ledgerStreamEntryIds).toEqual([
      currentSourceEntryId,
      missingIndexedSourceEntryId,
    ]);
    warn.mockRestore();
  });

  it("falls back to stream scanning for cross-session quarantined shared-state refs without an entry index", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-quarantine-fallback-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(25_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const quarantinedSourceEntryId = createStreamEntryId();
    const currentSourceEntryId = createStreamEntryId();
    const currentUserEntry = {
      id: currentSourceEntryId,
      kind: "user_msg",
      content: "Current placeholder source.",
      timestamp: 25_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-quarantine-fallback-current",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const quarantineWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: createSessionId(),
      clock,
    });
    cleanup.push(() => quarantineWriter.close());
    await quarantineWriter.append({
      kind: "internal_event",
      content: {
        event: QUARANTINED_USER_ENTRY_EVENT,
        source_stream_entry_id: quarantinedSourceEntryId,
        cited_stream_entry_ids: [],
      },
    });
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [
                  {
                    type: "add",
                    state_key: "decision.quarantined",
                    kind: "locked",
                    text: "A quarantined cross-session source should not be accepted.",
                    owner_entity_id: audienceEntityId,
                    source_stream_entry_ids: [quarantinedSourceEntryId],
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: true,
              },
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield currentUserEntry;
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-quarantine-fallback",
        audienceEntityId,
        currentUserMessage: "Current placeholder source.",
        currentUserEntry,
        globalTurnCounter: 25,
        workingMemory: {
          turn_counter: 25,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "idle",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "prior_session_memory",
            label: "Retrieved Evidence",
            entries: [
              {
                id: `retrieved_evidence:${quarantinedSourceEntryId}`,
                source_type: "prior_session_stream",
                session_scope: "prior_session",
                actor: "user",
                trust_rank: 1,
                text: "Quarantined cross-session evidence.",
              },
            ],
          },
        ],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "Quarantined cross-session evidence.",
    });

    expect(result.appliedOperationCount).toBe(0);
    expect(sharedStateRepository.get(audienceEntityId)?.entries ?? []).toHaveLength(0);
  });

  it("keeps shared-state entries cited by current retrieval results searchable while allowing low-salience demotion", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-retrieval-phase-retrieved-state-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const clock = new FixedClock(30_000);
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const oldSourceEntry = {
      id: createStreamEntryId(),
      kind: "user_msg",
      content: "Placeholder source for retrieved shared state.",
      timestamp: 1_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-1",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const fillerEntries = Array.from({ length: 8 }, (_, index) => ({
      id: createStreamEntryId(),
      kind: "user_msg",
      content: `Placeholder filler source ${index + 2}.`,
      timestamp: 2_000 + index,
      session_id: DEFAULT_SESSION_ID,
      turn_id: `turn-${index + 2}`,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    })) as StreamEntry[];
    const currentUserEntry = {
      id: createStreamEntryId(),
      kind: "user_msg",
      content: "Current placeholder source.",
      timestamp: 30_000,
      session_id: DEFAULT_SESSION_ID,
      turn_id: "turn-10",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const initial = sharedStateRepository.upsert(
      audienceEntityId,
      [
        {
          type: "add",
          state_key: "state.placeholder",
          kind: "live",
          text: "Placeholder retrieved shared state",
          provenance_stream_entry_ids: [oldSourceEntry.id],
          last_updated_stream_entry_ids: [oldSourceEntry.id],
          created_at: 1_000,
          last_updated_at: 1_000,
        },
      ],
      {
        now: 1_000,
      },
    );
    const entryId = initial?.entries[0]?.id;
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state",
              name: SHARED_STATE_TOOL_NAME,
              input: {
                operations: [],
              },
            },
          ],
        },
      ],
    });
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            decisionArtifact: {
              ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
              compilerPrefilter: {
                enabled: false,
              },
              recentTurnThreshold: 5,
              dormantTurnThreshold: 15,
            },
          },
        },
      },
      sharedStateRepository,
      llmFactory: () => llmClient,
      clock,
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
        get: () => null,
      },
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            for (const entry of [oldSourceEntry, ...fillerEntries, currentUserEntry]) {
              yield entry;
            }
          },
        }) as StreamReader,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await compileSharedStateArtifactForEvidenceLedgerResult({
      options,
      input: {
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn-retrieved-shared-state",
        audienceEntityId,
        currentUserMessage: "Current placeholder source.",
        currentUserEntry,
        globalTurnCounter: 10,
        workingMemory: {
          turn_counter: 10,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [
          {
            id: "retrieved-placeholder-source",
            source: "raw_stream",
            text: "Placeholder retrieved evidence.",
            provenance: {
              streamIds: [oldSourceEntry.id],
            },
            recallIntentId: "intent-placeholder",
            matchedTerms: [],
            score: 1,
            scoreBreakdown: {},
          },
        ] as never,
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [],
        participantRoster: null,
        isUserTurn: true,
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        } satisfies PerceptionResult,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [],
        transcriptIncluded: false,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 0,
        rawPreservedUserTranscriptEntryCount: 0,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "",
    });

    expect(result.renderOptions?.recentlyRetrievedEntryIds).toEqual([entryId]);
    expect(sharedStateRepository.get(audienceEntityId)?.entries[0]?.kind).toBe("low_salience_live");
  });
});

describe("runRetrievalPhase session re-entry continuity", () => {
  it("renders when an autonomous turn precedes the first user-origin turn", async () => {
    const audienceEntityId = createEntityId();
    const currentUserEntryId = createStreamEntryId();
    const priorAutonomousEntryId = createStreamEntryId();
    const artifact = makeSharedStateArtifact([
      makeLockedSharedStateEntry({
        audience_entity_id: audienceEntityId,
        state_key: "project.decision",
      }),
    ]);
    const currentUserEntry = {
      id: currentUserEntryId,
      kind: "user_msg",
      content: "Start a decision log for the project.",
      timestamp: 11_000,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    } as StreamEntry;
    const priorAutonomousEntry = {
      id: priorAutonomousEntryId,
      kind: "perception",
      content: {
        mode: "problem_solving",
        entities: [],
      },
      timestamp: 10_000,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
    } as StreamEntry;
    const retrieval = {
      evidence: [],
      episodes: [],
      semantic: null,
      open_questions: [],
      recall_intents: [],
      contradiction_present: false,
      contradictionRouting: {
        contradictions: [],
      },
      confidence: null,
    } as never;
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        generation: {
          ...DEFAULT_CONFIG.generation,
          evidenceLedger: {
            ...DEFAULT_CONFIG.generation.evidenceLedger,
            enabled: false,
          },
        },
      },
      sharedStateRepository: {
        get: () => artifact,
      },
      selfContextBuilder: {
        build: vi.fn(async () => ({
          selfSnapshot: {
            values: [],
            goals: [],
            traits: [],
          },
          activeScoringValues: [],
          selfScoringFeatures: {
            goalVectors: [],
            valueVectors: [],
          },
          retrievalScoringFeatures: {
            goalVectors: [],
            valueVectors: [],
          },
          executiveFocus: {
            selected_goal: null,
            selected_score: null,
            candidates: [],
            threshold: 0,
          },
        })),
      },
      turnRetrievalCoordinator: {
        coordinate: vi.fn(async () => ({
          applicableCommitments: [],
          pendingCorrections: [],
          affectiveTrajectory: [],
          retrieval,
          retrievedEpisodes: [],
          retrievedSemantic: null,
          proceduralContext: null,
          selectedSkill: null,
          retrievalOptions: {},
          reRetrieve: vi.fn(async () => retrieval),
        })),
      },
      relationalSlotRepository: {
        list: () => [],
        listConstrained: () => [],
      },
      openQuestionsRepository: {
        get: () => null,
      },
      createStreamReader: () =>
        ({
          async *iterate() {
            yield priorAutonomousEntry;
            yield currentUserEntry;
          },
        }) as StreamReader,
      clock: new FixedClock(11_000),
      tracer: {
        enabled: false,
        emit: vi.fn(),
      },
      entityRepository: {
        findByName: () => null,
      },
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await runRetrievalPhase({
      options,
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-first-user-after-autonomous",
      turnInput: {
        userMessage: "Start a decision log for the project.",
        audience: "project-team",
        origin: "user",
      },
      isSelfAudience: false,
      isUserTurn: true,
      cognitionInput: "Start a decision log for the project.",
      llmClient: new FakeLLMClient({ responses: [] }),
      recencyMessages: [],
      audienceEntityId,
      audienceEntity: null,
      audienceProfile: null,
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: {
          valence: 0,
          arousal: 0,
          dominant_emotion: null,
        },
        temporalCue: null,
      } satisfies PerceptionResult,
      workingMemory: {
        turn_counter: 2,
      } as never,
      suppressionSet: {} as never,
      actionLinkSelfContext: null,
      persistedPromotions: {
        goalIds: [],
        executiveStepIds: [],
      },
      correctiveCommitment: null,
      activeParticipants: [],
      participantRoster: null,
      participantProfiles: [],
      persistedUserEntry: currentUserEntry,
      currentTurnFrameAnomaly: null,
      closureLoopAssessment: null,
    });

    expect(result.evidenceLedgerContext.sessionReentryContinuityPromptSection).toContain(
      `<${SESSION_REENTRY_CONTINUITY_TAG}>`,
    );
    expect(result.evidenceLedgerContext.sessionReentryContinuityPromptSection).toContain(
      "active_entry_count=1",
    );
  });
});
