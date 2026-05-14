import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { decisionArtifactMigrations } from "../../memory/decision-artifacts/migrations.js";
import { DecisionArtifactRepository } from "../../memory/decision-artifacts/repository.js";
import type { DecisionArtifactEntryKind } from "../../memory/decision-artifacts/types.js";
import { selfMigrations } from "../../memory/self/migrations.js";
import { GoalsRepository } from "../../memory/self/goals-repository.js";
import {
  composeMigrations,
  openDatabase,
  type SqliteDatabase,
} from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import type { JsonValue } from "../../util/json-value.js";
import {
  createActionId,
  createCommitmentId,
  createEntityId,
  createGoalId,
  createOpenQuestionId,
  createStreamEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/index.js";
import { renderDecisionStateArtifact, renderEvidenceLedger } from "../evidence-ledger/index.js";
import { buildDecisionArtifactLedgerPromptContext } from "../lifecycle/turn-phase-coordinator.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import {
  compileDecisionArtifact,
  DECISION_ARTIFACT_TOOL_NAME,
  type EmitDecisionArtifactPatch,
} from "./compiler.js";

function emitDecisionArtifactPatchResponse(patch: EmitDecisionArtifactPatch) {
  return {
    text: "",
    input_tokens: 12,
    output_tokens: 8,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_decision_patch",
        name: DECISION_ARTIFACT_TOOL_NAME,
        input: patch,
      },
    ],
  };
}

function throwingResponse(): never {
  throw new Error("llm down");
}

function createTraceRecorder(): TurnTracer & {
  events: Array<{ event: TurnTraceEventName; data: TurnTraceData }>;
} {
  const events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];

  return {
    enabled: true,
    includePayloads: true,
    events,
    emit: vi.fn((event: TurnTraceEventName, data: TurnTraceData) => {
      events.push({ event, data });
    }),
  };
}

function ledgerEntry(input: {
  streamEntryId: StreamEntryId;
  streamIndex: number;
  text: string;
}): EvidenceLedgerEntry {
  return {
    id: `current_session_stream:${input.streamEntryId}`,
    source_type: "current_session_stream",
    session_scope: "current_session",
    actor: "user",
    trust_rank: 95,
    text: input.text,
    taint: "none",
    stream_index: input.streamIndex,
  };
}

function evidenceLedger(entries: readonly EvidenceLedgerEntry[]): EvidenceLedger {
  return {
    transcriptIncluded: true,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
    sections: [
      {
        id: "current_session_transcript",
        label: "2. Current-Session Transcript",
        entries: [...entries],
      },
    ],
  };
}

describe("compileDecisionArtifact", () => {
  let db: SqliteDatabase;
  let repository: DecisionArtifactRepository;
  let clock: FixedClock;
  let audience: EntityId;
  let self: EntityId;
  let alice: EntityId;
  let currentStreamEntryId: StreamEntryId;

  beforeEach(() => {
    db = openDatabase(":memory:", {
      migrations: composeMigrations(decisionArtifactMigrations, selfMigrations),
    });
    clock = new FixedClock(2_000);
    repository = new DecisionArtifactRepository({
      db,
      clock,
    });
    audience = createEntityId();
    self = createEntityId();
    alice = createEntityId();
    currentStreamEntryId = createStreamEntryId();
  });

  afterEach(() => {
    db.close();
  });

  function baseInput(llmClient: FakeLLMClient) {
    return {
      llmClient,
      model: "claude-haiku-test",
      repository,
      audienceEntityId: audience,
      selfEntityId: self,
      speakerEntityId: alice,
      participants: [{ entityId: alice, displayName: "Alice" }],
      currentUserMessage: "Madrid 3, SS 3, Seville 4, Granada 3 is locked.",
      currentUserStreamEntryId: currentStreamEntryId,
      promptVisibleLedger: "Commitments: route order confirmed.",
      allowedSourceStreamEntryIds: [currentStreamEntryId],
      clock,
      turnId: "turn_decision_artifact_test",
    };
  }

  function activeEntries() {
    return (repository.get(audience)?.entries ?? []).filter(
      (entry) => entry.superseded_by_id === null,
    );
  }

  it("adds a locked decision emitted by the LLM", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact(baseInput(llmClient));

    const artifact = repository.get(audience);
    expect(artifact?.entries).toHaveLength(1);
    expect(artifact?.entries[0]).toMatchObject({
      kind: "locked",
      text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
      owner_entity_id: audience,
      provenance_stream_entry_ids: [currentStreamEntryId],
    });
    expect(llmClient.requests[0]).toMatchObject({
      model: "claude-haiku-test",
      max_tokens: 1536,
      budget: "decision-artifact-compiler",
      tool_choice: { type: "tool", name: DECISION_ARTIFACT_TOOL_NAME },
    });
  });

  it("retains valid canonicalization ids in the normalized patch", async () => {
    const goalId = createGoalId();
    const commitmentId = createCommitmentId();
    const actionId = createActionId();
    const openQuestionId = createOpenQuestionId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Granada is locked for 3 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goalId],
                commitment_ids: [commitmentId],
                action_ids: [actionId],
                open_question_ids: [openQuestionId],
              },
            },
          ],
        }),
      ],
    });

    const patch = await compileDecisionArtifact({
      ...baseInput(llmClient),
      canonicalizationCandidates: {
        goals: [{ id: goalId, text: "Lock Granada for 3 nights" }],
        commitments: [{ id: commitmentId, text: "Remember Granada is locked" }],
        actions: [{ id: actionId, text: "Track Granada planning" }],
        openQuestions: [{ id: openQuestionId, text: "Is Granada final?" }],
      },
    });

    expect(patch.operations[0]).toMatchObject({
      canonicalizes: {
        goal_ids: [goalId],
        commitment_ids: [commitmentId],
        action_ids: [actionId],
        open_question_ids: [openQuestionId],
      },
    });
    expect(repository.get(audience)?.entries[0]?.canonicalizes).toEqual({
      goal_ids: [goalId],
      commitment_ids: [commitmentId],
      action_ids: [actionId],
      open_question_ids: [openQuestionId],
    });
  });

  it("drops invalid canonicalization ids and reports them in reconciliation trace", async () => {
    const trace = createTraceRecorder();
    const unknownGoalId = createGoalId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Granada is locked for 3 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: ["goal_invalid", unknownGoalId],
              },
            },
          ],
        }),
      ],
    });

    const patch = await compileDecisionArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      canonicalizationCandidates: {
        goals: [],
      },
    });

    expect(patch.operations[0]).toMatchObject({
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [],
        action_ids: [],
        open_question_ids: [],
      },
    });
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "decision_artifact_reconciliation_completed",
        data: expect.objectContaining({
          goals_retired: 0,
          unknown_ids: [
            expect.objectContaining({
              channel: "goal",
              id: "goal_invalid",
              reason: "invalid_id",
            }),
            expect.objectContaining({
              channel: "goal",
              id: unknownGoalId,
              reason: "unknown_id",
            }),
          ] satisfies JsonValue,
        }),
      }),
    );
  });

  it("drops duplicate canonicalization ids across artifact operations before persisting", async () => {
    const trace = createTraceRecorder();
    const goalId = createGoalId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Granada is locked for 3 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goalId],
              },
            },
            {
              type: "add",
              kind: "locked",
              text: "Granada nights are canonical",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goalId],
              },
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      canonicalizationCandidates: {
        goals: [{ id: goalId, text: "Lock Granada for 3 nights" }],
      },
    });

    const entries = repository.get(audience)?.entries ?? [];
    expect(entries).toHaveLength(2);
    expect(entries[0]?.canonicalizes.goal_ids).toEqual([goalId]);
    expect(entries[1]?.canonicalizes.goal_ids).toEqual([]);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "decision_artifact_reconciliation_completed",
        data: expect.objectContaining({
          canonicalization_duplicates_dropped: [
            expect.objectContaining({
              kind: "locked",
              dropped_ids: expect.objectContaining({
                goal_ids: [goalId],
              }),
            }),
          ],
        }),
      }),
    );
  });

  it("lets a surviving entry keep canonicalization ids claimed by a pruned entry", async () => {
    const trace = createTraceRecorder();
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const goal = goalsRepository.add({
      description: "Lock Granada for 3 nights",
      priority: 1,
      provenance: {
        kind: "online",
        process: "test",
      },
      audienceEntityId: audience,
      sourceStreamEntryIds: [currentStreamEntryId],
    });
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Older Granada lock duplicate",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goal.id],
              },
            },
            {
              type: "add",
              kind: "locked",
              text: "Surviving Granada lock",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goal.id],
              },
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      tracer: trace,
      canonicalizationCandidates: {
        goals: [{ id: goal.id, text: goal.description }],
      },
      reconciliation: {
        goalsRepository,
      },
      lifecycle: {
        maxActiveEntries: 1,
      },
    });

    const entries = repository.get(audience)?.entries ?? [];
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      text: "Surviving Granada lock",
      canonicalizes: {
        goal_ids: [goal.id],
      },
    });
    expect(goalsRepository.get(goal.id)).toMatchObject({
      status: "done",
      canonicalized_by_artifact_entry_id: entries[0]?.id,
    });
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "decision_artifact_reconciliation_completed",
        data: expect.objectContaining({
          canonicalization_duplicates_dropped: [],
        }),
      }),
    );
  });

  it("canonicalizes a locked entry and retires an active goal after upsert", async () => {
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const goal = goalsRepository.add({
      description: "Lock Granada for 3 nights",
      priority: 1,
      provenance: {
        kind: "online",
        process: "test",
      },
      audienceEntityId: audience,
      sourceStreamEntryIds: [currentStreamEntryId],
    });
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Granada is locked for 3 nights",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
              canonicalizes: {
                goal_ids: [goal.id],
              },
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      canonicalizationCandidates: {
        goals: [{ id: goal.id, text: goal.description }],
      },
      reconciliation: {
        goalsRepository,
      },
    });

    const artifactEntry = repository.get(audience)?.entries[0];
    expect(goalsRepository.get(goal.id)).toMatchObject({
      status: "done",
      canonicalized_by_artifact_entry_id: artifactEntry?.id,
    });
  });

  it("rejects an invalid owner entity id with a traced reason", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
              owner_entity_id: createEntityId(),
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(repository.get(audience)).toBeNull();
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "decision_artifact_compile_completed",
        data: expect.objectContaining({
          rejectedCount: 1,
          rejectionReasons: ["invalid_owner_entity_id"] satisfies JsonValue,
          applied: false,
        }),
      }),
    );
  });

  it("supersedes an existing entry with a replacement entry", async () => {
    const firstSource = createStreamEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          kind: "locked",
          text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 2",
          provenance_stream_entry_ids: [firstSource],
        },
      ],
      {
        lastCompiledStreamEntryId: firstSource,
      },
    );
    const oldEntryId = initial?.entries[0]?.id;

    expect(oldEntryId).toBeDefined();

    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "supersede",
              id: oldEntryId!,
              replacement: {
                kind: "locked",
                text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
                owner_entity_id: audience,
                source_stream_entry_ids: [currentStreamEntryId],
              },
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [firstSource, currentStreamEntryId],
    });

    const artifact = repository.get(audience);
    const oldEntry = artifact?.entries.find((entry) => entry.id === oldEntryId);
    const replacement = artifact?.entries.find((entry) => entry.id !== oldEntryId);

    expect(artifact?.entries).toHaveLength(2);
    expect(oldEntry?.superseded_by_id).toBe(replacement?.id);
    expect(replacement).toMatchObject({
      kind: "locked",
      text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
      provenance_stream_entry_ids: [currentStreamEntryId],
    });
  });

  it("skips gracefully when the LLM call fails", async () => {
    const onDegraded = vi.fn();
    const llmClient = new FakeLLMClient({
      responses: [throwingResponse],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      onDegraded,
    });

    expect(repository.get(audience)).toBeNull();
    expect(onDegraded).toHaveBeenCalledWith("llm_failed", expect.any(Error));
  });

  it("advances compile metadata and record version for a no-op compile", async () => {
    const initial = repository.upsert(audience, [
      {
        type: "add",
        kind: "live",
        text: "Question: Granada pacing",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const llmClient = new FakeLLMClient({
      responses: [emitDecisionArtifactPatchResponse({ operations: [] })],
    });

    await compileDecisionArtifact(baseInput(llmClient));

    expect(repository.get(audience)?.record_version).toBe((initial?.record_version ?? 0) + 1);
    expect(repository.get(audience)?.last_compiled_stream_entry_id).toBe(currentStreamEntryId);
  });

  it("creates an empty artifact on a first no-op compile so later turns can delta from it", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const llmClient = new FakeLLMClient({
      responses: [emitDecisionArtifactPatchResponse({ operations: [] })],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      currentUserStreamEntryId: firstSource,
      allowedSourceStreamEntryIds: [firstSource],
    });

    const artifact = repository.get(audience);
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: firstSource, streamIndex: 0, text: "first no-op turn" }),
      ledgerEntry({ streamEntryId: secondSource, streamIndex: 1, text: "second turn" }),
    ]);
    const context = buildDecisionArtifactLedgerPromptContext({
      ledger,
      previousArtifact: artifact,
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 1,
    });

    expect(artifact).toMatchObject({
      record_version: 1,
      last_compiled_stream_entry_id: firstSource,
      entries: [],
    });
    expect(context.ledgerMode).toBe("delta");
    expect(context.promptVisibleLedger).not.toContain("first no-op turn");
    expect(context.promptVisibleLedger).toContain("second turn");
  });

  it("advances no-op compile anchors so the next ledger delta starts after the no-op turn", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const thirdSource = createStreamEntryId();
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "live",
              text: "Live planning decision",
              owner_entity_id: audience,
              source_stream_entry_ids: [firstSource],
            },
          ],
        }),
        emitDecisionArtifactPatchResponse({ operations: [] }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      currentUserStreamEntryId: firstSource,
      allowedSourceStreamEntryIds: [firstSource],
    });
    const afterFirst = repository.get(audience);

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      currentUserStreamEntryId: secondSource,
      allowedSourceStreamEntryIds: [firstSource, secondSource],
    });

    const afterNoOp = repository.get(audience);
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: firstSource, streamIndex: 0, text: "first compile turn" }),
      ledgerEntry({ streamEntryId: secondSource, streamIndex: 1, text: "no-op compile turn" }),
      ledgerEntry({ streamEntryId: thirdSource, streamIndex: 2, text: "third compile turn" }),
    ]);
    const context = buildDecisionArtifactLedgerPromptContext({
      ledger,
      previousArtifact: afterNoOp,
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 1,
    });

    expect(afterNoOp?.record_version).toBe((afterFirst?.record_version ?? 0) + 1);
    expect(afterNoOp?.last_compiled_stream_entry_id).toBe(secondSource);
    expect(context.ledgerMode).toBe("delta");
    expect(context.promptVisibleLedger).not.toContain("first compile turn");
    expect(context.promptVisibleLedger).not.toContain("no-op compile turn");
    expect(context.promptVisibleLedger).toContain("third compile turn");
  });

  it("sends a summarized previous artifact instead of the full artifact JSON", async () => {
    repository.upsert(audience, [
      {
        type: "add",
        kind: "live",
        text: "Live planning decision",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const llmClient = new FakeLLMClient({
      responses: [emitDecisionArtifactPatchResponse({ operations: [] })],
    });

    await compileDecisionArtifact(baseInput(llmClient));

    const prompt = JSON.parse(llmClient.requests[0]?.messages[0]?.content ?? "{}") as {
      previous_artifact?: unknown;
      previous_artifact_summary?: {
        active_entries?: {
          live?: Array<{ text: string }>;
        };
      };
    };

    expect(prompt.previous_artifact).toBeUndefined();
    expect(prompt.previous_artifact_summary?.active_entries?.live).toEqual([
      expect.objectContaining({
        text: "Live planning decision",
      }),
    ]);
  });

  it("warns when the compiler input estimate exceeds the prompt budget", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [emitDecisionArtifactPatchResponse({ operations: [] })],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      promptVisibleLedger: "large ledger entry ".repeat(180_000),
      tracer: trace,
      ledgerMode: "delta",
    });

    const warning = trace.events.find(
      (event) => event.event === "decision_artifact_compile_over_budget",
    );
    const completed = trace.events.find(
      (event) => event.event === "decision_artifact_compile_completed",
    );

    expect(warning).toBeDefined();
    expect(warning?.data.ledger_mode).toBe("delta");
    expect(typeof warning?.data.input_token_estimate).toBe("number");
    expect(warning?.data.input_token_estimate as number).toBeGreaterThan(35_000);
    expect(warning?.data.breakdown).toEqual(
      expect.objectContaining({
        prompt_visible_ledger: expect.any(Number),
      }),
    );
    expect(completed?.data).toEqual(
      expect.objectContaining({
        ledger_mode: "delta",
        input_token_estimate: warning?.data.input_token_estimate,
      }),
    );
  });

  it("enforces the active-entry lifecycle budget on a no-op patch", async () => {
    const source = createStreamEntryId();

    repository.upsert(
      audience,
      Array.from({ length: 50 }, (_, index) => ({
        type: "add" as const,
        kind: "locked" as const,
        text: `Locked planning entry ${index}`,
        provenance_stream_entry_ids: [source],
        created_at: 1_000 + index,
        last_updated_at: 1_000 + index,
        rank: index,
      })),
      {
        lastCompiledStreamEntryId: source,
      },
    );

    const llmClient = new FakeLLMClient({
      responses: [emitDecisionArtifactPatchResponse({ operations: [] })],
    });
    const trace = createTraceRecorder();
    const patch = await compileDecisionArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [source, currentStreamEntryId],
      tracer: trace,
    });

    expect(activeEntries()).toHaveLength(40);
    expect(patch.operations.filter((operation) => operation.type === "prune")).toHaveLength(10);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "decision_artifact_compile_completed",
        data: expect.objectContaining({
          artifact_total_entry_count: 40,
          artifact_active_entry_count: 40,
          artifact_omitted_entry_count: 26,
          artifact_pruned_entry_count_this_turn: 10,
          artifact_superseded_count_this_turn: 0,
          rendered_by_kind: expect.objectContaining({
            locked: 14,
          }),
        }),
      }),
    );
  });

  it("keeps long compile sequences inside the active artifact budget while reserving live render slots", async () => {
    const maxActiveEntries = 40;
    const liveRenderReservation = 8;
    const responses = Array.from({ length: 60 }, (_, index) =>
      emitDecisionArtifactPatchResponse({
        operations: [
          {
            type: "add",
            kind: "locked",
            text: `Locked long-plan route invariant ${index}`,
            owner_entity_id: audience,
            source_stream_entry_ids: [currentStreamEntryId],
          },
          {
            type: "add",
            kind: "live",
            text: `Live long-plan detail ${index}`,
            owner_entity_id: audience,
            source_stream_entry_ids: [currentStreamEntryId],
          },
          {
            type: "add",
            kind: "pending",
            text: `Pending long-plan decision ${index}`,
            owner_entity_id: audience,
            source_stream_entry_ids: [currentStreamEntryId],
          },
        ],
      }),
    );
    const llmClient = new FakeLLMClient({ responses });
    const trace = createTraceRecorder();
    let sawRenderedOmission = false;
    let sawLifecyclePrune = false;

    for (let index = 0; index < responses.length; index += 1) {
      await compileDecisionArtifact({
        ...baseInput(llmClient),
        currentUserMessage: `Long planning turn ${index}`,
        tracer: trace,
      });

      const artifact = repository.get(audience);
      const active = activeEntries();
      const activeLive = active.filter((entry) => entry.kind === "live");
      const expectedLiveEntries = [...activeLive]
        .sort(
          (left, right) =>
            right.last_updated_at - left.last_updated_at ||
            left.rank - right.rank ||
            right.created_at - left.created_at ||
            left.id.localeCompare(right.id),
        )
        .slice(0, Math.min(activeLive.length, liveRenderReservation));
      const rendered = renderDecisionStateArtifact(artifact) ?? "";
      const renderedLiveEntryCount = rendered.match(/kind=live/g)?.length ?? 0;
      const completed = trace.events
        .filter((event) => event.event === "decision_artifact_compile_completed")
        .at(-1);
      const artifactTotalEntryCount = completed?.data.artifact_total_entry_count;
      const artifactActiveEntryCount = completed?.data.artifact_active_entry_count;
      const artifactOmittedEntryCount = completed?.data.artifact_omitted_entry_count;
      const artifactRenderedEntryCount = completed?.data.artifactEntryCount;
      const artifactPrunedEntryCount = completed?.data.artifact_pruned_entry_count_this_turn;

      expect(active.length).toBeLessThanOrEqual(maxActiveEntries);
      expect(renderedLiveEntryCount).toBeGreaterThanOrEqual(expectedLiveEntries.length);
      expect(expectedLiveEntries.every((entry) => rendered.includes(entry.text))).toBe(true);
      expect(completed?.data).toEqual(
        expect.objectContaining({
          artifact_total_entry_count: expect.any(Number),
          artifact_active_entry_count: expect.any(Number),
          artifact_omitted_entry_count: expect.any(Number),
          artifact_pruned_entry_count_this_turn: expect.any(Number),
          artifact_superseded_count_this_turn: expect.any(Number),
          rendered_by_kind: expect.any(Object),
        }),
      );
      expect(artifactTotalEntryCount as number).toBeGreaterThanOrEqual(active.length);
      expect(artifactActiveEntryCount as number).toBe(active.length);
      if (typeof artifactOmittedEntryCount === "number" && artifactOmittedEntryCount > 0) {
        sawRenderedOmission = true;
      }
      if (typeof artifactPrunedEntryCount === "number" && artifactPrunedEntryCount > 0) {
        sawLifecyclePrune = true;
      }
      if (
        typeof artifactRenderedEntryCount === "number" &&
        typeof artifactOmittedEntryCount === "number" &&
        active.length > artifactRenderedEntryCount
      ) {
        expect(artifactOmittedEntryCount).toBe(active.length - artifactRenderedEntryCount);
      }
    }

    expect(sawRenderedOmission).toBe(true);
    expect(sawLifecyclePrune).toBe(true);
  });

  it("prunes superseded dependencies before pruning a referenced replacement", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const extraSource = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        kind: "locked",
        text: "Original locked route",
        provenance_stream_entry_ids: [firstSource],
        created_at: 1_000,
        last_updated_at: 1_000,
        rank: 0,
      },
    ]);
    const originalId = initial?.entries[0]?.id;

    expect(originalId).toBeDefined();
    const superseded = repository.upsert(audience, [
      {
        type: "supersede",
        id: originalId!,
        replacement: {
          kind: "locked",
          text: "Replacement locked route",
          provenance_stream_entry_ids: [secondSource],
          created_at: 1_100,
          last_updated_at: 1_100,
          rank: 1,
        },
        last_updated_stream_entry_ids: [secondSource],
      },
      {
        type: "add",
        kind: "locked",
        text: "Extra locked route 1",
        provenance_stream_entry_ids: [extraSource],
        created_at: 2_000,
        last_updated_at: 2_000,
        rank: 2,
      },
      {
        type: "add",
        kind: "locked",
        text: "Extra locked route 2",
        provenance_stream_entry_ids: [extraSource],
        created_at: 3_000,
        last_updated_at: 3_000,
        rank: 3,
      },
    ]);
    const replacementId = superseded?.entries.find((entry) => entry.id !== originalId)?.id;

    expect(replacementId).toBeDefined();

    const llmClient = new FakeLLMClient({
      responses: [emitDecisionArtifactPatchResponse({ operations: [] })],
    });

    const patch = await compileDecisionArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [firstSource, secondSource, extraSource, currentStreamEntryId],
      lifecycle: {
        maxActiveEntries: 1,
        kindSoftCaps: {
          locked: 0,
        },
      },
    });

    const artifact = repository.get(audience);

    expect(activeEntries()).toHaveLength(1);
    expect(activeEntries()[0]?.text).toBe("Extra locked route 2");
    expect(artifact?.entries.find((entry) => entry.id === originalId)).toBeUndefined();
    expect(artifact?.entries.find((entry) => entry.id === replacementId)).toBeUndefined();
    expect(
      patch.operations
        .filter((operation) => operation.type === "prune")
        .map((operation) => operation.id),
    ).toEqual(expect.arrayContaining([originalId, replacementId]));
  });

  it("keeps the active-entry cap hard when every replacement has a superseded referrer", async () => {
    const source = createStreamEntryId();
    const originalIds: string[] = [];
    const replacementIds: string[] = [];

    for (let index = 0; index < 42; index += 1) {
      const originalText = `Original locked route ${index}`;
      const replacementText = `Replacement locked route ${index}`;
      const original = repository.upsert(audience, [
        {
          type: "add",
          kind: "locked",
          text: originalText,
          provenance_stream_entry_ids: [source],
          created_at: 1_000 + index,
          last_updated_at: 1_000 + index,
          rank: index,
        },
      ]);
      const originalId = original?.entries.find((entry) => entry.text === originalText)?.id;

      expect(originalId).toBeDefined();

      const superseded = repository.upsert(audience, [
        {
          type: "supersede",
          id: originalId!,
          replacement: {
            kind: "locked",
            text: replacementText,
            provenance_stream_entry_ids: [source],
            created_at: 10_000 + index,
            last_updated_at: 10_000 + index,
            rank: index,
          },
          last_updated_stream_entry_ids: [source],
        },
      ]);
      const replacementId = superseded?.entries.find((entry) => entry.text === replacementText)?.id;

      expect(replacementId).toBeDefined();
      originalIds.push(originalId!);
      replacementIds.push(replacementId!);
    }

    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [emitDecisionArtifactPatchResponse({ operations: [] })],
    });
    const patch = await compileDecisionArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [source, currentStreamEntryId],
      tracer: trace,
      lifecycle: {
        maxActiveEntries: 40,
        kindSoftCaps: {
          locked: 40,
        },
      },
    });
    const pruneIds = patch.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => operation.id);

    expect(activeEntries()).toHaveLength(40);
    expect(pruneIds).toHaveLength(4);
    for (const replacementId of replacementIds.filter((id) => pruneIds.includes(id))) {
      const originalId = originalIds[replacementIds.indexOf(replacementId)];

      expect(pruneIds.indexOf(originalId!)).toBeGreaterThanOrEqual(0);
      expect(pruneIds.indexOf(originalId!)).toBeLessThan(pruneIds.indexOf(replacementId));
    }
    expect(
      trace.events.find((event) => event.event === "decision_artifact_lifecycle_unable_to_cap"),
    ).toBeUndefined();
  });

  it("expands dependencies for an LLM-emitted prune of a referenced replacement", async () => {
    const firstSource = createStreamEntryId();
    const secondSource = createStreamEntryId();
    const initial = repository.upsert(audience, [
      {
        type: "add",
        kind: "locked",
        text: "Original locked route",
        provenance_stream_entry_ids: [firstSource],
        created_at: 1_000,
        last_updated_at: 1_000,
        rank: 0,
      },
    ]);
    const originalId = initial?.entries[0]?.id;

    expect(originalId).toBeDefined();

    const superseded = repository.upsert(audience, [
      {
        type: "supersede",
        id: originalId!,
        replacement: {
          kind: "locked",
          text: "Replacement locked route",
          provenance_stream_entry_ids: [secondSource],
          created_at: 1_100,
          last_updated_at: 1_100,
          rank: 1,
        },
        last_updated_stream_entry_ids: [secondSource],
      },
    ]);
    const replacementId = superseded?.entries.find((entry) => entry.id !== originalId)?.id;

    expect(replacementId).toBeDefined();

    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "prune",
              id: replacementId!,
            },
          ],
        }),
      ],
    });
    const patch = await compileDecisionArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [firstSource, secondSource, currentStreamEntryId],
    });
    const pruneIds = patch.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => operation.id);
    const artifact = repository.get(audience);

    expect(pruneIds).toEqual([originalId, replacementId]);
    expect(artifact?.entries.find((entry) => entry.id === originalId)).toBeUndefined();
    expect(artifact?.entries.find((entry) => entry.id === replacementId)).toBeUndefined();
  });

  it("accepts all decision artifact kinds emitted by the compiler", async () => {
    const kinds = [
      "locked",
      "live",
      "tentative",
      "invalidated",
      "pending",
    ] as const satisfies readonly DecisionArtifactEntryKind[];
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: kinds.map((kind) => ({
            type: "add" as const,
            kind,
            text: `Artifact entry kind ${kind}`,
            owner_entity_id: audience,
            source_stream_entry_ids: [currentStreamEntryId],
          })),
        }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(
      repository
        .get(audience)
        ?.entries.map((entry) => entry.kind)
        .sort(),
    ).toEqual([...kinds].sort());
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "decision_artifact_compile_completed",
        data: expect.objectContaining({
          rejectedCount: 0,
          rejectionReasons: [],
          applied: true,
        }),
      }),
    );
  });
});
