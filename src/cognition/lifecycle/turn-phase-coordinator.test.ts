import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type {
  DecisionArtifact,
  DecisionArtifactEntry,
  DecisionArtifactEntryKind,
} from "../../memory/decision-artifacts/index.js";
import {
  createDecisionArtifactEntryId,
  createEntityId,
  createGoalId,
  createStreamEntryId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { EvidenceLedger, EvidenceLedgerEntry } from "../evidence-ledger/index.js";
import { renderEvidenceLedger } from "../evidence-ledger/index.js";
import {
  DECISION_ARTIFACT_TOOL_NAME,
  findUnsettledDecisionArtifactReconciliation,
} from "../decision-artifact/index.js";
import {
  buildDecisionArtifactLedgerPromptContext,
  shouldSkipDecisionArtifactCompile,
  TurnPhaseCoordinator,
} from "./turn-phase-coordinator.js";

function decisionArtifactEntry(input: {
  audience: DecisionArtifact["audience_entity_id"];
  kind: DecisionArtifactEntryKind;
  source: StreamEntryId;
  index?: number;
  canonicalizes?: DecisionArtifactEntry["canonicalizes"];
}): DecisionArtifactEntry {
  const index = input.index ?? 0;

  return {
    id: createDecisionArtifactEntryId(),
    audience_entity_id: input.audience,
    kind: input.kind,
    text: `${input.kind} decision`,
    owner_entity_id: input.audience,
    provenance_stream_entry_ids: [input.source],
    last_updated_stream_entry_ids: [input.source],
    created_at: 1_000 + index,
    last_updated_at: 1_000 + index,
    superseded_by_id: null,
    rank: index,
    canonicalizes: input.canonicalizes ?? {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [],
      open_question_ids: [],
    },
  };
}

function decisionArtifact(input: {
  entries?: readonly DecisionArtifactEntry[];
  lastCompiledStreamEntryId?: StreamEntryId | null;
}): DecisionArtifact {
  const source = input.lastCompiledStreamEntryId ?? createStreamEntryId();
  const audience = input.entries?.[0]?.audience_entity_id ?? createEntityId();

  return {
    audience_entity_id: audience,
    record_version: 1,
    created_at: 1_000,
    updated_at: 1_000,
    last_compiled_at: 1_000,
    last_compiled_stream_entry_id: source,
    entries: [...(input.entries ?? [])],
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

describe("shouldSkipDecisionArtifactCompile", () => {
  it("does not skip frame-anomaly turns", () => {
    const skip = shouldSkipDecisionArtifactCompile({
      enabled: true,
      previousArtifact: null,
      perceptionMode: "problem_solving",
      frameAnomaly: {
        status: "ok",
        kind: "frame_assignment_claim",
        confidence: 1,
        rationale: "test",
      },
      closureLoopAssessment: null,
    });

    expect(skip).toBeNull();
  });

  it("skips idle turns when the previous artifact has no active in-flight decisions", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const previousArtifact = decisionArtifact({
      entries: [decisionArtifactEntry({ audience, source, kind: "locked" })],
      lastCompiledStreamEntryId: source,
    });
    const skip = shouldSkipDecisionArtifactCompile({
      enabled: true,
      previousArtifact,
      perceptionMode: "idle",
      frameAnomaly: null,
      closureLoopAssessment: null,
    });

    expect(skip).toMatchObject({
      reason: "idle_no_active_decisions",
      previousActiveEntryCount: 1,
      perceptionMode: "idle",
    });
  });

  it("does not skip idle turns when a live decision is active", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const previousArtifact = decisionArtifact({
      entries: [decisionArtifactEntry({ audience, source, kind: "live" })],
      lastCompiledStreamEntryId: source,
    });

    expect(
      shouldSkipDecisionArtifactCompile({
        enabled: true,
        previousArtifact,
        perceptionMode: "idle",
        frameAnomaly: null,
        closureLoopAssessment: null,
      }),
    ).toBeNull();
  });

  it("does not skip when a locked canonicalized goal is still unsettled", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const goalId = createGoalId();
    const previousArtifact = decisionArtifact({
      entries: [
        decisionArtifactEntry({
          audience,
          source,
          kind: "locked",
          canonicalizes: {
            goal_ids: [goalId],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        }),
      ],
      lastCompiledStreamEntryId: source,
    });
    const unsettledReconciliation = findUnsettledDecisionArtifactReconciliation({
      previousArtifact,
      repositories: {
        goalsRepository: {
          get: (id) =>
            id === goalId
              ? ({
                  id,
                  status: "active",
                } as never)
              : null,
        },
        commitmentRepository: { get: () => null },
        actionRepository: { get: () => null },
        openQuestionsRepository: { get: () => null },
      },
    });
    const skip = shouldSkipDecisionArtifactCompile({
      enabled: true,
      previousArtifact,
      perceptionMode: "idle",
      frameAnomaly: null,
      closureLoopAssessment: {
        closureLoopDetected: true,
        currentUserAct: "signoff",
        currentUserClosureShaped: true,
        currentUserSubstantive: false,
        mutualClosureCycles: 1,
        sourceStreamEntryIds: [source],
        reason: "test",
      },
      unsettledReconciliation: unsettledReconciliation?.summary ?? null,
    });

    expect(unsettledReconciliation?.summary).toMatchObject({
      active_locked_canonicalizing_entry_count: 1,
      referenced_goal_count: 1,
      unsettled_goal_count: 1,
      unsettled_total_count: 1,
    });
    expect(skip).toBeNull();
  });
});

describe("TurnPhaseCoordinator decision artifact prefilter", () => {
  it("emits an unblocked trace and compiles when canonicalized goal reconciliation is unsettled", async () => {
    const audience = createEntityId();
    const self = createEntityId();
    const source = createStreamEntryId();
    const current = createStreamEntryId();
    const goalId = createGoalId();
    let artifact: DecisionArtifact | null = decisionArtifact({
      entries: [
        decisionArtifactEntry({
          audience,
          source,
          kind: "locked",
          canonicalizes: {
            goal_ids: [goalId],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        }),
      ],
      lastCompiledStreamEntryId: source,
    });
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const llmClient = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_decision_patch",
              name: DECISION_ARTIFACT_TOOL_NAME,
              input: { operations: [] },
            },
          ],
        },
      ],
    });
    const coordinator = new TurnPhaseCoordinator({
      config: DEFAULT_CONFIG,
      decisionArtifactRepository: {
        get: () => artifact,
        upsert: (
          _audienceEntityId: unknown,
          _operations: unknown,
          metadata:
            | {
                now?: number;
                lastCompiledAt?: number;
                lastCompiledStreamEntryId?: StreamEntryId;
              }
            | undefined,
        ) => {
          artifact =
            artifact === null
              ? null
              : {
                  ...artifact,
                  record_version: artifact.record_version + 1,
                  updated_at: metadata?.now ?? artifact.updated_at,
                  last_compiled_at: metadata?.lastCompiledAt ?? artifact.last_compiled_at,
                  last_compiled_stream_entry_id:
                    metadata?.lastCompiledStreamEntryId ??
                    artifact.last_compiled_stream_entry_id,
                };

          return artifact;
        },
      },
      goalsRepository: {
        get: (id: string) =>
          id === goalId
            ? ({
                id,
                status: "active",
              } as never)
            : null,
        list: () => [],
        updateStatus: () => ({}),
      },
      commitmentRepository: {
        get: () => null,
        list: () => [],
      },
      actionRepository: {
        get: () => null,
        list: () => [],
      },
      openQuestionsRepository: {
        get: () => null,
        list: () => [],
      },
      entityRepository: {
        resolve: () => self,
      },
      llmFactory: () => llmClient,
      clock: {
        now: () => 2_000,
      },
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: (event: string, data: Record<string, unknown>) => {
          events.push({ event, data });
        },
      },
    } as never);
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: source, streamIndex: 0, text: "anchor planning turn" }),
      ledgerEntry({ streamEntryId: current, streamIndex: 1, text: "closure-shaped turn" }),
    ]);

    await (
      coordinator as unknown as {
        compileDecisionArtifactForEvidenceLedger(input: {
          input: Record<string, unknown>;
          ledger: EvidenceLedger;
          promptVisibleLedger: string;
        }): Promise<DecisionArtifact | null>;
      }
    ).compileDecisionArtifactForEvidenceLedger({
      input: {
        audienceEntityId: audience,
        isUserTurn: true,
        currentUserEntry: {
          id: current,
          sender_entity_id: null,
        },
        currentUserMessage: "closure-shaped turn",
        perception: {
          mode: "idle",
        },
        frameAnomaly: null,
        closureLoopAssessment: {
          closureLoopDetected: true,
          currentUserAct: "signoff",
          currentUserClosureShaped: true,
          currentUserSubstantive: false,
          mutualClosureCycles: 1,
          sourceStreamEntryIds: [current],
          reason: "test",
        },
        activeParticipants: [],
        turnId: "turn_unsettled_reconciliation",
      },
      ledger,
      promptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
    });

    expect(llmClient.requests).toHaveLength(1);
    expect(events.find((event) => event.event === "decision_artifact_compile_skipped")).toBe(
      undefined,
    );
    expect(events).toContainEqual(
      expect.objectContaining({
        event: "decision_artifact_compile_unblocked",
        data: expect.objectContaining({
          decision_artifact_compile_unblocked_reason: "unsettled_reconciliation",
          unsettled_goal_count: 1,
          unsettled_total_count: 1,
        }),
      }),
    );
  });
});

describe("buildDecisionArtifactLedgerPromptContext", () => {
  it("renders only delta ledger entries after the previous compile anchor", () => {
    const older = createStreamEntryId();
    const anchor = createStreamEntryId();
    const nextOne = createStreamEntryId();
    const nextTwo = createStreamEntryId();
    const current = createStreamEntryId();
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: older, streamIndex: 0, text: "older transcript" }),
      ledgerEntry({ streamEntryId: anchor, streamIndex: 1, text: "anchor transcript" }),
      ledgerEntry({ streamEntryId: nextOne, streamIndex: 2, text: "new transcript one" }),
      ledgerEntry({ streamEntryId: nextTwo, streamIndex: 3, text: "new transcript two" }),
      ledgerEntry({ streamEntryId: current, streamIndex: 4, text: "current transcript" }),
    ]);
    const context = buildDecisionArtifactLedgerPromptContext({
      ledger,
      previousArtifact: decisionArtifact({ lastCompiledStreamEntryId: anchor }),
      fullPromptVisibleLedger: renderEvidenceLedger(ledger) ?? "",
      enabled: true,
      minTailPerSection: 3,
    });

    expect(context.ledgerMode).toBe("delta");
    expect(context.promptVisibleLedger).not.toContain("older transcript");
    expect(context.promptVisibleLedger).not.toContain("anchor transcript");
    expect(context.promptVisibleLedger).toContain("new transcript one");
    expect(context.promptVisibleLedger).toContain("new transcript two");
    expect(context.promptVisibleLedger).toContain("current transcript");
    expect(context.visibleStreamEntryIds).toEqual([nextOne, nextTwo, current]);
  });

  it("falls back to the full ledger when the previous compile anchor is missing", () => {
    const older = createStreamEntryId();
    const current = createStreamEntryId();
    const ledger = evidenceLedger([
      ledgerEntry({ streamEntryId: older, streamIndex: 0, text: "older transcript" }),
      ledgerEntry({ streamEntryId: current, streamIndex: 1, text: "current transcript" }),
    ]);
    const fullPromptVisibleLedger = renderEvidenceLedger(ledger) ?? "";
    const context = buildDecisionArtifactLedgerPromptContext({
      ledger,
      previousArtifact: decisionArtifact({ lastCompiledStreamEntryId: createStreamEntryId() }),
      fullPromptVisibleLedger,
      enabled: true,
      minTailPerSection: 3,
    });

    expect(context).toEqual({
      promptVisibleLedger: fullPromptVisibleLedger,
      ledgerMode: "full_fallback",
      visibleStreamEntryIds: [older, current],
    });
  });

  it("falls back when the anchor exists only outside the retained current-session window", () => {
    const anchor = createStreamEntryId();
    const retainedOne = createStreamEntryId();
    const retainedTwo = createStreamEntryId();
    const ledger: EvidenceLedger = {
      transcriptIncluded: true,
      transcriptCompacted: true,
      originalTranscriptTokenEstimate: 0,
      compactedTranscriptEntryCount: 2,
      rawPreservedUserTranscriptEntryCount: 0,
      estimatedTokens: 0,
      sections: [
        {
          id: "current_session_transcript",
          label: "2. Current-Session Transcript",
          entries: [
            ledgerEntry({
              streamEntryId: retainedOne,
              streamIndex: 5,
              text: "retained transcript one",
            }),
            ledgerEntry({
              streamEntryId: retainedTwo,
              streamIndex: 6,
              text: "retained transcript two",
            }),
          ],
        },
        {
          id: "retrieved_memory_evidence",
          label: "10. Retrieved Memory Evidence",
          entries: [
            {
              id: "episode:pruned_anchor",
              source_type: "episode",
              session_scope: "current_session",
              actor: "memory",
              trust_rank: 50,
              text: "side metadata for pruned anchor",
              taint: "none",
              stream_index: 1,
              state_metadata: {
                source_stream_ids: [anchor],
              },
            },
          ],
        },
      ],
    };
    const fullPromptVisibleLedger = renderEvidenceLedger(ledger) ?? "";
    const context = buildDecisionArtifactLedgerPromptContext({
      ledger,
      previousArtifact: decisionArtifact({ lastCompiledStreamEntryId: anchor }),
      fullPromptVisibleLedger,
      enabled: true,
      minTailPerSection: 1,
    });

    expect(context).toEqual({
      promptVisibleLedger: fullPromptVisibleLedger,
      ledgerMode: "full_fallback",
      visibleStreamEntryIds: [retainedOne, retainedTwo, anchor],
    });
  });
});
