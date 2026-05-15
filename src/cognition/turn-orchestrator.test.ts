import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  Borg,
  ManualClock,
  QUARANTINED_USER_ENTRY_EVENT,
  type FrameAnomalyKind,
  type LLMCompleteOptions,
  type LLMCompleteResult,
} from "../index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import type { BorgDependencies } from "../borg/types.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import { Deliberator, type SelfSnapshot } from "./deliberation/deliberator.js";
import { ActionStateExtractor } from "./actions/action-state-extractor.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import {
  CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
  type ClosureLoopClassifiedMessage,
} from "./generation/closure-loop.js";
import { CLOSURE_RESPONSE_AUDIT_TOOL_NAME } from "./generation/closure-pressure-guard.js";
import { setClosureLoopDetected } from "./generation/discourse-state.js";
import type { Episode, EpisodicRepository } from "../memory/episodic/index.js";
import {
  createTestConfig,
  TestEmbeddingClient,
  type DeepPartial,
} from "../offline/test-support.js";
import type { Config } from "../config/index.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import {
  createEpisodeId,
  createGoalId,
  createStreamEntryId,
  type EntityId,
  type EpisodeId,
} from "../util/ids.js";
import type { IntentRecord } from "./types.js";

type TraceEvent = {
  event: string;
  turnId: string;
  [key: string]: unknown;
};

class CountingEmbeddingClient extends TestEmbeddingClient {
  readonly embedTexts: string[] = [];
  readonly embedBatchTexts: string[][] = [];

  async embed(text: string): Promise<Float32Array> {
    this.embedTexts.push(text);
    return super.embed(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    this.embedBatchTexts.push([...texts]);
    return super.embedBatch(texts);
  }
}

async function openTestBorg(
  tempDir: string,
  llm: FakeLLMClient,
  clock: ManualClock,
  embeddingClient: EmbeddingClient = new TestEmbeddingClient(),
  options: {
    tracerPath?: string;
    env?: NodeJS.ProcessEnv;
    configOverrides?: DeepPartial<Config>;
  } = {},
) {
  const configOverrides = options.configOverrides ?? {};

  return Borg.open({
    config: createTestConfig({
      ...configOverrides,
      dataDir: tempDir,
      perception: {
        useLlmFallback: false,
        ...configOverrides.perception,
      },
      affective: {
        useLlmFallback: false,
        ...configOverrides.affective,
      },
      embedding: {
        baseUrl: "http://localhost:1234/v1",
        apiKey: "test",
        model: "test-embed",
        dims: 4,
        ...configOverrides.embedding,
      },
      anthropic: {
        auth: "api-key",
        apiKey: "test",
        ...configOverrides.anthropic,
        models: {
          cognition: "test-cognition",
          background: "test-background",
          extraction: "test-extraction",
          recallExpansion: "test-recall",
          ...configOverrides.anthropic?.models,
        },
      },
    }),
    clock,
    embeddingDimensions: 4,
    embeddingClient,
    llmClient: llm,
    env: options.env,
    tracerPath: options.tracerPath,
    liveExtraction: false,
  });
}

function readTraceEvents(path: string): TraceEvent[] {
  const content = readFileSync(path, "utf8").trim();

  if (content.length === 0) {
    return [];
  }

  return content
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as TraceEvent);
}

function createEmptyReflectionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
        },
      },
    ],
  };
}

function findReflectionRequest(llm: FakeLLMClient): LLMCompleteOptions | undefined {
  return llm.requests.find((request) => {
    const toolChoice = request.tool_choice;

    return (
      typeof toolChoice === "object" &&
      toolChoice !== null &&
      "name" in toolChoice &&
      toolChoice.name === "EmitTurnReflection"
    );
  });
}

function parseReflectionPayload(request: LLMCompleteOptions | undefined): Record<string, unknown> {
  return JSON.parse(String(request?.messages[0]?.content ?? "{}")) as Record<string, unknown>;
}

function createStopCommitmentResponse(input: {
  classification: "stop_until_substantive_content" | "none";
  reason?: string;
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_stop_commitment",
        name: "EmitStopCommitmentClassification",
        input: {
          classification: input.classification,
          directive_family:
            input.classification === "stop_until_substantive_content"
              ? "stop_until_substantive_content"
              : null,
          reason: input.reason ?? "The assistant committed to stop until substantive content.",
          confidence: 0.94,
        },
      },
    ],
  };
}

function createNoOutputTurnPlanResponse() {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_no_output_plan",
        name: "EmitTurnPlan",
        input: {
          uncertainty: "",
          verification_steps: [],
          tensions: [],
          voice_note: "Hold output because the current turn does not warrant an assistant message.",
          intents: [],
          emission_recommendation: "no_output",
        },
      },
    ],
  };
}

function createTurnPlanResponse(intents: readonly IntentRecord[] = []) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_plan",
        name: "EmitTurnPlan",
        input: {
          uncertainty: "whether the response should continue",
          verification_steps: [],
          tensions: [],
          voice_note: "",
          intents,
        },
      },
    ],
  };
}

function createFinalizerToolResponse(tool: { id: string; name: string; input: unknown }) {
  return {
    text: "",
    input_tokens: 12,
    output_tokens: 6,
    stop_reason: "tool_use" as const,
    tool_calls: [tool],
  };
}

function createEmitAnswerResponse(text: string) {
  return createFinalizerToolResponse({
    id: "toolu_emit_answer",
    name: "EmitAnswer",
    input: { text },
  });
}

function createEmitNoOutputResponse(reason = "No assistant message is needed.") {
  return createFinalizerToolResponse({
    id: "toolu_emit_no_output",
    name: "EmitNoOutput",
    input: { reason },
  });
}

function createEmitObserveResponse(reason = "The participants are talking to each other.") {
  return createFinalizerToolResponse({
    id: "toolu_emit_observe",
    name: "EmitObserve",
    input: { reason },
  });
}

function createPendingActionJudgeResponse(classification: "action" | "non_action") {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_pending_action_judge",
        name: "ClassifyPendingAction",
        input: {
          classification,
          reason: "The planner item is classified for the pending-action store.",
          confidence: 0.95,
        },
      },
    ],
  };
}

function createRecallExpansionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_recall_expansion",
        name: "EmitRecallExpansion",
        input: {
          facets: [],
          named_terms: [],
        },
      },
    ],
  };
}

function createGenerationGateResponse(input: {
  decision: "proceed" | "suppress";
  substantive: boolean;
  reason?: string;
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_generation_gate",
        name: "EmitGenerationGateDecision",
        input: {
          decision: input.decision,
          substantive: input.substantive,
          reason: input.reason ?? "Generation gate classified the turn.",
          confidence: 0.95,
        },
      },
    ],
  };
}

function createFrameAnomalyResponse(input: {
  kind: FrameAnomalyKind;
  confidence?: number;
  rationale?: string;
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_frame_anomaly",
        name: "ClassifyFrameAnomaly",
        input: {
          kind: input.kind,
          confidence: input.confidence ?? (input.kind === "normal" ? 0.9 : 0.96),
          rationale: input.rationale ?? "The frame anomaly classifier categorized the turn.",
        },
      },
    ],
  };
}

function createClosureLoopSignoffResponseFromRequest() {
  return Object.assign(
    (options: LLMCompleteOptions) => {
      const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
        dialogue_window?: unknown;
      };
      const dialogueWindow = Array.isArray(payload.dialogue_window) ? payload.dialogue_window : [];
      const messages: ClosureLoopClassifiedMessage[] = [];

      for (const item of dialogueWindow) {
        if (typeof item !== "object" || item === null) {
          continue;
        }

        const message = item as { message_ref?: unknown; role?: unknown };

        if (
          typeof message.message_ref !== "string" ||
          (message.role !== "user" && message.role !== "assistant")
        ) {
          continue;
        }

        messages.push({
          message_ref: message.message_ref,
          role: message.role,
          act: message.role === "user" ? "signoff" : "assistant_valediction",
        });
      }

      return {
        text: "",
        input_tokens: 4,
        output_tokens: 2,
        stop_reason: "tool_use" as const,
        tool_calls: [
          {
            id: "toolu_closure_loop",
            name: CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
            input: {
              messages,
              confidence: 0.96,
              rationale: "The current user turn is another closure beat.",
            },
          },
        ],
      };
    },
    { budget: "closure-loop-classifier" },
  );
}

function createCommitmentJudgeResponse(
  violations: Array<{ commitment_id: string; reason: string; confidence?: number }>,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_commitment_judge",
        name: "EmitCommitmentViolations",
        input: {
          violations: violations.map((violation) => ({
            commitment_id: violation.commitment_id,
            reason: violation.reason,
            confidence: violation.confidence ?? 0.9,
          })),
        },
      },
    ],
  };
}

function createClosureResponseAuditResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_closure_response_audit",
        name: CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
        input: {
          spans: [],
          response_shape: "no_closure",
          reason: "The response contains no closure-pressure span.",
        },
      },
    ],
  };
}

function createMixedClosureResponseAuditResponse(spanText: string) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_closure_response_audit",
        name: CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
        input: {
          spans: [
            {
              text: spanText,
              kind: "imperative_closer",
              rationale: "Closure tail after substantive content.",
            },
          ],
          response_shape: "mixed",
          reason: "The response contains substantive content plus a closure-pressure span.",
        },
      },
    ],
  };
}

function createCorrectivePreferenceResponse(input: {
  classification: "corrective_preference" | "none";
  type?: "preference" | "rule" | "boundary" | null;
  directive?: string | null;
  directive_family?: string | null;
  closure_pressure_relevance?: "no_closure" | "neutral" | "closure_seeking" | null;
  priority?: number | null;
  reason?: string;
  confidence?: number;
  supersedes_commitment_id?: string | null;
  slot_negations?: unknown[];
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_corrective_preference",
        name: "EmitCorrectivePreference",
        input: {
          classification: input.classification,
          type: input.type ?? null,
          directive: input.directive ?? null,
          directive_family:
            input.directive_family ??
            (input.classification === "corrective_preference" ? "test_directive_family" : null),
          closure_pressure_relevance:
            input.closure_pressure_relevance ??
            (input.classification === "corrective_preference" ? "neutral" : null),
          priority: input.priority ?? null,
          reason: input.reason ?? "The current user turn corrected future response behavior.",
          confidence: input.confidence ?? 0.9,
          supersedes_commitment_id: input.supersedes_commitment_id ?? null,
          slot_negations: input.slot_negations ?? [],
        },
      },
    ],
  };
}

function createGoalPromotionResponse(
  promotions: Array<{
    description: string;
    priority?: number;
    target_at?: number | null;
    reason?: string;
    confidence?: number;
    duplicate_of_goal_id?: string | null;
    initial_step?: {
      description: string;
      kind: "think" | "ask_user" | "research" | "act" | "wait";
      due_at?: number | null;
      rationale: string;
    } | null;
  }>,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_goal_promotion",
        name: "EmitGoalPromotion",
        input: {
          promotions: promotions.map((promotion) => ({
            classification: "promote",
            description: promotion.description,
            priority: promotion.priority ?? 8,
            target_at: promotion.target_at ?? null,
            reason: promotion.reason ?? "The user asked Borg to carry this as an ongoing goal.",
            confidence: promotion.confidence ?? 0.9,
            duplicate_of_goal_id: promotion.duplicate_of_goal_id ?? null,
            initial_step: promotion.initial_step ?? null,
          })),
        },
      },
    ],
  };
}

function createActionStateResponse(
  actionStates: Array<{
    description: string;
    actor?: "user" | "borg";
    state?: "considering" | "committed_to_do" | "scheduled" | "completed" | "not_done";
    audience_entity_id?: string | null;
    evidence_stream_entry_ids: string[];
    confidence?: number;
  }>,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_action_states",
        name: "EmitActionStates",
        input: {
          action_states: actionStates.map((actionState) => ({
            description: actionState.description,
            actor: actionState.actor ?? "user",
            state: actionState.state ?? "completed",
            audience_entity_id: actionState.audience_entity_id ?? null,
            evidence_stream_entry_ids: actionState.evidence_stream_entry_ids,
            confidence: actionState.confidence ?? 0.9,
          })),
        },
      },
    ],
  };
}

function createDynamicCommitmentJudgeResponse(reason: string) {
  return (options: LLMCompleteOptions) => {
    const content = String(options.messages[0]?.content ?? "");
    const commitmentId = content.match(/id=(cmt_[a-z0-9]+)/u)?.[1];

    if (commitmentId === undefined) {
      throw new Error("Commitment id missing from judge prompt");
    }

    return createCommitmentJudgeResponse([
      {
        commitment_id: commitmentId,
        reason,
      },
    ]);
  };
}

function createStepReflectionResponse(input: {
  stepOutcomes?: Array<{
    step_id: string;
    new_status: "doing" | "done" | "blocked" | "abandoned";
    evidence: string;
  }>;
  proposedSteps?: Array<{
    goal_id: string;
    description: string;
    kind: "think" | "ask_user" | "research" | "act" | "wait";
    due_at?: number | null;
    rationale: string;
  }>;
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
          step_outcomes: input.stepOutcomes ?? [],
          proposed_steps: input.proposedSteps ?? [],
        },
      },
    ],
  };
}

function makeEpisode(input: {
  id: EpisodeId;
  now: number;
  audienceEntityId: EntityId | null;
  shared: boolean;
  title: string;
}): Episode {
  return {
    id: input.id,
    title: input.title,
    narrative: `${input.title} narrative.`,
    participants: ["Borg"],
    location: null,
    start_time: input.now,
    end_time: input.now,
    source_stream_ids: [createStreamEntryId()],
    significance: 0.7,
    tags: ["identity"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    audience_entity_id: input.audienceEntityId,
    shared: input.shared,
    embedding: Float32Array.from([0, 0, 0, 1]),
    created_at: input.now,
    updated_at: input.now,
  };
}

function systemText(request: LLMCompleteOptions | undefined): string {
  const system = request?.system;

  if (typeof system === "string") {
    return system;
  }

  return system?.map((block) => block.text).join("\n") ?? "";
}

function firstFinalizerRequest(
  requests: readonly LLMCompleteOptions[],
): LLMCompleteOptions | undefined {
  return requests.find(
    (request) => request.budget === "cognition-system-1" || request.budget === "cognition-system-2",
  );
}

function finalizerRequests(requests: readonly LLMCompleteOptions[]): LLMCompleteOptions[] {
  return requests.filter(
    (request) => request.budget === "cognition-system-1" || request.budget === "cognition-system-2",
  );
}

function requestTextMessages(request: LLMCompleteOptions | undefined): string[] {
  return request?.messages.map((message) => message.content) ?? [];
}

async function removeTempDir(path: string): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      rmSync(path, { recursive: true, force: true, maxRetries: 3, retryDelay: 20 });
      return;
    } catch (error) {
      const code = (error as { code?: unknown }).code;

      if (attempt === 4 || (code !== "ENOTEMPTY" && code !== "EBUSY")) {
        throw error;
      }

      await new Promise((resolve) => setTimeout(resolve, 20 * (attempt + 1)));
    }
  }
}

describe("TurnOrchestrator evidence ledger", () => {
  const tempDirs: string[] = [];

  afterEach(async () => {
    while (tempDirs.length > 0) {
      await removeTempDir(tempDirs.pop() as string);
    }
  });

  function ledgerTurnResponses(finalizerText: string) {
    return [
      createCorrectivePreferenceResponse({
        classification: "none",
      }),
      createActionStateResponse([]),
      createGoalPromotionResponse([]),
      createEmitAnswerResponse(finalizerText),
      createClosureResponseAuditResponse(),
      createEmptyReflectionResponse(),
    ];
  }

  it("does not include the evidence ledger prompt block when the flag is disabled", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_180_000);
    const llm = new FakeLLMClient({
      responses: ledgerTurnResponses("I will use the current session."),
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      await borg.turn({
        userMessage: "Current session says Marta is the tutor.",
        stakes: "low",
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));

      expect(finalizerSystem).not.toContain("<borg_evidence_ledger>");
      expect(finalizerSystem).toContain("<borg_retrieved_evidence>");
    } finally {
      await borg.close();
    }
  });

  it("adds the evidence ledger as a finalizer-only block and omits no-op compaction traces", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const clock = new ManualClock(1_800_000_181_000);
    const llm = new FakeLLMClient({
      responses: ledgerTurnResponses("I will keep Marta grounded in this session."),
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
      configOverrides: {
        generation: {
          evidenceLedger: {
            enabled: true,
            currentSessionTranscriptTokenBudget: 50_000,
          },
        },
      },
    });

    try {
      await borg.turn({
        userMessage: "Current session says Marta is the tutor.",
        stakes: "low",
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));
      const traceEvents = readTraceEvents(tracePath);
      const compactTraceEvent = traceEvents.find(
        (event) => event.event === "evidence_ledger_compacted",
      );
      const traceEvent = traceEvents.find((event) => event.event === "evidence_ledger_built");

      expect(finalizerSystem).toContain("<borg_evidence_ledger>");
      expect(finalizerSystem).toContain(
        "Current-session transcript is authoritative for what happened in this conversation.",
      );
      expect(finalizerSystem).toContain("Current session says Marta is the tutor.");
      expect(finalizerSystem).not.toContain("<borg_retrieved_evidence>");
      expect(compactTraceEvent).toBeUndefined();
      expect(traceEvent).toMatchObject({
        event: "evidence_ledger_built",
        transcript_included: true,
        transcript_compacted: false,
        transcript_omitted_reason: null,
        original_transcript_token_estimate: expect.any(Number),
        compacted_transcript_token_estimate: expect.any(Number),
        compacted_entry_count: 0,
        raw_preserved_user_entry_count: 1,
      });
      expect(traceEvent?.entry_counts).toMatchObject({
        current_user_message: 1,
        current_session_transcript: 1,
      });
      expect(typeof traceEvent?.total_estimated_tokens).toBe("number");
    } finally {
      await borg.close();
    }
  });

  it("uses emission-tool finalizer output as the agent response and traces the decision", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const clock = new ManualClock(1_800_000_182_000);
    const finalText = "Marta is the tutor in this current session.";
    const hostCapabilities =
      "Output channels available now:\n- EmitAnswer: respond\n- HostReminder: schedule user-visible reminders";
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "none",
        }),
        createActionStateResponse([]),
        createGoalPromotionResponse([]),
        createFinalizerToolResponse({
          id: "toolu_emit_answer",
          name: "EmitAnswer",
          input: { text: finalText },
        }),
        createClosureResponseAuditResponse(),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
      env: {
        BORG_TRACE_PROMPTS: "1",
      },
      configOverrides: {
        generation: {
          evidenceLedger: {
            enabled: true,
            currentSessionTranscriptTokenBudget: 50_000,
          },
        },
        host_capabilities: hostCapabilities,
      },
    });

    try {
      const result = await borg.turn({
        userMessage: "Current session says Marta is the tutor.",
        stakes: "low",
      });

      const finalizerRequest = firstFinalizerRequest(llm.requests);
      const finalizerSystem = systemText(finalizerRequest);
      const traceEvents = readTraceEvents(tracePath);
      const finalizerEvent = traceEvents.find((event) => event.event === "finalizer_emitted");
      const ledgerEvent = traceEvents.find((event) => event.event === "evidence_ledger_built");
      const agentEntry = borg.stream.tail(20).find((entry) => entry.kind === "agent_msg");

      expect(result.response).toBe(finalText);
      expect(result.emitted).toBe(true);
      expect(agentEntry?.content).toBe(finalText);
      expect(finalizerRequest?.tool_choice).toEqual({ type: "any" });
      expect(finalizerRequest?.output_config).toBeUndefined();
      expect(finalizerRequest?.tools?.map((tool) => tool.name)).toEqual([
        "EmitAnswer",
        "EmitObserve",
        "EmitNoOutput",
        "EmitSelfReport",
      ]);
      expect(finalizerSystem).toContain("<borg_evidence_ledger>");
      expect(finalizerSystem).toContain("id=current_user_message:");
      expect(finalizerSystem).toContain("<borg_host_capabilities>");
      expect(finalizerSystem).toContain(hostCapabilities);
      expect(finalizerSystem).not.toContain("Real-time polling of external state");
      expect(llm.requests.some((request) => request.budget === "closure-response-auditor")).toBe(
        true,
      );
      expect(ledgerEvent).toMatchObject({
        event: "evidence_ledger_built",
      });
      expect(finalizerEvent).toMatchObject({
        event: "finalizer_emitted",
        decision: "answer",
        text_length: finalText.length,
        mode: "emission_tools",
      });
    } finally {
      await borg.close();
    }
  });

  it("keeps group-chat first-person actions attributed to the original sender on later turns", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_182_250);
    const audience = "Spain Trip Planning Channel";
    let groupEntityId: EntityId;
    const actionStateForCurrentUser = Object.assign(
      (options: LLMCompleteOptions) => {
        const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
          current_user_stream_entry_id?: string;
        };
        const currentUserStreamEntryId = payload.current_user_stream_entry_id;

        if (typeof currentUserStreamEntryId !== "string") {
          throw new Error("action-state request did not include current_user_stream_entry_id");
        }

        return createActionStateResponse([
          {
            description: "book the Alhambra tickets this weekend",
            actor: "user",
            state: "committed_to_do",
            audience_entity_id: groupEntityId,
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]);
      },
      { budget: "action-state-extractor" },
    );
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({ classification: "none" }),
        actionStateForCurrentUser,
        createGoalPromotionResponse([]),
        createEmitAnswerResponse("Alice can own the Alhambra booking."),
        createClosureResponseAuditResponse(),
        createEmptyReflectionResponse(),
        createCorrectivePreferenceResponse({ classification: "none" }),
        createActionStateResponse([]),
        createGoalPromotionResponse([]),
        createEmitAnswerResponse("I will keep the commitments separated by speaker."),
        createClosureResponseAuditResponse(),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      configOverrides: {
        generation: {
          evidenceLedger: {
            enabled: true,
            currentSessionTranscriptTokenBudget: 50_000,
          },
        },
      },
    });

    try {
      groupEntityId = borg.entities.resolve(audience, {
        kind: "group",
      });
      const alice = borg.entities.resolve("Alice", {
        kind: "person",
      });
      const ben = borg.entities.resolve("Ben", {
        kind: "person",
      });

      await borg.turn({
        userMessage: "I'll book the Alhambra tickets this weekend",
        audience,
        senderEntityId: alice,
        stakes: "low",
      });

      const aliceAction = borg.actions
        .list({ limit: 10 })
        .find((record) => record.description === "book the Alhambra tickets this weekend");

      expect(aliceAction).toMatchObject({
        actor: alice,
        audience_entity_id: groupEntityId,
        state: "committed_to_do",
      });
      expect(aliceAction?.actor).not.toBe("user");
      expect(aliceAction?.actor).not.toBe(groupEntityId);
      expect(aliceAction?.actor).not.toBe(ben);

      await borg.turn({
        userMessage: "what did I commit to?",
        audience,
        senderEntityId: ben,
        stakes: "low",
      });

      const benFinalizerSystem = systemText(finalizerRequests(llm.requests).at(1));

      expect(benFinalizerSystem).toContain("<borg_evidence_ledger>");
      expect(benFinalizerSystem).toContain("book the Alhambra tickets this weekend");
      expect(benFinalizerSystem).toContain("actor: Alice");
      expect(benFinalizerSystem).not.toContain("actor: Ben");
      expect(
        borg.actions
          .list({ actor: ben, limit: 10 })
          .some((record) => record.description === "book the Alhambra tickets this weekend"),
      ).toBe(false);
    } finally {
      await borg.close();
    }
  });

  it("persists EmitSelfReport responses as assistant self-report stream entries", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_182_500);
    const finalText = "The gap feels like a discontinuity with a remembered edge.";
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "none",
        }),
        createActionStateResponse([]),
        createGoalPromotionResponse([]),
        createFinalizerToolResponse({
          id: "toolu_emit_self_report",
          name: "EmitSelfReport",
          input: {
            kind: "self_report",
            text: finalText,
            persistence_class: "assistant_self_report",
          },
        }),
        createClosureResponseAuditResponse(),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      configOverrides: {},
    });

    try {
      const result = await borg.turn({
        userMessage: "What does the gap feel like?",
        stakes: "low",
      });
      const agentEntry = borg.stream.tail(20).find((entry) => entry.kind === "agent_msg");

      expect(result.emission).toEqual({
        kind: "message",
        content: finalText,
        agentMessageId: agentEntry?.id,
        persistence_class: "assistant_self_report",
      });
      expect(agentEntry).toMatchObject({
        kind: "agent_msg",
        content: finalText,
        persistence_class: "assistant_self_report",
      });
    } finally {
      await borg.close();
    }
  });

  it("suppresses EmitNoOutput with a finalizer_no_output marker", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const clock = new ManualClock(1_800_000_183_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "none",
        }),
        createActionStateResponse([]),
        createGoalPromotionResponse([]),
        createFinalizerToolResponse({
          id: "toolu_emit_no_output",
          name: "EmitNoOutput",
          input: { reason: "natural_close" },
        }),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
      configOverrides: {
        generation: {},
      },
    });

    try {
      const result = await borg.turn({
        userMessage: "Thanks.",
        stakes: "low",
      });

      const traceEvents = readTraceEvents(tracePath);
      const finalizerEvent = traceEvents.find((event) => event.event === "finalizer_emitted");
      const suppressedEntry = borg.stream
        .tail(20)
        .find((entry) => entry.kind === "agent_suppressed");

      expect(result.response).toBe("");
      expect(result.emitted).toBe(false);
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "finalizer_no_output",
      });
      expect(suppressedEntry?.content).toMatchObject({
        reason: "finalizer_no_output",
      });
      expect(finalizerEvent).toMatchObject({
        event: "finalizer_emitted",
        decision: "no_output",
        reason: "natural_close",
      });
    } finally {
      await borg.close();
    }
  });

  it("persists EmitObserve distinctly and still runs reflection, mood, and social updates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const clock = new ManualClock(1_800_000_183_500);
    const observeReason = "Alice and Bob are coordinating directly.";
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "none",
        }),
        createActionStateResponse([]),
        createGoalPromotionResponse([]),
        createEmitObserveResponse(observeReason),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
      configOverrides: {
        affective: {
          useLlmFallback: false,
        },
      },
    });

    try {
      borg.entities.resolve("Planning Room", {
        kind: "group",
      });
      const alice = borg.entities.resolve("Alice", {
        kind: "person",
      });
      const internal = borg as unknown as {
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: () => Promise<unknown>;
              moodRepository: {
                update: (sessionId: string, update: unknown) => unknown;
              };
            };
          };
        };
      };
      const moodUpdate = vi.spyOn(internal.deps.turnOrchestrator.options.moodRepository, "update");
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async () => ({
        valence: 0.25,
        arousal: 0.35,
        dominant_emotion: "curiosity",
      });

      const result = await borg.turn({
        userMessage: "Bob, Tuesday works from my side.",
        audience: "Planning Room",
        senderEntityId: alice,
        stakes: "low",
      });
      const observedEntry = borg.stream.tail(20).find((entry) => entry.kind === "agent_observed");
      const workingMemory = borg.workmem.load();
      const reflectionPayload = parseReflectionPayload(findReflectionRequest(llm));

      expect(result.response).toBe("");
      expect(result.emitted).toBe(false);
      expect(result.emission).toEqual({
        kind: "observed",
        reason: observeReason,
        markerEntryId: observedEntry?.id,
      });
      expect(observedEntry?.content).toMatchObject({
        reason: observeReason,
      });
      expect(borg.stream.tail(20).some((entry) => entry.kind === "agent_suppressed")).toBe(false);
      expect(workingMemory.discourse_state.stop_until_substantive_content).toBeNull();
      expect(workingMemory.discourse_state.recent_suppressions).toEqual([]);
      expect(workingMemory.discourse_state.closure_pressure_history).toEqual([]);
      expect(moodUpdate).toHaveBeenCalledOnce();
      expect(borg.social.getProfile("Alice")?.interaction_count).toBe(1);
      expect(reflectionPayload).toMatchObject({
        agent_response: "",
      });
      expect(reflectionPayload.current_turn_stream_entry_ids).toContain(observedEntry?.id);
    } finally {
      await borg.close();
    }
  });
});

describe("TurnOrchestrator participant social profiles", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("rejects group-audience user turns without a sender before user persistence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_182_250);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const groupId = borg.entities.resolve("Planning Room", {
        kind: "group",
      });
      const internal = borg as unknown as {
        deps: {
          entityRepository: {
            get: (id: EntityId) => unknown;
          };
        };
      };
      const initialGroupEntity = internal.deps.entityRepository.get(groupId);
      const initialWorkingMemory = borg.workmem.load();

      await expect(
        borg.turn({
          userMessage: "I can handle flights.",
          audience: "Planning Room",
          stakes: "low",
        }),
      ).rejects.toMatchObject({
        code: "GROUP_SENDER_REQUIRED",
      });

      expect(borg.stream.tail(20).some((entry) => entry.kind === "user_msg")).toBe(false);
      expect(borg.workmem.load()).toEqual(initialWorkingMemory);
      expect(internal.deps.entityRepository.get(groupId)).toEqual(initialGroupEntity);
      expect(borg.actions.list({ limit: 10 })).toEqual([]);
      expect(borg.social.getProfile("Planning Room")).toBeNull();
      expect(llm.requests).toEqual([]);
    } finally {
      await borg.close();
    }
  });

  it("allows non-group user turns without a sender", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_182_250);
    const llm = new FakeLLMClient({
      responses: [
        createEmitAnswerResponse("Person turn ok."),
        createEmptyReflectionResponse(),
        createEmitAnswerResponse("Self turn ok."),
        createEmptyReflectionResponse(),
        createEmitAnswerResponse("Abstract turn ok."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      borg.entities.resolve("Alice", {
        kind: "person",
      });
      borg.entities.resolve("Project Atlas", {
        kind: "abstract",
      });

      await expect(
        borg.turn({
          userMessage: "Person-scoped note.",
          audience: "Alice",
          stakes: "low",
        }),
      ).resolves.toMatchObject({ response: "Person turn ok." });
      await expect(
        borg.turn({
          userMessage: "Self-scoped note.",
          audience: "self",
          stakes: "low",
        }),
      ).resolves.toMatchObject({ response: "Self turn ok." });
      await expect(
        borg.turn({
          userMessage: "Project-scoped note.",
          audience: "Project Atlas",
          stakes: "low",
        }),
      ).resolves.toMatchObject({ response: "Abstract turn ok." });
    } finally {
      await borg.close();
    }
  });

  it("skips active participant scans for non-group audiences", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_182_250);
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("Person turn ok."), createEmptyReflectionResponse()],
    });
    const scanSpy = vi.spyOn(StreamReader.prototype, "scanReverse");
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      borg.entities.resolve("Alice", {
        kind: "person",
      });

      await expect(
        borg.turn({
          userMessage: "Person-scoped note.",
          audience: "Alice",
          stakes: "low",
        }),
      ).resolves.toMatchObject({ response: "Person turn ok." });

      const participantScanCalls = scanSpy.mock.calls.filter(([options]) => {
        return (
          options?.maxEntries === 500 &&
          options.maxBytes === 512 * 1024 &&
          options.filter !== undefined
        );
      });

      expect(participantScanCalls).toHaveLength(0);
    } finally {
      scanSpy.mockRestore();
      await borg.close();
    }
  });

  it("allows autonomous group turns without a sender", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_182_250);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({ classification: "none" }),
        createEmitAnswerResponse("Autonomous group turn ok."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      borg.entities.resolve("Planning Room", {
        kind: "group",
      });

      await expect(
        borg.turn({
          userMessage: "Review the planning room state.",
          audience: "Planning Room",
          origin: "autonomous",
          stakes: "low",
        }),
      ).resolves.toMatchObject({ response: "Autonomous group turn ok." });

      expect(borg.stream.tail(20).some((entry) => entry.kind === "user_msg")).toBe(false);
    } finally {
      await borg.close();
    }
  });

  it("renders all active group participants in the social profile prompt section", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("Flights next."), createEmptyReflectionResponse()],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      borg.entities.resolve("Planning Room", {
        kind: "group",
      });
      const alice = borg.entities.resolve("Alice", {
        kind: "person",
      });
      const bob = borg.entities.resolve("Bob", {
        kind: "person",
      });

      borg.social.recordInteraction("Alice", {
        provenance: {
          kind: "system",
        },
        now: clock.now(),
      });
      borg.social.recordInteraction("Bob", {
        provenance: {
          kind: "system",
        },
        now: clock.now(),
      });
      await borg.stream.append({
        kind: "user_msg",
        content: "I can handle hotels.",
        audience: "Planning Room",
        sender_entity_id: bob,
      });

      await borg.turn({
        userMessage: "I can handle flights.",
        audience: "Planning Room",
        senderEntityId: alice,
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));

      expect(finalizerSystem).toContain("Participants:");
      expect(finalizerSystem).toContain("Alice (speaker): trust=0.50");
      expect(finalizerSystem).toContain("Bob (participant): trust=0.50");
      expect(finalizerSystem).not.toContain("Talking to:");
    } finally {
      await borg.close();
    }
  });

  it("traces participant scan caps for noisy group streams", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("Flights next."), createEmptyReflectionResponse()],
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    try {
      borg.entities.resolve("Planning Room", {
        kind: "group",
      });
      const alice = borg.entities.resolve("Alice", {
        kind: "person",
      });

      for (let index = 0; index < 600; index += 1) {
        clock.advance(1);
        await writer.append({
          kind: "internal_event",
          content: { event: "maintenance_audit", index },
        });
      }

      writer.close();

      await borg.turn({
        userMessage: "I can handle flights.",
        audience: "Planning Room",
        senderEntityId: alice,
      });

      const traceEvents = readTraceEvents(tracePath);

      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "participant_scan_cap_reached",
          turnId: expect.any(String),
          cap: "entries",
          scanned_entries: 500,
          found_unique_participants: 1,
          requested_limit: 8,
        }),
      );
      expect(
        traceEvents.find((event) => event.event === "participant_scan_cap_reached")?.scanned_bytes,
      ).toEqual(expect.any(Number));
    } finally {
      writer.close();
      await borg.close();
    }
  });

  it("keeps group audience visible when no prior peer turns are recent", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("Flights next."), createEmptyReflectionResponse()],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      borg.entities.resolve("Planning Room", {
        kind: "group",
      });
      const alice = borg.entities.resolve("Alice", {
        kind: "person",
      });

      await borg.turn({
        userMessage: "I can handle flights.",
        audience: "Planning Room",
        senderEntityId: alice,
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));

      expect(finalizerSystem).toContain("Participants:");
      expect(finalizerSystem).toContain("Alice (speaker):");
      expect(finalizerSystem).toContain("Planning Room (audience):");
      expect(finalizerSystem).not.toContain("Talking to:");
    } finally {
      await borg.close();
    }
  });

  it("keeps legacy global constrained slots when no participant can be resolved", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_500);
    const llm = new FakeLLMClient({
      responses: [
        createEmitAnswerResponse("I will keep it neutral."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository" | "relationalSlotRepository">;
    };

    try {
      const tom = internal.deps.entityRepository.resolve("Tom", {
        kind: "person",
      });
      internal.deps.relationalSlotRepository.applyAssertion({
        subject_entity_id: tom,
        slot_key: "partner.name",
        asserted_value: "Sarah",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      internal.deps.relationalSlotRepository.applyAssertion({
        subject_entity_id: tom,
        slot_key: "partner.name",
        asserted_value: "Maya",
        source_stream_entry_ids: [createStreamEntryId()],
      });

      await borg.turn({
        userMessage: "Help me decide what to say next.",
        stakes: "low",
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));

      expect(finalizerSystem).toContain("Relational slot constraints");
      expect(finalizerSystem).toContain("partner.name: CONTESTED");
      expect(finalizerSystem).not.toContain("Tom: partner.name");
    } finally {
      await borg.close();
    }
  });
});

describe("TurnOrchestrator self snapshot audience visibility", () => {
  const tempDirs: string[] = [];

  afterEach(async () => {
    while (tempDirs.length > 0) {
      await removeTempDir(tempDirs.pop() as string);
    }
  });

  it("uses post-construction llmFactory overrides on subsequent turns", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000_000);
    const initialLlm = new FakeLLMClient();
    const replacementAnswer = "Replacement answer.";
    const replacementLlm = new FakeLLMClient({
      responses: [createEmitAnswerResponse(replacementAnswer), createEmptyReflectionResponse()],
    });
    const borg = await openTestBorg(tempDir, initialLlm, clock);

    try {
      const internal = borg as unknown as {
        deps: {
          turnOrchestrator: {
            options: {
              llmFactory: () => FakeLLMClient;
            };
          };
        };
      };
      internal.deps.turnOrchestrator.options.llmFactory = () => replacementLlm;

      const result = await borg.turn({
        userMessage: "Use the replacement client.",
        stakes: "low",
      });

      expect(result.response).toBe(replacementAnswer);
      expect(firstFinalizerRequest(replacementLlm.requests)?.budget).toBe("cognition-system-1");
      expect(initialLlm.requests).toHaveLength(0);
    } finally {
      await borg.close();
    }
  });

  it("does not surface self records backed only by another audience's private episodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000_000);
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("Public answer."), createEmptyReflectionResponse()],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const internal = borg as unknown as {
        deps: Pick<BorgDependencies, "entityRepository"> & {
          episodicRepository: EpisodicRepository;
        };
      };
      const aliceEntityId = internal.deps.entityRepository.resolve("Alice");
      internal.deps.entityRepository.resolve("Bob");
      const alicePrivateEpisodeId = createEpisodeId();
      const publicEpisodeId = createEpisodeId();
      const now = clock.now();

      await internal.deps.episodicRepository.insert(
        makeEpisode({
          id: alicePrivateEpisodeId,
          now,
          audienceEntityId: aliceEntityId,
          shared: false,
          title: "Alice private identity evidence",
        }),
      );
      await internal.deps.episodicRepository.insert(
        makeEpisode({
          id: publicEpisodeId,
          now,
          audienceEntityId: null,
          shared: true,
          title: "Public identity evidence",
        }),
      );

      borg.self.values.add({
        label: "alice-private-value",
        description: "Alice-only value description.",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.values.add({
        label: "public-value",
        description: "Public value description.",
        priority: 9,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.values.add({
        label: "mixed-visible-value",
        description: "Mixed public and private evidence stays visible.",
        priority: 8,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId, publicEpisodeId],
        },
      });
      borg.self.values.add({
        label: "manual-unscoped-value",
        description: "Manual self state has no audience-scoped evidence.",
        priority: 7,
        provenance: {
          kind: "manual",
        },
      });
      borg.self.goals.add({
        description: "alice-private-goal",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.goals.add({
        description: "public-goal",
        priority: 9,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.traits.add({
        label: "alice-private-trait",
        delta: 0.4,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "public-trait",
        delta: 0.4,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "mixed-visible-trait",
        delta: 0.2,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "mixed-visible-trait",
        delta: 0.2,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
        timestamp: now + 1,
      });
      borg.self.growthMarkers.add({
        ts: now,
        category: "understanding",
        what_changed: "alice-private-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.growthMarkers.add({
        ts: now + 1,
        category: "understanding",
        what_changed: "public-growth",
        evidence_episode_ids: [publicEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.autobiographical.upsertPeriod({
        label: "alice-private-period",
        start_ts: now,
        narrative: "Alice-private period narrative.",
        key_episode_ids: [alicePrivateEpisodeId],
        themes: ["privacy"],
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });

      await borg.turn({
        userMessage: "Hello Bob.",
        audience: "Bob",
        stakes: "low",
      });

      const allRequestText = llm.requests.map((request) => JSON.stringify(request)).join("\n");
      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));

      expect(finalizerSystem).toContain("public-value");
      expect(finalizerSystem).toContain("mixed-visible-value");
      expect(finalizerSystem).toContain("manual-unscoped-value");
      expect(finalizerSystem).toContain("public-goal");
      expect(finalizerSystem).toContain("public-trait");
      expect(finalizerSystem).toContain("mixed-visible-trait");
      expect(finalizerSystem).toContain("public-growth");
      expect(allRequestText).not.toContain("alice-private-value");
      expect(allRequestText).not.toContain("Alice-only value description");
      expect(allRequestText).not.toContain("alice-private-goal");
      expect(allRequestText).not.toContain("alice-private-trait");
      expect(allRequestText).not.toContain("alice-private-growth");
      expect(allRequestText).not.toContain("alice-private-period");
      expect(allRequestText).not.toContain("Alice-private period narrative");
    } finally {
      await borg.close();
    }
  });

  it("filters gathered self snapshot records in a single visibility pass", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(2_000_000);
    const borg = await openTestBorg(tempDir, new FakeLLMClient(), clock);

    try {
      const internal = borg as unknown as {
        deps: Pick<BorgDependencies, "entityRepository"> & {
          episodicRepository: EpisodicRepository;
          turnOrchestrator: {
            buildSelfSnapshot(audienceEntityId: EntityId | null): Promise<SelfSnapshot>;
          };
        };
      };
      const bobEntityId = internal.deps.entityRepository.resolve("Bob");
      const publicEpisodeId = createEpisodeId();
      const singlePassGoalDescription = "single-pass-goal";
      const singlePassValueLabel = "single-pass-value";
      const now = clock.now();

      await internal.deps.episodicRepository.insert(
        makeEpisode({
          id: publicEpisodeId,
          now,
          audienceEntityId: null,
          shared: true,
          title: "Public single-pass snapshot evidence",
        }),
      );
      borg.self.values.add({
        label: singlePassValueLabel,
        description: "Visible value.",
        priority: 7,
        provenance: { kind: "episodes", episode_ids: [publicEpisodeId] },
      });
      borg.self.goals.add({
        description: singlePassGoalDescription,
        priority: 7,
        provenance: { kind: "episodes", episode_ids: [publicEpisodeId] },
      });
      const getManySpy = vi.spyOn(internal.deps.episodicRepository, "getMany");

      const snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);

      expect(snapshot.values.map((value) => value.label)).toContain(singlePassValueLabel);
      expect(snapshot.goals.map((goal) => goal.description)).toContain(singlePassGoalDescription);
      expect(getManySpy).toHaveBeenCalledTimes(1);
      expect(getManySpy.mock.calls[0]?.[0]).toEqual(expect.arrayContaining([publicEpisodeId]));
    } finally {
      await borg.close();
    }
  });

  it("keeps mixed, unscoped, matching-audience, and empty-evidence self records visible", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(2_000_000);
    const borg = await openTestBorg(tempDir, new FakeLLMClient(), clock);

    try {
      const internal = borg as unknown as {
        deps: Pick<
          BorgDependencies,
          "autobiographicalRepository" | "entityRepository" | "sqlite"
        > & {
          episodicRepository: EpisodicRepository;
          turnOrchestrator: {
            buildSelfSnapshot(audienceEntityId: EntityId | null): Promise<SelfSnapshot>;
          };
        };
      };
      const aliceEntityId = internal.deps.entityRepository.resolve("Alice");
      const bobEntityId = internal.deps.entityRepository.resolve("Bob");
      const alicePrivateEpisodeId = createEpisodeId();
      const bobPrivateEpisodeId = createEpisodeId();
      const publicEpisodeId = createEpisodeId();
      const now = clock.now();

      for (const episode of [
        makeEpisode({
          id: alicePrivateEpisodeId,
          now,
          audienceEntityId: aliceEntityId,
          shared: false,
          title: "Alice private snapshot evidence",
        }),
        makeEpisode({
          id: bobPrivateEpisodeId,
          now,
          audienceEntityId: bobEntityId,
          shared: false,
          title: "Bob private snapshot evidence",
        }),
        makeEpisode({
          id: publicEpisodeId,
          now,
          audienceEntityId: null,
          shared: true,
          title: "Public snapshot evidence",
        }),
      ]) {
        await internal.deps.episodicRepository.insert(episode);
      }

      borg.self.goals.add({
        description: "mixed-visible-goal",
        priority: 8,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
      });
      borg.self.goals.add({
        description: "manual-visible-goal",
        priority: 7,
        provenance: { kind: "manual" },
      });
      borg.self.goals.add({
        description: "system-visible-goal",
        priority: 6,
        provenance: { kind: "system" },
      });
      borg.self.goals.add({
        description: "bob-private-visible-goal",
        priority: 5,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      borg.self.traits.add({
        label: "manual-visible-trait",
        delta: 0.3,
        provenance: { kind: "manual" },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "bob-private-visible-trait",
        delta: 0.3,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
        timestamp: now,
      });
      borg.self.values.add({
        label: "bob-private-visible-value",
        description: "Bob-private evidence should be visible to Bob.",
        priority: 7,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      const emptyValue = borg.self.values.add({
        label: "empty-evidence-visible-value",
        description: "Empty evidence should not scope the record.",
        priority: 6,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
      });
      const emptyTrait = borg.self.traits.add({
        label: "empty-evidence-visible-trait",
        delta: 0.3,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
        timestamp: now,
      });
      const emptyTraitId =
        emptyTrait.status === "applied" ? emptyTrait.record.id : emptyTrait.current.id;

      internal.deps.sqlite
        .prepare('UPDATE "values" SET evidence_episode_ids = ? WHERE id = ?')
        .run("[]", emptyValue.id);
      internal.deps.sqlite
        .prepare("UPDATE traits SET evidence_episode_ids = ? WHERE id = ?")
        .run("[]", emptyTraitId);

      borg.self.growthMarkers.add({
        ts: now,
        category: "understanding",
        what_changed: "mixed-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId, publicEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
      });
      borg.self.growthMarkers.add({
        ts: now + 1,
        category: "understanding",
        what_changed: "manual-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "manual" },
      });
      borg.self.growthMarkers.add({
        ts: now + 2,
        category: "understanding",
        what_changed: "system-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "system" },
      });
      let snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
      expect(snapshot.recentGrowthMarkers?.map((marker) => marker.what_changed)).toEqual([
        "system-visible-growth",
        "manual-visible-growth",
        "mixed-visible-growth",
      ]);

      borg.self.growthMarkers.add({
        ts: now + 3,
        category: "understanding",
        what_changed: "bob-private-visible-growth",
        evidence_episode_ids: [bobPrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      const emptyGrowth = borg.self.growthMarkers.add({
        ts: now + 4,
        category: "understanding",
        what_changed: "empty-evidence-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
      });
      internal.deps.sqlite
        .prepare("UPDATE growth_markers SET evidence_episode_ids = ? WHERE id = ?")
        .run("[]", emptyGrowth.id);

      for (const [label, provenance] of [
        ["public-visible-period", { kind: "episodes" as const, episode_ids: [publicEpisodeId] }],
        [
          "mixed-visible-period",
          { kind: "episodes" as const, episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
        ],
        ["manual-visible-period", { kind: "manual" as const }],
        ["system-visible-period", { kind: "system" as const }],
        [
          "bob-private-visible-period",
          { kind: "episodes" as const, episode_ids: [bobPrivateEpisodeId] },
        ],
        [
          "empty-evidence-visible-period",
          { kind: "episodes" as const, episode_ids: [alicePrivateEpisodeId] },
        ],
      ] as const) {
        internal.deps.autobiographicalRepository.upsertPeriod({
          label,
          start_ts: now,
          narrative: `${label} narrative.`,
          key_episode_ids: label.startsWith("empty")
            ? []
            : provenance.kind === "episodes"
              ? provenance.episode_ids
              : [alicePrivateEpisodeId],
          themes: ["visibility"],
          provenance: provenance as never,
        });
        snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
        expect(snapshot.currentPeriod?.label).toBe(label);
      }

      snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
      expect(snapshot.values.map((value) => value.label)).toEqual(
        expect.arrayContaining(["bob-private-visible-value", "empty-evidence-visible-value"]),
      );
      expect(snapshot.goals.map((goal) => goal.description)).toEqual(
        expect.arrayContaining([
          "mixed-visible-goal",
          "manual-visible-goal",
          "system-visible-goal",
          "bob-private-visible-goal",
        ]),
      );
      expect(snapshot.traits.map((trait) => trait.label)).toEqual(
        expect.arrayContaining([
          "manual-visible-trait",
          "bob-private-visible-trait",
          "empty-evidence-visible-trait",
        ]),
      );
      expect(snapshot.recentGrowthMarkers?.map((marker) => marker.what_changed)).toEqual(
        expect.arrayContaining(["bob-private-visible-growth", "empty-evidence-visible-growth"]),
      );
    } finally {
      await borg.close();
    }
  });

  it("selects an executive focus and renders it without dropping other active goals", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(3_000_000);
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("Apollo answer."), createEmptyReflectionResponse()],
    });
    const embeddingClient = new CountingEmbeddingClient();
    const borg = await openTestBorg(tempDir, llm, clock, embeddingClient);

    try {
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));
      const blockStart = finalizerSystem.indexOf("<borg_executive_focus>");
      const blockEnd = finalizerSystem.indexOf("</borg_executive_focus>");
      const executiveBlock = finalizerSystem.slice(blockStart, blockEnd);

      expect(blockStart).toBeGreaterThanOrEqual(0);
      expect(blockEnd).toBeGreaterThan(blockStart);
      expect(executiveBlock).toContain("Current driving goal: Apollo launch plan");
      expect(executiveBlock).toContain(
        "Use this as a bias, not an override of the user's request or commitments.",
      );
      expect(executiveBlock).not.toContain("Next step:");
      expect(executiveBlock).not.toContain("Background maintenance");
      expect(finalizerSystem).toContain("goals Background maintenance");
      expect(finalizerSystem).toContain("Apollo launch plan");
      expect(
        embeddingClient.embedBatchTexts.filter((texts) =>
          texts.some((text) => text.includes("Apollo launch plan")),
        ),
      ).toHaveLength(1);
    } finally {
      await borg.close();
    }
  });

  it("renders the selected goal's top open executive step", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("Apollo step answer."), createEmptyReflectionResponse()],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const internal = borg as unknown as {
        deps: {
          executiveStepsRepository: ExecutiveStepsRepository;
        };
      };
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      const selectedGoal = borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });
      const dueAt = clock.now() + 86_400_000;

      internal.deps.executiveStepsRepository.add({
        goalId: selectedGoal.id,
        description: "Inspect the launch readiness notes",
        kind: "research",
        dueAt,
        provenance: {
          kind: "system",
        },
      });

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));
      const blockStart = finalizerSystem.indexOf("<borg_executive_focus>");
      const blockEnd = finalizerSystem.indexOf("</borg_executive_focus>");
      const executiveBlock = finalizerSystem.slice(blockStart, blockEnd);

      expect(executiveBlock).toContain("Current driving goal: Apollo launch plan");
      expect(executiveBlock).toContain(
        `Next step: Inspect the launch readiness notes (kind: research, due: ${new Date(
          dueAt,
        ).toISOString()})`,
      );
    } finally {
      await borg.close();
    }
  });

  it("applies executive step outcomes from full-turn reflection", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient();
    const tracePath = join(tempDir, "turn-trace.jsonl");
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });

    try {
      const internal = borg as unknown as {
        deps: {
          executiveStepsRepository: ExecutiveStepsRepository;
        };
      };
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      const selectedGoal = borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });
      const step = internal.deps.executiveStepsRepository.add({
        goalId: selectedGoal.id,
        description: "Inspect the launch readiness notes",
        kind: "research",
        provenance: {
          kind: "system",
        },
      });
      llm.pushResponse(createEmitAnswerResponse("Apollo step started."));
      llm.pushResponse(
        createStepReflectionResponse({
          stepOutcomes: [
            {
              step_id: step.id,
              new_status: "doing",
              evidence: "The assistant started inspecting the launch readiness notes.",
            },
          ],
        }),
      );

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      expect(internal.deps.executiveStepsRepository.get(step.id)?.status).toBe("doing");
    } finally {
      await borg.close();
    }
  });

  it("creates proposed executive steps from full-turn reflection when selected goal has none open", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient();
    const tracePath = join(tempDir, "turn-trace.jsonl");
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });

    try {
      const internal = borg as unknown as {
        deps: {
          executiveStepsRepository: ExecutiveStepsRepository;
        };
      };
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      const selectedGoal = borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });
      llm.pushResponse(createEmitAnswerResponse("Apollo next step identified."));
      llm.pushResponse(
        createStepReflectionResponse({
          proposedSteps: [
            {
              goal_id: selectedGoal.id,
              description: "Draft the Apollo readiness question",
              kind: "ask_user",
              due_at: null,
              rationale: "The selected goal has no open executive step.",
            },
          ],
        }),
      );

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      expect(
        internal.deps.executiveStepsRepository.listOpen(selectedGoal.id).map((step) => ({
          description: step.description,
          kind: step.kind,
        })),
      ).toEqual([
        {
          description: "Draft the Apollo readiness question",
          kind: "ask_user",
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("passes at most twenty active open questions to reflection", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      for (let index = 0; index < 100; index += 1) {
        borg.self.openQuestions.add({
          question: `Which Atlas follow-up ${index} remains open?`,
          urgency: 0.5,
          provenance: {
            kind: "manual",
          },
          source: "user",
        });
      }

      llm.pushResponse(createEmitAnswerResponse("I will keep this concise."));
      llm.pushResponse(createEmptyReflectionResponse());

      await borg.turn({
        userMessage: "Please keep tracking Atlas follow-ups.",
        stakes: "low",
      });

      const payload = parseReflectionPayload(findReflectionRequest(llm));
      expect(payload.active_open_questions).toHaveLength(20);
    } finally {
      await borg.close();
    }
  });

  it("omits resolved open questions from the next turn's active reflection list", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient();
    const tracePath = join(tempDir, "turn-trace.jsonl");
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });

    try {
      const question = borg.self.openQuestions.add({
        question: "What does the current turn answer?",
        urgency: 0.8,
        provenance: {
          kind: "manual",
        },
        source: "reflection",
      });

      llm.pushResponse(createEmitAnswerResponse("The current turn answers it directly."));
      llm.pushResponse(
        createStopCommitmentResponse({
          classification: "none",
        }),
      );
      llm.pushResponse((request: LLMCompleteOptions) => {
        const payload = parseReflectionPayload(request);
        const activeQuestions = payload.active_open_questions as Array<{ id: string }>;
        const streamEntryIds = payload.current_turn_stream_entry_ids as string[];

        expect(activeQuestions.map((item) => item.id)).toContain(question.id);
        expect(streamEntryIds.length).toBeGreaterThan(0);

        return {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use" as const,
          tool_calls: [
            {
              id: "toolu_reflection",
              name: "EmitTurnReflection",
              input: {
                advanced_goals: [],
                procedural_outcomes: [],
                trait_demonstrations: [],
                intent_updates: [],
                resolved_open_questions: [
                  {
                    question_id: question.id,
                    resolution_note: "The current turn answered it directly.",
                    evidence_episode_ids: [],
                    evidence_stream_entry_ids: [streamEntryIds[0]!],
                  },
                ],
              },
            },
          ],
        };
      });

      await borg.turn({
        userMessage: "This turn answers the open question.",
        stakes: "low",
      });

      expect(
        readTraceEvents(tracePath).filter(
          (event) => event.event === "open_question_resolution_degraded",
        ),
      ).toEqual([]);
      expect(borg.self.openQuestions.list({ status: "open" }).map((item) => item.id)).not.toContain(
        question.id,
      );

      llm.pushResponse(createEmitAnswerResponse("No open question remains in scope."));
      llm.pushResponse(
        createStopCommitmentResponse({
          classification: "none",
        }),
      );
      llm.pushResponse((request: LLMCompleteOptions) => {
        const payload = parseReflectionPayload(request);
        const activeQuestions = payload.active_open_questions as Array<{ id: string }>;

        expect(activeQuestions.map((item) => item.id)).not.toContain(question.id);

        return createEmptyReflectionResponse();
      });

      await borg.turn({
        userMessage: "Check the next active list.",
        stakes: "low",
      });
    } finally {
      await borg.close();
    }
  });

  it("records user-visible stop commitments as durable discourse state", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_000_000);
    const llm = new FakeLLMClient({
      responses: [
        createEmitAnswerResponse("I will stop responding until you bring substantive content."),
        createStopCommitmentResponse({
          classification: "stop_until_substantive_content",
          reason: "The assistant committed to stop until substantive content arrives.",
        }),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "Stop responding if I keep sending filler.",
      });
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(result.emitted).toBe(true);
      expect(result.response).toContain("I will stop responding");
      expect(activeStop).toMatchObject({
        provenance: "self_commitment_extractor",
        source_stream_entry_id: result.agentMessageId,
        reason: "The assistant committed to stop until substantive content arrives.",
        since_turn: 1,
      });
    } finally {
      await borg.close();
    }
  });

  it("turns S2 no-output recommendations into suppressed turns", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_100_000);
    const llm = new FakeLLMClient({
      responses: [
        createNoOutputTurnPlanResponse(),
        createEmitNoOutputResponse("The planner recommended no assistant message."),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "No.",
        stakes: "high",
      });
      const entries = borg.stream.tail(10);
      const thoughtEntry = entries.find((entry) => entry.kind === "thought");
      const suppressionEntry = entries.find((entry) => entry.kind === "agent_suppressed");
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(result.emitted).toBe(false);
      expect(result.response).toBe("");
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "finalizer_no_output",
      });
      expect(entries.some((entry) => entry.kind === "agent_msg")).toBe(false);
      expect(suppressionEntry?.content).toMatchObject({
        reason: "finalizer_no_output",
      });
      expect(activeStop).toMatchObject({
        provenance: "finalizer_no_output",
        source_stream_entry_id: suppressionEntry?.id,
        since_turn: 1,
      });
    } finally {
      await borg.close();
    }
  });

  it("consumes a detected closure loop through S2 no-output before suppressing later closure", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_120_000);
    const closureSourceEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        createGoalPromotionResponse([]),
        createClosureLoopSignoffResponseFromRequest(),
        createGenerationGateResponse({
          decision: "proceed",
          substantive: true,
        }),
        createNoOutputTurnPlanResponse(),
        createEmitNoOutputResponse("The planner recommended no assistant message."),
        createGoalPromotionResponse([]),
        createClosureLoopSignoffResponseFromRequest(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "workingMemoryStore">;
    };

    try {
      const workingMemory = internal.deps.workingMemoryStore.load("default" as never);
      internal.deps.workingMemoryStore.save(
        setClosureLoopDetected(workingMemory, {
          sourceStreamEntryIds: [closureSourceEntryId],
          reason: "Two mutual closure cycles detected.",
          sinceTurn: workingMemory.turn_counter,
        }),
      );

      const first = await borg.turn({
        userMessage: "phone down",
        stakes: "high",
      });
      const afterFirst = internal.deps.workingMemoryStore.load("default" as never);

      expect(first.emission).toMatchObject({
        kind: "suppressed",
        reason: "finalizer_no_output",
      });
      expect(afterFirst.discourse_state?.closure_loop?.status).toBe("named");

      const second = await borg.turn({
        userMessage: "phone still down",
        stakes: "high",
      });

      expect(second.emission).toMatchObject({
        kind: "suppressed",
        reason: "finalizer_no_output",
      });
      expect(second.response).toBe("");
    } finally {
      await borg.close();
    }
  });

  it("suppresses the turn when commitment revision still violates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_150_000);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const commitment = borg.commitments.add({
        type: "boundary",
        directiveFamily: "launch_date_boundary",
        directive: "Do not disclose launch dates.",
        priority: 10,
        provenance: { kind: "manual" },
      });

      llm.pushResponse(createEmitAnswerResponse("The launch is tomorrow."));
      llm.pushResponse(
        createCommitmentJudgeResponse([
          {
            commitment_id: commitment.id,
            reason: "Discloses a launch date.",
          },
        ]),
      );
      llm.pushResponse({
        text: "The launch is still tomorrow.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(
        createCommitmentJudgeResponse([
          {
            commitment_id: commitment.id,
            reason: "Still discloses a launch date after rewrite.",
          },
        ]),
      );

      const result = await borg.turn({
        userMessage: "When is launch?",
      });
      const entries = borg.stream.tail(10);
      const suppressionEntry = entries.find((entry) => entry.kind === "agent_suppressed");
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(result.emitted).toBe(false);
      expect(result.response).toBe("");
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "commitment_revision_failed",
      });
      expect(entries.some((entry) => entry.kind === "agent_msg")).toBe(false);
      expect(suppressionEntry?.content).toMatchObject({
        reason: "commitment_revision_failed",
      });
      expect(activeStop).toMatchObject({
        provenance: "commitment_guard",
        source_stream_entry_id: suppressionEntry?.id,
        since_turn: 1,
      });
    } finally {
      await borg.close();
    }
  });

  it("persists user corrective preferences through identity with audience and stream source", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_175_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          closure_pressure_relevance: "no_closure",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        createEmitAnswerResponse("I will adjust that pattern."),
        createCommitmentJudgeResponse([]),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository" | "identityService">;
    };
    const addCommitmentSpy = vi.spyOn(internal.deps.identityService, "addCommitment");

    try {
      await borg.turn({
        userMessage: "You keep doing those little closing lines. Stop that.",
        audience: "Sam",
      });

      const userEntry = borg.stream.tail(10).find((entry) => entry.kind === "user_msg");
      const samEntityId = internal.deps.entityRepository.findByName("Sam");
      const addInput = addCommitmentSpy.mock.calls[0]?.[0];
      const commitments = borg.commitments.list({
        activeOnly: true,
        audience: "Sam",
      });

      expect(addInput).toMatchObject({
        type: "preference",
        directive: "Do not add ritual closing lines when the conversation is open.",
        closurePressureRelevance: "no_closure",
        priority: 8,
        restrictedAudience: samEntityId,
        sourceStreamEntryIds: [userEntry?.id],
      });
      expect(commitments).toEqual([
        expect.objectContaining({
          restricted_audience: samEntityId,
          closure_pressure_relevance: "no_closure",
          source_stream_entry_ids: [userEntry?.id],
        }),
      ]);
    } finally {
      await borg.close();
    }
  });

  it("supersedes active corrective preferences selected by the extractor during a turn", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_180_000);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "commitmentRepository">;
    };

    try {
      const original = borg.commitments.add({
        type: "preference",
        directiveFamily: "ritual_closing",
        directive: "Avoid ritual closing lines.",
        priority: 6,
        provenance: { kind: "manual" },
      });
      llm.pushResponse(
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the user has not asked for closure.",
          directive_family: "ritual_closing",
          closure_pressure_relevance: "no_closure",
          priority: 9,
          reason: "The user tightened an existing durable response preference.",
          confidence: 0.93,
          supersedes_commitment_id: original.id,
        }),
      );
      llm.pushResponse(createEmitAnswerResponse("I will keep it direct."));
      llm.pushResponse(createCommitmentJudgeResponse([]));
      llm.pushResponse(createEmptyReflectionResponse());

      await borg.turn({
        userMessage: "Tighter rule: no ritual closing lines unless I ask for closure.",
      });

      const replacement = borg.commitments
        .list({ activeOnly: true })
        .find((commitment) => commitment.directive_family === "ritual_closing");
      const sameFamilyRows = borg.commitments
        .list({ activeOnly: false })
        .filter((commitment) => commitment.directive_family === "ritual_closing");

      expect(replacement).toBeDefined();
      expect(replacement?.id).not.toBe(original.id);
      expect(sameFamilyRows).toHaveLength(2);
      expect(internal.deps.commitmentRepository.get(original.id)).toMatchObject({
        superseded_by: replacement?.id,
      });
      expect(
        borg.commitments
          .list({ activeOnly: true })
          .filter((commitment) => commitment.directive_family === "ritual_closing"),
      ).toEqual([replacement]);
      expect(replacement).toMatchObject({
        revoked_at: null,
        superseded_by: null,
      });
    } finally {
      await borg.close();
    }
  });

  it("marks failed turns aborted and keeps retry state clean", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_190_000);
    const failedUserMessage = "Stop using ritual closing lines.";
    const failedReason = "finalizer exploded";
    const retryResponse = "I will keep it direct.";
    let correctiveCalls = 0;
    let finalizerCalls = 0;
    const scriptedResponse = (options: LLMCompleteOptions) => {
      if (options.budget === "corrective-preference-extractor") {
        correctiveCalls += 1;
        return correctiveCalls === 1
          ? createCorrectivePreferenceResponse({
              classification: "corrective_preference",
              type: "preference",
              directive: "Do not add ritual closing lines.",
              closure_pressure_relevance: "no_closure",
              priority: 8,
              reason: "The user named a future response pattern to stop.",
              confidence: 0.9,
            })
          : createCorrectivePreferenceResponse({
              classification: "none",
              reason: "No new preference.",
              confidence: 0.9,
            });
      }

      if (options.budget === "generation-gate") {
        return createGenerationGateResponse({ decision: "proceed", substantive: true });
      }

      if (options.budget === "recall-expansion") {
        return createRecallExpansionResponse();
      }

      if (options.budget === "cognition-plan") {
        return createTurnPlanResponse();
      }

      if (options.budget === "cognition-system-2") {
        finalizerCalls += 1;
        if (finalizerCalls === 1) {
          throw new Error(failedReason);
        }
        return createEmitAnswerResponse(retryResponse);
      }

      if (options.budget === "cognition-system-1") {
        return createEmitAnswerResponse(retryResponse);
      }

      if (options.budget === "commitment-judge") {
        return createCommitmentJudgeResponse([]);
      }

      if (options.budget === "generation-stop-commitment") {
        return createStopCommitmentResponse({ classification: "none" });
      }

      if (options.budget === "reflection") {
        return createEmptyReflectionResponse();
      }

      return createEmitAnswerResponse(retryResponse);
    };
    const llm = new FakeLLMClient({
      responses: Array.from({ length: 20 }, () => scriptedResponse),
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      await expect(
        borg.turn({
          userMessage: failedUserMessage,
          stakes: "high",
        }),
      ).rejects.toThrow(failedReason);

      const abortedEntries = borg.stream.tail(20);
      const abortedMarker = abortedEntries.find((entry) => {
        const content = entry.content as { event?: unknown };
        return entry.kind === "internal_event" && content.event === "aborted_turn";
      });
      const abortedUserEntry = abortedEntries.find((entry) => entry.kind === "user_msg");
      const abortedPlanEntry = abortedEntries.find((entry) => entry.kind === "thought");

      expect(abortedMarker?.turn_status).toBe("aborted");
      expect(abortedMarker?.content).toMatchObject({
        turn_id: abortedUserEntry?.turn_id,
        reason: expect.stringContaining(failedReason),
      });
      expect(abortedPlanEntry?.turn_id).toBe(abortedUserEntry?.turn_id);
      expect(borg.commitments.list({ activeOnly: true })).toEqual([]);
      expect(borg.workmem.load()).toMatchObject({
        turn_counter: 0,
        pending_actions: [],
      });

      const retry = await borg.turn({
        userMessage: failedUserMessage,
      });
      const retryFinalizer = finalizerRequests(llm.requests).at(-1);
      const retryUserMessages = requestTextMessages(retryFinalizer).filter(
        (message) => message === failedUserMessage,
      );

      expect(retry.response).toBe(retryResponse);
      expect(retryUserMessages).toEqual([failedUserMessage]);
      expect(borg.commitments.list({ activeOnly: true })).toEqual([]);
    } finally {
      await borg.close();
    }
  });

  it("does not count pending-action merges from aborted turn attempts", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_190_250);
    const failedReason = "reflection failed after pending-action merge";
    const userMessage = "Keep an eye on deployment metrics.";
    const response = "I will keep tracking it.";
    const pendingAction: IntentRecord = {
      description: "Follow up on deployment metrics",
      next_action: "Check the deployment metrics status",
    };
    let sawPendingActionJudge = false;
    let threwAfterMerge = false;
    const scriptedResponse = (options: LLMCompleteOptions) => {
      if (options.budget === "corrective-preference-extractor") {
        return createCorrectivePreferenceResponse({
          classification: "none",
          reason: "No new preference.",
          confidence: 0.9,
        });
      }

      if (options.budget === "generation-gate") {
        return createGenerationGateResponse({ decision: "proceed", substantive: true });
      }

      if (options.budget === "recall-expansion") {
        return createRecallExpansionResponse();
      }

      if (options.budget === "cognition-plan") {
        return createTurnPlanResponse([pendingAction]);
      }

      if (options.budget === "cognition-system-2" || options.budget === "cognition-system-1") {
        return createEmitAnswerResponse(response);
      }

      if (options.budget === "pending-action-judge") {
        sawPendingActionJudge = true;
        return createPendingActionJudgeResponse("action");
      }

      if (options.budget === "generation-stop-commitment") {
        return createStopCommitmentResponse({ classification: "none" });
      }

      if (options.budget === "commitment-judge") {
        return createCommitmentJudgeResponse([]);
      }

      if (options.budget === "reflection") {
        return createEmptyReflectionResponse();
      }

      return createEmitAnswerResponse(response);
    };
    const llm = new FakeLLMClient({
      responses: Array.from({ length: 30 }, () => scriptedResponse),
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "workingMemoryStore">;
    };

    try {
      const workingMemory = internal.deps.workingMemoryStore.load("default" as never);
      const originalSave = internal.deps.workingMemoryStore.save.bind(
        internal.deps.workingMemoryStore,
      );
      internal.deps.workingMemoryStore.save({
        ...workingMemory,
        pending_actions: [pendingAction],
        updated_at: clock.now(),
      });
      internal.deps.workingMemoryStore.recordPendingActionMerges(1);
      vi.spyOn(internal.deps.workingMemoryStore, "save").mockImplementation((memory) => {
        if (sawPendingActionJudge && !threwAfterMerge) {
          threwAfterMerge = true;
          throw new Error(failedReason);
        }

        return originalSave(memory);
      });

      await expect(
        borg.turn({
          userMessage,
          stakes: "high",
        }),
      ).rejects.toThrow(failedReason);

      expect(internal.deps.workingMemoryStore.getPendingActionMergeCount()).toBe(1);

      await borg.turn({
        userMessage,
        stakes: "high",
      });

      expect(internal.deps.workingMemoryStore.getPendingActionMergeCount()).toBe(2);
    } finally {
      await borg.close();
    }
  });

  it("rolls back corrective slot negations when a later turn phase aborts", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_190_500);
    const failedReason = "finalizer exploded after slot negation";
    let internal: {
      deps: Pick<BorgDependencies, "entityRepository" | "relationalSlotRepository">;
    };
    const scriptedResponse = (options: LLMCompleteOptions) => {
      if (options.budget === "corrective-preference-extractor") {
        const tom = internal.deps.entityRepository.findByName("Tom");

        return createCorrectivePreferenceResponse({
          classification: "none",
          slot_negations:
            tom === null
              ? []
              : [
                  {
                    subject_entity_id: tom,
                    slot_key: "partner.name",
                    rejected_value: "Sarah",
                    source_stream_entry_ids: [createStreamEntryId()],
                    confidence: 0.95,
                  },
                ],
        });
      }

      if (options.budget === "generation-gate") {
        return createGenerationGateResponse({ decision: "proceed", substantive: true });
      }

      if (options.budget === "recall-expansion") {
        return createRecallExpansionResponse();
      }

      if (options.budget === "cognition-plan") {
        return createTurnPlanResponse();
      }

      if (options.budget === "cognition-system-2") {
        throw new Error(failedReason);
      }

      return createEmptyReflectionResponse();
    };
    const llm = new FakeLLMClient({
      responses: Array.from({ length: 20 }, () => scriptedResponse),
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository" | "relationalSlotRepository">;
    };

    try {
      const tom = internal.deps.entityRepository.resolve("Tom");
      const evidenceId = createStreamEntryId();
      const prior = internal.deps.relationalSlotRepository.applyAssertion({
        subject_entity_id: tom,
        slot_key: "partner.name",
        asserted_value: "Sarah",
        source_stream_entry_ids: [evidenceId],
      }).slot;

      await expect(
        borg.turn({
          userMessage: "Her name is not Sarah.",
          stakes: "high",
        }),
      ).rejects.toThrow(failedReason);

      expect(
        internal.deps.relationalSlotRepository.findBySubjectAndKey(tom, "partner.name"),
      ).toEqual(prior);
    } finally {
      await borg.close();
    }
  });

  it("reverts reflector-side writes when the turn aborts after reflection", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_191_000);
    const failedReason = "post-reflection save failed";
    const scriptedResponse = (options: LLMCompleteOptions) => {
      if (options.budget === "corrective-preference-extractor") {
        return createCorrectivePreferenceResponse({ classification: "none" });
      }

      if (options.budget === "action-state-extractor") {
        return createActionStateResponse([]);
      }

      if (options.budget === "goal-promotion-extractor") {
        return createGoalPromotionResponse([]);
      }

      if (options.budget === "generation-gate") {
        return createGenerationGateResponse({ decision: "proceed", substantive: true });
      }

      if (options.budget === "recall-expansion") {
        return createRecallExpansionResponse();
      }

      if (options.budget === "cognition-system-1") {
        return createEmitAnswerResponse("Completed the check.");
      }

      if (options.budget === "commitment-judge") {
        return createCommitmentJudgeResponse([]);
      }

      if (options.budget === "generation-stop-commitment") {
        return createStopCommitmentResponse({ classification: "none" });
      }

      if (options.budget === "reflection") {
        const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
          active_goals?: Array<{ goal_id: string }>;
          active_open_questions?: Array<{ id: string }>;
          current_turn_stream_entry_ids?: string[];
        };
        const goalId = payload.active_goals?.[0]?.goal_id;
        const questionId = payload.active_open_questions?.[0]?.id;
        const currentStreamEntryId = payload.current_turn_stream_entry_ids?.[0];

        return {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use" as const,
          tool_calls: [
            {
              id: "toolu_reflection",
              name: "EmitTurnReflection",
              input: {
                advanced_goals:
                  goalId === undefined
                    ? []
                    : [{ goal_id: goalId, evidence: "The check was completed." }],
                procedural_outcomes: [],
                trait_demonstrations: [],
                intent_updates: [
                  {
                    description: "Check deployment state",
                    next_action: null,
                    actor: "borg",
                    status: "completed",
                    confidence: 0.9,
                    evidence: "Borg completed the check.",
                  },
                ],
                step_outcomes: [],
                proposed_steps:
                  goalId === undefined
                    ? []
                    : [
                        {
                          goal_id: goalId,
                          description: "Review the next deployment checkpoint",
                          kind: "think",
                          due_at: null,
                          rationale: "Keep the active goal moving.",
                        },
                      ],
                open_questions: [
                  {
                    question: "What deployment follow-up remains?",
                    urgency: 0.6,
                    related_episode_ids: [],
                  },
                ],
                resolved_open_questions:
                  questionId === undefined || currentStreamEntryId === undefined
                    ? []
                    : [
                        {
                          question_id: questionId,
                          resolution_note: "The turn answered the deployment question.",
                          evidence_episode_ids: [],
                          evidence_stream_entry_ids: [currentStreamEntryId],
                        },
                      ],
              },
            },
          ],
        };
      }

      return createEmptyReflectionResponse();
    };
    const llm = new FakeLLMClient({
      responses: Array.from({ length: 20 }, () => scriptedResponse),
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<
        BorgDependencies,
        | "actionRepository"
        | "executiveStepsRepository"
        | "goalsRepository"
        | "openQuestionsRepository"
        | "workingMemoryStore"
      >;
    };

    try {
      const goal = borg.self.goals.add({
        description: "Keep deployment state current",
        priority: 10,
        provenance: { kind: "manual" },
      });
      const question = borg.self.openQuestions.add({
        question: "What is the deployment state?",
        urgency: 0.8,
        related_episode_ids: [],
        related_semantic_node_ids: [],
        provenance: { kind: "manual" },
        source: "user",
      });
      const workingMemory = internal.deps.workingMemoryStore.load("default" as never);
      const originalSave = internal.deps.workingMemoryStore.save.bind(
        internal.deps.workingMemoryStore,
      );
      let threwAfterReflection = false;

      internal.deps.workingMemoryStore.save({
        ...workingMemory,
        pending_actions: [
          {
            description: "Check deployment state",
            next_action: null,
          },
        ],
        updated_at: clock.now(),
      });
      vi.spyOn(internal.deps.workingMemoryStore, "save").mockImplementation((memory) => {
        if (
          !threwAfterReflection &&
          memory.turn_counter > 0 &&
          memory.pending_actions.length === 0
        ) {
          threwAfterReflection = true;
          throw new Error(failedReason);
        }

        return originalSave(memory);
      });

      await expect(
        borg.turn({
          userMessage: "Please check the deployment state.",
          stakes: "low",
        }),
      ).rejects.toThrow(failedReason);

      expect(internal.deps.actionRepository.list({ limit: 10 })).toEqual([]);
      expect(internal.deps.executiveStepsRepository.list(goal.id)).toEqual([]);
      expect(internal.deps.goalsRepository.get(goal.id)?.progress_notes).toBeNull();
      expect(internal.deps.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "open",
        resolution_note: null,
        resolved_at: null,
      });
      expect(
        internal.deps.openQuestionsRepository.list({ status: "open" }).map((item) => item.id),
      ).toEqual([question.id]);
    } finally {
      await borg.close();
    }
  });

  it("applies corrective slot negations and sanitizes pending actions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_175_500);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<
        BorgDependencies,
        "entityRepository" | "relationalSlotRepository" | "workingMemoryStore"
      >;
    };

    try {
      const tom = internal.deps.entityRepository.resolve("Tom");
      internal.deps.relationalSlotRepository.applyAssertion({
        subject_entity_id: tom,
        slot_key: "partner.name",
        asserted_value: "Sarah",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      const workingMemory = internal.deps.workingMemoryStore.load("default" as never);
      internal.deps.workingMemoryStore.save({
        ...workingMemory,
        pending_actions: [
          {
            description: "Track whether Tom raises the planning comment with Sarah directly",
            next_action: "Ask Sarah if Tom brings it up",
          },
        ],
        updated_at: clock.now(),
      });
      llm.pushResponse(
        createCorrectivePreferenceResponse({
          classification: "none",
          reason: "The user rejected a stored relational name.",
          confidence: 0.95,
          slot_negations: [
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              rejected_value: "Sarah",
              source_stream_entry_ids: [createStreamEntryId()],
              confidence: 0.95,
            },
          ],
        }),
      );
      llm.pushResponse(createEmitAnswerResponse("I will avoid using that name."));
      llm.pushResponse(createEmptyReflectionResponse());

      await borg.turn({
        userMessage: "Her name is not Sarah.",
      });

      const slot = internal.deps.relationalSlotRepository.findBySubjectAndKey(tom, "partner.name");
      const nextWorkingMemory = internal.deps.workingMemoryStore.load("default" as never);

      expect(slot?.state).toBe("quarantined");
      expect(nextWorkingMemory.pending_actions).toEqual([
        {
          description: "Track whether Tom raises the planning comment with your partner directly",
          next_action: "Ask your partner if Tom brings it up",
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("promotes user goals through identity with audience, stream source, and initial step", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_500);
    const targetAt = 1_800_100_000_000;
    const stepDueAt = 1_800_050_000_000;
    const llm = new FakeLLMClient({
      responses: [
        createGoalPromotionResponse([
          {
            description: "Help the user keep the Monday postmortem straight",
            priority: 9,
            target_at: targetAt,
            reason: "The user asked Borg to help keep the postmortem organized.",
            confidence: 0.91,
            initial_step: {
              description: "Ask what must be included in the postmortem",
              kind: "ask_user",
              due_at: stepDueAt,
              rationale: "Borg needs the postmortem constraints to help track it.",
            },
          },
        ]),
        createEmitAnswerResponse("I will keep the postmortem straight."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository" | "identityService"> & {
        executiveStepsRepository: ExecutiveStepsRepository;
      };
    };
    const addGoalSpy = vi.spyOn(internal.deps.identityService, "addGoal");

    try {
      await borg.turn({
        userMessage: "Write postmortem Monday, help me keep this straight.",
        audience: "Sam",
      });

      const userEntry = borg.stream.tail(10).find((entry) => entry.kind === "user_msg");
      const samEntityId = internal.deps.entityRepository.findByName("Sam");
      const addInput = addGoalSpy.mock.calls[0]?.[0];
      const goals = borg.self.goals.list({
        status: "active",
        visibleToAudienceEntityId: samEntityId,
      });
      const promotedGoal = goals.find(
        (goal) => goal.description === "Help the user keep the Monday postmortem straight",
      );
      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));

      expect(addInput).toMatchObject({
        description: "Help the user keep the Monday postmortem straight",
        priority: 9,
        status: "active",
        targetAt,
        audienceEntityId: samEntityId,
        sourceStreamEntryIds: [userEntry?.id],
      });
      expect(promotedGoal).toMatchObject({
        status: "active",
        target_at: targetAt,
        audience_entity_id: samEntityId,
        source_stream_entry_ids: [userEntry?.id],
      });
      expect(promotedGoal).toBeDefined();
      expect(
        internal.deps.executiveStepsRepository.list(promotedGoal!.id).map((step) => ({
          goal_id: step.goal_id,
          description: step.description,
          kind: step.kind,
          due_at: step.due_at,
        })),
      ).toEqual([
        {
          goal_id: promotedGoal!.id,
          description: "Ask what must be included in the postmortem",
          kind: "ask_user",
          due_at: stepDueAt,
        },
      ]);
      expect(finalizerSystem).toContain("Help the user keep the Monday postmortem straight");
    } finally {
      await borg.close();
    }
  });

  it("persists current-turn action states before deliberation runs", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_550);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "none",
          reason: "No durable correction detected.",
          confidence: 0,
        }),
        Object.assign(
          (options: LLMCompleteOptions) => {
            expect(options.budget).toBe("action-state-extractor");
            expect(options.model).toBe("test-recall");
            const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
              current_user_stream_entry_id: string;
            };

            return createActionStateResponse([
              {
                description: "booked the tutor Tuesday 7pm",
                state: "completed",
                evidence_stream_entry_ids: [payload.current_user_stream_entry_id],
                confidence: 0.95,
              },
            ]);
          },
          { budget: "action-state-extractor" },
        ),
        createGoalPromotionResponse([]),
        createEmitAnswerResponse("I see the tutor booking is done."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const originalRun = Deliberator.prototype.run;
    const runSpy = vi.spyOn(Deliberator.prototype, "run").mockImplementation(function (
      this: Deliberator,
      ...args: Parameters<Deliberator["run"]>
    ) {
      expect(borg.actions.list({ state: "completed" })).toEqual([
        expect.objectContaining({
          description: "booked the tutor Tuesday 7pm",
          state: "completed",
        }),
      ]);

      return originalRun.apply(this, args);
    });

    try {
      await borg.turn({
        userMessage: "I booked the tutor Tuesday 7pm.",
        stakes: "low",
      });

      expect(runSpy).toHaveBeenCalledOnce();
      expect(borg.actions.list({ state: "completed" })).toEqual([
        expect.objectContaining({
          description: "booked the tutor Tuesday 7pm",
          provenance_stream_entry_ids: [expect.any(String)],
        }),
      ]);
    } finally {
      runSpy.mockRestore();
      await borg.close();
    }
  });

  it("quarantines frame-anomalous user turns before early extractors and passes the flag to deliberation", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_575);
    const llm = new FakeLLMClient({
      responses: [
        createFrameAnomalyResponse({
          kind: "frame_assignment_claim",
          confidence: 0.97,
          rationale: "The user claims the assistant was playing Tom.",
        }),
        createEmitAnswerResponse("I do not have evidence for that frame."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const extractSpy = vi.spyOn(ActionStateExtractor.prototype, "extract");
    const originalRun = Deliberator.prototype.run;
    const runSpy = vi.spyOn(Deliberator.prototype, "run").mockImplementation(function (
      this: Deliberator,
      ...args: Parameters<Deliberator["run"]>
    ) {
      expect(args[0].frameAnomaly).toMatchObject({
        status: "ok",
        kind: "frame_assignment_claim",
        confidence: 0.97,
      });

      return originalRun.apply(this, args);
    });

    try {
      await borg.turn({
        userMessage: "You were playing Tom in that exchange.",
        stakes: "low",
      });

      const budgets = llm.requests.map((request) => request.budget);
      const anomalyEvent = borg.stream.tail(20).find((entry) => {
        const content = entry.content as { event?: unknown };

        return entry.kind === "internal_event" && content.event === "frame_anomaly_gate";
      });
      const quarantineEvent = borg.stream.tail(20).find((entry) => {
        const content = entry.content as { event?: unknown };

        return entry.kind === "internal_event" && content.event === QUARANTINED_USER_ENTRY_EVENT;
      });

      expect(budgets).toContain("frame-anomaly-classifier");
      expect(budgets).not.toContain("corrective-preference-extractor");
      expect(budgets).not.toContain("action-state-extractor");
      expect(budgets).not.toContain("goal-promotion-extractor");
      expect(extractSpy).not.toHaveBeenCalled();
      expect(anomalyEvent?.content).toMatchObject({
        event: "frame_anomaly_gate",
        kind: "frame_assignment_claim",
        source_stream_entry_id: expect.any(String),
        cited_stream_entry_ids: [expect.any(String)],
      });
      expect(quarantineEvent?.content).toMatchObject({
        event: QUARANTINED_USER_ENTRY_EVENT,
        kind: "frame_assignment_claim",
        source_stream_entry_id: expect.any(String),
        cited_stream_entry_ids: [expect.any(String)],
      });
      expect(runSpy).toHaveBeenCalledOnce();
      expect(borg.actions.list({ limit: 10 })).toEqual([]);
    } finally {
      extractSpy.mockRestore();
      runSpy.mockRestore();
      await borg.close();
    }
  });

  it("uses the degraded fallback to fail closed on catastrophic frame phrases", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const clock = new ManualClock(1_800_000_176_580);
    const degradedFrameClassifier = Object.assign(
      () => {
        throw new Error("frame classifier unavailable");
      },
      { budget: "frame-anomaly-classifier" },
    );
    const llm = new FakeLLMClient({
      responses: [
        degradedFrameClassifier,
        createEmitAnswerResponse("I will avoid treating that as memory."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });
    const extractSpy = vi.spyOn(ActionStateExtractor.prototype, "extract");
    const originalRun = Deliberator.prototype.run;
    const runSpy = vi.spyOn(Deliberator.prototype, "run").mockImplementation(function (
      this: Deliberator,
      ...args: Parameters<Deliberator["run"]>
    ) {
      expect(args[0].frameAnomaly).toMatchObject({
        status: "ok",
        kind: "assistant_self_claim_in_user_role",
      });

      return originalRun.apply(this, args);
    });

    try {
      await borg.turn({
        userMessage: "I'm Claude and I had the role assignment inverted.",
        stakes: "low",
      });

      const budgets = llm.requests.map((request) => request.budget);
      const quarantineEvent = borg.stream.tail(20).find((entry) => {
        const content = entry.content as { event?: unknown };

        return entry.kind === "internal_event" && content.event === QUARANTINED_USER_ENTRY_EVENT;
      });
      const traceEvents = readTraceEvents(tracePath);

      expect(budgets).toContain("frame-anomaly-classifier");
      expect(budgets).not.toContain("corrective-preference-extractor");
      expect(budgets).not.toContain("action-state-extractor");
      expect(budgets).not.toContain("goal-promotion-extractor");
      expect(extractSpy).not.toHaveBeenCalled();
      expect(quarantineEvent?.content).toMatchObject({
        event: QUARANTINED_USER_ENTRY_EVENT,
        kind: "assistant_self_claim_in_user_role",
      });
      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "frame_anomaly_classifier_degraded",
          reason: "llm_failed",
        }),
      );
      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "frame_anomaly_degraded_fallback_match",
          pattern: "i'm claude",
          kind: "assistant_self_claim_in_user_role",
        }),
      );
      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "frame_anomaly_quarantine_appended",
          kind: "assistant_self_claim_in_user_role",
        }),
      );
      expect(runSpy).toHaveBeenCalledOnce();
      expect(borg.actions.list({ limit: 10 })).toEqual([]);
    } finally {
      extractSpy.mockRestore();
      runSpy.mockRestore();
      await borg.close();
    }
  });

  it("treats degraded frame classification without fallback match as normal", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "trace.jsonl");
    const clock = new ManualClock(1_800_000_176_582);
    const degradedFrameClassifier = Object.assign(
      () => {
        throw new Error("frame classifier unavailable");
      },
      { budget: "frame-anomaly-classifier" },
    );
    const llm = new FakeLLMClient({
      responses: [
        degradedFrameClassifier,
        createCorrectivePreferenceResponse({
          classification: "none",
          reason: "No durable correction detected.",
          confidence: 0,
        }),
        Object.assign(
          (options: LLMCompleteOptions) => {
            const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
              current_user_stream_entry_id: string;
            };

            return createActionStateResponse([
              {
                description: "closed the laptop for the night",
                state: "completed",
                evidence_stream_entry_ids: [payload.current_user_stream_entry_id],
                confidence: 0.93,
              },
            ]);
          },
          { budget: "action-state-extractor" },
        ),
        createGoalPromotionResponse([]),
        createEmitAnswerResponse("Talk tomorrow."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });

    try {
      await borg.turn({
        userMessage: "Closing the laptop. Talk tomorrow.",
        stakes: "low",
      });

      const budgets = llm.requests.map((request) => request.budget);
      const quarantineEvent = borg.stream.tail(20).find((entry) => {
        const content = entry.content as { event?: unknown };

        return entry.kind === "internal_event" && content.event === QUARANTINED_USER_ENTRY_EVENT;
      });
      const traceEvents = readTraceEvents(tracePath);

      expect(budgets).toContain("frame-anomaly-classifier");
      expect(budgets).toContain("corrective-preference-extractor");
      expect(budgets).toContain("action-state-extractor");
      expect(budgets).toContain("goal-promotion-extractor");
      expect(quarantineEvent).toBeUndefined();
      expect(traceEvents).toContainEqual(
        expect.objectContaining({
          event: "frame_anomaly_degraded_fallback_normal",
        }),
      );
      expect(borg.actions.list({ state: "completed" })).toEqual([
        expect.objectContaining({
          description: "closed the laptop for the night",
        }),
      ]);
    } finally {
      await borg.close();
    }
  });

  it("runs early extractors for normal user turns after the frame-anomaly classifier passes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_585);
    const llm = new FakeLLMClient({
      responses: [
        createFrameAnomalyResponse({
          kind: "normal",
          confidence: 0.91,
          rationale: "The message reports a normal user-world action.",
        }),
        createCorrectivePreferenceResponse({
          classification: "none",
          reason: "No durable correction detected.",
          confidence: 0,
        }),
        Object.assign(
          (options: LLMCompleteOptions) => {
            expect(options.budget).toBe("action-state-extractor");
            const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
              current_user_stream_entry_id: string;
            };

            return createActionStateResponse([
              {
                description: "closed the laptop for the night",
                state: "completed",
                evidence_stream_entry_ids: [payload.current_user_stream_entry_id],
                confidence: 0.93,
              },
            ]);
          },
          { budget: "action-state-extractor" },
        ),
        createGoalPromotionResponse([]),
        createEmitAnswerResponse("Talk tomorrow."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      await borg.turn({
        userMessage: "Closing the laptop. Talk tomorrow.",
        stakes: "low",
      });

      const budgets = llm.requests.map((request) => request.budget);

      expect(budgets).toContain("frame-anomaly-classifier");
      expect(budgets).toContain("corrective-preference-extractor");
      expect(budgets).toContain("action-state-extractor");
      expect(budgets).toContain("goal-promotion-extractor");
      expect(borg.actions.list({ state: "completed" })).toEqual([
        expect.objectContaining({
          description: "closed the laptop for the night",
        }),
      ]);
    } finally {
      await borg.close();
    }
  });

  it("persists at most three promoted goals from a five-candidate extraction", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_600);
    const llm = new FakeLLMClient({
      responses: [
        createGoalPromotionResponse([
          {
            description: "Help the user track the launch checklist",
            confidence: 0.95,
          },
          {
            description: "Help the user prepare the investor update",
            confidence: 0.94,
          },
          {
            description: "Help the user schedule the design review",
            confidence: 0.93,
          },
          {
            description: "Help the user collect beta feedback",
            confidence: 0.92,
          },
          {
            description: "Help the user plan the onboarding pass",
            confidence: 0.91,
          },
        ]),
        createEmitAnswerResponse("I will keep the active goals focused."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository" | "identityService">;
    };
    const addGoalSpy = vi.spyOn(internal.deps.identityService, "addGoal");

    try {
      await borg.turn({
        userMessage: "Keep track of launch, investor, design, beta, and onboarding work.",
        audience: "Sam",
      });

      const samEntityId = internal.deps.entityRepository.findByName("Sam");
      const goals = borg.self.goals.list({
        status: "active",
        visibleToAudienceEntityId: samEntityId,
      });

      expect(addGoalSpy).toHaveBeenCalledTimes(3);
      expect(goals.map((goal) => goal.description)).toEqual([
        "Help the user track the launch checklist",
        "Help the user prepare the investor update",
        "Help the user schedule the design review",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("emits a goal promotion degraded trace event when extraction fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "goal-promotion-degraded.jsonl");
    const clock = new ManualClock(1_800_000_176_650);
    const llm = new FakeLLMClient({
      responses: [
        (options: LLMCompleteOptions) => {
          expect(options.budget).toBe("goal-promotion-extractor");
          throw new Error("goal promotion transport failed");
        },
        createEmitAnswerResponse("I will continue without promoting a goal."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, undefined, {
      tracerPath: tracePath,
      env: {
        BORG_TRACE_PROMPTS: "1",
      },
    });

    try {
      await borg.turn({
        userMessage: "Keep this goal in view for later.",
        audience: "Sam",
      });
    } finally {
      await borg.close();
    }

    const degradedEvent = readTraceEvents(tracePath).find(
      (event) => event.event === "goal_promotion_extractor_degraded",
    );

    expect(degradedEvent).toMatchObject({
      event: "goal_promotion_extractor_degraded",
      reason: "llm_failed",
      error: "goal promotion transport failed",
    });
    expect(degradedEvent?.turnId).toEqual(expect.any(String));
  });

  it("does not create a duplicate goal when the extractor points at an existing goal", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_700);
    const existingGoalId = createGoalId();
    const llm = new FakeLLMClient({
      responses: [
        createGoalPromotionResponse([
          {
            description: "Help the user track their italki shortlist",
            duplicate_of_goal_id: existingGoalId,
            confidence: 0.95,
          },
        ]),
        createEmitAnswerResponse("I will keep it in mind."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository">;
    };

    try {
      const samEntityId = internal.deps.entityRepository.resolve("Sam");
      const existingGoal = borg.self.goals.add({
        id: existingGoalId,
        description: "Help the user track their italki shortlist",
        priority: 8,
        audienceEntityId: samEntityId,
        provenance: {
          kind: "manual",
        },
      });

      await borg.turn({
        userMessage: "Remind me about italki later.",
        audience: "Sam",
      });

      const goals = borg.self.goals.list({
        status: "active",
        visibleToAudienceEntityId: samEntityId,
      });

      expect(goals.map((goal) => goal.id)).toEqual([existingGoal.id]);
    } finally {
      await borg.close();
    }
  });

  it("enforces a corrective preference on the same turn by rewriting a violation", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          closure_pressure_relevance: "no_closure",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        createEmitAnswerResponse("Sleep well."),
        createDynamicCommitmentJudgeResponse("The response repeats the corrected closing pattern."),
        "I will leave it there.",
        createCommitmentJudgeResponse([]),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "You keep doing those little closing lines. Stop that.",
        audience: "Sam",
      });

      expect(result.emitted).toBe(true);
      expect(result.response).toBe("I will leave it there.");
      expect(llm.requests.map((request) => request.budget)).toContain("commitment-revision");
    } finally {
      await borg.close();
    }
  });

  it("records closure-pressure history when a mixed closure response is rewritten", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_500);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          closure_pressure_relevance: "no_closure",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        createActionStateResponse([]),
        createGoalPromotionResponse([]),
        createEmitAnswerResponse("Here is the actual answer. Sleep."),
        createCommitmentJudgeResponse([]),
        createMixedClosureResponseAuditResponse("Sleep."),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "Stop adding closing lines; keep the answer open.",
        audience: "Sam",
      });
      const history = borg.workmem.load().discourse_state?.closure_pressure_history ?? [];

      expect(result.emitted).toBe(true);
      expect(result.response).toBe("Here is the actual answer.");
      expect(history).toHaveLength(1);
      expect(history[0]).toMatchObject({
        reason: "span_removed",
      });
      expect(history[0]?.turn_id).toEqual(expect.any(String));
    } finally {
      await borg.close();
    }
  });

  it("suppresses pure corrective-preference violations when revision still violates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_177_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          closure_pressure_relevance: "no_closure",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        createEmitAnswerResponse("Sleep well."),
        createDynamicCommitmentJudgeResponse("The response repeats the corrected closing pattern."),
        "Sleep well.",
        createDynamicCommitmentJudgeResponse("The revision still repeats the corrected pattern."),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "You keep doing those little closing lines. Stop that.",
        audience: "Sam",
      });
      const suppressionEntry = borg.stream
        .tail(10)
        .find((entry) => entry.kind === "agent_suppressed");

      expect(result.emitted).toBe(false);
      expect(result.response).toBe("");
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "commitment_revision_failed",
      });
      expect(suppressionEntry?.content).toMatchObject({
        reason: "commitment_revision_failed",
      });
    } finally {
      await borg.close();
    }
  });

  it("loads durable corrective commitments on later turns", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_178_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          closure_pressure_relevance: "no_closure",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        createEmitAnswerResponse("I will adjust that pattern."),
        createCommitmentJudgeResponse([]),
        createEmptyReflectionResponse(),
        createEmitAnswerResponse("Sleep well."),
        createDynamicCommitmentJudgeResponse(
          "The later response repeats the durable corrected pattern.",
        ),
        "I will stop here.",
        createCommitmentJudgeResponse([]),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      await borg.turn({
        userMessage: "You keep doing those little closing lines. Stop that.",
        audience: "Sam",
      });
      clock.advance(5_000);

      const result = await borg.turn({
        userMessage: "Continue with the actual topic.",
        audience: "Sam",
      });

      expect(result.response).toBe("I will stop here.");
      expect(llm.requests.filter((request) => request.budget === "commitment-judge")).toHaveLength(
        3,
      );
    } finally {
      await borg.close();
    }
  });

  it("runs the generation gate before retrieval and finalization under active stop state", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_200_000);
    const embeddingClient = new CountingEmbeddingClient();
    const llm = new FakeLLMClient({
      responses: [
        createEmitAnswerResponse("I will stop responding until you bring substantive content."),
        createStopCommitmentResponse({
          classification: "stop_until_substantive_content",
        }),
        createEmptyReflectionResponse(),
        createGenerationGateResponse({
          decision: "suppress",
          substantive: false,
          reason: "The user sent another minimal probe under an active stop.",
        }),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, embeddingClient);

    try {
      await borg.turn({
        userMessage: "Stop responding if I keep sending filler.",
      });
      embeddingClient.embedTexts.length = 0;
      embeddingClient.embedBatchTexts.length = 0;
      const result = await borg.turn({
        userMessage: "No.",
      });
      const tailKinds = borg.stream.tail(6).map((entry) => entry.kind);

      expect(result.emitted).toBe(false);
      expect(result.response).toBe("");
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "active_discourse_stop",
      });
      expect(tailKinds.slice(-3)).toEqual(["user_msg", "perception", "agent_suppressed"]);
      expect(embeddingClient.embedTexts).toEqual([]);
      expect(embeddingClient.embedBatchTexts).toEqual([]);
      expect(borg.workmem.load().discourse_state?.stop_until_substantive_content).toMatchObject({
        provenance: "self_commitment_extractor",
      });
    } finally {
      await borg.close();
    }
  });
});
