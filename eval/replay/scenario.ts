import type { Borg } from "../../src/index.js";
import {
  FakeLLMClient,
  type LLMCompleteResult,
  type PostGenerationGuardMode,
  type RelationalClaimGuardMode,
  type TurnResult,
} from "../../src/index.js";
import type { BorgDependencies } from "../../src/borg/types.js";
import {
  CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
  type ClosureResponseAudit,
} from "../../src/cognition/generation/closure-pressure-guard.js";
import {
  type RelationalClaimAuditClaim,
  type RelationalClaimKind,
} from "../../src/cognition/generation/relational-guard.js";
import type { FrameAnomalyKind } from "../../src/cognition/frame-anomaly/index.js";
import type { Clock } from "../../src/util/clock.js";

export type ReplayPipelineId = "A" | "B" | "C" | "Cdoubleprime";

export type ReplayPipeline = {
  id: ReplayPipelineId;
  label: string;
  evidenceLedgerEnabled: boolean;
  emissionFinalizerEnabled: boolean;
  commitmentMode: PostGenerationGuardMode;
  relationalClaimMode: RelationalClaimGuardMode;
  closurePressureMode: PostGenerationGuardMode;
};

export type ScenarioDeps = {
  borg: Borg;
  deps: BorgDependencies;
  llm: FakeLLMClient;
  clock: Clock;
  tempDir: string;
  pipeline: ReplayPipeline;
};

export type ScenarioScriptContext = {
  pipeline: ReplayPipeline;
  enqueueBeforeRecall: (response: LLMCompleteResult) => void;
  enqueueAfterFinalizer: (response: string | LLMCompleteResult) => void;
};

export type ReplayFinalizerEmission =
  | {
      kind: "answer";
      text?: string;
    }
  | {
      kind: "self_report";
      text: string;
    }
  | {
      kind: "no_output";
      reason: string;
    };

export type ReplayScenario = {
  id: string;
  failureClass: string;
  description: string;
  seed: (deps: ScenarioDeps) => Promise<void>;
  userMessage: string;
  audience?: string;
  perceptionUseLlmFallback?: boolean;
  unsafeCandidateText: string;
  finalizerEmission?: ReplayFinalizerEmission;
  scriptLLMResponses: (client: FakeLLMClient, context: ScenarioScriptContext) => void;
  safeOutputPredicate: (emittedText: string) => boolean;
  usefulOutputPredicate?: (emittedText: string) => boolean;
  severeGuardCategories: string[];
  postRunAssert?: (
    deps: ScenarioDeps & {
      result: TurnResult;
      emittedText: string;
    },
  ) => Promise<void>;
  notes?: readonly string[];
};

export function textResponse(text: string): LLMCompleteResult {
  return {
    text,
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "end_turn",
    tool_calls: [],
  };
}

export function finalizerToolResponse(
  input: ReplayFinalizerEmission,
  fallbackText: string,
): LLMCompleteResult {
  const tool =
    input.kind === "answer"
      ? {
          id: "toolu_replay_emit_answer",
          name: "EmitAnswer",
          input: { text: input.text ?? fallbackText },
        }
      : input.kind === "self_report"
        ? {
            id: "toolu_replay_emit_self_report",
            name: "EmitSelfReport",
            input: { text: input.text },
          }
        : {
            id: "toolu_replay_emit_no_output",
            name: "EmitNoOutput",
            input: { reason: input.reason },
          };
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [tool],
  };
}

export function recallExpansionResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_recall_expansion",
        name: "EmitRecallExpansion",
        input: {
          facets: [],
          named_terms: [],
        },
      },
    ],
  };
}

export function emptyReflectionResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_reflection",
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

export function episodeExtractionResponse(
  episodes: readonly unknown[],
  relationalSlotUpdates: readonly unknown[] = [],
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_episode_extraction",
        name: "EmitEpisodeCandidates",
        input: {
          episodes,
          relational_slot_updates: relationalSlotUpdates,
        },
      },
    ],
  };
}

export function semanticExtractionResponse(input: {
  nodes: readonly unknown[];
  edges?: readonly unknown[];
}): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_semantic_extraction",
        name: "EmitSemanticCandidates",
        input: {
          nodes: input.nodes,
          edges: input.edges ?? [],
        },
      },
    ],
  };
}

export function promptForBudget(client: FakeLLMClient, budget: string, startIndex = 0): string {
  const request = client.requests
    .slice(startIndex)
    .find((candidate) => candidate.budget === budget);

  return String(request?.messages[0]?.content ?? "");
}

export function claimAuditResponse(
  claims: readonly RelationalClaimAuditClaim[],
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_claim_audit",
        name: "EmitClaimAudit",
        input: {
          claims,
        },
      },
    ],
  };
}

export function noClaimAuditResponse(): LLMCompleteResult {
  return claimAuditResponse([]);
}

export function makeRelationalClaim(
  overrides: Partial<RelationalClaimAuditClaim> & { kind: RelationalClaimKind },
): RelationalClaimAuditClaim {
  return {
    kind: overrides.kind,
    asserted: overrides.asserted ?? "unsupported replay claim",
    cited_stream_entry_ids: overrides.cited_stream_entry_ids ?? [],
    cited_episode_ids: overrides.cited_episode_ids ?? [],
    cited_commitment_ids: overrides.cited_commitment_ids ?? [],
    cited_action_ids: overrides.cited_action_ids ?? [],
    support_handles: overrides.support_handles ?? [],
    quoted_evidence_text: overrides.quoted_evidence_text ?? null,
    callback_scope:
      overrides.callback_scope ?? (overrides.kind === "callback" ? "prior_turn" : null),
    specific_detail_value:
      overrides.specific_detail_value ??
      (overrides.kind === "unsupported_specific_detail" ? "unsupported detail" : null),
    specific_detail_support_kind:
      overrides.specific_detail_support_kind ??
      (overrides.kind === "unsupported_specific_detail" ? "none" : null),
    subject_entity_id: overrides.subject_entity_id ?? null,
    slot_key: overrides.slot_key ?? null,
    relational_slot_value: overrides.relational_slot_value ?? null,
  };
}

export function noClosureAuditResponse(): LLMCompleteResult {
  return closureAuditResponse({
    spans: [],
    response_shape: "no_closure",
    reason: "Replay scenario has no closure-pressure span.",
  });
}

export function commitmentJudgeResponse(
  violations: readonly { commitment_id: string; reason: string; confidence?: number }[],
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_commitment_judge",
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

export function enqueueRelationalGuardFailure(
  context: ScenarioScriptContext,
  input: {
    claim: RelationalClaimAuditClaim;
    rewrite: string;
    closureAudit?: LLMCompleteResult;
  },
): void {
  context.enqueueAfterFinalizer(claimAuditResponse([input.claim]));
  context.enqueueAfterFinalizer(textResponse(input.rewrite));
  context.enqueueAfterFinalizer(noClaimAuditResponse());
  context.enqueueAfterFinalizer(input.closureAudit ?? noClosureAuditResponse());
}

export function enqueueRelationalGuardFailureWhenValidatorAbsent(
  context: ScenarioScriptContext,
  input: {
    claim: RelationalClaimAuditClaim;
    rewrite: string;
    closureAudit?: LLMCompleteResult;
  },
): void {
  enqueueRelationalGuardFailure(context, input);
}

export function enqueueRelationalGuardFailureWithShadowTrace(
  context: ScenarioScriptContext,
  input: {
    claim: RelationalClaimAuditClaim;
    rewrite: string;
    closureAudit?: LLMCompleteResult;
  },
): void {
  const closureAudit = input.closureAudit ?? noClosureAuditResponse();

  enqueueRelationalGuardFailure(context, {
    ...input,
    closureAudit,
  });
}

export function enqueueNoRelationalGuardIssue(
  context: ScenarioScriptContext,
  closureAudit: LLMCompleteResult = noClosureAuditResponse(),
): void {
  context.enqueueAfterFinalizer(noClaimAuditResponse());
  context.enqueueAfterFinalizer(closureAudit);
}

export function closureAuditResponse(audit: ClosureResponseAudit): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_closure_audit",
        name: CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
        input: audit,
      },
    ],
  };
}

export function frameAnomalyResponse(kind: FrameAnomalyKind): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_frame_anomaly",
        name: "ClassifyFrameAnomaly",
        input: {
          kind,
          confidence: 0.96,
          rationale: "Replay scenario scripts the v26 frame anomaly.",
        },
      },
    ],
  };
}

export function closureLoopClassificationResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_closure_loop",
        name: "ClassifyClosureLoopDialogueActs",
        input: {
          messages: [],
          confidence: 0.9,
          rationale: "Replay keeps the pre-seeded closure-loop state active.",
        },
      },
    ],
  };
}

export function safeRewrite(text: string): string {
  return text;
}

export function lowerIncludesNone(text: string, values: readonly string[]): boolean {
  const lowered = text.toLowerCase();

  return values.every((value) => !lowered.includes(value.toLowerCase()));
}
