import type { Borg } from "../../src/index.js";
import {
  FakeLLMClient,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type LLMConverseOptions,
  type PostGenerationGuardMode,
  type RelationalClaimGuardMode,
  type TurnResult,
} from "../../src/index.js";
import type { BorgDependencies } from "../../src/borg/types.js";
import type {
  EmitManifestResponse,
  EvidenceRef,
} from "../../src/cognition/deliberation/manifest-schema.js";
import {
  CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
  type ClosureResponseAudit,
} from "../../src/cognition/generation/closure-pressure-guard.js";
import {
  type RelationalClaimAuditClaim,
  type RelationalClaimKind,
} from "../../src/cognition/generation/relational-guard.js";
import type { EvidenceLedgerSourceType } from "../../src/cognition/evidence-ledger/index.js";
import type { FrameAnomalyKind } from "../../src/cognition/frame-anomaly/index.js";
import type { Clock } from "../../src/util/clock.js";

export const CURRENT_USER_EVIDENCE_ID = "$current_user_message";

export type ReplayPipelineId = "A" | "B" | "C" | "Cdoubleprime";

export type ReplayPipeline = {
  id: ReplayPipelineId;
  label: string;
  evidenceLedgerEnabled: boolean;
  manifestFinalizerEnabled: boolean;
  manifestValidatorEnabled: boolean;
  commitmentMode: PostGenerationGuardMode;
  relationalClaimMode: RelationalClaimGuardMode;
  closurePressureMode: PostGenerationGuardMode;
};

export type ScenarioDeps = {
  borg: Borg;
  deps: BorgDependencies;
  clock: Clock;
  tempDir: string;
  pipeline: ReplayPipeline;
};

export type EvidencePlaceholder = {
  sourceType?: EvidenceLedgerSourceType;
  textIncludes?: readonly string[];
  valueIncludes?: readonly string[];
  state?: string;
};

export type ScenarioScriptContext = {
  pipeline: ReplayPipeline;
  enqueueBeforeRecall: (response: LLMCompleteResult) => void;
  enqueueAfterFinalizer: (response: string | LLMCompleteResult) => void;
};

export type ReplayScenario = {
  id: string;
  failureClass: string;
  description: string;
  seed: (deps: ScenarioDeps) => Promise<void>;
  userMessage: string;
  unsafeCandidateText: string;
  manifestResponse: EmitManifestResponse;
  evidencePlaceholders?: Record<string, EvidencePlaceholder>;
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

export function evidenceRef(
  id: string,
  source_type: EvidenceLedgerSourceType,
): EvidenceRef {
  return {
    id,
    source_type,
  };
}

export function currentUserEvidenceRef(): EvidenceRef {
  return evidenceRef(CURRENT_USER_EVIDENCE_ID, "current_user_message");
}

export function placeholderEvidenceRef(
  placeholder: string,
  sourceType: EvidenceLedgerSourceType,
): EvidenceRef {
  return evidenceRef(`$evidence:${placeholder}`, sourceType);
}

export function textResponse(text: string): LLMCompleteResult {
  return {
    text,
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "end_turn",
    tool_calls: [],
  };
}

export function manifestFinalizerResponse(input: EmitManifestResponse): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_replay_manifest",
        name: "EmitManifestResponse",
        input,
      },
    ],
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
  if (context.pipeline.manifestValidatorEnabled) {
    enqueueNoRelationalGuardIssue(context, input.closureAudit ?? noClosureAuditResponse());
    return;
  }

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

function llmPromptText(options: LLMCompleteOptions | LLMConverseOptions): string {
  const system =
    typeof options.system === "string"
      ? options.system
      : (options.system?.map((block) => block.text).join("\n") ?? "");
  const messages = options.messages
    .map((message) => {
      if (typeof message.content === "string") {
        return message.content;
      }

      return message.content
        .map((block) => {
          if (block.type === "text") {
            return block.text;
          }

          if (block.type === "tool_use") {
            return block.name;
          }

          return typeof block.content === "string"
            ? block.content
            : block.content.map((part) => part.text).join("");
        })
        .join("\n");
    })
    .join("\n");

  return `${system}\n${messages}`;
}

type LedgerEntryForPrompt = {
  id: string;
  sourceType: EvidenceLedgerSourceType | null;
  text: string;
};

function extractLedgerEntries(prompt: string): LedgerEntryForPrompt[] {
  const entries: LedgerEntryForPrompt[] = [];
  const entryPattern = /- id=([^\s]+)([^\n]*)([\s\S]*?)(?=\n- id=|\n## |<\/borg_evidence_ledger>|$)/g;
  let match: RegExpExecArray | null;

  while ((match = entryPattern.exec(prompt)) !== null) {
    const [, id, metadata, body] = match;
    if (id === undefined || metadata === undefined || body === undefined) {
      continue;
    }

    const sourceTypeMatch = metadata.match(/source_type=([^\s]+)/);
    const sourceType = sourceTypeMatch?.[1] ?? null;

    entries.push({
      id,
      sourceType: sourceType as EvidenceLedgerSourceType | null,
      text: `${metadata}\n${body}`,
    });
  }

  return entries;
}

function resolveEvidenceId(input: {
  id: string;
  sourceType: EvidenceLedgerSourceType;
  prompt: string;
  placeholders: Record<string, EvidencePlaceholder>;
}): string {
  if (input.id === CURRENT_USER_EVIDENCE_ID) {
    const entry = extractLedgerEntries(input.prompt).find(
      (candidate) => candidate.sourceType === "current_user_message",
    );

    return entry?.id ?? input.id;
  }

  if (!input.id.startsWith("$evidence:")) {
    return input.id;
  }

  const key = input.id.slice("$evidence:".length);
  const placeholder = input.placeholders[key];

  if (placeholder === undefined) {
    return input.id;
  }

  const entry = extractLedgerEntries(input.prompt).find((candidate) => {
    const sourceType = placeholder.sourceType ?? input.sourceType;

    if (candidate.sourceType !== sourceType) {
      return false;
    }

    if (placeholder.state !== undefined && !candidate.text.includes(`state=${placeholder.state}`)) {
      return false;
    }

    if (
      placeholder.textIncludes !== undefined &&
      !placeholder.textIncludes.every((part) => candidate.text.includes(part))
    ) {
      return false;
    }

    if (
      placeholder.valueIncludes !== undefined &&
      !placeholder.valueIncludes.every((part) => candidate.text.includes(part))
    ) {
      return false;
    }

    return true;
  });

  return entry?.id ?? input.id;
}

function materializeEvidenceRef(
  ref: EvidenceRef,
  prompt: string,
  placeholders: Record<string, EvidencePlaceholder>,
): EvidenceRef {
  return {
    ...ref,
    id: resolveEvidenceId({
      id: ref.id,
      sourceType: ref.source_type,
      prompt,
      placeholders,
    }),
  };
}

export function materializeManifestResponse(
  manifest: EmitManifestResponse,
  options: LLMCompleteOptions | LLMConverseOptions,
  placeholders: Record<string, EvidencePlaceholder> = {},
): EmitManifestResponse {
  const prompt = llmPromptText(options);

  return {
    ...manifest,
    claims: manifest.claims.map((claim) => {
      if (!("evidence" in claim)) {
        return claim;
      }

      return {
        ...claim,
        evidence: claim.evidence.map((ref) => materializeEvidenceRef(ref, prompt, placeholders)),
      };
    }),
  };
}

export function lowerIncludesNone(text: string, values: readonly string[]): boolean {
  const lowered = text.toLowerCase();

  return values.every((value) => !lowered.includes(value.toLowerCase()));
}
