import type { StreamEntryPersistenceClass } from "../../stream/index.js";
import type { ClosurePressureHistoryReason } from "../../memory/working/index.js";
import { entityIdHelpers, type EntityId, type StreamEntryId } from "../../util/ids.js";
import { z } from "zod";

export type GenerationSuppressionReason =
  | "generation_gate"
  | "active_discourse_stop"
  | "empty_finalizer"
  | "finalizer_failed"
  | "finalizer_no_output"
  | "manifest_no_output"
  | "legacy_manifest_validation_failed_critical"
  | "manifest_validation_failed_critical"
  | "no_output_tool"
  | "s2_planner_no_output"
  | "closure_pressure_only"
  | "closure_response_audit_failed_closed"
  | "commitment_violation"
  | "commitment_violation_after_regenerate"
  | "commitment_revision_failed"
  | "internal_identifier_leak"
  | "rewrite_unsupported_or_empty";

export const FINALIZER_NO_OUTPUT_SEMANTIC_CATEGORIES = [
  "user_to_user",
  "when_borg_addressed",
  "closure",
] as const;

export const FINALIZER_NO_OUTPUT_PRIMARY_REASONS = [
  "closure",
  "user_to_user",
  "when_borg_addressed",
  "low_value_echo",
  "other",
] as const;

export const FINALIZER_NO_OUTPUT_STRUCTURAL_CATEGORIES = [
  "with_state_delta",
  "with_open_question",
] as const;

export const FINALIZER_NO_OUTPUT_STRUCTURAL_FLAGS = [
  "with_state_delta",
  "current_turn_state_delta",
  "with_open_question",
  "open_question_rendered",
  "borg_directly_addressed",
] as const;

export const FINALIZER_NO_OUTPUT_CATEGORIES = [
  ...FINALIZER_NO_OUTPUT_SEMANTIC_CATEGORIES,
  ...FINALIZER_NO_OUTPUT_STRUCTURAL_CATEGORIES,
] as const;

export type FinalizerNoOutputPrimaryReason = (typeof FINALIZER_NO_OUTPUT_PRIMARY_REASONS)[number];
export type FinalizerNoOutputSemanticCategory =
  (typeof FINALIZER_NO_OUTPUT_SEMANTIC_CATEGORIES)[number];
export type FinalizerNoOutputStructuralCategory =
  (typeof FINALIZER_NO_OUTPUT_STRUCTURAL_CATEGORIES)[number];
export type FinalizerNoOutputStructuralFlag = (typeof FINALIZER_NO_OUTPUT_STRUCTURAL_FLAGS)[number];
export type FinalizerNoOutputCategory = (typeof FINALIZER_NO_OUTPUT_CATEGORIES)[number];

export const NATURAL_SILENCE_SUPPRESSION_REASONS = [
  "generation_gate",
  "active_discourse_stop",
  "empty_finalizer",
  "finalizer_no_output",
  "manifest_no_output",
  "no_output_tool",
  "s2_planner_no_output",
  "closure_pressure_only",
] as const satisfies readonly GenerationSuppressionReason[];

const NATURAL_SILENCE_SUPPRESSION_REASON_SET: ReadonlySet<GenerationSuppressionReason> = new Set(
  NATURAL_SILENCE_SUPPRESSION_REASONS,
);

export function isNaturalSilenceSuppressionReason(reason: GenerationSuppressionReason): boolean {
  return NATURAL_SILENCE_SUPPRESSION_REASON_SET.has(reason);
}

export function deriveFinalizerNoOutputPrimaryReason(
  semanticCategories: readonly FinalizerNoOutputSemanticCategory[],
): FinalizerNoOutputPrimaryReason {
  if (semanticCategories.includes("when_borg_addressed")) {
    return "when_borg_addressed";
  }

  if (semanticCategories.includes("user_to_user")) {
    return "user_to_user";
  }

  if (semanticCategories.includes("closure")) {
    return "closure";
  }

  return "other";
}

const replyTargetEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid reply target entity id",
  })
  .transform((value) => value as EntityId);

export const replyTargetSchema = z.discriminatedUnion("kind", [
  z
    .object({
      kind: z.literal("audience"),
    })
    .strict(),
  z
    .object({
      kind: z.literal("entity"),
      entity_id: replyTargetEntityIdSchema,
    })
    .strict(),
]);

export type ReplyTarget = z.infer<typeof replyTargetSchema>;

export function replyTargetEntityId(replyTarget: ReplyTarget | undefined): EntityId | null {
  return replyTarget?.kind === "entity" ? replyTarget.entity_id : null;
}

export type PendingTurnEmission =
  | {
      kind: "message";
      content: string;
      reply_target?: ReplyTarget;
      persistence_class?: StreamEntryPersistenceClass;
      closure_pressure_history_reason?: ClosurePressureHistoryReason;
    }
  | {
      kind: "observed";
      reason: string;
      markerEntryId?: StreamEntryId;
    }
  | {
      kind: "suppressed";
      reason: GenerationSuppressionReason;
      markerEntryId?: StreamEntryId;
      closure_pressure_history_reason?: ClosurePressureHistoryReason;
      no_output_categories?: FinalizerNoOutputCategory[];
      primary_no_output_reason?: FinalizerNoOutputPrimaryReason;
      structural_no_output_flags?: FinalizerNoOutputStructuralFlag[];
    };

export type TurnEmission =
  | {
      kind: "message";
      content: string;
      reply_target?: ReplyTarget;
      agentMessageId: StreamEntryId;
      persistence_class?: StreamEntryPersistenceClass;
    }
  | {
      kind: "observed";
      reason: string;
      markerEntryId?: StreamEntryId;
    }
  | {
      kind: "suppressed";
      reason: GenerationSuppressionReason;
      markerEntryId?: StreamEntryId;
      no_output_categories?: FinalizerNoOutputCategory[];
      primary_no_output_reason?: FinalizerNoOutputPrimaryReason;
      structural_no_output_flags?: FinalizerNoOutputStructuralFlag[];
    };

export type AgentSuppressedStreamContent = {
  reason: GenerationSuppressionReason;
  user_entry_id?: StreamEntryId;
  turn_id?: string;
  no_output_categories?: FinalizerNoOutputCategory[];
  primary_no_output_reason?: FinalizerNoOutputPrimaryReason;
  structural_no_output_flags?: FinalizerNoOutputStructuralFlag[];
};

export type AgentObservedStreamContent = {
  reason: string;
  user_entry_id?: StreamEntryId;
  turn_id?: string;
};

export type EmissionRecommendation = "emit" | "no_output";
