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
    };

export type AgentSuppressedStreamContent = {
  reason: GenerationSuppressionReason;
  user_entry_id?: StreamEntryId;
  turn_id?: string;
};

export type AgentObservedStreamContent = {
  reason: string;
  user_entry_id?: StreamEntryId;
  turn_id?: string;
};

export type EmissionRecommendation = "emit" | "no_output";
