import type { StreamEntryPersistenceClass } from "../../stream/index.js";
import type { ClosurePressureHistoryReason } from "../../memory/working/index.js";
import type { StreamEntryId } from "../../util/ids.js";

export type GenerationSuppressionReason =
  | "generation_gate"
  | "active_discourse_stop"
  | "empty_finalizer"
  | "finalizer_failed"
  | "manifest_no_output"
  | "manifest_validation_failed_critical"
  | "no_output_tool"
  | "s2_planner_no_output"
  | "closure_pressure_only"
  | "closure_response_audit_failed_closed"
  | "commitment_revision_failed"
  | "rewrite_unsupported_or_empty"
  | "relational_guard_self_correction"
  | "relational_guard_audit_failed"
  | "relational_guard_rewrite_call_failed"
  | "relational_guard_rewrite_empty"
  | "relational_guard_reaudit_failed"
  | "relational_guard_rewrite_unsupported";

export const NATURAL_SILENCE_SUPPRESSION_REASONS = [
  "generation_gate",
  "active_discourse_stop",
  "empty_finalizer",
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

export type PendingTurnEmission =
  | {
      kind: "message";
      content: string;
      persistence_class?: StreamEntryPersistenceClass;
      closure_pressure_history_reason?: ClosurePressureHistoryReason;
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
      agentMessageId: StreamEntryId;
      persistence_class?: StreamEntryPersistenceClass;
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

export type EmissionRecommendation = "emit" | "no_output";
