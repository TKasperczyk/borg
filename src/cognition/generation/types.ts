import type { StreamEntryId } from "../../util/ids.js";

export type GenerationSuppressionReason =
  | "generation_gate"
  | "active_discourse_stop"
  | "empty_finalizer"
  | "no_output_tool"
  | "s2_planner_no_output"
  | "closure_pressure_only"
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
    }
  | {
      kind: "suppressed";
      reason: GenerationSuppressionReason;
      markerEntryId?: StreamEntryId;
    };

export type TurnEmission =
  | {
      kind: "message";
      content: string;
      agentMessageId: StreamEntryId;
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
