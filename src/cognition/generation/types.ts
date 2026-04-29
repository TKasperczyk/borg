import type { StreamEntryId } from "../../util/ids.js";

export type GenerationSuppressionReason =
  | "generation_gate"
  | "active_discourse_stop"
  | "output_validator"
  | "empty_finalizer"
  | "invalid_non_generation_text"
  | "s2_planner_no_output";

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
