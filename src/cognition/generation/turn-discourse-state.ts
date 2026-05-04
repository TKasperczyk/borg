import type { StreamWriter } from "../../stream/index.js";
import type { StreamEntryId } from "../../util/ids.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import {
  clearStopUntilSubstantiveContent,
  setStopUntilSubstantiveContent,
} from "./discourse-state.js";
import type { AgentSuppressedStreamContent, PendingTurnEmission } from "./types.js";

const ACTIVE_TURN_STATUS = "active";
const DISCOURSE_STATE_NAME = "stop_until_substantive_content";
type SuppressionReason = Extract<PendingTurnEmission, { kind: "suppressed" }>["reason"];
const RELATIONAL_GUARD_SUPPRESSION_REASONS: ReadonlySet<SuppressionReason> = new Set([
  "relational_guard_self_correction",
  "relational_guard_audit_failed",
  "relational_guard_rewrite_call_failed",
  "relational_guard_rewrite_empty",
  "relational_guard_reaudit_failed",
  "relational_guard_rewrite_unsupported",
] satisfies SuppressionReason[]);

export type TurnDiscourseStateServiceOptions = {
  tracer: TurnTracer;
};

export type SetTurnDiscourseStopStateInput = {
  workingMemory: WorkingMemory;
  provenance: Parameters<typeof setStopUntilSubstantiveContent>[1]["provenance"];
  sourceStreamEntryId?: StreamEntryId;
  reason: string;
  turnId: string;
};

export type AppendSuppressionMarkerInput = {
  streamWriter: Pick<StreamWriter, "append">;
  reason: SuppressionReason;
  userEntryId?: AgentSuppressedStreamContent["user_entry_id"];
  turnId: string;
  audience?: string;
};

export class TurnDiscourseStateService {
  constructor(private readonly options: TurnDiscourseStateServiceOptions) {}

  setStopState(input: SetTurnDiscourseStopStateInput): WorkingMemory {
    const next = setStopUntilSubstantiveContent(input.workingMemory, {
      provenance: input.provenance,
      sourceStreamEntryId: input.sourceStreamEntryId,
      reason: input.reason,
      sinceTurn: input.workingMemory.turn_counter,
    });

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state_set", {
        turnId: input.turnId,
        state: DISCOURSE_STATE_NAME,
        provenance: input.provenance,
        reason: input.reason,
        ...(input.sourceStreamEntryId === undefined
          ? {}
          : { sourceStreamEntryId: input.sourceStreamEntryId }),
      });
    }

    return next;
  }

  clearStopState(input: {
    workingMemory: WorkingMemory;
    reason: string;
    turnId: string;
  }): WorkingMemory {
    const active = input.workingMemory.discourse_state?.stop_until_substantive_content ?? null;
    const next = clearStopUntilSubstantiveContent(input.workingMemory);

    if (active !== null && this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state_cleared", {
        turnId: input.turnId,
        state: DISCOURSE_STATE_NAME,
        provenance: active.provenance,
        reason: input.reason,
      });
    }

    return next;
  }

  async appendHardCapEvent(input: {
    streamWriter: Pick<StreamWriter, "append">;
    turnId: string;
    activeTurns: number;
    hardCapTurns: number;
    stateReason: string;
  }): Promise<void> {
    if (this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state_hard_cap", {
        turnId: input.turnId,
        state: DISCOURSE_STATE_NAME,
        activeTurns: input.activeTurns,
        hardCapTurns: input.hardCapTurns,
      });
    }

    try {
      await input.streamWriter.append({
        kind: "internal_event",
        content: {
          hook: "discourse_state_hard_cap",
          turn_id: input.turnId,
          active_turns: input.activeTurns,
          hard_cap_turns: input.hardCapTurns,
          state_reason: input.stateReason,
        },
      });
    } catch {
      // Best-effort telemetry only.
    }
  }

  appendSuppressionMarker(input: AppendSuppressionMarkerInput) {
    return input.streamWriter.append({
      kind: "agent_suppressed",
      turn_id: input.turnId,
      turn_status: ACTIVE_TURN_STATUS,
      content: {
        reason: input.reason,
        user_entry_id: input.userEntryId,
        turn_id: input.turnId,
      } satisfies AgentSuppressedStreamContent,
      ...(input.audience === undefined ? {} : { audience: input.audience }),
    });
  }

  applySuppressedEmissionState(input: {
    workingMemory: WorkingMemory;
    reason: SuppressionReason;
    sourceStreamEntryId: StreamEntryId;
    turnId: string;
  }): WorkingMemory {
    if (input.reason === "no_output_tool") {
      return this.setStopState({
        workingMemory: input.workingMemory,
        provenance: "no_output_tool",
        sourceStreamEntryId: input.sourceStreamEntryId,
        reason: "Finalizer called no_output for this turn.",
        turnId: input.turnId,
      });
    }

    if (
      input.reason === "commitment_revision_failed" ||
      input.reason === "rewrite_unsupported_or_empty"
    ) {
      return this.setStopState({
        workingMemory: input.workingMemory,
        provenance: "commitment_guard",
        sourceStreamEntryId: input.sourceStreamEntryId,
        reason:
          input.reason === "commitment_revision_failed"
            ? "Commitment guard suppressed this turn because revision still violated an active commitment."
            : "Commitment guard suppressed this turn because rewrite produced no supported output.",
        turnId: input.turnId,
      });
    }

    if (RELATIONAL_GUARD_SUPPRESSION_REASONS.has(input.reason)) {
      return this.setStopState({
        workingMemory: input.workingMemory,
        provenance: "relational_guard",
        sourceStreamEntryId: input.sourceStreamEntryId,
        reason:
          input.reason === "relational_guard_self_correction"
            ? "Relational guard suppressed this turn because the response contained an unsupported self-correction claim."
            : "Relational guard suppressed this turn because it could not produce a supported relational response.",
        turnId: input.turnId,
      });
    }

    return input.workingMemory;
  }
}
