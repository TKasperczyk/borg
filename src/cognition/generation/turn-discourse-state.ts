import type { StreamWriter } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { StreamEntryId } from "../../util/ids.js";
import type { ClosurePressureHistoryReason, WorkingMemory } from "../../memory/working/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import {
  appendClosurePressureHistory,
  appendRecentSuppression,
  clearClosureLoop,
  clearStopUntilSubstantiveContent,
  markClosureLoopNamed,
  setClosureLoopDetected,
  setStopUntilSubstantiveContent,
} from "./discourse-state.js";
import {
  type AgentObservedStreamContent,
  isNaturalSilenceSuppressionReason,
  type AgentSuppressedStreamContent,
  type PendingTurnEmission,
} from "./types.js";

const ACTIVE_TURN_STATUS = "active";
const DISCOURSE_STATE_NAME = "stop_until_substantive_content";
const CLOSURE_LOOP_STATE_NAME = "closure_loop";
type SuppressionReason = Extract<PendingTurnEmission, { kind: "suppressed" }>["reason"];

function isRelationalGuardSuppressionReason(reason: SuppressionReason): boolean {
  return (
    reason === "relational_guard_self_correction" ||
    reason === "relational_guard_audit_failed" ||
    reason === "relational_guard_rewrite_call_failed" ||
    reason === "relational_guard_rewrite_empty" ||
    reason === "relational_guard_reaudit_failed" ||
    reason === "relational_guard_rewrite_unsupported"
  );
}

export type TurnDiscourseStateServiceOptions = {
  tracer: TurnTracer;
  clock?: Clock;
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

export type AppendObservationMarkerInput = {
  streamWriter: Pick<StreamWriter, "append">;
  reason: string;
  userEntryId?: AgentObservedStreamContent["user_entry_id"];
  turnId: string;
  audience?: string;
};

export class TurnDiscourseStateService {
  private readonly clock: Clock;

  constructor(private readonly options: TurnDiscourseStateServiceOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

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

  setClosureLoopDetected(input: {
    workingMemory: WorkingMemory;
    sourceStreamEntryIds: readonly StreamEntryId[];
    reason: string;
    turnId: string;
  }): WorkingMemory {
    const detected = setClosureLoopDetected(input.workingMemory, {
      sourceStreamEntryIds: input.sourceStreamEntryIds,
      reason: input.reason,
      sinceTurn: input.workingMemory.turn_counter,
    });
    const next = appendClosurePressureHistory(detected, {
      turnId: input.turnId,
      reason: "loop_detected",
      ts: this.clock.now(),
    });

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state_set", {
        turnId: input.turnId,
        state: CLOSURE_LOOP_STATE_NAME,
        provenance: "closure_loop_classifier",
        reason: input.reason,
        sourceStreamEntryIds: [...input.sourceStreamEntryIds],
      });
    }

    return next;
  }

  appendClosurePressureHistory(input: {
    workingMemory: WorkingMemory;
    turnId: string;
    reason: ClosurePressureHistoryReason;
  }): WorkingMemory {
    return appendClosurePressureHistory(input.workingMemory, {
      turnId: input.turnId,
      reason: input.reason,
      ts: this.clock.now(),
    });
  }

  markClosureLoopNamed(input: {
    workingMemory: WorkingMemory;
    reason: string;
    turnId: string;
    sourceStreamEntryId?: StreamEntryId;
  }): WorkingMemory {
    const next = markClosureLoopNamed(input.workingMemory, {
      sourceStreamEntryId: input.sourceStreamEntryId,
      reason: input.reason,
      turn: input.workingMemory.turn_counter,
    });

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state_set", {
        turnId: input.turnId,
        state: CLOSURE_LOOP_STATE_NAME,
        provenance: "closure_loop_named",
        reason: input.reason,
        ...(input.sourceStreamEntryId === undefined
          ? {}
          : { sourceStreamEntryId: input.sourceStreamEntryId }),
      });
    }

    return next;
  }

  clearClosureLoop(input: {
    workingMemory: WorkingMemory;
    reason: string;
    turnId: string;
  }): WorkingMemory {
    const active = input.workingMemory.discourse_state?.closure_loop ?? null;
    const next = clearClosureLoop(input.workingMemory);

    if (active !== null && this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state_cleared", {
        turnId: input.turnId,
        state: CLOSURE_LOOP_STATE_NAME,
        provenance: "closure_loop_classifier",
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

  appendObservationMarker(input: AppendObservationMarkerInput) {
    return input.streamWriter.append({
      kind: "agent_observed",
      turn_id: input.turnId,
      turn_status: ACTIVE_TURN_STATUS,
      content: {
        reason: input.reason,
        user_entry_id: input.userEntryId,
        turn_id: input.turnId,
      } satisfies AgentObservedStreamContent,
      ...(input.audience === undefined ? {} : { audience: input.audience }),
    });
  }

  applySuppressedEmissionState(input: {
    workingMemory: WorkingMemory;
    reason: SuppressionReason;
    sourceStreamEntryId: StreamEntryId;
    turnId: string;
  }): WorkingMemory {
    let workingMemory = appendRecentSuppression(input.workingMemory, {
      turnId: input.turnId,
      reason: input.reason,
      sourceStreamEntryId: input.sourceStreamEntryId,
      ts: this.clock.now(),
    });

    if (input.reason === "closure_pressure_only") {
      workingMemory = appendClosurePressureHistory(workingMemory, {
        turnId: input.turnId,
        reason: "span_removed",
        ts: this.clock.now(),
      });
    } else if (input.reason === "closure_response_audit_failed_closed") {
      workingMemory = appendClosurePressureHistory(workingMemory, {
        turnId: input.turnId,
        reason: "audit_caught",
        ts: this.clock.now(),
      });
    }

    if (input.reason === "no_output_tool" || input.reason === "manifest_no_output") {
      workingMemory = this.setStopState({
        workingMemory,
        provenance: input.reason,
        sourceStreamEntryId: input.sourceStreamEntryId,
        reason:
          input.reason === "manifest_no_output"
            ? "Legacy finalizer emitted no_output for this turn."
            : "Finalizer called no_output for this turn.",
        turnId: input.turnId,
      });
    }

    if (
      input.reason === "commitment_revision_failed" ||
      input.reason === "rewrite_unsupported_or_empty"
    ) {
      return this.setStopState({
        workingMemory,
        provenance: "commitment_guard",
        sourceStreamEntryId: input.sourceStreamEntryId,
        reason:
          input.reason === "commitment_revision_failed"
            ? "Commitment guard suppressed this turn because revision still violated an active commitment."
            : "Commitment guard suppressed this turn because rewrite produced no supported output.",
        turnId: input.turnId,
      });
    }

    if (isRelationalGuardSuppressionReason(input.reason)) {
      return this.setStopState({
        workingMemory,
        provenance: "relational_guard",
        sourceStreamEntryId: input.sourceStreamEntryId,
        reason:
          input.reason === "relational_guard_self_correction"
            ? "Relational guard suppressed this turn because the response contained an unsupported self-correction claim."
            : "Relational guard suppressed this turn because it could not produce a supported relational response.",
        turnId: input.turnId,
      });
    }

    if (
      isNaturalSilenceSuppressionReason(input.reason) &&
      (workingMemory.discourse_state?.closure_loop ?? null)?.status === "detected"
    ) {
      return this.markClosureLoopNamed({
        workingMemory,
        sourceStreamEntryId: input.sourceStreamEntryId,
        reason:
          input.reason === "no_output_tool"
            ? "Closure loop detected; finalizer chose no_output."
            : input.reason === "manifest_no_output"
              ? "Closure loop detected; legacy finalizer chose no_output."
              : `Closure loop detected; turn ended without assistant output (${input.reason}).`,
        turnId: input.turnId,
      });
    }

    return workingMemory;
  }
}
