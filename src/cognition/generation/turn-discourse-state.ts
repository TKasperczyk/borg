import type { StreamResponseTo, StreamWriter } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
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
  type FinalizerNoOutputCategory,
  type FinalizerNoOutputPrimaryReason,
  type FinalizerNoOutputStructuralFlag,
  isNaturalSilenceSuppressionReason,
  type AgentSuppressedStreamContent,
  type PendingTurnEmission,
} from "./types.js";

const ACTIVE_TURN_STATUS = "active";
const DISCOURSE_STATE_NAME = "stop_until_substantive_content";
const CLOSURE_LOOP_STATE_NAME = "closure_loop";
type SuppressionReason = Extract<PendingTurnEmission, { kind: "suppressed" }>["reason"];

function isFinalizerNoOutputSuppressionReason(
  reason: SuppressionReason,
): reason is Extract<
  SuppressionReason,
  "finalizer_no_output" | "manifest_no_output" | "no_output_tool"
> {
  return (
    reason === "finalizer_no_output" ||
    reason === "manifest_no_output" ||
    reason === "no_output_tool"
  );
}

function canonicalFinalizerNoOutputSuppressionReason(
  reason: SuppressionReason,
): "finalizer_no_output" | null {
  return isFinalizerNoOutputSuppressionReason(reason) ? "finalizer_no_output" : null;
}

export type TurnDiscourseStateServiceOptions = {
  tracer: TurnTracer;
  clock?: Clock;
};

export type SetTurnDiscourseStopStateInput = {
  workingMemory: WorkingMemory;
  provenance: Parameters<typeof setStopUntilSubstantiveContent>[1]["provenance"];
  sourceStreamEntryId?: StreamEntryId;
  sourceStreamEntryIds?: readonly StreamEntryId[];
  reason: string;
  turnId: string;
  sessionId?: SessionId;
};

export type AppendSuppressionMarkerInput = {
  streamWriter: Pick<StreamWriter, "append">;
  reason: SuppressionReason;
  userEntryId?: AgentSuppressedStreamContent["user_entry_id"];
  userEntryIds?: readonly StreamEntryId[];
  responseTo?: StreamResponseTo;
  turnId: string;
  audience?: string;
  noOutputCategories?: readonly FinalizerNoOutputCategory[];
  primaryNoOutputReason?: FinalizerNoOutputPrimaryReason;
  structuralNoOutputFlags?: readonly FinalizerNoOutputStructuralFlag[];
};

export type AppendObservationMarkerInput = {
  streamWriter: Pick<StreamWriter, "append">;
  reason: string;
  userEntryId?: AgentObservedStreamContent["user_entry_id"];
  userEntryIds?: readonly StreamEntryId[];
  responseTo?: StreamResponseTo;
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
      sourceStreamEntryIds: input.sourceStreamEntryIds,
      reason: input.reason,
      sinceTurn: input.workingMemory.turn_counter,
    });

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state.transitioned", {
        turnId: input.turnId,
        ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
        state: DISCOURSE_STATE_NAME,
        transition: "set",
        provenance: input.provenance,
        reason: input.reason,
        ...(input.sourceStreamEntryId === undefined
          ? {}
          : { sourceStreamEntryId: input.sourceStreamEntryId }),
        ...(input.sourceStreamEntryIds === undefined || input.sourceStreamEntryIds.length === 0
          ? {}
          : { sourceStreamEntryIds: [...input.sourceStreamEntryIds] }),
      });
    }

    return next;
  }

  clearStopState(input: {
    workingMemory: WorkingMemory;
    reason: string;
    turnId: string;
    sessionId?: SessionId;
  }): WorkingMemory {
    const active = input.workingMemory.discourse_state?.stop_until_substantive_content ?? null;
    const next = clearStopUntilSubstantiveContent(input.workingMemory);

    if (active !== null && this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state.transitioned", {
        turnId: input.turnId,
        ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
        state: DISCOURSE_STATE_NAME,
        transition: "cleared",
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
    sessionId?: SessionId;
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
      this.options.tracer.emit("discourse_state.transitioned", {
        turnId: input.turnId,
        ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
        state: CLOSURE_LOOP_STATE_NAME,
        transition: "detected",
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
    sourceStreamEntryIds?: readonly StreamEntryId[];
    sessionId?: SessionId;
  }): WorkingMemory {
    const next = markClosureLoopNamed(input.workingMemory, {
      sourceStreamEntryId: input.sourceStreamEntryId,
      sourceStreamEntryIds: input.sourceStreamEntryIds,
      reason: input.reason,
      turn: input.workingMemory.turn_counter,
    });

    if (this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state.transitioned", {
        turnId: input.turnId,
        ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
        state: CLOSURE_LOOP_STATE_NAME,
        transition: "named",
        provenance: "closure_loop_named",
        reason: input.reason,
        ...(input.sourceStreamEntryId === undefined
          ? {}
          : { sourceStreamEntryId: input.sourceStreamEntryId }),
        ...(input.sourceStreamEntryIds === undefined || input.sourceStreamEntryIds.length === 0
          ? {}
          : { sourceStreamEntryIds: [...input.sourceStreamEntryIds] }),
      });
    }

    return next;
  }

  clearClosureLoop(input: {
    workingMemory: WorkingMemory;
    reason: string;
    turnId: string;
    sessionId?: SessionId;
  }): WorkingMemory {
    const active = input.workingMemory.discourse_state?.closure_loop ?? null;
    const next = clearClosureLoop(input.workingMemory);

    if (active !== null && this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state.transitioned", {
        turnId: input.turnId,
        ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
        state: CLOSURE_LOOP_STATE_NAME,
        transition: "cleared",
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
    sessionId?: SessionId;
  }): Promise<void> {
    if (this.options.tracer.enabled) {
      this.options.tracer.emit("discourse_state.rejected", {
        turnId: input.turnId,
        ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
        state: DISCOURSE_STATE_NAME,
        activeTurns: input.activeTurns,
        hardCapTurns: input.hardCapTurns,
      });
    }

    try {
      await input.streamWriter.append({
        kind: "internal_event",
        content: {
          hook: "discourse_state.rejected",
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
        turn_id: input.turnId,
        ...(input.userEntryId === undefined ? {} : { user_entry_id: input.userEntryId }),
        ...(input.userEntryIds === undefined || input.userEntryIds.length === 0
          ? {}
          : { user_entry_ids: [...input.userEntryIds] }),
        ...(input.noOutputCategories === undefined
          ? {}
          : { no_output_categories: [...input.noOutputCategories] }),
        ...(input.primaryNoOutputReason === undefined
          ? {}
          : { primary_no_output_reason: input.primaryNoOutputReason }),
        ...(input.structuralNoOutputFlags === undefined
          ? {}
          : { structural_no_output_flags: [...input.structuralNoOutputFlags] }),
      } satisfies AgentSuppressedStreamContent,
      ...(input.responseTo === undefined ? {} : { response_to: input.responseTo }),
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
        turn_id: input.turnId,
        ...(input.userEntryId === undefined ? {} : { user_entry_id: input.userEntryId }),
        ...(input.userEntryIds === undefined || input.userEntryIds.length === 0
          ? {}
          : { user_entry_ids: [...input.userEntryIds] }),
      } satisfies AgentObservedStreamContent,
      ...(input.responseTo === undefined ? {} : { response_to: input.responseTo }),
      ...(input.audience === undefined ? {} : { audience: input.audience }),
    });
  }

  applySuppressedEmissionState(input: {
    workingMemory: WorkingMemory;
    reason: SuppressionReason;
    sourceStreamEntryId: StreamEntryId;
    sourceStreamEntryIds?: readonly StreamEntryId[];
    turnId: string;
    sessionId?: SessionId;
  }): WorkingMemory {
    let workingMemory = appendRecentSuppression(input.workingMemory, {
      turnId: input.turnId,
      reason: input.reason,
      sourceStreamEntryId: input.sourceStreamEntryId,
      sourceStreamEntryIds: input.sourceStreamEntryIds,
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

    const finalizerNoOutputReason = canonicalFinalizerNoOutputSuppressionReason(input.reason);

    if (finalizerNoOutputReason !== null) {
      workingMemory = this.setStopState({
        workingMemory,
        provenance: finalizerNoOutputReason,
        sourceStreamEntryId: input.sourceStreamEntryId,
        sourceStreamEntryIds: input.sourceStreamEntryIds,
        reason:
          input.reason === "manifest_no_output"
            ? "Legacy finalizer emitted no_output for this turn."
            : "Finalizer called no_output for this turn.",
        turnId: input.turnId,
        sessionId: input.sessionId,
      });
    }

    if (
      input.reason === "commitment_violation" ||
      input.reason === "commitment_violation_after_regenerate" ||
      input.reason === "commitment_revision_failed" ||
      input.reason === "rewrite_unsupported_or_empty"
    ) {
      return this.setStopState({
        workingMemory,
        provenance: "commitment_guard",
        sourceStreamEntryId: input.sourceStreamEntryId,
        sourceStreamEntryIds: input.sourceStreamEntryIds,
        reason:
          input.reason === "commitment_violation"
            ? "Commitment guard suppressed this turn because output violated an enforceable commitment."
            : input.reason === "commitment_violation_after_regenerate"
              ? "Commitment guard suppressed this turn because regenerated output still violated an enforceable commitment."
              : input.reason === "commitment_revision_failed"
                ? "Commitment guard suppressed this turn because revision still violated an active commitment."
                : "Commitment guard suppressed this turn because rewrite produced no supported output.",
        turnId: input.turnId,
        sessionId: input.sessionId,
      });
    }

    if (
      isNaturalSilenceSuppressionReason(input.reason) &&
      (workingMemory.discourse_state?.closure_loop ?? null)?.status === "detected"
    ) {
      return this.markClosureLoopNamed({
        workingMemory,
        sourceStreamEntryId: input.sourceStreamEntryId,
        sourceStreamEntryIds: input.sourceStreamEntryIds,
        reason:
          input.reason === "no_output_tool" || input.reason === "finalizer_no_output"
            ? "Closure loop detected; finalizer chose no_output."
            : input.reason === "manifest_no_output"
              ? "Closure loop detected; legacy finalizer chose no_output."
              : `Closure loop detected; turn ended without assistant output (${input.reason}).`,
        turnId: input.turnId,
        sessionId: input.sessionId,
      });
    }

    return workingMemory;
  }
}
