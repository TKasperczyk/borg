import type {
  AgentSuppressedStreamContent,
  FinalizerInvalidToolDiagnostic,
  FinalizerNoOutputCategory,
  FinalizerNoOutputPrimaryReason,
  FinalizerNoOutputStructuralFlag,
  GenerationSuppressionReason,
} from "./generation/types.js";
import type {
  RecentRegenerationCommitment,
  RecentRegenerationEntry,
  RecentSuppressionEntry,
  WorkingMemory,
} from "../memory/working/index.js";
import type {
  AutonomySchedulerBudgetDescription,
  AutonomySchedulerFleetBrakeDescription,
} from "../autonomy/index.js";
import {
  RECENT_REGENERATIONS_LIMIT,
  RECENT_SUPPRESSIONS_LIMIT,
} from "./generation/discourse-state.js";
import {
  hydrateStreamEntriesById,
  type StreamEntry,
  type StreamEntryIndexRepository,
  type StreamReader,
} from "../stream/index.js";
import type { SessionId, StreamEntryId } from "../util/ids.js";

export type HydratedSuppressionDiagnostic = {
  noOutputCategories?: readonly FinalizerNoOutputCategory[];
  primaryNoOutputReason?: FinalizerNoOutputPrimaryReason;
  structuralNoOutputFlags?: readonly FinalizerNoOutputStructuralFlag[];
  finalizerInvalidTool?: FinalizerInvalidToolDiagnostic;
};

export type HydratedRecentSuppression = {
  turnId: string;
  reason: GenerationSuppressionReason | string;
  ts: number;
  sourceStreamEntryId?: StreamEntryId;
  diagnostic?: HydratedSuppressionDiagnostic;
};

export type HydratedRecentRegeneration = {
  turnId: string;
  mechanism: RecentRegenerationEntry["mechanism"];
  ts: number;
  sourceStreamEntryId?: StreamEntryId;
  commitments?: readonly RecentRegenerationCommitment[];
};

// The scheduler's `describe()` builds six top-level fields and this object used to keep one.
// `budget` alone answers "is there room for another wake", but three of the discarded fields
// refuse a wake independently of room: `enabled` (loop off -> nothing fires), `next_tick_at`
// (null -> no interval handle, so nothing fires), and `fleet_brake` (empty-streak cooldown or
// error-streak pause -- a second gate the budget line knows nothing about). Rendering only the
// budget therefore produced a page on which "budget has headroom" and "a wake can happen" looked
// like the same statement. The discriminators existed upstream and were dropped at the provider
// call site; they are carried here so the surface can say which gate is holding.
//
// `scheduledTickAt` is carried alongside `nextTickAt` for the same reason. The scheduler floors
// `next_tick_at` forward to the read clock so a UI never shows a "next evaluation" that has
// already been and gone -- but the floor subtracts, and what it subtracts is exactly how overdue
// the tick was. Rendering only the floored stamp produced a line that reports lateness by
// refusing to be late: a tick 12s behind and a tick due this instant print the same instant, and
// the age hung on that stamp is time since the read, not tick lateness. The unfloored value is
// two lines away in the scheduler; carrying it here is what makes the discarded quantity
// recoverable at the surface instead of destroyed upstream of it.
//
// `tickInFlight` is the third of these, and the one the stamps could not supply. `enabled` is the
// config flag; `scheduledTickAt` says how far behind the loop is; neither says *why*, and the two
// causes take opposite repairs. A tick stamps the anchor on entry and the interval drops every
// fire while it runs, so a stuck tick holds one stamp while the overdue amount climbs -- the same
// page a lagging interval draws. Telling them apart used to require two reads far enough apart to
// watch whether the stamp moved, which is a discriminator no single prompt can carry. The flag is
// already in the scheduler at describe() time; carrying it here makes one read enough.
export type AutonomySchedulerMechanismEvidence = {
  observedAt: number;
  enabled: boolean;
  tickInFlight: boolean;
  nextTickAt: number | null;
  scheduledTickAt: number | null;
  budget: AutonomySchedulerBudgetDescription;
  fleetBrake: AutonomySchedulerFleetBrakeDescription;
};

export type TurnMechanismEvidence = {
  recentSuppressions: readonly HydratedRecentSuppression[];
  recentRegenerations: readonly HydratedRecentRegeneration[];
  autonomySchedulerState?: AutonomySchedulerMechanismEvidence;
};

export type HydrateTurnMechanismEvidenceInput = {
  dataDir: string;
  sessionId: SessionId;
  workingMemory: WorkingMemory;
  autonomySchedulerState?: AutonomySchedulerMechanismEvidence;
  entryIndex?: Pick<StreamEntryIndexRepository, "lookupMany">;
  createStreamReader: (sessionId: SessionId) => StreamReader;
};

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string")
    ? [...value]
    : undefined;
}

function finalizerInvalidToolDiagnostic(
  value: unknown,
): FinalizerInvalidToolDiagnostic | undefined {
  if (!isObjectRecord(value)) {
    return undefined;
  }

  return typeof value.tool_name === "string" &&
    typeof value.reason === "string" &&
    (value.attempt === "initial" || value.attempt === "regenerate")
    ? {
        tool_name: value.tool_name,
        reason: value.reason,
        attempt: value.attempt,
      }
    : undefined;
}

function suppressionDiagnosticFromEntry(
  entry: StreamEntry | undefined,
): HydratedSuppressionDiagnostic {
  if (entry?.kind !== "agent_suppressed" || !isObjectRecord(entry.content)) {
    return {};
  }

  const content = entry.content as Partial<AgentSuppressedStreamContent>;
  const noOutputCategories = stringArray(content.no_output_categories);
  const structuralNoOutputFlags = stringArray(content.structural_no_output_flags);
  const primaryNoOutputReason =
    typeof content.primary_no_output_reason === "string"
      ? content.primary_no_output_reason
      : undefined;
  const finalizerInvalidTool = finalizerInvalidToolDiagnostic(content.finalizer_invalid_tool);

  return {
    ...(noOutputCategories === undefined
      ? {}
      : { noOutputCategories: noOutputCategories as FinalizerNoOutputCategory[] }),
    ...(primaryNoOutputReason === undefined
      ? {}
      : { primaryNoOutputReason: primaryNoOutputReason as FinalizerNoOutputPrimaryReason }),
    ...(structuralNoOutputFlags === undefined
      ? {}
      : { structuralNoOutputFlags: structuralNoOutputFlags as FinalizerNoOutputStructuralFlag[] }),
    ...(finalizerInvalidTool === undefined ? {} : { finalizerInvalidTool }),
  };
}

function hydratedRecentRegeneration(entry: RecentRegenerationEntry): HydratedRecentRegeneration {
  return {
    turnId: entry.turn_id,
    mechanism: entry.mechanism,
    ts: entry.ts,
    ...(entry.source_stream_entry_id === undefined
      ? {}
      : { sourceStreamEntryId: entry.source_stream_entry_id }),
    // Absent and empty are two different silences on the stored entry (see
    // appendRecentRegeneration); carry both through so the render can say which.
    ...(entry.commitments === undefined ? {} : { commitments: entry.commitments }),
  };
}

// Scope of "recent silences from my side" (system-prompt.ts renders this list): it is a
// POST-GENERATION register, not a register of turns that produced nothing. Its only writer is
// `discourseStateService.appendSuppressionMarker`, called exclusively from post-generation-phase --
// so an entry exists only for a turn that generated a candidate and then had it suppressed by a
// guard. A turn that died before or during generation (provider error, superseding inbound) never
// reaches that call site and is therefore absent here by construction, not by filtering: nothing
// removes aborts from this list, they were never added. Measured on the live demo store
// (2026-08-20): 2165 stream-index rows carry turn_status='aborted', every one active=0; 1463
// `agent_suppressed` rows, every one active=1 -- the two classes do not overlap and only the
// second one can ever appear below. See the comment on `isAbortedTurnMarker` in
// src/stream/turn-status.ts for the independent second reason the abort's `reason` string is
// unreadable. Consequence worth stating plainly before anyone reasons from this block: a run of
// aborted turns renders here as an unbroken record of turns that spoke.
//
// Second property, independent of the first: neither list is a time window. Both are count-capped
// rings (capNewest at RECENT_*_LIMIT in generation/discourse-state.ts), and working memory is
// per-session (working/<session_id>.json), so an entry survives in its own session until that many
// newer ones displace it -- days or weeks. Two entries side by side carry no implication of being
// contemporaries, and neither list can be compared against a time-bounded census of the other.
// Measured on the live demo store (2026-08-23), one session's suppression ring held 8 entries
// spanning 11 days, its head a reason code from a guard that had been configured off for that
// session's source type two hours after it fired: still rendered, no longer possible. That is why
// system-prompt.ts renders each entry's age -- the ts is here, and dropping it at render was the
// whole gap.
export async function hydrateTurnMechanismEvidence(
  input: HydrateTurnMechanismEvidenceInput,
): Promise<TurnMechanismEvidence> {
  const recentSuppressions = (input.workingMemory.discourse_state?.recent_suppressions ?? []).slice(
    -RECENT_SUPPRESSIONS_LIMIT,
  );
  const recentRegenerations = (
    input.workingMemory.discourse_state?.recent_regenerations ?? []
  ).slice(-RECENT_REGENERATIONS_LIMIT);
  const sourceStreamEntryIds = recentSuppressions.flatMap((entry) =>
    entry.source_stream_entry_id === undefined ? [] : [entry.source_stream_entry_id],
  );
  const entriesById = await hydrateStreamEntriesById({
    dataDir: input.dataDir,
    sessionId: input.sessionId,
    streamEntryIds: sourceStreamEntryIds,
    entryIndex: input.entryIndex,
    createStreamReader: input.createStreamReader,
  });

  return {
    recentSuppressions: recentSuppressions.map((entry: RecentSuppressionEntry) => ({
      turnId: entry.turn_id,
      reason: entry.reason,
      ts: entry.ts,
      ...(entry.source_stream_entry_id === undefined
        ? {}
        : {
            sourceStreamEntryId: entry.source_stream_entry_id,
            diagnostic: suppressionDiagnosticFromEntry(
              entriesById.get(entry.source_stream_entry_id),
            ),
          }),
    })),
    recentRegenerations: recentRegenerations.map(hydratedRecentRegeneration),
    ...(input.autonomySchedulerState === undefined
      ? {}
      : { autonomySchedulerState: input.autonomySchedulerState }),
  };
}
