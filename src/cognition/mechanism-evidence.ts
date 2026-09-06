import type { OperatorAttentionIndex } from "../memory/operator-attention/types.js";
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
  AutonomySchedulerDescription,
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
//
// `sources` is the sixth, and it is the only prospective field the scheduler produces. Everything
// else on this object is retrospective or about the loop itself: what has already fired, what the
// window has already spent, how far behind the tick is. A reader holding only those can describe
// the scheduler's past and its health and can state nothing falsifiable about its next move -- the
// wakes-by-trigger groups are the same source names counted backwards, which is why carrying this
// is not a duplicate of them. The per-trigger due stamps are the one field on which a prediction
// made before a wake fires can turn out wrong, and being wrong is the property the rest of the
// block cannot offer at any number of reads.
//
// `intervalMs` and `droppedIntervalFires` are the fourth and fifth, and both were dropped at the
// same call site as the three above. The interval length is what turns a series of
// `scheduledTickAt` stamps into a grid: the stamp is the last tick entry plus the interval, so
// consecutive reads differ by a multiple of it while one handle stays armed, and a delta that is
// not a multiple means it was re-armed. Without the period a reader holding two stamps can see
// that they differ and nothing more. The refused-fire counts answer the question `tickInFlight`
// raises and cannot close -- a tick holding the anchor is refusing an interval fire every period,
// and the early return that refuses it is the only event in the scheduler that writes nothing
// anywhere. A frozen hour and a quiet hour therefore leave the same trace in the budget and
// outcome counts, which is none; these are that difference, and unlike `tickInFlight` the
// per-tick count still carries information on an autonomous turn.
export type AutonomySchedulerMechanismEvidence = {
  observedAt: number;
  enabled: boolean;
  tickInFlight: boolean;
  intervalMs: number;
  droppedIntervalFires: AutonomySchedulerDescription["dropped_interval_fires"];
  intervalArmedAt: number | null;
  nextTickAt: number | null;
  scheduledTickAt: number | null;
  /** Optional only for historical captures and direct harness fixtures. */
  windowWakes?: AutonomySchedulerDescription["window_wakes"];
  /** Filing metadata only; independent of wake admission and outcome. */
  operatorAttentionIndex?: OperatorAttentionIndex;
  budget: AutonomySchedulerBudgetDescription;
  fleetBrake: AutonomySchedulerFleetBrakeDescription;
  sources: AutonomySchedulerDescription["sources"];
};

/**
 * Per-field disposition at the one boundary where `describe()`'s output crosses
 * into turn evidence (`retrieval-phase.ts`, hand-written and renamed).
 *
 * Five of the ten carried fields were lost at exactly that line and recovered
 * only when a reader noticed the page was thin. That failure has no symptom: the
 * target type is unchanged, so nothing is missing from any literal, no assertion
 * fires, no test goes red -- the surface simply renders less than it could, and
 * only someone holding both the type and the page can tell. It is not a name
 * asserting semantics the mechanism lacks; it is an absence, which is why reading
 * the render never finds it.
 *
 * The `Record` key is what makes this load-bearing: a new field on
 * `AutonomySchedulerDescription` fails typecheck here until someone says which
 * it is. Carrying it is then a normal edit; dropping it is a decision with a
 * written reason. Either way the loss stops being silent.
 *
 * The dropped reasons are rendered on the block itself, so the page states its
 * own gap against the read rather than leaving the count of missing fields
 * unknowable to anyone who cannot also read this file. With nothing dropped the
 * block says that instead, which is the same sentence in its empty case rather
 * than a silence: a page that stops mentioning its gap and a page that has none
 * would otherwise be the same page.
 */
export type AutonomySchedulerFieldDisposition = "carried" | { readonly dropped: string };

export const AUTONOMY_SCHEDULER_DESCRIPTION_FIELD_DISPOSITION: Record<
  keyof AutonomySchedulerDescription,
  AutonomySchedulerFieldDisposition
> = {
  observed_at: "carried",
  enabled: "carried",
  tick_in_flight: "carried",
  interval_ms: "carried",
  dropped_interval_fires: "carried",
  interval_armed_at: "carried",
  next_tick_at: "carried",
  scheduled_tick_at: "carried",
  window_wakes: "carried",
  budget: "carried",
  fleet_brake: "carried",
  sources: "carried",
};

/**
 * The dropped entries above, in declaration order, for surfaces that state their
 * own gap. Derived rather than restated so a later drop cannot make the rendered
 * sentence quietly false.
 */
export const AUTONOMY_SCHEDULER_DESCRIPTION_DROPPED_FIELDS: readonly string[] = Object.values(
  AUTONOMY_SCHEDULER_DESCRIPTION_FIELD_DISPOSITION,
)
  .filter((disposition): disposition is { readonly dropped: string } => disposition !== "carried")
  .map((disposition) => disposition.dropped);

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

// The regeneration ring has the same shape of scope trap as the suppression ring below, one
// step further in: it holds regenerations whose redrafted answer was then emitted, not
// regenerations. The breadcrumb is minted only under `finalAnswerRegenerated &&
// guardedEmission.kind === "message"` (turn-action/turn-action-coordinator.ts) and appended
// only on the message branch of post-generation-phase.ts, whose suppressed branch returns
// first. So a draft the commitment guard redrafted and a guard then suppressed lands in the
// suppression ring and nowhere here -- the two lists cannot share a turn id, ever. Measured on
// the live store (2026-08-27): zero overlap across all nine sessions, one of them carrying
// seven `commitment_violation_after_regenerate` silences beside a full regeneration ring that
// names none of the seven. Two reason codes (`commitment_violation_after_regenerate`,
// `invalid_tool_after_regenerate` -- the latter the finalizer's own retry, not this guard)
// still carry the redraft in their own name; under any other one it is unrecorded in both.
// system-prompt.ts states that scope on the rendered line, which is the fix for a gap of this
// shape here: name it, do not widen the ring.
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
