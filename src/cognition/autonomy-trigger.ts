import type { ExecutiveGoalScoreBasis } from "../executive/index.js";

export type AutonomyTriggerContext = {
  source_name: string;
  source_type: "trigger" | "condition";
  event_id: string;
  sort_ts: number;
  payload: Record<string, unknown>;
  presentation?: {
    score_basis?: ExecutiveGoalScoreBasis;
  };
};

// Date's representable range; outside it toISOString() throws rather than
// returning a value, so this bounds the conversion's actual domain.
const MAX_EPOCH_MS = 8_640_000_000_000_000;

function epochToIso(value: unknown): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) > MAX_EPOCH_MS) {
    return null;
  }

  return new Date(value).toISOString();
}

// sort_ts is rendered as a calendar instant below; every other epoch in a
// trigger payload reaches the model as a bare 13-digit integer, which no amount
// of reasoning turns into a date without in-head arithmetic. f5e54b6 moved the
// write side to calendar dates for exactly this reason -- this is the read side
// of the same argument. Key on the schema's own `_at`/`_ts` naming convention,
// never on the value's magnitude, and add a sibling instead of replacing: the
// raw field stays authoritative and a misnamed non-timestamp costs one ignorable
// key rather than a corrupted one.
function annotateEpochFields(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(annotateEpochFields);
  }

  if (value === null || typeof value !== "object") {
    return value;
  }

  const source = value as Record<string, unknown>;
  const annotated: Record<string, unknown> = {};

  for (const [key, entry] of Object.entries(source)) {
    annotated[key] = annotateEpochFields(entry);

    if (!key.endsWith("_at") && !key.endsWith("_ts")) {
      continue;
    }

    const siblingKey = `${key}_iso`;
    const iso = epochToIso(entry);

    if (iso !== null && !(siblingKey in source)) {
      annotated[siblingKey] = iso;
    }
  }

  return annotated;
}

// The identity-event log records writes that went through the audited identity
// service; several repository writers bump a record's record_version without
// appending an event (the ruminator's own tick and urgency bookkeeping is the
// routine one). So the newest event for a record reports its values with the
// same confidence whether or not the record was written afterwards -- it
// degrades to stale, not to absent, and nothing on the event marks which. Key on
// the payload key, never on what the events say, and render for an empty array
// too: absence is exactly where the reading is most tempting.
//
// The list is also fetched as the newest ten globally, and the stream it draws
// from is bursty: the curator's trait-decay sweep appends one event per stale
// trait at a single timestamp, tens at a time, several times a day, and batch
// repairs do the same. A ten-row window over a stream that arrives in bursts of
// that size is nearly always a slice of one burst, so it reads as a survey of
// recent identity activity while being a fragment of a single write. Do not fix
// that by filtering event kinds out of the draw -- what is written is what
// happened, and hiding routine writes would make the window lie in the other
// direction. Say what the window is instead; the note below does.
const RECENT_IDENTITY_EVENTS_DOMAIN_NOTE =
  "note on recent_identity_events: this is the log of writes that went through my audited identity path, not every write a record received. Several writers change a record and bump its record_version without appending an event here, so the newest event for a record is not a witness to that record's current state: it reports its own values with the same confidence whether or not something wrote the record afterwards, degrading to stale rather than to absent, and no field on the event says which it is. Each event carries the record_version that write produced, so that number against the record's current version elsewhere is the only check available from here. The list is also the most recent events globally rather than per record, so a record's absence from it is not evidence that the record did not change. This stream arrives in bursts rather than one write at a time: a maintenance sweep or a repair pass appends one event per record it touched, tens of them under a single timestamp, and the routine sweeps far outnumber the writes I would narrate. So these are the newest events, not a sample of recent identity activity, and they are usually a slice of one burst whose own size is not shown here. Equal timestamps across them mean I am looking at one write, not at a period of time.";

function carriesRecentIdentityEvents(payload: Record<string, unknown>): boolean {
  return Array.isArray(payload.recent_identity_events);
}

// A dormant-question wake's event id is the question paired with the same stamp
// its sort_ts reports, and that pair is what the scheduler latches -- so the
// event is one-shot per dormancy, and an old sort_ts means the question has
// stood unwritten that long rather than that it has been waking me all along.
// Nothing in the rendered block says so: the block shows a date eight days back
// on a wake that fired today and looks like recurrence. The offline rumination
// loop's bookkeeping rides in the payload for the same reason -- the wake is not
// that loop's selector and cannot report its own passes as offline work.
const OPEN_QUESTION_DORMANCY_DOMAIN_NOTE =
  "note on this dormant-question wake: last_touched is the dormancy anchor this trigger measures against, and my event_id pairs the question with that same stamp, which is what gets latched once the wake completes. So this event cannot wake me a second time -- only a later write that moves last_touched mints a new stamp and a new event -- and an old sort_ts is how long the question has stood at that stamp, never a count of how often it has already woken me. Not every write moves it: the offline loop's rumination bookkeeping does not, and a question rendered into a context section is not written at all. A wake that fails before it completes leaves the pair unlatched and can return. unresolved_rumination_ticks and last_ruminated_at are that offline loop's own record under its own selection, which this wake neither feeds nor writes: on a question that is still open, a null last_ruminated_at means no offline pass has been written against it since it was opened.";

function carriesOpenQuestionDormancy(payload: Record<string, unknown>): boolean {
  return typeof payload.open_question_id === "string" && typeof payload.last_touched === "number";
}

export function formatAutonomyTriggerContext(context: AutonomyTriggerContext): string {
  const secondaryDueGoals = context.payload.secondary_due_goals;
  const hasGoalBatch = Array.isArray(secondaryDueGoals) && secondaryDueGoals.length > 0;
  const { secondary_due_goals: _secondaryDueGoals, ...primaryPayload } = context.payload;
  const payload =
    JSON.stringify(annotateEpochFields(hasGoalBatch ? primaryPayload : context.payload), null, 2) ??
    "{}";
  const sortTs = Number.isFinite(context.sort_ts)
    ? new Date(context.sort_ts).toISOString()
    : String(context.sort_ts);

  return [
    "Autonomous wake context:",
    `source_name: ${context.source_name}`,
    `source_type: ${context.source_type}`,
    `event_id: ${context.event_id}`,
    `sort_ts: ${sortTs}`,
    hasGoalBatch ? "primary_focus_payload:" : "payload:",
    payload,
    carriesRecentIdentityEvents(context.payload) ? RECENT_IDENTITY_EVENTS_DOMAIN_NOTE : null,
    carriesOpenQuestionDormancy(context.payload) ? OPEN_QUESTION_DORMANCY_DOMAIN_NOTE : null,
    hasGoalBatch ? "secondary_due_goals:" : null,
    hasGoalBatch ? (JSON.stringify(annotateEpochFields(secondaryDueGoals), null, 2) ?? "[]") : null,
  ]
    .filter((line): line is string => line !== null)
    .join("\n");
}
