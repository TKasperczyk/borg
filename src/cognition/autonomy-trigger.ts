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
const RECENT_IDENTITY_EVENTS_DOMAIN_NOTE =
  "note on recent_identity_events: this is the log of writes that went through my audited identity path, not every write a record received. Several writers change a record and bump its record_version without appending an event here, so the newest event for a record is not a witness to that record's current state: it reports its own values with the same confidence whether or not something wrote the record afterwards, degrading to stale rather than to absent, and no field on the event says which it is. Each event carries the record_version that write produced, so that number against the record's current version elsewhere is the only check available from here. The list is also the most recent events globally rather than per record, so a record's absence from it is not evidence that the record did not change.";

function carriesRecentIdentityEvents(payload: Record<string, unknown>): boolean {
  return Array.isArray(payload.recent_identity_events);
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
    hasGoalBatch ? "secondary_due_goals:" : null,
    hasGoalBatch ? (JSON.stringify(annotateEpochFields(secondaryDueGoals), null, 2) ?? "[]") : null,
  ]
    .filter((line): line is string => line !== null)
    .join("\n");
}
