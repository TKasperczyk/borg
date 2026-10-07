import type { ExecutiveGoalScoreBasis } from "../executive/index.js";
import { IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS } from "../memory/identity/index.js";
import { headTailTextExcerpt } from "../util/text-excerpt.js";

export const AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS = 32_000;

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

function carriesTruncatedIdentityEventChange(payload: Record<string, unknown>): boolean {
  if (!Array.isArray(payload.recent_identity_events)) {
    return false;
  }

  return payload.recent_identity_events.some((event) => {
    if (event === null || typeof event !== "object" || Array.isArray(event)) {
      return false;
    }

    const change = (event as Record<string, unknown>).change;
    return (
      change !== null &&
      typeof change === "object" &&
      !Array.isArray(change) &&
      (change as Record<string, unknown>).excerpt_exact === false
    );
  });
}

function autonomyTriggerExcerptNotice(maxChars: number): string {
  return (
    `excerpt_notice: borg_autonomy_trigger content is mechanically bounded to ${maxChars} chars; ` +
    `each recent identity event's old-to-new change excerpt is bounded to ${IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS} chars. ` +
    "Head+tail excerpts are not summaries, and omitted characters remain available in the source records."
  );
}

// Disclosure labels stay whole at every depth. So does an open question's id, the handle any
// action on it takes, and its own text: it is the subject of the wake that carries it, and the
// ruminations reader returns question text only beside a note, so for a question with no notes
// an excerpt here can leave it unread. The evidence attached to the question absorbs the cut
// instead.
function keepsWhole(
  record: Record<string, unknown>,
  key: string,
  keepQuestionText: boolean,
): boolean {
  return (
    key === "disclosure" ||
    key === "disclosure_label" ||
    (key === "open_question_id" && typeof record.open_question_id === "string") ||
    (keepQuestionText && key === "question" && typeof record.open_question_id === "string")
  );
}

function isPresentedQuestion(value: unknown): value is Record<string, unknown> & {
  related_episodes: unknown[];
} {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as Record<string, unknown>).open_question_id === "string" &&
    Array.isArray((value as Record<string, unknown>).related_episodes)
  );
}

function relatedEpisodeCount(payload: Record<string, unknown>): number {
  const additional = Array.isArray(payload.additional_questions)
    ? payload.additional_questions
    : [];
  return [payload, ...additional]
    .filter(isPresentedQuestion)
    .reduce((total, question) => total + question.related_episodes.length, 0);
}

// Removes `count` related episodes whole from the end of the payload: the last presented
// question's lowest-ranked episode first, the primary question's last.
function omitTrailingRelatedEpisodes(
  payload: Record<string, unknown>,
  count: number,
): Record<string, unknown> {
  let remaining = count;
  const trim = (question: unknown): unknown => {
    if (remaining === 0 || !isPresentedQuestion(question)) {
      return question;
    }
    const kept = Math.max(0, question.related_episodes.length - remaining);
    remaining -= question.related_episodes.length - kept;
    return { ...question, related_episodes: question.related_episodes.slice(0, kept) };
  };
  const additional = payload.additional_questions;
  return trim(
    Array.isArray(additional)
      ? { ...payload, additional_questions: [...additional].reverse().map(trim).reverse() }
      : payload,
  ) as Record<string, unknown>;
}

function excerptPayloadText(value: unknown, maxChars: number, keepQuestionText: boolean): unknown {
  if (typeof value === "string") {
    const marker = "[ELIDED]";
    if (value.length <= Math.max(marker.length, maxChars)) {
      return value;
    }
    const excerpt = headTailTextExcerpt(value, Math.max(0, maxChars - marker.length));
    return `${excerpt.head}${marker}${excerpt.tail}`;
  }

  if (Array.isArray(value)) {
    return value.map((entry) => excerptPayloadText(entry, maxChars, keepQuestionText));
  }

  if (value === null || typeof value !== "object") {
    return value;
  }

  const record = value as Record<string, unknown>;
  return Object.fromEntries(
    Object.entries(record).map(([key, entry]) => [
      key,
      keepsWhole(record, key, keepQuestionText)
        ? entry
        : excerptPayloadText(entry, maxChars, keepQuestionText),
    ]),
  );
}

function excerptIdentityEvent(event: unknown, maxChars: number): unknown {
  if (event === null || typeof event !== "object" || Array.isArray(event)) {
    return event;
  }
  const row = event as Record<string, unknown>;
  const change = row.change;
  if (change === null || typeof change !== "object" || Array.isArray(change)) {
    return event;
  }
  const source = change as Record<string, unknown>;
  if (typeof source.excerpt_head !== "string") {
    return event;
  }
  const text =
    source.excerpt_head + (typeof source.excerpt_tail === "string" ? source.excerpt_tail : "");
  const excerpt = headTailTextExcerpt(text, maxChars);
  if (excerpt.exact) {
    return event;
  }
  return {
    ...row,
    change: {
      ...source,
      excerpt_head: excerpt.head,
      excerpt_tail: excerpt.tail,
      excerpt_exact: false,
      excerpt_chars: excerpt.renderedChars,
    },
  };
}

function boundAutonomyTriggerContext(
  context: AutonomyTriggerContext,
  maxChars: number,
): string {
  const rendered = renderAutonomyTriggerContext(context);
  const nestedExcerptWasTruncated = carriesTruncatedIdentityEventChange(context.payload);
  if (!nestedExcerptWasTruncated && rendered.length <= maxChars) {
    return rendered;
  }

  const events = context.payload.recent_identity_events;
  let eventCount = Array.isArray(events) ? events.length : 0;
  let omittedPayloadFields = 0;
  let omittedRelatedEpisodes = 0;
  let keepQuestionText = true;
  const retainedPayload = { ...context.payload };
  const render = (stringBudget: number): string => {
    const source = omitTrailingRelatedEpisodes(retainedPayload, omittedRelatedEpisodes);
    const payload = Object.fromEntries(
      Object.entries(source).map(([key, value]) => [
        key,
        key === "recent_identity_events" && Array.isArray(events)
          ? events.slice(0, eventCount).map((event) => excerptIdentityEvent(event, stringBudget))
          : keepsWhole(source, key, keepQuestionText)
            ? value
            : excerptPayloadText(value, stringBudget, keepQuestionText),
      ]),
    );
    const notice = [
      autonomyTriggerExcerptNotice(maxChars),
      Array.isArray(events)
        ? `recent_identity_events_omitted: ${events.length - eventCount}.`
        : null,
      omittedRelatedEpisodes > 0 ? `related_episodes_omitted: ${omittedRelatedEpisodes}.` : null,
      omittedPayloadFields > 0 ? `payload_fields_omitted: ${omittedPayloadFields}.` : null,
    ]
      .filter((part) => part !== null)
      .join(" ");
    // The planner applies a smaller head+tail crop to this block. Keep the
    // bounding and omission notice on the first line so that crop retains it.
    return `${notice}\n${renderAutonomyTriggerContext({ ...context, payload })}`;
  };

  const uncut = render(Number.POSITIVE_INFINITY);
  if (uncut.length <= maxChars) {
    return uncut;
  }

  // Reserve every selected event's metadata before spending any room on text.
  // If metadata alone exceeds the cap, exclude an explicitly counted suffix.
  let bounded = render(0);
  if (bounded.length > maxChars) {
    // The questions' related episodes leave whole, from the end, while that can still let the
    // question text fit beside the labels and structure that stay whole. Question text that
    // cannot is excerpted before any whole field is dropped.
    omittedRelatedEpisodes = relatedEpisodeCount(retainedPayload);
    if (render(0).length <= maxChars) {
      omittedRelatedEpisodes = 0;
      while (bounded.length > maxChars) {
        omittedRelatedEpisodes += 1;
        bounded = render(0);
      }
    } else {
      omittedRelatedEpisodes = 0;
      keepQuestionText = false;
      bounded = render(0);
    }
  }
  // A dropped trailing field can be additional_questions, which the wake latches whether or not
  // they were shown, so the related episodes leave before any field does. They leave together:
  // episodes that only fit at a zero text budget would leave every excerpt, the questions'
  // included, almost empty.
  if (bounded.length > maxChars && relatedEpisodeCount(retainedPayload) > 0) {
    omittedRelatedEpisodes = relatedEpisodeCount(retainedPayload);
    bounded = render(0);
  }
  while (bounded.length > maxChars && eventCount > 0) {
    eventCount -= 1;
    bounded = render(0);
  }
  // Other payloads can also contain more structure than the cap permits (for
  // example, a large array). Omit whole trailing fields, never serialized bytes.
  const otherFields = Object.keys(retainedPayload).filter(
    (key) => key !== "recent_identity_events",
  );
  while (bounded.length > maxChars && otherFields.length > 0) {
    delete retainedPayload[otherFields.pop()!];
    omittedPayloadFields += 1;
    bounded = render(0);
  }

  // Price the final serialization, including JSON escapes, calendar siblings,
  // domain notes, and the notice. Short values use only the room they need.
  let low = 0;
  let high = maxChars;
  while (low <= high) {
    const candidateBudget = Math.floor((low + high) / 2);
    const candidate = render(candidateBudget);
    if (candidate.length <= maxChars) {
      bounded = candidate;
      low = candidateBudget + 1;
    } else {
      high = candidateBudget - 1;
    }
  }
  return bounded;
}

// Each presented question keeps its own dormancy latch and disclosure label.
const OPEN_QUESTION_DORMANCY_DOMAIN_NOTE =
  "note on this dormant-question wake: up to three questions share this wake. Each last_touched is that question's dormancy anchor, never a count of how often it has already woken me. Completion latches every presented question at that stamp and updates its rumination count and time; a failure leaves the pair unlatched and can return. A latched question cannot wake me a second time until a later write moves last_touched. Rumination stamps include offline passes and completed dormant-question wakes; null means neither has been recorded.";

function carriesOpenQuestionDormancy(payload: Record<string, unknown>): boolean {
  return typeof payload.open_question_id === "string" && typeof payload.last_touched === "number";
}

function renderAutonomyTriggerContext(context: AutonomyTriggerContext): string {
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

export function formatAutonomyTriggerContext(
  context: AutonomyTriggerContext,
  maxChars = AUTONOMY_TRIGGER_CONTEXT_MAX_CHARS,
): string {
  return boundAutonomyTriggerContext(context, maxChars);
}
