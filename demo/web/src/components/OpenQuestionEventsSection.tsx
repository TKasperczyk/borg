import { useMemo } from "react";

import type { IdentityEvent } from "../api/types";
import { formatTime } from "../lib/stream-utils";
import { dateLabel, displayValue, isRecord, shortId } from "../screens/screen-utils";
import { Empty } from "./Empty";
import { IdRef } from "./Inspector/IdRef";
import { Tag } from "./Tag";

function eventActionTag(event: IdentityEvent) {
  if (event.overwrite_without_review) {
    return "bad";
  }
  if (event.reason === "open_question_duplicate_merge") {
    return "purple";
  }
  if (event.action === "create") {
    return "info";
  }
  if (event.action === "resolve") {
    return "acc";
  }
  if (event.action === "update") {
    return "warn";
  }
  return "";
}

function eventTimestamp(ts: number): string {
  return `${dateLabel(ts)} ${formatTime(ts)}`;
}

function provenanceLabel(provenance: IdentityEvent["provenance"]): string {
  if (!isRecord(provenance)) {
    return "provenance -";
  }

  const kind = typeof provenance.kind === "string" ? provenance.kind : "unknown";
  const process = typeof provenance.process === "string" ? provenance.process : null;
  return process === null ? `provenance ${kind}` : `provenance ${kind}/${process}`;
}

function eventRecordSummary(value: unknown): string {
  if (value === null || value === undefined) {
    return "none";
  }

  if (!isRecord(value)) {
    return displayValue(value);
  }

  const parts: string[] = [];
  const status = value.status;
  const urgency = value.urgency;
  const question = value.question;

  if (typeof status === "string") {
    parts.push(status);
  }
  if (typeof urgency === "number") {
    parts.push(`urg ${urgency.toFixed(2)}`);
  }
  if (typeof question === "string" && question.length > 0) {
    parts.push(question);
  }

  return parts.length === 0 ? displayValue(value) : parts.join(" · ");
}

export function OpenQuestionEventsSection({
  events,
  title = "open question events",
  ariaLabel = "open question events history",
  emptyLabel = "no open question events recorded",
  gridColumn = "span 7",
}: {
  events: readonly IdentityEvent[];
  title?: string;
  ariaLabel?: string;
  emptyLabel?: string;
  gridColumn?: string;
}) {
  const sortedEvents = useMemo(
    () => [...events].sort((left, right) => right.ts - left.ts || right.id - left.id),
    [events],
  );

  return (
    <div className="id-card" style={{ gridColumn }} aria-label={ariaLabel}>
      <div className="h">
        <span className="ttl">{title}</span>
        <span className="n">{events.length}</span>
      </div>
      <div className="body">
        <div className="timeline identity-events">
          {sortedEvents.map((event) => {
            const isDuplicateMerge = event.reason === "open_question_duplicate_merge";
            return (
              <div
                key={event.id}
                className={`ev ${event.overwrite_without_review ? "bad" : ""}`}
                data-testid="identity-event-row"
              >
                <div className="identity-event-head">
                  <Tag kind={eventActionTag(event)} dot>
                    {event.action}
                  </Tag>
                  {isDuplicateMerge ? <Tag kind="purple">duplicate merge</Tag> : null}
                  {event.overwrite_without_review ? (
                    <Tag kind="bad">without review gate</Tag>
                  ) : null}
                  {event.review_item_id === null ? null : (
                    <Tag kind="info">
                      <IdRef
                        id={String(event.review_item_id)}
                        type="review"
                        label={`review ${event.review_item_id}`}
                      />
                    </Tag>
                  )}
                  <span className="dim tab-num">{eventTimestamp(event.ts)}</span>
                </div>
                <div className="identity-event-record">
                  <IdRef
                    id={event.record_id}
                    label={shortId(event.record_id)}
                    title={event.record_id}
                  />
                  <span className="dim"> · {provenanceLabel(event.provenance)}</span>
                </div>
                <div className="identity-event-change">
                  <span>{eventRecordSummary(event.old_value)}</span>
                  <span className="dim"> -&gt; </span>
                  <span>{eventRecordSummary(event.new_value)}</span>
                </div>
                <div className="identity-event-reason">
                  reason <span>{event.reason ?? "-"}</span>
                </div>
              </div>
            );
          })}
          {sortedEvents.length === 0 ? (
            <div className="identity-empty-band">
              <Empty>{emptyLabel}</Empty>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
