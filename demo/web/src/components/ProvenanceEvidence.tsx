import type { ReactNode } from "react";

import { ApiError, getWhy } from "../api/client";
import { useApi } from "../hooks/use-api";
import { compactStreamText, formatTimestampForKey, streamContentText } from "../lib/stream-utils";
import { displayValue, fieldLabel, isInternalId, isRecord, shortId } from "../screens/screen-utils";
import { Empty } from "./Empty";
import { ErrorState } from "./ErrorState";
import { IdChip } from "./Inspector/IdChip";
import { IdRef } from "./Inspector/IdRef";
import { JsonValueView } from "./JsonValueView";
import { Loading } from "./Loading";
import { Tag } from "./Tag";

const PROVENANCE_PRIORITY_KEYS = [
  "citation_chain",
  "source_stream_ids",
  "source_stream_entry_ids",
  "source_episode_ids",
  "evidence_episode_ids",
  "identity_events",
  "reinforcement_events",
  "direct_edges",
  "walked_edges",
  "from_node",
  "to_node",
] as const;

type ProvenanceEntry = [key: string, value: unknown];

function meaningful(value: unknown): boolean {
  return value !== null && value !== undefined;
}

function jsonEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) {
    return true;
  }

  try {
    return JSON.stringify(left) === JSON.stringify(right);
  } catch {
    return false;
  }
}

function orderedEntries(data: Record<string, unknown>): ProvenanceEntry[] {
  const entries = Object.entries(data).filter(
    ([key, value]) => key !== "target_type" && key !== "record" && meaningful(value),
  );
  const byKey = new Map(entries);
  const ordered: ProvenanceEntry[] = [];

  for (const key of PROVENANCE_PRIORITY_KEYS) {
    if (byKey.has(key)) {
      ordered.push([key, byKey.get(key)]);
      byKey.delete(key);
    }
  }

  ordered.push(...entries.filter(([key]) => byKey.has(key)));

  if (meaningful(data.record)) {
    ordered.push(["record", data.record]);
  }

  return ordered;
}

function idArray(value: unknown): string[] | null {
  if (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((entry) => typeof entry === "string" && isInternalId(entry))
  ) {
    return value;
  }

  return null;
}

function ProvenanceSection({
  title,
  children,
  defaultOpen = true,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="provenance-section" open={defaultOpen}>
      <summary>{fieldLabel(title)}</summary>
      <div className="provenance-section-body">{children}</div>
    </details>
  );
}

function ProvenanceValue({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  const timestamp = formatTimestampForKey(fieldKey, value);
  if (timestamp !== null) {
    return <>{timestamp}</>;
  }

  if (typeof value === "string" && isInternalId(value)) {
    return <IdRef id={value} />;
  }

  const ids = idArray(value);
  if (ids !== null) {
    return (
      <span className="id-chip-list">
        {ids.map((id) => (
          <IdRef key={id} id={id} />
        ))}
      </span>
    );
  }

  if (value === null || value === undefined) {
    return null;
  }

  if (typeof value !== "object") {
    return <>{displayValue(value)}</>;
  }

  return <JsonValueView value={value} />;
}

function ProvenanceProps({ value }: { value: Record<string, unknown> }) {
  const entries = Object.entries(value).filter(([, entry]) => meaningful(entry));

  if (entries.length === 0) {
    return <Empty>no fields</Empty>;
  }

  return (
    <div className="props">
      {entries.map(([key, entry]) => (
        <div className="row" key={key}>
          <span className="k">{fieldLabel(key)}</span>
          <span className="v">
            <ProvenanceValue fieldKey={key} value={entry} />
          </span>
        </div>
      ))}
    </div>
  );
}

function CitationChain({ value }: { value: unknown }) {
  if (!Array.isArray(value) || value.length === 0) {
    return <Empty>no citations retained</Empty>;
  }

  return (
    <div className="timeline provenance-events">
      {value.map((entry, index) => {
        if (!isRecord(entry)) {
          return <JsonValueView key={index} value={entry} />;
        }

        const id = typeof entry.id === "string" ? entry.id : null;
        const kind = typeof entry.kind === "string" ? entry.kind : "stream";
        const timestamp = formatTimestampForKey("timestamp", entry.timestamp);
        const content = compactStreamText(streamContentText(entry.content), 180);

        return (
          <div className="ev" key={id ?? index}>
            <div className="identity-event-head">
              <Tag>{kind}</Tag>
              {id === null ? null : <IdRef id={id} type="stream_entry" label={shortId(id)} />}
              {timestamp === null ? null : <span className="dim tab-num">{timestamp}</span>}
            </div>
            <div className="identity-event-record">{content}</div>
            <details className="provenance-inline-raw">
              <summary>raw citation</summary>
              <JsonValueView value={entry} />
            </details>
          </div>
        );
      })}
    </div>
  );
}

type IdentityEventLike = {
  id?: unknown;
  record_type?: unknown;
  record_id?: unknown;
  action?: unknown;
  old_value?: unknown;
  new_value?: unknown;
  reason?: unknown;
  provenance?: unknown;
  review_item_id?: unknown;
  overwrite_without_review?: unknown;
  ts?: unknown;
};

function changedFields(oldValue: unknown, newValue: unknown): ProvenanceEntry[] {
  if (isRecord(oldValue) || isRecord(newValue)) {
    const oldRecord = isRecord(oldValue) ? oldValue : {};
    const newRecord = isRecord(newValue) ? newValue : {};
    const keys = [...new Set([...Object.keys(oldRecord), ...Object.keys(newRecord)])].sort();
    return keys
      .map((key) => [key, { old: oldRecord[key], next: newRecord[key] }] as ProvenanceEntry)
      .filter(([, change]) => isRecord(change) && !jsonEqual(change.old, change.next));
  }

  return jsonEqual(oldValue, newValue) ? [] : [["value", { old: oldValue, next: newValue }]];
}

function IdentityEventDiff({ event }: { event: IdentityEventLike }) {
  const changes = changedFields(event.old_value, event.new_value);

  if (changes.length === 0) {
    return <div className="dim">no changed fields retained</div>;
  }

  return (
    <div className="props">
      {changes.map(([key, value]) => {
        const oldValue = isRecord(value) ? value.old : undefined;
        const nextValue = isRecord(value) ? value.next : undefined;
        return (
          <div className="row" key={key}>
            <span className="k">{fieldLabel(key)}</span>
            <span className="v provenance-diff-value">
              <span>
                <ProvenanceValue fieldKey={key} value={oldValue} />
              </span>
              <span className="dim">-&gt;</span>
              <span>
                <ProvenanceValue fieldKey={key} value={nextValue} />
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

function IdentityEvents({ value }: { value: unknown }) {
  if (!Array.isArray(value) || value.length === 0) {
    return <Empty>no identity events retained</Empty>;
  }

  return (
    <div className="timeline provenance-events">
      {value.map((rawEvent, index) => {
        if (!isRecord(rawEvent)) {
          return <JsonValueView key={index} value={rawEvent} />;
        }

        const event = rawEvent as IdentityEventLike;
        const eventId = typeof event.id === "number" ? event.id : index;
        const action = typeof event.action === "string" ? event.action : "event";
        const recordId = typeof event.record_id === "string" ? event.record_id : null;
        const recordType = typeof event.record_type === "string" ? event.record_type : null;
        const timestamp = formatTimestampForKey("ts", event.ts);
        const reviewId =
          typeof event.review_item_id === "number" ? String(event.review_item_id) : null;
        const reason = typeof event.reason === "string" ? event.reason : null;

        return (
          <div
            className={`ev ${event.overwrite_without_review === true ? "bad" : ""}`}
            key={eventId}
          >
            <div className="identity-event-head">
              <Tag kind={event.overwrite_without_review === true ? "bad" : ""}>{action}</Tag>
              {recordType === null ? null : <Tag kind="info">{recordType}</Tag>}
              {reviewId === null ? null : (
                <Tag kind="info">
                  <IdRef id={reviewId} type="review" label={`review ${reviewId}`} />
                </Tag>
              )}
              {timestamp === null ? null : <span className="dim tab-num">{timestamp}</span>}
            </div>
            {recordId === null ? null : (
              <div className="identity-event-record">
                <IdRef id={recordId} label={shortId(recordId)} title={recordId} />
              </div>
            )}
            <IdentityEventDiff event={event} />
            {reason === null ? null : (
              <div className="identity-event-reason">
                reason <span>{reason}</span>
              </div>
            )}
            <details className="provenance-inline-raw">
              <summary>raw event</summary>
              <JsonValueView value={rawEvent} />
            </details>
          </div>
        );
      })}
    </div>
  );
}

function RecordList({ value }: { value: unknown }) {
  if (!Array.isArray(value)) {
    return <JsonValueView value={value} />;
  }

  if (value.length === 0) {
    return <Empty>none retained</Empty>;
  }

  return (
    <div className="provenance-list">
      {value.map((entry, index) => (
        <div
          className="provenance-list-item"
          key={isRecord(entry) && typeof entry.id === "string" ? entry.id : index}
        >
          {isRecord(entry) ? <ProvenanceProps value={entry} /> : <JsonValueView value={entry} />}
        </div>
      ))}
    </div>
  );
}

function ProvenanceSectionContent({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  if (fieldKey === "citation_chain") {
    return <CitationChain value={value} />;
  }

  if (fieldKey === "identity_events") {
    return <IdentityEvents value={value} />;
  }

  if (isRecord(value)) {
    return <ProvenanceProps value={value} />;
  }

  if (Array.isArray(value) && value.some((entry) => isRecord(entry))) {
    return <RecordList value={value} />;
  }

  return <ProvenanceValue fieldKey={fieldKey} value={value} />;
}

function ProvenancePayload({ data }: { data: Record<string, unknown> }) {
  const targetType = typeof data.target_type === "string" ? data.target_type : "unknown";
  const entries = orderedEntries(data);

  if (entries.length === 0) {
    return <Empty>no provenance fields</Empty>;
  }

  return (
    <div className="provenance-evidence">
      <div className="provenance-evidence-head">
        <Tag kind="info">{targetType}</Tag>
        {isRecord(data.record) && typeof data.record.id === "string" ? (
          <IdChip id={data.record.id} />
        ) : null}
      </div>
      {entries.map(([key, value]) => (
        <ProvenanceSection key={key} title={key} defaultOpen={key !== "record"}>
          <ProvenanceSectionContent fieldKey={key} value={value} />
        </ProvenanceSection>
      ))}
    </div>
  );
}

export function ProvenanceEvidence({ id }: { id: string }) {
  const api = useApi(() => getWhy(id), [id]);

  if (api.loading) {
    return <Loading>loading provenance</Loading>;
  }

  if (api.error !== null) {
    if (api.error instanceof ApiError && api.error.status === 404) {
      return <Empty>no provenance retained</Empty>;
    }
    return <ErrorState>{api.error.message}</ErrorState>;
  }

  if (api.data === null) {
    return <Empty>no provenance fields</Empty>;
  }

  const data = api.data;
  if (!isRecord(data) || Object.keys(data).length === 0) {
    return <Empty>no provenance fields</Empty>;
  }

  return <ProvenancePayload data={data} />;
}
