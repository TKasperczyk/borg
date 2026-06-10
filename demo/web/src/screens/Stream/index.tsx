import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react";

import { getAttachmentMetadata, getAttachmentStatuses } from "../../api/client";
import type {
  AttachmentMetadataResponse,
  AttachmentStatusItem,
  ImagePerceptionRecord,
  StreamEntry,
  StreamEntryKind,
  WsState,
} from "../../api/types";
import { Empty } from "../../components/Empty";
import { ErrorState } from "../../components/ErrorState";
import { ImagePlaceholder } from "../../components/ImagePlaceholder";
import { IdChip } from "../../components/Inspector/IdChip";
import { IdRef } from "../../components/Inspector/IdRef";
import { Loading } from "../../components/Loading";
import { Tag, type TagKind } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { useStreamWindow } from "../../hooks/use-stream-window";
import { copyText } from "../../lib/clipboard";
import { activateOnEnterOrSpace } from "../../lib/keyboard";
import {
  UNCLAIMED_STREAM_GROUP_LABEL,
  applyStreamStructuralFilters,
  groupStreamEntriesByTurn,
  hasStreamAttachment,
  isAbortedTurnEntry,
  isCompressed,
  streamEntryAttachmentId,
  type StreamStructuralFilterId,
  type StreamStructuralFilterState,
  type StreamTurnGroup,
} from "../../lib/stream-grouping";
import { streamOutcomeSummary, type StreamOutcomeSummary } from "../../lib/stream-outcomes";
import { compactStreamText, formatTime, streamContentText } from "../../lib/stream-utils";
import {
  contentField,
  displayValue,
  fieldLabel,
  isRecord,
  jsonText,
  shortId,
} from "../screen-utils";

const STREAM_KINDS: StreamEntryKind[] = [
  "user_msg",
  "user_image_attachment",
  "agent_msg",
  "agent_suppressed",
  "agent_observed",
  "thought",
  "tool_call",
  "tool_result",
  "perception",
  "internal_event",
  "dream_report",
];

function kindTag(kind: StreamEntryKind): TagKind {
  switch (kind) {
    case "user_msg":
    case "user_image_attachment":
    case "perception":
      return "info";
    case "agent_msg":
      return "acc";
    case "agent_suppressed":
      return "bad";
    case "agent_observed":
    case "tool_call":
    case "tool_result":
      return "warn";
    case "thought":
    case "dream_report":
      return "purple";
    case "internal_event":
      return "";
  }
}

function streamConnectionTagKind(wsState: WsState): TagKind {
  if (wsState === "live") {
    return "acc";
  }
  if (wsState === "reconnecting") {
    return "warn";
  }
  return "bad";
}

function streamConnectionLabel(wsState: WsState): string {
  if (wsState === "live") {
    return "tailing";
  }
  if (wsState === "reconnecting") {
    return "reconnecting";
  }
  return "offline";
}

function mediaType(entry: StreamEntry): string | undefined {
  return contentField(entry.content, "media_type");
}

function StreamOutcomeTags({ summary }: { summary: StreamOutcomeSummary }) {
  const invalidTool = summary.finalizerInvalidTool;
  const reasonLabel =
    summary.reason === null
      ? null
      : summary.outcome.outcomeClass === "observed"
        ? summary.reason
        : fieldLabel(summary.reason);

  return (
    <>
      <Tag kind={summary.outcome.tagKind} dot>
        {summary.outcome.label}
      </Tag>
      {reasonLabel === null ? null : <Tag kind={summary.outcome.tagKind}>reason {reasonLabel}</Tag>}
      {summary.primaryNoOutputReason === undefined ? null : (
        <Tag>primary {fieldLabel(summary.primaryNoOutputReason)}</Tag>
      )}
      {summary.noOutputCategories.map((category) => (
        <Tag key={`category:${category}`}>category {fieldLabel(category)}</Tag>
      ))}
      {summary.structuralNoOutputFlags.map((flag) => (
        <Tag key={`flag:${flag}`}>flag {fieldLabel(flag)}</Tag>
      ))}
      {invalidTool === undefined ? null : (
        <>
          <Tag kind="bad">tool {displayValue(invalidTool.tool_name)}</Tag>
          <Tag kind="bad">attempt {fieldLabel(invalidTool.attempt)}</Tag>
          <Tag kind="bad">invalid {displayValue(invalidTool.reason)}</Tag>
        </>
      )}
    </>
  );
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

type AutonomousWakeInfo = {
  sourceName: string;
  triggerType?: string;
  intervalMs?: number;
};

// An autonomy wake (scheduled reflection, scheduled wake, condition trigger, etc.) is recorded as
// an internal_event whose content.kind is "autonomous_wake". Surface it distinctly so reflection
// phases are recognizable in the stream instead of reading as a generic internal event.
function autonomousWakeInfo(entry: StreamEntry): AutonomousWakeInfo | null {
  if (entry.kind !== "internal_event") {
    return null;
  }
  const content = isRecord(entry.content) ? entry.content : null;
  if (content === null || content.kind !== "autonomous_wake") {
    return null;
  }
  const payload = isRecord(content.payload) ? content.payload : null;
  return {
    sourceName: optionalString(content.source_name) ?? "autonomous wake",
    triggerType: optionalString(content.trigger_type),
    intervalMs: payload === null ? undefined : optionalNumber(payload.interval_ms),
  };
}

function formatWakeInterval(ms: number): string {
  if (ms >= 3_600_000) {
    const hours = ms / 3_600_000;
    return `${Number.isInteger(hours) ? hours : hours.toFixed(1)}h`;
  }
  if (ms >= 60_000) {
    return `${Math.round(ms / 60_000)}m`;
  }
  return `${Math.round(ms / 1000)}s`;
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function arrayCount(value: unknown): number | undefined {
  return Array.isArray(value) ? value.length : undefined;
}

function compactDisplayValue(value: unknown, maxLength = 80): string {
  return compactStreamText(displayValue(value), maxLength);
}

function structuralHint(content: unknown): string {
  if (typeof content === "string") {
    return compactStreamText(content, 140) || "empty text";
  }

  if (Array.isArray(content)) {
    return `${content.length} ${content.length === 1 ? "item" : "items"}`;
  }

  if (isRecord(content)) {
    const keys = Object.keys(content);
    if (keys.length === 0) {
      return "empty object";
    }
    const visibleKeys = keys.slice(0, 4).map(fieldLabel).join(", ");
    return `${keys.length} ${keys.length === 1 ? "field" : "fields"}: ${visibleKeys}${
      keys.length > 4 ? ", ..." : ""
    }`;
  }

  return displayValue(content);
}

function textSnippet(
  content: unknown,
  fieldNames: readonly string[] = ["text", "summary", "caption", "note"],
): string | undefined {
  if (typeof content === "string") {
    const snippet = compactStreamText(content, 180);
    return snippet.length > 0 ? snippet : undefined;
  }

  if (Array.isArray(content)) {
    const textBlocks = content.flatMap((block) =>
      isRecord(block) && typeof block.text === "string" ? [block.text] : [],
    );
    if (textBlocks.length > 0) {
      const snippet = compactStreamText(textBlocks.join("\n"), 180);
      return snippet.length > 0 ? snippet : undefined;
    }
    return undefined;
  }

  if (!isRecord(content)) {
    return undefined;
  }

  for (const fieldName of fieldNames) {
    const value = content[fieldName];
    if (typeof value === "string" && value.length > 0) {
      return compactStreamText(value, 180);
    }
  }

  return undefined;
}

function messageSummary(content: unknown): string {
  return textSnippet(content, ["text"]) ?? structuralHint(content);
}

function ThoughtSummary({ entry }: { entry: StreamEntry }) {
  const snippet = textSnippet(entry.content, ["text", "summary", "note", "thought"]);

  return (
    <>
      <Tag kind="purple" dot>
        thought
      </Tag>
      <span>{snippet ?? structuralHint(entry.content)}</span>
    </>
  );
}

function PerceptionSummary({ entry }: { entry: StreamEntry }) {
  const content = isRecord(entry.content) ? entry.content : null;
  const mode = content === null ? undefined : optionalString(content.mode);
  const entityCount = content === null ? undefined : arrayCount(content.entities);
  const identityCount = content === null ? undefined : arrayCount(content.userIdentityNames);
  const affectiveSignal =
    content !== null && isRecord(content.affectiveSignal)
      ? optionalString(content.affectiveSignal.dominant_emotion)
      : undefined;
  const snippet = textSnippet(entry.content, ["text", "summary", "caption", "note"]);

  return (
    <>
      <Tag kind="info" dot>
        perception
      </Tag>
      {mode === undefined ? null : <Tag>mode {fieldLabel(mode)}</Tag>}
      {entityCount === undefined ? null : <Tag>{entityCount} entities</Tag>}
      {identityCount === undefined || identityCount === 0 ? null : (
        <Tag>{identityCount} identities</Tag>
      )}
      {affectiveSignal === undefined ? null : <Tag>emotion {fieldLabel(affectiveSignal)}</Tag>}
      {snippet === undefined ? null : <span className="dim">{snippet}</span>}
      {content === null && snippet === undefined ? (
        <span>{structuralHint(entry.content)}</span>
      ) : null}
    </>
  );
}

function DreamReportSummary({ entry }: { entry: StreamEntry }) {
  const content: Record<string, unknown> = isRecord(entry.content) ? entry.content : {};
  const runId = optionalString(content.run_id);
  const processes = stringArray(content.processes);
  const changes = optionalNumber(content.changes) ?? 0;
  const tokensUsed = optionalNumber(content.tokens_used) ?? 0;
  const errorCount = arrayCount(content.errors) ?? 0;
  const budgetExhaustedProcesses = stringArray(content.budget_exhausted_processes);

  return (
    <>
      <Tag kind="purple" dot>
        dream
      </Tag>
      {runId === undefined ? null : <Tag>run {shortId(runId)}</Tag>}
      <span className="dim">{processes.length === 0 ? "no processes" : processes.join(", ")}</span>
      <Tag kind={changes > 0 ? "acc" : ""}>{changes} changes</Tag>
      <Tag>{tokensUsed} tok</Tag>
      <Tag kind={errorCount > 0 ? "bad" : "acc"}>{errorCount} errors</Tag>
      {content.dry_run === true ? <Tag kind="warn">dry run</Tag> : null}
      {budgetExhaustedProcesses.length === 0 ? null : (
        <Tag kind="warn">budget {budgetExhaustedProcesses.join(", ")}</Tag>
      )}
    </>
  );
}

function InternalEventSummary({ entry }: { entry: StreamEntry }) {
  const wake = autonomousWakeInfo(entry);
  if (wake !== null) {
    return (
      <>
        <Tag kind="info" dot>
          {fieldLabel(wake.sourceName)}
        </Tag>
        {wake.triggerType === undefined ? null : <Tag>{fieldLabel(wake.triggerType)}</Tag>}
        {wake.intervalMs === undefined ? null : (
          <Tag>every {formatWakeInterval(wake.intervalMs)}</Tag>
        )}
      </>
    );
  }

  const content = isRecord(entry.content) ? entry.content : null;

  if (content === null) {
    return (
      <>
        <Tag>internal</Tag>
        <span>{structuralHint(entry.content)}</span>
      </>
    );
  }

  const eventKind =
    optionalString(content.event) ??
    optionalString(content.kind) ??
    optionalString(content.hook) ??
    "internal event";
  const salientFields = [
    "trigger",
    "outcome_summary",
    "status",
    "outcome",
    "reason",
    "source",
    "process",
    "action",
    "phase",
  ];
  const sourceEntryCount = arrayCount(content.source_stream_entry_ids);
  const citedEntryCount = arrayCount(content.cited_stream_entry_ids);

  return (
    <>
      <Tag>event</Tag>
      <span>{compactStreamText(eventKind, 80)}</span>
      {salientFields.flatMap((fieldName) => {
        const value = content[fieldName];
        if (
          value === undefined ||
          value === null ||
          (typeof value === "string" && value.length === 0)
        ) {
          return [];
        }
        return [
          <Tag key={fieldName}>
            {fieldLabel(fieldName)} {compactDisplayValue(value)}
          </Tag>,
        ];
      })}
      {sourceEntryCount === undefined ? null : <Tag>{sourceEntryCount} source refs</Tag>}
      {citedEntryCount === undefined ? null : <Tag>{citedEntryCount} cited refs</Tag>}
    </>
  );
}

function ToolCallSummary({ entry }: { entry: StreamEntry }) {
  const content: Record<string, unknown> = isRecord(entry.content) ? entry.content : {};
  const toolName = optionalString(content.tool_name) ?? optionalString(content.name) ?? "tool";
  const callId = optionalString(content.call_id);
  const origin = optionalString(content.origin);

  return (
    <>
      <Tag kind="warn" dot>
        call
      </Tag>
      <Tag>{compactStreamText(toolName, 80)}</Tag>
      {callId === undefined ? null : <Tag>id {shortId(callId)}</Tag>}
      {origin === undefined ? null : <Tag>origin {fieldLabel(origin)}</Tag>}
      {content.skipped === true ? <Tag kind="warn">skipped</Tag> : null}
      {optionalString(content.skip_reason) === undefined ? null : (
        <span className="dim">
          {compactStreamText(optionalString(content.skip_reason) ?? "", 100)}
        </span>
      )}
    </>
  );
}

function ToolResultSummary({ entry }: { entry: StreamEntry }) {
  const content: Record<string, unknown> = isRecord(entry.content) ? entry.content : {};
  const callId = optionalString(content.call_id);
  const ok = typeof content.ok === "boolean" ? content.ok : undefined;
  const durationMs = optionalNumber(content.duration_ms);
  const outputHint = content.output === undefined ? undefined : structuralHint(content.output);
  const error = optionalString(content.error);

  return (
    <>
      <Tag kind={ok === false ? "bad" : ok === true ? "acc" : "warn"} dot>
        {ok === false ? "failed" : ok === true ? "ok" : "result"}
      </Tag>
      {callId === undefined ? null : <Tag>id {shortId(callId)}</Tag>}
      {durationMs === undefined ? null : <Tag>{durationMs} ms</Tag>}
      {error === undefined ? null : <span className="dim">{compactStreamText(error, 120)}</span>}
      {error !== undefined || outputHint === undefined ? null : (
        <span className="dim">output {outputHint}</span>
      )}
    </>
  );
}

function attachmentSummary(entry: StreamEntry): string {
  const content: Record<string, unknown> = isRecord(entry.content) ? entry.content : {};
  const id = streamEntryAttachmentId(entry) ?? optionalString(content.id) ?? "attachment";
  const type = mediaType(entry) ?? optionalString(content.kind);
  const perceptionId = optionalString(content.perception_id);
  const parts = [shortId(id)];

  if (type !== undefined) {
    parts.push(type);
  }
  if (perceptionId !== undefined) {
    parts.push(`perception ${shortId(perceptionId)}`);
  }

  return parts.join(" · ");
}

function streamAttachmentPerceptionId(
  entry: StreamEntry,
  attachment?: AttachmentMetadataResponse | null,
): string | undefined {
  if (
    attachment?.attachment.perception_id !== null &&
    attachment?.attachment.perception_id !== undefined
  ) {
    return attachment.attachment.perception_id;
  }

  const content: Record<string, unknown> = isRecord(entry.content) ? entry.content : {};
  return optionalString(content.perception_id);
}

function StreamProvenanceRows({ entry, status }: { entry: StreamEntry; status: string }) {
  const responseSourceEntryIds = sourceEntryIds(entry);

  return (
    <div className="props">
      <div className="row">
        <span className="k">kind</span>
        <span className="v">{entry.kind}</span>
      </div>
      <div className="row">
        <span className="k">session_id</span>
        <span className="v">
          <IdChip id={entry.session_id} type="session" />
        </span>
      </div>
      <div className="row">
        <span className="k">turn_id</span>
        <span className="v">
          {entry.turn_id === undefined ? "—" : <IdChip id={entry.turn_id} type="turn" />}
        </span>
      </div>
      <div className="row">
        <span className="k">audience</span>
        <span className="v">{entry.audience ?? "global"}</span>
      </div>
      <div className="row">
        <span className="k">sender_entity_id</span>
        <span className="v">
          {entry.sender_entity_id === null ? (
            "—"
          ) : (
            <IdChip id={entry.sender_entity_id} type="entity" />
          )}
        </span>
      </div>
      <div className="row">
        <span className="k">reply_target_entity_id</span>
        <span className="v">
          {entry.reply_target_entity_id === null ? (
            "—"
          ) : (
            <IdChip id={entry.reply_target_entity_id} type="entity" />
          )}
        </span>
      </div>
      <div className="row">
        <span className="k">status</span>
        <span className="v">{status}</span>
      </div>
      {entry.entry_index === undefined ? null : (
        <div className="row">
          <span className="k">entry_index</span>
          <span className="v">{entry.entry_index}</span>
        </div>
      )}
      <div className="row">
        <span className="k">compressed</span>
        <span className="v">{String(entry.compressed)}</span>
      </div>
      {entry.persistence_class === undefined ? null : (
        <div className="row">
          <span className="k">persistence_class</span>
          <span className="v">{entry.persistence_class}</span>
        </div>
      )}
      {entry.token_estimate === undefined ? null : (
        <div className="row">
          <span className="k">token_estimate</span>
          <span className="v">{entry.token_estimate}</span>
        </div>
      )}
      {entry.source_message_key === undefined ? null : (
        <div className="row">
          <span className="k">source_message_key</span>
          <span className="v">
            {entry.source_message_key.source_type}/{entry.source_message_key.source_external_id}/
            {entry.source_message_key.external_message_id}
          </span>
        </div>
      )}
      {responseSourceEntryIds.length === 0 ? null : (
        <div className="row">
          <span className="k">response_to</span>
          <span className="v idref-list">
            {responseSourceEntryIds.map((id) => (
              <IdChip key={id} id={id} type="stream_entry" />
            ))}
          </span>
        </div>
      )}
    </div>
  );
}

function StreamRowSummary({ entry }: { entry: StreamEntry }): ReactNode {
  switch (entry.kind) {
    case "user_msg":
    case "agent_msg":
      return messageSummary(entry.content);
    case "thought":
      return <ThoughtSummary entry={entry} />;
    case "perception":
      return <PerceptionSummary entry={entry} />;
    case "dream_report":
      return <DreamReportSummary entry={entry} />;
    case "internal_event":
      return <InternalEventSummary entry={entry} />;
    case "tool_call":
      return <ToolCallSummary entry={entry} />;
    case "tool_result":
      return <ToolResultSummary entry={entry} />;
    case "user_image_attachment":
      return attachmentSummary(entry);
    case "agent_suppressed":
    case "agent_observed":
      return structuralHint(entry.content);
    default:
      return `${(entry as { kind: string }).kind} · ${structuralHint(entry.content)}`;
  }
}

function usesTagSummary(entry: StreamEntry): boolean {
  return (
    entry.kind === "thought" ||
    entry.kind === "perception" ||
    entry.kind === "dream_report" ||
    entry.kind === "internal_event" ||
    entry.kind === "tool_call" ||
    entry.kind === "tool_result"
  );
}

function summarizeStatus(
  entry: StreamEntry,
  attachment?: AttachmentMetadataResponse | null,
): string {
  if (attachment?.status.quarantined === true) {
    return "quarantined";
  }
  if (entry.turn_status === "aborted") {
    return "aborted";
  }
  return "active";
}

function groupStatusTagKind(status: StreamTurnGroup["status"]): TagKind {
  switch (status) {
    case "active":
      return "acc";
    case "aborted":
      return "bad";
    case "mixed":
      return "warn";
    case "maintenance":
      return "info";
  }
}

function groupTimeRange(group: StreamTurnGroup): string {
  if (group.startTimestamp === group.endTimestamp) {
    return formatTime(group.endTimestamp);
  }
  return `${formatTime(group.startTimestamp)}-${formatTime(group.endTimestamp)}`;
}

function sourceEntryIds(entry: StreamEntry): string[] {
  const ids = entry.response_to?.source_entry_ids;
  return Array.isArray(ids) ? ids.filter((id): id is string => typeof id === "string") : [];
}

function formatBytes(value: number | undefined): string {
  if (value === undefined) {
    return "-";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatIsoTimestamp(value: number | undefined): string {
  return value === undefined ? "-" : new Date(value).toISOString();
}

function CopyableHash({ value }: { value: string | undefined }) {
  if (value === undefined || value.length === 0) {
    return <span>-</span>;
  }

  return (
    <span className="copyable-hash">
      <span>{value}</span>
      <button
        type="button"
        className="btn sm ghost"
        aria-label="copy attachment sha256"
        onClick={() => {
          void copyText(value);
        }}
      >
        copy
      </button>
    </span>
  );
}

function BooleanStatusChip({ label, value }: { label: string; value: boolean | undefined }) {
  if (value === undefined) {
    return null;
  }

  return (
    <Tag kind={value ? "acc" : "bad"}>
      {label} {String(value)}
    </Tag>
  );
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function perceptionString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function perceptionBoolean(value: unknown): string | undefined {
  return typeof value === "boolean" ? String(value) : undefined;
}

function PerceptionField({
  label,
  value,
  chips = false,
}: {
  label: string;
  value: string | readonly string[] | undefined;
  chips?: boolean;
}) {
  if (value === undefined || (Array.isArray(value) && value.length === 0)) {
    return null;
  }

  return (
    <div className="row">
      <span className="k">{label}</span>
      <span className={chips ? "v chip-lines" : "v"}>
        {Array.isArray(value)
          ? chips
            ? value.map((item) => <Tag key={item}>{item}</Tag>)
            : value.join("; ")
          : value}
      </span>
    </div>
  );
}

function AttachmentPreview({
  attachmentId,
  mediaTypeValue,
  audience,
  quarantined,
}: {
  attachmentId: string;
  mediaTypeValue?: string;
  audience?: string;
  quarantined: boolean;
}) {
  if (audience === undefined) {
    return (
      <div className="img-ph stream-no-audience-preview" title={attachmentId}>
        <span>preview unavailable</span>
        <span>no audience</span>
      </div>
    );
  }

  return (
    <ImagePlaceholder
      attachmentId={attachmentId}
      mediaType={mediaTypeValue}
      audience={audience}
      size="lg"
      quarantined={quarantined}
    />
  );
}

function AttachmentPerceptionRows({
  perception,
}: {
  perception: ImagePerceptionRecord | null | undefined;
}) {
  if (perception === null || perception === undefined) {
    return null;
  }

  return (
    <>
      <div className="upper dim" style={{ marginTop: 14, marginBottom: 6 }}>
        perception
      </div>
      <div className="attachment-props compact-props">
        <PerceptionField label="payload_id" value={perceptionString(perception.payload_id)} />
        <PerceptionField label="image_kind" value={perceptionString(perception.image_kind)} />
        <PerceptionField label="active" value={perceptionBoolean(perception.active)} />
        <PerceptionField label="audience" value={perceptionString(perception.audience)} />
        <PerceptionField label="visible_text" value={stringList(perception.visible_text)} />
        <PerceptionField label="objects" value={stringList(perception.objects)} chips />
        <PerceptionField
          label="people_or_roles"
          value={stringList(perception.people_or_roles)}
          chips
        />
        <PerceptionField label="scene" value={perceptionString(perception.scene)} />
        <PerceptionField
          label="colors_and_visual_attributes"
          value={stringList(perception.colors_and_visual_attributes)}
          chips
        />
        <PerceptionField
          label="spatial_relationships"
          value={stringList(perception.spatial_relationships)}
        />
        <PerceptionField
          label="possible_user_relevant_details"
          value={stringList(perception.possible_user_relevant_details)}
        />
        <PerceptionField label="search_terms" value={stringList(perception.search_terms)} chips />
        <PerceptionField label="uncertainties" value={stringList(perception.uncertainties)} />
        <PerceptionField
          label="embedding_status"
          value={perceptionString(perception.embedding_status)}
        />
      </div>
    </>
  );
}

function AttachmentDetail({
  selected,
  attachmentId,
  metadata,
  perceptionId,
}: {
  selected: StreamEntry;
  attachmentId: string;
  metadata: AttachmentMetadataResponse | null;
  perceptionId?: string;
}) {
  const attachment = metadata?.attachment;
  const status = metadata?.status;
  const previewAudience = selected.audience ?? attachment?.audience ?? undefined;
  const mediaTypeValue = attachment?.media_type ?? mediaType(selected);

  return (
    <>
      <div className="upper dim" style={{ marginBottom: 6 }}>
        attachment
      </div>
      <div className="att-card">
        <AttachmentPreview
          attachmentId={attachmentId}
          mediaTypeValue={mediaTypeValue}
          audience={previewAudience}
          quarantined={status?.quarantined === true}
        />
        <div className="att-card-meta">
          <div className="att-card-id">
            <IdRef id={attachmentId} type="attachment" label={attachmentId} hint={attachment} />
            {perceptionId === undefined ? null : (
              <IdRef
                id={perceptionId}
                type="image_perception"
                label={`perception ${perceptionId}`}
                hint={metadata?.perception}
              />
            )}
          </div>
          <div className="att-card-caption">
            {metadata?.perception?.caption ?? "perception unavailable"}
          </div>
          <div className="att-card-stats">
            <span>
              {attachment?.width ?? "?"}x{attachment?.height ?? "?"}
            </span>
            <span>{shortId(attachment?.sha256)}</span>
            <span>{status?.quarantined === true ? "quarantined cascade" : "active"}</span>
          </div>
          <div className="att-status-chips">
            <BooleanStatusChip label="active" value={status?.active} />
            <BooleanStatusChip label="quarantined" value={status?.quarantined} />
            <BooleanStatusChip label="stream_active" value={status?.stream_active} />
            <BooleanStatusChip label="parent_active" value={status?.parent_active} />
          </div>
        </div>
      </div>
      <div className="attachment-props compact-props">
        <div className="row">
          <span className="k">media_type</span>
          <span className="v">{mediaTypeValue ?? "-"}</span>
        </div>
        <div className="row">
          <span className="k">byte_size</span>
          <span className="v">{formatBytes(attachment?.byte_size)}</span>
        </div>
        <div className="row">
          <span className="k">sha256</span>
          <span className="v">
            <CopyableHash value={attachment?.sha256} />
          </span>
        </div>
        <div className="row">
          <span className="k">created_at</span>
          <span className="v">{formatIsoTimestamp(attachment?.created_at)}</span>
        </div>
        <div className="row">
          <span className="k">parent_entry_id</span>
          <span className="v">
            {attachment?.parent_entry_id == null ? (
              "-"
            ) : (
              <IdRef
                id={attachment.parent_entry_id}
                type="stream_entry"
                label={attachment.parent_entry_id}
              />
            )}
          </span>
        </div>
        <div className="row">
          <span className="k">parent_turn_id</span>
          <span className="v">
            {attachment?.parent_turn_id == null ? (
              "-"
            ) : (
              <IdRef id={attachment.parent_turn_id} type="turn" label={attachment.parent_turn_id} />
            )}
          </span>
        </div>
      </div>
      <AttachmentPerceptionRows perception={metadata?.perception} />
    </>
  );
}

function StreamTimelineRow({
  entry,
  selected,
  attachmentStatus,
  onSelect,
}: {
  entry: StreamEntry;
  selected: boolean;
  attachmentStatus?: AttachmentStatusItem["status"];
  onSelect: (id: string) => void;
}) {
  const attId = streamEntryAttachmentId(entry);
  const isAttachment = entry.kind === "user_image_attachment";
  const outcomeSummary = streamOutcomeSummary(entry);
  const wake = autonomousWakeInfo(entry);

  return (
    <div
      className={`stream-row ${selected ? "selected " : ""}${isAttachment ? "kind-attachment" : ""}`}
      role="button"
      tabIndex={0}
      aria-label={`select stream entry ${entry.id}`}
      aria-pressed={selected}
      onClick={() => onSelect(entry.id)}
      onKeyDown={(event) => activateOnEnterOrSpace(event, () => onSelect(entry.id))}
    >
      <span className="t">{formatTime(entry.timestamp)}</span>
      <span className={`k ${wake !== null ? "info" : kindTag(entry.kind)}`}>
        {wake !== null ? "wake" : entry.kind}
      </span>
      <span
        className={
          isAttachment
            ? "att-inline"
            : outcomeSummary === null && !usesTagSummary(entry)
              ? "body"
              : "body outcome-tags"
        }
      >
        {isAttachment ? (
          <>
            <ImagePlaceholder
              attachmentId={attId}
              mediaType={mediaType(entry)}
              audience={entry.audience}
              size="xs"
              quarantined={attachmentStatus?.quarantined === true}
            />
            <span className="body-txt">{attachmentSummary(entry)}</span>
            {attachmentStatus?.quarantined === true ? <Tag kind="bad">quarantined</Tag> : null}
          </>
        ) : outcomeSummary !== null ? (
          <StreamOutcomeTags summary={outcomeSummary} />
        ) : (
          <StreamRowSummary entry={entry} />
        )}
      </span>
      <span className="aud">{entry.audience ?? "global"}</span>
    </div>
  );
}

function StreamGroupHeader({
  group,
  collapsed,
  onToggle,
}: {
  group: StreamTurnGroup;
  collapsed: boolean;
  onToggle: (groupId: string) => void;
}) {
  return (
    <div className="stream-group-head">
      <button
        type="button"
        className="stream-group-toggle"
        aria-label={`${collapsed ? "expand" : "collapse"} ${group.label}`}
        aria-expanded={!collapsed}
        onClick={() => onToggle(group.id)}
      >
        {collapsed ? "+" : "-"}
      </button>
      <span className="stream-group-id">
        {group.turnId === null ? (
          UNCLAIMED_STREAM_GROUP_LABEL
        ) : (
          <IdRef id={group.turnId} type="turn" label={group.turnId} />
        )}
      </span>
      <span className="stream-group-meta">{groupTimeRange(group)}</span>
      <span className="stream-group-meta">{group.entryCount} entries</span>
      <Tag kind={groupStatusTagKind(group.status)}>{group.status}</Tag>
    </div>
  );
}

export function StreamScreen({ sessionId }: { sessionId: string }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectionSeeded, setSelectionSeeded] = useState(false);
  const [selectedKinds, setSelectedKinds] = useState<Set<StreamEntryKind>>(
    () => new Set(STREAM_KINDS),
  );
  const [audience, setAudience] = useState("all");
  const [structuralFilters, setStructuralFilters] = useState<StreamStructuralFilterState>({});
  const [collapsedGroupIds, setCollapsedGroupIds] = useState<Set<string>>(() => new Set());
  const [attachmentStatusById, setAttachmentStatusById] = useState<
    Record<string, AttachmentStatusItem["status"]>
  >({});
  const selectedKindList = useMemo(
    () => STREAM_KINDS.filter((kind) => selectedKinds.has(kind)),
    [selectedKinds],
  );
  const selectedKindKey = selectedKindList.join(",");
  const serverKinds =
    selectedKindList.length === STREAM_KINDS.length ? undefined : selectedKindList;
  const serverAudience = audience === "all" ? undefined : audience;
  const invalidateAttachmentStatuses = useCallback((ids: readonly string[]) => {
    setAttachmentStatusById((current) => {
      const next = { ...current };
      for (const id of ids) {
        delete next[id];
      }
      return next;
    });
  }, []);
  const streamWindow = useStreamWindow({
    sessionId,
    kinds: serverKinds,
    audience: serverAudience,
    limit: 120,
    onAttachmentStatusesInvalidated: invalidateAttachmentStatuses,
  });
  const entries = streamWindow.entries;
  const filtered = useMemo(
    () => applyStreamStructuralFilters(entries, structuralFilters),
    [entries, structuralFilters],
  );
  const groups = useMemo(() => groupStreamEntriesByTurn(filtered), [filtered]);
  const selected =
    selectedId === null ? null : (filtered.find((entry) => entry.id === selectedId) ?? null);
  const selectedAttachmentId = selected === null ? undefined : streamEntryAttachmentId(selected);
  const attachmentApi = useApi<AttachmentMetadataResponse | null>(
    () =>
      selectedAttachmentId === undefined
        ? Promise.resolve(null)
        : getAttachmentMetadata(selectedAttachmentId).catch(() => null),
    [selectedAttachmentId],
  );

  useEffect(() => {
    setSelectedId(null);
    setSelectionSeeded(false);
    setCollapsedGroupIds(new Set());
    setAttachmentStatusById({});
  }, [audience, selectedKindKey, sessionId]);

  useEffect(() => {
    if (streamWindow.loading || selectionSeeded) {
      return;
    }
    setSelectedId(filtered[0]?.id ?? null);
    setSelectionSeeded(true);
  }, [filtered, selectionSeeded, streamWindow.loading]);

  useEffect(() => {
    if (selectedId !== null && filtered.find((entry) => entry.id === selectedId) === undefined) {
      setSelectedId(null);
    }
  }, [filtered, selectedId]);

  const visibleAttachmentIds = useMemo(
    () => [
      ...new Set(
        filtered.flatMap((entry) => {
          const id = streamEntryAttachmentId(entry);
          return id === undefined ? [] : [id];
        }),
      ),
    ],
    [filtered],
  );
  const missingAttachmentIds = useMemo(
    () => visibleAttachmentIds.filter((id) => attachmentStatusById[id] === undefined),
    [attachmentStatusById, visibleAttachmentIds],
  );
  const missingAttachmentKey = missingAttachmentIds.join(",");
  const attachmentStatusesApi = useApi<AttachmentStatusItem[]>(
    () =>
      missingAttachmentIds.length === 0
        ? Promise.resolve([])
        : getAttachmentStatuses(missingAttachmentIds),
    [missingAttachmentKey],
  );

  useEffect(() => {
    const rows = attachmentStatusesApi.data;
    if (rows === null || rows.length === 0) {
      return;
    }

    setAttachmentStatusById((current) => {
      const next = { ...current };
      for (const item of rows) {
        next[item.id] = item.status;
      }
      return next;
    });
  }, [attachmentStatusesApi.data]);

  const windowAudiences = useMemo(
    () =>
      [
        ...new Set([
          ...entries.flatMap((entry) => (entry.audience === undefined ? [] : [entry.audience])),
          ...(serverAudience === undefined ? [] : [serverAudience]),
        ]),
      ].sort(),
    [entries, serverAudience],
  );
  const kindCounts = useMemo(
    () =>
      Object.fromEntries(
        STREAM_KINDS.map((kind) => [kind, entries.filter((entry) => entry.kind === kind).length]),
      ) as Record<StreamEntryKind, number>,
    [entries],
  );
  const audienceCounts = useMemo(
    () =>
      Object.fromEntries(
        windowAudiences.map((audience) => [
          audience,
          entries.filter((entry) => entry.audience === audience).length,
        ]),
      ) as Record<string, number>,
    [entries, windowAudiences],
  );
  const structuralCounts = useMemo(
    () => ({
      aborted: entries.filter(isAbortedTurnEntry).length,
      hasAttachment: entries.filter(hasStreamAttachment).length,
      hasTurnId: entries.filter((entry) => entry.turn_id !== undefined).length,
      hasSourceMessageKey: entries.filter((entry) => entry.source_message_key !== undefined).length,
      compressed: entries.filter(isCompressed).length,
    }),
    [entries],
  );
  const selectedStatus = selected === null ? null : summarizeStatus(selected, attachmentApi.data);
  const selectedAttachmentPerceptionId =
    selected === null ? undefined : streamAttachmentPerceptionId(selected, attachmentApi.data);
  const hasOlder = streamWindow.nextCursor !== null;
  const honestyLabel = hasOlder ? "loaded window only · older entries available" : "loaded window";

  const toggleKind = (kind: StreamEntryKind) => {
    setSelectedKinds((current) => {
      if (current.has(kind) && current.size === 1) {
        return current;
      }
      const next = new Set(current);
      if (next.has(kind)) {
        next.delete(kind);
      } else {
        next.add(kind);
      }
      return next;
    });
  };
  const toggleStructuralFilter = (filterId: StreamStructuralFilterId) => {
    setStructuralFilters((current) => ({
      ...current,
      [filterId]: current[filterId] !== true,
    }));
  };
  const toggleGroup = (groupId: string) => {
    setCollapsedGroupIds((current) => {
      const next = new Set(current);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  };

  return (
    <div className="stream-screen">
      <div className="stream-filters" style={{ overflowY: "auto" }}>
        <div className="group">
          <div className="label">stream</div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--text-mute)" }}>
            append-only · {entries.length} window events · limit 120
          </div>
          <div className="stream-honesty">{honestyLabel}</div>
        </div>
        <div className="group">
          <div className="label">server filters</div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--text-mute)", marginBottom: 6 }}>
            counts are loaded window counts
          </div>
        </div>
        <div className="group">
          <div className="label">kinds</div>
          {STREAM_KINDS.map((kind) => (
            <button
              type="button"
              key={kind}
              className={`opt ${selectedKinds.has(kind) ? "on" : ""}`}
              onClick={() => toggleKind(kind)}
            >
              <span>
                <span className={`dot ${kindTag(kind)}`}></span> {kind}
              </span>
              <span className="count">{kindCounts[kind]}</span>
            </button>
          ))}
        </div>
        <div className="group">
          <div className="label">audience</div>
          <select
            aria-label="stream audience filter"
            value={audience}
            onChange={(event) => setAudience(event.currentTarget.value)}
          >
            <option value="all">all audiences ({entries.length} window)</option>
            {windowAudiences.map((item) => (
              <option key={item} value={item}>
                {item} ({audienceCounts[item] ?? 0} window)
              </option>
            ))}
          </select>
        </div>
        <div className="group">
          <div className="label">loaded-window filters</div>
          {(
            [
              ["aborted", "aborted-only", structuralCounts.aborted],
              ["hasAttachment", "has attachment", structuralCounts.hasAttachment],
              ["hasTurnId", "has turn id", structuralCounts.hasTurnId],
              ["hasSourceMessageKey", "has source key", structuralCounts.hasSourceMessageKey],
              ["compressed", "compressed", structuralCounts.compressed],
            ] as const
          ).map(([filterId, label, count]) => (
            <button
              type="button"
              key={filterId}
              className={`opt ${structuralFilters[filterId] === true ? "on" : ""}`}
              onClick={() => toggleStructuralFilter(filterId)}
            >
              <span>{label}</span>
              <span className="count">{count}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="stream-main">
        <div className="stream-main-head">
          <span>{filtered.length} window events</span>
          <span>{groups.length} groups</span>
          <span>{honestyLabel}</span>
          <span className="spacer"></span>
          <Tag kind={streamConnectionTagKind(streamWindow.wsState)} dot>
            {streamConnectionLabel(streamWindow.wsState)}
          </Tag>
        </div>
        {streamWindow.loading && entries.length === 0 ? <Loading>loading stream</Loading> : null}
        {streamWindow.error !== null ? <ErrorState>{streamWindow.error.message}</ErrorState> : null}
        {groups.length === 0 && !streamWindow.loading ? <Empty>no entries in window</Empty> : null}
        {groups.map((group) => {
          const collapsed = collapsedGroupIds.has(group.id);
          return (
            <div key={group.id} className="stream-group">
              <StreamGroupHeader group={group} collapsed={collapsed} onToggle={toggleGroup} />
              {collapsed
                ? null
                : group.entries.map((entry) => {
                    const attId = streamEntryAttachmentId(entry);
                    return (
                      <StreamTimelineRow
                        key={entry.id}
                        entry={entry}
                        selected={entry.id === selected?.id}
                        attachmentStatus={
                          attId === undefined ? undefined : attachmentStatusById[attId]
                        }
                        onSelect={setSelectedId}
                      />
                    );
                  })}
            </div>
          );
        })}
        <div className="stream-load-older">
          {hasOlder ? (
            <button
              type="button"
              className="btn sm"
              onClick={() => {
                void streamWindow.loadOlder();
              }}
              disabled={streamWindow.loadingOlder}
            >
              {streamWindow.loadingOlder ? "loading older" : "load older"}
            </button>
          ) : (
            <span className="dim">end of loaded stream window</span>
          )}
        </div>
      </div>

      <div className="stream-detail">
        {selected === null ? (
          <Empty>select a stream entry</Empty>
        ) : (
          <>
            <div className="det-head">
              <div className="id">
                <IdChip
                  id={selected.id}
                  type="stream_entry"
                  label={`[${shortId(selected.id)}]`}
                  hint={selected}
                />
              </div>
              <div className="ts">
                {new Date(selected.timestamp).toISOString()} · {selected.kind} ·{" "}
                {selected.audience ?? "global"}
              </div>
              <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
                <Tag kind={selectedStatus === "active" ? "acc" : "bad"} dot>
                  {selectedStatus}
                </Tag>
                <Tag>
                  <IdChip id={selected.session_id} type="session" />
                </Tag>
                {selected.turn_id === undefined ? null : (
                  <Tag>
                    <IdChip
                      id={selected.turn_id}
                      type="turn"
                      label={`turn ${shortId(selected.turn_id)}`}
                    />
                  </Tag>
                )}
                {selected.kind === "agent_suppressed" || selected.kind === "agent_observed" ? (
                  <Tag kind="warn">turn-action visible</Tag>
                ) : null}
              </div>
            </div>
            <div className="det-body">
              {selectedAttachmentId === undefined ? null : (
                <AttachmentDetail
                  selected={selected}
                  attachmentId={selectedAttachmentId}
                  metadata={attachmentApi.data}
                  perceptionId={selectedAttachmentPerceptionId}
                />
              )}
              <div className="upper dim" style={{ marginTop: 16, marginBottom: 6 }}>
                body
              </div>
              <pre>{streamContentText(selected.content)}</pre>
              <div className="upper dim" style={{ marginTop: 16, marginBottom: 6 }}>
                raw
              </div>
              <pre>{jsonText(selected.content)}</pre>
              <div className="upper dim" style={{ marginTop: 16, marginBottom: 6 }}>
                provenance
              </div>
              <StreamProvenanceRows entry={selected} status={selectedStatus ?? "active"} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
