import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import { getAttachmentMetadata, getAttachmentStatuses, getStream } from "../../api/client";
import type {
  AttachmentMetadataResponse,
  AttachmentStatusItem,
  StreamEntry,
  StreamEntryKind,
} from "../../api/types";
import { ImagePlaceholder } from "../../components/ImagePlaceholder";
import { IdRef } from "../../components/Inspector/IdRef";
import { Tag, type TagKind } from "../../components/Tag";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { streamOutcomeSummary, type StreamOutcomeSummary } from "../../lib/stream-outcomes";
import {
  compactStreamText,
  formatTime,
  mergeEntries,
  streamContentText,
} from "../../lib/stream-utils";
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

function attachmentId(entry: StreamEntry): string | undefined {
  return contentField(entry.content, "attachment_id");
}

function attachmentStatusInvalidationIds(entries: readonly StreamEntry[]): string[] {
  return [
    ...new Set(
      entries.flatMap((entry) => {
        if (entry.kind !== "user_image_attachment" && entry.kind !== "internal_event") {
          return [];
        }

        const id = attachmentId(entry);
        return id === undefined ? [] : [id];
      }),
    ),
  ];
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
        <Tag kind="purple" dot>
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
  const id = attachmentId(entry) ?? optionalString(content.id) ?? "attachment";
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
  return (
    <div className="props">
      <div className="row">
        <span className="k">kind</span>
        <span className="v">{entry.kind}</span>
      </div>
      <div className="row">
        <span className="k">session_id</span>
        <span className="v">
          <IdRef id={entry.session_id} type="session" label={entry.session_id} />
        </span>
      </div>
      <div className="row">
        <span className="k">turn_id</span>
        <span className="v">
          {entry.turn_id === undefined ? (
            "—"
          ) : (
            <IdRef id={entry.turn_id} type="turn" label={entry.turn_id} />
          )}
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
            <IdRef id={entry.sender_entity_id} type="entity" label={entry.sender_entity_id} />
          )}
        </span>
      </div>
      <div className="row">
        <span className="k">reply_target_entity_id</span>
        <span className="v">
          {entry.reply_target_entity_id === null ? (
            "—"
          ) : (
            <IdRef
              id={entry.reply_target_entity_id}
              type="entity"
              label={entry.reply_target_entity_id}
            />
          )}
        </span>
      </div>
      <div className="row">
        <span className="k">status</span>
        <span className="v">{status}</span>
      </div>
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

export function StreamScreen({ sessionId }: { sessionId: string }) {
  const live = useLiveEventsContext();
  const streamApi = useApi(() => getStream({ session: sessionId, limit: 120 }), [sessionId]);
  const refetchStream = streamApi.refetch;
  const previousConnectionCountRef = useRef(live.connectionCount);
  const [entries, setEntries] = useState<StreamEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [kinds, setKinds] = useState<Set<StreamEntryKind>>(() => new Set(STREAM_KINDS));
  const [audiences, setAudiences] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    const streamData = streamApi.data;

    if (streamData === null) {
      return;
    }
    setEntries((current) => mergeEntries(current, streamData.entries, "desc"));
    setSelectedId(
      (current) => current ?? mergeEntries([], streamData.entries, "desc")[0]?.id ?? null,
    );
  }, [streamApi.data]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type !== "stream:append") {
        return;
      }
      const matchingEntries = frame.entries.filter((entry) => entry.session_id === sessionId);
      if (matchingEntries.length === 0) {
        return;
      }
      setEntries((current) => mergeEntries(current, matchingEntries, "desc"));
      setSelectedId((current) => current ?? matchingEntries.at(-1)?.id ?? null);
      const invalidatedAttachmentIds = attachmentStatusInvalidationIds(matchingEntries);
      if (invalidatedAttachmentIds.length > 0) {
        setAttachmentStatusById((current) => {
          const next = { ...current };
          for (const id of invalidatedAttachmentIds) {
            delete next[id];
          }
          return next;
        });
      }
    });
  }, [live, sessionId]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    void refetchStream();
  }, [live.connectionCount, refetchStream]);

  const windowAudiences = useMemo(
    () =>
      [
        ...new Set(
          entries.flatMap((entry) => (entry.audience === undefined ? [] : [entry.audience])),
        ),
      ].sort(),
    [entries],
  );

  useEffect(() => {
    setAudiences((current) => {
      if (current.size > 0 || windowAudiences.length === 0) {
        return current;
      }
      return new Set(windowAudiences);
    });
  }, [windowAudiences]);

  const filtered = useMemo(
    () =>
      entries.filter(
        (entry) =>
          kinds.has(entry.kind) &&
          (entry.audience === undefined || audiences.size === 0 || audiences.has(entry.audience)),
      ),
    [audiences, entries, kinds],
  );
  const selected = filtered.find((entry) => entry.id === selectedId) ?? filtered[0] ?? null;
  const selectedAttachmentId = selected === null ? undefined : attachmentId(selected);
  const attachmentApi = useApi<AttachmentMetadataResponse | null>(
    () =>
      selectedAttachmentId === undefined
        ? Promise.resolve(null)
        : getAttachmentMetadata(selectedAttachmentId).catch(() => null),
    [selectedAttachmentId],
  );
  const [attachmentStatusById, setAttachmentStatusById] = useState<
    Record<string, AttachmentStatusItem["status"]>
  >({});

  useEffect(() => {
    setEntries([]);
    setSelectedId(null);
    setAudiences(new Set());
    setAttachmentStatusById({});
  }, [sessionId]);
  const visibleAttachmentIds = useMemo(
    () => [
      ...new Set(
        filtered.flatMap((entry) => {
          const id = attachmentId(entry);
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
  const selectedStatus = selected === null ? null : summarizeStatus(selected, attachmentApi.data);
  const selectedAttachmentPerceptionId =
    selected === null ? undefined : streamAttachmentPerceptionId(selected, attachmentApi.data);

  const toggleKind = (kind: StreamEntryKind) => {
    setKinds((current) => {
      const next = new Set(current);
      if (next.has(kind)) {
        next.delete(kind);
      } else {
        next.add(kind);
      }
      return next;
    });
  };
  const toggleAudience = (audience: string) => {
    setAudiences((current) => {
      const next = new Set(current);
      if (next.has(audience)) {
        next.delete(audience);
      } else {
        next.add(audience);
      }
      return next;
    });
  };

  return (
    <div className="stream-screen">
      <div className="stream-filters" style={{ overflowY: "auto" }}>
        <div className="group">
          <div className="label">stream</div>
          <div style={{ fontSize: 10.5, color: "var(--text-mute)" }}>
            append-only · {entries.length} events · window counts
          </div>
        </div>
        <div className="group">
          <div className="label">kinds</div>
          {STREAM_KINDS.map((kind) => (
            <div
              key={kind}
              className={`opt ${kinds.has(kind) ? "on" : ""}`}
              onClick={() => toggleKind(kind)}
            >
              <span>
                <span className={`dot ${kindTag(kind)}`}></span> {kind}
              </span>
              <span className="count">{kindCounts[kind]}</span>
            </div>
          ))}
        </div>
        <div className="group">
          <div className="label">audience</div>
          {windowAudiences.map((audience) => (
            <div
              key={audience}
              className={`opt ${audiences.has(audience) ? "on" : ""}`}
              onClick={() => toggleAudience(audience)}
            >
              <span>{audience}</span>
              <span className="count">{audienceCounts[audience]}</span>
            </div>
          ))}
          {windowAudiences.length === 0 ? (
            <div className="opt readonly">
              <span style={{ color: "var(--text-faint)" }}>none in window</span>
            </div>
          ) : null}
        </div>
        <div className="group">
          <div className="label">status</div>
          <div className="opt readonly">
            <span>active</span>
            <span className="count">
              {entries.filter((entry) => entry.turn_status !== "aborted").length}
            </span>
          </div>
          <div className="opt readonly">
            <span>aborted-turn</span>
            <span className="count">
              {entries.filter((entry) => entry.turn_status === "aborted").length}
            </span>
          </div>
        </div>
      </div>

      <div className="stream-main">
        <div className="stream-main-head">
          <span>{filtered.length} events</span>
          <span className="spacer"></span>
          <span className="live-dot"></span>
          <span className="acc upper">tailing</span>
        </div>
        {streamApi.loading && entries.length === 0 ? (
          <div className="notice">loading stream</div>
        ) : null}
        {streamApi.error !== null ? (
          <div className="notice bad">{streamApi.error.message}</div>
        ) : null}
        {filtered.map((entry) => {
          const attId = attachmentId(entry);
          const attachmentStatus = attId === undefined ? undefined : attachmentStatusById[attId];
          const isAttachment = entry.kind === "user_image_attachment";
          const outcomeSummary = streamOutcomeSummary(entry);
          const wake = autonomousWakeInfo(entry);
          return (
            <div
              key={entry.id}
              className={`stream-row ${entry.id === selected?.id ? "selected " : ""}${
                isAttachment ? "kind-attachment" : ""
              }`}
              onClick={() => setSelectedId(entry.id)}
            >
              <span className="t">{formatTime(entry.timestamp)}</span>
              <span className={`k ${wake !== null ? "purple" : kindTag(entry.kind)}`}>
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
                    {attachmentStatus?.quarantined === true ? (
                      <Tag kind="bad">quarantined</Tag>
                    ) : null}
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
        })}
      </div>

      <div className="stream-detail">
        {selected === null ? null : (
          <>
            <div className="det-head">
              <div className="id">
                <IdRef
                  id={selected.id}
                  type="stream_entry"
                  label={`[${selected.id}]`}
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
                  <IdRef id={selected.session_id} type="session" label={selected.session_id} />
                </Tag>
                {selected.turn_id === undefined ? null : (
                  <Tag>
                    <IdRef id={selected.turn_id} type="turn" label={`turn ${selected.turn_id}`} />
                  </Tag>
                )}
                {selected.kind === "agent_suppressed" || selected.kind === "agent_observed" ? (
                  <Tag kind="warn">turn-action visible</Tag>
                ) : null}
              </div>
            </div>
            <div className="det-body">
              {selectedAttachmentId === undefined ? null : (
                <>
                  <div className="upper dim" style={{ marginBottom: 6 }}>
                    attachment
                  </div>
                  <div className="att-card">
                    <ImagePlaceholder
                      attachmentId={selectedAttachmentId}
                      mediaType={mediaType(selected)}
                      audience={selected.audience}
                      size="lg"
                      quarantined={attachmentApi.data?.status.quarantined === true}
                    />
                    <div className="att-card-meta">
                      <div className="att-card-id">
                        <IdRef
                          id={selectedAttachmentId}
                          type="attachment"
                          label={selectedAttachmentId}
                          hint={attachmentApi.data?.attachment}
                        />
                        {selectedAttachmentPerceptionId === undefined ? null : (
                          <IdRef
                            id={selectedAttachmentPerceptionId}
                            type="image_perception"
                            label={`perception ${selectedAttachmentPerceptionId}`}
                            hint={attachmentApi.data?.perception}
                          />
                        )}
                      </div>
                      <div className="att-card-caption">
                        {attachmentApi.data?.perception?.caption ?? "perception unavailable"}
                      </div>
                      <div className="att-card-stats">
                        <span>
                          {attachmentApi.data?.attachment.width ?? "?"}x
                          {attachmentApi.data?.attachment.height ?? "?"}
                        </span>
                        <span>{shortId(attachmentApi.data?.attachment.sha256)}</span>
                        <span>
                          {attachmentApi.data?.status.quarantined === true
                            ? "quarantined cascade"
                            : "active"}
                        </span>
                      </div>
                    </div>
                  </div>
                </>
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
