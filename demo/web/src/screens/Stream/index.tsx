import { useEffect, useMemo, useRef, useState } from "react";

import { getAttachmentMetadata, getAttachmentStatuses, getStream } from "../../api/client";
import type {
  AttachmentMetadataResponse,
  AttachmentStatusItem,
  StreamEntry,
  StreamEntryKind,
} from "../../api/types";
import { ImagePlaceholder } from "../../components/ImagePlaceholder";
import { Tag, type TagKind } from "../../components/Tag";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { formatTime, mergeEntries, streamContentText } from "../../lib/stream-utils";
import { contentField, jsonText, shortId } from "../screen-utils";

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
          return (
            <div
              key={entry.id}
              className={`stream-row ${entry.id === selected?.id ? "selected " : ""}${
                isAttachment ? "kind-attachment" : ""
              }`}
              onClick={() => setSelectedId(entry.id)}
            >
              <span className="t">{formatTime(entry.timestamp)}</span>
              <span className={`k ${kindTag(entry.kind)}`}>{entry.kind}</span>
              <span className={isAttachment ? "att-inline" : "body"}>
                {isAttachment ? (
                  <>
                    <ImagePlaceholder
                      attachmentId={attId}
                      mediaType={mediaType(entry)}
                      audience={entry.audience}
                      size="xs"
                      quarantined={attachmentStatus?.quarantined === true}
                    />
                    <span className="body-txt">
                      {attId ?? "attachment"} · {streamContentText(entry.content)}
                    </span>
                    {attachmentStatus?.quarantined === true ? (
                      <Tag kind="bad">quarantined</Tag>
                    ) : null}
                  </>
                ) : (
                  streamContentText(entry.content)
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
              <div className="id">[{selected.id}]</div>
              <div className="ts">
                {new Date(selected.timestamp).toISOString()} · {selected.kind} ·{" "}
                {selected.audience ?? "global"}
              </div>
              <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap" }}>
                <Tag
                  kind={summarizeStatus(selected, attachmentApi.data) === "active" ? "acc" : "bad"}
                  dot
                >
                  {summarizeStatus(selected, attachmentApi.data)}
                </Tag>
                <Tag>{selected.session_id}</Tag>
                {selected.turn_id === undefined ? null : <Tag>turn {selected.turn_id}</Tag>}
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
                      <div className="att-card-id">{selectedAttachmentId}</div>
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
              <pre>{`source_kind:   ${selected.kind}
session:       ${selected.session_id}
turn:          ${selected.turn_id ?? "—"}
audience:      ${selected.audience ?? "global"}
sender:        ${selected.sender_entity_id ?? "—"}
reply_target:  ${selected.reply_target_entity_id ?? "—"}
status:        ${summarizeStatus(selected, attachmentApi.data)}`}</pre>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
