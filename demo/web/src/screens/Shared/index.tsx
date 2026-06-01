import { useMemo, useState } from "react";

import { getSharedState, getState } from "../../api/client";
import type { SharedStateEntry, SharedStateEntryKind } from "../../api/types";
import { Tag, type TagKind } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { dateLabel, shortId } from "../screen-utils";

type PrimaryLifecycleKind = "locked" | "live" | "tentative" | "invalidated";
type LifecycleFilter = PrimaryLifecycleKind | "all";

const LIFECYCLE: PrimaryLifecycleKind[] = [
  "locked",
  "live",
  "tentative",
  "invalidated"
];

function primaryLifecycleKind(kind: SharedStateEntryKind): PrimaryLifecycleKind {
  if (kind === "low_salience_live" || kind === "dormant_live") {
    return "live";
  }
  if (kind === "pending") {
    return "tentative";
  }
  return kind;
}

function lifecycleLabel(kind: SharedStateEntryKind | PrimaryLifecycleKind): string {
  if (kind === "low_salience_live") {
    return "live - low salience";
  }
  if (kind === "dormant_live") {
    return "live - dormant";
  }
  if (kind === "pending") {
    return "pending (legacy)";
  }
  return kind;
}

function cssState(kind: SharedStateEntryKind): string {
  if (kind === "dormant_live" || kind === "low_salience_live") {
    return "live";
  }
  if (kind === "pending") {
    return "tentative";
  }
  return kind;
}

function lifecycleColor(kind: PrimaryLifecycleKind): string {
  if (kind === "locked") {
    return "var(--acc)";
  }
  if (kind === "live") {
    return "var(--info)";
  }
  if (kind === "tentative") {
    return "var(--warn)";
  }
  if (kind === "invalidated") {
    return "var(--bad)";
  }
  return "var(--text-mute)";
}

function tagKind(kind: SharedStateEntryKind): TagKind {
  if (kind === "locked") {
    return "acc";
  }
  if (kind === "live" || kind === "low_salience_live" || kind === "dormant_live") {
    return "info";
  }
  if (kind === "tentative" || kind === "pending") {
    return "warn";
  }
  if (kind === "invalidated") {
    return "bad";
  }
  return "";
}

export function SharedScreen({ sessionId }: { sessionId: string }) {
  // Audience options derive from the SELECTED session's stream (like every other
  // session-scoped screen). Without this the screen was pinned to the default
  // session and only ever offered default/self, hiding entries compiled under
  // other audiences (e.g. the operator/Tom audience).
  const stateApi = useApi(() => getState({ session: sessionId }), [sessionId]);
  const audiences = stateApi.data?.audiences ?? [];
  const [audience, setAudience] = useState<string | null>(null);
  // Fall back to the session's first audience when no valid manual pick is held
  // (a manual pick from a previous session may not exist in this one).
  const selectedAudience =
    audience !== null && audiences.includes(audience) ? audience : (audiences[0] ?? "self");
  const sharedApi = useApi(() => getSharedState(selectedAudience), [selectedAudience]);
  const [filter, setFilter] = useState<LifecycleFilter>("all");
  const entries = sharedApi.data?.entries ?? [];
  const filtered =
    filter === "all" ? entries : entries.filter((entry) => primaryLifecycleKind(entry.kind) === filter);
  const counts = useMemo(
    () => {
      const next = Object.fromEntries(LIFECYCLE.map((kind) => [kind, 0])) as Record<
        PrimaryLifecycleKind,
        number
      >;
      for (const entry of entries) {
        next[primaryLifecycleKind(entry.kind)] += 1;
      }
      return next;
    },
    [entries]
  );
  const lastCompile = entries.length === 0 ? null : Math.max(...entries.map((entry) => entry.last_updated_at));

  if ((stateApi.loading || sharedApi.loading) && stateApi.data === null && sharedApi.data === null) {
    return <div className="notice">loading shared state</div>;
  }

  if (stateApi.error !== null || sharedApi.error !== null) {
    return <div className="notice bad">{stateApi.error?.message ?? sharedApi.error?.message}</div>;
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <h1>shared state</h1>
        <span className="desc">per-audience compiled lifecycle view</span>
        <span className="spacer"></span>
        <div className="filter-pills">
          {(audiences.length === 0 ? ["self"] : audiences).map((value) => (
            <span
              key={value}
              className={`pill ${selectedAudience === value ? "on" : ""}`}
              onClick={() => setAudience(value)}
            >
              {value}
            </span>
          ))}
        </div>
      </div>

      <div
        style={{
          padding: "10px 20px",
          borderBottom: "1px solid var(--line)",
          background: "var(--bg-0)",
          display: "flex",
          gap: 18,
          alignItems: "center",
          flexWrap: "wrap"
        }}
      >
        <div className="filter-pills">
          {(["all", ...LIFECYCLE] as const).map((kind) => (
            <span
              key={kind}
              className={`pill ${filter === kind ? "on" : ""}`}
              style={kind !== "all" && filter !== kind ? { color: lifecycleColor(kind) } : undefined}
              onClick={() => setFilter(kind)}
            >
              {kind === "all" ? kind : lifecycleLabel(kind)}
              {kind === "all" ? "" : ` ${counts[kind] ?? 0}`}
            </span>
          ))}
        </div>
        <div style={{ flex: 1 }}></div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
          <span style={{ fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--text-mute)" }}>
            last compile
          </span>
          <span style={{ fontSize: 12, color: "var(--text)", fontVariantNumeric: "tabular-nums" }}>
            {dateLabel(lastCompile)}
          </span>
        </div>
      </div>

      <div className="shared">
        {filtered.length === 0 ? (
          <div className="notice">
            no {filter === "all" ? "" : `${lifecycleLabel(filter)} `}entries for audience '{selectedAudience}'
          </div>
        ) : null}
        {filtered.map((entry) => (
          <SharedEntryCard key={entry.id} entry={entry} audience={selectedAudience} />
        ))}
      </div>
    </div>
  );
}

function SharedEntryCard({ entry, audience }: { entry: SharedStateEntry; audience: string }) {
  const canonical = [
    ...entry.canonicalizes.goal_ids,
    ...entry.canonicalizes.commitment_ids,
    ...entry.canonicalizes.action_ids,
    ...entry.canonicalizes.open_question_ids
  ];

  return (
    <div className={`ss-entry ${cssState(entry.kind)}`}>
      <div className="h">
        <Tag kind={tagKind(entry.kind)} dot>
          {lifecycleLabel(entry.kind)}
        </Tag>
        <span className="id">[{shortId(entry.id)}]</span>
        <span style={{ flex: 1 }}></span>
        <span className="dim tab-num" style={{ fontSize: 10 }}>
          {entry.last_updated_turn_global === null ? "" : `turn ${entry.last_updated_turn_global} · `}
          {entry.provenance_stream_entry_ids.length} src
        </span>
      </div>
      <div className="text">{entry.text}</div>
      <div className="meta">
        {canonical.length === 0 ? null : (
          <span>
            canon → <span className="acc">{canonical.map(shortId).join(", ")}</span>
          </span>
        )}
        {entry.superseded_by_id === null ? null : (
          <span>
            superseded by <span className="info">{shortId(entry.superseded_by_id)}</span>
          </span>
        )}
        <span>
          audience <span className="acc">{audience}</span>
        </span>
        <span>updated {dateLabel(entry.last_updated_at)}</span>
      </div>
    </div>
  );
}
