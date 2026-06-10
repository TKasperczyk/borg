import { useEffect, useMemo, useState } from "react";

import {
  getMemoryBand,
  getMemoryBands,
  getReviews,
  getSemanticGraph,
  getSemanticNode,
  postCorrectionCorrect,
  postCorrectionForget,
  postSemanticEdgeInvalidate,
} from "../../api/client";
import type {
  CommitmentItem,
  EpisodeMemoryItem,
  MemoryBandDetail,
  MemoryBandId,
  MemoryBandSummary,
  ProceduralMemoryItem,
  RelationalMemoryItem,
  SemanticGraphResponse,
  SemanticMemoryNode,
  SocialMemoryItem,
} from "../../api/types";
import { Empty } from "../../components/Empty";
import { ErrorState } from "../../components/ErrorState";
import { IdChip } from "../../components/Inspector/IdChip";
import { Loading } from "../../components/Loading";
import { Modal } from "../../components/Modal";
import { Panel } from "../../components/Panel";
import { SemanticNodeDetail } from "../../components/SemanticNodeDetail";
import { Tag } from "../../components/Tag";
import { WhyDrawer } from "../../components/WhyDrawer";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { activateOnEnterOrSpace } from "../../lib/keyboard";
import { formatTimestamp, formatTimestampRange } from "../../lib/stream-utils";
import { isInternalId, jsonText, parseJsonPatch, shortId } from "../screen-utils";
import { SocialTrustScatter, ValenceArousalPlane } from "./AtlasPlots";
import { SemanticTopology } from "./SemanticTopology";

const BAND_ORDER: MemoryBandId[] = [
  "episodic",
  "semantic",
  "procedural",
  "affective",
  "self",
  "commitments",
  "social",
  "relational",
];

const BAND_DESCRIPTIONS: Record<MemoryBandId, string> = {
  episodic: "what happened",
  semantic: "what Borg believes",
  procedural: "how Borg solves things",
  affective: "mood and trajectory",
  self: "values, goals, traits, narrative",
  commitments: "scoped promises and boundaries",
  social: "per-entity trust and history",
  relational: "evidence-backed relationship facts",
};

const MEMORY_PAGE_LIMIT = 50;
const SEMANTIC_TOPOLOGY_LIMIT = 300;
const SEARCHABLE_BANDS = new Set<MemoryBandId>(["episodic", "semantic", "procedural"]);

type SortMode =
  | "backend"
  | "updated_desc"
  | "updated_asc"
  | "created_desc"
  | "created_asc"
  | "audience_group";
type SemanticViewMode = "browser" | "topology";

type MemoryRow = {
  id: string;
  title: string;
  meta: string;
  body: string;
  order: number;
  rowKind: string;
  audience?: string;
  tags?: string[];
  kind?: string;
  status?: string;
  state?: string;
  enforcement?: string;
  contradictedCount?: number;
  alternateCount?: number;
  updatedAt?: number;
  createdAt?: number;
  startTime?: number;
  searchScore?: number;
};

type MemoryCorrectionAction =
  | { kind: "forget"; id: string; title: string }
  | { kind: "correct"; id: string; title: string; patch: string; reason: string }
  | { kind: "invalidate-edge"; id: string; title: string; reason: string; at: string };

function correctionActionKind(id: string): "episode" | "semantic_node" | "semantic_edge" | null {
  if (id.startsWith("ep_")) {
    return "episode";
  }
  if (id.startsWith("semn_")) {
    return "semantic_node";
  }
  if (id.startsWith("seme_")) {
    return "semantic_edge";
  }
  return null;
}

function defaultMemoryPatch(
  row: { title: string; body: string },
  kind: NonNullable<ReturnType<typeof correctionActionKind>>,
): string {
  if (kind === "episode") {
    return JSON.stringify(
      {
        title: row.title,
        narrative: row.body,
      },
      null,
      2,
    );
  }

  if (kind === "semantic_node") {
    return JSON.stringify(
      {
        label: row.title,
        description: row.body,
      },
      null,
      2,
    );
  }

  return "{}";
}

function scoreMeta(score: number | undefined): string {
  return score === undefined ? "" : `score ${score.toFixed(2)} · `;
}

function semanticNodeRow(node: SemanticMemoryNode, order: number): MemoryRow {
  const title =
    node.display_label ?? (isInternalId(node.label) ? `${node.kind} memory` : node.label);

  return {
    id: node.id,
    title,
    meta: `${scoreMeta(node.search_score)}${node.kind} · ${node.status} · ${node.source_count} src`,
    body: node.description,
    order,
    rowKind: "node",
    kind: node.kind,
    status: node.status,
    createdAt: node.created_at,
    updatedAt: node.updated_at,
    searchScore: node.search_score,
  };
}

function detailRows(detail: MemoryBandDetail): MemoryRow[] {
  switch (detail.band) {
    case "episodic":
      return detail.items.map((item, order) => ({
        id: item.id,
        title: item.title,
        meta: `${scoreMeta(item.search_score)}${formatTimestamp(item.start_time)} · ${item.audience ?? "global"} · ${item.source_count} src`,
        body: item.narrative,
        order,
        rowKind: "episode",
        audience: item.audience ?? "global",
        tags: item.tags,
        createdAt: item.created_at,
        updatedAt: item.updated_at,
        startTime: item.start_time,
        searchScore: item.search_score,
      }));
    case "semantic":
      return [
        ...detail.nodes.map((node, order) => semanticNodeRow(node, order)),
        ...detail.edges.map((edge, index) => ({
          id: edge.id,
          title: `${edge.from_node_id} --${edge.relation}-> ${edge.to_node_id}`,
          meta: `edge · confidence ${edge.confidence.toFixed(2)} · ${edge.source_count} src`,
          body: edge.invalidated_reason ?? "active edge",
          order: detail.nodes.length + index,
          rowKind: "edge",
          kind: "edge",
          status: edge.invalidated_at === null ? "active" : "invalidated",
          createdAt: edge.valid_from,
          updatedAt: edge.invalidated_at ?? edge.valid_from,
        })),
      ];
    case "procedural":
      return detail.items.map((skill, order) => ({
        id: skill.id,
        title: skill.applies_when,
        meta: `${scoreMeta(skill.search_score)}${skill.status} · alpha ${skill.alpha.toFixed(1)} · beta ${skill.beta.toFixed(1)} · ${skill.sample_count} samples`,
        body: skill.approach,
        order,
        rowKind: "skill",
        status: skill.status,
        createdAt: skill.created_at,
        updatedAt: skill.updated_at,
        searchScore: skill.search_score,
      }));
    case "affective":
      return detail.history.map((point, order) => ({
        id: String(point.id),
        title: `${formatTimestamp(point.ts)} · valence ${point.valence.toFixed(2)}`,
        meta: `arousal ${point.arousal.toFixed(2)} · ${point.trigger_reason ?? "no trigger"}`,
        body: jsonText(point.provenance),
        order,
        rowKind: "mood",
        createdAt: point.ts,
        updatedAt: point.ts,
      }));
    case "self":
      return [
        ...detail.values.map((value, order) => ({
          id: value.id,
          title: value.label,
          meta: `value · confidence ${value.confidence.toFixed(2)}`,
          body: value.description,
          order,
          rowKind: "value",
          state: value.state,
          createdAt: value.created_at,
          updatedAt: value.last_affirmed ?? value.created_at,
        })),
        ...detail.goals.map((goal, index) => ({
          id: goal.id,
          title: goal.description,
          meta: `goal · ${goal.status} · priority ${goal.priority.toFixed(2)}`,
          body: goal.progress_notes ?? "no progress notes",
          order: detail.values.length + index,
          rowKind: "goal",
          status: goal.status,
          createdAt: goal.created_at,
          updatedAt: goal.created_at,
        })),
        ...detail.traits.map((trait, index) => ({
          id: trait.id,
          title: trait.label,
          meta: `trait · confidence ${trait.confidence.toFixed(2)}`,
          body: `${trait.support_count} support · ${trait.contradiction_count} contradiction`,
          order: detail.values.length + detail.goals.length + index,
          rowKind: "trait",
          state: trait.state,
        })),
        ...detail.open_questions.map((question, index) => ({
          id: question.id,
          title: question.question,
          meta: `open_question · ${question.status} · urgency ${question.urgency.toFixed(2)}`,
          body: question.resolution_note ?? question.abandoned_reason ?? "unresolved",
          order: detail.values.length + detail.goals.length + detail.traits.length + index,
          rowKind: "open_question",
          status: question.status,
          createdAt: question.created_at,
          updatedAt: question.last_touched,
        })),
      ];
    case "commitments":
      return detail.items.map((commitment, order) => ({
        id: commitment.id,
        title: commitment.text,
        meta: `${commitment.state} · ${commitment.enforcement_class} · ${commitment.audience ?? "global"}`,
        body: `${commitment.type} · ${commitment.kind}`,
        order,
        rowKind: "commitment",
        audience: commitment.audience ?? "global",
        status: commitment.state,
        state: commitment.state,
        enforcement: commitment.enforcement_class,
        createdAt: commitment.created_at,
        updatedAt: commitment.last_reinforced_at,
      }));
    case "social":
      return detail.items.map((profile, order) => ({
        id: profile.entity_id,
        title: profile.name ?? profile.entity_id,
        meta: `trust ${profile.trust.toFixed(2)} · ${profile.history_count} interactions`,
        body: `attachment ${profile.attachment.toFixed(2)} · commitments ${profile.commitment_count}`,
        order,
        rowKind: "profile",
        updatedAt: profile.updated_at,
        createdAt: profile.updated_at,
      }));
    case "relational":
      return detail.items.map((slot, order) => ({
        id: slot.id,
        title: slot.slot,
        meta: `${slot.state} · ${slot.sources_count} src · ${slot.alternate_count} alternates`,
        body: slot.value,
        order,
        rowKind: "slot",
        state: slot.state,
        status: slot.state,
        contradictedCount: slot.contradicted_count,
        alternateCount: slot.alternate_count,
        createdAt: slot.created_at,
        updatedAt: slot.updated_at,
      }));
  }
}

export function MemoryScreen({
  sessionId,
  onOpenWorkbench,
  onOpenReview,
  onOpenIdentity,
  onOpenCommitments,
}: {
  sessionId: string;
  onOpenWorkbench?: () => void;
  onOpenReview?: () => void;
  onOpenIdentity?: () => void;
  onOpenCommitments?: () => void;
}) {
  const api = useApi(() => getMemoryBands({ session: sessionId }), [sessionId]);
  const [activeBand, setActiveBand] = useState<MemoryBandId | null>(null);
  const bands = BAND_ORDER.map(
    (id, index) =>
      api.data?.bands.find((band) => band.id === id) ?? {
        id,
        n: String(index + 1).padStart(2, "0"),
        name: id,
        desc: BAND_DESCRIPTIONS[id],
        count: 0,
        count_is_lower_bound: false,
        stats: [],
      },
  );
  const totalMemories = bands.reduce((total, band) => total + band.count, 0);

  if (activeBand !== null) {
    const activeSummary = bands.find((item) => item.id === activeBand);
    return (
      <MemoryDrill
        band={activeBand}
        bands={bands}
        totalCount={activeSummary?.count ?? 0}
        totalIsLowerBound={activeSummary?.count_is_lower_bound ?? false}
        sessionId={sessionId}
        back={() => setActiveBand(null)}
        onSelectBand={setActiveBand}
        onOpenIdentity={onOpenIdentity}
        onOpenCommitments={onOpenCommitments}
        onMemoryChanged={api.refetch}
      />
    );
  }

  if (api.loading && api.data === null) {
    return <Loading>loading memory bands</Loading>;
  }

  if (api.error !== null) {
    return <ErrorState onRetry={api.refetch}>{api.error.message}</ErrorState>;
  }

  return (
    <div className="bands memory-atlas">
      <div className="bands-head">
        <h1>memory atlas</h1>
        <div className="desc">
          raw memory store browser · audience scoping applies during retrieval/evidence ledger
        </div>
      </div>
      <BandOverviewBar bands={bands} activeBand={null} onSelectBand={setActiveBand} />
      {api.data !== null && totalMemories === 0 ? (
        <div className="memory-zero-state">
          <span>no memories yet -- memory forms automatically as turns are ingested</span>
          {onOpenWorkbench === undefined ? null : (
            <button type="button" className="btn sm ghost" onClick={onOpenWorkbench}>
              open workbench
            </button>
          )}
        </div>
      ) : null}
      <div className="divider" style={{ marginTop: 22 }}>
        governance
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <Panel title="identity governance" badge="guards">
          <div style={{ padding: 12 }}>
            <div className="panel-note">
              identity-bearing writes are guarded by provenance, confidence, and review routing.
            </div>
            <div className="props">
              <div className="row">
                <span className="k">bands</span>
                <span className="v">8</span>
              </div>
              <div className="row">
                <span className="k">review hint</span>
                <span className="v">see review for open queue rows</span>
              </div>
            </div>
          </div>
        </Panel>
        <Panel title="correction reviews" badge="open">
          <CorrectionReviewSummaryPanel onOpenReview={onOpenReview} />
        </Panel>
      </div>
    </div>
  );
}

function BandOverviewBar({
  bands,
  activeBand,
  onSelectBand,
}: {
  bands: readonly MemoryBandSummary[];
  activeBand: MemoryBandId | null;
  onSelectBand: (band: MemoryBandId) => void;
}) {
  return (
    <div className="band-overview-bar" aria-label="memory bands">
      {bands.map((band) => (
        <BandCard
          key={band.id}
          band={band}
          active={activeBand === band.id}
          onClick={() => onSelectBand(band.id)}
        />
      ))}
    </div>
  );
}

function BandCard({
  band,
  active,
  onClick,
}: {
  band: MemoryBandSummary;
  active: boolean;
  onClick: () => void;
}) {
  const countLabel = `${band.count_is_lower_bound ? "≥" : ""}${band.count.toLocaleString()}`;

  return (
    <button
      type="button"
      className={`band-card ${active ? "active" : ""}`}
      aria-label={`open ${band.name} memory band`}
      onClick={onClick}
    >
      <div className="head">
        <span>band {band.n ?? "—"}</span>
        <span className="n">{countLabel}</span>
      </div>
      <div className="name">{band.name}</div>
      <div className="stat-row">
        {band.stats.slice(0, 2).map((stat) => (
          <div key={stat.k} className="stat">
            <div className="k">{stat.k}</div>
            <div className="v">{stat.v}</div>
          </div>
        ))}
      </div>
    </button>
  );
}

function detailNextCursor(detail: MemoryBandDetail | null): string | null {
  if (detail === null) {
    return null;
  }
  if (detail.band === "episodic" || detail.band === "semantic") {
    return detail.next_cursor;
  }
  return null;
}

function loadedBandCount(detail: MemoryBandDetail | null, rows: readonly MemoryRow[]): number {
  if (detail === null) {
    return 0;
  }
  if (detail.band === "episodic") {
    return detail.items.length;
  }
  if (detail.band === "semantic") {
    return detail.nodes.length;
  }
  return rows.length;
}

function mergeMemoryDetail(
  current: MemoryBandDetail | null,
  next: MemoryBandDetail,
): MemoryBandDetail {
  if (current === null || current.band !== next.band || current.mode === "search") {
    return next;
  }

  if (current.band === "episodic" && next.band === "episodic") {
    return {
      ...next,
      items: [...current.items, ...next.items],
    };
  }

  if (current.band === "semantic" && next.band === "semantic") {
    return {
      ...next,
      nodes: [...current.nodes, ...next.nodes],
      edges: current.edges,
    };
  }

  return next;
}

function uniqueSorted(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => value !== undefined))].sort();
}

function structuralFilterRows(
  rows: readonly MemoryRow[],
  filters: Readonly<Record<string, string>>,
): MemoryRow[] {
  const rowKind = filters.rowKind ?? "all";
  const tag = filters.tag ?? "all";
  const kind = filters.kind ?? "all";
  const status = filters.status ?? "all";
  const state = filters.state ?? "all";
  const enforcement = filters.enforcement ?? "all";
  const conflict = filters.conflict ?? "all";

  return rows.filter((row) => {
    if (rowKind !== "all" && row.rowKind !== rowKind) {
      return false;
    }
    if (tag !== "all" && !row.tags?.includes(tag)) {
      return false;
    }
    if (kind !== "all" && row.kind !== kind) {
      return false;
    }
    if (status !== "all" && row.status !== status) {
      return false;
    }
    if (state !== "all" && row.state !== state) {
      return false;
    }
    if (enforcement !== "all" && row.enforcement !== enforcement) {
      return false;
    }
    if (
      conflict === "conflicts" &&
      row.state === "established" &&
      (row.contradictedCount ?? 0) === 0 &&
      (row.alternateCount ?? 0) === 0
    ) {
      return false;
    }
    return true;
  });
}

function rowSortTime(row: MemoryRow): number {
  return row.updatedAt ?? row.startTime ?? row.createdAt ?? 0;
}

function sortedRows(rows: readonly MemoryRow[], sortMode: SortMode): MemoryRow[] {
  const next = [...rows];
  if (sortMode === "backend") {
    return next.sort((left, right) => left.order - right.order);
  }
  if (sortMode === "updated_desc") {
    return next.sort(
      (left, right) => rowSortTime(right) - rowSortTime(left) || left.order - right.order,
    );
  }
  if (sortMode === "updated_asc") {
    return next.sort(
      (left, right) => rowSortTime(left) - rowSortTime(right) || left.order - right.order,
    );
  }
  if (sortMode === "created_desc") {
    return next.sort(
      (left, right) => (right.createdAt ?? 0) - (left.createdAt ?? 0) || left.order - right.order,
    );
  }
  if (sortMode === "audience_group") {
    return next.sort((left, right) => {
      const audienceCompare = (left.audience ?? "global").localeCompare(right.audience ?? "global");
      return audienceCompare || left.order - right.order;
    });
  }
  return next.sort(
    (left, right) => (left.createdAt ?? 0) - (right.createdAt ?? 0) || left.order - right.order,
  );
}

function defaultFiltersForBand(_band: MemoryBandId): Record<string, string> {
  return {
    conflict: "all",
    enforcement: "all",
    kind: "all",
    rowKind: "all",
    state: "all",
    status: "all",
    tag: "all",
  };
}

function sortLabel(sort: SortMode): string {
  if (sort === "backend") {
    return "backend";
  }
  if (sort === "updated_desc") {
    return "newest";
  }
  if (sort === "updated_asc") {
    return "oldest";
  }
  if (sort === "created_desc") {
    return "created new";
  }
  if (sort === "audience_group") {
    return "group audience";
  }
  return "created old";
}

function FilterPillGroup({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: string[];
  value: string;
  onChange: (value: string) => void;
}) {
  if (options.length <= 1) {
    return null;
  }

  // Many options (e.g. episodic tags can be 100+) would render as an enormous
  // pill list and blow the page height -- collapse those into a dropdown.
  if (options.length > 8) {
    return (
      <>
        <span className="dim" style={{ fontSize: "var(--fs-xs)" }}>
          {label}
        </span>
        <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </>
    );
  }

  return (
    <>
      <span className="dim" style={{ fontSize: "var(--fs-xs)" }}>
        {label}
      </span>
      <div className="filter-pills">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            className={`pill ${value === option ? "on" : ""}`}
            onClick={() => onChange(option)}
          >
            {option}
          </button>
        ))}
      </div>
    </>
  );
}

function MemoryStructuralControls({
  band,
  rows,
  filters,
  onFilter,
}: {
  band: MemoryBandId;
  rows: readonly MemoryRow[];
  filters: Record<string, string>;
  onFilter: (key: string, value: string) => void;
}) {
  const rowKinds = ["all", ...uniqueSorted(rows.map((row) => row.rowKind))];
  const tags = ["all", ...uniqueSorted(rows.flatMap((row) => row.tags ?? []))];
  const kinds = ["all", ...uniqueSorted(rows.map((row) => row.kind))];
  const statuses = ["all", ...uniqueSorted(rows.map((row) => row.status))];
  const states = ["all", ...uniqueSorted(rows.map((row) => row.state))];
  const enforcement = ["all", ...uniqueSorted(rows.map((row) => row.enforcement))];
  const selected = (key: string) => filters[key] ?? "all";

  if (band === "episodic") {
    return (
      <FilterPillGroup
        label="tag"
        options={tags}
        value={selected("tag")}
        onChange={(value) => onFilter("tag", value)}
      />
    );
  }

  if (band === "semantic") {
    return (
      <>
        <FilterPillGroup
          label="kind"
          options={kinds}
          value={selected("kind")}
          onChange={(value) => onFilter("kind", value)}
        />
        <FilterPillGroup
          label="status"
          options={statuses}
          value={selected("status")}
          onChange={(value) => onFilter("status", value)}
        />
      </>
    );
  }

  if (band === "self") {
    return (
      <>
        <FilterPillGroup
          label="type"
          options={rowKinds}
          value={selected("rowKind")}
          onChange={(value) => onFilter("rowKind", value)}
        />
        <FilterPillGroup
          label="status"
          options={statuses}
          value={selected("status")}
          onChange={(value) => onFilter("status", value)}
        />
        <FilterPillGroup
          label="state"
          options={states}
          value={selected("state")}
          onChange={(value) => onFilter("state", value)}
        />
      </>
    );
  }

  if (band === "commitments") {
    return (
      <>
        <FilterPillGroup
          label="state"
          options={states}
          value={selected("state")}
          onChange={(value) => onFilter("state", value)}
        />
        <FilterPillGroup
          label="enforce"
          options={enforcement}
          value={selected("enforcement")}
          onChange={(value) => onFilter("enforcement", value)}
        />
      </>
    );
  }

  if (band === "relational") {
    return (
      <>
        <FilterPillGroup
          label="state"
          options={states}
          value={selected("state")}
          onChange={(value) => onFilter("state", value)}
        />
        <FilterPillGroup
          label="lens"
          options={["all", "conflicts"]}
          value={selected("conflict")}
          onChange={(value) => onFilter("conflict", value)}
        />
      </>
    );
  }

  return (
    <FilterPillGroup
      label="status"
      options={statuses}
      value={selected("status")}
      onChange={(value) => onFilter("status", value)}
    />
  );
}

function audienceLabel(audience: string | null | undefined): string {
  return audience ?? "global";
}

function participantRefs(episode: EpisodeMemoryItem) {
  return (
    episode.participant_refs ??
    episode.participants.map((participant) => ({
      value: participant,
      id: null,
      label: participant,
    }))
  );
}

function ParticipantLabel({
  participant,
}: {
  participant: ReturnType<typeof participantRefs>[number];
}) {
  return (
    <span className="identity-inline">
      <span>{participant.label ?? "unknown"}</span>
      {participant.id === null ? null : <IdChip id={participant.id} type="entity" />}
    </span>
  );
}

function timeRangeLabel(start: number, end: number): string {
  return formatTimestampRange(start, end);
}

function betaMean(skill: ProceduralMemoryItem): number {
  const total = skill.alpha + skill.beta;
  return total <= 0 ? 0 : skill.alpha / total;
}

function stateTagKind(state: string): "acc" | "warn" | "bad" | "info" | "" {
  if (state === "established" || state === "active") {
    return "acc";
  }
  if (state === "contested" || state === "superseded") {
    return "warn";
  }
  if (state === "quarantined" || state === "revoked") {
    return "bad";
  }
  return "";
}

function orderedByRows<T>(
  rows: readonly MemoryRow[],
  items: readonly T[],
  itemId: (item: T) => string,
): T[] {
  const byId = new Map(items.map((item) => [itemId(item), item]));
  return rows.map((row) => byId.get(row.id)).filter((item): item is T => item !== undefined);
}

function MemoryBandBrowser({
  band,
  detail,
  rows,
  filteredRows,
  selectedId,
  loading,
  error,
  loadedCount,
  sortMode,
  onSortMode,
  filters,
  onFilter,
  onSelectRow,
  nextCursor,
  loadingMore,
  onLoadMore,
  onRetry,
  onOpenIdentity,
  onOpenCommitments,
}: {
  band: MemoryBandId;
  detail: MemoryBandDetail | null;
  rows: readonly MemoryRow[];
  filteredRows: readonly MemoryRow[];
  selectedId: string | null;
  loading: boolean;
  error: Error | null;
  loadedCount: number;
  sortMode: SortMode;
  onSortMode: (mode: SortMode) => void;
  filters: Record<string, string>;
  onFilter: (key: string, value: string) => void;
  onSelectRow: (id: string) => void;
  nextCursor: string | null;
  loadingMore: boolean;
  onLoadMore: () => void;
  onRetry?: () => void;
  onOpenIdentity?: () => void;
  onOpenCommitments?: () => void;
}) {
  const sortModes: SortMode[] = rows.some((row) => row.audience !== undefined)
    ? ["backend", "updated_desc", "updated_asc", "created_desc", "created_asc", "audience_group"]
    : ["backend", "updated_desc", "updated_asc", "created_desc", "created_asc"];

  return (
    <div className={`list matlas-browser matlas-${band}-browser`}>
      <div className="matlas-browser-toolbar">
        <span>
          {filteredRows.length} visible · {loadedCount} loaded
        </span>
        <span className="spacer"></span>
        <div className="filter-pills" aria-label="memory sort">
          {sortModes.map((mode) => (
            <button
              key={mode}
              type="button"
              className={`pill ${sortMode === mode ? "on" : ""}`}
              onClick={() => onSortMode(mode)}
            >
              {sortLabel(mode)}
            </button>
          ))}
        </div>
      </div>
      <div className="matlas-browser-filters">
        <MemoryStructuralControls band={band} rows={rows} filters={filters} onFilter={onFilter} />
      </div>
      {loading && rows.length === 0 && detail === null ? <Loading>loading {band}</Loading> : null}
      {error !== null ? <ErrorState onRetry={onRetry}>{error.message}</ErrorState> : null}
      {detail?.band === "episodic" ? (
        <EpisodicTimeline
          detail={detail}
          rows={filteredRows}
          selectedId={selectedId}
          onSelect={onSelectRow}
        />
      ) : detail?.band === "procedural" ? (
        <ProceduralSkillCards
          detail={detail}
          rows={filteredRows}
          selectedId={selectedId}
          onSelect={onSelectRow}
        />
      ) : detail?.band === "relational" ? (
        <RelationalFactTable
          detail={detail}
          rows={filteredRows}
          selectedId={selectedId}
          onSelect={onSelectRow}
        />
      ) : detail?.band === "affective" ? (
        <AffectiveAtlas
          detail={detail}
          rows={filteredRows}
          selectedId={selectedId}
          onSelect={onSelectRow}
        />
      ) : detail?.band === "social" ? (
        <SocialAtlas
          detail={detail}
          rows={filteredRows}
          selectedId={selectedId}
          onSelect={onSelectRow}
        />
      ) : detail?.band === "self" ? (
        <SelfAtlas
          detail={detail}
          rows={filteredRows}
          selectedId={selectedId}
          onSelect={onSelectRow}
          onOpenIdentity={onOpenIdentity}
        />
      ) : detail?.band === "commitments" ? (
        <CommitmentsAtlas
          detail={detail}
          rows={filteredRows}
          selectedId={selectedId}
          onSelect={onSelectRow}
          onOpenCommitments={onOpenCommitments}
        />
      ) : (
        <GenericMemoryRows rows={filteredRows} selectedId={selectedId} onSelect={onSelectRow} />
      )}
      {filteredRows.length === 0 && !loading && detail?.band !== "affective" ? (
        <Empty>no records in current filter</Empty>
      ) : null}
      {nextCursor !== null ? (
        <div style={{ padding: 12 }}>
          <button className="btn sm ghost" disabled={loadingMore} onClick={onLoadMore}>
            {loadingMore ? "loading" : "load more"}
          </button>
        </div>
      ) : null}
    </div>
  );
}

function GenericMemoryRows({
  rows,
  selectedId,
  onSelect,
}: {
  rows: readonly MemoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <>
      {rows.map((row) => (
        <div
          key={row.id}
          className={`list-row ${row.id === selectedId ? "selected" : ""}`}
          role="button"
          tabIndex={0}
          aria-pressed={row.id === selectedId}
          onClick={() => onSelect(row.id)}
          onKeyDown={(event) => activateOnEnterOrSpace(event, () => onSelect(row.id))}
        >
          <div className="ttl">{row.title}</div>
          <div className="meta">
            <span>[{shortId(row.id)}]</span>
            <span>·</span>
            <span>{row.meta}</span>
          </div>
        </div>
      ))}
    </>
  );
}

function EpisodicTimeline({
  detail,
  rows,
  selectedId,
  onSelect,
}: {
  detail: Extract<MemoryBandDetail, { band: "episodic" }>;
  rows: readonly MemoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const items = orderedByRows(rows, detail.items, (item) => item.id);

  return (
    <div className="matlas-timeline">
      {items.map((episode) => (
        <article
          key={episode.id}
          className={`list-row matlas-timeline-card ${episode.id === selectedId ? "selected" : ""}`}
          role="button"
          tabIndex={0}
          aria-pressed={episode.id === selectedId}
          onClick={() => onSelect(episode.id)}
          onKeyDown={(event) => activateOnEnterOrSpace(event, () => onSelect(episode.id))}
        >
          <div className="matlas-card-head">
            <div className="ttl">{episode.title}</div>
            <Tag kind="info">{audienceLabel(episode.audience)}</Tag>
          </div>
          <div className="matlas-card-body">{episode.narrative}</div>
          <div className="matlas-chip-row">
            <Tag>{timeRangeLabel(episode.start_time, episode.end_time)}</Tag>
            {participantRefs(episode).map((participant) => (
              <Tag key={participant.value}>
                <ParticipantLabel participant={participant} />
              </Tag>
            ))}
            <Tag>sig {episode.significance.toFixed(2)}</Tag>
            <Tag>conf {episode.confidence.toFixed(2)}</Tag>
            <Tag>{episode.source_count} src</Tag>
          </div>
          {episode.tags.length === 0 ? null : (
            <div className="matlas-tag-row">
              {episode.tags.map((tag) => (
                <Tag key={tag}>{tag}</Tag>
              ))}
            </div>
          )}
        </article>
      ))}
    </div>
  );
}

function ProceduralSkillCards({
  detail,
  rows,
  selectedId,
  onSelect,
}: {
  detail: Extract<MemoryBandDetail, { band: "procedural" }>;
  rows: readonly MemoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const skills = orderedByRows(rows, detail.items, (item) => item.id);

  return (
    <div className="matlas-skill-grid">
      {skills.map((skill) => {
        const mean = betaMean(skill);
        return (
          <article
            key={skill.id}
            className={`list-row matlas-skill-card ${skill.id === selectedId ? "selected" : ""}`}
            role="button"
            tabIndex={0}
            aria-pressed={skill.id === selectedId}
            onClick={() => onSelect(skill.id)}
            onKeyDown={(event) => activateOnEnterOrSpace(event, () => onSelect(skill.id))}
          >
            <div className="matlas-card-head">
              <div className="ttl">{skill.applies_when}</div>
              <Tag kind={stateTagKind(skill.status)}>{skill.status}</Tag>
            </div>
            <div className="matlas-card-body">{skill.approach}</div>
            <div className="matlas-chip-row">
              <Tag>{skill.successes} success</Tag>
              <Tag>{skill.failures} failure</Tag>
              <Tag>{skill.attempts} attempts</Tag>
              {skill.requires_manual_review ? <Tag kind="warn">manual review</Tag> : null}
            </div>
            <div
              className="matlas-beta-bar"
              role="meter"
              aria-label={`beta posterior ${skill.id}`}
              aria-valuemin={0}
              aria-valuemax={1}
              aria-valuenow={Number(mean.toFixed(3))}
            >
              <span style={{ width: `${mean * 100}%` }}></span>
            </div>
            <div className="matlas-card-meta">
              alpha {skill.alpha.toFixed(1)} · beta {skill.beta.toFixed(1)} · {skill.sample_count}{" "}
              samples · last used{" "}
              {skill.last_used === null ? "never" : formatTimestamp(skill.last_used)} · last
              successful{" "}
              {skill.last_successful === null ? "never" : formatTimestamp(skill.last_successful)}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function RelationalFactTable({
  detail,
  rows,
  selectedId,
  onSelect,
}: {
  detail: Extract<MemoryBandDetail, { band: "relational" }>;
  rows: readonly MemoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const slots = orderedByRows(rows, detail.items, (item) => item.id);

  return (
    <div className="matlas-table-wrap">
      <table className="matlas-rel-table">
        <thead>
          <tr>
            <th>subject</th>
            <th>slot</th>
            <th>value</th>
            <th>state</th>
            <th>evidence</th>
            <th>name provenance</th>
          </tr>
        </thead>
        <tbody>
          {slots.map((slot) => (
            <tr
              key={slot.id}
              className={slot.id === selectedId ? "selected" : ""}
              role="button"
              tabIndex={0}
              aria-pressed={slot.id === selectedId}
              onClick={() => onSelect(slot.id)}
              onKeyDown={(event) => activateOnEnterOrSpace(event, () => onSelect(slot.id))}
            >
              <td>{slot.subject ?? shortId(slot.subject_entity_id)}</td>
              <td>{slot.slot_key}</td>
              <td>{slot.value}</td>
              <td>
                <Tag kind={stateTagKind(slot.state)}>{slot.state}</Tag>
              </td>
              <td>
                {slot.sources_count} src · {slot.contradicted_count} contra · {slot.alternate_count}{" "}
                alt
              </td>
              <td>{slot.name_provenance}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AffectiveAtlas({
  detail,
  rows,
  selectedId,
  onSelect,
}: {
  detail: Extract<MemoryBandDetail, { band: "affective" }>;
  rows: readonly MemoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const points = orderedByRows(rows, detail.history, (item) => String(item.id));

  return (
    <div className="matlas-affective">
      <ValenceArousalPlane current={detail.current} history={detail.history} />
      <div className="props compact-props">
        <div className="row">
          <span className="k">half life</span>
          <span className="v">{detail.current.half_life_hours}h</span>
        </div>
        <div className="row">
          <span className="k">recent triggers</span>
          <span className="v">
            {detail.current.recent_triggers.length === 0
              ? "none"
              : detail.current.recent_triggers.join(", ")}
          </span>
        </div>
      </div>
      <div className="divider">history</div>
      <div className="matlas-history-list">
        {points.map((point) => (
          <div
            key={point.id}
            className={`list-row ${String(point.id) === selectedId ? "selected" : ""}`}
            role="button"
            tabIndex={0}
            aria-pressed={String(point.id) === selectedId}
            onClick={() => onSelect(String(point.id))}
            onKeyDown={(event) => activateOnEnterOrSpace(event, () => onSelect(String(point.id)))}
          >
            <div className="ttl">
              {formatTimestamp(point.ts)} · v {point.valence.toFixed(2)} / a{" "}
              {point.arousal.toFixed(2)}
            </div>
            <div className="meta">{point.trigger_reason ?? "no trigger"}</div>
          </div>
        ))}
        {points.length === 0 ? <Empty>no mood history</Empty> : null}
      </div>
    </div>
  );
}

function SocialAtlas({
  detail,
  rows,
  selectedId,
  onSelect,
}: {
  detail: Extract<MemoryBandDetail, { band: "social" }>;
  rows: readonly MemoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const items = orderedByRows(rows, detail.items, (item) => item.entity_id);

  return (
    <div className="matlas-social">
      <SocialTrustScatter items={items} selectedId={selectedId} onSelect={onSelect} />
      <div className="matlas-entity-grid">
        {items.map((profile) => (
          <article
            key={profile.entity_id}
            className={`list-row matlas-entity-card ${
              profile.entity_id === selectedId ? "selected" : ""
            }`}
            role="button"
            tabIndex={0}
            aria-pressed={profile.entity_id === selectedId}
            onClick={() => onSelect(profile.entity_id)}
            onKeyDown={(event) => activateOnEnterOrSpace(event, () => onSelect(profile.entity_id))}
          >
            <div className="ttl">{profile.name ?? profile.entity_id}</div>
            <div className="matlas-chip-row">
              <Tag>trust {profile.trust.toFixed(2)}</Tag>
              <Tag>attachment {profile.attachment.toFixed(2)}</Tag>
              <Tag>{profile.interaction_count} interactions</Tag>
            </div>
            <div className="matlas-card-meta">
              {profile.history_count} history · {profile.commitment_count} commitments · updated{" "}
              {formatTimestamp(profile.updated_at)}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function SelfAtlas({
  detail,
  rows,
  selectedId,
  onSelect,
  onOpenIdentity,
}: {
  detail: Extract<MemoryBandDetail, { band: "self" }>;
  rows: readonly MemoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onOpenIdentity?: () => void;
}) {
  const recentRows = sortedRows(rows, "updated_desc").slice(0, 5);

  return (
    <div className="matlas-self">
      <div className="matlas-snapshot-grid">
        <MetricTile label="values" value={detail.values.length} />
        <MetricTile label="goals" value={detail.goals.length} />
        <MetricTile label="traits" value={detail.traits.length} />
        <MetricTile label="open questions" value={detail.open_questions.length} />
        <MetricTile label="growth" value={detail.growth_markers.length} />
        <MetricTile label="periods" value={detail.periods.length} />
      </div>
      <div className="operator-actions">
        <button
          className="btn sm primary"
          type="button"
          disabled={onOpenIdentity === undefined}
          onClick={onOpenIdentity}
        >
          open Identity Studio
        </button>
      </div>
      <div className="divider">recent identity items</div>
      <GenericMemoryRows rows={recentRows} selectedId={selectedId} onSelect={onSelect} />
    </div>
  );
}

function CommitmentsAtlas({
  detail,
  rows,
  selectedId,
  onSelect,
  onOpenCommitments,
}: {
  detail: Extract<MemoryBandDetail, { band: "commitments" }>;
  rows: readonly MemoryRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onOpenCommitments?: () => void;
}) {
  const items = orderedByRows(rows, detail.items, (item) => item.id);
  const groups = new Map<string, CommitmentItem[]>();
  for (const item of items) {
    const key = `${item.enforcement_class} / ${item.state}`;
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }

  return (
    <div className="matlas-commitments">
      <div className="operator-actions">
        <button
          className="btn sm primary"
          type="button"
          disabled={onOpenCommitments === undefined}
          onClick={onOpenCommitments}
        >
          open commitments
        </button>
      </div>
      {[...groups.entries()].map(([group, groupItems]) => (
        <section key={group} className="matlas-commit-group">
          <div className="matlas-group-title">{group}</div>
          {groupItems.map((commitment) => (
            <article
              key={commitment.id}
              className={`list-row matlas-commit-card ${
                commitment.id === selectedId ? "selected" : ""
              }`}
              role="button"
              tabIndex={0}
              aria-pressed={commitment.id === selectedId}
              onClick={() => onSelect(commitment.id)}
              onKeyDown={(event) => activateOnEnterOrSpace(event, () => onSelect(commitment.id))}
            >
              <div className="ttl">{commitment.text}</div>
              <div className="matlas-chip-row">
                <Tag kind={commitment.enforcement_class === "critical" ? "warn" : ""}>
                  {commitment.enforcement_class}
                </Tag>
                <Tag kind={stateTagKind(commitment.state)}>{commitment.state}</Tag>
                <Tag>{commitment.directive_family}</Tag>
                <Tag kind="info">{audienceLabel(commitment.audience)}</Tag>
              </div>
            </article>
          ))}
        </section>
      ))}
    </div>
  );
}

function MetricTile({ label, value }: { label: string; value: number }) {
  return (
    <div className="matlas-metric">
      <div className="k">{label}</div>
      <div className="v">{value}</div>
    </div>
  );
}

function SemanticTopologyPanel({
  selectedId,
  onSelectNode,
}: {
  selectedId: string | null;
  onSelectNode: (nodeId: string) => void;
}) {
  const live = useLiveEventsContext();
  const api = useApi<SemanticGraphResponse>(() => getSemanticGraph(SEMANTIC_TOPOLOGY_LIMIT), []);
  const refetch = api.refetch;

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type === "maintenance:tick") {
        void refetch();
      }
    });
  }, [live, refetch]);

  if (api.loading && api.data === null) {
    return <Loading>loading semantic topology</Loading>;
  }

  if (api.error !== null) {
    return <ErrorState onRetry={api.refetch}>{api.error.message}</ErrorState>;
  }

  if (api.data === null) {
    return <Empty>semantic topology unavailable</Empty>;
  }

  return <SemanticTopology graph={api.data} selectedId={selectedId} onSelectNode={onSelectNode} />;
}

function MemoryDrill({
  band,
  bands,
  totalCount,
  totalIsLowerBound,
  sessionId,
  back,
  onSelectBand,
  onOpenIdentity,
  onOpenCommitments,
  onMemoryChanged,
}: {
  band: MemoryBandId;
  bands: readonly MemoryBandSummary[];
  totalCount: number;
  totalIsLowerBound: boolean;
  sessionId: string;
  back: () => void;
  onSelectBand: (band: MemoryBandId) => void;
  onOpenIdentity?: () => void;
  onOpenCommitments?: () => void;
  onMemoryChanged: () => Promise<void>;
}) {
  const [searchText, setSearchText] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [detail, setDetail] = useState<MemoryBandDetail | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>("backend");
  const [semanticViewMode, setSemanticViewMode] = useState<SemanticViewMode>("browser");
  const [filters, setFilters] = useState<Record<string, string>>(() => defaultFiltersForBand(band));
  const [fetchedSemanticNode, setFetchedSemanticNode] = useState<SemanticMemoryNode | null>(null);
  const [fetchedSemanticNodeLoading, setFetchedSemanticNodeLoading] = useState(false);
  const [fetchedSemanticNodeError, setFetchedSemanticNodeError] = useState<string | null>(null);
  const api = useApi(
    () =>
      getMemoryBand(band, {
        session: sessionId,
        limit: MEMORY_PAGE_LIMIT,
        ...(searchQuery.length === 0 ? {} : { query: searchQuery }),
      }),
    [band, sessionId, searchQuery],
  );
  const rows = useMemo(() => (detail === null ? [] : detailRows(detail)), [detail]);
  const filteredRows = useMemo(
    () => sortedRows(structuralFilterRows(rows, filters), sortMode),
    [filters, rows, sortMode],
  );
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [whyId, setWhyId] = useState<string | null>(null);
  const [action, setAction] = useState<MemoryCorrectionAction | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const loadedSelected =
    selectedId === null ? null : (rows.find((row) => row.id === selectedId) ?? null);
  const shouldFetchSelectedSemanticNode =
    band === "semantic" &&
    selectedId !== null &&
    loadedSelected === null &&
    correctionActionKind(selectedId) === "semantic_node";
  const fetchedSelected =
    shouldFetchSelectedSemanticNode && fetchedSemanticNode !== null
      ? semanticNodeRow(fetchedSemanticNode, rows.length)
      : null;
  const selected =
    loadedSelected ?? fetchedSelected ?? (selectedId === null ? (filteredRows[0] ?? null) : null);
  const selectedCorrectionKind = selected === null ? null : correctionActionKind(selected.id);
  const selectedSemanticNode =
    band === "semantic" && selected?.rowKind === "node"
      ? fetchedSemanticNode?.id === selected.id
        ? fetchedSemanticNode
        : detail?.band === "semantic"
          ? (detail.nodes.find((node) => node.id === selected.id) ?? null)
          : null
      : null;
  const searchable = SEARCHABLE_BANDS.has(band);
  const searchActive = searchQuery.length > 0;
  const nextCursor = searchActive ? null : detailNextCursor(detail);
  const loadedCount = loadedBandCount(detail, rows);
  const displayTotal = Math.max(totalCount, loadedCount);
  const totalLabel =
    totalIsLowerBound && nextCursor !== null
      ? `≥${displayTotal.toLocaleString()}`
      : displayTotal.toLocaleString();
  const topologySelectedId =
    selectedId !== null && correctionActionKind(selectedId) === "semantic_node" ? selectedId : null;

  useEffect(() => {
    setDetail(api.data);
  }, [api.data]);

  useEffect(() => {
    setDetail(null);
    setSelectedId(null);
    setFilters(defaultFiltersForBand(band));
    setSortMode("backend");
    setSemanticViewMode("browser");
    setSearchText("");
    setSearchQuery("");
    setFetchedSemanticNode(null);
    setFetchedSemanticNodeError(null);
    setFetchedSemanticNodeLoading(false);
  }, [band, sessionId]);

  useEffect(() => {
    if (!shouldFetchSelectedSemanticNode || selectedId === null) {
      setFetchedSemanticNode(null);
      setFetchedSemanticNodeError(null);
      setFetchedSemanticNodeLoading(false);
      return undefined;
    }

    let cancelled = false;
    setFetchedSemanticNode(null);
    setFetchedSemanticNodeError(null);
    setFetchedSemanticNodeLoading(true);

    void getSemanticNode(selectedId)
      .then((node) => {
        if (!cancelled) {
          setFetchedSemanticNode(node);
        }
      })
      .catch((caught: unknown) => {
        if (!cancelled) {
          setFetchedSemanticNodeError(caught instanceof Error ? caught.message : String(caught));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setFetchedSemanticNodeLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId, shouldFetchSelectedSemanticNode]);

  function setFilter(key: string, value: string): void {
    setFilters((current) => ({ ...current, [key]: value }));
    setSelectedId(null);
  }

  async function loadMore(): Promise<void> {
    if (nextCursor === null || loadingMore) {
      return;
    }

    setLoadingMore(true);
    setOperatorError(null);
    try {
      const next = await getMemoryBand(band, {
        session: sessionId,
        limit: MEMORY_PAGE_LIMIT,
        cursor: nextCursor,
      });
      setDetail((current) => mergeMemoryDetail(current, next));
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoadingMore(false);
    }
  }

  function submitSearch(): void {
    const trimmed = searchText.trim();
    if (!searchable) {
      return;
    }

    if (trimmed.length === 0 && searchQuery.length === 0) {
      return;
    }

    setDetail(null);
    setSearchQuery(trimmed);
    setSelectedId(null);
  }

  function clearSearch(): void {
    setSearchText("");
    setDetail(null);
    setSearchQuery("");
    setSelectedId(null);
  }

  function selectSemanticViewMode(mode: SemanticViewMode): void {
    if (mode === semanticViewMode) {
      return;
    }

    if (mode === "topology") {
      if (selectedId !== null && correctionActionKind(selectedId) !== "semantic_node") {
        setSelectedId(null);
      }
    } else if (selectedId !== null && rows.find((row) => row.id === selectedId) === undefined) {
      setSelectedId(null);
    }

    setSemanticViewMode(mode);
  }

  async function refetchAfterMemoryCorrection(): Promise<void> {
    // Invalidates GET /api/memory/bands and GET /api/memory/bands/:band.
    // Semantic topology is refreshed independently from GET /api/semantic/graph.
    await Promise.all([api.refetch(), onMemoryChanged()]);
  }

  async function runMemoryAction(label: string, callback: () => Promise<void>): Promise<void> {
    setBusy(label);
    setOperatorError(null);
    try {
      await callback();
      setAction(null);
      await refetchAfterMemoryCorrection();
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function submitMemoryAction(): Promise<void> {
    if (action === null) {
      return;
    }

    if (action.kind === "forget") {
      await runMemoryAction("forget", async () => {
        await postCorrectionForget(action.id);
      });
      return;
    }

    if (action.kind === "correct") {
      await runMemoryAction("correct", async () => {
        const patch = parseJsonPatch(action.patch);
        await postCorrectionCorrect(action.id, {
          patch,
          ...(action.reason.trim().length === 0 ? {} : { reason: action.reason.trim() }),
        });
      });
      return;
    }

    await runMemoryAction("invalidate-edge", async () => {
      const parsedAt = action.at.trim().length === 0 ? undefined : Number(action.at);
      if (parsedAt !== undefined && !Number.isFinite(parsedAt)) {
        throw new Error("at must be a finite number");
      }
      await postSemanticEdgeInvalidate(action.id, {
        ...(parsedAt === undefined ? {} : { at: parsedAt }),
        ...(action.reason.trim().length === 0 ? {} : { reason: action.reason.trim() }),
      });
    });
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <button className="btn sm ghost" type="button" onClick={back}>
          ← memory
        </button>
        <h1>memory atlas</h1>
        <span className="desc">{BAND_DESCRIPTIONS[band]}</span>
        <span className="spacer"></span>
        {band === "semantic" ? (
          <div className="filter-pills" role="tablist" aria-label="semantic memory view">
            {(["browser", "topology"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                role="tab"
                aria-selected={semanticViewMode === mode}
                className={`pill ${semanticViewMode === mode ? "on" : ""}`}
                onClick={() => selectSemanticViewMode(mode)}
              >
                {mode}
              </button>
            ))}
          </div>
        ) : null}
        <Tag>
          {searchActive
            ? `${rows.length} results`
            : `loaded ${loadedCount.toLocaleString()} of ${totalLabel}`}
        </Tag>
      </div>
      <BandOverviewBar bands={bands} activeBand={band} onSelectBand={onSelectBand} />
      {searchable ? (
        <form
          className="operator-actions"
          style={{ padding: "0 0 10px", gap: 8 }}
          onSubmit={(event) => {
            event.preventDefault();
            submitSearch();
          }}
        >
          <input
            value={searchText}
            placeholder={`search ${band}`}
            onChange={(event) => setSearchText(event.target.value)}
            style={{ minWidth: 260 }}
          />
          <button className="btn sm primary" type="submit">
            search
          </button>
          {searchActive ? (
            <button className="btn sm ghost" type="button" onClick={clearSearch}>
              clear search
            </button>
          ) : null}
        </form>
      ) : null}
      {operatorError === null ? null : <ErrorState>{operatorError}</ErrorState>}
      <div
        className={`band-detail ${
          band === "semantic" && semanticViewMode === "topology" ? "semantic-topology-mode" : ""
        }`}
        style={{ flex: 1 }}
      >
        {band === "semantic" && semanticViewMode === "topology" ? (
          <div className="semantic-topology-pane">
            <SemanticTopologyPanel selectedId={topologySelectedId} onSelectNode={setSelectedId} />
          </div>
        ) : (
          <MemoryBandBrowser
            band={band}
            detail={detail}
            rows={rows}
            filteredRows={filteredRows}
            selectedId={selected?.id ?? null}
            loading={api.loading}
            error={api.error}
            loadedCount={loadedCount}
            sortMode={sortMode}
            onSortMode={setSortMode}
            filters={filters}
            onFilter={setFilter}
            onSelectRow={setSelectedId}
            nextCursor={nextCursor}
            loadingMore={loadingMore}
            onLoadMore={() => void loadMore()}
            onRetry={api.refetch}
            onOpenIdentity={onOpenIdentity}
            onOpenCommitments={onOpenCommitments}
          />
        )}
        <div className="detail">
          {selected === null ? (
            fetchedSemanticNodeLoading ? (
              <Loading>loading selected semantic node</Loading>
            ) : fetchedSemanticNodeError !== null ? (
              <ErrorState>{fetchedSemanticNodeError}</ErrorState>
            ) : (
              <Empty>no records in this band</Empty>
            )
          ) : selectedSemanticNode === null ? (
            <>
              <h2>{selected.title}</h2>
              <div className="meta-line">
                <span>
                  <IdChip id={selected.id} />
                </span>
                <span>·</span>
                <span>{selected.meta}</span>
              </div>
              <div className="divider">body</div>
              <div
                style={{
                  fontFamily: "var(--sans)",
                  color: "var(--text-dim)",
                  fontSize: 13,
                  lineHeight: 1.6,
                }}
              >
                {selected.body}
              </div>
              {detail === null ? null : (
                <BandSpecificDetail detail={detail} selectedId={selected.id} />
              )}
            </>
          ) : (
            <SemanticNodeDetail node={selectedSemanticNode} />
          )}
        </div>
        <div
          className="panel"
          style={{
            borderLeft: "1px solid var(--line)",
            borderTop: 0,
            borderRight: 0,
            borderBottom: 0,
          }}
        >
          <div className="panel-header">
            <span className="title">properties</span>
          </div>
          <div className="panel-body pad">
            <div className="props">
              <div className="row">
                <span className="k">band</span>
                <span className="v">{band}</span>
              </div>
              <div className="row">
                <span className="k">id</span>
                <span className="v">
                  {selected?.id === undefined && selectedId === null ? (
                    "—"
                  ) : (
                    <IdChip
                      id={selected?.id ?? selectedId ?? ""}
                      hint={selectedSemanticNode ?? undefined}
                    />
                  )}
                </span>
              </div>
              <div className="row">
                <span className="k">rows</span>
                <span className="v">{filteredRows.length}</span>
              </div>
              <div className="row">
                <span className="k">source policy</span>
                <span className="v">source-linked records only</span>
              </div>
            </div>
            <div className="divider" style={{ marginTop: 16 }}>
              operations
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {selected !== null && selectedCorrectionKind !== null ? (
                <>
                  <button
                    className="btn sm"
                    disabled={busy !== null}
                    onClick={() => setWhyId(selected.id)}
                  >
                    why
                  </button>
                  {selectedCorrectionKind === "semantic_edge" ? (
                    <button
                      className="btn sm danger"
                      disabled={busy !== null}
                      onClick={() =>
                        setAction({
                          kind: "invalidate-edge",
                          id: selected.id,
                          title: selected.title,
                          reason: "",
                          at: "",
                        })
                      }
                    >
                      invalidate
                    </button>
                  ) : (
                    <>
                      <button
                        className="btn sm danger"
                        disabled={busy !== null}
                        onClick={() =>
                          setAction({ kind: "forget", id: selected.id, title: selected.title })
                        }
                      >
                        forget
                      </button>
                      <button
                        className="btn sm ghost"
                        disabled={busy !== null}
                        onClick={() =>
                          setAction({
                            kind: "correct",
                            id: selected.id,
                            title: selected.title,
                            patch: defaultMemoryPatch(selected, selectedCorrectionKind),
                            reason: "",
                          })
                        }
                      >
                        correct
                      </button>
                    </>
                  )}
                </>
              ) : (
                <span className="dim" style={{ fontSize: "var(--fs-xs)" }}>
                  no correction actions for this row
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
      <WhyDrawer open={whyId !== null} id={whyId} onClose={() => setWhyId(null)} />
      <Modal
        open={action !== null}
        title={
          <span className="identity-inline">
            <span>{action === null ? "correction" : action.kind}</span>
            {action === null ? null : <IdChip id={action.id} />}
          </span>
        }
        onClose={() => {
          if (busy === null) {
            setAction(null);
          }
        }}
        footer={
          <>
            <button
              className="btn sm ghost"
              disabled={busy !== null}
              onClick={() => setAction(null)}
            >
              cancel
            </button>
            <button
              className={`btn sm ${
                action?.kind === "forget" || action?.kind === "invalidate-edge"
                  ? "danger"
                  : "primary"
              }`}
              disabled={busy !== null}
              onClick={() => void submitMemoryAction()}
            >
              {busy === null
                ? action?.kind === "invalidate-edge"
                  ? "invalidate"
                  : action?.kind === "correct"
                    ? "queue"
                    : action?.kind
                : "saving"}
            </button>
          </>
        }
      >
        {action?.kind === "forget" ? (
          <div className="modal-form">
            <div className="dim">{action.title}</div>
          </div>
        ) : null}
        {action?.kind === "correct" ? (
          <div className="modal-form">
            <div className="dim">{action.title}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                value={action.reason}
                onChange={(event) => setAction({ ...action, reason: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>json patch</span>
              <textarea
                value={action.patch}
                onChange={(event) => setAction({ ...action, patch: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {action?.kind === "invalidate-edge" ? (
          <div className="modal-form">
            <div className="dim">{action.title}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                value={action.reason}
                onChange={(event) => setAction({ ...action, reason: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>at ms</span>
              <input
                type="number"
                value={action.at}
                onChange={(event) => setAction({ ...action, at: event.target.value })}
              />
            </label>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}

function CorrectionReviewSummaryPanel({ onOpenReview }: { onOpenReview?: () => void }) {
  const api = useApi(() => getReviews({ openOnly: true, kind: "correction" }), []);
  const count = api.data?.rows.length ?? 0;

  if (api.loading && api.data === null) {
    return <Loading>loading corrections</Loading>;
  }

  if (api.error !== null) {
    return <ErrorState onRetry={api.refetch}>{api.error.message}</ErrorState>;
  }

  return (
    <div style={{ padding: 12 }}>
      <div className="panel-note">
        {count === 0 ? "No pending corrections." : `${count} pending correction review rows.`}
      </div>
      <div className="operator-actions" style={{ marginTop: 10 }}>
        <button className="btn sm primary" type="button" onClick={onOpenReview}>
          open review
        </button>
      </div>
    </div>
  );
}

function BandSpecificDetail({
  detail,
  selectedId,
}: {
  detail: MemoryBandDetail;
  selectedId: string;
}) {
  const renderIds = (ids: readonly string[], type?: "stream_entry") =>
    ids.length === 0 ? (
      <span className="dim" style={{ fontSize: "var(--fs-xs)" }}>
        none
      </span>
    ) : (
      <span className="matlas-id-list">
        {ids.map((id) => (
          <IdChip key={id} id={id} type={type} />
        ))}
      </span>
    );

  if (detail.band === "episodic") {
    const episode = detail.items.find((item) => item.id === selectedId) as
      | EpisodeMemoryItem
      | undefined;
    if (episode === undefined) {
      return null;
    }
    return (
      <>
        <div className="divider">episode fields</div>
        <div className="props">
          <div className="row">
            <span className="k">time range</span>
            <span className="v">{timeRangeLabel(episode.start_time, episode.end_time)}</span>
          </div>
          <div className="row">
            <span className="k">audience</span>
            <span className="v">{audienceLabel(episode.audience)}</span>
          </div>
          <div className="row">
            <span className="k">participants</span>
            <span className="v">
              {participantRefs(episode).length === 0
                ? "none"
                : participantRefs(episode).map((participant, index) => (
                    <span key={participant.value}>
                      {index === 0 ? null : ", "}
                      <ParticipantLabel participant={participant} />
                    </span>
                  ))}
            </span>
          </div>
          <div className="row">
            <span className="k">location</span>
            <span className="v">{episode.location ?? "—"}</span>
          </div>
          <div className="row">
            <span className="k">significance</span>
            <span className="v">{episode.significance.toFixed(2)}</span>
          </div>
          <div className="row">
            <span className="k">confidence</span>
            <span className="v">{episode.confidence.toFixed(2)}</span>
          </div>
        </div>
        <div className="divider">citations</div>
        {renderIds(episode.source_stream_ids, "stream_entry")}
        <div className="divider">lineage</div>
        <div className="props">
          <div className="row">
            <span className="k">derived from</span>
            <span className="v">{renderIds(episode.lineage.derived_from)}</span>
          </div>
          <div className="row">
            <span className="k">supersedes</span>
            <span className="v">{renderIds(episode.lineage.supersedes)}</span>
          </div>
        </div>
      </>
    );
  }

  if (detail.band === "procedural") {
    const skill = detail.items.find((item) => item.id === selectedId);
    if (skill === undefined) {
      return null;
    }
    const mean = betaMean(skill);
    return (
      <>
        <div className="divider">skill evidence</div>
        <div className="props">
          <div className="row">
            <span className="k">status</span>
            <span className="v">{skill.status}</span>
          </div>
          <div className="row">
            <span className="k">beta posterior</span>
            <span className="v">
              mean {mean.toFixed(3)} · alpha {skill.alpha.toFixed(1)} · beta {skill.beta.toFixed(1)}{" "}
              · {skill.sample_count} samples
            </span>
          </div>
          <div className="row">
            <span className="k">outcomes</span>
            <span className="v">
              {skill.successes} success · {skill.failures} failure · {skill.attempts} attempts
            </span>
          </div>
          <div className="row">
            <span className="k">last used</span>
            <span className="v">
              {skill.last_used === null ? "never" : formatTimestamp(skill.last_used)}
            </span>
          </div>
          <div className="row">
            <span className="k">last successful</span>
            <span className="v">
              {skill.last_successful === null ? "never" : formatTimestamp(skill.last_successful)}
            </span>
          </div>
          <div className="row">
            <span className="k">manual review</span>
            <span className="v">{skill.requires_manual_review ? "required" : "not required"}</span>
          </div>
        </div>
        <div className="divider">source episodes</div>
        {renderIds(skill.source_episode_ids)}
      </>
    );
  }

  if (detail.band === "affective") {
    const point = detail.history.find((item) => String(item.id) === selectedId);
    return (
      <>
        <div className="divider">current mood</div>
        <div className="props">
          <div className="row">
            <span className="k">valence</span>
            <span className="v">{detail.current.valence.toFixed(2)}</span>
          </div>
          <div className="row">
            <span className="k">arousal</span>
            <span className="v">{detail.current.arousal.toFixed(2)}</span>
          </div>
          <div className="row">
            <span className="k">updated</span>
            <span className="v">{formatTimestamp(detail.current.updated_at)}</span>
          </div>
          <div className="row">
            <span className="k">half life</span>
            <span className="v">{detail.current.half_life_hours}h</span>
          </div>
          <div className="row">
            <span className="k">recent triggers</span>
            <span className="v">
              {detail.current.recent_triggers.length === 0
                ? "none"
                : detail.current.recent_triggers.join(", ")}
            </span>
          </div>
        </div>
        {point === undefined ? null : (
          <>
            <div className="divider">selected mood history</div>
            <div className="props">
              <div className="row">
                <span className="k">trigger</span>
                <span className="v">{point.trigger_reason ?? "none"}</span>
              </div>
              <div className="row">
                <span className="k">provenance</span>
                <span className="v">{jsonText(point.provenance)}</span>
              </div>
            </div>
          </>
        )}
      </>
    );
  }

  if (detail.band === "self") {
    const value = detail.values.find((item) => item.id === selectedId);
    const goal = detail.goals.find((item) => item.id === selectedId);
    const trait = detail.traits.find((item) => item.id === selectedId);
    const question = detail.open_questions.find((item) => item.id === selectedId);
    return (
      <>
        <div className="divider">identity record</div>
        <div className="props">
          {value === undefined ? null : (
            <>
              <div className="row">
                <span className="k">state</span>
                <span className="v">{value.state}</span>
              </div>
              <div className="row">
                <span className="k">support</span>
                <span className="v">
                  {value.support_count} support · {value.contradiction_count} contradiction
                </span>
              </div>
              <div className="row">
                <span className="k">evidence</span>
                <span className="v">{renderIds(value.evidence_episode_ids)}</span>
              </div>
            </>
          )}
          {goal === undefined ? null : (
            <>
              <div className="row">
                <span className="k">status</span>
                <span className="v">{goal.status}</span>
              </div>
              <div className="row">
                <span className="k">priority</span>
                <span className="v">{goal.priority.toFixed(2)}</span>
              </div>
              <div className="row">
                <span className="k">target</span>
                <span className="v">
                  {goal.target_at === null ? "none" : formatTimestamp(goal.target_at)}
                </span>
              </div>
            </>
          )}
          {trait === undefined ? null : (
            <>
              <div className="row">
                <span className="k">strength</span>
                <span className="v">{trait.strength.toFixed(2)}</span>
              </div>
              <div className="row">
                <span className="k">confidence</span>
                <span className="v">{trait.confidence.toFixed(2)}</span>
              </div>
              <div className="row">
                <span className="k">evidence</span>
                <span className="v">{renderIds(trait.evidence_episode_ids)}</span>
              </div>
            </>
          )}
          {question === undefined ? null : (
            <>
              <div className="row">
                <span className="k">status</span>
                <span className="v">{question.status}</span>
              </div>
              <div className="row">
                <span className="k">urgency</span>
                <span className="v">{question.urgency.toFixed(2)}</span>
              </div>
              <div className="row">
                <span className="k">rumination ticks</span>
                <span className="v">{question.unresolved_rumination_ticks}</span>
              </div>
            </>
          )}
        </div>
      </>
    );
  }

  if (detail.band === "commitments") {
    const commitment = detail.items.find((item) => item.id === selectedId);
    if (commitment === undefined) {
      return null;
    }
    return (
      <>
        <div className="divider">commitment fields</div>
        <div className="props">
          <div className="row">
            <span className="k">audience</span>
            <span className="v">{audienceLabel(commitment.audience)}</span>
          </div>
          <div className="row">
            <span className="k">directive family</span>
            <span className="v">{commitment.directive_family}</span>
          </div>
          <div className="row">
            <span className="k">priority</span>
            <span className="v">{commitment.priority}</span>
          </div>
          <div className="row">
            <span className="k">type / kind</span>
            <span className="v">
              {commitment.type} · {commitment.kind}
            </span>
          </div>
          <div className="row">
            <span className="k">sources</span>
            <span className="v">
              {renderIds(commitment.source_stream_entry_ids, "stream_entry")}
            </span>
          </div>
          {commitment.superseded_by_id === null ? null : (
            <div className="row">
              <span className="k">superseded by</span>
              <span className="v">{renderIds([commitment.superseded_by_id])}</span>
            </div>
          )}
          {commitment.canonicalized_by_artifact_entry_id === null ? null : (
            <div className="row">
              <span className="k">canonicalized by</span>
              <span className="v">
                {renderIds([commitment.canonicalized_by_artifact_entry_id])}
              </span>
            </div>
          )}
        </div>
      </>
    );
  }

  if (detail.band === "social") {
    const profile = detail.items.find((item) => item.entity_id === selectedId);
    if (profile === undefined) {
      return null;
    }
    return (
      <>
        <div className="divider">social entity</div>
        <div className="props">
          <div className="row">
            <span className="k">entity</span>
            <span className="v">
              <IdChip id={profile.entity_id} hint={profile} />
            </span>
          </div>
          <div className="row">
            <span className="k">trust</span>
            <span className="v">{profile.trust.toFixed(2)}</span>
          </div>
          <div className="row">
            <span className="k">attachment</span>
            <span className="v">{profile.attachment.toFixed(2)}</span>
          </div>
          <div className="row">
            <span className="k">interactions</span>
            <span className="v">
              {profile.interaction_count} total · {profile.history_count} history
            </span>
          </div>
          <div className="row">
            <span className="k">last interaction</span>
            <span className="v">
              {profile.last_interaction_at === null
                ? "none"
                : formatTimestamp(profile.last_interaction_at)}
            </span>
          </div>
        </div>
      </>
    );
  }

  if (detail.band === "relational") {
    const slot = detail.items.find((item) => item.id === selectedId);
    return (
      <>
        {slot === undefined ? null : (
          <>
            <div className="divider">fact evidence</div>
            <div className="props">
              <div className="row">
                <span className="k">subject entity</span>
                <span className="v">
                  <IdChip id={slot.subject_entity_id} hint={slot} />
                </span>
              </div>
              <div className="row">
                <span className="k">slot key</span>
                <span className="v">{slot.slot_key}</span>
              </div>
              <div className="row">
                <span className="k">evidence</span>
                <span className="v">
                  {slot.sources_count} source · {slot.contradicted_count} contradicted ·{" "}
                  {slot.alternate_count} alternate
                </span>
              </div>
              <div className="row">
                <span className="k">name provenance</span>
                <span className="v">{slot.name_provenance}</span>
              </div>
            </div>
          </>
        )}
        <div className="divider">state counts</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {Object.entries(detail.counts).map(([state, count]) => (
            <Tag key={state}>
              {state} {count}
            </Tag>
          ))}
        </div>
      </>
    );
  }

  return null;
}
