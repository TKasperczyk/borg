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
  EpisodeMemoryItem,
  MemoryBandDetail,
  MemoryBandId,
  MemoryBandSummary,
  SemanticGraphResponse,
  SemanticMemoryNode,
} from "../../api/types";
import { Modal } from "../../components/Modal";
import { Panel } from "../../components/Panel";
import { SemanticNodeDetail } from "../../components/SemanticNodeDetail";
import { Spark } from "../../components/Spark";
import { Tag } from "../../components/Tag";
import { WhyDrawer } from "../../components/WhyDrawer";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { formatTime } from "../../lib/stream-utils";
import { dateLabel, jsonText, parseJsonPatch, shortId } from "../screen-utils";
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

type SortMode = "backend" | "updated_desc" | "updated_asc" | "created_desc" | "created_asc";
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
  return {
    id: node.id,
    title: node.label,
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
        meta: `${scoreMeta(item.search_score)}${dateLabel(item.start_time)} · ${item.audience ?? "global"} · ${item.source_count} src`,
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
        title: `${formatTime(point.ts)} · valence ${point.valence.toFixed(2)}`,
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
        createdAt: slot.created_at,
        updatedAt: slot.updated_at,
      }));
  }
}

export function MemoryScreen({
  sessionId,
  onOpenReview,
}: {
  sessionId: string;
  onOpenReview?: () => void;
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
        growth: [1, 1, 1],
        stats: [],
      },
  );

  if (activeBand !== null) {
    const activeSummary = bands.find((item) => item.id === activeBand);
    return (
      <MemoryDrill
        band={activeBand}
        totalCount={activeSummary?.count ?? 0}
        totalIsLowerBound={activeSummary?.count_is_lower_bound ?? false}
        sessionId={sessionId}
        back={() => setActiveBand(null)}
        onMemoryChanged={api.refetch}
      />
    );
  }

  if (api.loading && api.data === null) {
    return <div className="notice">loading memory bands</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  return (
    <div className="bands">
      <div className="bands-head">
        <h1>memory::bands</h1>
        <div className="desc">
          raw memory store browser · audience scoping applies during retrieval/evidence ledger
        </div>
      </div>
      <div className="bands-grid">
        {bands.map((band) => (
          <BandCard key={band.id} band={band} onClick={() => setActiveBand(band.id)} />
        ))}
      </div>
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

function BandCard({ band, onClick }: { band: MemoryBandSummary; onClick: () => void }) {
  const countLabel = `${band.count.toLocaleString()}${band.count_is_lower_bound ? "+" : ""}`;

  return (
    <div className="band-card" onClick={onClick}>
      <div className="head">
        <span>band {band.n ?? "—"}</span>
        <span className="n">{countLabel}</span>
      </div>
      <div className="name">{band.name}</div>
      <div className="desc-line">{band.desc ?? BAND_DESCRIPTIONS[band.id]}</div>
      <Spark data={band.growth ?? [1, 1, 1]} />
      <div className="stat-row">
        {band.stats.slice(0, 2).map((stat) => (
          <div key={stat.k} className="stat">
            <div className="k">{stat.k}</div>
            <div className="v">{stat.v}</div>
          </div>
        ))}
      </div>
      <div className="explore">browse ▸</div>
    </div>
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
  const audience = filters.audience ?? "all";
  const tag = filters.tag ?? "all";
  const kind = filters.kind ?? "all";
  const status = filters.status ?? "all";
  const state = filters.state ?? "all";
  const enforcement = filters.enforcement ?? "all";

  return rows.filter((row) => {
    if (rowKind !== "all" && row.rowKind !== rowKind) {
      return false;
    }
    if (audience !== "all" && row.audience !== audience) {
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
  return next.sort(
    (left, right) => (left.createdAt ?? 0) - (right.createdAt ?? 0) || left.order - right.order,
  );
}

function defaultFiltersForBand(_band: MemoryBandId): Record<string, string> {
  return {
    audience: "all",
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

  return (
    <>
      <span className="dim" style={{ fontSize: 10.5 }}>
        {label}
      </span>
      <div className="filter-pills">
        {options.map((option) => (
          <span
            key={option}
            className={`pill ${value === option ? "on" : ""}`}
            onClick={() => onChange(option)}
          >
            {option}
          </span>
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
  const audiences = ["all", ...uniqueSorted(rows.map((row) => row.audience))];
  const tags = ["all", ...uniqueSorted(rows.flatMap((row) => row.tags ?? []))];
  const kinds = ["all", ...uniqueSorted(rows.map((row) => row.kind))];
  const statuses = ["all", ...uniqueSorted(rows.map((row) => row.status))];
  const states = ["all", ...uniqueSorted(rows.map((row) => row.state))];
  const enforcement = ["all", ...uniqueSorted(rows.map((row) => row.enforcement))];
  const selected = (key: string) => filters[key] ?? "all";

  if (band === "episodic") {
    return (
      <>
        <FilterPillGroup
          label="audience"
          options={audiences}
          value={selected("audience")}
          onChange={(value) => onFilter("audience", value)}
        />
        <FilterPillGroup
          label="tag"
          options={tags}
          value={selected("tag")}
          onChange={(value) => onFilter("tag", value)}
        />
      </>
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
        <FilterPillGroup
          label="audience"
          options={audiences}
          value={selected("audience")}
          onChange={(value) => onFilter("audience", value)}
        />
      </>
    );
  }

  if (band === "relational") {
    return (
      <FilterPillGroup
        label="state"
        options={states}
        value={selected("state")}
        onChange={(value) => onFilter("state", value)}
      />
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
    return <div className="notice">loading semantic topology</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  if (api.data === null) {
    return <div className="notice">semantic topology unavailable</div>;
  }

  return <SemanticTopology graph={api.data} selectedId={selectedId} onSelectNode={onSelectNode} />;
}

function MemoryDrill({
  band,
  totalCount,
  totalIsLowerBound,
  sessionId,
  back,
  onMemoryChanged,
}: {
  band: MemoryBandId;
  totalCount: number;
  totalIsLowerBound: boolean;
  sessionId: string;
  back: () => void;
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
      ? `${displayTotal.toLocaleString()}+`
      : displayTotal.toLocaleString();

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
        <span className="btn sm ghost" style={{ cursor: "pointer" }} onClick={back}>
          ← memory
        </span>
        <h1>{band} memory</h1>
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
                onClick={() => setSemanticViewMode(mode)}
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
      {operatorError === null ? null : (
        <div className="notice bad" style={{ padding: 12 }}>
          {operatorError}
        </div>
      )}
      <div
        className={`band-detail ${
          band === "semantic" && semanticViewMode === "topology" ? "semantic-topology-mode" : ""
        }`}
        style={{ flex: 1 }}
      >
        {band === "semantic" && semanticViewMode === "topology" ? (
          <div className="semantic-topology-pane">
            <SemanticTopologyPanel selectedId={selectedId} onSelectNode={setSelectedId} />
          </div>
        ) : (
          <div className="list">
            <div
              style={{
                padding: "8px 14px",
                borderBottom: "1px solid var(--line)",
                display: "flex",
                gap: 8,
                alignItems: "center",
                fontSize: 10.5,
                color: "var(--text-mute)",
              }}
            >
              <span>
                {filteredRows.length} visible · {loadedCount} loaded
              </span>
              <span style={{ flex: 1 }}></span>
              <div className="filter-pills">
                {(
                  ["backend", "updated_desc", "updated_asc", "created_desc", "created_asc"] as const
                ).map((mode) => (
                  <span
                    key={mode}
                    className={`pill ${sortMode === mode ? "on" : ""}`}
                    onClick={() => setSortMode(mode)}
                  >
                    {sortLabel(mode)}
                  </span>
                ))}
              </div>
            </div>
            <div
              style={{
                padding: "8px 14px",
                borderBottom: "1px solid var(--line)",
                display: "flex",
                gap: 8,
                alignItems: "center",
                flexWrap: "wrap",
              }}
            >
              <MemoryStructuralControls
                band={band}
                rows={rows}
                filters={filters}
                onFilter={setFilter}
              />
            </div>
            {api.loading && rows.length === 0 ? <div className="notice">loading {band}</div> : null}
            {api.error !== null ? <div className="notice bad">{api.error.message}</div> : null}
            {filteredRows.map((row) => (
              <div
                key={row.id}
                className={`list-row ${row.id === selected?.id ? "selected" : ""}`}
                onClick={() => setSelectedId(row.id)}
              >
                <div className="ttl">{row.title}</div>
                <div className="meta">
                  <span>[{shortId(row.id)}]</span>
                  <span>·</span>
                  <span>{row.meta}</span>
                </div>
              </div>
            ))}
            {filteredRows.length === 0 && !api.loading ? (
              <div className="notice">no records in current filter</div>
            ) : null}
            {nextCursor !== null ? (
              <div style={{ padding: 12 }}>
                <button
                  className="btn sm ghost"
                  disabled={loadingMore}
                  onClick={() => void loadMore()}
                >
                  {loadingMore ? "loading" : "load more"}
                </button>
              </div>
            ) : null}
          </div>
        )}
        <div className="detail">
          {selected === null ? (
            fetchedSemanticNodeLoading ? (
              <div className="notice">loading selected semantic node</div>
            ) : fetchedSemanticNodeError !== null ? (
              <div className="notice bad">{fetchedSemanticNodeError}</div>
            ) : (
              <div className="notice">no records in this band</div>
            )
          ) : selectedSemanticNode === null ? (
            <>
              <h2>{selected.title}</h2>
              <div className="meta-line">
                <span>[{selected.id}]</span>
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
                <span className="v">{selected?.id ?? selectedId ?? "—"}</span>
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
                      className="btn sm ghost"
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
                        className="btn sm ghost"
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
                <span className="dim" style={{ fontSize: 11 }}>
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
        title={action === null ? "correction" : `${action.kind} ${action.id}`}
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
              className="btn sm primary"
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
    return <div className="notice">loading corrections</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
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
  if (detail.band === "episodic") {
    const episode = detail.items.find((item) => item.id === selectedId) as
      | EpisodeMemoryItem
      | undefined;
    if (episode === undefined) {
      return null;
    }
    return (
      <>
        <div className="divider">citations</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {episode.source_stream_ids.map((id) => (
            <span key={id} className="tag info">
              {id}
            </span>
          ))}
        </div>
      </>
    );
  }

  if (detail.band === "affective") {
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
            <span className="v">{formatTime(detail.current.updated_at)}</span>
          </div>
        </div>
      </>
    );
  }

  if (detail.band === "relational") {
    return (
      <>
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
