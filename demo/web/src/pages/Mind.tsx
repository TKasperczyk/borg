import { type FormEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation } from "wouter";

import {
  ApiError,
  fetchBandDetail,
  fetchCommitments,
  fetchCreatorDirectives,
  fetchIdentity,
  fetchMemoryBands,
  fetchSemanticGraph,
  fetchSemanticNode,
  fetchState,
  invalidateSemanticEdge,
  patchGoal,
  patchOpenQuestion,
  postIdentityValue,
  revokeCommitment,
  revokeCreatorDirective,
  supersedeCreatorDirective,
} from "../api/client";
import type {
  BandDetailResponse,
  Commitment,
  CreatorDirective,
  CreatorDirectiveKind,
  IdentityGoal,
  IdentityOpenQuestion,
  IdentityPeriod,
  IdentityResponse,
  MemoryBandId,
  MemoryBandSummary,
  SemanticGraphEdge,
  SemanticGraphNode,
  SemanticGraphResponse,
  SemanticNodeDetail,
} from "../api/types";
import { useQuery } from "../api/useQuery";
import { dayLabel, hm } from "../format/time";
import { moodLabel } from "../state/mood";
import { edgeStyleForType, layoutGraph, nodeStatusColor } from "./mind/graph";

const INSPECTOR_SECTIONS = ["identity", "directives", "graph"] as const;
const BAND_IDS: MemoryBandId[] = [
  "episodic",
  "semantic",
  "procedural",
  "affective",
  "self",
  "commitments",
  "social",
  "relational",
];

type InspectorSection = (typeof INSPECTOR_SECTIONS)[number] | MemoryBandId;
type MindAtlasTab = "identity" | "directives" | "memory" | "graph";
type Filter = "active" | "all";
type Toast = { text: string; tone: "ok" | "error" };

const MIND_ATLAS_TABS: Array<{ id: MindAtlasTab; label: string; path: string }> = [
  { id: "identity", label: "IDENTITY", path: "/mind" },
  { id: "directives", label: "DIRECTIVES", path: "/mind/directives" },
  { id: "memory", label: "MEMORY", path: "/mind/memory" },
  { id: "graph", label: "GRAPH", path: "/mind/graph" },
];

function isInspectorSection(value: string): value is InspectorSection {
  return [...INSPECTOR_SECTIONS, ...BAND_IDS].includes(value as InspectorSection);
}

function activeInspectorSection(path: string): InspectorSection | null {
  const prefix = "/mind/inspect/";
  if (!path.startsWith(prefix)) {
    return null;
  }

  const raw = decodeURIComponent(path.slice(prefix.length));
  return isInspectorSection(raw) ? raw : null;
}

function activeAtlasTab(path: string): MindAtlasTab {
  if (path === "/mind" || path === "/mind/") {
    return "identity";
  }

  const prefix = "/mind/";
  if (!path.startsWith(prefix)) {
    return "identity";
  }

  const raw = decodeURIComponent(path.slice(prefix.length));
  return MIND_ATLAS_TABS.some((tab) => tab.id === raw) ? (raw as MindAtlasTab) : "identity";
}

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.status} ${error.message}`;
  }

  return error instanceof Error ? error.message : String(error);
}

function clamp(value: number, min = 0, max = 1): number {
  return Math.min(max, Math.max(min, value));
}

function barValue(value: number): number {
  return value > 1 ? clamp(value / 10) : clamp(value);
}

const TRAIT_PREVIEW_COUNT = 10;
const GOAL_PREVIEW_COUNT = 6;
const OPEN_QUESTION_PREVIEW_COUNT = 8;

function dateText(ts: number | null | undefined): string {
  return typeof ts === "number" ? dayLabel(new Date(ts)) : "—";
}

function rangeText(period: IdentityPeriod): string {
  const start = dayLabel(new Date(period.start_ts));
  const end = period.end_ts === null ? "now" : dayLabel(new Date(period.end_ts));
  return `${start} – ${end}`;
}

function currentPeriod(periods: readonly IdentityPeriod[]): IdentityPeriod | null {
  return periods.find((period) => period.end_ts === null) ?? periods[0] ?? null;
}

function flattenGoals(goals: readonly IdentityGoal[]): IdentityGoal[] {
  return goals.flatMap((goal) => [goal, ...flattenGoals(goal.children ?? [])]);
}

function statline(stats: MemoryBandSummary["stats"]): string {
  return stats.map((stat) => `${stat.k} ${stat.v}`).join(" · ");
}

function directiveKindChip(kind: CreatorDirectiveKind): string {
  const labels: Record<CreatorDirectiveKind, string> = {
    self_identity: "SELF",
    subject_fact: "FACT",
    disclosure_boundary: "BOUND",
    response_policy: "RESP",
    routing_instruction: "ROUTE",
  };
  return labels[kind];
}

function directiveCount(directives: readonly CreatorDirective[]): string {
  const active = directives.filter((directive) => directive.status === "active").length;
  const inactive = directives.length - active;
  return `${active} active · ${inactive} revoked/superseded`;
}

function commitmentCount(commitments: readonly Commitment[]): string {
  const active = commitments.filter((commitment) => commitment.state === "active").length;
  const inactive = commitments.length - active;
  return `${active} active · ${inactive} revoked/expired`;
}

export function commitmentExpiresSoon(commitment: Pick<Commitment, "expires_at" | "state">, now = Date.now()): boolean {
  if (commitment.state !== "active" || commitment.expires_at === null) {
    return false;
  }

  return commitment.expires_at > now && commitment.expires_at - now <= 24 * 60 * 60 * 1000;
}

function recordText(record: Record<string, unknown>): string {
  const keys = [
    "title",
    "narrative",
    "description",
    "applies_when",
    "approach",
    "label",
    "slot",
    "value",
    "text",
    "question",
    "what_changed",
  ];
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.length > 0) {
      return value;
    }
  }

  return record.id === undefined ? "record" : String(record.id);
}

function recordMeta(record: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const key of ["status", "state", "kind", "source", "disclosure_class", "audience"]) {
    const value = record[key];
    if (typeof value === "string" && value.length > 0) {
      parts.push(`${key} ${value}`);
    }
  }
  for (const key of ["created_at", "updated_at", "start_time", "last_interaction_at"]) {
    const value = record[key];
    if (typeof value === "number") {
      parts.push(`${key} ${dateText(value)}`);
      break;
    }
  }
  for (const key of ["source_count", "sample_count", "sources_count"]) {
    const value = record[key];
    if (typeof value === "number") {
      parts.push(`${key} ${value}`);
    }
  }

  return parts.join(" · ");
}

function recordRight(record: Record<string, unknown>): string {
  for (const key of ["salience", "significance", "confidence", "trust", "priority"]) {
    const value = record[key];
    if (typeof value === "number") {
      return `${key} ${value.toFixed(2)}`;
    }
  }

  for (const key of ["state", "status"]) {
    const value = record[key];
    if (typeof value === "string") {
      return value.toUpperCase();
    }
  }

  return "";
}

function mergeByStableId<T extends { id?: unknown }>(existing: readonly T[] = [], incoming: readonly T[] = []): T[] {
  const merged: T[] = [];
  const seen = new Set<string>();

  for (const record of [...existing, ...incoming]) {
    const key = record.id === undefined || record.id === null ? JSON.stringify(record) : String(record.id);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(record);
  }

  return merged;
}

function identityRecordCount(identity: IdentityResponse | undefined): number {
  if (identity === undefined) {
    return 0;
  }

  return (
    identity.values.length +
    identity.goals.length +
    identity.open_questions.length +
    identity.growth_markers.length +
    identity.periods.length
  );
}

function useToast(): [Toast | null, (toast: Toast) => void] {
  const [toast, setToast] = useState<Toast | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const showToast = useCallback((next: Toast) => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
    }
    setToast(next);
    timeoutRef.current = window.setTimeout(() => {
      setToast(null);
      timeoutRef.current = null;
    }, 2400);
  }, []);

  useEffect(
    () => () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
      }
    },
    [],
  );

  return [toast, showToast];
}

export function MindPage() {
  const [location, navigate] = useLocation();
  const section = activeInspectorSection(location);
  const activeTab = activeAtlasTab(location);
  const [toast, showToast] = useToast();

  const state = useQuery("state:mind-default", () => fetchState());
  const identity = useQuery("identity", fetchIdentity);
  const directives = useQuery("directives:all", () => fetchCreatorDirectives("all"));
  const commitments = useQuery("commitments:all", () => fetchCommitments("all"));
  const bands = useQuery("bands", () => fetchMemoryBands());
  const graph = useQuery("graph:60", () => fetchSemanticGraph(60));

  const refetchMind = useCallback(() => {
    identity.refetch();
    directives.refetch();
    commitments.refetch();
    bands.refetch();
    graph.refetch();
  }, [bands, commitments, directives, graph, identity]);

  const mood = state.data?.current_mood ?? null;
  const moodText = mood === null ? "level" : moodLabel(mood.valence, mood.arousal);
  const moodNums =
    mood === null
      ? "v +0.00 · a 0.00"
      : `v ${mood.valence >= 0 ? "+" : ""}${mood.valence.toFixed(2)} · a ${mood.arousal.toFixed(2)}`;

  return (
    <main className="mind-page">
      <header className="mind-header">
        <span className="page-title">MIND</span>
        <span className="page-subtitle">live internals -- identity / directives / memory / belief graph</span>
        {section === null ? <MindTabBar activeTab={activeTab} navigate={navigate} /> : null}
        <div className="mind-mood-chip" aria-label="Mood">
          <span>MOOD</span>
          <span className="mind-mood-square" />
          <strong>{moodText}</strong>
          <span>{moodNums}</span>
        </div>
      </header>

      {section === null ? (
        <MindAtlas
          activeTab={activeTab}
          identity={identity.data}
          identityLoading={identity.loading}
          directives={directives.data?.directives ?? []}
          commitments={commitments.data?.commitments ?? []}
          bands={bands.data?.bands ?? []}
          graph={graph.data}
          navigate={navigate}
          showToast={showToast}
          refetchMind={refetchMind}
        />
      ) : (
        <MindInspector
          section={section}
          identity={identity.data}
          directives={directives.data?.directives ?? []}
          commitments={commitments.data?.commitments ?? []}
          bands={bands.data?.bands ?? []}
          graph={graph.data}
          navigate={navigate}
          showToast={showToast}
          refetchMind={refetchMind}
        />
      )}

      {toast === null ? null : <ToastView toast={toast} />}
    </main>
  );
}

function MindTabBar({
  activeTab,
  navigate,
}: {
  activeTab: MindAtlasTab;
  navigate: (path: string) => void;
}) {
  return (
    <div className="activity-tabs mind-tabs" aria-label="Mind sections">
      {MIND_ATLAS_TABS.map((tab) => (
        <button
          className={activeTab === tab.id ? "activity-tab activity-tab-active" : "activity-tab"}
          key={tab.id}
          type="button"
          onClick={() => navigate(tab.path)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function MindAtlas({
  activeTab,
  identity,
  identityLoading,
  directives,
  commitments,
  bands,
  graph,
  navigate,
  showToast,
  refetchMind,
}: {
  activeTab: MindAtlasTab;
  identity: IdentityResponse | undefined;
  identityLoading: boolean;
  directives: CreatorDirective[];
  commitments: Commitment[];
  bands: MemoryBandSummary[];
  graph: SemanticGraphResponse | undefined;
  navigate: (path: string) => void;
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const [directiveFilter, setDirectiveFilter] = useState<Filter>("active");
  const [commitmentFilter, setCommitmentFilter] = useState<Filter>("active");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const graphNodes = graph?.nodes ?? [];
  const graphEdges = graph?.edges ?? [];
  const selectedNode = graphNodes.find((node) => node.id === selectedNodeId) ?? graphNodes[0] ?? null;
  const graphLayout = useMemo(
    () => layoutGraph(graphNodes, graphEdges),
    [graphEdges, graphNodes],
  );

  useEffect(() => {
    if (selectedNodeId === null && graphNodes[0] !== undefined) {
      setSelectedNodeId(graphNodes[0].id);
    }
  }, [graphNodes, selectedNodeId]);

  const period = currentPeriod(identity?.periods ?? []);
  const shownDirectives =
    directiveFilter === "active" ? directives.filter((directive) => directive.status === "active") : directives;
  const shownCommitments =
    commitmentFilter === "active"
      ? commitments.filter((commitment) => commitment.state === "active")
      : commitments;

  return (
    <div className="mind-atlas">
      {activeTab === "identity" ? (
        <section className="mind-panel mind-tab-panel">
          <PanelHead
            title="IDENTITY"
            subtitle="/api/identity -- values, goals, traits, open questions, narrative"
            right={period === null ? "" : `period: ${period.label} (${rangeText(period)})`}
          />
          {identityLoading && identity === undefined ? (
            <div className="mind-empty">loading identity…</div>
          ) : (
            <IdentityGrid identity={identity} showToast={showToast} refetchMind={refetchMind} />
          )}
        </section>
      ) : null}

      {activeTab === "directives" ? (
        <div className="mind-tab-panel mind-ledger-grid">
          <section className="mind-panel">
            <PanelHead
              title="CREATOR-DIRECTIVES"
              titleAccent
              subtitle="factual / operational guidance · trusted briefing"
              right={directiveCount(directives)}
            />
            <FilterRow value={directiveFilter} onChange={setDirectiveFilter} />
            <DirectiveLedger
              directives={shownDirectives}
              allDirectives={directives}
              showToast={showToast}
              refetchMind={refetchMind}
            />
          </section>
          <section className="mind-panel">
            <PanelHead
              title="COMMITMENTS"
              titleAccent
              subtitle="scoped promises & boundaries · enforced at emission"
              right={commitmentCount(commitments)}
            />
            <FilterRow value={commitmentFilter} onChange={setCommitmentFilter} />
            <CommitmentLedger
              commitments={shownCommitments}
              showToast={showToast}
              refetchMind={refetchMind}
            />
          </section>
        </div>
      ) : null}

      {activeTab === "memory" ? (
        <section className="mind-panel mind-tab-panel">
          <PanelHead title="MEMORY BANDS" subtitle="/api/memory/bands -- 8 bands, recall is global" />
          <div className="band-card-grid">
            {bands.map((band) => (
              <button
                key={band.id}
                className="band-card"
                type="button"
                onClick={() => navigate(`/mind/inspect/${band.id}`)}
              >
                <div className="band-card-top">
                  <span>{band.n}</span>
                  <strong>{band.name}</strong>
                  <b>{band.count_is_lower_bound ? `${band.count}+` : band.count}</b>
                </div>
                <div className="band-card-desc">{band.desc}</div>
                <div className="band-card-stats">{statline(band.stats)}</div>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {activeTab === "graph" ? (
        <section className="mind-panel mind-tab-panel graph-panel graph-tab-panel">
          <PanelHead
            title="BELIEF GRAPH"
            subtitle={`/api/semantic/graph · ${graph?.total_nodes ?? 0} nodes · ${graph?.total_edges ?? 0} edges · rendered ${graph?.rendered?.nodes ?? 0}/${graph?.rendered?.edges ?? 0}`}
          />
          <GraphCanvas
            layout={graphLayout}
            selectedNodeId={selectedNode?.id ?? null}
            onSelect={setSelectedNodeId}
          />
          <GraphSelectionStrip
            node={selectedNode}
            graphEdges={graphEdges}
            showToast={showToast}
            refetchMind={refetchMind}
          />
        </section>
      ) : null}
    </div>
  );
}

function IdentityGrid({
  identity,
  showToast,
  refetchMind,
}: {
  identity: IdentityResponse | undefined;
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const [showAllGoals, setShowAllGoals] = useState(false);
  const [showAllOpenQuestions, setShowAllOpenQuestions] = useState(false);
  const [showAllTraits, setShowAllTraits] = useState(false);
  const goals = [...flattenGoals(identity?.goals ?? [])].sort((left, right) => right.priority - left.priority);
  const visibleGoals = showAllGoals ? goals : goals.slice(0, GOAL_PREVIEW_COUNT);
  const openQuestions = [...(identity?.open_questions ?? [])].sort((left, right) => right.urgency - left.urgency);
  const visibleOpenQuestions = showAllOpenQuestions
    ? openQuestions
    : openQuestions.slice(0, OPEN_QUESTION_PREVIEW_COUNT);
  const traits = [...(identity?.traits ?? [])].sort((left, right) => right.strength - left.strength);
  const visibleTraits = showAllTraits ? traits : traits.slice(0, TRAIT_PREVIEW_COUNT);
  const traitStateCounts = new Map<string, number>();
  for (const trait of traits) {
    traitStateCounts.set(trait.state, (traitStateCounts.get(trait.state) ?? 0) + 1);
  }
  const traitStateMeta = [...traitStateCounts.entries()]
    .map(([state, count]) => `${count} ${state}`)
    .join(" · ");
  const markers = [...(identity?.growth_markers ?? [])]
    .sort((left, right) => right.ts - left.ts)
    .slice(0, 3);

  return (
    <div className="identity-grid">
      <div className="identity-cell">
        <Subhead>VALUES</Subhead>
        {(identity?.values ?? []).length === 0 ? <div className="quiet-line">none recorded</div> : null}
        {(identity?.values ?? []).map((value) => {
          const width = barValue(value.priority) * 100;
          return (
            <div key={value.id} className="value-row">
              <span>{value.label}</span>
              <div className="mini-track">
                <div className="mini-fill" style={{ width: `${width}%` }} />
              </div>
              <b>{value.priority.toFixed(2)}</b>
            </div>
          );
        })}
        <AddValueForm showToast={showToast} refetchMind={refetchMind} />
      </div>
      <div className="identity-cell">
        <Subhead>
          TRAITS
          {traitStateMeta.length > 0 ? <span className="mind-subhead-meta">{traitStateMeta}</span> : null}
        </Subhead>
        {traits.length === 0 ? <div className="quiet-line">none recorded</div> : null}
        {visibleTraits.map((trait) => (
          <div
            key={trait.id}
            className={trait.state === "established" ? "value-row trait-row trait-row-established" : "value-row trait-row"}
            title={trait.label}
          >
            <span>{trait.label}</span>
            <div className="mini-track">
              <div className="mini-fill" style={{ width: `${barValue(trait.strength) * 100}%` }} />
            </div>
            <b>{trait.strength.toFixed(2)}</b>
          </div>
        ))}
        <PreviewToggle
          total={traits.length}
          previewCount={TRAIT_PREVIEW_COUNT}
          expanded={showAllTraits}
          collapsedLabel={`ALL ${traits.length}`}
          expandedLabel={`STRONGEST ${TRAIT_PREVIEW_COUNT}`}
          onToggle={() => setShowAllTraits((open) => !open)}
        />
        <Subhead className="subhead-spaced">GROWTH MARKERS</Subhead>
        {markers.map((marker) => (
          <div key={marker.id} className="marker-line">
            {dayLabel(new Date(marker.ts))} -- {marker.what_changed}
          </div>
        ))}
      </div>
      <div className="identity-cell">
        <Subhead>GOALS</Subhead>
        {visibleGoals.map((goal) => (
          <GoalRow key={goal.id} goal={goal} showToast={showToast} refetchMind={refetchMind} />
        ))}
        <PreviewToggle
          total={goals.length}
          previewCount={GOAL_PREVIEW_COUNT}
          expanded={showAllGoals}
          collapsedLabel={`ALL ${goals.length}`}
          expandedLabel={`TOP ${GOAL_PREVIEW_COUNT}`}
          onToggle={() => setShowAllGoals((open) => !open)}
        />
      </div>
      <div className="identity-cell">
        <Subhead>OPEN QUESTIONS</Subhead>
        {visibleOpenQuestions.map((question) => (
          <OpenQuestionRow
            key={question.id}
            question={question}
            showToast={showToast}
            refetchMind={refetchMind}
          />
        ))}
        <PreviewToggle
          total={openQuestions.length}
          previewCount={OPEN_QUESTION_PREVIEW_COUNT}
          expanded={showAllOpenQuestions}
          collapsedLabel={`ALL ${openQuestions.length}`}
          expandedLabel={`TOP ${OPEN_QUESTION_PREVIEW_COUNT}`}
          onToggle={() => setShowAllOpenQuestions((open) => !open)}
        />
      </div>
    </div>
  );
}

function PreviewToggle({
  total,
  previewCount,
  expanded,
  collapsedLabel,
  expandedLabel,
  onToggle,
}: {
  total: number;
  previewCount: number;
  expanded: boolean;
  collapsedLabel: string;
  expandedLabel: string;
  onToggle: () => void;
}) {
  if (total <= previewCount) {
    return null;
  }

  return (
    <button type="button" className="load-more trait-toggle" onClick={onToggle}>
      {expanded ? expandedLabel : collapsedLabel}
    </button>
  );
}

function AddValueForm({
  showToast,
  refetchMind,
}: {
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [pending, setPending] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const trimmedName = name.trim();
    const trimmedDescription = description.trim();
    if (pending || trimmedName.length === 0) {
      return;
    }

    setPending(true);
    try {
      await postIdentityValue({
        name: trimmedName,
        ...(trimmedDescription.length === 0 ? {} : { description: trimmedDescription }),
      });
      showToast({ text: "value added", tone: "ok" });
      setName("");
      setDescription("");
      setOpen(false);
      refetchMind();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPending(false);
    }
  };

  if (!open) {
    return (
      <button type="button" className="ensure-session add-value-button" onClick={() => setOpen(true)}>
        + ADD VALUE
      </button>
    );
  }

  return (
    <form className="inline-confirm add-value-form" onSubmit={(event) => void submit(event)}>
      <input
        value={name}
        onChange={(event) => setName(event.target.value)}
        placeholder="value"
        disabled={pending}
      />
      <input
        value={description}
        onChange={(event) => setDescription(event.target.value)}
        placeholder="description"
        disabled={pending}
      />
      <button type="submit" disabled={pending || name.trim().length === 0}>
        CONFIRM
      </button>
      <button type="button" onClick={() => setOpen(false)} disabled={pending}>
        CANCEL
      </button>
    </form>
  );
}

function GoalRow({
  goal,
  showToast,
  refetchMind,
}: {
  goal: IdentityGoal;
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const [blocking, setBlocking] = useState(false);
  const [reason, setReason] = useState("");
  const [pendingAction, setPendingAction] = useState<"bump" | "block" | null>(null);

  const bump = async () => {
    if (pendingAction !== null) {
      return;
    }
    setPendingAction("bump");
    try {
      await patchGoal(goal.id, {
        action: "progress",
        progress: Math.min(100, Math.round(barValue(goal.priority) * 100 + 10)),
        note: "operator bump +0.1",
      });
      showToast({ text: "goal progress updated", tone: "ok" });
      refetchMind();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPendingAction(null);
    }
  };

  const block = async (event: FormEvent) => {
    event.preventDefault();
    if (pendingAction !== null || reason.trim().length === 0) {
      return;
    }
    setPendingAction("block");
    try {
      await patchGoal(goal.id, { action: "block", note: reason.trim() });
      showToast({ text: "goal blocked", tone: "ok" });
      setBlocking(false);
      setReason("");
      refetchMind();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPendingAction(null);
    }
  };

  return (
    <div className="goal-row">
      <div className="goal-main">
        <span className={goal.status === "active" ? "status-chip status-active" : "status-chip"}>
          {goal.status.toUpperCase()}
        </span>
        <span>{goal.description}</span>
      </div>
      <div className="goal-actions">
        <span>prio {goal.priority.toFixed(2)}</span>
        <button type="button" onClick={() => void bump()} disabled={pendingAction !== null}>
          BUMP +
        </button>
        {goal.status === "active" ? (
          <button type="button" onClick={() => setBlocking(true)} disabled={pendingAction !== null}>
            BLOCK
          </button>
        ) : null}
      </div>
      {blocking ? (
        <form className="inline-confirm" onSubmit={(event) => void block(event)}>
          <input
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            placeholder="reason"
            disabled={pendingAction !== null}
          />
          <button type="submit" disabled={pendingAction !== null || reason.trim().length === 0}>
            CONFIRM
          </button>
          <button type="button" onClick={() => setBlocking(false)} disabled={pendingAction !== null}>
            CANCEL
          </button>
        </form>
      ) : null}
    </div>
  );
}

function OpenQuestionRow({
  question,
  showToast,
  refetchMind,
}: {
  question: IdentityOpenQuestion;
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const [resolving, setResolving] = useState(false);
  const [resolution, setResolution] = useState("");
  const [pending, setPending] = useState(false);
  const barColor = question.urgency > 0.6 ? "#C9A227" : "var(--ac)";

  const resolve = async (event: FormEvent) => {
    event.preventDefault();
    if (pending || resolution.trim().length === 0) {
      return;
    }
    setPending(true);
    try {
      await patchOpenQuestion(question.id, { action: "resolve", resolution: resolution.trim() });
      showToast({ text: "open question resolved", tone: "ok" });
      setResolving(false);
      setResolution("");
      refetchMind();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="question-row">
      <div>{question.question}</div>
      <div className="question-meta">
        <span>urgency</span>
        <div className="mini-track">
          <div
            className="mini-fill"
            style={{ width: `${clamp(question.urgency) * 100}%`, background: barColor }}
          />
        </div>
        <span>
          {question.source} · touched {dateText(question.last_touched)}
        </span>
        {question.status === "open" ? (
          <button type="button" onClick={() => setResolving(true)}>
            RESOLVE
          </button>
        ) : null}
      </div>
      {resolving ? (
        <form className="inline-confirm" onSubmit={(event) => void resolve(event)}>
          <input
            value={resolution}
            onChange={(event) => setResolution(event.target.value)}
            placeholder="resolution"
            disabled={pending}
          />
          <button type="submit" disabled={pending || resolution.trim().length === 0}>
            CONFIRM
          </button>
          <button type="button" onClick={() => setResolving(false)} disabled={pending}>
            CANCEL
          </button>
        </form>
      ) : null}
    </div>
  );
}

function DirectiveLedger({
  directives,
  allDirectives,
  showToast,
  refetchMind,
}: {
  directives: CreatorDirective[];
  allDirectives: CreatorDirective[];
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  if (directives.length === 0) {
    return <div className="mind-empty">0 records</div>;
  }

  return (
    <>
      {directives.map((directive) => (
        <DirectiveRow
          key={directive.id}
          directive={directive}
          activeReplacements={allDirectives.filter(
            (candidate) => candidate.status === "active" && candidate.id !== directive.id,
          )}
          showToast={showToast}
          refetchMind={refetchMind}
        />
      ))}
    </>
  );
}

function DirectiveRow({
  directive,
  activeReplacements,
  showToast,
  refetchMind,
}: {
  directive: CreatorDirective;
  activeReplacements: CreatorDirective[];
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const [mode, setMode] = useState<"none" | "revoke" | "supersede">("none");
  const [reason, setReason] = useState("");
  const [replacementId, setReplacementId] = useState(activeReplacements[0]?.id ?? "");
  const [pending, setPending] = useState(false);
  const inactive = directive.status !== "active";

  const revoke = async (event: FormEvent) => {
    event.preventDefault();
    if (pending || reason.trim().length === 0) {
      return;
    }
    setPending(true);
    try {
      await revokeCreatorDirective(directive.id, reason.trim());
      showToast({ text: "creator directive revoked", tone: "ok" });
      setMode("none");
      setReason("");
      refetchMind();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPending(false);
    }
  };

  const supersede = async (event: FormEvent) => {
    event.preventDefault();
    if (pending || replacementId.length === 0) {
      return;
    }
    setPending(true);
    try {
      await supersedeCreatorDirective(directive.id, replacementId);
      showToast({ text: "creator directive superseded", tone: "ok" });
      setMode("none");
      refetchMind();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="ledger-row">
      <div className="ledger-main">
        <span className={directive.kind === "subject_fact" ? "kind-chip kind-dim" : "kind-chip"}>
          {directiveKindChip(directive.kind)}
        </span>
        <span className={inactive ? "ledger-text ledger-text-inactive" : "ledger-text"}>
          {directive.text ?? directive.id}
        </span>
      </div>
      <div className="ledger-meta">
        <span>{directive.id}</span>
        <span>scope: {directive.content_scope}</span>
        <span>mention: {directive.mention_policy}</span>
        <span>prio {directive.priority}</span>
        <span>status: {directive.status}</span>
        {directive.status === "active" ? (
          <>
            <button type="button" onClick={() => setMode("supersede")} disabled={pending}>
              SUPERSEDE
            </button>
            <button type="button" onClick={() => setMode("revoke")} disabled={pending}>
              REVOKE
            </button>
          </>
        ) : null}
      </div>
      {mode === "revoke" ? (
        <form className="inline-confirm" onSubmit={(event) => void revoke(event)}>
          <input
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            placeholder="reason"
            disabled={pending}
          />
          <button type="submit" disabled={pending || reason.trim().length === 0}>
            CONFIRM
          </button>
          <button type="button" onClick={() => setMode("none")} disabled={pending}>
            CANCEL
          </button>
        </form>
      ) : null}
      {mode === "supersede" ? (
        <form className="replacement-picker" onSubmit={(event) => void supersede(event)}>
          {activeReplacements.map((candidate) => (
            <label key={candidate.id}>
              <input
                type="radio"
                name={`replacement-${directive.id}`}
                value={candidate.id}
                checked={replacementId === candidate.id}
                onChange={() => setReplacementId(candidate.id)}
                disabled={pending}
              />
              <span>{candidate.text ?? candidate.id}</span>
            </label>
          ))}
          <button type="submit" disabled={pending || replacementId.length === 0}>
            CONFIRM
          </button>
          <button type="button" onClick={() => setMode("none")} disabled={pending}>
            CANCEL
          </button>
        </form>
      ) : null}
    </div>
  );
}

function CommitmentLedger({
  commitments,
  showToast,
  refetchMind,
}: {
  commitments: Commitment[];
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  if (commitments.length === 0) {
    return <div className="mind-empty">0 records</div>;
  }

  return (
    <>
      {commitments.map((commitment) => (
        <CommitmentRow
          key={commitment.id}
          commitment={commitment}
          showToast={showToast}
          refetchMind={refetchMind}
        />
      ))}
    </>
  );
}

function CommitmentRow({
  commitment,
  showToast,
  refetchMind,
}: {
  commitment: Commitment;
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const [revoking, setRevoking] = useState(false);
  const [reason, setReason] = useState("");
  const [pending, setPending] = useState(false);
  const inactive = commitment.state !== "active";
  const soon = commitmentExpiresSoon(commitment);

  const revoke = async (event: FormEvent) => {
    event.preventDefault();
    if (pending) {
      return;
    }
    setPending(true);
    try {
      await revokeCommitment(commitment.id, reason.trim());
      showToast({ text: "commitment revoked", tone: "ok" });
      setRevoking(false);
      setReason("");
      refetchMind();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="ledger-row">
      <div className="ledger-main">
        <span
          className={
            commitment.enforcement_class === "critical"
              ? "enforcement-chip enforcement-critical"
              : "enforcement-chip"
          }
        >
          {commitment.enforcement_class.toUpperCase()}
        </span>
        <span className={inactive ? "ledger-text ledger-text-inactive" : "ledger-text"}>
          {commitment.text}
        </span>
      </div>
      <div className="ledger-meta">
        <span>{commitment.id}</span>
        <span>{commitment.kind}</span>
        <span>audience: {commitment.audience ?? "—"}</span>
        <span className={soon ? "expires-soon" : ""}>
          {commitment.expires_at === null ? "no expiry" : `expires ${dateText(commitment.expires_at)}`}
        </span>
        <span>state: {commitment.state}</span>
        {commitment.state === "active" ? (
          <button type="button" onClick={() => setRevoking(true)} disabled={pending}>
            REVOKE
          </button>
        ) : null}
      </div>
      {revoking ? (
        <form className="inline-confirm" onSubmit={(event) => void revoke(event)}>
          <input
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            placeholder="reason"
            disabled={pending}
          />
          <button type="submit" disabled={pending}>
            CONFIRM
          </button>
          <button type="button" onClick={() => setRevoking(false)} disabled={pending}>
            CANCEL
          </button>
        </form>
      ) : null}
    </div>
  );
}

function GraphCanvas({
  layout,
  selectedNodeId,
  onSelect,
}: {
  layout: ReturnType<typeof layoutGraph>;
  selectedNodeId: string | null;
  onSelect: (nodeId: string) => void;
}) {
  return (
    <div className="graph-canvas-wrap">
      <svg viewBox="0 0 618 312" className="graph-canvas" role="img" aria-label="Belief graph">
        {layout.edges.map((edge) => {
          const style = edgeStyleForType(edge.type);
          return (
            <line
              key={edge.id}
              x1={edge.x1}
              y1={edge.y1}
              x2={edge.x2}
              y2={edge.y2}
              stroke={style.stroke}
              strokeWidth={style.strokeWidth}
              strokeDasharray={style.strokeDasharray}
            />
          );
        })}
        {layout.nodes.map((node) => {
          const color = nodeStatusColor(node.status);
          const selected = node.id === selectedNodeId;
          return (
            <g key={node.id}>
              <circle
                cx={node.x}
                cy={node.y}
                r={node.r}
                fill={node.kind === "entity" ? "#0B0B09" : color}
                stroke={color}
                strokeWidth={selected ? 2.5 : 1.4}
                role="button"
                aria-label={`select ${node.display_label ?? node.label}`}
                tabIndex={0}
                onClick={() => onSelect(node.id)}
              />
              <text x={node.x} y={node.y + node.r + 11} textAnchor="middle">
                {(node.display_label ?? node.label).slice(0, 24)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="graph-legend">
        <span>
          <b className="legend-active">●</b> active
        </span>
        <span>
          <b className="legend-contested">●</b> contested
        </span>
        <span>
          <b className="legend-contradicted">●</b> contradicted
        </span>
        <span>
          <b className="legend-quarantined">●</b> quarantined
        </span>
        <span>-- supports · -- contradicts</span>
      </div>
    </div>
  );
}

function GraphSelectionStrip({
  node,
  graphEdges,
  showToast,
  refetchMind,
}: {
  node: SemanticGraphNode | null;
  graphEdges: SemanticGraphEdge[];
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const [showEdges, setShowEdges] = useState(false);
  const [reasonByEdge, setReasonByEdge] = useState<Record<string, string>>({});
  const [pendingEdgeId, setPendingEdgeId] = useState<string | null>(null);
  const detail = useQuery<{ node: SemanticNodeDetail | null }>(
    `graph-node:${node?.id ?? ""}`,
    () => (node === null ? Promise.resolve({ node: null }) : fetchSemanticNode(node.id)),
  );
  const nodeEdges =
    node === null ? [] : graphEdges.filter((edge) => edge.source === node.id || edge.target === node.id);
  const detailNode = node !== null && detail.data?.node?.id === node.id ? detail.data.node : null;
  const loadingSelectedDetail = node !== null && detailNode === null && detail.loading;

  const invalidate = async (edgeId: string) => {
    if (pendingEdgeId !== null) {
      return;
    }
    setPendingEdgeId(edgeId);
    try {
      await invalidateSemanticEdge(edgeId, reasonByEdge[edgeId]);
      showToast({ text: "semantic edge invalidated", tone: "ok" });
      refetchMind();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPendingEdgeId(null);
    }
  };

  if (node === null) {
    return <div className="graph-strip mind-empty">select a node to inspect</div>;
  }

  return (
    <div className="graph-strip">
      <div className="graph-selected-main">
        <span className={`graph-status graph-status-${node.status}`}>{node.status.toUpperCase()}</span>
        <strong>{node.display_label ?? node.label}</strong>
        <span>
          {node.kind} · {node.edge_count} edges
        </span>
      </div>
      {loadingSelectedDetail ? <div className="graph-detail-line mind-empty">loading node detail…</div> : null}
      {detailNode === null ? null : (
        <div className="graph-detail-line">
          confidence {detailNode.confidence.toFixed(2)} · sources {detailNode.source_count}
          {detailNode.domain === null ? "" : ` · domain ${detailNode.domain}`}
        </div>
      )}
      <div className="graph-actions">
        <span>evidence: {detailNode?.description ?? `${node.edge_count} graph edges`}</span>
        {nodeEdges.length === 0 ? null : (
          <button type="button" onClick={() => setShowEdges((current) => !current)}>
            INVALIDATE EDGE
          </button>
        )}
        <Link href="/reviews">OPEN IN REVIEWS →</Link>
      </div>
      {showEdges ? (
        <div className="edge-list">
          {nodeEdges.map((edge) => (
            <div key={edge.id} className="edge-action-row">
              <span>
                {edge.type} · {edge.source} → {edge.target}
              </span>
              <input
                value={reasonByEdge[edge.id] ?? ""}
                onChange={(event) =>
                  setReasonByEdge((current) => ({ ...current, [edge.id]: event.target.value }))
                }
                placeholder="reason"
                disabled={pendingEdgeId !== null}
              />
              <button type="button" onClick={() => void invalidate(edge.id)} disabled={pendingEdgeId !== null}>
                POST
              </button>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function MindInspector({
  section,
  identity,
  directives,
  commitments,
  bands,
  graph,
  navigate,
  showToast,
  refetchMind,
}: {
  section: InspectorSection;
  identity: IdentityResponse | undefined;
  directives: CreatorDirective[];
  commitments: Commitment[];
  bands: MemoryBandSummary[];
  graph: SemanticGraphResponse | undefined;
  navigate: (path: string) => void;
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const activeBand = bands.find((band) => band.id === section) ?? null;
  const title =
    section === "identity"
      ? "identity"
      : section === "directives"
        ? "directives"
        : section === "graph"
          ? "belief graph"
          : `${activeBand?.n ?? ""} / ${section}`;
  const meta =
    section === "identity"
      ? "/api/identity -- governed records"
      : section === "directives"
        ? "creator-directives vs commitments"
        : section === "graph"
          ? `${graph?.total_nodes ?? 0} nodes · ${graph?.total_edges ?? 0} edges`
          : `${activeBand?.desc ?? ""} -- ${activeBand === null ? "" : statline(activeBand.stats)}`;

  return (
    <div className="mind-inspector">
      <div className="mind-index">
        <IndexRow
          code="ID"
          name="identity"
          count={String(identityRecordCount(identity))}
          desc="values, goals, traits, questions"
          active={section === "identity"}
          onClick={() => navigate("/mind/inspect/identity")}
        />
        <IndexRow
          code="DR"
          name="directives"
          count={`${directives.length}+${commitments.length}`}
          desc="creator-directives · commitments"
          active={section === "directives"}
          onClick={() => navigate("/mind/inspect/directives")}
        />
        <IndexRow
          code="GR"
          name="belief graph"
          count={String(graph?.total_nodes ?? 0)}
          desc="semantic nodes + edges"
          active={section === "graph"}
          onClick={() => navigate("/mind/inspect/graph")}
        />
        {bands.map((band) => (
          <IndexRow
            key={band.id}
            code={band.n}
            name={band.name}
            count={String(band.count)}
            desc={band.desc}
            active={section === band.id}
            onClick={() => navigate(`/mind/inspect/${band.id}`)}
          />
        ))}
      </div>
      <div className="inspector-detail">
        <div className="inspector-head">
          <strong>{title}</strong>
          <span>{meta}</span>
        </div>
        <div className="inspector-rows">
          {section === "identity" ? (
            <IdentityInspectorRows identity={identity} showToast={showToast} refetchMind={refetchMind} />
          ) : section === "directives" ? (
            <DirectiveInspectorRows directives={directives} commitments={commitments} />
          ) : section === "graph" ? (
            <GraphInspectorRows graph={graph} />
          ) : (
            <BandInspectorRows band={section} />
          )}
        </div>
      </div>
    </div>
  );
}

function IdentityInspectorRows({
  identity,
  showToast,
  refetchMind,
}: {
  identity: IdentityResponse | undefined;
  showToast: (toast: Toast) => void;
  refetchMind: () => void;
}) {
  const rows = [
    ...(identity?.values ?? []).map((value) => ({
      chip: "VALUE",
      text: value.label,
      meta: `weight ${value.priority.toFixed(2)} · ${value.state}`,
      right: "",
    })),
    ...flattenGoals(identity?.goals ?? []).map((goal) => ({
      chip: "GOAL",
      text: goal.description,
      meta: `priority ${goal.priority.toFixed(2)} · ${goal.status}`,
      right: "BUMP / BLOCK",
      goal,
    })),
    ...(identity?.open_questions ?? []).map((question) => ({
      chip: "OPEN_Q",
      text: question.question,
      meta: `${question.source} · urgency ${question.urgency.toFixed(2)}`,
      right: question.status.toUpperCase(),
      question,
    })),
    ...(identity?.growth_markers ?? []).map((marker) => ({
      chip: "MARKER",
      text: marker.what_changed,
      meta: `${dateText(marker.ts)} · ${marker.category}`,
      right: "",
    })),
    ...(identity?.periods ?? []).map((period) => ({
      chip: "PERIOD",
      text: period.label,
      meta: rangeText(period),
      right: "",
    })),
  ];

  if (rows.length === 0) {
    return <div className="mind-empty">0 records</div>;
  }

  return (
    <>
      {rows.map((row, index) => {
        const goal = "goal" in row ? (row.goal as IdentityGoal | undefined) : undefined;
        return (
          <InspectorRow
            key={`${row.chip}:${index}`}
            chip={row.chip}
            text={row.text}
            meta={row.meta}
            right={row.right}
          >
            {goal === undefined ? null : (
              <GoalRow goal={goal} showToast={showToast} refetchMind={refetchMind} />
            )}
          </InspectorRow>
        );
      })}
    </>
  );
}

function DirectiveInspectorRows({
  directives,
  commitments,
}: {
  directives: CreatorDirective[];
  commitments: Commitment[];
}) {
  const rows = [
    ...directives.map((directive) => ({
      chip: `CD · ${directiveKindChip(directive.kind)}`,
      text: directive.text ?? directive.id,
      meta: `${directive.id} · scope ${directive.content_scope} · mention ${directive.mention_policy}`,
      right: directive.status.toUpperCase(),
    })),
    ...commitments.map((commitment) => ({
      chip: `CM · ${commitment.enforcement_class.toUpperCase()}`,
      text: commitment.text,
      meta: `${commitment.id} · ${commitment.kind} · audience ${commitment.audience ?? "—"}`,
      right: commitment.state.toUpperCase(),
    })),
  ];

  return rows.length === 0 ? (
    <div className="mind-empty">0 records</div>
  ) : (
    <>
      {rows.map((row) => (
        <InspectorRow key={`${row.chip}:${row.meta}`} {...row} />
      ))}
    </>
  );
}

function GraphInspectorRows({ graph }: { graph: SemanticGraphResponse | undefined }) {
  const [selected, setSelected] = useState<SemanticGraphNode | null>(null);
  const nodes = graph?.nodes ?? [];

  if (nodes.length === 0) {
    return <div className="mind-empty">0 records</div>;
  }

  return (
    <>
      {nodes.map((node) => (
        <InspectorRow
          key={node.id}
          chip={node.status.toUpperCase()}
          text={node.display_label ?? node.label}
          meta={`${node.kind} · ${node.edge_count} edges`}
          right="INSPECT"
        >
          <button className="row-mini-action" type="button" onClick={() => setSelected(node)}>
            INSPECT
          </button>
        </InspectorRow>
      ))}
      {selected === null ? null : (
        <div className="inline-graph-detail">
          {selected.status.toUpperCase()} · {selected.display_label ?? selected.label} · {selected.kind} ·{" "}
          {selected.edge_count} edges
        </div>
      )}
    </>
  );
}

function BandInspectorRows({ band }: { band: MemoryBandId }) {
  const initial = useQuery(`bands:${band}:detail`, () => fetchBandDetail({ band, limit: 50 }));
  const [detail, setDetail] = useState<BandDetailResponse | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);

  useEffect(() => {
    setDetail(null);
  }, [band]);

  useEffect(() => {
    if (initial.data !== undefined) {
      setDetail(initial.data);
    }
  }, [initial.data]);

  const loadMore = async () => {
    if (detail?.next_cursor === undefined || detail.next_cursor === null) {
      return;
    }

    setLoadingMore(true);
    try {
      const next = await fetchBandDetail({ band, cursor: detail.next_cursor, limit: 50 });
      setDetail({
        ...next,
        items: mergeByStableId(detail.items ?? [], next.items ?? []),
        nodes: mergeByStableId(detail.nodes ?? [], next.nodes ?? []),
        edges: mergeByStableId(detail.edges ?? [], next.edges ?? []),
      });
    } finally {
      setLoadingMore(false);
    }
  };

  if (detail === null) {
    return <div className="mind-empty">loading records…</div>;
  }

  if (band === "affective") {
    return <AffectiveRows detail={detail} />;
  }

  if (band === "self") {
    return <IdentityInspectorRows identity={detail as IdentityResponse} showToast={() => {}} refetchMind={() => {}} />;
  }

  const records = [
    ...(detail.items ?? []),
    ...(detail.nodes ?? []).map((node) => node as unknown as Record<string, unknown>),
    ...(detail.edges ?? []).map((edge) => edge as unknown as Record<string, unknown>),
  ];

  return (
    <>
      {records.length === 0 ? (
        <div className="mind-empty">0 records</div>
      ) : (
        records.map((record) => (
          <InspectorRow
            key={String(record.id ?? recordText(record))}
            chip={String(record.kind ?? record.status ?? record.state ?? band).toUpperCase().slice(0, 10)}
            text={recordText(record)}
            meta={recordMeta(record)}
            right={recordRight(record)}
          />
        ))
      )}
      {detail.next_cursor === null || detail.next_cursor === undefined ? null : (
        <button className="load-more" type="button" onClick={() => void loadMore()} disabled={loadingMore}>
          LOAD MORE
        </button>
      )}
    </>
  );
}

function AffectiveRows({ detail }: { detail: BandDetailResponse }) {
  const rows: Array<{ chip: string; text: string; meta: string; right: string }> = [];
  if (detail.current !== undefined && detail.current !== null) {
    rows.push({
      chip: "NOW",
      text: `valence ${detail.current.valence.toFixed(2)} · arousal ${detail.current.arousal.toFixed(2)}`,
      meta: `current mood · updated ${hm(new Date(detail.current.updated_at))}`,
      right: moodLabel(detail.current.valence, detail.current.arousal),
    });
  }
  for (const item of detail.history ?? []) {
    rows.push({
      chip: "HIST",
      text: `valence ${item.valence.toFixed(2)} · arousal ${item.arousal.toFixed(2)}`,
      meta: `updated ${dayLabel(new Date(item.updated_at))} ${hm(new Date(item.updated_at))}`,
      right: item.session_id,
    });
  }

  return rows.length === 0 ? (
    <div className="mind-empty">0 records</div>
  ) : (
    <>
      {rows.map((row, index) => (
        <InspectorRow key={`${row.chip}:${index}`} {...row} />
      ))}
    </>
  );
}

function FilterRow({ value, onChange }: { value: Filter; onChange: (value: Filter) => void }) {
  return (
    <div className="filter-row">
      <button className={value === "active" ? "filter-active" : ""} type="button" onClick={() => onChange("active")}>
        active
      </button>
      <button className={value === "all" ? "filter-active" : ""} type="button" onClick={() => onChange("all")}>
        all
      </button>
    </div>
  );
}

function IndexRow({
  code,
  name,
  count,
  desc,
  active,
  onClick,
}: {
  code: string;
  name: string;
  count: string;
  desc: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button className={active ? "mind-index-row mind-index-active" : "mind-index-row"} type="button" onClick={onClick}>
      <div>
        <span>{code}</span>
        <strong>{name}</strong>
        <b>{count}</b>
      </div>
      <p>{desc}</p>
    </button>
  );
}

function InspectorRow({
  chip,
  text,
  meta,
  right,
  children,
}: {
  chip: string;
  text: string;
  meta: string;
  right: string;
  children?: ReactNode;
}) {
  return (
    <div className="inspector-row">
      <span className="row-chip">{chip}</span>
      <div>
        <div>{text}</div>
        <p>{meta}</p>
        {children}
      </div>
      <span>{right}</span>
    </div>
  );
}

function PanelHead({
  title,
  subtitle,
  right = "",
  titleAccent = false,
}: {
  title: string;
  subtitle: string;
  right?: string;
  titleAccent?: boolean;
}) {
  return (
    <div className="mind-panel-head">
      <span className={titleAccent ? "panel-title-accent" : ""}>{title}</span>
      <span>{subtitle}</span>
      {right.length === 0 ? null : <span>{right}</span>}
    </div>
  );
}

function Subhead({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`mind-subhead ${className}`}>{children}</div>;
}

function ToastView({ toast }: { toast: Toast }) {
  return <div className={toast.tone === "error" ? "mind-toast mind-toast-error" : "mind-toast"}>{toast.text}</div>;
}
