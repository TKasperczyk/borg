import { type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  fetchCommitments,
  fetchCorrectionReviews,
  fetchCorrectionWhy,
  fetchCreatorDirectives,
  fetchEpisode,
  fetchIdentity,
  fetchReviews,
  fetchSemanticEdge,
  fetchSemanticNode,
  patchCorrectionReview,
  patchDreamReview,
  patchReview,
  postCreatorDirectiveReconciliation,
} from "../api/client";
import type {
  Commitment,
  CorrectionWhyResponse,
  CreatorDirective,
  EpisodeDetail,
  IdentityResponse,
  ReviewKind,
  ReviewResolution,
  ReviewRow,
  SemanticEdgeDetail,
  SemanticNodeDetail,
} from "../api/types";
import { useQuery } from "../api/useQuery";
import { ExpandableText } from "../components/ExpandableText";
import { dayLabel, relativeAge } from "../format/time";

type StatusFilter = "open" | "resolved";
type Toast = { text: string; tone: "ok" | "error" };
type RefCard = {
  id: string;
  label: string;
  sub: string;
  evidence: string;
  description?: string;
  audienceLabels?: string[];
  disclosureClass?: string;
  tag?: string;
};

type PairEvidenceResult = {
  key: string;
  episodes: EpisodeDetail[];
  failed: boolean;
};
type CurrentTargetState = {
  record: Record<string, unknown> | null;
  loading: boolean;
};
type ProposedInsight = {
  mode: string;
  nodeId: string | null;
  node: Record<string, unknown> | null;
  patch: Record<string, unknown> | null;
  evidenceEpisodeIds: string[];
};
type ProposedValueMode = "current" | "proposed" | "insert";

const MAX_PAIR_EVIDENCE_EPISODES = 6;
const LONG_TEXT_MIN_LENGTH = 120;
// Universal record metadata, not reviewable content.
const UNIVERSAL_RECORD_METADATA_FIELDS = new Set(["id", "created_at", "updated_at", "last_verified_at"]);
const IDENTITY_TARGET_COLLECTIONS: Record<string, string> = {
  autobiographical_period: "periods",
  goal: "goals",
  open_question: "open_questions",
  trait: "traits",
  value: "values",
};

const GENERIC_ACTIONS: Record<ReviewKind, ReviewResolution[]> = {
  contradiction: ["keep_both", "supersede", "invalidate", "dismiss"],
  duplicate: ["keep_both", "supersede", "invalidate", "dismiss"],
  new_insight: ["accept", "invalidate", "dismiss"],
  misattribution: ["accept", "reject", "dismiss"],
  temporal_drift: ["accept", "reject", "dismiss"],
  identity_inconsistency: ["accept", "reject", "dismiss"],
  correction: ["accept", "reject"],
  belief_revision: ["dismiss"],
  skill_split: ["accept", "reject"],
  creator_directive_reconciliation: ["supersede", "keep"],
  commitment_reconciliation: ["dismiss", "reject", "accept", "keep"],
  relationship_claim_ungrounded: [],
};

const ACTION_LABELS: Record<ReviewResolution, string> = {
  keep_both: "KEEP BOTH",
  supersede: "SUPERSEDE -> WINNER",
  invalidate: "INVALIDATE",
  dismiss: "DISMISS",
  accept: "ACCEPT",
  reject: "REJECT",
  keep: "KEEP",
  weaken: "WEAKEN",
  archive_node: "ARCHIVE NODE",
  invalidate_edge: "INVALIDATE EDGE",
};

export function reviewKindColor(kind: ReviewKind): string {
  const colors: Record<ReviewKind, string> = {
    contradiction: "var(--error-bright)",
    duplicate: "var(--gold)",
    creator_directive_reconciliation: "var(--ac)",
    commitment_reconciliation: "var(--ac)",
    belief_revision: "var(--purple)",
    correction: "var(--blue)",
    new_insight: "var(--text-dim)",
    misattribution: "var(--text-dim)",
    temporal_drift: "var(--text-dim)",
    identity_inconsistency: "var(--text-dim)",
    skill_split: "var(--text-dim)",
    relationship_claim_ungrounded: "var(--text-dim)",
  };
  return colors[kind];
}

export function actionsForReviewKind(kind: ReviewKind): ReviewResolution[] {
  return GENERIC_ACTIONS[kind];
}

export function mergeReviewRows(mainRows: readonly ReviewRow[], correctionRows: readonly ReviewRow[]): ReviewRow[] {
  const merged: ReviewRow[] = [];
  const seen = new Set<string>();

  for (const row of [...mainRows, ...correctionRows]) {
    const key = reviewKey(row);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(row);
  }

  return merged.sort((left, right) => right.created_at - left.created_at || left.id - right.id);
}

function reviewKey(row: Pick<ReviewRow, "kind" | "id">): string {
  return `${row.kind}:${row.id}`;
}

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }

  return error instanceof Error ? error.message : String(error);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function stringField(record: Record<string, unknown>, key: string): string | null {
  const value = record[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function numberField(record: Record<string, unknown>, key: string): number | null {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringArrayField(record: Record<string, unknown>, key: string): string[] {
  const value = record[key];
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

function stringArrayValue(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

function nonEmptyStringArrayValue(value: unknown): string[] | null {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }
  if (!value.every((entry) => typeof entry === "string" && entry.length > 0)) {
    return null;
  }
  return value as string[];
}

function hasOwn(record: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function upperLabel(value: string): string {
  return value.replace(/_/g, " ").toUpperCase();
}

function shortId(id: number): string {
  return String(id);
}

function resolutionText(row: ReviewRow): string {
  return row.resolution === null ? "unresolved" : upperLabel(row.resolution);
}

function sourceText(row: ReviewRow): string | null {
  return stringField(row.refs, "source_process") ?? stringField(row.refs, "source");
}

function actionTone(action: ReviewResolution): "primary" | "danger" | "plain" {
  if (action === "accept" || action === "supersede") {
    return "primary";
  }
  if (action === "reject" || action === "invalidate" || action === "archive_node" || action === "invalidate_edge") {
    return "danger";
  }
  return "plain";
}

function needsWinner(row: ReviewRow, action: ReviewResolution): boolean {
  if (row.kind !== "contradiction" && row.kind !== "duplicate") {
    return false;
  }
  if (action !== "supersede" && action !== "invalidate") {
    return false;
  }
  return stringArrayField(row.refs, "node_ids").length > 0;
}

function semanticNodeIds(row: ReviewRow | null): string[] {
  if (row === null) {
    return [];
  }
  return isNodePairReview(row) ? stringArrayField(row.refs, "node_ids") : [];
}

function isNodePairReview(row: ReviewRow): boolean {
  return row.kind === "contradiction" || row.kind === "duplicate";
}

function nodePairEdgeId(row: ReviewRow | null): string | null {
  if (row === null || !isNodePairReview(row)) {
    return null;
  }
  return stringField(row.refs, "edge_id");
}

function orderedUnique(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    if (seen.has(value)) {
      continue;
    }
    seen.add(value);
    result.push(value);
  }
  return result;
}

function pairEvidenceEpisodeIds(input: {
  row: ReviewRow | null;
  nodes: Record<string, SemanticNodeDetail | null>;
  edge: SemanticEdgeDetail | null;
}): string[] {
  if (input.row === null || !isNodePairReview(input.row)) {
    return [];
  }

  const nodeIds = stringArrayField(input.row.refs, "node_ids");
  return orderedUnique([
    ...nodeIds.flatMap((id) => input.nodes[id]?.source_episode_ids ?? []),
    ...(input.edge?.evidence_episode_ids ?? []),
  ]);
}

function episodeIdsFromRecord(record: Record<string, unknown>): string[] {
  const ids: string[] = [];
  for (const [key, value] of Object.entries(record)) {
    if (key === "episode_ids" || key.endsWith("_episode_ids")) {
      ids.push(...stringArrayValue(value));
    }
  }
  return ids;
}

function proposedInsight(row: ReviewRow | null): ProposedInsight | null {
  const pending = asRecord(row?.refs.reflector_pending_insight);
  if (pending === null) {
    return null;
  }

  const target = asRecord(pending.target);
  const topLevelNode = asRecord(pending.node);
  const targetNode = asRecord(target?.node);
  const node = targetNode ?? topLevelNode;
  const patch = asRecord(target?.patch) ?? asRecord(pending.patch);
  const targetMode = target === null ? null : stringField(target, "mode");
  const pendingMode = stringField(pending, "mode");
  const mode = targetMode ?? pendingMode ?? (patch !== null ? "update" : "insert");
  const nodeId = stringField(target ?? {}, "node_id") ?? stringField(node ?? {}, "id");
  const evidenceCluster = asRecord(pending.evidence_cluster);
  const evidenceEpisodeIds = orderedUnique([
    ...stringArrayField(row?.refs ?? {}, "episode_ids"),
    ...episodeIdsFromRecord(evidenceCluster ?? {}),
    ...episodeIdsFromRecord(node ?? {}),
    ...episodeIdsFromRecord(patch ?? {}),
  ]);

  if (node === null && patch === null && nodeId === null) {
    return null;
  }

  return { mode, nodeId, node, patch, evidenceEpisodeIds };
}

function pendingInsightUpdateNodeId(row: ReviewRow | null): string | null {
  const insight = proposedInsight(row);
  return insight?.mode === "update" ? insight.nodeId : null;
}

function reviewProposedEvidenceEpisodeIds(row: ReviewRow | null): string[] {
  if (row === null) {
    return [];
  }

  const patch = asRecord(row.refs.patch);
  const insight = proposedInsight(row);
  return orderedUnique([
    ...stringArrayField(row.refs, "evidence_episode_ids"),
    ...stringArrayField(row.refs, "episode_ids"),
    ...(patch === null ? [] : episodeIdsFromRecord(patch)),
    ...(insight?.evidenceEpisodeIds ?? []),
  ]);
}

function patchTargetNeedsIdentity(row: ReviewRow | null): boolean {
  if (row === null || asRecord(row.refs.patch) === null) {
    return false;
  }

  const targetType = stringField(row.refs, "target_type");
  return targetType !== null && hasOwn(IDENTITY_TARGET_COLLECTIONS, targetType);
}

function flattenRecordTree(values: readonly unknown[]): Record<string, unknown>[] {
  return values.flatMap((value) => {
    const record = asRecord(value);
    if (record === null) {
      return [];
    }

    const children = Array.isArray(record.children) ? flattenRecordTree(record.children) : [];
    return [record, ...children];
  });
}

function identityCurrentTarget(input: {
  row: ReviewRow | null;
  identity: IdentityResponse | null | undefined;
}): Record<string, unknown> | null {
  if (input.row === null || input.identity === null || input.identity === undefined) {
    return null;
  }

  const targetType = stringField(input.row.refs, "target_type");
  const targetId = stringField(input.row.refs, "target_id");
  if (targetType === null || targetId === null) {
    return null;
  }

  const collectionKey = IDENTITY_TARGET_COLLECTIONS[targetType];
  if (collectionKey === undefined) {
    return null;
  }

  const snapshot = input.identity as unknown as Record<string, unknown>;
  const collection = snapshot[collectionKey];
  if (!Array.isArray(collection)) {
    return null;
  }

  return flattenRecordTree(collection).find((record) => stringField(record, "id") === targetId) ?? null;
}

function publicDisclosure(value: string | undefined): boolean {
  return value === undefined || value === "public";
}

function labelRefLabels(refs: SemanticNodeDetail["origin_audience_refs"]): string[] {
  if (refs === undefined || refs.length === 0) {
    return [];
  }
  return refs.map((ref) => ref.label ?? ref.value);
}

function correctionTargetId(row: ReviewRow): string | null {
  return stringField(row.refs, "target_id");
}

function selectedStatusRows(rows: readonly ReviewRow[], status: StatusFilter): ReviewRow[] {
  return rows.filter((row) => (status === "open" ? row.resolved_at === null : row.resolved_at !== null));
}

function useToast(): [Toast | null, (toast: Toast) => void] {
  const [toast, setToast] = useState<Toast | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const showToast = (next: Toast) => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
    }
    setToast(next);
    timeoutRef.current = window.setTimeout(() => {
      setToast(null);
      timeoutRef.current = null;
    }, 2600);
  };

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

function useEvidenceEpisodes(scope: string, episodeIds: readonly string[]) {
  const cappedEpisodeIds = episodeIds.slice(0, MAX_PAIR_EVIDENCE_EPISODES);
  const queryKey =
    cappedEpisodeIds.length === 0 ? `${scope}:none` : `${scope}:${cappedEpisodeIds.join(",")}`;
  const query = useQuery<PairEvidenceResult>(queryKey, async () => {
    if (cappedEpisodeIds.length === 0) {
      return { key: queryKey, episodes: [], failed: false };
    }

    let failed = false;
    const episodes = await Promise.all(
      cappedEpisodeIds.map(async (id): Promise<EpisodeDetail | null> => {
        try {
          return (await fetchEpisode(id)).episode;
        } catch {
          failed = true;
          return null;
        }
      }),
    );

    return {
      key: queryKey,
      episodes: episodes.filter((episode): episode is EpisodeDetail => episode !== null),
      failed,
    };
  });
  const evidence =
    query.data?.key === queryKey ? query.data : { key: queryKey, episodes: [], failed: false };

  return {
    evidence,
    loading: query.loading && episodeIds.length > 0,
    omittedCount: Math.max(0, episodeIds.length - MAX_PAIR_EVIDENCE_EPISODES),
  };
}

export function ReviewsPage() {
  const [kindFilter, setKindFilter] = useState<ReviewKind | "all">("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("open");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [toast, showToast] = useToast();

  const reviews = useQuery("reviews:all", () => fetchReviews({ openOnly: false }));
  const correctionReviews = useQuery("reviews:correction", fetchCorrectionReviews);
  const commitments = useQuery("reviews:commitments:all", () => fetchCommitments("all"));
  const directives = useQuery("reviews:directives:all", () => fetchCreatorDirectives("all"));

  const allRows = useMemo(
    () => mergeReviewRows(reviews.data?.rows ?? [], correctionReviews.data?.rows ?? []),
    [correctionReviews.data?.rows, reviews.data?.rows],
  );
  const openCount = allRows.filter((row) => row.resolved_at === null).length;
  const statusRows = selectedStatusRows(allRows, statusFilter);
  const kindCounts = useMemo(() => {
    const counts = new Map<ReviewKind, number>();
    for (const row of statusRows) {
      counts.set(row.kind, (counts.get(row.kind) ?? 0) + 1);
    }
    return counts;
  }, [statusRows]);
  const visibleRows = statusRows.filter((row) => kindFilter === "all" || row.kind === kindFilter);
  const selected = visibleRows.find((row) => reviewKey(row) === selectedKey) ?? visibleRows[0] ?? null;

  useEffect(() => {
    if (visibleRows.length === 0) {
      setSelectedKey(null);
      return;
    }
    if (selected === null || selectedKey !== reviewKey(selected)) {
      setSelectedKey(reviewKey(visibleRows[0]!));
    }
  }, [selected, selectedKey, visibleRows]);

  const refetchReviews = () => {
    reviews.refetch();
    correctionReviews.refetch();
  };

  return (
    <main className="page reviews-page">
      <header className="page-header reviews-header">
        <span className="page-title">REVIEWS</span>
        <span className="page-subtitle">substrate issue resolution -- the entity flags, the operator decides</span>
        <span className="reviews-open-count">{openCount} OPEN</span>
      </header>

      <div className="reviews-filter-row">
        <button
          className={kindFilter === "all" ? "review-filter review-filter-active" : "review-filter"}
          type="button"
          onClick={() => setKindFilter("all")}
        >
          ALL {statusRows.length}
        </button>
        {[...kindCounts.entries()].map(([kind, count]) => (
          <button
            key={kind}
            className={kindFilter === kind ? "review-filter review-filter-active" : "review-filter"}
            type="button"
            onClick={() => setKindFilter(kind)}
          >
            {upperLabel(kind)} {count}
          </button>
        ))}
        <div className="review-status-filters">
          <button
            className={statusFilter === "open" ? "review-filter review-filter-active" : "review-filter"}
            type="button"
            onClick={() => setStatusFilter("open")}
          >
            OPEN {openCount}
          </button>
          <button
            className={statusFilter === "resolved" ? "review-filter review-filter-active" : "review-filter"}
            type="button"
            onClick={() => setStatusFilter("resolved")}
          >
            RESOLVED {allRows.length - openCount}
          </button>
        </div>
      </div>

      <div className="reviews-split">
        <ReviewList
          rows={visibleRows}
          selectedKey={selected === null ? null : reviewKey(selected)}
          onSelect={(row) => setSelectedKey(reviewKey(row))}
          loading={reviews.loading && reviews.data === undefined}
        />
        <ReviewDetail
          row={selected}
          semanticNodeIds={semanticNodeIds(selected)}
          commitments={commitments.data?.commitments ?? []}
          directives={directives.data?.directives ?? []}
          showToast={showToast}
          refetchReviews={refetchReviews}
        />
      </div>
      {toast === null ? null : <div className={`reviews-toast reviews-toast-${toast.tone}`}>{toast.text}</div>}
    </main>
  );
}

function ReviewList({
  rows,
  selectedKey,
  onSelect,
  loading,
}: {
  rows: ReviewRow[];
  selectedKey: string | null;
  onSelect: (row: ReviewRow) => void;
  loading: boolean;
}) {
  if (loading) {
    return <div className="reviews-list reviews-empty">loading reviews...</div>;
  }

  if (rows.length === 0) {
    return <div className="reviews-list reviews-empty">0 records</div>;
  }

  return (
    <div className="reviews-list">
      {rows.map((row) => {
        const selected = reviewKey(row) === selectedKey;
        const resolved = row.resolved_at !== null;
        return (
          <button
            key={reviewKey(row)}
            className={[
              "review-list-row",
              selected ? "review-list-row-selected" : "",
              resolved ? "review-list-row-resolved" : "",
            ].join(" ")}
            type="button"
            onClick={() => onSelect(row)}
          >
            <div className="review-row-top">
              <span className="review-short-id">#{shortId(row.id)}</span>
              <KindChip kind={row.kind} dim={resolved} />
              {resolved ? (
                <span className="review-check">✓</span>
              ) : (
                <span className="review-age">{relativeAge(new Date(row.created_at))}</span>
              )}
            </div>
            <div className="review-row-reason">{row.reason}</div>
          </button>
        );
      })}
    </div>
  );
}

function ReviewDetail({
  row,
  semanticNodeIds,
  commitments,
  directives,
  showToast,
  refetchReviews,
}: {
  row: ReviewRow | null;
  semanticNodeIds: string[];
  commitments: Commitment[];
  directives: CreatorDirective[];
  showToast: (toast: Toast) => void;
  refetchReviews: () => void;
}) {
  const [winnerByReview, setWinnerByReview] = useState<Record<string, string>>({});
  const [localResolvedByKey, setLocalResolvedByKey] = useState<Record<string, ReviewRow>>({});
  const [note, setNote] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [why, setWhy] = useState<CorrectionWhyResponse | null>(null);
  const [whyLoading, setWhyLoading] = useState(false);
  const nodeQueryKey = semanticNodeIds.length === 0 ? "reviews:nodes:none" : `reviews:nodes:${semanticNodeIds.join(",")}`;
  const edgeId = nodePairEdgeId(row);
  const edgeQueryKey = edgeId === null ? "reviews:edge:none" : `reviews:edge:${edgeId}`;
  const insightUpdateNodeId = pendingInsightUpdateNodeId(row);
  const insightUpdateNodeQueryKey =
    insightUpdateNodeId === null ? "reviews:insight-current-node:none" : `reviews:insight-current-node:${insightUpdateNodeId}`;
  const identityQueryKey = patchTargetNeedsIdentity(row) ? "reviews:identity:current-targets" : "reviews:identity:none";
  const nodeDetails = useQuery<Record<string, SemanticNodeDetail | null>>(nodeQueryKey, async () => {
    const entries = await Promise.all(
      semanticNodeIds.map(async (id): Promise<[string, SemanticNodeDetail | null]> => {
        try {
          const response = await fetchSemanticNode(id);
          return [id, response.node];
        } catch {
          return [id, null];
        }
      }),
    );
    return Object.fromEntries(entries);
  });
  const edgeDetail = useQuery<SemanticEdgeDetail | null>(edgeQueryKey, async () => {
    if (edgeId === null) {
      return null;
    }
    try {
      return (await fetchSemanticEdge(edgeId)).edge;
    } catch {
      return null;
    }
  });
  const insightCurrentNode = useQuery<SemanticNodeDetail | null>(insightUpdateNodeQueryKey, async () => {
    if (insightUpdateNodeId === null) {
      return null;
    }
    try {
      return (await fetchSemanticNode(insightUpdateNodeId)).node;
    } catch {
      return null;
    }
  });
  const identitySnapshot = useQuery<IdentityResponse | null>(identityQueryKey, async () => {
    if (!patchTargetNeedsIdentity(row)) {
      return null;
    }
    try {
      return await fetchIdentity();
    } catch {
      return null;
    }
  });
  const currentEdge = edgeDetail.data?.id === edgeId ? edgeDetail.data : null;
  const currentInsightNode =
    insightCurrentNode.data?.id === insightUpdateNodeId ? insightCurrentNode.data : null;
  const currentIdentityTarget = identityCurrentTarget({
    row,
    identity: identitySnapshot.data,
  });
  const currentPatchTarget: CurrentTargetState = {
    record: currentIdentityTarget,
    loading: identitySnapshot.loading && patchTargetNeedsIdentity(row),
  };
  const currentInsightTarget: CurrentTargetState = {
    record: currentInsightNode === null ? null : (currentInsightNode as unknown as Record<string, unknown>),
    loading: insightCurrentNode.loading && insightUpdateNodeId !== null,
  };
  const nodeDetailsReady =
    semanticNodeIds.length === 0 ||
    (nodeDetails.data !== undefined &&
      semanticNodeIds.every((id) => Object.prototype.hasOwnProperty.call(nodeDetails.data, id)));
  const edgeDetailsReady = edgeId === null || !edgeDetail.loading;
  const evidenceEpisodeIds = useMemo(
    () =>
      nodeDetailsReady && edgeDetailsReady
        ? pairEvidenceEpisodeIds({
            row,
            nodes: nodeDetails.data ?? {},
            edge: currentEdge,
          })
        : [],
    [currentEdge, edgeDetailsReady, nodeDetails.data, nodeDetailsReady, row],
  );
  const pairEvidence = useEvidenceEpisodes("reviews:pair-episodes", evidenceEpisodeIds);
  const proposedEvidenceEpisodeIds = useMemo(() => reviewProposedEvidenceEpisodeIds(row), [row]);
  const proposedEvidence = useEvidenceEpisodes("reviews:proposed-episodes", proposedEvidenceEpisodeIds);

  useEffect(() => {
    setNote("");
    setWhy(null);
  }, [row?.id, row?.kind]);

  if (row === null) {
    return <div className="reviews-detail reviews-empty">select a review</div>;
  }

  const key = reviewKey(row);
  const effectiveRow = localResolvedByKey[key] ?? row;
  const selectedWinner = winnerByReview[key];
  const resolved = effectiveRow.resolved_at !== null;
  const source = sourceText(effectiveRow);
  const refCards = reviewRefCards({
    row: effectiveRow,
    nodes: nodeDetails.data ?? {},
    commitments,
    directives,
  });

  const chooseWinner = (id: string) => {
    setWinnerByReview((current) => ({ ...current, [key]: id }));
  };

  const runAction = async (action: ReviewResolution) => {
    if (pending !== null || resolved) {
      return;
    }
    if (needsWinner(effectiveRow, action) && selectedWinner === undefined) {
      showToast({ text: "pick a winner first", tone: "error" });
      return;
    }

    setPending(action);
    try {
      const noteText = note.trim();
      let result: ReviewRow;
      if (effectiveRow.kind === "belief_revision") {
        result = await patchDreamReview(effectiveRow.id, noteText);
      } else if (effectiveRow.kind === "correction") {
        if (action !== "accept" && action !== "reject") {
          return;
        }
        result = await patchCorrectionReview(effectiveRow.id, {
          action,
          ...(noteText.length === 0 ? {} : { note: noteText }),
        });
      } else {
        result = await patchReview(effectiveRow.id, {
          action,
          ...(needsWinner(effectiveRow, action) && selectedWinner !== undefined
            ? { winner_node_id: selectedWinner }
            : {}),
          ...(noteText.length === 0 ? {} : { note: noteText }),
        });
      }
      setLocalResolvedByKey((current) => ({ ...current, [key]: result }));
      showToast({ text: `resolved -- ${resolutionText(result)}`, tone: "ok" });
      refetchReviews();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPending(null);
    }
  };

  const runCreatorDirectiveAction = async (action: "supersede" | "keep") => {
    if (pending !== null || resolved) {
      return;
    }
    if (action === "supersede" && selectedWinner === undefined) {
      showToast({ text: "pick a survivor first", tone: "error" });
      return;
    }

    setPending(action);
    try {
      const noteText = note.trim();
      const result = await postCreatorDirectiveReconciliation(
        effectiveRow.id,
        action === "supersede"
          ? {
              action,
              survivor_id: selectedWinner!,
              ...(noteText.length === 0 ? {} : { reason: noteText }),
            }
          : { action, ...(noteText.length === 0 ? {} : { reason: noteText }) },
      );
      setLocalResolvedByKey((current) => ({ ...current, [key]: result }));
      showToast({ text: `resolved -- ${resolutionText(result)}`, tone: "ok" });
      refetchReviews();
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setPending(null);
    }
  };

  const fetchWhy = async () => {
    if (pending !== null || row.kind !== "correction") {
      return;
    }
    const targetId = correctionTargetId(row);
    if (targetId === null) {
      showToast({ text: "correction review has no target_id", tone: "error" });
      return;
    }

    setWhyLoading(true);
    try {
      setWhy(await fetchCorrectionWhy(targetId));
    } catch (error) {
      showToast({ text: formatError(error), tone: "error" });
    } finally {
      setWhyLoading(false);
    }
  };

  return (
    <div className="reviews-detail">
      <div className="review-detail-head">
        <span className="review-detail-id">#{shortId(effectiveRow.id)}</span>
        <KindChip kind={effectiveRow.kind} dim={resolved} />
        <span className="review-detail-age">
          {relativeAge(new Date(effectiveRow.created_at))}
          {source === null ? "" : ` · queued by ${source}`}
        </span>
      </div>
      <div className="review-detail-body">
        <div className="review-full-reason">{effectiveRow.reason}</div>
        <KindBody
          row={effectiveRow}
          refCards={refCards}
          selectedId={selectedWinner}
          onPick={chooseWinner}
          why={why}
          whyLoading={whyLoading}
          evidence={pairEvidence.evidence}
          evidenceLoading={pairEvidence.loading}
          evidenceOmittedCount={pairEvidence.omittedCount}
          proposedEvidence={proposedEvidence.evidence}
          proposedEvidenceLoading={proposedEvidence.loading}
          proposedEvidenceOmittedCount={proposedEvidence.omittedCount}
          currentPatchTarget={currentPatchTarget}
          currentInsightTarget={currentInsightTarget}
        />
        {isNodePairReview(effectiveRow) && currentEdge !== null ? <EdgeLine edge={currentEdge} /> : null}
        {resolved ? (
          <div className="review-resolved-banner">
            ✓ resolved -- {resolutionText(effectiveRow)}
            {effectiveRow.resolved_at === null ? "" : ` · ${dayLabel(new Date(effectiveRow.resolved_at))}`}
          </div>
        ) : (
          <>
            <form className="review-note" onSubmit={(event) => event.preventDefault()}>
              <input
                value={note}
                onChange={(event) => setNote(event.target.value)}
                placeholder="optional note"
                disabled={pending !== null}
              />
            </form>
            <div className="review-actions">
              <ActionButtons
                row={effectiveRow}
                selectedWinner={selectedWinner}
                pending={pending}
                onGenericAction={(action) => void runAction(action)}
                onCreatorDirectiveAction={(action) => void runCreatorDirectiveAction(action)}
                onWhy={() => void fetchWhy()}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function ActionButtons({
  row,
  selectedWinner,
  pending,
  onGenericAction,
  onCreatorDirectiveAction,
  onWhy,
}: {
  row: ReviewRow;
  selectedWinner: string | undefined;
  pending: string | null;
  onGenericAction: (action: ReviewResolution) => void;
  onCreatorDirectiveAction: (action: "supersede" | "keep") => void;
  onWhy: () => void;
}) {
  if (row.kind === "creator_directive_reconciliation") {
    const disabled = selectedWinner === undefined || pending !== null;
    return (
      <>
        <button
          className="review-action review-action-primary"
          type="button"
          disabled={disabled}
          onClick={() => onCreatorDirectiveAction("supersede")}
        >
          SUPERSEDE FAMILY -&gt; SURVIVOR
        </button>
        {selectedWinner === undefined ? <span className="review-action-hint">pick a survivor first</span> : null}
        <button
          className="review-action review-action-plain"
          type="button"
          disabled={pending !== null}
          onClick={() => onCreatorDirectiveAction("keep")}
        >
          KEEP ALL
        </button>
      </>
    );
  }

  if (row.kind === "correction") {
    return (
      <>
        <button className="review-action review-action-plain" type="button" disabled={pending !== null} onClick={onWhy}>
          WHY?
        </button>
        {actionsForReviewKind(row.kind).map((action) => (
          <ActionButton
            key={action}
            action={action}
            disabled={pending !== null}
            onClick={() => onGenericAction(action)}
          />
        ))}
      </>
    );
  }

  const actions = actionsForReviewKind(row.kind);
  if (actions.length === 0) {
    return <span className="review-action-hint">no supported action for this legacy review kind</span>;
  }

  return (
    <>
      {actions.map((action) => (
        <ActionButton
          key={action}
          action={action}
          disabled={pending !== null || (needsWinner(row, action) && selectedWinner === undefined)}
          onClick={() => onGenericAction(action)}
        />
      ))}
      {actions.some((action) => needsWinner(row, action)) && selectedWinner === undefined ? (
        <span className="review-action-hint">pick a winner first</span>
      ) : null}
    </>
  );
}

function ActionButton({
  action,
  disabled,
  onClick,
}: {
  action: ReviewResolution;
  disabled: boolean;
  onClick: () => void;
}) {
  const tone = actionTone(action);
  return (
    <button
      className={`review-action review-action-${tone}`}
      type="button"
      disabled={disabled}
      onClick={onClick}
    >
      {ACTION_LABELS[action]}
    </button>
  );
}

function KindBody({
  row,
  refCards,
  selectedId,
  onPick,
  why,
  whyLoading,
  evidence,
  evidenceLoading,
  evidenceOmittedCount,
  proposedEvidence,
  proposedEvidenceLoading,
  proposedEvidenceOmittedCount,
  currentPatchTarget,
  currentInsightTarget,
}: {
  row: ReviewRow;
  refCards: RefCard[];
  selectedId: string | undefined;
  onPick: (id: string) => void;
  why: CorrectionWhyResponse | null;
  whyLoading: boolean;
  evidence: PairEvidenceResult;
  evidenceLoading: boolean;
  evidenceOmittedCount: number;
  proposedEvidence: PairEvidenceResult;
  proposedEvidenceLoading: boolean;
  proposedEvidenceOmittedCount: number;
  currentPatchTarget: CurrentTargetState;
  currentInsightTarget: CurrentTargetState;
}) {
  if (row.kind === "creator_directive_reconciliation") {
    return (
      <>
        <MemberCards cards={refCards} selectedId={selectedId} onPick={onPick} />
        <div className="review-note-line">
          This reconciliation uses a distinct atomic flow: pick one active survivor to supersede the
          rest, or keep every directive active.
        </div>
      </>
    );
  }

  if (refCards.length > 0) {
    return (
      <>
        <PairCards cards={refCards} selectedId={selectedId} onPick={onPick} />
        {isNodePairReview(row) ? (
          <PairEvidence evidence={evidence} loading={evidenceLoading} omittedCount={evidenceOmittedCount} />
        ) : null}
        <div className="review-note-line">{pairNote(row)}</div>
      </>
    );
  }

  return (
    <>
      <ReviewBodyBlock row={row} />
      <ProposedContentBlock
        row={row}
        evidence={proposedEvidence}
        evidenceLoading={proposedEvidenceLoading}
        evidenceOmittedCount={proposedEvidenceOmittedCount}
        currentPatchTarget={currentPatchTarget}
        currentInsightTarget={currentInsightTarget}
      />
      {row.kind === "belief_revision" ? (
        <div className="review-note-line">
          Applying a revision happens through the belief-reviser apply step; this queue exposes
          dismissal only.
        </div>
      ) : null}
      {row.kind === "correction" ? (
        <WhyBlock why={why} loading={whyLoading} />
      ) : null}
    </>
  );
}

function PairCards({
  cards,
  selectedId,
  onPick,
}: {
  cards: RefCard[];
  selectedId: string | undefined;
  onPick: (id: string) => void;
}) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set());

  const toggle = (id: string) => {
    onPick(id);
    setExpandedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <div className="review-pair-grid">
      {cards.map((card) => {
        const expanded = expandedIds.has(card.id);
        return (
          <button
            key={card.id}
            className={selectedId === card.id ? "review-ref-card review-ref-card-selected" : "review-ref-card"}
            type="button"
            onClick={() => toggle(card.id)}
          >
            <span className="review-radio" />
            <div className="review-ref-card-content">
              <span className="review-ref-title-line">
                <strong>{card.label}</strong>
                {card.disclosureClass === undefined ? null : (
                  <span className="review-disclosure-chip">{card.disclosureClass}</span>
                )}
              </span>
              {card.description === undefined ? null : (
                <ExpandableText
                  text={card.description}
                  expanded={expanded}
                  className={
                    "review-ref-description"
                  }
                  expandedClassName="review-ref-description-expanded"
                />
              )}
              <p>{card.sub}</p>
              <small>{card.evidence}</small>
              {card.audienceLabels === undefined || card.audienceLabels.length === 0 ? null : (
                <small>origin {card.audienceLabels.join(" · ")}</small>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function EdgeLine({ edge }: { edge: SemanticEdgeDetail }) {
  return (
    <div className="review-edge-line">
      {edge.relation} edge · confidence {edge.confidence.toFixed(2)} · recorded{" "}
      {dayLabel(new Date(edge.valid_from))}
    </div>
  );
}

function PairEvidence({
  evidence,
  loading,
  omittedCount,
}: {
  evidence: PairEvidenceResult;
  loading: boolean;
  omittedCount: number;
}) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set());

  if (loading) {
    return <div className="review-evidence review-evidence-muted">loading evidence...</div>;
  }

  if (evidence.failed && evidence.episodes.length === 0) {
    return <div className="review-evidence review-evidence-muted">evidence unavailable</div>;
  }

  if (evidence.episodes.length === 0 && !evidence.failed) {
    return null;
  }

  const toggle = (id: string) => {
    setExpandedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <section className="review-evidence">
      <div className="review-evidence-head">EVIDENCE</div>
      {evidence.episodes.map((episode) => {
        const expanded = expandedIds.has(episode.id);
        return (
          <button className="review-evidence-row" key={episode.id} type="button" onClick={() => toggle(episode.id)}>
            <strong>{episode.title}</strong>
            <span>{dayLabel(new Date(episode.start_time))}</span>
            <ExpandableText
              text={episode.narrative}
              expanded={expanded}
              className="review-evidence-narrative"
              expandedClassName="review-evidence-narrative-expanded"
            />
          </button>
        );
      })}
      {omittedCount > 0 ? <div className="review-evidence-muted">{omittedCount} more not shown</div> : null}
      {evidence.failed ? <div className="review-evidence-muted">evidence unavailable</div> : null}
    </section>
  );
}

function MemberCards(props: {
  cards: RefCard[];
  selectedId: string | undefined;
  onPick: (id: string) => void;
}) {
  return (
    <div className="review-member-list">
      {props.cards.map((card) => (
        <button
          key={card.id}
          className={props.selectedId === card.id ? "review-member-card review-ref-card-selected" : "review-member-card"}
          type="button"
          onClick={() => props.onPick(card.id)}
        >
          <span className="review-radio" />
          <div>
            <strong>{card.label}</strong>
            <p>{card.sub}</p>
          </div>
          {card.tag === undefined ? null : <span className="review-member-tag">{card.tag}</span>}
        </button>
      ))}
    </div>
  );
}

function ProposedContentBlock({
  row,
  evidence,
  evidenceLoading,
  evidenceOmittedCount,
  currentPatchTarget,
  currentInsightTarget,
}: {
  row: ReviewRow;
  evidence: PairEvidenceResult;
  evidenceLoading: boolean;
  evidenceOmittedCount: number;
  currentPatchTarget: CurrentTargetState;
  currentInsightTarget: CurrentTargetState;
}) {
  const patch = asRecord(row.refs.patch);
  const insight = proposedInsight(row);
  const hasPatchContent = patch !== null && renderableEntries(patch).length > 0;
  const hasInsightContent = insight !== null;
  const hasEvidence =
    evidenceLoading || evidence.failed || evidence.episodes.length > 0 || evidenceOmittedCount > 0;

  if (!hasPatchContent && !hasInsightContent && !hasEvidence) {
    return null;
  }

  return (
    <section className="review-proposed-block">
      <div className="review-proposed-head">PROPOSED CONTENT</div>
      {patch === null ? null : (
        <ProposedPatch patch={patch} currentTarget={currentPatchTarget} />
      )}
      {insight === null ? null : (
        <ProposedInsightBlock insight={insight} currentTarget={currentInsightTarget} />
      )}
      <PairEvidence
        evidence={evidence}
        loading={evidenceLoading}
        omittedCount={evidenceOmittedCount}
      />
    </section>
  );
}

function ProposedPatch({
  patch,
  currentTarget,
}: {
  patch: Record<string, unknown>;
  currentTarget: CurrentTargetState;
}) {
  const entries = renderableEntries(patch);
  if (entries.length === 0) {
    return null;
  }

  return (
    <div className="review-proposed-section">
      <div className="review-proposed-section-title">PATCH</div>
      {entries.map(([field, value]) => (
        <ProposedFieldRow
          key={field}
          field={field}
          proposedValue={value}
          currentTarget={currentTarget}
          proposedMode="proposed"
        />
      ))}
    </div>
  );
}

function ProposedInsightBlock({
  insight,
  currentTarget,
}: {
  insight: ProposedInsight;
  currentTarget: CurrentTargetState;
}) {
  const proposed = proposedInsightFields(insight);
  if (proposed.length === 0) {
    return null;
  }

  const label =
    valueAsString(proposed.find(([field]) => field === "label")?.[1]) ??
    valueAsString(currentTarget.record?.display_label) ??
    valueAsString(currentTarget.record?.label) ??
    insight.nodeId ??
    "pending insight";
  const contextKind =
    valueAsString(proposed.find(([field]) => field === "kind")?.[1]) ??
    valueAsString(currentTarget.record?.kind);
  const title = ["INSIGHT", insight.mode.toUpperCase(), contextKind, label].filter(
    (part): part is string => part !== null,
  );

  return (
    <div className="review-proposed-section">
      <div className="review-proposed-section-title">{title.join(" · ")}</div>
      {proposed.map(([field, value]) => (
        <ProposedFieldRow
          key={field}
          field={field}
          proposedValue={value}
          currentTarget={currentTarget}
          proposedMode={insight.mode === "insert" ? "insert" : "proposed"}
        />
      ))}
    </div>
  );
}

function proposedInsightFields(insight: ProposedInsight): Array<[string, unknown]> {
  const source = insight.mode === "update" ? insight.patch ?? insight.node ?? {} : insight.node ?? insight.patch ?? {};
  const entries = renderableEntries(source);
  const fields: Array<[string, unknown]> = [];
  const headlineValues: Array<[string, unknown]> = [
    ["label", source.display_label ?? source.label],
    ["description", source.description],
    ["kind", source.kind],
    ["confidence", source.confidence],
  ];

  for (const [field, value] of headlineValues) {
    if (value !== undefined && isRenderableProposedValue(value)) {
      fields.push([field, value]);
    }
  }

  const headlineFields = new Set(["display_label", "label", "description", "kind", "confidence"]);
  for (const [field, value] of entries) {
    if (!headlineFields.has(field)) {
      fields.push([field, value]);
    }
  }

  return fields;
}

function ProposedFieldRow({
  field,
  proposedValue,
  currentTarget,
  proposedMode,
}: {
  field: string;
  proposedValue: unknown;
  currentTarget: CurrentTargetState;
  proposedMode: ProposedValueMode;
}) {
  if (!isRenderableProposedValue(proposedValue)) {
    return null;
  }

  const hasCurrent = currentTarget.record !== null && hasOwn(currentTarget.record, field);
  const currentValue = hasCurrent ? currentTarget.record![field] : undefined;
  const proposedLabel =
    proposedMode === "insert" ? "INSERT" : hasCurrent ? "PROPOSED" : "PROPOSED ONLY";

  return (
    <div className="review-proposed-row">
      <strong>{upperLabel(field)}</strong>
      <div className="review-proposed-values">
        {currentTarget.loading ? (
          <div className="review-proposed-value review-proposed-current">
            <span className="review-proposed-value-label">CURRENT</span>
            <span className="review-proposed-muted">loading...</span>
          </div>
        ) : hasCurrent ? (
          <div className="review-proposed-value review-proposed-current">
            <span className="review-proposed-value-label">CURRENT</span>
            <ProposedValue field={field} value={currentValue} />
          </div>
        ) : null}
        <div className="review-proposed-value">
          <span className="review-proposed-value-label">{proposedLabel}</span>
          <ProposedValue field={field} value={proposedValue} />
        </div>
      </div>
    </div>
  );
}

function ProposedValue({ field, value }: { field: string; value: unknown }) {
  const [expanded, setExpanded] = useState(false);

  if (!isRenderableProposedValue(value)) {
    return <span className="review-proposed-muted">not shown</span>;
  }

  if (typeof value === "string") {
    if (value.length >= LONG_TEXT_MIN_LENGTH) {
      return (
        <button
          type="button"
          className="review-proposed-text-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((current) => !current)}
        >
          <ExpandableText
            text={value}
            expanded={expanded}
            className="review-proposed-long-text"
            expandedClassName="review-proposed-long-text-expanded"
          />
        </button>
      );
    }

    return <span>{value}</span>;
  }

  if (typeof value === "number") {
    return <span>{field === "confidence" ? value.toFixed(2) : String(value)}</span>;
  }

  if (typeof value === "boolean") {
    return <span>{String(value)}</span>;
  }

  const strings = nonEmptyStringArrayValue(value);
  if (strings !== null) {
    return <ChipList values={strings} />;
  }

  const record = asRecord(value);
  if (record !== null && isDisclosureLabelRecord(record)) {
    return <DisclosureLabelSummary record={record} />;
  }

  return <span className="review-proposed-muted">not shown</span>;
}

function ChipList({ values }: { values: string[] }) {
  return (
    <span className="review-proposed-chip-list">
      {values.map((value) => (
        <span className="review-proposed-chip" key={value}>
          {value}
        </span>
      ))}
    </span>
  );
}

function DisclosureLabelSummary({ record }: { record: Record<string, unknown> }) {
  const disclosureClass = stringField(record, "disclosure_class");
  const privateTo = stringArrayField(record, "private_to_entity_ids");
  const publicTo = stringArrayField(record, "public_to_entity_ids");
  const origin = stringArrayField(record, "origin_audience_entity_ids");
  const summary = [
    privateTo.length > 0 ? `private to ${privateTo.length}` : null,
    publicTo.length > 0 ? `public to ${publicTo.length}` : null,
    origin.length > 0 ? `origin ${origin.length}` : null,
  ].filter((entry): entry is string => entry !== null);

  return (
    <span className="review-proposed-disclosure">
      {disclosureClass === null ? null : (
        <span className="review-disclosure-chip">{disclosureClass}</span>
      )}
      {summary.length === 0 ? null : <span>{summary.join(" · ")}</span>}
    </span>
  );
}

function renderableEntries(record: Record<string, unknown>): Array<[string, unknown]> {
  return Object.entries(record).filter(
    ([field, value]) => !UNIVERSAL_RECORD_METADATA_FIELDS.has(field) && isRenderableProposedValue(value),
  );
}

function isRenderableProposedValue(value: unknown): boolean {
  if (typeof value === "string") {
    return value.length > 0;
  }
  if (typeof value === "number") {
    return Number.isFinite(value);
  }
  if (typeof value === "boolean") {
    return true;
  }
  if (nonEmptyStringArrayValue(value) !== null) {
    return true;
  }

  const record = asRecord(value);
  return record !== null && isDisclosureLabelRecord(record);
}

function isDisclosureLabelRecord(record: Record<string, unknown>): boolean {
  return (
    hasOwn(record, "disclosure_class") ||
    hasOwn(record, "origin_audience_entity_ids") ||
    hasOwn(record, "private_to_entity_ids") ||
    hasOwn(record, "public_to_entity_ids")
  );
}

function valueAsString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function ReviewBodyBlock({ row }: { row: ReviewRow }) {
  const lines = bodyLines(row);
  if (lines.length === 0) {
    return null;
  }

  return (
    <div className="review-body-block">
      {lines.map((line) => (
        <div key={line.label}>
          <strong>{line.label}</strong>
          <span>{line.value}</span>
        </div>
      ))}
    </div>
  );
}

function WhyBlock({ why, loading }: { why: CorrectionWhyResponse | null; loading: boolean }) {
  if (loading) {
    return <div className="review-why">loading provenance...</div>;
  }
  if (why === null) {
    return null;
  }

  return (
    <div className="review-why">
      <strong>WHY</strong>
      {Object.entries(why).map(([key, value]) => (
        <div key={key}>
          <span>{key}</span>
          <code>{summarizeUnknown(value)}</code>
        </div>
      ))}
    </div>
  );
}

function KindChip({ kind, dim = false }: { kind: ReviewKind; dim?: boolean }) {
  return (
    <span className="review-kind-chip" style={{ color: dim ? "var(--text-ghost)" : reviewKindColor(kind) }}>
      {upperLabel(kind)}
    </span>
  );
}

function reviewRefCards(input: {
  row: ReviewRow;
  nodes: Record<string, SemanticNodeDetail | null>;
  commitments: Commitment[];
  directives: CreatorDirective[];
}): RefCard[] {
  if (input.row.kind === "contradiction" || input.row.kind === "duplicate") {
    const nodeIds = stringArrayField(input.row.refs, "node_ids");
    const labels = stringArrayField(input.row.refs, "node_labels");
    return nodeIds.map((id, index) => {
      const node = input.nodes[id] ?? null;
      const description = node?.description.trim();
      const domain = node?.domain === null || node?.domain === undefined ? [] : [node.domain];
      const audiences = labelRefLabels(node?.origin_audience_refs);
      return {
        id,
        label: node?.display_label ?? node?.label ?? labels[index] ?? id,
        sub:
          node === null
            ? "semantic node"
            : [node.kind, ...domain, `confidence ${node.confidence.toFixed(2)}`, node.status].join(" · "),
        evidence:
          node === null
            ? id
            : `recorded ${dayLabel(new Date(node.created_at))} · updated ${dayLabel(
                new Date(node.updated_at),
              )} · ${id}`,
        description: description === undefined || description.length === 0 ? undefined : description,
        audienceLabels: audiences.length === 0 ? undefined : audiences,
        disclosureClass: publicDisclosure(node?.disclosure_class) ? undefined : node?.disclosure_class,
      };
    });
  }

  if (input.row.kind === "creator_directive_reconciliation") {
    const ids = stringArrayField(input.row.refs, "directive_ids");
    const createdAts = ids
      .map((id) => input.directives.find((directive) => directive.id === id)?.created_at)
      .filter((value): value is number => typeof value === "number");
    const newest = createdAts.length > 1 ? Math.max(...createdAts) : null;
    const oldest = createdAts.length > 1 ? Math.min(...createdAts) : null;
    return ids.map((id) => {
      const directive = input.directives.find((entry) => entry.id === id);
      const tag =
        directive?.created_at === newest && newest !== oldest
          ? "NEWER"
          : directive?.created_at === oldest && newest !== oldest
            ? "OLDER"
            : undefined;
      return {
        id,
        label: directive?.text ?? id,
        sub:
          directive === undefined
            ? id
            : `${directive.kind} · ${directive.status} · scope ${directive.content_scope} · mention ${directive.mention_policy}`,
        evidence:
          directive === undefined
            ? ""
            : `priority ${directive.priority.toFixed(2)} · created ${dayLabel(new Date(directive.created_at))}`,
        tag,
      };
    });
  }

  if (input.row.kind === "commitment_reconciliation") {
    const ids = stringArrayField(input.row.refs, "commitment_ids");
    const members = Array.isArray(input.row.refs.members)
      ? input.row.refs.members.flatMap((member) => {
          const record = asRecord(member);
          return record === null ? [] : [record];
        })
      : [];
    return ids.map((id) => {
      const commitment = input.commitments.find((entry) => entry.id === id);
      const member = members.find((entry) => stringField(entry, "id") === id);
      return {
        id,
        label: commitment?.text ?? stringField(member ?? {}, "directive") ?? id,
        sub:
          commitment === undefined
            ? `${stringField(member ?? {}, "kind") ?? "commitment"} · ${stringField(member ?? {}, "type") ?? ""}`
            : `${commitment.kind} · ${commitment.enforcement_class} · ${commitment.state}`,
        evidence:
          commitment === undefined
            ? id
            : `${commitment.audience ?? "no audience"} · source ${commitment.source} · priority ${commitment.priority}`,
      };
    });
  }

  return [];
}

function pairNote(row: ReviewRow): string {
  if (row.kind === "commitment_reconciliation") {
    const judgment = asRecord(row.refs.judgment);
    return stringField(judgment ?? {}, "reason") ?? "resolution records the commitment reconciliation judgment";
  }
  if (row.kind === "duplicate") {
    return "supersede keeps the selected semantic node active and marks the other as superseded; dismiss leaves the queue only.";
  }
  if (row.kind === "contradiction") {
    return "keep both leaves the tension visible; supersede or invalidate requires selecting the winning node.";
  }
  return "";
}

function bodyLines(row: ReviewRow): Array<{ label: string; value: string }> {
  const refs = row.refs;
  const lines: Array<{ label: string; value: string }> = [];
  const targetType = stringField(refs, "target_type");
  const targetId = stringField(refs, "target_id");
  if (targetType !== null) {
    lines.push({ label: "target", value: targetId === null ? targetType : `${targetType} · ${targetId}` });
  }

  const invalidatedEdgeId = stringField(refs, "invalidated_edge_id");
  if (invalidatedEdgeId !== null) {
    lines.push({ label: "invalidated edge", value: invalidatedEdgeId });
  }

  const evidenceIds = stringArrayField(refs, "evidence_episode_ids");
  if (evidenceIds.length > 0) {
    lines.push({ label: "evidence", value: `${evidenceIds.length} episodes` });
  }

  const revision = asRecord(refs.belief_revision_llm);
  if (revision !== null) {
    const verdict = stringField(revision, "verdict");
    const rationale = stringField(revision, "rationale");
    if (verdict !== null) {
      lines.push({ label: "proposed", value: rationale === null ? verdict : `${verdict} · ${rationale}` });
    }
  }

  const rationale = stringField(refs, "rationale");
  if (rationale !== null) {
    lines.push({ label: "rationale", value: rationale });
  }

  const reason = stringField(refs, "reason");
  if (reason !== null) {
    lines.push({ label: "reason", value: reason });
  }

  const operatorReason = stringField(refs, "operator_reason");
  if (operatorReason !== null) {
    lines.push({ label: "operator reason", value: operatorReason });
  }

  const promptSummary = stringField(refs, "prompt_summary");
  if (promptSummary !== null) {
    lines.push({ label: "summary", value: promptSummary });
  }

  const similarity = numberField(refs, "vector_similarity");
  if (similarity !== null) {
    lines.push({ label: "vector similarity", value: similarity.toFixed(3) });
  }

  return lines;
}

function summarizeUnknown(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return `${value.length} items`;
  }
  const record = asRecord(value);
  if (record !== null) {
    const label =
      stringField(record, "label") ??
      stringField(record, "display_label") ??
      stringField(record, "id") ??
      stringField(record, "target_type");
    const keys = Object.keys(record);
    return label === null ? keys.join(", ") : `${label} · ${keys.length} fields`;
  }
  return String(value);
}
