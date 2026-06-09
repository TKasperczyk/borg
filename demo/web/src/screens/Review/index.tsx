import { useEffect, useMemo, useRef, useState } from "react";

import {
  getCreatorDirectives,
  getSemanticEdge,
  getSemanticNode,
  getReviews,
  patchReview,
  resolveCreatorDirectiveReconciliation,
} from "../../api/client";
import type {
  CreatorDirectiveItem,
  ReviewKind,
  ReviewResolution,
  ReviewRow,
  SemanticMemoryEdge,
  SemanticMemoryNode,
} from "../../api/types";
import { SemanticEdgeDetail } from "../../components/SemanticEdgeDetail";
import { SemanticNodeDetail } from "../../components/SemanticNodeDetail";
import { Tag } from "../../components/Tag";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { formatTime } from "../../lib/stream-utils";
import { displayValue, fieldLabel, isRecord, shortId } from "../screen-utils";

const REVIEW_KIND_ORDER: ReviewKind[] = [
  "creator_directive_reconciliation",
  "commitment_reconciliation",
  "correction",
  "belief_revision",
  "contradiction",
  "duplicate",
  "new_insight",
  "misattribution",
  "temporal_drift",
  "identity_inconsistency",
  "skill_split",
];

const GENERIC_REVIEW_ACTIONS: Record<ReviewKind, ReviewResolution[]> = {
  contradiction: ["keep_both", "supersede", "invalidate", "dismiss"],
  duplicate: ["keep_both", "supersede", "invalidate", "dismiss"],
  new_insight: ["accept", "invalidate", "dismiss"],
  misattribution: ["accept", "reject", "dismiss"],
  temporal_drift: ["accept", "reject", "dismiss"],
  identity_inconsistency: ["accept", "reject", "dismiss"],
  correction: ["accept", "reject"],
  belief_revision: ["dismiss"],
  skill_split: ["accept", "reject"],
  creator_directive_reconciliation: [],
  commitment_reconciliation: ["accept", "reject", "dismiss", "keep"],
};

const REVIEW_RESOLVER_REF_PREFIX = "__borg_review_resolver_";

type ReviewData = {
  rows: ReviewRow[];
  directives: CreatorDirectiveItem[];
};

type BusyState = {
  id: number;
  label: string;
} | null;

type ScopeField =
  | "content_scope"
  | "mention_policy"
  | "disclosure_allowed"
  | "disclosure_excluded"
  | "activation_scope"
  | "activation_allowed"
  | "activation_excluded";

function recordValue(record: Record<string, unknown>, key: string): unknown {
  return Object.hasOwn(record, key) ? record[key] : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

type DetailField = [string, unknown];

function addField(fields: DetailField[], label: string, value: unknown): void {
  if (value === undefined || value === null) {
    return;
  }

  if (typeof value === "string" && value.length === 0) {
    return;
  }

  fields.push([label, value]);
}

function addArrayCount(fields: DetailField[], label: string, value: unknown): void {
  if (Array.isArray(value)) {
    fields.push([label, value.length]);
  }
}

function firstString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function compactPatchSummary(value: unknown): string | null {
  if (!isRecord(value)) {
    return firstString(value);
  }

  const entries = Object.entries(value).filter(([, item]) => item !== null && item !== undefined);
  if (entries.length === 0) {
    return null;
  }

  const visible = entries.slice(0, 6).map(([key, item]) => {
    if (Array.isArray(item)) {
      return `${fieldLabel(key)}: ${item.length}`;
    }
    return `${fieldLabel(key)}: ${displayValue(item)}`;
  });
  if (entries.length > visible.length) {
    visible.push(`+${entries.length - visible.length} fields`);
  }
  return visible.join("; ");
}

function addTopLevelArrayCounts(fields: DetailField[], refs: Record<string, unknown>): void {
  const seen = new Set(fields.map(([label]) => label));

  for (const [key, value] of Object.entries(refs)) {
    if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
      continue;
    }

    const label = `${fieldLabel(key)} count`;
    if (!seen.has(label)) {
      fields.push([label, value.length]);
      seen.add(label);
    }
  }
}

function sourceOverlapSummary(value: unknown): string | null {
  if (!isRecord(value)) {
    return firstString(value);
  }

  const parts: string[] = [];
  const overlapCount = recordValue(value, "overlap_count");
  if (typeof overlapCount === "number") {
    parts.push(`${overlapCount} overlapping`);
  }

  const candidateSourceEpisodes = recordValue(value, "candidate_source_episode_ids");
  if (Array.isArray(candidateSourceEpisodes)) {
    parts.push(`${candidateSourceEpisodes.length} candidate`);
  }

  const matchedSourceEpisodes = recordValue(value, "matched_source_episode_ids");
  if (Array.isArray(matchedSourceEpisodes)) {
    parts.push(`${matchedSourceEpisodes.length} matched`);
  }

  return parts.length === 0 ? null : parts.join(", ");
}

function reviewDiagnosticReason(refs: Record<string, unknown>): string | null {
  const diagnostic = recordValue(refs, `${REVIEW_RESOLVER_REF_PREFIX}diagnostic`);
  if (!isRecord(diagnostic)) {
    return null;
  }

  return firstString(diagnostic.reason);
}

function pendingInsightTargetNode(refs: Record<string, unknown>): Record<string, unknown> {
  const pendingInsight = recordValue(refs, "reflector_pending_insight");
  if (!isRecord(pendingInsight)) {
    return {};
  }

  const target = recordValue(pendingInsight, "target");
  if (!isRecord(target)) {
    return {};
  }

  const node = recordValue(target, "node");
  if (isRecord(node)) {
    return node;
  }

  const patch = recordValue(target, "patch");
  return isRecord(patch) ? patch : {};
}

function nodeIds(row: ReviewRow): string[] {
  return stringArray(recordValue(row.refs, "node_ids"));
}

function nodeLabels(row: ReviewRow): string[] {
  return stringArray(recordValue(row.refs, "node_labels"));
}

function nodeOptionLabel(row: ReviewRow, id: string, index: number): string {
  const label = nodeLabels(row)[index];
  return label === undefined ? shortId(id) : `${label} [${shortId(id)}]`;
}

function reviewEdgeId(row: ReviewRow): string | null {
  return firstString(recordValue(row.refs, "edge_id"));
}

function isPairReview(row: ReviewRow): boolean {
  return row.kind === "contradiction" || row.kind === "duplicate";
}

function directiveIds(row: ReviewRow): string[] {
  return stringArray(recordValue(row.refs, "directive_ids"));
}

function actionLabel(action: ReviewResolution): string {
  if (action === "keep_both") {
    return "keep both";
  }
  if (action === "archive_node") {
    return "archive node";
  }
  if (action === "invalidate_edge") {
    return "invalidate edge";
  }
  return action;
}

function kindLabel(kind: ReviewKind): string {
  return kind.replaceAll("_", " ");
}

function diagnosticEntries(row: ReviewRow): Array<[string, unknown]> {
  const refDiagnostics = Object.entries(row.refs).filter(([key]) =>
    key.startsWith(REVIEW_RESOLVER_REF_PREFIX),
  );

  if (row.resolver_diagnostic === undefined) {
    return refDiagnostics;
  }

  return [["resolver_diagnostic", row.resolver_diagnostic], ...refDiagnostics];
}

function newInsightDetailFields(refs: Record<string, unknown>): DetailField[] {
  const fields: DetailField[] = [];
  const node = pendingInsightTargetNode(refs);

  addField(fields, "node label", recordValue(node, "label"));
  addField(fields, "node description", recordValue(node, "description"));
  addField(fields, "evidence cluster size", recordValue(refs, "evidence_cluster_size"));
  addArrayCount(fields, "episode count", recordValue(refs, "episode_ids"));
  addField(fields, "diagnostic reason", reviewDiagnosticReason(refs));

  return fields;
}

function pairDetailFields(refs: Record<string, unknown>): DetailField[] {
  const fields: DetailField[] = [];

  addField(fields, "node labels", recordValue(refs, "node_labels"));
  addField(fields, "vector similarity", recordValue(refs, "vector_similarity"));
  addField(fields, "source overlap", sourceOverlapSummary(recordValue(refs, "source_overlap")));
  addField(fields, "duplicate subtype", recordValue(refs, "duplicate_subtype"));
  addField(fields, "suggested valid to", recordValue(refs, "suggested_valid_to"));
  addField(fields, "repair text", recordValue(refs, "reason"));

  return fields;
}

type EpisodeRefGroup = {
  label: string;
  ids: string[];
};

function pairEpisodeRefGroups(refs: Record<string, unknown>): EpisodeRefGroup[] {
  const groups: EpisodeRefGroup[] = [];
  const reviewEpisodeIds = stringArray(recordValue(refs, "episode_ids"));
  if (reviewEpisodeIds.length > 0) {
    groups.push({ label: "review episodes", ids: reviewEpisodeIds });
  }

  const sourceOverlap = recordValue(refs, "source_overlap");
  if (!isRecord(sourceOverlap)) {
    return groups;
  }

  groups.push(
    {
      label: "candidate episodes",
      ids: stringArray(recordValue(sourceOverlap, "candidate_source_episode_ids")),
    },
    {
      label: "matched episodes",
      ids: stringArray(recordValue(sourceOverlap, "matched_source_episode_ids")),
    },
    {
      label: "overlap episodes",
      ids: stringArray(recordValue(sourceOverlap, "overlapping_source_episode_ids")),
    },
  );

  return groups.filter((group) => group.ids.length > 0);
}

function repairDetailFields(refs: Record<string, unknown>): DetailField[] {
  const fields: DetailField[] = [];

  addField(fields, "target type", recordValue(refs, "target_type"));
  addField(fields, "repair op", recordValue(refs, "repair_op"));
  addField(fields, "patch description", recordValue(refs, "patch_description"));
  addField(fields, "corrected start time", recordValue(refs, "corrected_start_time"));
  addField(fields, "corrected end time", recordValue(refs, "corrected_end_time"));
  addField(fields, "suggested valid to", recordValue(refs, "suggested_valid_to"));
  addField(fields, "repair text", recordValue(refs, "reason"));
  addField(fields, "patch", compactPatchSummary(recordValue(refs, "patch")));

  const nextPeriod = recordValue(refs, "next_period_open_payload");
  if (isRecord(nextPeriod)) {
    addField(fields, "next period field count", Object.keys(nextPeriod).length);
  }

  addTopLevelArrayCounts(fields, refs);
  return fields;
}

function correctionDetailFields(refs: Record<string, unknown>): DetailField[] {
  const fields: DetailField[] = [];

  addField(fields, "prompt summary", recordValue(refs, "prompt_summary"));
  addField(fields, "operator reason", recordValue(refs, "operator_reason"));

  return fields;
}

function skillSplitDetailFields(refs: Record<string, unknown>): DetailField[] {
  const fields: DetailField[] = [];
  const proposal =
    firstString(recordValue(refs, "rationale")) ?? compactPatchSummary(refs.proposal);
  const children = Array.isArray(refs.proposed_children) ? refs.proposed_children : refs.splits;

  addField(fields, "proposal", proposal);
  addArrayCount(fields, "split count", children);

  return fields;
}

function creatorDirectiveReconciliationDetailFields(refs: Record<string, unknown>): DetailField[] {
  const fields: DetailField[] = [];
  const judgment = recordValue(refs, "judgment");

  addField(fields, "subkind", recordValue(refs, "subkind"));
  addArrayCount(fields, "member count", recordValue(refs, "members"));
  addField(fields, "rationale", isRecord(judgment) ? judgment.rationale : undefined);

  return fields;
}

function commitmentReconciliationDetailFields(refs: Record<string, unknown>): DetailField[] {
  const fields: DetailField[] = [];
  const judgment = recordValue(refs, "judgment");

  addField(fields, "subkind", recordValue(refs, "subkind"));
  addArrayCount(fields, "commitment count", recordValue(refs, "commitment_ids"));
  addField(fields, "reason", recordValue(refs, "reason"));
  addField(fields, "judgment reason", isRecord(judgment) ? judgment.reason : undefined);

  return fields;
}

function detailFields(row: ReviewRow): DetailField[] {
  switch (row.kind) {
    case "contradiction":
    case "duplicate":
      return pairDetailFields(row.refs);
    case "new_insight":
      return newInsightDetailFields(row.refs);
    case "belief_revision":
    case "misattribution":
    case "identity_inconsistency":
    case "temporal_drift":
      return repairDetailFields(row.refs);
    case "skill_split":
      return skillSplitDetailFields(row.refs);
    case "correction":
      return correctionDetailFields(row.refs);
    case "creator_directive_reconciliation":
      return creatorDirectiveReconciliationDetailFields(row.refs);
    case "commitment_reconciliation":
      return commitmentReconciliationDetailFields(row.refs);
  }

  return [];
}

function memberId(member: unknown): string | null {
  return isRecord(member) ? firstString(member.id) : null;
}

function memberScope(member: unknown): Record<string, unknown> {
  if (!isRecord(member) || !isRecord(member.scope_equivalence)) {
    return {};
  }

  return member.scope_equivalence;
}

function scopeFieldValue(member: unknown, field: ScopeField): unknown {
  const scope = memberScope(member);
  const disclosure = isRecord(scope.disclosure_policy) ? scope.disclosure_policy : {};
  const activation = isRecord(scope.activation_policy) ? scope.activation_policy : {};

  switch (field) {
    case "content_scope":
      return disclosure.content_scope;
    case "mention_policy":
      return disclosure.mention_policy;
    case "disclosure_allowed":
      return disclosure.allowed_entity_ids;
    case "disclosure_excluded":
      return disclosure.excluded_entity_ids;
    case "activation_scope":
      return activation.scope;
    case "activation_allowed":
      return activation.allowed_entity_ids;
    case "activation_excluded":
      return activation.excluded_entity_ids;
  }
}

function scopeFieldDisplayValue(member: unknown, field: ScopeField): unknown {
  const value = scopeFieldValue(member, field);

  if (
    field === "disclosure_allowed" ||
    field === "disclosure_excluded" ||
    field === "activation_allowed" ||
    field === "activation_excluded"
  ) {
    return Array.isArray(value) ? `${value.length} entities` : value;
  }

  return value;
}

function scopeDifferences(members: readonly unknown[]): Set<ScopeField> {
  const fields: ScopeField[] = [
    "content_scope",
    "mention_policy",
    "disclosure_allowed",
    "disclosure_excluded",
    "activation_scope",
    "activation_allowed",
    "activation_excluded",
  ];
  const differing = new Set<ScopeField>();

  for (const field of fields) {
    const values = new Set(members.map((member) => displayValue(scopeFieldValue(member, field))));
    if (values.size > 1) {
      differing.add(field);
    }
  }

  return differing;
}

function ReviewDetail({ row }: { row: ReviewRow }) {
  const diagnostics = diagnosticEntries(row);

  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div className="props">
        {detailFields(row).map(([key, value]) => (
          <div className="row" key={key}>
            <span className="k">{key}</span>
            <span className="v" style={{ whiteSpace: "pre-wrap" }}>
              {displayValue(value)}
            </span>
          </div>
        ))}
      </div>
      {diagnostics.length === 0 ? null : (
        <div>
          <div className="divider">resolver diagnostic</div>
          <div className="props">
            {diagnostics.map(([key, value]) => (
              <div className="row" key={key}>
                <span className="k">{key}</span>
                <span className="v" style={{ whiteSpace: "pre-wrap" }}>
                  {displayValue(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function PairEpisodeRefs({ groups }: { groups: readonly EpisodeRefGroup[] }) {
  if (groups.length === 0) {
    return null;
  }

  return (
    <div>
      <div className="divider">episode refs</div>
      <div className="props">
        {groups.map((group) => (
          <div className="row" key={group.label}>
            <span className="k">{group.label}</span>
            <span className="v">{group.ids.join(", ")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

type SemanticNodeDrillResult = {
  id: string;
  node: SemanticMemoryNode | null;
  error: string | null;
};

type SemanticEdgeDrillResult = {
  id: string;
  edge: SemanticMemoryEdge | null;
  error: string | null;
};

async function fetchSemanticNodeDrillResult(id: string): Promise<SemanticNodeDrillResult> {
  try {
    return { id, node: await getSemanticNode(id), error: null };
  } catch (caught) {
    return {
      id,
      node: null,
      error: caught instanceof Error ? caught.message : String(caught),
    };
  }
}

async function fetchSemanticEdgeDrillResult(id: string): Promise<SemanticEdgeDrillResult> {
  try {
    return { id, edge: await getSemanticEdge(id), error: null };
  } catch (caught) {
    return {
      id,
      edge: null,
      error: caught instanceof Error ? caught.message : String(caught),
    };
  }
}

function SemanticNodeUnavailable({
  label,
  result,
}: {
  label: string;
  result?: SemanticNodeDrillResult;
}) {
  return (
    <div className="notice bad">
      <div>{label} semantic node unavailable</div>
      <div style={{ marginTop: 6, overflowWrap: "anywhere" }}>
        {result?.id ?? "missing node id"}
      </div>
      {result?.error === null || result?.error === undefined ? null : (
        <div style={{ marginTop: 4, overflowWrap: "anywhere" }}>{result.error}</div>
      )}
    </div>
  );
}

function SemanticNodeDrillSlot({
  result,
  label,
}: {
  result?: SemanticNodeDrillResult;
  label: string;
}) {
  if (result === undefined || result.node === null) {
    return <SemanticNodeUnavailable label={label} result={result} />;
  }

  return <SemanticNodeDetail node={result.node} label={label} />;
}

function SemanticEdgeDrillSlot({
  result,
  nodes,
}: {
  result: SemanticEdgeDrillResult | null;
  nodes: readonly SemanticMemoryNode[];
}) {
  if (result === null) {
    return <div className="notice">semantic edge unavailable</div>;
  }

  if (result.edge === null) {
    return (
      <div className="notice bad">
        <div>semantic edge unavailable</div>
        <div style={{ marginTop: 6, overflowWrap: "anywhere" }}>{result.id}</div>
        {result.error === null ? null : (
          <div style={{ marginTop: 4, overflowWrap: "anywhere" }}>{result.error}</div>
        )}
      </div>
    );
  }

  return <SemanticEdgeDetail edge={result.edge} nodes={nodes} />;
}

function ReviewPairDrillthrough({ row, open }: { row: ReviewRow; open: boolean }) {
  const ids = nodeIds(row);
  const edgeId = reviewEdgeId(row);
  const nodeKey = ids.join("|");
  const [nodeResults, setNodeResults] = useState<SemanticNodeDrillResult[] | null>(null);
  const [edgeResult, setEdgeResult] = useState<SemanticEdgeDrillResult | null>(null);
  const [loading, setLoading] = useState(false);
  const episodeGroups = pairEpisodeRefGroups(row.refs);
  const availableNodes =
    nodeResults === null
      ? []
      : nodeResults.flatMap((result) => (result.node === null ? [] : [result.node]));

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    let cancelled = false;
    setLoading(true);
    setNodeResults(null);
    setEdgeResult(null);

    void Promise.all([
      Promise.all(ids.map((id) => fetchSemanticNodeDrillResult(id))),
      edgeId === null ? Promise.resolve(null) : fetchSemanticEdgeDrillResult(edgeId),
    ])
      .then(([nextNodeResults, nextEdgeResult]) => {
        if (!cancelled) {
          setNodeResults(nextNodeResults);
          setEdgeResult(nextEdgeResult);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [edgeId, nodeKey, open]);

  if (!open) {
    return null;
  }

  return (
    <div style={{ display: "grid", gap: 10, marginBottom: 14 }}>
      {loading ? <div className="notice">loading semantic drill-through</div> : null}
      {nodeResults === null ? null : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              edgeId === null ? "repeat(2, minmax(220px, 1fr))" : "1fr minmax(220px, 300px) 1fr",
            gap: 10,
          }}
        >
          <SemanticNodeDrillSlot result={nodeResults[0]} label="candidate 1" />
          {edgeId === null ? null : (
            <SemanticEdgeDrillSlot result={edgeResult} nodes={availableNodes} />
          )}
          <SemanticNodeDrillSlot result={nodeResults[1]} label="candidate 2" />
        </div>
      )}
      <PairEpisodeRefs groups={episodeGroups} />
    </div>
  );
}

function GenericReviewActions({
  row,
  busy,
  note,
  winner,
  onNote,
  onWinner,
  onAction,
}: {
  row: ReviewRow;
  busy: BusyState;
  note: string;
  winner: string;
  onNote: (value: string) => void;
  onWinner: (value: string) => void;
  onAction: (action: ReviewResolution) => void;
}) {
  const actions = GENERIC_REVIEW_ACTIONS[row.kind];
  const ids = nodeIds(row);
  const hasWinnerPicker = ids.length > 0 && actions.some((action) => action === "supersede");

  return (
    <div style={{ display: "grid", gap: 8 }}>
      <label className="modal-field">
        <span>note</span>
        <input value={note} onChange={(event) => onNote(event.target.value)} />
      </label>
      {hasWinnerPicker ? (
        <label className="modal-field">
          <span>winner node</span>
          <select value={winner} onChange={(event) => onWinner(event.target.value)}>
            {ids.map((id, index) => (
              <option value={id} key={id}>
                {nodeOptionLabel(row, id, index)}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <div className="operator-actions">
        {actions.map((action) => {
          const needsWinner = ids.length > 0 && (action === "supersede" || action === "invalidate");
          return (
            <button
              className={
                action === "accept" || action === "supersede" ? "btn sm primary" : "btn sm ghost"
              }
              disabled={busy !== null || (needsWinner && winner.length === 0)}
              key={action}
              type="button"
              onClick={() => onAction(action)}
            >
              {busy?.id === row.id && busy.label === action ? "saving" : actionLabel(action)}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ReconciliationReview({
  row,
  directivesById,
  busy,
  survivor,
  onSurvivor,
  onSupersede,
  onKeep,
}: {
  row: ReviewRow;
  directivesById: Map<string, CreatorDirectiveItem>;
  busy: BusyState;
  survivor: string;
  onSurvivor: (value: string) => void;
  onSupersede: () => void;
  onKeep: () => void;
}) {
  const refs = row.refs;
  const members = Array.isArray(refs.members) ? refs.members : [];
  const ids = directiveIds(row);
  const judgment = isRecord(refs.judgment) ? refs.judgment : {};
  const differences = scopeDifferences(members);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="props">
        <div className="row">
          <span className="k">subkind</span>
          <span className="v">{displayValue(refs.subkind)}</span>
        </div>
        <div className="row">
          <span className="k">member count</span>
          <span className="v">{members.length}</span>
        </div>
        <div className="row">
          <span className="k">rationale</span>
          <span className="v">{displayValue(judgment.rationale)}</span>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 10,
        }}
      >
        {members.map((member, index) => {
          const id = memberId(member) ?? ids[index] ?? `member-${index + 1}`;
          const label = `member ${index + 1}`;
          const directive = directivesById.get(id);
          return (
            <DirectiveMemberCard
              key={id}
              label={label}
              member={member}
              directive={directive}
              differences={differences}
            />
          );
        })}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(220px, 320px) minmax(240px, 1fr)",
          gap: 10,
          alignItems: "end",
        }}
      >
        <label className="modal-field">
          <span>scope survivor</span>
          <select value={survivor} onChange={(event) => onSurvivor(event.target.value)}>
            {ids.map((id, index) => {
              const directive = directivesById.get(id);
              return (
                <option value={id} key={id}>
                  member {index + 1} {directive?.content_scope ?? ""}
                </option>
              );
            })}
          </select>
        </label>
        <div className="operator-actions">
          <button
            type="button"
            className="btn sm primary"
            disabled={busy !== null || survivor.length === 0}
            onClick={onSupersede}
          >
            {busy?.id === row.id && busy.label === "supersede" ? "saving" : "supersede to survivor"}
          </button>
          <button type="button" className="btn sm ghost" disabled={busy !== null} onClick={onKeep}>
            {busy?.id === row.id && busy.label === "keep" ? "saving" : "keep both"}
          </button>
        </div>
      </div>
    </div>
  );
}

function DirectiveMemberCard({
  label,
  member,
  directive,
  differences,
}: {
  label: string;
  member: unknown;
  directive?: CreatorDirectiveItem;
  differences: Set<ScopeField>;
}) {
  const rows: Array<[ScopeField, string]> = [
    ["content_scope", "content"],
    ["mention_policy", "mention"],
    ["disclosure_allowed", "allowed"],
    ["disclosure_excluded", "excluded"],
    ["activation_scope", "activation"],
    ["activation_allowed", "act allowed"],
    ["activation_excluded", "act excluded"],
  ];

  return (
    <div className="item" style={{ padding: 12, border: "1px solid var(--line)" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <Tag>{directive?.kind ?? "directive"}</Tag>
        <span className="acc">{label}</span>
      </div>
      <div
        style={{
          marginTop: 8,
          color: "var(--text)",
          fontFamily: "var(--sans)",
          fontSize: 12,
          lineHeight: 1.45,
        }}
      >
        {directive?.text ?? "directive content unavailable"}
      </div>
      <div className="props" style={{ marginTop: 8 }}>
        {rows.map(([field, label]) => (
          <div className="row" key={field}>
            <span className={`k ${differences.has(field) ? "warn" : ""}`}>{label}</span>
            <span className={`v ${differences.has(field) ? "warn" : ""}`}>
              {displayValue(scopeFieldDisplayValue(member, field))}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ReviewScreen() {
  const live = useLiveEventsContext();
  const api = useApi<ReviewData>(async () => {
    const [reviews, directives] = await Promise.all([
      getReviews({ openOnly: true }),
      getCreatorDirectives(),
    ]);
    return { rows: reviews.rows, directives: directives.directives };
  }, []);
  const refetch = api.refetch;
  const previousConnectionCountRef = useRef(live.connectionCount);
  const [busy, setBusy] = useState<BusyState>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [winners, setWinners] = useState<Record<number, string>>({});
  const [survivors, setSurvivors] = useState<Record<number, string>>({});
  const [expandedRows, setExpandedRows] = useState<Record<number, boolean>>({});

  const directivesById = useMemo(() => {
    return new Map((api.data?.directives ?? []).map((directive) => [directive.id, directive]));
  }, [api.data?.directives]);

  const groups = useMemo(() => {
    const rows = api.data?.rows ?? [];
    return REVIEW_KIND_ORDER.map((kind) => ({
      kind,
      rows: rows.filter((row) => row.kind === kind),
    })).filter((group) => group.rows.length > 0);
  }, [api.data?.rows]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type === "maintenance:tick") {
        void refetch();
      }
    });
  }, [live, refetch]);

  useEffect(() => {
    if (live.connectionCount > previousConnectionCountRef.current) {
      void refetch();
    }
    previousConnectionCountRef.current = live.connectionCount;
  }, [live.connectionCount, refetch]);

  async function runReviewAction(id: number, label: string, callback: () => Promise<void>) {
    setBusy({ id, label });
    setOperatorError(null);
    try {
      await callback();
      await refetch();
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  function setNote(id: number, note: string): void {
    setNotes((current) => ({ ...current, [id]: note }));
  }

  function setWinner(id: number, winner: string): void {
    setWinners((current) => ({ ...current, [id]: winner }));
  }

  function setSurvivor(id: number, survivor: string): void {
    setSurvivors((current) => ({ ...current, [id]: survivor }));
  }

  function toggleExpanded(rowId: number): void {
    setExpandedRows((current) => ({ ...current, [rowId]: current[rowId] !== true }));
  }

  async function submitGeneric(row: ReviewRow, action: ReviewResolution): Promise<void> {
    const ids = nodeIds(row);
    const winner = winners[row.id] ?? ids[0] ?? "";
    const note = notes[row.id]?.trim();

    await runReviewAction(row.id, action, async () => {
      await patchReview(row.id, {
        action,
        ...(note === undefined || note.length === 0 ? {} : { note }),
        ...(ids.length > 0 && (action === "supersede" || action === "invalidate")
          ? { winner_node_id: winner }
          : {}),
      });
    });
  }

  async function submitReconciliationSupersede(row: ReviewRow): Promise<void> {
    const ids = directiveIds(row);
    const survivor = survivors[row.id] ?? ids[0] ?? "";
    await runReviewAction(row.id, "supersede", async () => {
      await resolveCreatorDirectiveReconciliation(row.id, {
        action: "supersede",
        survivor_id: survivor,
      });
    });
  }

  async function submitReconciliationKeep(row: ReviewRow): Promise<void> {
    await runReviewAction(row.id, "keep", async () => {
      await resolveCreatorDirectiveReconciliation(row.id, { action: "keep" });
    });
  }

  if (api.loading && api.data === null) {
    return <div className="notice">loading reviews</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <h1>review</h1>
        <span className="desc">open operator review queue</span>
        <span className="spacer" />
        <span className="sep">{api.data?.rows.length ?? 0} open</span>
      </div>

      <div className="page-body" style={{ padding: 18 }}>
        {operatorError === null ? null : (
          <div className="notice bad" style={{ padding: 10, marginBottom: 12 }}>
            {operatorError}
          </div>
        )}
        {groups.length === 0 ? (
          <div className="notice">no open review rows</div>
        ) : (
          <div style={{ display: "grid", gap: 18 }}>
            {groups.map((group) => (
              <section key={group.kind} style={{ display: "grid", gap: 10 }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    borderBottom: "1px solid var(--line)",
                    paddingBottom: 6,
                  }}
                >
                  <Tag>{kindLabel(group.kind)}</Tag>
                  <span className="dim">{group.rows.length} open</span>
                </div>
                <div style={{ display: "grid", gap: 10 }}>
                  {group.rows.map((row) => {
                    const ids = nodeIds(row);
                    const winner = winners[row.id] ?? ids[0] ?? "";
                    const creatorIds = directiveIds(row);
                    const survivor = survivors[row.id] ?? creatorIds[0] ?? "";
                    const pairReview = isPairReview(row);
                    const expanded = expandedRows[row.id] === true;
                    return (
                      <div
                        className="item"
                        key={row.id}
                        style={{ padding: 14, border: "1px solid var(--line)" }}
                      >
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            gap: 10,
                            alignItems: "flex-start",
                            marginBottom: 10,
                          }}
                        >
                          <div style={{ minWidth: 0 }}>
                            <div
                              style={{
                                display: "flex",
                                gap: 8,
                                alignItems: "center",
                                flexWrap: "wrap",
                              }}
                            >
                              <span className="acc">review {row.id}</span>
                              <Tag>{kindLabel(row.kind)}</Tag>
                              <span className="dim">{formatTime(row.created_at)}</span>
                            </div>
                            <div
                              style={{
                                color: "var(--text)",
                                fontFamily: "var(--sans)",
                                fontSize: 12.5,
                                lineHeight: 1.5,
                                marginTop: 7,
                                overflowWrap: "anywhere",
                              }}
                            >
                              {row.reason}
                            </div>
                          </div>
                          {pairReview ? (
                            <button
                              className="btn sm ghost"
                              type="button"
                              onClick={() => toggleExpanded(row.id)}
                            >
                              {expanded ? "hide drill" : "drill"}
                            </button>
                          ) : null}
                        </div>

                        {row.kind === "creator_directive_reconciliation" ? (
                          <ReconciliationReview
                            row={row}
                            directivesById={directivesById}
                            busy={busy}
                            survivor={survivor}
                            onSurvivor={(value) => setSurvivor(row.id, value)}
                            onSupersede={() => void submitReconciliationSupersede(row)}
                            onKeep={() => void submitReconciliationKeep(row)}
                          />
                        ) : (
                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns: "minmax(0, 1fr) 320px",
                              gap: 14,
                            }}
                          >
                            <div style={{ display: "grid", gap: 10, minWidth: 0 }}>
                              <ReviewPairDrillthrough row={row} open={pairReview && expanded} />
                              <ReviewDetail row={row} />
                            </div>
                            <GenericReviewActions
                              row={row}
                              busy={busy}
                              note={notes[row.id] ?? ""}
                              winner={winner}
                              onNote={(value) => setNote(row.id, value)}
                              onWinner={(value) => setWinner(row.id, value)}
                              onAction={(action) => void submitGeneric(row, action)}
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
