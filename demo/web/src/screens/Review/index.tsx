import { useEffect, useMemo, useRef, useState } from "react";

import {
  ApiError,
  getCommitments,
  getCreatorDirectives,
  getWhy,
  getSemanticEdge,
  getSemanticNode,
  getReviews,
  postCorrectionCorrect,
  postCorrectionForget,
  postSemanticEdgeInvalidate,
} from "../../api/client";
import type {
  CommitmentItem,
  CreatorDirectiveItem,
  ReviewKind,
  ReviewResolution,
  ReviewRow,
  SemanticMemoryEdge,
  SemanticMemoryNode,
} from "../../api/types";
import { Empty } from "../../components/Empty";
import { ErrorState } from "../../components/ErrorState";
import { IdRef } from "../../components/Inspector/IdRef";
import { resolveObjectType, type ObjectType } from "../../components/Inspector/inspector-id";
import { isWhySupported } from "../../components/Inspector/inspector-registry";
import { JsonValueView } from "../../components/JsonValueView";
import { Loading } from "../../components/Loading";
import { Modal } from "../../components/Modal";
import { SemanticEdgeDetail } from "../../components/SemanticEdgeDetail";
import { SemanticNodeDetail } from "../../components/SemanticNodeDetail";
import { Tag } from "../../components/Tag";
import { IdChip } from "../../components/Inspector/IdChip";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import {
  DESTRUCTIVE_REVIEW_ACTIONS,
  GENERIC_REVIEW_ACTIONS,
  resolveReviewAction,
} from "../../lib/review-actions";
import { isInteractiveDescendantEvent } from "../../lib/keyboard";
import { formatTimestamp } from "../../lib/stream-utils";
import { displayValue, fieldLabel, isRecord, parseJsonPatch, shortId } from "../screen-utils";

export { GENERIC_REVIEW_ACTIONS } from "../../lib/review-actions";

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

const REVIEW_RESOLVER_REF_PREFIX = "__borg_review_resolver_";

type ReviewData = {
  rows: ReviewRow[];
  directives: CreatorDirectiveItem[];
  commitments: CommitmentItem[];
};

type BusyState = {
  id: number;
  label: string;
} | null;

type ReviewMode = "queue" | "lab";
type AgeBucket = "all" | "hour" | "day" | "week" | "older";
type AffectedTypeFilter = "all" | ObjectType;

type PendingReviewAction = {
  row: ReviewRow;
  action: ReviewResolution;
  label: string;
} | null;

type PendingLabAction =
  | {
      kind: "forget";
      id: string;
    }
  | {
      kind: "invalidate";
      id: string;
    };

type ScopeField =
  | "content_scope"
  | "mention_policy"
  | "disclosure_allowed"
  | "disclosure_excluded"
  | "activation_scope"
  | "activation_allowed"
  | "activation_excluded";

const AGE_FILTERS: readonly { value: AgeBucket; label: string }[] = [
  { value: "all", label: "all ages" },
  { value: "hour", label: "last hour" },
  { value: "day", label: "last day" },
  { value: "week", label: "last week" },
  { value: "older", label: "older" },
];

function recordValue(record: Record<string, unknown>, key: string): unknown {
  return Object.hasOwn(record, key) ? record[key] : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function typedTargetType(value: unknown): ObjectType | null {
  switch (value) {
    case "semantic_node":
    case "semantic_edge":
    case "episode":
    case "creator_directive":
    case "commitment":
    case "entity":
      return value;
    default:
      return null;
  }
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

function optionalNote(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed === undefined || trimmed.length === 0 ? undefined : trimmed;
}

function isCorrectableId(id: string): boolean {
  const type = resolveObjectType(id);
  return type !== null && isWhySupported(type);
}

function parseLabPatch(text: string): Record<string, unknown> {
  return parseJsonPatch(text);
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

type ReviewRefGroup = {
  label: string;
  type: ObjectType;
  ids: string[];
};

function addReviewStringRef(
  groups: ReviewRefGroup[],
  label: string,
  type: ObjectType,
  value: unknown,
): void {
  const id = firstString(value);
  if (id !== null) {
    groups.push({ label, type, ids: [id] });
  }
}

function addReviewArrayRefs(
  groups: ReviewRefGroup[],
  label: string,
  type: ObjectType,
  value: unknown,
): void {
  const ids = stringArray(value);
  if (ids.length > 0) {
    groups.push({ label, type, ids });
  }
}

function reviewRefGroups(refs: Record<string, unknown>): ReviewRefGroup[] {
  const groups: ReviewRefGroup[] = [];
  addReviewArrayRefs(groups, "refs.node_ids", "semantic_node", recordValue(refs, "node_ids"));
  addReviewStringRef(groups, "refs.edge_id", "semantic_edge", recordValue(refs, "edge_id"));
  addReviewArrayRefs(
    groups,
    "refs.directive_ids",
    "creator_directive",
    recordValue(refs, "directive_ids"),
  );
  addReviewArrayRefs(
    groups,
    "refs.commitment_ids",
    "commitment",
    recordValue(refs, "commitment_ids"),
  );

  const targetId = firstString(recordValue(refs, "target_id"));
  if (targetId !== null) {
    const targetType =
      typedTargetType(recordValue(refs, "target_type")) ?? resolveObjectType(targetId);
    if (targetType !== null) {
      groups.push({ label: "refs.target_id", type: targetType, ids: [targetId] });
    }
  }

  addReviewStringRef(
    groups,
    "refs.invalidated_edge_id",
    "semantic_edge",
    recordValue(refs, "invalidated_edge_id"),
  );
  addReviewArrayRefs(
    groups,
    "refs.dependency_path_edge_ids",
    "semantic_edge",
    recordValue(refs, "dependency_path_edge_ids"),
  );
  addReviewArrayRefs(
    groups,
    "refs.surviving_support_edge_ids",
    "semantic_edge",
    recordValue(refs, "surviving_support_edge_ids"),
  );
  addReviewArrayRefs(groups, "refs.episode_ids", "episode", recordValue(refs, "episode_ids"));
  addReviewArrayRefs(
    groups,
    "refs.evidence_episode_ids",
    "episode",
    recordValue(refs, "evidence_episode_ids"),
  );

  const sourceOverlap = recordValue(refs, "source_overlap");
  if (isRecord(sourceOverlap)) {
    addReviewArrayRefs(
      groups,
      "refs.source_overlap.candidate_source_episode_ids",
      "episode",
      recordValue(sourceOverlap, "candidate_source_episode_ids"),
    );
    addReviewArrayRefs(
      groups,
      "refs.source_overlap.matched_source_episode_ids",
      "episode",
      recordValue(sourceOverlap, "matched_source_episode_ids"),
    );
    addReviewArrayRefs(
      groups,
      "refs.source_overlap.overlapping_source_episode_ids",
      "episode",
      recordValue(sourceOverlap, "overlapping_source_episode_ids"),
    );
  }

  return groups;
}

const KNOWN_PREFIXED_STRING_REF_KEYS = [
  "target_id",
  "source_target_id",
  "edge_id",
  "by_edge_id",
  "invalidated_edge_id",
  "invalidated_by_edge_id",
  "target_node_id",
  "survivor_commitment_id",
] as const;

const KNOWN_PREFIXED_ARRAY_REF_KEYS = [
  "node_ids",
  "edge_ids",
  "episode_ids",
  "evidence_episode_ids",
  "source_episode_ids",
  "key_episode_ids",
  "related_episode_ids",
  "resolution_evidence_episode_ids",
  "directive_ids",
  "commitment_ids",
  "member_ids",
  "dependency_path_edge_ids",
  "surviving_support_edge_ids",
  "superseded_commitment_ids",
] as const;

const KNOWN_SOURCE_OVERLAP_ARRAY_REF_KEYS = [
  "candidate_source_episode_ids",
  "matched_source_episode_ids",
  "overlapping_source_episode_ids",
] as const;

function addTypeForKnownId(value: unknown, output: Set<ObjectType>): void {
  if (typeof value !== "string") {
    return;
  }

  const type = resolveObjectType(value);
  if (type !== null) {
    output.add(type);
  }
}

function addTypesForKnownIds(values: unknown, output: Set<ObjectType>): void {
  for (const value of stringArray(values)) {
    addTypeForKnownId(value, output);
  }
}

function addMemberIdTypes(value: unknown, output: Set<ObjectType>): void {
  if (!Array.isArray(value)) {
    return;
  }

  for (const member of value) {
    if (isRecord(member)) {
      addTypeForKnownId(recordValue(member, "id"), output);
    }
  }
}

function affectedObjectTypes(row: ReviewRow): ObjectType[] {
  const types = new Set<ObjectType>();

  for (const group of reviewRefGroups(row.refs)) {
    types.add(group.type);
  }

  for (const key of KNOWN_PREFIXED_STRING_REF_KEYS) {
    addTypeForKnownId(recordValue(row.refs, key), types);
  }

  for (const key of KNOWN_PREFIXED_ARRAY_REF_KEYS) {
    addTypesForKnownIds(recordValue(row.refs, key), types);
  }

  addMemberIdTypes(recordValue(row.refs, "members"), types);

  const sourceOverlap = recordValue(row.refs, "source_overlap");
  if (isRecord(sourceOverlap)) {
    for (const key of KNOWN_SOURCE_OVERLAP_ARRAY_REF_KEYS) {
      addTypesForKnownIds(recordValue(sourceOverlap, key), types);
    }
  }

  const targetId = firstString(recordValue(row.refs, "target_id"));
  if (targetId !== null) {
    const targetType = typedTargetType(recordValue(row.refs, "target_type"));
    if (targetType !== null) {
      types.add(targetType);
    }
  }

  return [...types].sort();
}

function affectedTypeLabel(type: ObjectType): string {
  return type.replaceAll("_", " ");
}

function ageBucketForCreatedAt(createdAt: number, now = Date.now()): Exclude<AgeBucket, "all"> {
  const ageMs = Math.max(0, now - createdAt);
  const hourMs = 60 * 60 * 1000;
  const dayMs = 24 * hourMs;
  const weekMs = 7 * dayMs;

  if (ageMs < hourMs) {
    return "hour";
  }
  if (ageMs < dayMs) {
    return "day";
  }
  if (ageMs < weekMs) {
    return "week";
  }
  return "older";
}

function matchesAgeBucket(row: ReviewRow, bucket: AgeBucket): boolean {
  return bucket === "all" || ageBucketForCreatedAt(row.created_at) === bucket;
}

function reviewIsOpen(row: ReviewRow): boolean {
  return row.resolved_at === null && row.resolution === null;
}

function ReviewRefList({ group }: { group: ReviewRefGroup }) {
  return (
    <>
      {group.ids.map((id, index) => (
        <span key={id}>
          {index === 0 ? null : ", "}
          <IdChip id={id} type={group.type} />
        </span>
      ))}
    </>
  );
}

function ReviewReferenceSections({ refs }: { refs: Record<string, unknown> }) {
  const groups = reviewRefGroups(refs);
  if (groups.length === 0) {
    return null;
  }

  return (
    <div>
      <div className="divider">refs</div>
      <div className="props">
        {groups.map((group) => (
          <div className="row" key={group.label}>
            <span className="k">{group.label}</span>
            <span className="v">
              <ReviewRefList group={group} />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
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

function isEntityScopeField(field: ScopeField): boolean {
  return (
    field === "disclosure_allowed" ||
    field === "disclosure_excluded" ||
    field === "activation_allowed" ||
    field === "activation_excluded"
  );
}

function ScopeFieldRenderedValue({ member, field }: { member: unknown; field: ScopeField }) {
  const value = scopeFieldValue(member, field);

  if (!isEntityScopeField(field)) {
    return <>{displayValue(scopeFieldDisplayValue(member, field))}</>;
  }

  const ids =
    typeof value === "string"
      ? [value]
      : Array.isArray(value)
        ? value.filter((item): item is string => typeof item === "string")
        : [];

  if (ids.length === 0) {
    return <>{displayValue(scopeFieldDisplayValue(member, field))}</>;
  }

  return (
    <>
      {ids.length} {ids.length === 1 ? "entity" : "entities"}{" "}
      {ids.map((id, index) => (
        <span key={id}>
          {index === 0 ? null : ", "}
          <IdRef id={id} type="entity" label={shortId(id)} />
        </span>
      ))}
    </>
  );
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
      <ReviewReferenceSections refs={row.refs} />
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
            <span className="v">
              {group.ids.map((id, index) => (
                <span key={id}>
                  {index === 0 ? null : ", "}
                  <IdChip id={id} type="episode" />
                </span>
              ))}
            </span>
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
        {result?.id === undefined ? (
          "missing node id"
        ) : (
          <IdChip id={result.id} type="semantic_node" />
        )}
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
        <div style={{ marginTop: 6, overflowWrap: "anywhere" }}>
          <IdChip id={result.id} type="semantic_edge" />
        </div>
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
    <div className="review-pair-drillthrough">
      {loading ? <div className="notice">loading semantic drill-through</div> : null}
      {nodeResults === null ? null : (
        <div
          className={`review-pair-grid ${edgeId === null ? "review-pair-grid-two" : "review-pair-grid-three"}`}
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
        <div className="modal-field">
          <span>winner node</span>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <select
              aria-label="winner node"
              value={winner}
              onChange={(event) => onWinner(event.target.value)}
            >
              {ids.map((id, index) => (
                <option value={id} key={id}>
                  {nodeOptionLabel(row, id, index)}
                </option>
              ))}
            </select>
            {winner.length === 0 ? null : (
              <IdRef id={winner} type="semantic_node" label={shortId(winner)} />
            )}
          </div>
        </div>
      ) : null}
      <div className="operator-actions">
        {actions.map((action) => {
          const needsWinner = ids.length > 0 && (action === "supersede" || action === "invalidate");
          return (
            <button
              className={
                DESTRUCTIVE_REVIEW_ACTIONS.has(action)
                  ? "btn sm danger"
                  : action === "accept"
                    ? "btn sm primary"
                    : "btn sm ghost"
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

function CreatorDirectiveReconciliationComparison({
  row,
  directivesById,
  survivor,
  onSurvivor,
}: {
  row: ReviewRow;
  directivesById: Map<string, CreatorDirectiveItem>;
  survivor: string;
  onSurvivor: (value: string) => void;
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
      <ReviewReferenceSections refs={refs} />

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
          display: "flex",
          gap: 10,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <div className="modal-field">
          <span>scope survivor</span>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <select
              aria-label="scope survivor"
              value={survivor}
              onChange={(event) => onSurvivor(event.target.value)}
            >
              {ids.map((id, index) => {
                const directive = directivesById.get(id);
                return (
                  <option value={id} key={id}>
                    member {index + 1} {directive?.content_scope ?? ""}
                  </option>
                );
              })}
            </select>
            {survivor.length === 0 ? null : (
              <IdRef
                id={survivor}
                type="creator_directive"
                label={shortId(survivor)}
                hint={directivesById.get(survivor)}
              />
            )}
          </div>
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
              <ScopeFieldRenderedValue member={member} field={field} />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

type CommitmentScopeField =
  | "type"
  | "kind"
  | "enforcement_class"
  | "critical_domain"
  | "state"
  | "audience"
  | "made_to"
  | "about"
  | "committed_by"
  | "directive_family";

const COMMITMENT_SCOPE_ROWS: readonly [CommitmentScopeField, string][] = [
  ["type", "type"],
  ["kind", "kind"],
  ["enforcement_class", "enforcement"],
  ["critical_domain", "critical domain"],
  ["state", "state"],
  ["audience", "audience"],
  ["made_to", "made to"],
  ["about", "about"],
  ["committed_by", "committed by"],
  ["directive_family", "directive family"],
];

function commitmentIds(row: ReviewRow): string[] {
  return stringArray(recordValue(row.refs, "commitment_ids"));
}

function commitmentScopeValue(commitment: CommitmentItem, field: CommitmentScopeField): unknown {
  return commitment[field];
}

function commitmentScopeDifferences(
  commitments: readonly CommitmentItem[],
): Set<CommitmentScopeField> {
  const differing = new Set<CommitmentScopeField>();

  for (const [field] of COMMITMENT_SCOPE_ROWS) {
    const values = new Set(
      commitments.map((commitment) => displayValue(commitmentScopeValue(commitment, field))),
    );
    if (values.size > 1) {
      differing.add(field);
    }
  }

  return differing;
}

function MaybeIdValue({ value }: { value: unknown }) {
  if (typeof value !== "string" || value.length === 0) {
    return <>{displayValue(value)}</>;
  }

  const type = resolveObjectType(value);
  if (type === null) {
    return <>{value}</>;
  }

  return <IdChip id={value} type={type} />;
}

function CommitmentMemberCard({
  label,
  id,
  commitment,
  differences,
}: {
  label: string;
  id: string;
  commitment?: CommitmentItem;
  differences: Set<CommitmentScopeField>;
}) {
  if (commitment === undefined) {
    return (
      <div className="item" style={{ padding: 12, border: "1px solid var(--line)" }}>
        <div className="notice bad">
          commitment unavailable <IdChip id={id} type="commitment" />
        </div>
      </div>
    );
  }

  return (
    <div className="item" style={{ padding: 12, border: "1px solid var(--line)" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <Tag>{commitment.enforcement_class}</Tag>
        <span className="acc">{label}</span>
        <IdRef
          id={commitment.id}
          type="commitment"
          label={shortId(commitment.id)}
          hint={commitment}
        />
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
        {commitment.text}
      </div>
      <div className="props" style={{ marginTop: 8 }}>
        {COMMITMENT_SCOPE_ROWS.map(([field, label]) => (
          <div className="row" key={field}>
            <span className={`k ${differences.has(field) ? "warn" : ""}`}>{label}</span>
            <span className={`v ${differences.has(field) ? "warn" : ""}`}>
              <MaybeIdValue value={commitmentScopeValue(commitment, field)} />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CommitmentReconciliationComparison({
  row,
  commitmentsById,
}: {
  row: ReviewRow;
  commitmentsById: Map<string, CommitmentItem>;
}) {
  const ids = commitmentIds(row);
  const commitments = ids.flatMap((id) => {
    const commitment = commitmentsById.get(id);
    return commitment === undefined ? [] : [commitment];
  });
  const differences = commitmentScopeDifferences(commitments);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <ReviewDetail row={row} />
      <div className="notice">
        commitment comparison is read-only context; use the resolution panel for generic review
        actions
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 10,
        }}
      >
        {ids.length === 0 ? (
          <div className="notice">no commitment refs on this review</div>
        ) : (
          ids.map((id, index) => (
            <CommitmentMemberCard
              key={id}
              label={`member ${index + 1}`}
              id={id}
              commitment={commitmentsById.get(id)}
              differences={differences}
            />
          ))
        )}
      </div>
    </div>
  );
}

function InlineWhyEvidence({ id }: { id: string }) {
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

  const entries = Object.entries(api.data);
  if (entries.length === 0) {
    return <Empty>no provenance fields</Empty>;
  }

  return (
    <div className="why-drawer">
      {entries.map(([key, value]) => (
        <details key={key} className="why-section" open>
          <summary>{key}</summary>
          <JsonValueView value={value} />
        </details>
      ))}
    </div>
  );
}

function ReviewQueuePane({
  groups,
  selectedRowId,
  emptyMessage,
  onSelect,
}: {
  groups: readonly { kind: ReviewKind; rows: ReviewRow[] }[];
  selectedRowId: number | null;
  emptyMessage: string;
  onSelect: (row: ReviewRow) => void;
}) {
  if (groups.length === 0) {
    return <Empty>{emptyMessage}</Empty>;
  }

  return (
    <div className="review-queue-list">
      {groups.map((group) => (
        <section key={group.kind} className="review-kind-group">
          <div className="review-kind-head">
            <Tag>{kindLabel(group.kind)}</Tag>
            <span className="dim">{group.rows.length}</span>
          </div>
          <div className="review-kind-rows">
            {group.rows.map((row) => {
              const affected = affectedObjectTypes(row);
              const selected = row.id === selectedRowId;
              return (
                <div
                  key={row.id}
                  className={`review-queue-row${selected ? " selected" : ""}`}
                  onClick={(event) => {
                    if (!isInteractiveDescendantEvent(event.currentTarget, event.target)) {
                      onSelect(row);
                    }
                  }}
                >
                  <span className="review-row-top">
                    <span>
                      <IdChip
                        id={String(row.id)}
                        type="review"
                        label={`review ${row.id}`}
                        hint={row}
                      />
                    </span>
                    <span className="dim">{formatTimestamp(row.created_at)}</span>
                  </span>
                  <button
                    type="button"
                    className="review-row-reason review-row-select"
                    aria-pressed={selected}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelect(row);
                    }}
                  >
                    {row.reason}
                  </button>
                  <span className="review-row-meta">
                    {reviewIsOpen(row) ? (
                      <Tag>open</Tag>
                    ) : (
                      <Tag>
                        {row.resolution === null ? "resolved" : actionLabel(row.resolution)}
                      </Tag>
                    )}
                    {affected.length === 0 ? (
                      <span className="dim">affected unknown</span>
                    ) : (
                      affected.map((type) => (
                        <span className="dim" key={type}>
                          {affectedTypeLabel(type)}
                        </span>
                      ))
                    )}
                  </span>
                </div>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}

function ReviewQuietPane() {
  return (
    <div className="review-quiet-pane" aria-hidden="true">
      —
    </div>
  );
}

function ReviewEvidencePane({
  row,
  queueEmpty,
  directivesById,
  commitmentsById,
  survivor,
  onSurvivor,
}: {
  row: ReviewRow | null;
  queueEmpty: boolean;
  directivesById: Map<string, CreatorDirectiveItem>;
  commitmentsById: Map<string, CommitmentItem>;
  survivor: string;
  onSurvivor: (value: string) => void;
}) {
  if (row === null) {
    if (queueEmpty) {
      return <ReviewQuietPane />;
    }
    return <Empty>select a review row</Empty>;
  }

  return (
    <div className="review-evidence">
      <div className="review-selected-head">
        <div>
          <div className="eyebrow">selected review</div>
          <div className="review-title-line">
            <IdRef id={String(row.id)} type="review" label={`review ${row.id}`} hint={row} />
            <Tag>{kindLabel(row.kind)}</Tag>
            {reviewIsOpen(row) ? <Tag>open</Tag> : <Tag>{row.resolution ?? "resolved"}</Tag>}
          </div>
        </div>
        <span className="dim">{formatTimestamp(row.created_at)}</span>
      </div>
      <div className="review-reason">{row.reason}</div>

      {isPairReview(row) ? <ReviewPairDrillthrough row={row} open /> : null}
      {row.kind === "creator_directive_reconciliation" ? (
        <CreatorDirectiveReconciliationComparison
          row={row}
          directivesById={directivesById}
          survivor={survivor}
          onSurvivor={onSurvivor}
        />
      ) : row.kind === "commitment_reconciliation" ? (
        <CommitmentReconciliationComparison row={row} commitmentsById={commitmentsById} />
      ) : (
        <ReviewDetail row={row} />
      )}
    </div>
  );
}

function CreatorDirectiveResolutionPanel({
  row,
  busy,
  note,
  survivor,
  onNote,
  onSupersede,
  onKeep,
}: {
  row: ReviewRow;
  busy: BusyState;
  note: string;
  survivor: string;
  onNote: (value: string) => void;
  onSupersede: () => void;
  onKeep: () => void;
}) {
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <label className="modal-field">
        <span>note</span>
        <input value={note} onChange={(event) => onNote(event.target.value)} />
      </label>
      <div className="operator-actions">
        <button
          type="button"
          className="btn sm danger"
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
  );
}

function ReviewResolutionPanel({
  row,
  queueEmpty,
  busy,
  note,
  winner,
  survivor,
  onNote,
  onWinner,
  onAction,
  onCdrSupersede,
  onCdrKeep,
}: {
  row: ReviewRow | null;
  queueEmpty: boolean;
  busy: BusyState;
  note: string;
  winner: string;
  survivor: string;
  onNote: (value: string) => void;
  onWinner: (value: string) => void;
  onAction: (action: ReviewResolution) => void;
  onCdrSupersede: () => void;
  onCdrKeep: () => void;
}) {
  if (row === null) {
    if (queueEmpty) {
      return <ReviewQuietPane />;
    }
    return <Empty>select a review row to repair</Empty>;
  }

  if (!reviewIsOpen(row)) {
    return (
      <div className="notice">
        resolved as {row.resolution === null ? "resolved" : actionLabel(row.resolution)}
      </div>
    );
  }

  if (row.kind === "creator_directive_reconciliation") {
    return (
      <CreatorDirectiveResolutionPanel
        row={row}
        busy={busy}
        note={note}
        survivor={survivor}
        onNote={onNote}
        onSupersede={onCdrSupersede}
        onKeep={onCdrKeep}
      />
    );
  }

  return (
    <GenericReviewActions
      row={row}
      busy={busy}
      note={note}
      winner={winner}
      onNote={onNote}
      onWinner={onWinner}
      onAction={onAction}
    />
  );
}

function CorrectionLab({
  id,
  patch,
  reason,
  busy,
  error,
  status,
  onId,
  onPatch,
  onReason,
  onCorrect,
  onForget,
  onInvalidate,
}: {
  id: string;
  patch: string;
  reason: string;
  busy: string | null;
  error: string | null;
  status: string | null;
  onId: (value: string) => void;
  onPatch: (value: string) => void;
  onReason: (value: string) => void;
  onCorrect: () => void;
  onForget: () => void;
  onInvalidate: () => void;
}) {
  const trimmedId = id.trim();
  const objectType = trimmedId.length === 0 ? null : resolveObjectType(trimmedId);
  const correctable = trimmedId.length > 0 && objectType !== null && isWhySupported(objectType);

  return (
    <div className="page-body review-lab">
      <section className="panel">
        <div className="panel-header">
          <span className="title">Correction Lab</span>
          <span className="spacer" />
          {objectType === null ? null : <Tag>{affectedTypeLabel(objectType)}</Tag>}
        </div>
        <div className="panel-body pad">
          <div className="modal-form">
            <label className="modal-field">
              <span>object id</span>
              <input
                value={id}
                placeholder="stored-object id"
                onChange={(event) => onId(event.target.value)}
              />
            </label>
            {trimmedId.length === 0 ? (
              <div className="notice">enter a correctable stored-object id</div>
            ) : correctable && objectType !== null ? (
              <div className="notice">correctable {affectedTypeLabel(objectType)}</div>
            ) : (
              <div className="notice bad">not a correctable id</div>
            )}
            {status === null ? null : <div className="notice">{status}</div>}
            {error === null ? null : <div className="notice bad">{error}</div>}
          </div>
        </div>
      </section>

      {correctable ? (
        <div className="review-lab-grid">
          <section className="panel">
            <div className="panel-header">
              <span className="title">why?</span>
            </div>
            <div className="panel-body pad">
              <InlineWhyEvidence id={trimmedId} />
            </div>
          </section>
          <section className="panel">
            <div className="panel-header">
              <span className="title">repair actions</span>
            </div>
            <div className="panel-body pad">
              <div className="modal-form">
                <label className="modal-field">
                  <span>json patch</span>
                  <textarea value={patch} onChange={(event) => onPatch(event.target.value)} />
                </label>
                <label className="modal-field">
                  <span>reason</span>
                  <textarea value={reason} onChange={(event) => onReason(event.target.value)} />
                </label>
                <div className="operator-actions">
                  <button
                    type="button"
                    className="btn sm primary"
                    disabled={busy !== null}
                    onClick={onCorrect}
                  >
                    {busy === "correct" ? "saving" : "queue correction"}
                  </button>
                  <button
                    type="button"
                    className="btn sm danger"
                    disabled={busy !== null}
                    onClick={onForget}
                  >
                    forget
                  </button>
                  {objectType === "semantic_edge" ? (
                    <button
                      type="button"
                      className="btn sm danger"
                      disabled={busy !== null}
                      onClick={onInvalidate}
                    >
                      invalidate
                    </button>
                  ) : null}
                </div>
              </div>
            </div>
          </section>
        </div>
      ) : null}
    </div>
  );
}

export function ReviewScreen() {
  const live = useLiveEventsContext();
  const [mode, setMode] = useState<ReviewMode>("queue");
  const [openOnly, setOpenOnly] = useState(true);
  const [kindFilter, setKindFilter] = useState<ReviewKind | "all">("all");
  const [ageFilter, setAgeFilter] = useState<AgeBucket>("all");
  const [affectedTypeFilter, setAffectedTypeFilter] = useState<AffectedTypeFilter>("all");
  const [selectedRowId, setSelectedRowId] = useState<number | null>(null);
  const api = useApi<ReviewData>(async () => {
    const [reviews, directives, commitments] = await Promise.all([
      getReviews({ openOnly }),
      getCreatorDirectives({ status: "all" }),
      getCommitments({ state: "all", enforcement: "all" }),
    ]);
    return {
      rows: reviews.rows,
      directives: directives.directives,
      commitments: commitments.commitments,
    };
  }, [openOnly]);
  const refetch = api.refetch;
  const previousConnectionCountRef = useRef(live.connectionCount);
  const [busy, setBusy] = useState<BusyState>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [winners, setWinners] = useState<Record<number, string>>({});
  const [survivors, setSurvivors] = useState<Record<number, string>>({});
  const [pendingReviewAction, setPendingReviewAction] = useState<PendingReviewAction>(null);
  const [labId, setLabId] = useState("");
  const [labPatch, setLabPatch] = useState("{}");
  const [labReason, setLabReason] = useState("");
  const [labBusy, setLabBusy] = useState<string | null>(null);
  const [labError, setLabError] = useState<string | null>(null);
  const [labStatus, setLabStatus] = useState<string | null>(null);
  const [pendingLabAction, setPendingLabAction] = useState<PendingLabAction | null>(null);

  const directivesById = useMemo(() => {
    return new Map((api.data?.directives ?? []).map((directive) => [directive.id, directive]));
  }, [api.data?.directives]);

  const commitmentsById = useMemo(() => {
    return new Map((api.data?.commitments ?? []).map((commitment) => [commitment.id, commitment]));
  }, [api.data?.commitments]);

  const rows = api.data?.rows ?? [];

  const affectedTypeOptions = useMemo(() => {
    const types = new Set<ObjectType>();
    for (const row of rows) {
      for (const type of affectedObjectTypes(row)) {
        types.add(type);
      }
    }
    return [...types].sort();
  }, [rows]);

  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (kindFilter !== "all" && row.kind !== kindFilter) {
        return false;
      }
      if (!matchesAgeBucket(row, ageFilter)) {
        return false;
      }
      if (
        affectedTypeFilter !== "all" &&
        !affectedObjectTypes(row).some((type) => type === affectedTypeFilter)
      ) {
        return false;
      }
      return true;
    });
  }, [affectedTypeFilter, ageFilter, kindFilter, rows]);

  const groups = useMemo(() => {
    return REVIEW_KIND_ORDER.map((kind) => ({
      kind,
      rows: filteredRows.filter((row) => row.kind === kind),
    })).filter((group) => group.rows.length > 0);
  }, [filteredRows]);

  const selectedRow = useMemo(() => {
    if (selectedRowId === null) {
      return null;
    }
    return filteredRows.find((row) => row.id === selectedRowId) ?? null;
  }, [filteredRows, selectedRowId]);

  useEffect(() => {
    if (filteredRows.length === 0) {
      if (selectedRowId !== null) {
        setSelectedRowId(null);
      }
      return;
    }

    if (selectedRowId === null || !filteredRows.some((row) => row.id === selectedRowId)) {
      setSelectedRowId(filteredRows[0]?.id ?? null);
    }
  }, [filteredRows, selectedRowId]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (
        frame.type === "maintenance:tick" ||
        frame.type === "dream:process:completed" ||
        frame.type === "borg:reset"
      ) {
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

  function requestReviewAction(row: ReviewRow, action: ReviewResolution): void {
    if (DESTRUCTIVE_REVIEW_ACTIONS.has(action)) {
      setPendingReviewAction({ row, action, label: actionLabel(action) });
      return;
    }

    void submitReviewResolution(row, action);
  }

  function requestCreatorDirectiveSupersede(row: ReviewRow): void {
    setPendingReviewAction({ row, action: "supersede", label: "supersede to survivor" });
  }

  async function submitReviewResolution(row: ReviewRow, action: ReviewResolution): Promise<void> {
    const ids = nodeIds(row);
    const winner = winners[row.id] ?? ids[0] ?? "";
    const note = optionalNote(notes[row.id]);
    const directiveMemberIds = directiveIds(row);
    const survivor = survivors[row.id] ?? directiveMemberIds[0] ?? "";

    await runReviewAction(row.id, action, async () => {
      await resolveReviewAction({
        row,
        action,
        note,
        winnerNodeId:
          ids.length > 0 && (action === "supersede" || action === "invalidate")
            ? winner
            : undefined,
        survivorId:
          row.kind === "creator_directive_reconciliation" && action === "supersede"
            ? survivor
            : undefined,
      });
    });
  }

  async function submitReconciliationSupersede(row: ReviewRow): Promise<void> {
    await submitReviewResolution(row, "supersede");
  }

  async function submitReconciliationKeep(row: ReviewRow): Promise<void> {
    await submitReviewResolution(row, "keep");
  }

  async function submitPendingReviewAction(): Promise<void> {
    if (pendingReviewAction === null) {
      return;
    }

    const pending = pendingReviewAction;
    if (pending.row.kind === "creator_directive_reconciliation" && pending.action === "supersede") {
      await submitReconciliationSupersede(pending.row);
    } else {
      await submitReviewResolution(pending.row, pending.action);
    }
    setPendingReviewAction(null);
  }

  async function runLabAction(label: string, callback: () => Promise<void>): Promise<void> {
    setLabBusy(label);
    setLabError(null);
    setLabStatus(null);
    try {
      await callback();
    } catch (caught) {
      setLabError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLabBusy(null);
    }
  }

  function requireLabCorrectableId(): string {
    const id = labId.trim();
    if (!isCorrectableId(id)) {
      throw new Error("not a correctable id");
    }
    return id;
  }

  async function submitLabCorrect(): Promise<void> {
    await runLabAction("correct", async () => {
      const id = requireLabCorrectableId();
      const review = await postCorrectionCorrect(id, {
        patch: parseLabPatch(labPatch),
        ...(optionalNote(labReason) === undefined ? {} : { reason: optionalNote(labReason) }),
      });
      setLabStatus(`queued correction review ${review.id}`);
      await refetch();
      setOpenOnly(true);
      setKindFilter("all");
      setAgeFilter("all");
      setAffectedTypeFilter("all");
      setSelectedRowId(review.id);
      setMode("queue");
    });
  }

  async function submitPendingLabAction(): Promise<void> {
    if (pendingLabAction === null) {
      return;
    }

    const pending = pendingLabAction;
    setPendingLabAction(null);
    if (pending.kind === "forget") {
      await runLabAction("forget", async () => {
        await postCorrectionForget(pending.id);
        setLabStatus(`forgot ${pending.id}`);
        await refetch();
      });
      return;
    }

    await runLabAction("invalidate", async () => {
      await postSemanticEdgeInvalidate(pending.id, {
        ...(optionalNote(labReason) === undefined ? {} : { reason: optionalNote(labReason) }),
      });
      setLabStatus(`invalidated ${pending.id}`);
      await refetch();
    });
  }

  const selectedNodeIds = selectedRow === null ? [] : nodeIds(selectedRow);
  const selectedWinner =
    selectedRow === null ? "" : (winners[selectedRow.id] ?? selectedNodeIds[0] ?? "");
  const selectedDirectiveIds = selectedRow === null ? [] : directiveIds(selectedRow);
  const selectedSurvivor =
    selectedRow === null ? "" : (survivors[selectedRow.id] ?? selectedDirectiveIds[0] ?? "");
  const selectedNote = selectedRow === null ? "" : (notes[selectedRow.id] ?? "");
  const emptyQueueMessage =
    rows.length === 0
      ? openOnly
        ? "no open review rows; review rows are filed by extraction and overseer audits"
        : "no review rows yet; review rows are filed by extraction and overseer audits"
      : "none match this filter";
  const queueEmpty = filteredRows.length === 0;

  if (api.loading && api.data === null && mode === "queue") {
    return <Loading>loading reviews</Loading>;
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <h1>review & repair</h1>
        <span className="desc">operator queue and sanctioned repair lab</span>
        <span className="spacer" />
        <div className="filter-pills" role="tablist" aria-label="review mode">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "queue"}
            className={`pill ${mode === "queue" ? "on" : ""}`}
            onClick={() => setMode("queue")}
          >
            queue
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "lab"}
            className={`pill ${mode === "lab" ? "on" : ""}`}
            onClick={() => setMode("lab")}
          >
            lab
          </button>
        </div>
        <span className="sep">
          {filteredRows.length}/{rows.length} loaded
        </span>
      </div>

      {mode === "lab" ? (
        <CorrectionLab
          id={labId}
          patch={labPatch}
          reason={labReason}
          busy={labBusy}
          error={labError}
          status={labStatus}
          onId={(value) => {
            setLabId(value);
            setLabError(null);
            setLabStatus(null);
          }}
          onPatch={setLabPatch}
          onReason={setLabReason}
          onCorrect={() => void submitLabCorrect()}
          onForget={() => {
            const id = labId.trim();
            if (isCorrectableId(id)) {
              setPendingLabAction({ kind: "forget", id });
            }
          }}
          onInvalidate={() => {
            const id = labId.trim();
            if (resolveObjectType(id) === "semantic_edge") {
              setPendingLabAction({ kind: "invalidate", id });
            }
          }}
        />
      ) : (
        <div className="page-body review-repair-page">
          {api.error === null ? null : (
            <div className="notice bad review-wide-notice">{api.error.message}</div>
          )}
          {operatorError === null ? null : (
            <div className="notice bad review-wide-notice">{operatorError}</div>
          )}
          {api.loading && api.data !== null ? (
            <div className="notice review-wide-notice">refreshing review queue</div>
          ) : null}

          <div className="review-filter-bar">
            <label className="review-toggle">
              <input
                type="checkbox"
                checked={openOnly}
                onChange={(event) => setOpenOnly(event.currentTarget.checked)}
              />
              <span>open only</span>
            </label>
            <label className="modal-field">
              <span>kind</span>
              <select
                aria-label="kind filter"
                value={kindFilter}
                onChange={(event) => setKindFilter(event.target.value as ReviewKind | "all")}
              >
                <option value="all">all kinds</option>
                {REVIEW_KIND_ORDER.map((kind) => (
                  <option key={kind} value={kind}>
                    {kindLabel(kind)}
                  </option>
                ))}
              </select>
            </label>
            <label className="modal-field">
              <span>age</span>
              <select
                aria-label="age filter"
                value={ageFilter}
                onChange={(event) => setAgeFilter(event.target.value as AgeBucket)}
              >
                {AGE_FILTERS.map((filter) => (
                  <option key={filter.value} value={filter.value}>
                    {filter.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="modal-field">
              <span>affected type</span>
              <select
                aria-label="affected type filter"
                value={affectedTypeFilter}
                onChange={(event) =>
                  setAffectedTypeFilter(event.target.value as AffectedTypeFilter)
                }
              >
                <option value="all">all types</option>
                {affectedTypeOptions.map((type) => (
                  <option key={type} value={type}>
                    {affectedTypeLabel(type)}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="review-repair-grid">
            <section className="panel review-pane">
              <div className="panel-header">
                <span className="title">queue</span>
                <span className="spacer" />
                <span className="badge">{filteredRows.length}</span>
              </div>
              <div className="panel-body pad">
                <ReviewQueuePane
                  groups={groups}
                  selectedRowId={selectedRowId}
                  emptyMessage={emptyQueueMessage}
                  onSelect={(row) => setSelectedRowId(row.id)}
                />
              </div>
            </section>

            <section className="panel review-pane">
              <div className="panel-header">
                <span className="title">evidence comparison</span>
              </div>
              <div className="panel-body pad">
                <ReviewEvidencePane
                  row={selectedRow}
                  queueEmpty={queueEmpty}
                  directivesById={directivesById}
                  commitmentsById={commitmentsById}
                  survivor={selectedSurvivor}
                  onSurvivor={(value) => {
                    if (selectedRow !== null) {
                      setSurvivor(selectedRow.id, value);
                    }
                  }}
                />
              </div>
            </section>

            <section className="panel review-pane">
              <div className="panel-header">
                <span className="title">resolution</span>
              </div>
              <div className="panel-body pad">
                <ReviewResolutionPanel
                  row={selectedRow}
                  queueEmpty={queueEmpty}
                  busy={busy}
                  note={selectedNote}
                  winner={selectedWinner}
                  survivor={selectedSurvivor}
                  onNote={(value) => {
                    if (selectedRow !== null) {
                      setNote(selectedRow.id, value);
                    }
                  }}
                  onWinner={(value) => {
                    if (selectedRow !== null) {
                      setWinner(selectedRow.id, value);
                    }
                  }}
                  onAction={(action) => {
                    if (selectedRow !== null) {
                      requestReviewAction(selectedRow, action);
                    }
                  }}
                  onCdrSupersede={() => {
                    if (selectedRow !== null) {
                      requestCreatorDirectiveSupersede(selectedRow);
                    }
                  }}
                  onCdrKeep={() => {
                    if (selectedRow !== null) {
                      void submitReconciliationKeep(selectedRow);
                    }
                  }}
                />
              </div>
            </section>
          </div>
        </div>
      )}

      <Modal
        open={pendingReviewAction !== null}
        title={pendingReviewAction === null ? "confirm review action" : pendingReviewAction.label}
        onClose={() => setPendingReviewAction(null)}
        footer={
          <>
            <button
              type="button"
              className="btn sm ghost"
              disabled={busy !== null}
              onClick={() => setPendingReviewAction(null)}
            >
              cancel
            </button>
            <button
              type="button"
              className={`btn sm ${
                pendingReviewAction !== null &&
                DESTRUCTIVE_REVIEW_ACTIONS.has(pendingReviewAction.action)
                  ? "danger"
                  : "live-write"
              }`}
              disabled={busy !== null}
              onClick={() => void submitPendingReviewAction()}
            >
              confirm {pendingReviewAction?.label ?? "action"}
            </button>
          </>
        }
      >
        <div className="modal-form">
          <div>
            Confirm {pendingReviewAction?.label ?? "this action"} for review{" "}
            {pendingReviewAction?.row.id ?? ""}.
          </div>
          {pendingReviewAction?.row.kind === "creator_directive_reconciliation" &&
          pendingReviewAction.action === "supersede" ? (
            <div className="dim">survivor {selectedSurvivor}</div>
          ) : null}
        </div>
      </Modal>

      <Modal
        open={pendingLabAction !== null}
        title={
          <span className="identity-inline">
            <span>{pendingLabAction?.kind ?? "confirm lab action"}</span>
            {pendingLabAction === null ? null : <IdChip id={pendingLabAction.id} />}
          </span>
        }
        onClose={() => setPendingLabAction(null)}
        footer={
          <>
            <button
              type="button"
              className="btn sm ghost"
              disabled={labBusy !== null}
              onClick={() => setPendingLabAction(null)}
            >
              cancel
            </button>
            <button
              type="button"
              className="btn sm danger"
              disabled={labBusy !== null}
              onClick={() => void submitPendingLabAction()}
            >
              confirm {pendingLabAction?.kind ?? "action"}
            </button>
          </>
        }
      >
        <div className="modal-form">
          <div>
            Confirm {pendingLabAction?.kind ?? "this action"} for {pendingLabAction?.id ?? ""}.
          </div>
          {pendingLabAction?.kind === "invalidate" ? (
            <div className="dim">reason: {optionalNote(labReason) ?? "none"}</div>
          ) : null}
        </div>
      </Modal>
    </div>
  );
}
