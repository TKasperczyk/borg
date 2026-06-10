import { useMemo, useState, type ReactNode } from "react";

import {
  getCommitments,
  getCreatorDirectives,
  getSessions,
  getSharedState,
  getState,
  revokeCreatorDirective,
  supersedeCreatorDirective,
} from "../../api/client";
import type {
  CommitmentItem,
  CreatorDirectivesResponse,
  CreatorDirectiveItem,
  CreatorDirectiveStatus,
  CreatorDirectiveStatusFilter,
  SharedStateEntry,
} from "../../api/types";
import { IdChip } from "../../components/Inspector/IdChip";
import { IdRef } from "../../components/Inspector/IdRef";
import { resolveObjectType, type ObjectType } from "../../components/Inspector/inspector-id";
import { Modal } from "../../components/Modal";
import { SupersededByChip } from "../../components/SupersededByChip";
import { Tag } from "../../components/Tag";
import { useApi, type ApiHookState } from "../../hooks/use-api";
import { isInteractiveDescendantEvent } from "../../lib/keyboard";
import { lifecycleLabel, tagKind } from "../../lib/shared-state-lifecycle";
import { dateLabel, shortId } from "../screen-utils";

type SortMode = "priority_desc" | "priority_asc";

type DirectiveModal =
  | { kind: "revoke"; directive: CreatorDirectiveItem; reason: string }
  | { kind: "supersede"; directive: CreatorDirectiveItem; replacementId: string };
export type SharedStateAudienceEntries = {
  audience: string;
  entries: SharedStateEntry[];
};
export type DirectiveSupportData = {
  audienceDiscoveryTruncated: boolean;
  commitments: CommitmentItem[];
  sharedAudiences: SharedStateAudienceEntries[];
};
type SharedStateRelation =
  | {
      kind: "canonicalized_commitment";
      commitment: CommitmentItem;
      streamIds: string[];
    }
  | {
      kind: "shared_source";
      streamIds: string[];
    };
type RelatedSharedStateEntry = {
  audience: string;
  entry: SharedStateEntry;
  relations: SharedStateRelation[];
};

const STATUS_FILTERS: CreatorDirectiveStatusFilter[] = ["active", "revoked", "superseded", "all"];
const SORT_MODES: SortMode[] = ["priority_desc", "priority_asc"];
export const SESSION_AUDIENCE_DISCOVERY_CAP = 1000;

function uniqueStrings(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];

  for (const value of values) {
    if (value.length === 0 || seen.has(value)) {
      continue;
    }
    seen.add(value);
    result.push(value);
  }

  return result;
}

function intersectStrings(values: readonly string[], source: ReadonlySet<string>): string[] {
  return uniqueStrings(values.filter((value) => source.has(value)));
}

function directiveSourceStreamIds(directive: CreatorDirectiveItem): string[] {
  return uniqueStrings([
    ...directive.authorization_stream_entry_ids,
    ...directive.content_source_stream_entry_ids,
  ]);
}

export async function loadDirectiveSupportData(
  sessionId: string,
  commitments?: readonly CommitmentItem[],
): Promise<DirectiveSupportData> {
  const [sessionsResponse, commitmentsResponse, stateResponse] = await Promise.all([
    getSessions(),
    commitments === undefined
      ? getCommitments({ state: "all" })
      : Promise.resolve({ commitments: [...commitments] }),
    getState({ session: sessionId }),
  ]);
  const audienceLabels = uniqueStrings([
    "self",
    ...stateResponse.audiences,
    ...sessionsResponse.sessions.map((session) => session.audience_label),
  ]);
  const sharedAudiences = await Promise.all(
    audienceLabels.map(async (audience) => {
      const response = await getSharedState(audience);
      return { audience: response.audience, entries: response.entries };
    }),
  );

  return {
    audienceDiscoveryTruncated: sessionsResponse.sessions.length === SESSION_AUDIENCE_DISCOVERY_CAP,
    commitments: commitmentsResponse.commitments,
    sharedAudiences,
  };
}

function statusTag(status: CreatorDirectiveStatus) {
  if (status === "active") {
    return "acc";
  }
  if (status === "revoked") {
    return "bad";
  }
  return "warn";
}

function emptyLabel(value: string | null): string {
  return value === null || value.length === 0 ? "—" : value;
}

function joinedIds(ids: readonly string[]): string {
  return ids.length === 0 ? "—" : ids.map(shortId).join(", ");
}

function InlineIdRefList({ ids, type }: { ids: readonly string[]; type: ObjectType }): ReactNode {
  if (ids.length === 0) {
    return "—";
  }

  return ids.map((id, index) => (
    <span key={id}>
      {index === 0 ? null : ", "}
      <IdChip id={id} type={type} />
    </span>
  ));
}

function sortLabel(sort: SortMode): string {
  return sort === "priority_desc" ? "priority high" : "priority low";
}

function statusRank(status: CreatorDirectiveStatus): number {
  if (status === "active") {
    return 0;
  }
  if (status === "revoked") {
    return 1;
  }
  return 2;
}

function compareDirectives(sort: SortMode) {
  return (left: CreatorDirectiveItem, right: CreatorDirectiveItem): number => {
    const statusDiff = statusRank(left.status) - statusRank(right.status);
    if (statusDiff !== 0) {
      return statusDiff;
    }

    const priorityDiff =
      sort === "priority_desc" ? right.priority - left.priority : left.priority - right.priority;
    if (priorityDiff !== 0) {
      return priorityDiff;
    }

    return left.created_at - right.created_at;
  };
}

function supersededLabel(
  id: string | null,
  directivesById: ReadonlyMap<string, CreatorDirectiveItem>,
): string {
  if (id === null) {
    return "—";
  }

  const directive = directivesById.get(id);
  if (directive === undefined) {
    return shortId(id);
  }

  return `${shortId(id)} · ${directive.kind} · p:${directive.priority}`;
}

function CreatorDirectiveSupersededByChip({
  id,
  directivesById,
  onOpen,
  inspect = false,
}: {
  id: string;
  directivesById: ReadonlyMap<string, CreatorDirectiveItem>;
  onOpen: (id: string) => void;
  inspect?: boolean;
}) {
  return (
    <SupersededByChip
      id={id}
      label={supersededLabel(id, directivesById)}
      title={`Jump to directive ${id}`}
      ariaLabel={`jump to directive ${id}`}
      onOpen={onOpen}
      inspectType={inspect ? "creator_directive" : undefined}
      inspectHint={inspect ? directivesById.get(id) : undefined}
      inspectAriaLabel={inspect ? `inspect directive ${id}` : undefined}
    />
  );
}

function inactiveSummary(
  directive: CreatorDirectiveItem,
  directivesById: ReadonlyMap<string, CreatorDirectiveItem>,
  onOpenDirective: (id: string) => void,
): ReactNode {
  if (directive.status === "revoked") {
    return `revoked: ${directive.revoked_reason ?? "—"}`;
  }
  if (directive.status === "superseded") {
    if (directive.superseded_by_id === null) {
      return "superseded by: —";
    }
    return (
      <>
        superseded by:{" "}
        <CreatorDirectiveSupersededByChip
          id={directive.superseded_by_id}
          directivesById={directivesById}
          onOpen={onOpenDirective}
        />
      </>
    );
  }
  return null;
}

function canReplaceWith(candidate: CreatorDirectiveItem, directive: CreatorDirectiveItem): boolean {
  return candidate.status === "active" && candidate.id !== directive.id;
}

function defaultReplacementId(
  directives: readonly CreatorDirectiveItem[],
  directive: CreatorDirectiveItem,
): string {
  return directives.find((candidate) => canReplaceWith(candidate, directive))?.id ?? "";
}

function replacementLabel(candidate: CreatorDirectiveItem): string {
  return `${shortId(candidate.id)} · ${candidate.kind} · p:${candidate.priority}`;
}

function canSubmitModal(
  modal: DirectiveModal | null,
  directivesById: ReadonlyMap<string, CreatorDirectiveItem>,
): boolean {
  if (modal === null) {
    return false;
  }

  if (modal.kind === "revoke") {
    return modal.reason.trim().length > 0;
  }

  const replacement = directivesById.get(modal.replacementId);
  return replacement !== undefined && canReplaceWith(replacement, modal.directive);
}

function sharedStateRelationsForDirective(input: {
  directive: CreatorDirectiveItem;
  entry: SharedStateEntry;
  commitmentsById: ReadonlyMap<string, CommitmentItem>;
}): SharedStateRelation[] {
  const directiveSourceIds = new Set(directiveSourceStreamIds(input.directive));

  if (directiveSourceIds.size === 0) {
    return [];
  }

  const relations: SharedStateRelation[] = [];
  const sharedSourceIntersection = intersectStrings(
    [...input.entry.provenance_stream_entry_ids, ...input.entry.last_updated_stream_entry_ids],
    directiveSourceIds,
  );

  if (sharedSourceIntersection.length > 0) {
    relations.push({
      kind: "shared_source",
      streamIds: sharedSourceIntersection,
    });
  }

  for (const commitmentId of input.entry.canonicalizes.commitment_ids) {
    const commitment = input.commitmentsById.get(commitmentId);
    if (commitment === undefined) {
      continue;
    }

    const commitmentSourceIntersection = intersectStrings(
      commitment.source_stream_entry_ids,
      directiveSourceIds,
    );

    if (commitmentSourceIntersection.length > 0) {
      relations.push({
        kind: "canonicalized_commitment",
        commitment,
        streamIds: commitmentSourceIntersection,
      });
    }
  }

  return relations;
}

function relatedSharedStateEntries(input: {
  directive: CreatorDirectiveItem;
  sharedAudiences: readonly SharedStateAudienceEntries[];
  commitmentsById: ReadonlyMap<string, CommitmentItem>;
}): RelatedSharedStateEntry[] {
  const rows: RelatedSharedStateEntry[] = [];

  for (const audience of input.sharedAudiences) {
    for (const entry of audience.entries) {
      const relations = sharedStateRelationsForDirective({
        directive: input.directive,
        entry,
        commitmentsById: input.commitmentsById,
      });

      if (relations.length > 0) {
        rows.push({
          audience: audience.audience,
          entry,
          relations,
        });
      }
    }
  }

  return rows;
}

function uncorrelatedSharedStateEntries(input: {
  directives: readonly CreatorDirectiveItem[];
  sharedAudiences: readonly SharedStateAudienceEntries[];
  commitmentsById: ReadonlyMap<string, CommitmentItem>;
}): RelatedSharedStateEntry[] {
  const rows: RelatedSharedStateEntry[] = [];

  for (const audience of input.sharedAudiences) {
    for (const entry of audience.entries) {
      const relatesToAnyDirective = input.directives.some(
        (directive) =>
          sharedStateRelationsForDirective({
            directive,
            entry,
            commitmentsById: input.commitmentsById,
          }).length > 0,
      );

      if (!relatesToAnyDirective) {
        rows.push({
          audience: audience.audience,
          entry,
          relations: [],
        });
      }
    }
  }

  return rows;
}

function commitmentStatusLabel(commitment: CommitmentItem | undefined): string {
  if (commitment === undefined) {
    return "unknown";
  }
  return commitment.superseded_by_id === null ? commitment.state : "superseded";
}

function canonicalTargetRows(
  entry: SharedStateEntry,
  commitmentsById: ReadonlyMap<string, CommitmentItem>,
): Array<{ channel: string; id: string; status: string; hint?: unknown }> {
  return [
    ...entry.canonicalizes.goal_ids.map((id) => ({
      channel: "goal",
      id,
      status: "status unavailable",
    })),
    ...entry.canonicalizes.commitment_ids.map((id) => ({
      channel: "commitment",
      id,
      status: commitmentStatusLabel(commitmentsById.get(id)),
      hint: commitmentsById.get(id),
    })),
    ...entry.canonicalizes.action_ids.map((id) => ({
      channel: "action",
      id,
      status: "status unavailable",
    })),
    ...entry.canonicalizes.open_question_ids.map((id) => ({
      channel: "open question",
      id,
      status: "status unavailable",
    })),
  ];
}

function relationLabel(relation: SharedStateRelation): string {
  if (relation.kind === "canonicalized_commitment") {
    return `related via canonicalized commitment ${shortId(relation.commitment.id)} (${commitmentStatusLabel(
      relation.commitment,
    )}) source ${relation.streamIds.map(shortId).join(", ")}`;
  }

  return `related via shared source ${relation.streamIds.map(shortId).join(", ")}`;
}

function StructuredRelationLabel({ relation }: { relation: SharedStateRelation }) {
  if (relation.kind === "canonicalized_commitment") {
    return (
      <>
        related via canonicalized commitment{" "}
        <IdRef
          id={relation.commitment.id}
          type="commitment"
          label={shortId(relation.commitment.id)}
          hint={relation.commitment}
        />{" "}
        ({commitmentStatusLabel(relation.commitment)}) source{" "}
        <InlineIdRefList ids={relation.streamIds} type="stream_entry" />
      </>
    );
  }

  return (
    <>
      related via shared source <InlineIdRefList ids={relation.streamIds} type="stream_entry" />
    </>
  );
}

function canonicalTargetInspectType(id: string): ObjectType | undefined {
  const type = resolveObjectType(id);
  return type === "goal" ||
    type === "commitment" ||
    type === "action_record" ||
    type === "open_question"
    ? type
    : undefined;
}

function canonicalizedCommitmentWarningState(
  relations: readonly SharedStateRelation[],
): "revoked" | "superseded" | null {
  for (const relation of relations) {
    if (relation.kind !== "canonicalized_commitment") {
      continue;
    }

    const state = commitmentStatusLabel(relation.commitment);
    if (state === "revoked" || state === "superseded") {
      return state;
    }
  }

  return null;
}

function SharedStateLifecyclePanel({
  directive,
  allDirectives,
  commitmentsById,
  sharedAudiences,
  audienceDiscoveryTruncated,
  loading,
}: {
  directive: CreatorDirectiveItem;
  allDirectives: readonly CreatorDirectiveItem[];
  commitmentsById: ReadonlyMap<string, CommitmentItem>;
  sharedAudiences: readonly SharedStateAudienceEntries[];
  audienceDiscoveryTruncated: boolean;
  loading: boolean;
}) {
  const [focusedSharedEntryId, setFocusedSharedEntryId] = useState<string | null>(null);
  const [focusedCanonicalTargetId, setFocusedCanonicalTargetId] = useState<string | null>(null);
  const relatedRows = relatedSharedStateEntries({
    directive,
    sharedAudiences,
    commitmentsById,
  });
  const uncorrelatedRows = uncorrelatedSharedStateEntries({
    directives: allDirectives,
    sharedAudiences,
    commitmentsById,
  });
  const emptyAudiences = sharedAudiences
    .filter((audience) => audience.entries.length === 0)
    .map((audience) => audience.audience);
  const totalEntryCount = sharedAudiences.reduce(
    (sum, audience) => sum + audience.entries.length,
    0,
  );

  return (
    <>
      <div className="divider">shared-state lifecycle</div>
      {loading ? <div className="notice">loading shared-state lifecycle</div> : null}
      {!loading && audienceDiscoveryTruncated ? (
        <div
          className="notice"
          style={{ fontSize: "var(--fs-sm)", lineHeight: 1.55, marginBottom: 8 }}
        >
          audience discovery reached the 1000-session server cap; shared-state lifecycle rows from
          older audiences may be missing
        </div>
      ) : null}
      {!loading && sharedAudiences.length === 0 ? (
        <div className="dim" style={{ fontSize: "var(--fs-sm)", lineHeight: 1.55 }}>
          no shared-state audiences returned for lookup
        </div>
      ) : null}
      {!loading && emptyAudiences.length > 0 ? (
        <div
          className="dim"
          style={{ fontSize: "var(--fs-sm)", lineHeight: 1.55, marginBottom: 8 }}
        >
          empty shared-state audiences: {emptyAudiences.join(", ")}
        </div>
      ) : null}
      {!loading && totalEntryCount === 0 && sharedAudiences.length > 0 ? (
        <div className="dim" style={{ fontSize: "var(--fs-sm)", lineHeight: 1.55 }}>
          no shared-state rows across discovered audiences
        </div>
      ) : null}
      {!loading && totalEntryCount > 0 && relatedRows.length === 0 ? (
        <div
          className="dim"
          style={{ fontSize: "var(--fs-sm)", lineHeight: 1.55, marginBottom: 8 }}
        >
          no shared-state rows structurally relate to this directive's source stream ids
        </div>
      ) : null}
      {relatedRows.map((row) => (
        <SharedStateLifecycleRow
          key={`related:${row.audience}:${row.entry.id}`}
          row={row}
          directive={directive}
          commitmentsById={commitmentsById}
          focusedSharedEntryId={focusedSharedEntryId}
          focusedCanonicalTargetId={focusedCanonicalTargetId}
          onOpenSharedEntry={setFocusedSharedEntryId}
          onOpenCanonicalTarget={setFocusedCanonicalTargetId}
        />
      ))}

      <div className="divider">uncorrelated shared lifecycle</div>
      {!loading && totalEntryCount === 0 ? (
        <div className="dim" style={{ fontSize: "var(--fs-sm)", lineHeight: 1.55 }}>
          no uncorrelated rows because no shared-state rows were returned
        </div>
      ) : null}
      {!loading && totalEntryCount > 0 && uncorrelatedRows.length === 0 ? (
        <div className="dim" style={{ fontSize: "var(--fs-sm)", lineHeight: 1.55 }}>
          all non-empty shared-state rows have at least one structural directive relation
        </div>
      ) : null}
      {uncorrelatedRows.map((row) => (
        <SharedStateLifecycleRow
          key={`uncorrelated:${row.audience}:${row.entry.id}`}
          row={row}
          directive={directive}
          commitmentsById={commitmentsById}
          focusedSharedEntryId={focusedSharedEntryId}
          focusedCanonicalTargetId={focusedCanonicalTargetId}
          onOpenSharedEntry={setFocusedSharedEntryId}
          onOpenCanonicalTarget={setFocusedCanonicalTargetId}
        />
      ))}
    </>
  );
}

function SharedStateLifecycleRow({
  row,
  directive,
  commitmentsById,
  focusedSharedEntryId,
  focusedCanonicalTargetId,
  onOpenSharedEntry,
  onOpenCanonicalTarget,
}: {
  row: RelatedSharedStateEntry;
  directive: CreatorDirectiveItem;
  commitmentsById: ReadonlyMap<string, CommitmentItem>;
  focusedSharedEntryId: string | null;
  focusedCanonicalTargetId: string | null;
  onOpenSharedEntry: (id: string) => void;
  onOpenCanonicalTarget: (id: string) => void;
}) {
  const entry = row.entry;
  const canonicalTargets = canonicalTargetRows(entry, commitmentsById);
  const canonicalCommitmentWarningState =
    directive.status === "active" ? canonicalizedCommitmentWarningState(row.relations) : null;

  return (
    <div
      style={{
        border: `1px solid ${focusedSharedEntryId === entry.id ? "var(--acc-dim)" : "var(--line)"}`,
        background: "var(--bg-1)",
        padding: "10px 12px",
        marginBottom: 8,
      }}
    >
      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
        <Tag kind={tagKind(entry.kind)} dot>
          {lifecycleLabel(entry.kind)}
        </Tag>
        <IdRef id={entry.id} type="shared_state_entry" label={shortId(entry.id)} hint={entry} />
        <span className="dim" style={{ fontSize: "var(--fs-xs)" }}>
          audience {row.audience}
        </span>
      </div>
      <div className="props" style={{ marginTop: 8 }}>
        <div className="row">
          <span className="k">state key</span>
          <span className="v">{entry.state_key ?? "legacy/unkeyed"}</span>
        </div>
        <div className="row">
          <span className="k">updated</span>
          <span className="v">{dateLabel(entry.last_updated_at)}</span>
        </div>
        <div className="row">
          <span className="k">sources</span>
          <span className="v tab-num">
            {entry.provenance_stream_entry_ids.length} provenance /{" "}
            {entry.last_updated_stream_entry_ids.length} update
          </span>
        </div>
        {entry.superseded_by_id === null ? null : (
          <div className="row">
            <span className="k">superseded by</span>
            <span className="v">
              <SupersededByChip
                id={entry.superseded_by_id}
                label={shortId(entry.superseded_by_id)}
                active={focusedSharedEntryId === entry.superseded_by_id}
                title={`Focus shared-state entry ${entry.superseded_by_id}`}
                ariaLabel={`focus shared-state entry ${entry.superseded_by_id}`}
                onOpen={onOpenSharedEntry}
                inspectType="shared_state_entry"
                inspectAriaLabel={`inspect shared-state entry ${entry.superseded_by_id}`}
              />
            </span>
          </div>
        )}
      </div>
      <div
        style={{
          fontFamily: "var(--sans)",
          fontSize: 12,
          lineHeight: 1.5,
          color: "var(--text)",
          marginTop: 8,
          overflowWrap: "anywhere",
        }}
      >
        {entry.text}
      </div>
      {row.relations.length === 0 ? (
        <div className="dim" style={{ fontSize: "var(--fs-xs)", marginTop: 8 }}>
          no structural relation to any creator directive source stream id
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 8 }}>
          {row.relations.map((relation) => (
            <div
              key={`${relation.kind}:${relation.streamIds.join(",")}:${
                relation.kind === "canonicalized_commitment" ? relation.commitment.id : "source"
              }`}
              className="dim"
              title={relationLabel(relation)}
              style={{ fontSize: "var(--fs-xs)", lineHeight: 1.45 }}
            >
              <StructuredRelationLabel relation={relation} />
            </div>
          ))}
        </div>
      )}
      {canonicalTargets.length === 0 ? null : (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
          {canonicalTargets.map((target) => {
            const inspectType = canonicalTargetInspectType(target.id);
            return (
              <SupersededByChip
                key={`${target.channel}:${target.id}`}
                id={target.id}
                label={`${target.channel} ${shortId(target.id)} · ${target.status}`}
                active={focusedCanonicalTargetId === target.id}
                title={`Focus canonical target ${target.id}`}
                ariaLabel={`focus canonical target ${target.id}`}
                onOpen={onOpenCanonicalTarget}
                inspectType={inspectType}
                inspectHint={target.hint}
                inspectAriaLabel={
                  inspectType === undefined ? undefined : `inspect canonical target ${target.id}`
                }
              />
            );
          })}
        </div>
      )}
      {canonicalCommitmentWarningState === null ? null : (
        <div className="warn" style={{ fontSize: "var(--fs-xs)", marginTop: 8, lineHeight: 1.45 }}>
          canonicalized commitment is {canonicalCommitmentWarningState} while selected directive is
          active
        </div>
      )}
    </div>
  );
}

export type DirectivesTabProps = {
  sessionId?: string;
  embedded?: boolean;
};

export function DirectivesTab({ sessionId = "default", embedded = false }: DirectivesTabProps) {
  const api = useApi(() => getCreatorDirectives({ status: "all" }), []);
  const supportApi = useApi(() => loadDirectiveSupportData(sessionId), [sessionId]);
  return <DirectivesPanel api={api} supportApi={supportApi} embedded={embedded} />;
}

export function DirectivesPanel({
  api,
  supportApi,
  embedded = false,
}: {
  api: ApiHookState<CreatorDirectivesResponse>;
  supportApi: ApiHookState<DirectiveSupportData>;
  embedded?: boolean;
}) {
  const [statusFilter, setStatusFilter] = useState<CreatorDirectiveStatusFilter>("active");
  const [sortMode, setSortMode] = useState<SortMode>("priority_desc");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [modal, setModal] = useState<DirectiveModal | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const rawDirectives = api.data?.directives ?? [];
  const visibleRawDirectives = useMemo(
    () =>
      statusFilter === "all"
        ? rawDirectives
        : rawDirectives.filter((directive) => directive.status === statusFilter),
    [rawDirectives, statusFilter],
  );
  const directives = useMemo(
    () => [...visibleRawDirectives].sort(compareDirectives(sortMode)),
    [visibleRawDirectives, sortMode],
  );
  const directivesById = useMemo(
    () => new Map(rawDirectives.map((directive) => [directive.id, directive])),
    [rawDirectives],
  );
  const commitmentsById = useMemo(
    () => new Map((supportApi.data?.commitments ?? []).map((item) => [item.id, item])),
    [supportApi.data?.commitments],
  );
  const selected = directives.find((item) => item.id === selectedId) ?? directives[0] ?? null;

  function openRevoke(directive: CreatorDirectiveItem): void {
    setModal({ kind: "revoke", directive, reason: "" });
  }

  function openSupersede(directive: CreatorDirectiveItem): void {
    setModal({
      kind: "supersede",
      directive,
      replacementId: defaultReplacementId(directives, directive),
    });
  }

  function openDirectiveReference(id: string): void {
    setStatusFilter("all");
    setSelectedId(id);
  }

  async function runAction(label: string, action: () => Promise<void>): Promise<void> {
    setBusy(label);
    setOperatorError(null);
    try {
      await action();
      await api.refetch();
      setModal(null);
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  async function submitModal(): Promise<void> {
    if (modal === null) {
      return;
    }

    if (modal.kind === "revoke") {
      await runAction("creator-directive-revoke", async () => {
        await revokeCreatorDirective(modal.directive.id, modal.reason.trim());
      });
      return;
    }

    await runAction("creator-directive-supersede", async () => {
      await supersedeCreatorDirective(modal.directive.id, modal.replacementId);
    });
  }

  if (api.loading && api.data === null) {
    return <div className="notice">loading creator directives</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  return (
    <div className={embedded ? "governance-panel" : "full-page"}>
      <div className="page-head">
        {embedded ? null : (
          <>
            <h1>creator directives</h1>
            <span className="desc">identity · subject facts · disclosure · response policies</span>
          </>
        )}
        <span className="spacer"></span>
        <div className="filter-pills">
          {STATUS_FILTERS.map((value) => (
            <button
              key={value}
              type="button"
              className={`pill ${statusFilter === value ? "on" : ""}`}
              onClick={() => setStatusFilter(value)}
            >
              {value}
            </button>
          ))}
        </div>
        <span className="sep">|</span>
        <div className="filter-pills">
          {SORT_MODES.map((value) => (
            <button
              key={value}
              type="button"
              className={`pill ${sortMode === value ? "on" : ""}`}
              onClick={() => setSortMode(value)}
            >
              {sortLabel(value)}
            </button>
          ))}
        </div>
      </div>

      {operatorError === null ? null : (
        <div className="notice bad" style={{ padding: 12 }}>
          {operatorError}
        </div>
      )}
      {supportApi.error === null ? null : (
        <div className="notice bad" style={{ padding: 12 }}>
          shared-state lifecycle unavailable: {supportApi.error.message}
        </div>
      )}

      <div
        className="page-body"
        style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 320px" }}
      >
        <div style={{ overflow: "auto", borderRight: "1px solid var(--line)" }}>
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 92 }}>id</th>
                <th style={{ width: 150 }}>kind</th>
                <th style={{ minWidth: 260 }}>text</th>
                <th style={{ width: 132 }}>scope</th>
                <th style={{ width: 126 }}>content</th>
                <th style={{ width: 142 }}>mention</th>
                <th style={{ width: 84 }}>status</th>
                <th style={{ width: 50, textAlign: "right" }}>p</th>
                <th style={{ width: 100 }}>since</th>
                <th style={{ width: 150 }}></th>
              </tr>
            </thead>
            <tbody>
              {directives.map((item) => {
                const inactive = inactiveSummary(item, directivesById, openDirectiveReference);
                return (
                  <tr
                    key={item.id}
                    onClick={(event) => {
                      if (!isInteractiveDescendantEvent(event.currentTarget, event.target)) {
                        setSelectedId(item.id);
                      }
                    }}
                    className={item.id === selected?.id ? "selected" : ""}
                    style={{ cursor: "pointer" }}
                  >
                    <td>
                      <button
                        type="button"
                        className="row-select-button"
                        aria-pressed={item.id === selected?.id}
                        aria-label={`select creator directive ${item.id}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelectedId(item.id);
                        }}
                      >
                        {shortId(item.id)}
                      </button>
                    </td>
                    <td>
                      <Tag>{item.kind}</Tag>
                    </td>
                    <td
                      className="wrap"
                      style={{ fontFamily: "var(--sans)", fontSize: "12px", lineHeight: 1.45 }}
                    >
                      {emptyLabel(item.text)}
                      <div className="dim" style={{ fontSize: "var(--fs-xs)", marginTop: 2 }}>
                        subject:{item.subject_entity_name ?? item.subject_kind}
                      </div>
                      {inactive === null ? null : (
                        <div className="dim" style={{ fontSize: "var(--fs-xs)", marginTop: 2 }}>
                          {inactive}
                        </div>
                      )}
                    </td>
                    <td>{item.activation_scope}</td>
                    <td>{item.content_scope}</td>
                    <td>{item.mention_policy}</td>
                    <td>
                      <Tag kind={statusTag(item.status)} dot>
                        {item.status}
                      </Tag>
                    </td>
                    <td
                      className="tab-num"
                      style={{
                        textAlign: "right",
                        color: item.priority >= 80 ? "var(--warn)" : "var(--text-dim)",
                      }}
                    >
                      {item.priority}
                    </td>
                    <td className="dim" style={{ fontSize: "var(--fs-xs)" }}>
                      {dateLabel(item.created_at)}
                    </td>
                    <td>
                      {item.status === "active" ? (
                        <div style={{ display: "flex", gap: 6 }}>
                          <button
                            className="btn sm danger"
                            disabled={busy !== null}
                            onClick={(event) => {
                              event.stopPropagation();
                              openRevoke(item);
                            }}
                          >
                            revoke
                          </button>
                          <button
                            className="btn sm danger"
                            disabled={
                              busy !== null ||
                              directives.every((candidate) => !canReplaceWith(candidate, item))
                            }
                            onClick={(event) => {
                              event.stopPropagation();
                              openSupersede(item);
                            }}
                          >
                            supersede
                          </button>
                        </div>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ overflowY: "auto", background: "var(--bg-0)" }}>
          {selected === null ? (
            <div className="notice">
              {rawDirectives.length === 0 ? "no creator directives yet" : "none match this filter"}
            </div>
          ) : (
            <CreatorDirectiveDetail
              directive={selected}
              directives={directives}
              allDirectives={rawDirectives}
              directivesById={directivesById}
              commitmentsById={commitmentsById}
              sharedAudiences={supportApi.data?.sharedAudiences ?? []}
              audienceDiscoveryTruncated={supportApi.data?.audienceDiscoveryTruncated ?? false}
              sharedLifecycleLoading={supportApi.loading && supportApi.data === null}
              busy={busy !== null}
              onOpenDirective={openDirectiveReference}
              onRevoke={openRevoke}
              onSupersede={openSupersede}
            />
          )}
        </div>
      </div>

      <Modal
        open={modal !== null}
        title={
          modal?.kind === "revoke" ? "revoke creator directive" : "supersede creator directive"
        }
        onClose={() => {
          if (busy === null) {
            setModal(null);
          }
        }}
        footer={
          <>
            <span className="dim" style={{ fontSize: "var(--fs-xs)", marginRight: "auto" }}>
              {modal?.kind === "revoke" ? "reason required" : ""}
            </span>
            <button
              className="btn sm ghost"
              disabled={busy !== null}
              onClick={() => setModal(null)}
            >
              cancel
            </button>
            <button
              className="btn sm danger"
              disabled={busy !== null || !canSubmitModal(modal, directivesById)}
              onClick={() => void submitModal()}
            >
              {busy === null ? (modal?.kind === "revoke" ? "revoke" : "supersede") : "saving"}
            </button>
          </>
        }
      >
        {modal?.kind === "revoke" ? (
          <div className="modal-form">
            <div className="dim">{emptyLabel(modal.directive.text)}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                required
                maxLength={2000}
                value={modal.reason}
                onChange={(event) => setModal({ ...modal, reason: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "supersede" ? (
          <div className="modal-form">
            <div className="dim">{emptyLabel(modal.directive.text)}</div>
            <label className="modal-field">
              <span>replacement</span>
              <select
                value={modal.replacementId}
                onChange={(event) => setModal({ ...modal, replacementId: event.target.value })}
              >
                <option value="" disabled>
                  select replacement
                </option>
                {directives.map((candidate) => (
                  <option
                    key={candidate.id}
                    value={candidate.id}
                    disabled={!canReplaceWith(candidate, modal.directive)}
                  >
                    {replacementLabel(candidate)}
                    {candidate.id === modal.directive.id ? " · current" : ""}
                    {candidate.status === "active" ? "" : ` · ${candidate.status}`}
                  </option>
                ))}
              </select>
            </label>
            {directives.some((candidate) => canReplaceWith(candidate, modal.directive)) ? null : (
              <div className="warn">no other active directive is available as a replacement</div>
            )}
          </div>
        ) : null}
      </Modal>
    </div>
  );
}

function CreatorDirectiveDetail({
  directive,
  directives,
  allDirectives,
  directivesById,
  commitmentsById,
  sharedAudiences,
  audienceDiscoveryTruncated,
  sharedLifecycleLoading,
  busy,
  onOpenDirective,
  onRevoke,
  onSupersede,
}: {
  directive: CreatorDirectiveItem;
  directives: readonly CreatorDirectiveItem[];
  allDirectives: readonly CreatorDirectiveItem[];
  directivesById: ReadonlyMap<string, CreatorDirectiveItem>;
  commitmentsById: ReadonlyMap<string, CommitmentItem>;
  sharedAudiences: readonly SharedStateAudienceEntries[];
  audienceDiscoveryTruncated: boolean;
  sharedLifecycleLoading: boolean;
  busy: boolean;
  onOpenDirective: (id: string) => void;
  onRevoke: (directive: CreatorDirectiveItem) => void;
  onSupersede: (directive: CreatorDirectiveItem) => void;
}) {
  const hasReplacement = directives.some((candidate) => canReplaceWith(candidate, directive));
  return (
    <>
      <div style={{ padding: "16px 16px 10px 16px", borderBottom: "1px solid var(--line)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--text-mute)" }}>creator directive</div>
        <div
          style={{
            fontSize: 14,
            color: "var(--text)",
            fontFamily: "var(--sans)",
            margin: "6px 0 10px 0",
            overflowWrap: "anywhere",
          }}
        >
          {emptyLabel(directive.text)}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Tag kind={statusTag(directive.status)} dot>
            {directive.status}
          </Tag>
          <Tag>{directive.kind}</Tag>
          <Tag>{directive.activation_scope}</Tag>
        </div>
      </div>
      <div style={{ padding: 16 }}>
        <div className="divider">identity</div>
        <div className="props">
          <div className="row">
            <span className="k">id</span>
            <span className="v acc">
              <IdChip id={directive.id} type="creator_directive" hint={directive} />
            </span>
          </div>
          <div className="row">
            <span className="k">subject</span>
            <span className="v">{directive.subject_entity_name ?? directive.subject_kind}</span>
          </div>
          <div className="row">
            <span className="k">subject id</span>
            <span className="v">
              {directive.subject_entity_id === null ? (
                "—"
              ) : (
                <IdChip id={directive.subject_entity_id} type="entity" />
              )}
            </span>
          </div>
          <div className="row">
            <span className="k">content scope</span>
            <span className="v">{directive.content_scope}</span>
          </div>
          <div className="row">
            <span className="k">mention policy</span>
            <span className="v">{directive.mention_policy}</span>
          </div>
          <div className="row">
            <span className="k">priority</span>
            <span className="v tab-num">{directive.priority}</span>
          </div>
          <div className="row">
            <span className="k">created</span>
            <span className="v">{dateLabel(directive.created_at)}</span>
          </div>
          <div className="row">
            <span className="k">updated</span>
            <span className="v">{dateLabel(directive.updated_at)}</span>
          </div>
          {directive.revoked_reason === null ? null : (
            <div className="row">
              <span className="k">revoked reason</span>
              <span className="v">{directive.revoked_reason}</span>
            </div>
          )}
          {directive.superseded_by_id === null ? null : (
            <div className="row">
              <span className="k">superseded by</span>
              <span className="v">
                <CreatorDirectiveSupersededByChip
                  id={directive.superseded_by_id}
                  directivesById={directivesById}
                  onOpen={onOpenDirective}
                  inspect
                />
              </span>
            </div>
          )}
        </div>

        <div className="divider">sources</div>
        <div className="props">
          <div className="row">
            <span className="k">authorization</span>
            <span className="v">
              <InlineIdRefList ids={directive.authorization_stream_entry_ids} type="stream_entry" />
            </span>
          </div>
          <div className="row">
            <span className="k">content source</span>
            <span className="v">
              <InlineIdRefList
                ids={directive.content_source_stream_entry_ids}
                type="stream_entry"
              />
            </span>
          </div>
        </div>

        <div className="divider">activation</div>
        <div className="props">
          <div className="row">
            <span className="k">scope</span>
            <span className="v">{directive.activation_scope}</span>
          </div>
          <div className="row">
            <span className="k">allowed ids</span>
            <span className="v">
              <InlineIdRefList ids={directive.activation_allowed_entity_ids} type="entity" />
            </span>
          </div>
          <div className="row">
            <span className="k">excluded ids</span>
            <span className="v">
              <InlineIdRefList ids={directive.activation_excluded_entity_ids} type="entity" />
            </span>
          </div>
        </div>

        <div className="divider">content</div>
        <div className="props">
          <div className="row">
            <span className="k">canonical fact</span>
            <span className="v">{directive.canonical_fact ?? "—"}</span>
          </div>
          <div className="row">
            <span className="k">operational directive</span>
            <span className="v">{directive.operational_directive ?? "—"}</span>
          </div>
        </div>

        <SharedStateLifecyclePanel
          directive={directive}
          allDirectives={allDirectives}
          commitmentsById={commitmentsById}
          sharedAudiences={sharedAudiences}
          audienceDiscoveryTruncated={audienceDiscoveryTruncated}
          loading={sharedLifecycleLoading}
        />

        <div className="divider">operations</div>
        {directive.status === "active" ? (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            <button className="btn sm danger" disabled={busy} onClick={() => onRevoke(directive)}>
              revoke
            </button>
            <button
              className="btn sm danger"
              disabled={busy || !hasReplacement}
              onClick={() => onSupersede(directive)}
            >
              supersede
            </button>
          </div>
        ) : (
          <div className="dim" style={{ fontSize: "var(--fs-sm)" }}>
            no active operations for {directive.status} directives
          </div>
        )}
      </div>
    </>
  );
}
