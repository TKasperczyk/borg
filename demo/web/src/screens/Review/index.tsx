import { useEffect, useMemo, useRef, useState } from "react";

import {
  getCreatorDirectives,
  getReviews,
  patchReview,
  resolveCreatorDirectiveReconciliation,
} from "../../api/client";
import type {
  CreatorDirectiveItem,
  ReviewKind,
  ReviewResolution,
  ReviewRow,
} from "../../api/types";
import { Tag } from "../../components/Tag";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { formatTime } from "../../lib/stream-utils";
import { shortId } from "../screen-utils";

const REVIEW_KIND_ORDER: ReviewKind[] = [
  "creator_directive_reconciliation",
  "correction",
  "belief_revision",
  "contradiction",
  "duplicate",
  "new_insight",
  "misattribution",
  "temporal_drift",
  "identity_inconsistency",
  "skill_split",
  "relationship_claim_ungrounded",
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
  relationship_claim_ungrounded: ["accept", "reject", "dismiss", "keep"],
  creator_directive_reconciliation: [],
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "-";
  }

  if (typeof value === "string") {
    return value.length === 0 ? "-" : value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value.length === 0 ? "-" : value.map(displayValue).join(", ");
  }

  return JSON.stringify(value, null, 2);
}

function recordValue(record: Record<string, unknown>, key: string): unknown {
  return Object.hasOwn(record, key) ? record[key] : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function nodeIds(row: ReviewRow): string[] {
  return stringArray(recordValue(row.refs, "node_ids"));
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

function detailFields(row: ReviewRow): Array<[string, unknown]> {
  const keysByKind: Partial<Record<ReviewKind, string[]>> = {
    contradiction: ["node_ids", "node_labels", "edge_id", "vector_similarity", "source_overlap"],
    duplicate: [
      "node_ids",
      "node_labels",
      "duplicate_subtype",
      "vector_similarity",
      "source_overlap",
    ],
    correction: ["target_type", "target_id", "prompt_summary", "operator_reason", "patch"],
    belief_revision: [
      "target_type",
      "target_id",
      "invalidated_edge_id",
      "dependency_path_edge_ids",
      "surviving_support_edge_ids",
      "evidence_episode_ids",
    ],
    new_insight: ["target_type", "semantic_node", "candidate", "source_episode_ids"],
    misattribution: ["target_type", "target_id", "patch", "source_stream_entry_ids"],
    temporal_drift: [
      "target_type",
      "target_id",
      "corrected_start_time",
      "corrected_end_time",
      "patch_description",
    ],
    identity_inconsistency: ["target_type", "target_id", "patch", "repair"],
    skill_split: ["skill_id", "claimed_at", "proposal", "splits"],
    relationship_claim_ungrounded: [
      "target_type",
      "label",
      "description",
      "relationship_claim_label_families",
      "ungrounded_relationship_claims",
    ],
  };
  const preferredKeys = keysByKind[row.kind] ?? [];
  const preferred = preferredKeys
    .filter((key) => recordValue(row.refs, key) !== undefined)
    .map((key): [string, unknown] => [key, recordValue(row.refs, key)]);
  const preferredSet = new Set(preferredKeys);
  const extras = Object.entries(row.refs)
    .filter(([key]) => !preferredSet.has(key))
    .filter(([key]) => !key.startsWith(REVIEW_RESOLVER_REF_PREFIX))
    .slice(0, 8);

  return [...preferred, ...extras];
}

function firstString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
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
            {ids.map((id) => (
              <option value={id} key={id}>
                {id}
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
  revokeIds,
  revokeReason,
  onSurvivor,
  onToggleRevoke,
  onRevokeReason,
  onSupersede,
  onRevoke,
  onKeep,
}: {
  row: ReviewRow;
  directivesById: Map<string, CreatorDirectiveItem>;
  busy: BusyState;
  survivor: string;
  revokeIds: string[];
  revokeReason: string;
  onSurvivor: (value: string) => void;
  onToggleRevoke: (value: string) => void;
  onRevokeReason: (value: string) => void;
  onSupersede: () => void;
  onRevoke: () => void;
  onKeep: () => void;
}) {
  const refs = row.refs;
  const members = Array.isArray(refs.members) ? refs.members : [];
  const ids = directiveIds(row);
  const judgment = isRecord(refs.judgment) ? refs.judgment : {};
  const differences = scopeDifferences(members);
  const selectedRevokeIds = new Set(revokeIds);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="props">
        <div className="row">
          <span className="k">subkind</span>
          <span className="v">{displayValue(refs.subkind)}</span>
        </div>
        <div className="row">
          <span className="k">verdict</span>
          <span className="v">
            {displayValue(judgment.verdict)} / {displayValue(judgment.confidence)}
          </span>
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
          const directive = directivesById.get(id);
          return (
            <DirectiveMemberCard
              key={id}
              id={id}
              member={member}
              directive={directive}
              differences={differences}
              selectedForRevoke={selectedRevokeIds.has(id)}
              onToggleRevoke={() => onToggleRevoke(id)}
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
            {ids.map((id) => {
              const directive = directivesById.get(id);
              return (
                <option value={id} key={id}>
                  {shortId(id)} {directive?.content_scope ?? ""}
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

      <div style={{ display: "grid", gap: 8 }}>
        <label className="modal-field">
          <span>revoke reason</span>
          <textarea value={revokeReason} onChange={(event) => onRevokeReason(event.target.value)} />
        </label>
        <div className="operator-actions">
          <button
            type="button"
            className="btn sm ghost"
            disabled={busy !== null || revokeIds.length === 0 || revokeReason.trim().length === 0}
            onClick={onRevoke}
          >
            {busy?.id === row.id && busy.label === "revoke" ? "saving" : "revoke selected"}
          </button>
        </div>
      </div>
    </div>
  );
}

function DirectiveMemberCard({
  id,
  member,
  directive,
  differences,
  selectedForRevoke,
  onToggleRevoke,
}: {
  id: string;
  member: unknown;
  directive?: CreatorDirectiveItem;
  differences: Set<ScopeField>;
  selectedForRevoke: boolean;
  onToggleRevoke: () => void;
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
        <span className="acc">{shortId(id)}</span>
        <label style={{ display: "inline-flex", gap: 5, alignItems: "center" }}>
          <input type="checkbox" checked={selectedForRevoke} onChange={onToggleRevoke} />
          <span className="dim" style={{ fontSize: 10.5 }}>
            revoke
          </span>
        </label>
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
              {displayValue(scopeFieldValue(member, field))}
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
  const [revokeSelections, setRevokeSelections] = useState<Record<number, string[]>>({});
  const [revokeReasons, setRevokeReasons] = useState<Record<number, string>>({});

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

  function toggleRevokeId(id: number, directiveId: string): void {
    setRevokeSelections((current) => {
      const selected = new Set(current[id] ?? []);
      if (selected.has(directiveId)) {
        selected.delete(directiveId);
      } else {
        selected.add(directiveId);
      }
      return { ...current, [id]: [...selected] };
    });
  }

  function setRevokeReason(id: number, reason: string): void {
    setRevokeReasons((current) => ({ ...current, [id]: reason }));
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

  async function submitReconciliationRevoke(row: ReviewRow): Promise<void> {
    const revokeIds = revokeSelections[row.id] ?? [];
    const reason = revokeReasons[row.id]?.trim() ?? "";
    await runReviewAction(row.id, "revoke", async () => {
      await resolveCreatorDirectiveReconciliation(row.id, {
        action: "revoke",
        revoke_ids: revokeIds,
        reason,
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
                        </div>

                        {row.kind === "creator_directive_reconciliation" ? (
                          <ReconciliationReview
                            row={row}
                            directivesById={directivesById}
                            busy={busy}
                            survivor={survivor}
                            revokeIds={revokeSelections[row.id] ?? []}
                            revokeReason={revokeReasons[row.id] ?? ""}
                            onSurvivor={(value) => setSurvivor(row.id, value)}
                            onToggleRevoke={(value) => toggleRevokeId(row.id, value)}
                            onRevokeReason={(value) => setRevokeReason(row.id, value)}
                            onSupersede={() => void submitReconciliationSupersede(row)}
                            onRevoke={() => void submitReconciliationRevoke(row)}
                            onKeep={() => void submitReconciliationKeep(row)}
                          />
                        ) : (
                          <div
                            style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 14 }}
                          >
                            <ReviewDetail row={row} />
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
