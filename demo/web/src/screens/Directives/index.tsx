import { useMemo, useState, type ReactNode } from "react";

import {
  getCreatorDirectives,
  revokeCreatorDirective,
  supersedeCreatorDirective,
} from "../../api/client";
import type {
  CreatorDirectiveItem,
  CreatorDirectiveStatus,
  CreatorDirectiveStatusFilter,
} from "../../api/types";
import { Modal } from "../../components/Modal";
import { Tag } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { dateLabel, shortId } from "../screen-utils";

type SortMode = "priority_desc" | "priority_asc";

type DirectiveModal =
  | { kind: "revoke"; directive: CreatorDirectiveItem; reason: string }
  | { kind: "supersede"; directive: CreatorDirectiveItem; replacementId: string };

const STATUS_FILTERS: CreatorDirectiveStatusFilter[] = [
  "active",
  "revoked",
  "superseded",
  "all",
];
const SORT_MODES: SortMode[] = ["priority_desc", "priority_asc"];

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

function SupersededByChip({
  id,
  directivesById,
  onOpen,
}: {
  id: string;
  directivesById: ReadonlyMap<string, CreatorDirectiveItem>;
  onOpen: (id: string) => void;
}) {
  return (
    <button
      type="button"
      className="btn sm ghost"
      title={`Jump to directive ${id}`}
      aria-label={`jump to directive ${id}`}
      onClick={(event) => {
        event.stopPropagation();
        onOpen(id);
      }}
    >
      {supersededLabel(id, directivesById)}
    </button>
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
        <SupersededByChip
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

export function DirectivesScreen() {
  const [statusFilter, setStatusFilter] = useState<CreatorDirectiveStatusFilter>("active");
  const [sortMode, setSortMode] = useState<SortMode>("priority_desc");
  const api = useApi(() => getCreatorDirectives({ status: statusFilter }), [statusFilter]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [modal, setModal] = useState<DirectiveModal | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const rawDirectives = api.data?.directives ?? [];
  const directives = useMemo(
    () => [...rawDirectives].sort(compareDirectives(sortMode)),
    [rawDirectives, sortMode],
  );
  const directivesById = useMemo(
    () => new Map(directives.map((directive) => [directive.id, directive])),
    [directives],
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
    <div className="full-page">
      <div className="page-head">
        <h1>creator directives</h1>
        <span className="desc">identity · subject facts · disclosure · response policies</span>
        <span className="spacer"></span>
        <div className="filter-pills">
          {STATUS_FILTERS.map((value) => (
            <span
              key={value}
              className={`pill ${statusFilter === value ? "on" : ""}`}
              onClick={() => setStatusFilter(value)}
            >
              {value}
            </span>
          ))}
        </div>
        <span className="sep">|</span>
        <div className="filter-pills">
          {SORT_MODES.map((value) => (
            <span
              key={value}
              className={`pill ${sortMode === value ? "on" : ""}`}
              onClick={() => setSortMode(value)}
            >
              {sortLabel(value)}
            </span>
          ))}
        </div>
      </div>

      {operatorError === null ? null : (
        <div className="notice bad" style={{ padding: 12 }}>
          {operatorError}
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
                    onClick={() => setSelectedId(item.id)}
                    className={item.id === selected?.id ? "selected" : ""}
                    style={{ cursor: "pointer" }}
                  >
                    <td>
                      <span className="acc">{shortId(item.id)}</span>
                    </td>
                    <td>
                      <Tag>{item.kind}</Tag>
                    </td>
                    <td
                      className="wrap"
                      style={{ fontFamily: "var(--sans)", fontSize: "12px", lineHeight: 1.45 }}
                    >
                      {emptyLabel(item.text)}
                      <div className="dim" style={{ fontSize: 10, marginTop: 2 }}>
                        subject:{item.subject_entity_name ?? item.subject_kind}
                      </div>
                      {inactive === null ? null : (
                        <div className="dim" style={{ fontSize: 10, marginTop: 2 }}>
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
                        color: item.priority >= 80 ? "var(--bad)" : "var(--text-dim)",
                      }}
                    >
                      {item.priority}
                    </td>
                    <td className="dim" style={{ fontSize: 11 }}>
                      {dateLabel(item.created_at)}
                    </td>
                    <td>
                      {item.status === "active" ? (
                        <div style={{ display: "flex", gap: 6 }}>
                          <button
                            className="btn sm primary"
                            disabled={busy !== null}
                            onClick={(event) => {
                              event.stopPropagation();
                              openRevoke(item);
                            }}
                          >
                            revoke
                          </button>
                          <button
                            className="btn sm ghost"
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
            <div className="notice">no creator directives in filter</div>
          ) : (
            <CreatorDirectiveDetail
              directive={selected}
              directives={directives}
              directivesById={directivesById}
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
          modal?.kind === "revoke"
            ? "revoke creator directive"
            : "supersede creator directive"
        }
        onClose={() => {
          if (busy === null) {
            setModal(null);
          }
        }}
        footer={
          <>
            <span className="dim" style={{ fontSize: 10.5, marginRight: "auto" }}>
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
              className="btn sm primary"
              disabled={busy !== null || !canSubmitModal(modal, directivesById)}
              onClick={() => void submitModal()}
            >
              {busy === null
                ? modal?.kind === "revoke"
                  ? "revoke"
                  : "supersede"
                : "saving"}
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
  directivesById,
  busy,
  onOpenDirective,
  onRevoke,
  onSupersede,
}: {
  directive: CreatorDirectiveItem;
  directives: readonly CreatorDirectiveItem[];
  directivesById: ReadonlyMap<string, CreatorDirectiveItem>;
  busy: boolean;
  onOpenDirective: (id: string) => void;
  onRevoke: (directive: CreatorDirectiveItem) => void;
  onSupersede: (directive: CreatorDirectiveItem) => void;
}) {
  const hasReplacement = directives.some((candidate) => canReplaceWith(candidate, directive));
  return (
    <>
      <div style={{ padding: "16px 16px 10px 16px", borderBottom: "1px solid var(--line)" }}>
        <div style={{ fontSize: 10.5, color: "var(--text-mute)" }}>creator directive</div>
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
            <span className="v acc">{directive.id}</span>
          </div>
          <div className="row">
            <span className="k">subject</span>
            <span className="v">{directive.subject_entity_name ?? directive.subject_kind}</span>
          </div>
          <div className="row">
            <span className="k">subject id</span>
            <span className="v">{directive.subject_entity_id ?? "—"}</span>
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
                <SupersededByChip
                  id={directive.superseded_by_id}
                  directivesById={directivesById}
                  onOpen={onOpenDirective}
                />
              </span>
            </div>
          )}
        </div>

        <div className="divider">activation</div>
        <div className="props">
          <div className="row">
            <span className="k">scope</span>
            <span className="v">{directive.activation_scope}</span>
          </div>
          <div className="row">
            <span className="k">allowed ids</span>
            <span className="v">{joinedIds(directive.activation_allowed_entity_ids)}</span>
          </div>
          <div className="row">
            <span className="k">excluded ids</span>
            <span className="v">{joinedIds(directive.activation_excluded_entity_ids)}</span>
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

        <div className="divider">operations</div>
        {directive.status === "active" ? (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            <button className="btn sm primary" disabled={busy} onClick={() => onRevoke(directive)}>
              revoke
            </button>
            <button
              className="btn sm ghost"
              disabled={busy || !hasReplacement}
              onClick={() => onSupersede(directive)}
            >
              supersede
            </button>
          </div>
        ) : (
          <div className="dim" style={{ fontSize: 11.5 }}>
            no active operations for {directive.status} directives
          </div>
        )}
      </div>
    </>
  );
}
