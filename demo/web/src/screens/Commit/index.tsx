import { useMemo, useState } from "react";

import {
  getCommitments,
  postCommitment,
  postCommitmentRevoke,
  postCorrectionCorrect,
} from "../../api/client";
import {
  COMMITMENT_CREATE_TYPES,
  COMMITMENT_KINDS,
  type CommitmentCreateType,
  type CommitmentEnforcement,
  type CommitmentItem,
  type CommitmentKind,
  type CommitmentState,
  type CreateCommitmentRequest,
} from "../../api/types";
import { Modal } from "../../components/Modal";
import { Tag } from "../../components/Tag";
import { WhyDrawer } from "../../components/WhyDrawer";
import { useApi } from "../../hooks/use-api";
import { dateLabel, parseJsonPatch, shortId } from "../screen-utils";

type StateFilter = CommitmentState | "all";
type EnforcementFilter = CommitmentEnforcement | "all";
type CommitModal =
  | {
      kind: "create";
      type: CommitmentCreateType;
      commitmentKind: CommitmentKind;
      directive: string;
      priority: string;
      audience: string;
      made_to: string;
      about: string;
      directive_family: string;
      expires_at: string;
    }
  | { kind: "revoke"; commitment: CommitmentItem; reason: string }
  | { kind: "correct"; commitment: CommitmentItem; patch: string; reason: string };

function commitmentPatch(commitment: CommitmentItem): string {
  return JSON.stringify(
    {
      directive: commitment.text,
      priority: commitment.priority,
    },
    null,
    2,
  );
}

function stateTag(state: CommitmentState) {
  if (state === "active") {
    return "acc";
  }
  if (state === "revoked") {
    return "bad";
  }
  return "warn";
}

const COMMITMENT_TEXT_MAX_LENGTH = 2_000;
const COMMITMENT_DIRECTIVE_FAMILY_MAX_LENGTH = 64;

function requiredLabel(text: string) {
  return (
    <span>
      {text}{" "}
      <strong aria-hidden="true" style={{ color: "var(--bad)" }}>
        *
      </strong>
    </span>
  );
}

function createCommitmentModal(): CommitModal {
  return {
    kind: "create",
    type: "rule",
    commitmentKind: "process_norm",
    directive: "",
    priority: "5",
    audience: "",
    made_to: "",
    about: "",
    directive_family: "",
    expires_at: "",
  };
}

function optionalText(value: string): string | undefined {
  const trimmed = value.trim();
  return trimmed.length === 0 ? undefined : trimmed;
}

function dateInputToUnixMs(value: string): number | undefined {
  if (value.length === 0) {
    return undefined;
  }

  const parsed = new Date(`${value}T00:00:00`).getTime();
  return Number.isFinite(parsed) ? parsed : undefined;
}

function canSubmitModal(modal: CommitModal | null): boolean {
  if (modal === null) {
    return false;
  }

  if (modal.kind === "revoke") {
    return true;
  }

  if (modal.kind === "correct") {
    return modal.patch.trim().length > 0;
  }

  const priority = Number(modal.priority);
  const expiresAt = dateInputToUnixMs(modal.expires_at);

  return (
    modal.directive.trim().length > 0 &&
    Number.isInteger(priority) &&
    priority >= 1 &&
    priority <= 10 &&
    (modal.expires_at.length === 0 || expiresAt !== undefined)
  );
}

function createCommitmentRequest(modal: Extract<CommitModal, { kind: "create" }>) {
  const input: CreateCommitmentRequest = {
    type: modal.type,
    kind: modal.commitmentKind,
    directive: modal.directive,
    priority: Number(modal.priority),
  };
  const audience = optionalText(modal.audience);
  const madeTo = optionalText(modal.made_to);
  const about = optionalText(modal.about);
  const directiveFamily = optionalText(modal.directive_family);
  const expiresAt = dateInputToUnixMs(modal.expires_at);

  if (audience !== undefined) {
    input.audience = audience;
  }
  if (madeTo !== undefined) {
    input.made_to = madeTo;
  }
  if (about !== undefined) {
    input.about = about;
  }
  if (directiveFamily !== undefined) {
    input.directive_family = directiveFamily;
  }
  if (expiresAt !== undefined) {
    input.expires_at = expiresAt;
  }

  return input;
}

export function CommitScreen() {
  const api = useApi(() => getCommitments({ state: "all" }), []);
  const [state, setState] = useState<StateFilter>("active");
  const [enforcement, setEnforcement] = useState<EnforcementFilter>("all");
  const [audience, setAudience] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [modal, setModal] = useState<CommitModal | null>(null);
  const [whyId, setWhyId] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const commitments = api.data?.commitments ?? [];
  const audiences = useMemo(
    () => ["all", ...[...new Set(commitments.map((item) => item.audience ?? "global"))].sort()],
    [commitments],
  );
  const filtered = useMemo(
    () =>
      commitments.filter((item) => {
        if (state !== "all" && item.state !== state) {
          return false;
        }
        if (enforcement !== "all" && item.enforcement_class !== enforcement) {
          return false;
        }
        if (audience !== "all" && (item.audience ?? "global") !== audience) {
          return false;
        }
        return true;
      }),
    [audience, commitments, enforcement, state],
  );
  const selected = filtered.find((item) => item.id === selectedId) ?? filtered[0] ?? null;

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

    if (modal.kind === "create") {
      await runAction("commitment-create", async () => {
        await postCommitment(createCommitmentRequest(modal));
      });
      return;
    }

    if (modal.kind === "correct") {
      await runAction("commitment-correct", async () => {
        const patch = parseJsonPatch(modal.patch);
        // Invalidates GET /api/correction/reviews; accepted reviews later invalidate commitments.
        await postCorrectionCorrect(modal.commitment.id, {
          patch,
          ...(modal.reason.trim().length === 0 ? {} : { reason: modal.reason.trim() }),
        });
      });
      return;
    }

    await runAction("commitment-revoke", async () => {
      const reason = optionalText(modal.reason);
      await postCommitmentRevoke(modal.commitment.id, reason === undefined ? {} : { reason });
    });
  }

  if (api.loading && api.data === null) {
    return <div className="notice">loading commitments</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <h1>commitments</h1>
        <span className="desc">scoped promises · rules · preferences · boundaries</span>
        <span className="spacer"></span>
        <button
          className="btn sm primary"
          disabled={busy !== null}
          aria-label="add commitment"
          onClick={() => setModal(createCommitmentModal())}
        >
          + add
        </button>
        <div className="filter-pills">
          {(["all", "active", "revoked", "expired"] as const).map((value) => (
            <span
              key={value}
              className={`pill ${state === value ? "on" : ""}`}
              onClick={() => setState(value)}
            >
              {value}
            </span>
          ))}
        </div>
        <span className="sep">|</span>
        <div className="filter-pills">
          {(["all", "critical", "advisory"] as const).map((value) => (
            <span
              key={value}
              className={`pill ${enforcement === value ? "on" : ""}`}
              onClick={() => setEnforcement(value)}
            >
              {value}
            </span>
          ))}
        </div>
        <span className="sep">|</span>
        <div className="filter-pills">
          {audiences.map((value) => (
            <span
              key={value}
              className={`pill ${audience === value ? "on" : ""}`}
              onClick={() => setAudience(value)}
            >
              {value}
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
                <th style={{ minWidth: 240 }}>text</th>
                <th style={{ width: 96 }}>audience</th>
                <th style={{ width: 96 }}>enforce</th>
                <th style={{ width: 84 }}>state</th>
                <th style={{ width: 50, textAlign: "right" }}>p</th>
                <th style={{ width: 100 }}>since</th>
                <th style={{ width: 86 }}></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => (
                <tr
                  key={item.id}
                  onClick={() => setSelectedId(item.id)}
                  className={item.id === selected?.id ? "selected" : ""}
                  style={{ cursor: "pointer" }}
                >
                  <td>
                    <span className="acc">{shortId(item.id)}</span>
                  </td>
                  <td
                    className="wrap"
                    style={{ fontFamily: "var(--sans)", fontSize: "12px", lineHeight: 1.45 }}
                  >
                    {item.text}
                    <div className="dim" style={{ fontSize: 10, marginTop: 2 }}>
                      {item.type} · {item.kind}
                      {item.about === null ? "" : ` · about:${item.about}`}
                    </div>
                  </td>
                  <td>
                    <span
                      className={item.audience === null ? "mute" : "acc"}
                      title={item.audience ?? "global"}
                      style={{
                        display: "block",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {item.audience === null ? "global" : shortId(item.audience)}
                    </span>
                  </td>
                  <td>
                    <Tag kind={item.enforcement_class === "critical" ? "bad" : ""} dot>
                      {item.enforcement_class}
                    </Tag>
                  </td>
                  <td>
                    <Tag kind={stateTag(item.state)} dot>
                      {item.state}
                    </Tag>
                  </td>
                  <td
                    className="tab-num"
                    style={{
                      textAlign: "right",
                      color: item.priority >= 8 ? "var(--bad)" : "var(--text-dim)",
                    }}
                  >
                    {item.priority}
                  </td>
                  <td className="dim" style={{ fontSize: 11 }}>
                    {dateLabel(item.created_at)}
                  </td>
                  <td>
                    {item.state === "active" ? (
                      <button
                        className="btn sm ghost"
                        disabled={busy !== null}
                        onClick={(event) => {
                          event.stopPropagation();
                          setModal({ kind: "revoke", commitment: item, reason: "" });
                        }}
                      >
                        revoke
                      </button>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ overflowY: "auto", background: "var(--bg-0)" }}>
          {selected === null ? (
            <div className="notice">no commitments in filter</div>
          ) : (
            <CommitmentDetail
              commitment={selected}
              busy={busy !== null}
              onRevoke={(commitment) => setModal({ kind: "revoke", commitment, reason: "" })}
              onWhy={(commitment) => setWhyId(commitment.id)}
              onCorrect={(commitment) =>
                setModal({
                  kind: "correct",
                  commitment,
                  patch: commitmentPatch(commitment),
                  reason: "",
                })
              }
            />
          )}
        </div>
      </div>

      <Modal
        open={modal !== null}
        title={
          modal?.kind === "revoke"
            ? "revoke commitment"
            : modal?.kind === "correct"
              ? "correct commitment"
              : "add commitment"
        }
        onClose={() => {
          if (busy === null) {
            setModal(null);
          }
        }}
        footer={
          <>
            <span className="dim" style={{ fontSize: 10.5, marginRight: "auto" }}>
              {modal?.kind === "create" ? "marked as creator-authored advice" : ""}
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
              disabled={busy !== null || !canSubmitModal(modal)}
              onClick={() => void submitModal()}
            >
              {busy === null
                ? modal?.kind === "revoke"
                  ? "revoke"
                  : modal?.kind === "correct"
                    ? "queue"
                    : "save"
                : "saving"}
            </button>
          </>
        }
      >
        {modal?.kind === "create" ? (
          <div className="modal-form">
            <label className="modal-field">
              {requiredLabel("type")}
              <select
                required
                value={modal.type}
                onChange={(event) =>
                  setModal({ ...modal, type: event.target.value as CommitmentCreateType })
                }
              >
                {COMMITMENT_CREATE_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
            <label className="modal-field">
              {requiredLabel("kind")}
              <select
                required
                value={modal.commitmentKind}
                onChange={(event) =>
                  setModal({ ...modal, commitmentKind: event.target.value as CommitmentKind })
                }
              >
                {COMMITMENT_KINDS.map((kind) => (
                  <option key={kind} value={kind}>
                    {kind}
                  </option>
                ))}
              </select>
            </label>
            <label className="modal-field">
              {requiredLabel("directive")}
              <textarea
                required
                maxLength={COMMITMENT_TEXT_MAX_LENGTH}
                value={modal.directive}
                onChange={(event) => setModal({ ...modal, directive: event.target.value })}
              />
            </label>
            <label className="modal-field">
              {requiredLabel("priority")}
              <input
                required
                type="number"
                min="1"
                max="10"
                step="1"
                value={modal.priority}
                onChange={(event) => setModal({ ...modal, priority: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>audience: who this commitment is scoped to (when conversing with...)</span>
              <input
                maxLength={COMMITMENT_TEXT_MAX_LENGTH}
                value={modal.audience}
                onChange={(event) => setModal({ ...modal, audience: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>made_to: who borg promised this to (often same as audience)</span>
              <input
                maxLength={COMMITMENT_TEXT_MAX_LENGTH}
                value={modal.made_to}
                onChange={(event) => setModal({ ...modal, made_to: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>about: who/what this is about</span>
              <input
                maxLength={COMMITMENT_TEXT_MAX_LENGTH}
                value={modal.about}
                onChange={(event) => setModal({ ...modal, about: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>directive_family</span>
              <input
                maxLength={COMMITMENT_DIRECTIVE_FAMILY_MAX_LENGTH}
                value={modal.directive_family}
                onChange={(event) => setModal({ ...modal, directive_family: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>expires_at</span>
              <input
                type="date"
                value={modal.expires_at}
                onChange={(event) => setModal({ ...modal, expires_at: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "revoke" ? (
          <div className="modal-form">
            <div className="dim">{modal.commitment.text}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                maxLength={2000}
                value={modal.reason}
                onChange={(event) => setModal({ ...modal, reason: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "correct" ? (
          <div className="modal-form">
            <div className="dim">{modal.commitment.text}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                maxLength={2000}
                value={modal.reason}
                onChange={(event) => setModal({ ...modal, reason: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>json patch</span>
              <textarea
                maxLength={COMMITMENT_TEXT_MAX_LENGTH}
                value={modal.patch}
                onChange={(event) => setModal({ ...modal, patch: event.target.value })}
              />
            </label>
          </div>
        ) : null}
      </Modal>
      <WhyDrawer open={whyId !== null} id={whyId} onClose={() => setWhyId(null)} />
    </div>
  );
}

function CommitmentDetail({
  commitment,
  busy,
  onRevoke,
  onWhy,
  onCorrect,
}: {
  commitment: CommitmentItem;
  busy: boolean;
  onRevoke: (commitment: CommitmentItem) => void;
  onWhy: (commitment: CommitmentItem) => void;
  onCorrect: (commitment: CommitmentItem) => void;
}) {
  return (
    <>
      <div style={{ padding: "16px 16px 10px 16px", borderBottom: "1px solid var(--line)" }}>
        <div style={{ fontSize: 10.5, color: "var(--text-mute)" }}>commitment</div>
        <div
          style={{
            fontSize: 14,
            color: "var(--text)",
            fontFamily: "var(--sans)",
            margin: "6px 0 10px 0",
          }}
        >
          {commitment.text}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Tag kind={commitment.enforcement_class === "critical" ? "bad" : ""} dot>
            {commitment.enforcement_class}
          </Tag>
          <Tag kind={stateTag(commitment.state)} dot>
            {commitment.state}
          </Tag>
          <Tag>{commitment.type}</Tag>
          <Tag>{commitment.kind}</Tag>
        </div>
      </div>
      <div style={{ padding: 16 }}>
        <div className="props">
          <div className="row">
            <span className="k">id</span>
            <span className="v acc">{commitment.id}</span>
          </div>
          <div className="row">
            <span className="k">audience</span>
            <span className="v">{commitment.audience ?? "global"}</span>
          </div>
          <div className="row">
            <span className="k">made to</span>
            <span className="v">{commitment.made_to ?? <span className="mute">—</span>}</span>
          </div>
          <div className="row">
            <span className="k">about</span>
            <span className="v">{commitment.about ?? <span className="mute">—</span>}</span>
          </div>
          <div className="row">
            <span className="k">priority</span>
            <span className="v tab-num">{commitment.priority}</span>
          </div>
          <div className="row">
            <span className="k">source</span>
            <span className="v">{commitment.source}</span>
          </div>
          <div className="row">
            <span className="k">created</span>
            <span className="v">{dateLabel(commitment.created_at)}</span>
          </div>
          {commitment.revoked_at === null ? null : (
            <div className="row">
              <span className="k">revoked at</span>
              <span className="v">{dateLabel(commitment.revoked_at)}</span>
            </div>
          )}
          {commitment.expired_at === null ? null : (
            <div className="row">
              <span className="k">expired at</span>
              <span className="v">{dateLabel(commitment.expired_at)}</span>
            </div>
          )}
          {commitment.superseded_by_id === null ? null : (
            <div className="row">
              <span className="k">superseded by</span>
              <span className="v">{commitment.superseded_by_id}</span>
            </div>
          )}
        </div>

        <div className="divider">enforcement</div>
        <div style={{ fontSize: 11.5, color: "var(--text-dim)", lineHeight: 1.55 }}>
          {commitment.enforcement_class === "critical" ? (
            <>
              checked as a hard constraint. critical domain:{" "}
              <span className="bad">{commitment.critical_domain ?? "unspecified"}</span>.
            </>
          ) : (
            <>tracked as advisory context and surfaced before generation.</>
          )}
        </div>

        <div className="divider">provenance</div>
        <div
          style={{
            fontSize: 11,
            color: "var(--text-dim)",
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          {commitment.source_stream_entry_ids.length === 0 ? (
            <div className="dim">no stream source ids recorded</div>
          ) : (
            commitment.source_stream_entry_ids.map((id) => (
              <div key={id}>
                <span className="acc">[{id}]</span> source stream entry
              </div>
            ))
          )}
        </div>

        <div className="divider">operations</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          <button className="btn sm" disabled={busy} onClick={() => onWhy(commitment)}>
            why
          </button>
          <button
            className="btn sm ghost"
            disabled={busy || commitment.state !== "active"}
            onClick={() => onRevoke(commitment)}
          >
            revoke
          </button>
          <button className="btn sm ghost" disabled={busy} onClick={() => onCorrect(commitment)}>
            correct
          </button>
        </div>
      </div>
    </>
  );
}
