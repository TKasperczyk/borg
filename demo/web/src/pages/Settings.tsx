import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "wouter";

import {
  ApiError,
  fetchAssembledPrompts,
  fetchCreatorEntity,
  fetchDreamState,
  fetchPrompts,
  fetchSessions,
  resetBorg,
  resetPromptOverride,
  savePromptOverride,
  setCreatorEntity,
  setSessionParticipation,
} from "../api/client";
import type {
  AssembledPromptResponse,
  AssembledPromptSegment,
  PromptBlock,
  SessionParticipationPolicy,
  SessionRecord,
} from "../api/types";
import { useQuery } from "../api/useQuery";
import { humanMs } from "../format/time";
import { useAppState } from "../state/app-state";

type Toast = { text: string; tone: "ok" | "error" };
type PendingSwitch =
  | { type: "prompt"; key: string }
  | { type: "preview"; preview: boolean }
  | { type: "reset_prompt" };

const LIMIT = 50_000;
const POLICIES: SessionParticipationPolicy[] = ["active", "paused", "observing", "muted"];

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.status} ${error.message}`;
  }
  return error instanceof Error ? error.message : String(error);
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

function nextPolicy(policy: SessionParticipationPolicy): SessionParticipationPolicy {
  const index = POLICIES.indexOf(policy);
  return POLICIES[(index + 1) % POLICIES.length] ?? "active";
}

function segmentText(text: string, segment: AssembledPromptSegment): string {
  return text.slice(segment.start, segment.end);
}

function sessionLabel(session: SessionRecord): string {
  return session.label || session.session_id;
}

function AssembledBlocks({ data }: { data: AssembledPromptResponse }) {
  if (data.segments.length === 0) {
    return <pre>{data.text}</pre>;
  }

  return data.segments.map((segment) => (
    <div className="assembled-block" key={segment.id}>
      <b>▸ {segment.label}</b>
      <pre>{segmentText(data.text, segment)}</pre>
    </div>
  ));
}

export function SettingsPage() {
  const prompts = useQuery("prompts", fetchPrompts);
  const assembled = useQuery("prompts:assembled", fetchAssembledPrompts);
  const dream = useQuery("dream:state:settings", fetchDreamState);
  const creator = useQuery("entities:creator", fetchCreatorEntity);
  const sessions = useQuery("sessions", fetchSessions);
  const appState = useAppState();
  const [toast, showToast] = useToast();

  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [dirty, setDirty] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [pendingSwitch, setPendingSwitch] = useState<PendingSwitch | null>(null);
  const [saving, setSaving] = useState(false);
  const [resettingPrompt, setResettingPrompt] = useState(false);
  const [confirmResetPrompt, setConfirmResetPrompt] = useState(false);
  const [creatorName, setCreatorName] = useState("");
  const [creatorPending, setCreatorPending] = useState(false);
  const [participationEdit, setParticipationEdit] = useState<{
    sessionId: string;
    policy: SessionParticipationPolicy;
    reason: string;
  } | null>(null);
  const [participationPending, setParticipationPending] = useState<string | null>(null);
  const [resetToken, setResetToken] = useState("");
  const [adminResetPending, setAdminResetPending] = useState(false);

  const blocks = prompts.data?.blocks ?? [];
  const selectedBlock = useMemo(
    () => blocks.find((block) => block.key === selectedKey) ?? blocks[0] ?? null,
    [blocks, selectedKey],
  );

  useEffect(() => {
    if (selectedKey === null && blocks[0] !== undefined) {
      setSelectedKey(blocks[0].key);
    }
  }, [blocks, selectedKey]);

  useEffect(() => {
    if (selectedBlock !== null && !dirty) {
      setDraft(selectedBlock.current_text);
    }
  }, [dirty, selectedBlock]);

  useEffect(() => {
    if (creator.data !== undefined) {
      setCreatorName(creator.data?.canonical_name ?? "");
    }
  }, [creator.data]);

  useEffect(() => {
    if (!dirty) {
      return;
    }

    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [dirty]);

  const applySwitch = (next: PendingSwitch) => {
    setDirty(false);
    setPendingSwitch(null);
    if (next.type === "reset_prompt") {
      setConfirmResetPrompt(true);
      return;
    }
    if (next.type === "prompt") {
      const block = blocks.find((candidate) => candidate.key === next.key);
      setSelectedKey(next.key);
      setShowPreview(false);
      setConfirmResetPrompt(false);
      setDraft(block?.current_text ?? "");
      return;
    }
    setConfirmResetPrompt(false);
    setShowPreview(next.preview);
  };

  const requestSwitch = (next: PendingSwitch) => {
    if (dirty) {
      setPendingSwitch(next);
      return;
    }
    applySwitch(next);
  };

  const savePrompt = async () => {
    const trimmedLength = draft.trim().length;
    if (selectedBlock === null || saving || trimmedLength < 1 || trimmedLength > LIMIT) {
      return;
    }
    setSaving(true);
    try {
      const saved = await savePromptOverride(selectedBlock.key, draft);
      setDraft(saved.current_text);
      setDirty(false);
      showToast({ text: "override saved", tone: "ok" });
      prompts.refetch();
      assembled.refetch();
    } catch (caught) {
      showToast({ text: formatError(caught), tone: "error" });
    } finally {
      setSaving(false);
    }
  };

  const clearPrompt = async () => {
    if (selectedBlock === null || !selectedBlock.overridden || resettingPrompt) {
      return;
    }
    setResettingPrompt(true);
    try {
      const reset = await resetPromptOverride(selectedBlock.key);
      setDraft(reset.current_text);
      setDirty(false);
      setConfirmResetPrompt(false);
      showToast({ text: "override reset", tone: "ok" });
      prompts.refetch();
      assembled.refetch();
    } catch (caught) {
      showToast({ text: formatError(caught), tone: "error" });
    } finally {
      setResettingPrompt(false);
    }
  };

  const requestPromptReset = () => {
    if (dirty) {
      setPendingSwitch({ type: "reset_prompt" });
      return;
    }

    setConfirmResetPrompt(true);
  };

  const submitCreator = async (event: FormEvent) => {
    event.preventDefault();
    if (creatorPending || creatorName.trim().length === 0) {
      return;
    }
    setCreatorPending(true);
    try {
      await setCreatorEntity(creatorName);
      showToast({ text: "creator entity set", tone: "ok" });
      creator.refetch();
    } catch (caught) {
      showToast({ text: formatError(caught), tone: "error" });
    } finally {
      setCreatorPending(false);
    }
  };

  const submitParticipation = async () => {
    if (participationEdit === null || participationPending !== null) {
      return;
    }
    setParticipationPending(participationEdit.sessionId);
    try {
      await setSessionParticipation(
        participationEdit.sessionId,
        participationEdit.policy,
        participationEdit.reason,
      );
      setParticipationEdit(null);
      showToast({ text: "participation updated", tone: "ok" });
      sessions.refetch();
    } catch (caught) {
      showToast({ text: formatError(caught), tone: "error" });
    } finally {
      setParticipationPending(null);
    }
  };

  const submitReset = async () => {
    if (resetToken !== "RESET" || adminResetPending) {
      return;
    }
    setAdminResetPending(true);
    try {
      await resetBorg();
      setResetToken("");
      showToast({ text: "reset complete", tone: "ok" });
      prompts.refetch();
      assembled.refetch();
      dream.refetch();
      creator.refetch();
      sessions.refetch();
      appState.refetch();
    } catch (caught) {
      showToast({ text: formatError(caught), tone: "error" });
    } finally {
      setAdminResetPending(false);
    }
  };

  const trimmedLength = draft.trim().length;
  const invalidPromptLength = trimmedLength < 1 || trimmedLength > LIMIT;
  const overLimit = trimmedLength > LIMIT;
  const runtime = appState.data?.runtime;
  const scheduler = dream.data?.scheduler;
  const participationSessions =
    sessions.data?.sessions.filter((session) => session.participation_policy !== undefined) ?? [];

  return (
    <main className="page settings-page">
      <div className="page-header">
        <div className="page-title">SETTINGS</div>
        <div className="page-subtitle">prompt blocks · runtime · scheduler · entities · danger</div>
      </div>

      <div className="settings-layout">
        <aside className="settings-prompt-list">
          <div className="panel-head">
            <b>PROMPT BLOCKS</b>
            <span>{blocks.length}</span>
          </div>
          <div className="prompt-list-scroll">
            {blocks.map((block) => (
              <button
                className={selectedBlock?.key === block.key ? "prompt-row prompt-row-active" : "prompt-row"}
                key={block.key}
                type="button"
                onClick={() => requestSwitch({ type: "prompt", key: block.key })}
              >
                <b>{block.key}</b>
                <span>{block.description}</span>
                <i>{block.overridden ? "OVERRIDE" : "default"}</i>
              </button>
            ))}
          </div>
          <button
            className="settings-preview-toggle"
            type="button"
            onClick={() => requestSwitch({ type: "preview", preview: !showPreview })}
          >
            {showPreview ? "← BACK TO EDITOR" : "▸ PREVIEW ASSEMBLED FRAMING"}
          </button>
        </aside>

        <section className="settings-editor">
          {pendingSwitch === null ? null : (
            <div className="dirty-guard">
              <span>
                {pendingSwitch.type === "reset_prompt"
                  ? "reset to default discards the unsaved draft first"
                  : "discard unsaved prompt changes?"}
              </span>
              <button className="solid-button" type="button" onClick={() => applySwitch(pendingSwitch)}>
                DISCARD DRAFT
              </button>
              <button className="outline-button" type="button" onClick={() => setPendingSwitch(null)}>
                STAY
              </button>
            </div>
          )}

          {showPreview ? (
            <div className="assembled-view">
              <div className="editor-head">
                <div>
                  <b>assembled framing preview</b>
                  <span>exactly what deliberation sees</span>
                </div>
              </div>
              <div className="assembled-blocks">
                {assembled.data === undefined ? (
                  <div className="quiet-line">loading assembled framing…</div>
                ) : (
                  <AssembledBlocks data={assembled.data} />
                )}
              </div>
            </div>
          ) : selectedBlock === null ? (
            <div className="quiet-line">loading prompt blocks…</div>
          ) : (
            <>
              <div className="editor-head">
                <div>
                  <b>{selectedBlock.key}</b>
                  <span>{selectedBlock.label}</span>
                </div>
                <i>{selectedBlock.overridden ? "OVERRIDE ACTIVE" : "BUILT-IN DEFAULT"}</i>
                <strong className={overLimit ? "tone-error" : ""}>
                  {trimmedLength.toLocaleString()} trimmed / 50,000
                </strong>
                <button
                  className="solid-button"
                  type="button"
                  disabled={saving || invalidPromptLength}
                  onClick={savePrompt}
                >
                  SAVE OVERRIDE
                </button>
                {confirmResetPrompt ? (
                  <span className="inline-actions">
                    <button
                      className="outline-button danger"
                      type="button"
                      disabled={resettingPrompt}
                      onClick={clearPrompt}
                    >
                      CONFIRM RESET
                    </button>
                    <button className="ghost-button" type="button" onClick={() => setConfirmResetPrompt(false)}>
                      CANCEL
                    </button>
                  </span>
                ) : (
                  <button
                    className="outline-button danger"
                    type="button"
                    disabled={!selectedBlock.overridden || resettingPrompt}
                    onClick={requestPromptReset}
                  >
                    RESET TO DEFAULT
                  </button>
                )}
              </div>
              <textarea
                className="prompt-textarea"
                value={draft}
                onChange={(event) => {
                  setDraft(event.target.value);
                  setDirty(true);
                }}
              />
              <div className="settings-note">overrides take effect next turn, no restart</div>
            </>
          )}
        </section>

        <aside className="settings-rail">
          <section className="settings-card">
            <div className="panel-head">
              <b>RUNTIME</b>
              <span>{appState.data?.version === undefined ? "" : `v${appState.data.version}`}</span>
            </div>
            <dl className="settings-dl">
              {runtime?.model === undefined ? null : (
                <>
                  <dt>model</dt>
                  <dd>{runtime.model}</dd>
                </>
              )}
              {runtime?.embedding === undefined ? null : (
                <>
                  <dt>embedding</dt>
                  <dd>
                    {runtime.embedding.model} · {runtime.embedding.dims} dims
                  </dd>
                </>
              )}
            </dl>
          </section>

          <section className="settings-card">
            <div className="panel-head">
              <b>MAINTENANCE SCHEDULER</b>
              <span>read-only</span>
            </div>
            {scheduler === undefined ? (
              <div className="quiet-line">loading scheduler…</div>
            ) : (
              <div className="scheduler-readonly">
                <div>state · {scheduler.enabled ? "enabled" : "disabled"}</div>
                <div>light · {humanMs(scheduler.light_interval_ms)}</div>
                <p>{scheduler.light_processes.map((process) => process.replace(/-/g, " ")).join(" · ")}</p>
                <div>heavy · {humanMs(scheduler.heavy_interval_ms)}</div>
                <p>{scheduler.heavy_processes.map((process) => process.replace(/-/g, " ")).join(" · ")}</p>
                <div>optimize storage · {scheduler.optimize_storage ? "enabled" : "disabled"}</div>
                <Link href="/dream">per-process budgets → DREAM</Link>
              </div>
            )}
          </section>

          <section className="settings-card">
            <div className="panel-head">
              <b>ENTITIES</b>
              <span>{creator.data?.canonical_name ?? ""}</span>
            </div>
            <form className="entity-form" onSubmit={submitCreator}>
              <input
                value={creatorName}
                placeholder="creator name"
                onChange={(event) => setCreatorName(event.target.value)}
              />
              <button className="solid-button" type="submit" disabled={creatorPending}>
                SET
              </button>
            </form>
            <div className="settings-note">corrections from this entity are ground truth</div>
          </section>

          <section className="settings-card">
            <div className="panel-head">
              <b>SESSION PARTICIPATION</b>
              <span>{participationSessions.length}</span>
            </div>
            <div className="participation-list">
              {participationSessions.map((session) => {
                const editing = participationEdit?.sessionId === session.session_id;
                const target = editing ? participationEdit.policy : nextPolicy(session.participation_policy);
                return (
                  <div className="participation-row" key={session.session_id}>
                    <span>{sessionLabel(session)}</span>
                    <button
                      className="chip-button"
                      type="button"
                      onClick={() =>
                        setParticipationEdit({
                          sessionId: session.session_id,
                          policy: target,
                          reason: "",
                        })
                      }
                    >
                      {session.participation_policy}
                    </button>
                    {editing ? (
                      <div className="participation-edit">
                        <span>set {target}</span>
                        <input
                          value={participationEdit.reason}
                          placeholder="reason"
                          onChange={(event) =>
                            setParticipationEdit({
                              ...participationEdit,
                              reason: event.target.value,
                            })
                          }
                        />
                        <button
                          className="solid-button"
                          type="button"
                          disabled={participationPending === session.session_id}
                          onClick={submitParticipation}
                        >
                          APPLY
                        </button>
                        <button className="ghost-button" type="button" onClick={() => setParticipationEdit(null)}>
                          CANCEL
                        </button>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </section>

          <section className="settings-card danger-card">
            <div className="panel-head">
              <b>DANGER</b>
              <span>reset substrate</span>
            </div>
            <p>Clears demo memory, prompt overrides, cached live state, and opens a fresh substrate.</p>
            <input
              value={resetToken}
              placeholder="type RESET"
              onChange={(event) => setResetToken(event.target.value)}
            />
            <button
              className="danger-solid"
              type="button"
              disabled={resetToken !== "RESET" || adminResetPending}
              onClick={submitReset}
            >
              RESET
            </button>
          </section>
        </aside>
      </div>

      {toast === null ? null : <div className={`toast toast-${toast.tone}`}>{toast.text}</div>}
    </main>
  );
}
