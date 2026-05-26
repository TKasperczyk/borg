import { useMemo, useState } from "react";

import {
  getIdentity,
  patchGoal,
  patchOpenQuestion,
  postCorrectionCorrect,
  postCorrectionForget,
  postGoal,
  postGrowthMarker,
  postValue,
} from "../../api/client";
import type { IdentityGoal, IdentityTrait, IdentityValue, OpenQuestion } from "../../api/types";
import { Modal } from "../../components/Modal";
import { Tag } from "../../components/Tag";
import { WhyDrawer } from "../../components/WhyDrawer";
import { useApi } from "../../hooks/use-api";
import { clamp01, dateLabel, parseJsonPatch } from "../screen-utils";

type QuestionFilter = "all" | OpenQuestion["status"];

type IdentityModal =
  | { kind: "value"; name: string; description: string }
  | { kind: "goal"; description: string; priority: string }
  | { kind: "growth"; description: string; source: string }
  | { kind: "goal-progress"; goal: IdentityGoal; note: string; progress: string }
  | { kind: "question-resolve"; question: OpenQuestion; text: string }
  | { kind: "question-abandon"; question: OpenQuestion; text: string }
  | { kind: "forget"; id: string; label: string }
  | { kind: "correct"; id: string; label: string; patch: string; reason: string };

function valuePatch(value: IdentityValue): string {
  return JSON.stringify({ description: value.description }, null, 2);
}

function goalPatch(goal: IdentityGoal): string {
  return JSON.stringify({ description: goal.description }, null, 2);
}

function traitPatch(trait: IdentityTrait): string {
  return JSON.stringify({ label: trait.label }, null, 2);
}

function questionPatch(question: OpenQuestion): string {
  return JSON.stringify({ question: question.question }, null, 2);
}

function questionTag(status: OpenQuestion["status"]) {
  if (status === "open") {
    return "acc";
  }
  if (status === "resolved") {
    return "info";
  }
  return "warn";
}

function IdentityCorrectionButtons({
  busy,
  id,
  label,
  patch,
  onWhy,
  onModal,
}: {
  busy: boolean;
  id: string;
  label: string;
  patch: string;
  onWhy: (id: string) => void;
  onModal: (modal: IdentityModal) => void;
}) {
  return (
    <>
      <button className="btn sm" disabled={busy} onClick={() => onWhy(id)}>
        why
      </button>
      <button
        className="btn sm ghost"
        disabled={busy}
        onClick={() => onModal({ kind: "forget", id, label })}
      >
        forget
      </button>
      <button
        className="btn sm ghost"
        disabled={busy}
        onClick={() => onModal({ kind: "correct", id, label, patch, reason: "" })}
      >
        correct
      </button>
    </>
  );
}

export function IdentityScreen() {
  const api = useApi(getIdentity, []);
  const [questionFilter, setQuestionFilter] = useState<QuestionFilter>("all");
  const [modal, setModal] = useState<IdentityModal | null>(null);
  const [whyId, setWhyId] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const identity = api.data;
  const questions = useMemo(() => {
    const all = identity?.open_questions ?? [];
    return questionFilter === "all"
      ? all
      : all.filter((question) => question.status === questionFilter);
  }, [identity?.open_questions, questionFilter]);
  const currentPeriod =
    identity?.periods.find((period) => period.end_ts === null) ?? identity?.periods[0] ?? null;
  const activeGoals = identity?.goals.filter((goal) => goal.status === "active").length ?? 0;

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

    if (modal.kind === "value") {
      await runAction("value", async () => {
        await postValue({
          name: modal.name,
          ...(modal.description.trim().length === 0 ? {} : { description: modal.description }),
        });
      });
      return;
    }

    if (modal.kind === "goal") {
      await runAction("goal", async () => {
        await postGoal({
          description: modal.description,
          ...(modal.priority.trim().length === 0 ? {} : { priority: Number(modal.priority) }),
        });
      });
      return;
    }

    if (modal.kind === "growth") {
      await runAction("growth", async () => {
        await postGrowthMarker({
          description: modal.description,
          ...(modal.source.trim().length === 0 ? {} : { source: modal.source }),
        });
      });
      return;
    }

    if (modal.kind === "goal-progress") {
      await runAction("goal-progress", async () => {
        await patchGoal(modal.goal.id, {
          action: "progress",
          ...(modal.note.trim().length === 0 ? {} : { note: modal.note }),
          ...(modal.progress.trim().length === 0 ? {} : { progress: Number(modal.progress) }),
        });
      });
      return;
    }

    if (modal.kind === "question-resolve") {
      await runAction("question-resolve", async () => {
        await patchOpenQuestion(modal.question.id, {
          action: "resolve",
          resolution: modal.text,
        });
      });
      return;
    }

    if (modal.kind === "question-abandon") {
      await runAction("question-abandon", async () => {
        await patchOpenQuestion(modal.question.id, {
          action: "abandon",
          reason: modal.text,
        });
      });
      return;
    }

    if (modal.kind === "forget") {
      await runAction("forget", async () => {
        // Invalidates GET /api/identity and the self memory band.
        await postCorrectionForget(modal.id);
      });
      return;
    }

    await runAction("correct", async () => {
      const patch = parseJsonPatch(modal.patch);
      // Invalidates GET /api/correction/reviews; accepted reviews later invalidate identity.
      await postCorrectionCorrect(modal.id, {
        patch,
        ...(modal.reason.trim().length === 0 ? {} : { reason: modal.reason.trim() }),
      });
    });
  }

  if (api.loading && identity === null) {
    return <div className="notice">loading identity</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  if (identity === null) {
    return <div className="notice">no identity records</div>;
  }

  return (
    <div className="identity">
      <div className="id-hero">
        <div>
          <div
            style={{
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              color: "var(--text-mute)",
              marginBottom: 8,
            }}
          >
            self::current
          </div>
          <div className="stamp">
            borg <span className="acc">·</span> v89 identity substrate
          </div>
          <div className="quote">
            values, goals, traits, open questions, growth markers, and autobiography.
          </div>
          <div className="quote-attr">
            current period: {currentPeriod?.label ?? "none"} · {dateLabel(currentPeriod?.start_ts)}
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="mini-stat">
            <div className="k">autobiographical periods</div>
            <div className="v tab-num">{identity.periods.length}</div>
            <div className="sub">
              current: <span className="acc">{currentPeriod?.label ?? "—"}</span>
            </div>
          </div>
          <div className="mini-stat">
            <div className="k">growth markers</div>
            <div className="v tab-num">{identity.growth_markers.length}</div>
            <div className="sub">evidence-backed changes</div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="mini-stat">
            <div className="k">values · goals · traits</div>
            <div className="v tab-num">
              {identity.values.length} · {identity.goals.length} · {identity.traits.length}
            </div>
            <div className="sub">{activeGoals} active goals</div>
          </div>
          <div className="mini-stat">
            <div className="k">open-question events</div>
            <div className="v tab-num">{identity.open_question_events.length}</div>
            <div className="sub">create · resolve · abandon · bump</div>
          </div>
        </div>
      </div>

      {operatorError === null ? null : (
        <div className="notice bad" style={{ gridColumn: "span 12", padding: 12 }}>
          {operatorError}
        </div>
      )}

      <div className="id-card" style={{ gridColumn: "span 5" }}>
        <div className="h">
          <span className="ttl">values</span>
          <span className="n">{identity.values.length}</span>
          <span style={{ flex: 1 }}></span>
          <span className="dim" style={{ textTransform: "none" }}>
            preserved across turns
          </span>
          <button
            className="btn sm ghost"
            disabled={busy !== null}
            aria-label="add value"
            onClick={() => setModal({ kind: "value", name: "", description: "" })}
          >
            + add
          </button>
        </div>
        <div className="body">
          {identity.values.map((value) => (
            <div key={value.id} className="item">
              <div
                style={{
                  color: "var(--text)",
                  fontFamily: "var(--sans)",
                  fontSize: 12.5,
                  lineHeight: 1.45,
                  marginBottom: 6,
                }}
              >
                {value.description}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div className="bar-meter" style={{ flex: 1 }}>
                  <div
                    className="fill"
                    style={{ width: `${clamp01(value.confidence) * 100}%` }}
                  ></div>
                </div>
                <span
                  className="dim tab-num"
                  style={{ fontSize: 10.5, width: 34, textAlign: "right" }}
                >
                  {value.confidence.toFixed(2)}
                </span>
                <span className="dim" style={{ fontSize: 10.5, whiteSpace: "nowrap" }}>
                  {value.support_count} src · {dateLabel(value.created_at)}
                </span>
                <IdentityCorrectionButtons
                  busy={busy !== null}
                  id={value.id}
                  label={value.label}
                  patch={valuePatch(value)}
                  onWhy={setWhyId}
                  onModal={setModal}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 4" }}>
        <div className="h">
          <span className="ttl">goals</span>
          <span className="n">{activeGoals} active</span>
          <span style={{ flex: 1 }}></span>
          <button
            className="btn sm ghost"
            disabled={busy !== null}
            aria-label="add goal"
            onClick={() => setModal({ kind: "goal", description: "", priority: "" })}
          >
            + add
          </button>
        </div>
        <div className="body">
          {identity.goals.map((goal) => (
            <div key={goal.id} className="item">
              <div
                style={{
                  color: "var(--text)",
                  fontFamily: "var(--sans)",
                  fontSize: 13,
                  marginBottom: 4,
                }}
              >
                {goal.description}
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <Tag kind={goal.status === "active" ? "acc" : ""} dot>
                  {goal.status}
                </Tag>
                <span className="dim" style={{ fontSize: 10.5 }}>
                  priority {goal.priority.toFixed(2)}
                </span>
                <span className="dim" style={{ fontSize: 10.5 }}>
                  since {dateLabel(goal.created_at)}
                </span>
                <span style={{ flex: 1 }}></span>
                <button
                  className="btn sm"
                  disabled={busy !== null || goal.status !== "active"}
                  onClick={() =>
                    void runAction("goal-complete", async () => {
                      await patchGoal(goal.id, { action: "complete" });
                    })
                  }
                >
                  complete
                </button>
                <button
                  className="btn sm ghost"
                  disabled={busy !== null || goal.status !== "active"}
                  onClick={() =>
                    void runAction("goal-block", async () => {
                      await patchGoal(goal.id, { action: "block" });
                    })
                  }
                >
                  block
                </button>
                <button
                  className="btn sm ghost"
                  disabled={busy !== null}
                  onClick={() => setModal({ kind: "goal-progress", goal, note: "", progress: "" })}
                >
                  progress
                </button>
                <IdentityCorrectionButtons
                  busy={busy !== null}
                  id={goal.id}
                  label={goal.description}
                  patch={goalPatch(goal)}
                  onWhy={setWhyId}
                  onModal={setModal}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 3" }}>
        <div className="h">
          <span className="ttl">open questions</span>
          <span className="n">{questions.length}</span>
        </div>
        <div className="body">
          <div className="filter-pills" style={{ marginBottom: 8 }}>
            {(["all", "open", "resolved", "abandoned"] as const).map((filter) => (
              <span
                key={filter}
                className={`pill ${questionFilter === filter ? "on" : ""}`}
                onClick={() => setQuestionFilter(filter)}
              >
                {filter}
              </span>
            ))}
          </div>
          {questions.map((question) => (
            <div key={question.id} className="item">
              <div
                style={{
                  color: "var(--text-dim)",
                  fontFamily: "var(--sans)",
                  fontSize: 12,
                  lineHeight: 1.5,
                  marginBottom: 4,
                }}
              >
                {question.question}
              </div>
              <div
                style={{
                  display: "flex",
                  gap: 6,
                  alignItems: "center",
                  flexWrap: "wrap",
                  fontSize: 10.5,
                }}
              >
                <Tag kind={questionTag(question.status)} dot>
                  {question.status}
                </Tag>
                <span className="dim">urg {question.urgency.toFixed(2)}</span>
                <span className="dim">touched {dateLabel(question.last_touched)}</span>
                {question.resolved_at === null ? null : (
                  <span className="info">resolved {dateLabel(question.resolved_at)}</span>
                )}
                {question.abandoned_at === null ? null : (
                  <span className="warn">abandoned {dateLabel(question.abandoned_at)}</span>
                )}
                {question.last_ruminated_at === null ? null : (
                  <span className="purple">bumped {dateLabel(question.last_ruminated_at)}</span>
                )}
                <span style={{ flex: 1 }}></span>
                <button
                  className="btn sm"
                  disabled={busy !== null || question.status !== "open"}
                  onClick={() => setModal({ kind: "question-resolve", question, text: "" })}
                >
                  resolve
                </button>
                <button
                  className="btn sm ghost"
                  disabled={busy !== null || question.status !== "open"}
                  onClick={() => setModal({ kind: "question-abandon", question, text: "" })}
                >
                  abandon
                </button>
                <button
                  className="btn sm ghost"
                  disabled={busy !== null || question.status !== "open"}
                  onClick={() =>
                    void runAction("question-bump", async () => {
                      await patchOpenQuestion(question.id, { action: "bump" });
                    })
                  }
                >
                  bump
                </button>
                <IdentityCorrectionButtons
                  busy={busy !== null}
                  id={question.id}
                  label={question.question}
                  patch={questionPatch(question)}
                  onWhy={setWhyId}
                  onModal={setModal}
                />
              </div>
              <div className="bar-meter" style={{ marginTop: 4 }}>
                <div
                  className={`fill ${question.urgency > 0.6 ? "warn" : ""}`}
                  style={{ width: `${clamp01(question.urgency) * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 5" }}>
        <div className="h">
          <span className="ttl">traits</span>
          <span className="n">{identity.traits.length}</span>
        </div>
        <div className="body">
          {identity.traits.map((trait) => (
            <div
              key={trait.id}
              className="item"
              style={{ display: "flex", justifyContent: "space-between", gap: 10 }}
            >
              <span
                style={{
                  color: "var(--text-dim)",
                  fontFamily: "var(--sans)",
                  fontSize: 12.5,
                  lineHeight: 1.5,
                }}
              >
                {trait.label}
              </span>
              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                <span className="dim" style={{ fontSize: 10.5, whiteSpace: "nowrap" }}>
                  {trait.support_count} obs
                </span>
                <IdentityCorrectionButtons
                  busy={busy !== null}
                  id={trait.id}
                  label={trait.label}
                  patch={traitPatch(trait)}
                  onWhy={setWhyId}
                  onModal={setModal}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 7" }}>
        <div className="h">
          <span className="ttl">growth markers</span>
          <span className="n">{identity.growth_markers.length}</span>
          <span style={{ flex: 1 }}></span>
          <button
            className="btn sm ghost"
            disabled={busy !== null}
            aria-label="add growth marker"
            onClick={() => setModal({ kind: "growth", description: "", source: "" })}
          >
            + add
          </button>
        </div>
        <div className="body">
          <div className="timeline">
            {identity.growth_markers.map((marker) => (
              <div key={marker.id} className={`ev ${marker.confidence < 0.6 ? "warn" : ""}`}>
                <div className="t">
                  {dateLabel(marker.ts)} · {marker.source_process}
                </div>
                <div className="x">{marker.what_changed}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="id-card" style={{ gridColumn: "span 12" }}>
        <div className="h">
          <span className="ttl">autobiographical periods</span>
          <span className="n">{identity.periods.length}</span>
        </div>
        <div className="body" style={{ padding: 0 }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${Math.max(identity.periods.length, 1)}, 1fr)`,
              borderBottom: "1px solid var(--line)",
            }}
          >
            {identity.periods.map((period, index) => {
              const current = period.id === currentPeriod?.id;
              return (
                <div
                  key={period.id}
                  style={{
                    padding: 14,
                    borderRight:
                      index === identity.periods.length - 1 ? "0" : "1px solid var(--line)",
                    background: current ? "oklch(0.84 0.155 142 / 0.05)" : "transparent",
                  }}
                >
                  <div
                    style={{
                      fontSize: 10,
                      textTransform: "uppercase",
                      letterSpacing: "0.1em",
                      color: current ? "var(--acc)" : "var(--text-mute)",
                      marginBottom: 6,
                    }}
                  >
                    period {index + 1}
                    {current ? " · current" : ""}
                  </div>
                  <div
                    style={{ color: "var(--text)", fontSize: 14, fontWeight: 500, marginBottom: 4 }}
                  >
                    {period.label}
                  </div>
                  <div className="dim" style={{ fontSize: 10.5, marginBottom: 8 }}>
                    {dateLabel(period.start_ts)} to{" "}
                    {period.end_ts === null ? "present" : dateLabel(period.end_ts)}
                  </div>
                  <div
                    style={{
                      color: "var(--text-dim)",
                      fontFamily: "var(--sans)",
                      fontSize: 12,
                      lineHeight: 1.55,
                    }}
                  >
                    {period.narrative}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <Modal
        open={modal !== null}
        title={modal === null ? "identity" : modal.kind.replace("-", " ")}
        onClose={() => setModal(null)}
        footer={
          <>
            <button
              className="btn sm ghost"
              disabled={busy !== null}
              onClick={() => setModal(null)}
            >
              cancel
            </button>
            <button
              className="btn sm primary"
              disabled={busy !== null || modal === null}
              onClick={() => void submitModal()}
            >
              {busy === null
                ? modal?.kind === "forget"
                  ? "forget"
                  : modal?.kind === "correct"
                    ? "queue"
                    : "save"
                : "saving"}
            </button>
          </>
        }
      >
        {modal?.kind === "value" ? (
          <div className="modal-form">
            <label className="modal-field">
              <span>name</span>
              <input
                value={modal.name}
                onChange={(event) => setModal({ ...modal, name: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>description</span>
              <textarea
                value={modal.description}
                onChange={(event) => setModal({ ...modal, description: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "goal" ? (
          <div className="modal-form">
            <label className="modal-field">
              <span>description</span>
              <textarea
                value={modal.description}
                onChange={(event) => setModal({ ...modal, description: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>priority</span>
              <input
                type="number"
                step="0.1"
                value={modal.priority}
                onChange={(event) => setModal({ ...modal, priority: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "growth" ? (
          <div className="modal-form">
            <label className="modal-field">
              <span>description</span>
              <textarea
                value={modal.description}
                onChange={(event) => setModal({ ...modal, description: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>source</span>
              <input
                value={modal.source}
                onChange={(event) => setModal({ ...modal, source: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "goal-progress" ? (
          <div className="modal-form">
            <div className="dim">{modal.goal.description}</div>
            <label className="modal-field">
              <span>note</span>
              <textarea
                value={modal.note}
                onChange={(event) => setModal({ ...modal, note: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>progress</span>
              <input
                type="number"
                min="0"
                max="100"
                value={modal.progress}
                onChange={(event) => setModal({ ...modal, progress: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "question-resolve" ? (
          <div className="modal-form">
            <div className="dim">{modal.question.question}</div>
            <label className="modal-field">
              <span>resolution</span>
              <textarea
                value={modal.text}
                onChange={(event) => setModal({ ...modal, text: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "question-abandon" ? (
          <div className="modal-form">
            <div className="dim">{modal.question.question}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                value={modal.text}
                onChange={(event) => setModal({ ...modal, text: event.target.value })}
              />
            </label>
          </div>
        ) : null}
        {modal?.kind === "forget" ? (
          <div className="modal-form">
            <div className="dim">{modal.label}</div>
          </div>
        ) : null}
        {modal?.kind === "correct" ? (
          <div className="modal-form">
            <div className="dim">{modal.label}</div>
            <label className="modal-field">
              <span>reason</span>
              <textarea
                value={modal.reason}
                onChange={(event) => setModal({ ...modal, reason: event.target.value })}
              />
            </label>
            <label className="modal-field">
              <span>json patch</span>
              <textarea
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
