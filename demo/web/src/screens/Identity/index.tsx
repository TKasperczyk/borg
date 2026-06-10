import { useMemo, useState, type ReactNode } from "react";

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
import type {
  AutobiographicalPeriod,
  GrowthMarker,
  IdentityEvent,
  IdentityGoal,
  IdentityTrait,
  IdentityValue,
  OpenQuestion,
} from "../../api/types";
import { Empty } from "../../components/Empty";
import { IdRef } from "../../components/Inspector/IdRef";
import { Modal } from "../../components/Modal";
import { OpenQuestionEventsSection } from "../../components/OpenQuestionEventsSection";
import { Tag } from "../../components/Tag";
import { WhyDrawer } from "../../components/WhyDrawer";
import { useApi } from "../../hooks/use-api";
import { clamp01, dateLabel, parseJsonPatch, shortId } from "../screen-utils";

type QuestionFilter = "all" | OpenQuestion["status"];
type EventScope = "selected" | "all";
type FacetState = IdentityValue["state"];
type GoalStatus = IdentityGoal["status"];

type IdentityModal =
  | { kind: "value"; name: string; description: string }
  | { kind: "goal"; description: string; priority: string }
  | { kind: "growth"; description: string; source: string }
  | { kind: "goal-progress"; goal: IdentityGoal; note: string; progress: string }
  | { kind: "question-resolve"; question: OpenQuestion; text: string }
  | { kind: "question-abandon"; question: OpenQuestion; text: string }
  | { kind: "forget"; id: string; label: string }
  | { kind: "correct"; id: string; label: string; patch: string; reason: string };

type DirectCreateModal = Extract<IdentityModal, { kind: "value" | "goal" | "growth" }>;

const FACET_STATES: readonly FacetState[] = ["established", "candidate"];
const GOAL_STATUSES: readonly GoalStatus[] = ["active", "blocked", "done", "abandoned"];
const QUESTION_FILTERS: readonly QuestionFilter[] = ["all", "open", "resolved", "abandoned"];

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

function goalTag(status: GoalStatus) {
  if (status === "active") {
    return "acc";
  }
  if (status === "blocked") {
    return "warn";
  }
  if (status === "done") {
    return "info";
  }
  return "";
}

function isDirectCreateModal(modal: IdentityModal | null): modal is DirectCreateModal {
  return modal?.kind === "value" || modal?.kind === "goal" || modal?.kind === "growth";
}

function modalTitle(modal: IdentityModal | null): string {
  if (modal === null) {
    return "identity";
  }
  if (isDirectCreateModal(modal)) {
    return `operator-authored ${modal.kind}`;
  }
  return modal.kind.replace("-", " ");
}

function latestPeriod(periods: readonly AutobiographicalPeriod[]): AutobiographicalPeriod | null {
  return periods.reduce<AutobiographicalPeriod | null>((latest, period) => {
    if (latest === null || period.start_ts > latest.start_ts) {
      return period;
    }
    return latest;
  }, null);
}

function latestOpenQuestionEvent(events: readonly IdentityEvent[]): IdentityEvent | null {
  return events.reduce<IdentityEvent | null>((latest, event) => {
    if (
      latest === null ||
      event.ts > latest.ts ||
      (event.ts === latest.ts && event.id > latest.id)
    ) {
      return event;
    }
    return latest;
  }, null);
}

function formatMetric(value: number): string {
  return value.toFixed(2);
}

function DirectPatchLabel() {
  return (
    <Tag kind="warn" dot>
      writes live self-band
    </Tag>
  );
}

function DirectPatchNotice() {
  return (
    <div className="identity-live-write-note">
      <DirectPatchLabel />
      <span>direct patch -- writes the being's live self-band now</span>
    </div>
  );
}

function DirectWriteNotice({
  acknowledged,
  onAcknowledged,
}: {
  acknowledged: boolean;
  onAcknowledged: (acknowledged: boolean) => void;
}) {
  return (
    <div className="identity-live-write-warning">
      <div>
        <Tag kind="warn" dot>
          operator-authored
        </Tag>
      </div>
      <div>
        This writes the being's live self-band immediately. It is not self-derived and does not
        enter the correction review queue.
      </div>
      <label className="identity-ack">
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(event) => onAcknowledged(event.target.checked)}
        />
        <span>I acknowledge this direct live self-band write.</span>
      </label>
    </div>
  );
}

function IdentityEmptyBand({
  title,
  count,
  gridColumn = "span 4",
  children,
}: {
  title: string;
  count: number;
  gridColumn?: string;
  children: string;
}) {
  return (
    <div className="id-card identity-empty-card" style={{ gridColumn }}>
      <div className="h">
        <span className="ttl">{title}</span>
        <span className="n">{count}</span>
      </div>
      <div className="identity-empty-band" aria-label={`empty ${title}`}>
        <Empty>{children}</Empty>
      </div>
    </div>
  );
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
        className="btn sm danger"
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

function SnapshotStat({
  label,
  value,
  sub,
}: {
  label: string;
  value: number | string;
  sub: string;
}) {
  return (
    <div className="mini-stat identity-snapshot-stat" aria-label={`snapshot ${label}`}>
      <div className="k">{label}</div>
      <div className="v tab-num">{value}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}

function EvidenceRefs({ ids, label = "evidence" }: { ids: readonly string[]; label?: string }) {
  return (
    <div className="identity-idrefs">
      <span className="dim">{label}</span>
      {ids.length === 0 ? <span className="dim">none</span> : null}
      {ids.map((id) => (
        <IdRef key={id} id={id} label={shortId(id)} title={id} />
      ))}
    </div>
  );
}

function FieldRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="identity-field-row">
      <span className="k">{label}</span>
      <span className="v">{children}</span>
    </div>
  );
}

function FacetMetric({ label, value }: { label: string; value: number | string }) {
  return (
    <span className="identity-metric">
      <span>{label}</span>
      <strong className="tab-num">{value}</strong>
    </span>
  );
}

function ValueCard({
  value,
  busy,
  onWhy,
  onModal,
}: {
  value: IdentityValue;
  busy: boolean;
  onWhy: (id: string) => void;
  onModal: (modal: IdentityModal) => void;
}) {
  return (
    <div className="identity-record-card" data-testid="identity-value-card">
      <div className="identity-card-head">
        <div>
          <div className="identity-card-title">{value.label}</div>
          <div className="identity-card-copy">{value.description}</div>
        </div>
        <Tag kind={value.state === "established" ? "acc" : "warn"}>{value.state}</Tag>
      </div>
      <div className="identity-metric-row">
        <FacetMetric label="priority" value={formatMetric(value.priority)} />
        <FacetMetric label="confidence" value={formatMetric(value.confidence)} />
        <FacetMetric label="support" value={value.support_count} />
        <FacetMetric label="contradictions" value={value.contradiction_count} />
        {typeof value.last_affirmed === "number" ? (
          <FacetMetric label="last affirmed" value={dateLabel(value.last_affirmed)} />
        ) : null}
      </div>
      <div className="bar-meter" aria-label={`${value.label} confidence`}>
        <div className="fill" style={{ width: `${clamp01(value.confidence) * 100}%` }}></div>
      </div>
      <EvidenceRefs ids={value.evidence_episode_ids} />
      <div className="identity-card-actions">
        <IdentityCorrectionButtons
          busy={busy}
          id={value.id}
          label={value.label}
          patch={valuePatch(value)}
          onWhy={onWhy}
          onModal={onModal}
        />
      </div>
    </div>
  );
}

function TraitCard({
  trait,
  busy,
  onWhy,
  onModal,
}: {
  trait: IdentityTrait;
  busy: boolean;
  onWhy: (id: string) => void;
  onModal: (modal: IdentityModal) => void;
}) {
  return (
    <div className="identity-record-card" data-testid="identity-trait-card">
      <div className="identity-card-head">
        <div className="identity-card-title">{trait.label}</div>
        <Tag kind={trait.state === "established" ? "acc" : "warn"}>{trait.state}</Tag>
      </div>
      <div className="identity-metric-row">
        {typeof trait.strength === "number" ? (
          <FacetMetric label="strength" value={formatMetric(trait.strength)} />
        ) : null}
        <FacetMetric label="confidence" value={formatMetric(trait.confidence)} />
        <FacetMetric label="support" value={trait.support_count} />
        <FacetMetric label="contradictions" value={trait.contradiction_count} />
      </div>
      <div className="bar-meter" aria-label={`${trait.label} confidence`}>
        <div className="fill purple" style={{ width: `${clamp01(trait.confidence) * 100}%` }}></div>
      </div>
      <EvidenceRefs ids={trait.evidence_episode_ids} />
      <div className="identity-card-actions">
        <IdentityCorrectionButtons
          busy={busy}
          id={trait.id}
          label={trait.label}
          patch={traitPatch(trait)}
          onWhy={onWhy}
          onModal={onModal}
        />
      </div>
    </div>
  );
}

function GoalCard({
  goal,
  busy,
  onPatch,
  onWhy,
  onModal,
}: {
  goal: IdentityGoal;
  busy: boolean;
  onPatch: (label: string, action: () => Promise<void>) => void;
  onWhy: (id: string) => void;
  onModal: (modal: IdentityModal) => void;
}) {
  const canDirectPatch = goal.status === "active";

  return (
    <div className="identity-record-card" data-testid="identity-goal-card">
      <div className="identity-card-head">
        <div className="identity-card-title">{goal.description}</div>
        <Tag kind={goalTag(goal.status)} dot>
          {goal.status}
        </Tag>
      </div>
      <div className="identity-metric-row">
        <FacetMetric label="priority" value={formatMetric(goal.priority)} />
        <FacetMetric label="created" value={dateLabel(goal.created_at)} />
        <FacetMetric label="target" value={dateLabel(goal.target_at)} />
      </div>
      <FieldRow label="progress notes">{goal.progress_notes ?? "-"}</FieldRow>
      <div className="identity-card-actions">
        {canDirectPatch ? (
          <>
            <DirectPatchLabel />
            <button
              className="btn sm"
              disabled={busy}
              onClick={() =>
                onPatch("goal-complete", async () => {
                  await patchGoal(goal.id, { action: "complete" });
                })
              }
            >
              complete
            </button>
            <button
              className="btn sm ghost"
              disabled={busy}
              onClick={() =>
                onPatch("goal-block", async () => {
                  await patchGoal(goal.id, { action: "block" });
                })
              }
            >
              block
            </button>
            <button
              className="btn sm ghost"
              disabled={busy}
              onClick={() => onModal({ kind: "goal-progress", goal, note: "", progress: "" })}
            >
              progress
            </button>
          </>
        ) : null}
        <IdentityCorrectionButtons
          busy={busy}
          id={goal.id}
          label={goal.description}
          patch={goalPatch(goal)}
          onWhy={onWhy}
          onModal={onModal}
        />
      </div>
    </div>
  );
}

function OpenQuestionQueueItem({
  question,
  selected,
  onSelect,
}: {
  question: OpenQuestion;
  selected: boolean;
  onSelect: (id: string) => void;
}) {
  return (
    <button
      type="button"
      className={`identity-queue-item ${selected ? "selected" : ""}`}
      aria-label={`select open question ${shortId(question.id)}`}
      aria-current={selected ? "true" : undefined}
      onClick={() => onSelect(question.id)}
    >
      <span className="identity-queue-question">{question.question}</span>
      <span className="identity-queue-meta">
        <Tag kind={questionTag(question.status)} dot>
          {question.status}
        </Tag>
        <span className="tab-num">urg {formatMetric(question.urgency)}</span>
        <span>{dateLabel(question.last_touched)}</span>
      </span>
    </button>
  );
}

function OpenQuestionDetail({
  question,
  emptyLabel,
  busy,
  onPatch,
  onWhy,
  onModal,
}: {
  question: OpenQuestion | null;
  emptyLabel: string;
  busy: boolean;
  onPatch: (label: string, action: () => Promise<void>) => void;
  onWhy: (id: string) => void;
  onModal: (modal: IdentityModal) => void;
}) {
  if (question === null) {
    return (
      <div className="identity-empty-band">
        <Empty>{emptyLabel}</Empty>
      </div>
    );
  }

  return (
    <div className="identity-question-detail" aria-label="selected open question detail">
      <div className="identity-card-head">
        <div>
          <div className="identity-card-title">{question.question}</div>
          <div className="identity-card-copy">
            <IdRef id={question.id} label={shortId(question.id)} title={question.id} />
          </div>
        </div>
        <Tag kind={questionTag(question.status)} dot>
          {question.status}
        </Tag>
      </div>
      <div className="identity-question-meter">
        <span className="dim">urgency</span>
        <div className="bar-meter">
          <div
            className={`fill ${question.urgency > 0.6 ? "warn" : ""}`}
            style={{ width: `${clamp01(question.urgency) * 100}%` }}
          ></div>
        </div>
        <span className="tab-num">{formatMetric(question.urgency)}</span>
      </div>
      <div className="identity-detail-grid">
        <FieldRow label="goal">
          {question.goal_id === null ? (
            "-"
          ) : (
            <IdRef
              id={question.goal_id}
              label={shortId(question.goal_id)}
              title={question.goal_id}
            />
          )}
        </FieldRow>
        <FieldRow label="source">{question.source}</FieldRow>
        <FieldRow label="created">{dateLabel(question.created_at)}</FieldRow>
        <FieldRow label="last touched">{dateLabel(question.last_touched)}</FieldRow>
        <FieldRow label="rumination ticks">{question.unresolved_rumination_ticks}</FieldRow>
        <FieldRow label="last ruminated">{dateLabel(question.last_ruminated_at)}</FieldRow>
        {question.resolution_note === null ? null : (
          <FieldRow label="resolution">{question.resolution_note}</FieldRow>
        )}
        {question.abandoned_reason === null ? null : (
          <FieldRow label="abandoned reason">{question.abandoned_reason}</FieldRow>
        )}
      </div>
      <div className="identity-card-actions">
        {question.status === "open" ? (
          <>
            <DirectPatchLabel />
            <button
              className="btn sm"
              disabled={busy}
              onClick={() => onModal({ kind: "question-resolve", question, text: "" })}
            >
              resolve
            </button>
            <button
              className="btn sm ghost"
              disabled={busy}
              onClick={() => onModal({ kind: "question-abandon", question, text: "" })}
            >
              abandon
            </button>
            <button
              className="btn sm ghost"
              disabled={busy}
              onClick={() =>
                onPatch("question-bump", async () => {
                  await patchOpenQuestion(question.id, { action: "bump" });
                })
              }
            >
              bump
            </button>
          </>
        ) : null}
        <IdentityCorrectionButtons
          busy={busy}
          id={question.id}
          label={question.question}
          patch={questionPatch(question)}
          onWhy={onWhy}
          onModal={onModal}
        />
      </div>
    </div>
  );
}

function GrowthMarkerEvent({ marker }: { marker: GrowthMarker }) {
  return (
    <div className={`ev ${marker.confidence < 0.6 ? "warn" : ""}`} data-testid="growth-marker-row">
      <div className="identity-event-head">
        <Tag kind="purple">{marker.category}</Tag>
        <span className="dim tab-num">{dateLabel(marker.ts)}</span>
        <span className="dim">{marker.source_process}</span>
      </div>
      <div className="x">{marker.what_changed}</div>
      {marker.before_description === null && marker.after_description === null ? null : (
        <div className="identity-before-after">
          <span>{marker.before_description ?? "-"}</span>
          <span className="dim"> -&gt; </span>
          <span>{marker.after_description ?? "-"}</span>
        </div>
      )}
      <div className="identity-metric-row">
        <FacetMetric label="confidence" value={formatMetric(marker.confidence)} />
      </div>
      <EvidenceRefs ids={marker.evidence_episode_ids} />
    </div>
  );
}

function PeriodEvent({ period, current }: { period: AutobiographicalPeriod; current: boolean }) {
  return (
    <div className={`ev ${current ? "" : "warn"}`} data-testid="autobiographical-period-row">
      <div className="identity-event-head">
        <Tag kind={current ? "acc" : ""}>{current ? "current" : "period"}</Tag>
        <span className="dim tab-num">
          {dateLabel(period.start_ts)} to{" "}
          {period.end_ts === null ? "present" : dateLabel(period.end_ts)}
        </span>
      </div>
      <div className="identity-card-title">{period.label}</div>
      <div className="identity-card-copy">{period.narrative}</div>
      {period.themes.length === 0 ? null : (
        <div className="identity-chip-row">
          {period.themes.map((theme) => (
            <Tag key={theme} kind="info">
              {theme}
            </Tag>
          ))}
        </div>
      )}
      <EvidenceRefs ids={period.key_episode_ids} label="key episodes" />
    </div>
  );
}

export function IdentityScreen() {
  const api = useApi(getIdentity, []);
  const [questionFilter, setQuestionFilter] = useState<QuestionFilter>("all");
  const [selectedQuestionId, setSelectedQuestionId] = useState<string | null>(null);
  const [eventScope, setEventScope] = useState<EventScope>("selected");
  const [modal, setModal] = useState<IdentityModal | null>(null);
  const [whyId, setWhyId] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const [directWriteAcknowledged, setDirectWriteAcknowledged] = useState(false);
  const identity = api.data;

  const valueCounts = useMemo(() => {
    const values = identity?.values ?? [];
    return {
      established: values.filter((value) => value.state === "established").length,
      candidate: values.filter((value) => value.state === "candidate").length,
    };
  }, [identity?.values]);

  const traitCounts = useMemo(() => {
    const traits = identity?.traits ?? [];
    return {
      established: traits.filter((trait) => trait.state === "established").length,
      candidate: traits.filter((trait) => trait.state === "candidate").length,
    };
  }, [identity?.traits]);

  const goalCounts = useMemo(() => {
    const goals = identity?.goals ?? [];
    return {
      active: goals.filter((goal) => goal.status === "active").length,
      blocked: goals.filter((goal) => goal.status === "blocked").length,
      done: goals.filter((goal) => goal.status === "done").length,
      abandoned: goals.filter((goal) => goal.status === "abandoned").length,
    };
  }, [identity?.goals]);

  const questionCounts = useMemo(() => {
    const questions = identity?.open_questions ?? [];
    return {
      open: questions.filter((question) => question.status === "open").length,
      resolved: questions.filter((question) => question.status === "resolved").length,
      abandoned: questions.filter((question) => question.status === "abandoned").length,
    };
  }, [identity?.open_questions]);

  const queuedQuestions = useMemo(() => {
    const all = identity?.open_questions ?? [];
    const filtered =
      questionFilter === "all" ? all : all.filter((question) => question.status === questionFilter);
    return [...filtered].sort(
      (left, right) =>
        right.urgency - left.urgency ||
        right.last_touched - left.last_touched ||
        right.created_at - left.created_at,
    );
  }, [identity?.open_questions, questionFilter]);

  const selectedQuestion =
    queuedQuestions.find((question) => question.id === selectedQuestionId) ??
    queuedQuestions[0] ??
    null;
  const questionEmptyLabel =
    identity?.open_questions.length === 0 ? "no questions yet" : "none match this filter";

  const scopedEvents =
    eventScope === "all"
      ? (identity?.open_question_events ?? [])
      : (identity?.open_question_events ?? []).filter(
          (event) => selectedQuestion !== null && event.record_id === selectedQuestion.id,
        );

  const currentPeriod = latestPeriod(identity?.periods ?? []);
  const recentEvent = latestOpenQuestionEvent(identity?.open_question_events ?? []);

  async function runAction(label: string, action: () => Promise<void>): Promise<void> {
    setBusy(label);
    setOperatorError(null);
    try {
      await action();
      await api.refetch();
      setModal(null);
      setDirectWriteAcknowledged(false);
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  function openModal(nextModal: IdentityModal): void {
    setDirectWriteAcknowledged(false);
    setModal(nextModal);
  }

  function closeModal(): void {
    if (busy !== null) {
      return;
    }
    setDirectWriteAcknowledged(false);
    setModal(null);
  }

  function canSubmitModal(): boolean {
    if (modal === null || busy !== null) {
      return false;
    }
    return !isDirectCreateModal(modal) || directWriteAcknowledged;
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
    <div className="identity identity-studio">
      <div className="id-hero identity-snapshot" aria-label="self snapshot header">
        <div className="identity-snapshot-main">
          <div className="id-eyebrow">self::snapshot</div>
          <div className="stamp">Identity Studio</div>
          <div className="quote">
            values, goals, traits, open questions, growth markers, and autobiography.
          </div>
          <div className="quote-attr">
            current autobiographical period:{" "}
            <span className="acc">{currentPeriod?.label ?? "none"}</span> ·{" "}
            {dateLabel(currentPeriod?.start_ts)}
          </div>
          <div className="identity-recent-event" aria-label="recent open-question event">
            <span className="id-eyebrow">recent open-question event</span>
            {recentEvent === null ? (
              <span className="dim">none</span>
            ) : (
              <span>
                <Tag kind={recentEvent.action === "resolve" ? "acc" : "info"}>
                  {recentEvent.action}
                </Tag>
                <IdRef
                  id={recentEvent.record_id}
                  label={shortId(recentEvent.record_id)}
                  title={recentEvent.record_id}
                />
                <span className="dim">{dateLabel(recentEvent.ts)}</span>
              </span>
            )}
          </div>
        </div>
        <div className="identity-snapshot-grid">
          <SnapshotStat
            label="values"
            value={identity.values.length}
            sub={`${valueCounts.established} established · ${valueCounts.candidate} candidate`}
          />
          <SnapshotStat
            label="goals"
            value={identity.goals.length}
            sub={`${goalCounts.active} active · ${goalCounts.blocked} blocked · ${goalCounts.done} done · ${goalCounts.abandoned} abandoned`}
          />
          <SnapshotStat
            label="traits"
            value={identity.traits.length}
            sub={`${traitCounts.established} established · ${traitCounts.candidate} candidate`}
          />
          <SnapshotStat
            label="open questions"
            value={identity.open_questions.length}
            sub={`${questionCounts.open} open · ${questionCounts.resolved} resolved · ${questionCounts.abandoned} abandoned`}
          />
          <SnapshotStat
            label="growth markers"
            value={identity.growth_markers.length}
            sub="evidence-backed changes"
          />
          <SnapshotStat
            label="periods"
            value={identity.periods.length}
            sub={`latest starts ${dateLabel(currentPeriod?.start_ts)}`}
          />
        </div>
      </div>

      {operatorError === null ? null : (
        <div className="notice bad" style={{ gridColumn: "span 12", padding: 12 }}>
          {operatorError}
        </div>
      )}

      <div
        className="id-card identity-values-board"
        style={{ gridColumn: "span 6" }}
        aria-label="values board"
      >
        <div className="h">
          <span className="ttl">values board</span>
          <span className="n">{identity.values.length}</span>
          <span style={{ flex: 1 }}></span>
          <button
            className="btn sm live-write"
            disabled={busy !== null}
            aria-label="add value"
            onClick={() => openModal({ kind: "value", name: "", description: "" })}
          >
            + value
          </button>
        </div>
        <div className="body">
          <div className="identity-state-board">
            {FACET_STATES.map((state) => {
              const values = identity.values.filter((value) => value.state === state);
              return (
                <div key={state} className="identity-state-column" aria-label={`${state} values`}>
                  <div className="identity-lane-head">
                    <span>{state}</span>
                    <span className="n">{values.length}</span>
                  </div>
                  {values.map((value) => (
                    <ValueCard
                      key={value.id}
                      value={value}
                      busy={busy !== null}
                      onWhy={setWhyId}
                      onModal={openModal}
                    />
                  ))}
                  {values.length === 0 ? (
                    <div className="identity-empty-band">
                      <Empty>no {state} values</Empty>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div
        className="id-card identity-traits-board"
        style={{ gridColumn: "span 6" }}
        aria-label="traits board"
      >
        <div className="h">
          <span className="ttl">traits board</span>
          <span className="n">{identity.traits.length}</span>
        </div>
        <div className="body">
          <div className="identity-state-board">
            {FACET_STATES.map((state) => {
              const traits = identity.traits.filter((trait) => trait.state === state);
              return (
                <div key={state} className="identity-state-column" aria-label={`${state} traits`}>
                  <div className="identity-lane-head">
                    <span>{state}</span>
                    <span className="n">{traits.length}</span>
                  </div>
                  {traits.map((trait) => (
                    <TraitCard
                      key={trait.id}
                      trait={trait}
                      busy={busy !== null}
                      onWhy={setWhyId}
                      onModal={openModal}
                    />
                  ))}
                  {traits.length === 0 ? (
                    <div className="identity-empty-band">
                      <Empty>no {state} traits</Empty>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div
        className="id-card identity-goals-kanban"
        style={{ gridColumn: "span 12" }}
        aria-label="goals kanban"
      >
        <div className="h">
          <span className="ttl">goals kanban</span>
          <span className="n">{identity.goals.length}</span>
          <span style={{ flex: 1 }}></span>
          <button
            className="btn sm live-write"
            disabled={busy !== null}
            aria-label="add goal"
            onClick={() => openModal({ kind: "goal", description: "", priority: "" })}
          >
            + goal
          </button>
        </div>
        <div className="body">
          <div className="identity-kanban">
            {GOAL_STATUSES.map((status) => {
              const goals = identity.goals.filter((goal) => goal.status === status);
              return (
                <div key={status} className="identity-kanban-lane" aria-label={`${status} goals`}>
                  <div className="identity-lane-head">
                    <span>{status}</span>
                    <span className="n">{goals.length}</span>
                  </div>
                  {goals.map((goal) => (
                    <GoalCard
                      key={goal.id}
                      goal={goal}
                      busy={busy !== null}
                      onPatch={(label, action) => void runAction(label, action)}
                      onWhy={setWhyId}
                      onModal={openModal}
                    />
                  ))}
                  {goals.length === 0 ? (
                    <div className="identity-empty-band">
                      <Empty>no {status} goals</Empty>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <section
        className="identity-region identity-open-questions-region"
        aria-label="open questions queue"
        style={{ gridColumn: "span 12" }}
      >
        <div className="identity-region-head">
          <div>
            <div className="id-eyebrow">open questions</div>
            <h2>queue</h2>
          </div>
          <div className="filter-pills" aria-label="open question status filter">
            {QUESTION_FILTERS.map((filter) => (
              <button
                key={filter}
                type="button"
                className={`pill ${questionFilter === filter ? "on" : ""}`}
                onClick={() => setQuestionFilter(filter)}
              >
                {filter}
              </button>
            ))}
          </div>
        </div>
        <div className="identity-question-layout">
          <div
            className="id-card identity-question-queue-card"
            aria-label="open question queue list"
          >
            <div className="h">
              <span className="ttl">urgency queue</span>
              <span className="n">{queuedQuestions.length}</span>
            </div>
            <div className="body identity-queue-list">
              {queuedQuestions.map((question) => (
                <OpenQuestionQueueItem
                  key={question.id}
                  question={question}
                  selected={selectedQuestion?.id === question.id}
                  onSelect={setSelectedQuestionId}
                />
              ))}
              {queuedQuestions.length === 0 ? (
                <div className="identity-empty-band">
                  <Empty>{questionEmptyLabel}</Empty>
                </div>
              ) : null}
            </div>
          </div>
          <div
            className="id-card identity-question-detail-card"
            aria-label="open question selected record"
          >
            <div className="h">
              <span className="ttl">selected question</span>
              <span className="n">
                {selectedQuestion === null ? "none" : shortId(selectedQuestion.id)}
              </span>
            </div>
            <div className="body">
              <OpenQuestionDetail
                question={selectedQuestion}
                emptyLabel={questionEmptyLabel}
                busy={busy !== null}
                onPatch={(label, action) => void runAction(label, action)}
                onWhy={setWhyId}
                onModal={openModal}
              />
            </div>
          </div>
          <div className="identity-events-toolbar">
            <div className="id-eyebrow">events timeline</div>
            <div className="filter-pills" aria-label="open question events scope">
              {(["selected", "all"] as const).map((scope) => (
                <button
                  key={scope}
                  type="button"
                  className={`pill ${eventScope === scope ? "on" : ""}`}
                  onClick={() => setEventScope(scope)}
                >
                  {scope === "all" ? "all events" : "selected question"}
                </button>
              ))}
            </div>
          </div>
          <OpenQuestionEventsSection
            events={scopedEvents}
            title={eventScope === "all" ? "all open question events" : "selected question events"}
            emptyLabel={
              selectedQuestion === null
                ? "no selected question events"
                : `no events for ${shortId(selectedQuestion.id)}`
            }
            gridColumn="span 12"
          />
        </div>
      </section>

      <section
        className="identity-region identity-growth-region"
        aria-label="growth and autobiography timeline"
        style={{ gridColumn: "span 12" }}
      >
        <div className="identity-region-head">
          <div>
            <div className="id-eyebrow">growth and autobiography</div>
            <h2>timeline</h2>
          </div>
          <button
            className="btn sm live-write"
            disabled={busy !== null}
            aria-label="add growth marker"
            onClick={() => openModal({ kind: "growth", description: "", source: "" })}
          >
            + growth
          </button>
        </div>
        <div className="identity-timeline-layout">
          {identity.growth_markers.length === 0 ? (
            <IdentityEmptyBand
              title="growth markers"
              count={identity.growth_markers.length}
              gridColumn="auto"
            >
              no growth markers recorded
            </IdentityEmptyBand>
          ) : (
            <div className="id-card" aria-label="growth markers timeline">
              <div className="h">
                <span className="ttl">growth markers</span>
                <span className="n">{identity.growth_markers.length}</span>
              </div>
              <div className="body">
                <div className="timeline identity-growth-timeline">
                  {identity.growth_markers.map((marker) => (
                    <GrowthMarkerEvent key={marker.id} marker={marker} />
                  ))}
                </div>
              </div>
            </div>
          )}
          {identity.periods.length === 0 ? (
            <IdentityEmptyBand
              title="autobiographical periods"
              count={identity.periods.length}
              gridColumn="auto"
            >
              no autobiographical periods recorded
            </IdentityEmptyBand>
          ) : (
            <div className="id-card" aria-label="autobiographical periods timeline">
              <div className="h">
                <span className="ttl">autobiographical periods</span>
                <span className="n">{identity.periods.length}</span>
              </div>
              <div className="body">
                <div className="timeline identity-period-timeline">
                  {[...identity.periods]
                    .sort((left, right) => right.start_ts - left.start_ts)
                    .map((period) => (
                      <PeriodEvent
                        key={period.id}
                        period={period}
                        current={period.id === currentPeriod?.id}
                      />
                    ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      <Modal
        open={modal !== null}
        title={modalTitle(modal)}
        onClose={closeModal}
        footer={
          <>
            <span className="dim" style={{ fontSize: 10.5, marginRight: "auto" }}>
              {isDirectCreateModal(modal)
                ? "operator-authored live write"
                : modal?.kind === "correct"
                  ? "safe path: queues review"
                  : ""}
            </span>
            <button className="btn sm ghost" disabled={busy !== null} onClick={closeModal}>
              cancel
            </button>
            <button
              className={`btn sm ${
                modal?.kind === "forget"
                  ? "danger"
                  : `primary${isDirectCreateModal(modal) ? " live-write" : ""}`
              }`}
              disabled={!canSubmitModal()}
              onClick={() => void submitModal()}
            >
              {busy === null
                ? isDirectCreateModal(modal)
                  ? "write live self-band"
                  : modal?.kind === "forget"
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
            <DirectWriteNotice
              acknowledged={directWriteAcknowledged}
              onAcknowledged={setDirectWriteAcknowledged}
            />
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
            <DirectWriteNotice
              acknowledged={directWriteAcknowledged}
              onAcknowledged={setDirectWriteAcknowledged}
            />
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
            <DirectWriteNotice
              acknowledged={directWriteAcknowledged}
              onAcknowledged={setDirectWriteAcknowledged}
            />
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
            <DirectPatchNotice />
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
            <DirectPatchNotice />
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
            <DirectPatchNotice />
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
