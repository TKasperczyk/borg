import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  ApiError,
  deletePrompt,
  patchGoal,
  patchOpenQuestion,
  postCommitmentRevoke,
  postCorrectionCorrect,
  postCorrectionForget,
  postSemanticEdgeInvalidate,
  putPrompt,
  revertDreamAudit,
  revokeCreatorDirective,
  setSessionPolicy,
  supersedeCreatorDirective,
} from "../../api/client";
import type {
  AttachmentMetadataResponse,
  PromptKey,
  ReviewKind,
  ReviewResolution,
  ReviewRow,
  SemanticMemoryEdge,
  SemanticMemoryNode,
  SessionParticipationPolicy,
} from "../../api/types";
import { copyText } from "../../lib/clipboard";
import {
  DESTRUCTIVE_REVIEW_ACTIONS,
  GENERIC_REVIEW_ACTIONS,
  resolveReviewAction,
} from "../../lib/review-actions";
import { formatTimestamp, formatTimestampForKey } from "../../lib/stream-utils";
import { LedgerView } from "../../screens/Cognition/LedgerView";
import {
  displayValue,
  fieldLabel,
  isInternalId,
  isRecord,
  parseJsonPatch,
  shortId,
} from "../../screens/screen-utils";
import { AttachmentChip } from "../AttachmentChip";
import { DisclosureLabel } from "../DisclosureLabel";
import { Empty } from "../Empty";
import { ErrorState } from "../ErrorState";
import { JsonValueView } from "../JsonValueView";
import { Loading } from "../Loading";
import { Modal } from "../Modal";
import { ProvenanceEvidence } from "../ProvenanceEvidence";
import { SemanticEdgeDetail } from "../SemanticEdgeDetail";
import { SemanticNodeDetail } from "../SemanticNodeDetail";
import { Tag } from "../Tag";
import { IdChip } from "./IdChip";
import { IdRef } from "./IdRef";
import { useInspector, type InspectorTarget } from "./inspector-context";
import {
  INSPECTOR_TABS,
  isWhySupported,
  objectRegistry,
  type InspectorTab,
  type ObjectModel,
} from "./inspector-registry";
import type { ObjectType } from "./inspector-id";
import { useApi } from "../../hooks/use-api";

const PROMPT_KEYS: readonly PromptKey[] = [
  "base_identity_preamble",
  "self_architecture",
  "voice_and_posture",
  "epistemic_posture",
  "identity_posture",
  "participation_posture",
  "host_capabilities",
];

const SESSION_POLICIES: readonly SessionParticipationPolicy[] = [
  "active",
  "paused",
  "observing",
  "muted",
];

function isPromptKey(value: string): value is PromptKey {
  return PROMPT_KEYS.some((key) => key === value);
}

function tabLabel(tab: InspectorTab): string {
  if (tab === "raw") {
    return "Raw JSON";
  }
  return `${tab.slice(0, 1).toUpperCase()}${tab.slice(1)}`;
}

function objectUnavailableMessage(target: InspectorTarget, model: ObjectModel): string {
  if (model.reliability === "needs_backend") {
    return `${model.label} does not have a direct resolver for ${target.id}.`;
  }
  if (model.reliability === "in_list") {
    return `${model.label} is available only from the currently loaded list; ${target.id} was not found.`;
  }
  return `${model.label} was not found.`;
}

function errorMessage(error: Error, model: ObjectModel): string {
  if (error instanceof ApiError && error.status === 404) {
    return `${model.label} not retained or not found.`;
  }
  return error.message;
}

function timelineTimestamp(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

const TIMELINE_FIELDS: readonly [key: string, label: string][] = [
  ["created_at", "created"],
  ["updated_at", "updated"],
  ["last_updated_at", "last updated"],
  ["last_touched", "last touched"],
  ["last_reinforced_at", "last reinforced"],
  ["last_affirmed", "last affirmed"],
  ["last_ruminated_at", "last ruminated"],
  ["start_time", "started"],
  ["end_time", "ended"],
  ["valid_from", "valid from"],
  ["valid_to", "valid to"],
  ["resolved_at", "resolved"],
  ["abandoned_at", "abandoned"],
  ["revoked_at", "revoked"],
  ["expired_at", "expired"],
  ["applied_at", "applied"],
  ["reverted_at", "reverted"],
  ["timestamp", "timestamp"],
  ["ts", "timestamp"],
  ["started_at", "started"],
  ["last_activity_at", "last activity"],
];

type TimelineEvent = {
  key: string;
  label: string;
  ts: number;
};

function timelineEvents(value: unknown): TimelineEvent[] {
  if (!isRecord(value)) {
    return [];
  }

  const events: TimelineEvent[] = [];
  for (const [key, label] of TIMELINE_FIELDS) {
    const ts = timelineTimestamp(value[key]);
    if (ts !== null) {
      events.push({ key, label, ts });
    }
  }
  return events.sort((left, right) => left.ts - right.ts || left.key.localeCompare(right.key));
}

function TimestampLabel({ ts }: { ts: number }) {
  return <>{formatTimestamp(ts)}</>;
}

type SummaryDisclosureLabel = {
  key: string;
  value: unknown;
};

const SUMMARY_DISCLOSURE_LABEL_FIELDS = [
  "disclosure_label",
  "source_disclosure_label",
  "resolution_disclosure_label",
  "goal_disclosure_label",
] as const;

function disclosureClassFromMetadata(value: unknown): unknown {
  if (!isRecord(value)) {
    return undefined;
  }
  return value.disclosure_class;
}

function summaryDisclosureLabels(value: Record<string, unknown>): SummaryDisclosureLabel[] {
  const labels: SummaryDisclosureLabel[] = [];
  const primaryDisclosureClass = disclosureClassFromMetadata(value.disclosure_label);

  if (primaryDisclosureClass !== undefined) {
    labels.push({ key: "disclosure", value: primaryDisclosureClass });
  } else if (typeof value.disclosure_class === "string") {
    labels.push({ key: "disclosure", value: value.disclosure_class });
  }

  for (const field of SUMMARY_DISCLOSURE_LABEL_FIELDS) {
    if (field === "disclosure_label") {
      continue;
    }
    const disclosureClass = disclosureClassFromMetadata(value[field]);
    if (disclosureClass !== undefined) {
      labels.push({ key: fieldLabel(field), value: disclosureClass });
    }
  }

  return labels;
}

function GenericSummary({ value }: { value: unknown }) {
  if (!isRecord(value)) {
    return <JsonValueView value={value} />;
  }

  const entries = Object.entries(value);
  const labels = summaryDisclosureLabels(value);
  if (entries.length === 0 && labels.length === 0) {
    return <Empty>no summary fields</Empty>;
  }

  return (
    <div className="props">
      {labels.length === 0 ? null : (
        <div className="row">
          <span className="k">labels</span>
          <span className="v">
            {labels.map((label, index) => (
              <span key={`${label.key}:${index}`}>
                {index === 0 ? null : " "}
                <span className="dim">{label.key}</span> <DisclosureLabel value={label.value} />
              </span>
            ))}
          </span>
        </div>
      )}
      {entries.map(([key, entry]) => (
        <div className="row" key={key}>
          <span className="k">{fieldLabel(key)}</span>
          <span className="v">
            <SummaryValue fieldKey={key} value={entry} />
          </span>
        </div>
      ))}
    </div>
  );
}

function SummaryValue({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  const timestamp = formatTimestampForKey(fieldKey, value);
  if (timestamp !== null) {
    return <>{timestamp}</>;
  }

  if (typeof value === "string" && isInternalId(value)) {
    return <IdChip id={value} />;
  }

  if (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every((item) => typeof item === "string" && isInternalId(item))
  ) {
    return (
      <span className="id-chip-list">
        {value.map((id) => (
          <IdChip key={id} id={id} />
        ))}
      </span>
    );
  }

  return <>{displayValue(value)}</>;
}

function AttachmentSummary({ metadata }: { metadata: AttachmentMetadataResponse }) {
  return (
    <div className="inspector-section">
      <AttachmentChip
        attachmentId={metadata.attachment.attachment_id}
        mediaType={metadata.attachment.media_type}
        audience={metadata.attachment.audience ?? undefined}
        expanded
      />
      <div className="props">
        <div className="row">
          <span className="k">status</span>
          <span className="v">
            active {String(metadata.status.active)}, quarantined{" "}
            {String(metadata.status.quarantined)}
          </span>
        </div>
        <div className="row">
          <span className="k">byte size</span>
          <span className="v">{metadata.attachment.byte_size}</span>
        </div>
        <div className="row">
          <span className="k">perception</span>
          <span className="v">{metadata.perception?.perception_id ?? "none"}</span>
        </div>
      </div>
    </div>
  );
}

function SummaryTab({ target, data }: { target: InspectorTarget; data: unknown }) {
  if (target.type === "semantic_node" && isRecord(data)) {
    return <SemanticNodeDetail node={data as SemanticMemoryNode} />;
  }
  if (target.type === "semantic_edge" && isRecord(data)) {
    return <SemanticEdgeDetail edge={data as SemanticMemoryEdge} nodes={[]} />;
  }
  if (target.type === "attachment" && isRecord(data) && isRecord(data.attachment)) {
    return <AttachmentSummary metadata={data as AttachmentMetadataResponse} />;
  }

  return <GenericSummary value={data} />;
}

function EvidenceTab({ target, audience }: { target: InspectorTarget; audience: string | null }) {
  if (target.type === "turn") {
    return <LedgerView turnId={target.id} active audience={audience} />;
  }

  if (!isWhySupported(target.type)) {
    return <Empty>no evidence resolver for this object type</Empty>;
  }

  return <ProvenanceEvidence id={target.id} />;
}

function RelationshipsTab({ model, data }: { model: ObjectModel; data: unknown }) {
  const refs = model.pivots(data);

  if (refs.length === 0) {
    return <Empty>no schema-known related ids</Empty>;
  }

  return (
    <div className="inspector-rel-list">
      <div className="props">
        {refs.map((ref, index) => (
          <div className="row" key={`${ref.fieldLabel}:${ref.id}:${index}`}>
            <span className="k">{ref.fieldLabel}</span>
            <span className="v">
              <IdRef
                id={ref.id}
                type={ref.type}
                label={`${objectRegistry[ref.type].label} ${shortId(ref.id)}`}
              />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TimelineTab({ data }: { data: unknown }) {
  const events = timelineEvents(data);

  if (events.length === 0) {
    return <Empty>no lifecycle timestamps on this object</Empty>;
  }

  return (
    <div className="timeline inspector-timeline">
      {events.map((event) => (
        <div className="ev" key={event.key}>
          <div className="t">
            <TimestampLabel ts={event.ts} />
          </div>
          <div className="x">{event.label}</div>
        </div>
      ))}
    </div>
  );
}

function RawJsonTab({ data }: { data: unknown }) {
  if (data === null || data === undefined) {
    return <Empty>no raw JSON available</Empty>;
  }

  return <JsonValueView value={data} />;
}

type ActionTitle = {
  action: string;
  id: string | null;
  type: ObjectType | null;
};

type ConfirmAction = {
  title: ActionTitle;
  body: string;
  confirmLabel: string;
  reasonLabel?: string;
  requireReason?: boolean;
  danger?: boolean;
  run: (reason: string) => Promise<unknown>;
};

type TextAction = {
  title: ActionTitle;
  label: string;
  initialValue: string;
  requireValue?: boolean;
  trimValue?: boolean;
  danger?: boolean;
  run: (value: string) => Promise<unknown>;
};

function numericId(id: string): number | null {
  const value = Number(id);
  return Number.isFinite(value) ? value : null;
}

function optionalText(value: string): string | undefined {
  const trimmed = value.trim();
  return trimmed.length === 0 ? undefined : trimmed;
}

function reviewKind(value: unknown): ReviewKind | null {
  const kind = recordString(value, "kind");
  return kind !== null && Object.hasOwn(GENERIC_REVIEW_ACTIONS, kind) ? (kind as ReviewKind) : null;
}

function reviewRowForAction(value: unknown, id: number, kind: ReviewKind): ReviewRow {
  if (!isRecord(value)) {
    return {
      id,
      kind,
      refs: {},
      reason: "",
      created_at: 0,
      resolved_at: null,
      resolution: null,
    };
  }

  return {
    ...value,
    id,
    kind,
    refs: isRecord(value.refs) ? value.refs : {},
    reason: recordString(value, "reason") ?? "",
    created_at: typeof value.created_at === "number" ? value.created_at : 0,
    resolved_at: typeof value.resolved_at === "number" ? value.resolved_at : null,
    resolution:
      typeof value.resolution === "string" ? (value.resolution as ReviewResolution) : null,
  };
}

function reviewNodeIds(value: unknown): string[] {
  if (!isRecord(value) || !isRecord(value.refs) || !Array.isArray(value.refs.node_ids)) {
    return [];
  }
  return value.refs.node_ids.filter(
    (item): item is string => typeof item === "string" && item.length > 0,
  );
}

function reviewDirectiveIds(value: unknown): string[] {
  if (!isRecord(value) || !isRecord(value.refs) || !Array.isArray(value.refs.directive_ids)) {
    return [];
  }
  return value.refs.directive_ids.filter(
    (item): item is string => typeof item === "string" && item.length > 0,
  );
}

function reviewActionLabel(action: ReviewResolution): string {
  return action.replaceAll("_", " ");
}

function targetActionTitle(action: string, target: InspectorTarget): ActionTitle {
  return { action, id: target.id, type: target.type };
}

function ActionModalTitle({ title, fallback }: { title: ActionTitle | null; fallback: string }) {
  if (title === null) {
    return <span>{fallback}</span>;
  }

  return (
    <span className="identity-inline">
      <span>{title.action}</span>
      {title.id === null ? null : <IdChip id={title.id} type={title.type} />}
    </span>
  );
}

function recordString(value: unknown, key: string): string | null {
  return isRecord(value) && typeof value[key] === "string" ? value[key] : null;
}

function recordBoolean(value: unknown, key: string): boolean | null {
  return isRecord(value) && typeof value[key] === "boolean" ? value[key] : null;
}

function hasReversalPayload(value: unknown): boolean {
  return isRecord(value) && isRecord(value.reversal) && Object.keys(value.reversal).length > 0;
}

function ActionsTab({
  target,
  data,
  onRefresh,
  onModalOpenChange,
}: {
  target: InspectorTarget;
  data: unknown;
  onRefresh: () => Promise<void>;
  onModalOpenChange: (open: boolean) => void;
}) {
  const [confirmAction, setConfirmAction] = useState<ConfirmAction | null>(null);
  const [textAction, setTextAction] = useState<TextAction | null>(null);
  const [correctOpen, setCorrectOpen] = useState(false);
  const [correctPatch, setCorrectPatch] = useState("{}");
  const [correctReason, setCorrectReason] = useState("");
  const [reason, setReason] = useState("");
  const [textValue, setTextValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const anyModalOpen = confirmAction !== null || textAction !== null || correctOpen;

  useEffect(() => {
    onModalOpenChange(anyModalOpen);
    return () => onModalOpenChange(false);
  }, [anyModalOpen, onModalOpenChange]);

  const runAndRefresh = async (runner: () => Promise<unknown>, doneMessage: string) => {
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      await runner();
      setStatus(doneMessage);
      await onRefresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const openConfirm = (action: ConfirmAction) => {
    setReason("");
    setConfirmAction(action);
  };

  const openText = (action: TextAction) => {
    setTextValue(action.initialValue);
    setTextAction(action);
  };

  const actionButtons: ReactNode[] = [];
  const addButton = (key: string, label: string, onClick: () => void, className = "btn sm") => {
    actionButtons.push(
      <button key={key} type="button" className={className} onClick={onClick} disabled={busy}>
        {label}
      </button>,
    );
  };

  if (isWhySupported(target.type)) {
    addButton("correct", "correct", () => setCorrectOpen(true), "btn sm primary");
    addButton(
      "forget",
      "forget",
      () =>
        openConfirm({
          title: targetActionTitle("forget", target),
          body: "Queue a sanctioned correction forget for this stored object.",
          confirmLabel: "forget",
          danger: true,
          run: () => postCorrectionForget(target.id),
        }),
      "btn sm danger",
    );
  }

  if (target.type === "semantic_edge") {
    addButton(
      "invalidate",
      "invalidate edge",
      () =>
        openConfirm({
          title: targetActionTitle("invalidate", target),
          body: "Invalidate this semantic edge through the correction governance endpoint.",
          confirmLabel: "invalidate",
          reasonLabel: "reason",
          danger: true,
          run: (inputReason) =>
            postSemanticEdgeInvalidate(target.id, {
              ...(optionalText(inputReason) === undefined
                ? {}
                : { reason: optionalText(inputReason) }),
            }),
        }),
      "btn sm danger",
    );
  }

  if (target.type === "commitment") {
    addButton(
      "revoke",
      "revoke commitment",
      () =>
        openConfirm({
          title: targetActionTitle("revoke", target),
          body: "Revoke this commitment through the commitment governance endpoint.",
          confirmLabel: "revoke",
          reasonLabel: "reason",
          danger: true,
          run: (inputReason) => {
            const trimmed = optionalText(inputReason);
            return postCommitmentRevoke(
              target.id,
              trimmed === undefined ? {} : { reason: trimmed },
            );
          },
        }),
      "btn sm danger",
    );
  }

  if (target.type === "creator_directive") {
    addButton(
      "revoke-directive",
      "revoke directive",
      () =>
        openConfirm({
          title: targetActionTitle("revoke", target),
          body: "Revoke this creator directive. Creator directives are not correction targets.",
          confirmLabel: "revoke",
          reasonLabel: "reason",
          requireReason: true,
          danger: true,
          run: (inputReason) => revokeCreatorDirective(target.id, inputReason),
        }),
      "btn sm danger",
    );
    addButton(
      "supersede-directive",
      "supersede directive",
      () =>
        openText({
          title: targetActionTitle("supersede", target),
          label: "replacement directive id",
          initialValue: "",
          requireValue: true,
          danger: true,
          run: (replacementId) => supersedeCreatorDirective(target.id, replacementId.trim()),
        }),
      "btn sm danger",
    );
  }

  if (target.type === "goal") {
    addButton("goal-complete", "complete", () =>
      openConfirm({
        title: targetActionTitle("complete", target),
        body: "Mark this goal complete.",
        confirmLabel: "complete",
        reasonLabel: "note",
        run: (note) => {
          const trimmed = optionalText(note);
          return patchGoal(target.id, {
            action: "complete",
            ...(trimmed === undefined ? {} : { note: trimmed }),
          });
        },
      }),
    );
    addButton("goal-block", "block", () =>
      openConfirm({
        title: targetActionTitle("block", target),
        body: "Mark this goal blocked.",
        confirmLabel: "block",
        reasonLabel: "note",
        run: (note) => {
          const trimmed = optionalText(note);
          return patchGoal(target.id, {
            action: "block",
            ...(trimmed === undefined ? {} : { note: trimmed }),
          });
        },
      }),
    );
    addButton("goal-progress", "progress", () =>
      openConfirm({
        title: targetActionTitle("progress", target),
        body: "Append progress notes to this goal.",
        confirmLabel: "save progress",
        reasonLabel: "note",
        run: (note) => {
          const trimmed = optionalText(note);
          return patchGoal(target.id, {
            action: "progress",
            ...(trimmed === undefined ? {} : { note: trimmed }),
          });
        },
      }),
    );
  }

  if (target.type === "open_question") {
    addButton("question-resolve", "resolve", () =>
      openText({
        title: targetActionTitle("resolve", target),
        label: "resolution",
        initialValue: "",
        requireValue: true,
        run: (resolution) => patchOpenQuestion(target.id, { action: "resolve", resolution }),
      }),
    );
    addButton("question-abandon", "abandon", () =>
      openText({
        title: targetActionTitle("abandon", target),
        label: "reason",
        initialValue: "",
        requireValue: true,
        run: (abandonReason) =>
          patchOpenQuestion(target.id, { action: "abandon", reason: abandonReason }),
      }),
    );
    addButton("question-bump", "bump", () =>
      openConfirm({
        title: targetActionTitle("bump", target),
        body: "Increase this open question urgency.",
        confirmLabel: "bump",
        run: () => patchOpenQuestion(target.id, { action: "bump" }),
      }),
    );
  }

  if (target.type === "review") {
    const id = numericId(target.id);
    if (id !== null) {
      const kind = reviewKind(data);
      const row = kind === null ? null : reviewRowForAction(data, id, kind);
      if (kind === "creator_directive_reconciliation") {
        const directiveIds = reviewDirectiveIds(data);
        addButton(
          "creator-reconcile-supersede",
          "supersede directive",
          () =>
            openText({
              title: targetActionTitle("supersede directive review", target),
              label: "survivor directive id",
              initialValue: directiveIds[0] ?? "",
              requireValue: true,
              danger: true,
              run: (survivorId) =>
                resolveReviewAction({
                  row: row ?? reviewRowForAction(data, id, "creator_directive_reconciliation"),
                  action: "supersede",
                  survivorId,
                }),
            }),
          "btn sm danger",
        );
        addButton("creator-reconcile-keep", "keep directives", () =>
          openConfirm({
            title: targetActionTitle("keep directive review", target),
            body: "Resolve this creator-directive reconciliation by keeping the member directives.",
            confirmLabel: "keep",
            reasonLabel: "reason",
            run: (inputReason) => {
              const trimmed = optionalText(inputReason);
              return resolveReviewAction({
                row: row ?? reviewRowForAction(data, id, "creator_directive_reconciliation"),
                action: "keep",
                note: trimmed,
              });
            },
          }),
        );
      } else if (kind !== null && row !== null) {
        const ids = reviewNodeIds(data);
        for (const action of GENERIC_REVIEW_ACTIONS[kind]) {
          addButton(
            `review-${action}`,
            reviewActionLabel(action),
            () =>
              openConfirm({
                title: targetActionTitle(`${reviewActionLabel(action)} review`, target),
                body: `Resolve this ${kind.replaceAll("_", " ")} review as ${reviewActionLabel(action)}.`,
                confirmLabel: reviewActionLabel(action),
                reasonLabel: "note",
                danger: DESTRUCTIVE_REVIEW_ACTIONS.has(action),
                run: (note) => {
                  const trimmed = optionalText(note);
                  return resolveReviewAction({
                    row,
                    action,
                    note: trimmed,
                    winnerNodeId:
                      ids.length > 0 && (action === "supersede" || action === "invalidate")
                        ? ids[0]
                        : undefined,
                  });
                },
              }),
            DESTRUCTIVE_REVIEW_ACTIONS.has(action) ? "btn sm danger" : "btn sm",
          );
        }
      }
    }
  }

  if (target.type === "dream_audit" && numericId(target.id) !== null) {
    const id = numericId(target.id);
    const alreadyReverted =
      (isRecord(data) && data.reverted_at !== null && data.reverted_at !== undefined) ||
      recordString(data, "reverted_by") !== null ||
      recordBoolean(data, "reverted") === true;
    if (id !== null && hasReversalPayload(data) && !alreadyReverted) {
      addButton(
        "revert-dream-audit",
        "revert audit",
        () =>
          openConfirm({
            title: targetActionTitle("revert audit", target),
            body: "Apply the stored reversal payload for this dream audit row.",
            confirmLabel: "revert",
            run: () => revertDreamAudit(id),
          }),
        "btn sm live-write",
      );
    }
  }

  if (target.type === "prompt_block" && isPromptKey(target.id)) {
    const promptKey = target.id;
    const currentText = recordString(data, "current_text") ?? "";
    addButton("prompt-save", "save prompt", () =>
      openText({
        title: targetActionTitle("save", target),
        label: "prompt text",
        initialValue: currentText,
        run: (text) => putPrompt(promptKey, text),
      }),
    );
    addButton(
      "prompt-reset",
      "reset prompt",
      () =>
        openConfirm({
          title: targetActionTitle("reset", target),
          body: "Delete the stored override and return this prompt block to its default/runtime text.",
          confirmLabel: "reset",
          danger: true,
          run: () => deletePrompt(promptKey),
        }),
      "btn sm danger",
    );
  }

  if (target.type === "session") {
    for (const policy of SESSION_POLICIES) {
      addButton(`policy-${policy}`, `set ${policy}`, () =>
        openConfirm({
          title: targetActionTitle(`set session ${policy}`, target),
          body: `Set participation policy for ${target.id} to ${policy}.`,
          confirmLabel: `set ${policy}`,
          reasonLabel: "reason",
          run: (inputReason) => setSessionPolicy(target.id, policy, optionalText(inputReason)),
        }),
      );
    }
  }

  const submitConfirm = async () => {
    if (confirmAction === null) {
      return;
    }
    const action = confirmAction;
    await runAndRefresh(() => action.run(reason.trim()), "action completed");
    setConfirmAction(null);
  };

  const submitText = async () => {
    if (textAction === null) {
      return;
    }
    const action = textAction;
    const submittedValue = action.trimValue === false ? textValue : textValue.trim();
    await runAndRefresh(() => action.run(submittedValue), "action completed");
    setTextAction(null);
  };

  const submitCorrection = async () => {
    await runAndRefresh(
      () =>
        postCorrectionCorrect(target.id, {
          patch: parseJsonPatch(correctPatch),
          ...(correctReason.trim().length === 0 ? {} : { reason: correctReason.trim() }),
        }),
      "correction queued",
    );
    setCorrectOpen(false);
  };

  if (actionButtons.length === 0) {
    return <Empty>no sanctioned actions for this object type</Empty>;
  }

  return (
    <div className="inspector-actions">
      <div className="operator-actions">{actionButtons}</div>
      {status === null ? null : <div className="notice">{status}</div>}
      {error === null ? null : <ErrorState>{error}</ErrorState>}

      <Modal
        open={confirmAction !== null}
        title={<ActionModalTitle title={confirmAction?.title ?? null} fallback="confirm action" />}
        onClose={() => setConfirmAction(null)}
        footer={
          <>
            <button type="button" className="btn sm ghost" onClick={() => setConfirmAction(null)}>
              cancel
            </button>
            <button
              type="button"
              className={`btn sm ${confirmAction?.danger === true ? "danger" : "live-write"}`}
              onClick={submitConfirm}
              disabled={
                busy || (confirmAction?.requireReason === true && reason.trim().length === 0)
              }
            >
              {confirmAction?.confirmLabel ?? "confirm"}
            </button>
          </>
        }
      >
        <div className="modal-form">
          <div>{confirmAction?.body}</div>
          {confirmAction?.reasonLabel === undefined ? null : (
            <label className="modal-field">
              <span>{confirmAction.reasonLabel}</span>
              <textarea value={reason} onChange={(event) => setReason(event.currentTarget.value)} />
            </label>
          )}
        </div>
      </Modal>

      <Modal
        open={textAction !== null}
        title={<ActionModalTitle title={textAction?.title ?? null} fallback="edit action" />}
        onClose={() => setTextAction(null)}
        footer={
          <>
            <button type="button" className="btn sm ghost" onClick={() => setTextAction(null)}>
              cancel
            </button>
            <button
              type="button"
              className={`btn sm ${textAction?.danger === true ? "danger" : "live-write"}`}
              onClick={submitText}
              disabled={
                busy ||
                (textAction?.requireValue === true &&
                  (textAction.trimValue === false ? textValue : textValue.trim()).length === 0)
              }
            >
              save
            </button>
          </>
        }
      >
        <label className="modal-field">
          <span>{textAction?.label ?? "value"}</span>
          <textarea
            value={textValue}
            onChange={(event) => setTextValue(event.currentTarget.value)}
          />
        </label>
      </Modal>

      <Modal
        open={correctOpen}
        title={
          <span className="identity-inline">
            <span>correct</span>
            <IdChip id={target.id} type={target.type} />
          </span>
        }
        onClose={() => setCorrectOpen(false)}
        footer={
          <>
            <button type="button" className="btn sm ghost" onClick={() => setCorrectOpen(false)}>
              cancel
            </button>
            <button
              type="button"
              className="btn sm primary"
              onClick={submitCorrection}
              disabled={busy}
            >
              queue correction
            </button>
          </>
        }
      >
        <div className="modal-form">
          <label className="modal-field">
            <span>json patch object</span>
            <textarea
              value={correctPatch}
              onChange={(event) => setCorrectPatch(event.currentTarget.value)}
            />
          </label>
          <label className="modal-field">
            <span>reason</span>
            <textarea
              value={correctReason}
              onChange={(event) => setCorrectReason(event.currentTarget.value)}
            />
          </label>
        </div>
      </Modal>
    </div>
  );
}

function TabContent({
  activeTab,
  target,
  model,
  data,
  onRefresh,
  audience,
  onActionModalOpenChange,
}: {
  activeTab: InspectorTab;
  target: InspectorTarget;
  model: ObjectModel;
  data: unknown;
  onRefresh: () => Promise<void>;
  audience: string | null;
  onActionModalOpenChange: (open: boolean) => void;
}) {
  if (data === null) {
    return <Empty>{objectUnavailableMessage(target, model)}</Empty>;
  }

  if (activeTab === "summary") {
    return <SummaryTab target={target} data={data} />;
  }
  if (activeTab === "evidence") {
    return <EvidenceTab target={target} audience={audience} />;
  }
  if (activeTab === "relationships") {
    return <RelationshipsTab model={model} data={data} />;
  }
  if (activeTab === "timeline") {
    return <TimelineTab data={data} />;
  }
  if (activeTab === "actions") {
    return (
      <ActionsTab
        target={target}
        data={data}
        onRefresh={onRefresh}
        onModalOpenChange={onActionModalOpenChange}
      />
    );
  }
  return <RawJsonTab data={data} />;
}

function InspectorBody({
  target,
  activeTab,
  model,
  onActionModalOpenChange,
}: {
  target: InspectorTarget;
  activeTab: InspectorTab;
  model: ObjectModel;
  onActionModalOpenChange: (open: boolean) => void;
}) {
  const { sessionId, audience } = useInspector();
  const api = useApi(async () => {
    if (model.reliability === "needs_backend") {
      return null;
    }
    return model.fetch(target.id, { sessionId, audience });
  }, [target.type, target.id, target.hint, model, sessionId, audience]);

  if (api.loading && target.hint !== undefined && target.hint !== null) {
    return (
      <TabContent
        activeTab={activeTab}
        target={target}
        model={model}
        data={target.hint}
        onRefresh={api.refetch}
        audience={audience}
        onActionModalOpenChange={onActionModalOpenChange}
      />
    );
  }

  if (api.loading) {
    return <Loading>loading {model.label.toLowerCase()}</Loading>;
  }

  if (api.error !== null) {
    return <ErrorState>{errorMessage(api.error, model)}</ErrorState>;
  }

  return (
    <TabContent
      activeTab={activeTab}
      target={target}
      model={model}
      data={api.data}
      onRefresh={api.refetch}
      audience={audience}
      onActionModalOpenChange={onActionModalOpenChange}
    />
  );
}

export function Inspector() {
  const inspector = useInspector();
  const target = inspector.target;
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const [activeTab, setActiveTab] = useState<InspectorTab>("summary");
  const [actionModalOpen, setActionModalOpen] = useState(false);

  const model = target === null ? null : objectRegistry[target.type];
  const visibleTabs = useMemo(
    () => (model === null ? [] : INSPECTOR_TABS.filter((tab) => model.tabs.includes(tab))),
    [model],
  );

  useEffect(() => {
    if (target === null || model === null) {
      return;
    }
    setActionModalOpen(false);
    const requestedTab =
      target.presetTab !== undefined && model.tabs.includes(target.presetTab)
        ? target.presetTab
        : (model.tabs[0] ?? "summary");
    setActiveTab(requestedTab);
  }, [model, target]);

  useEffect(() => {
    if (target === null) {
      return;
    }
    closeButtonRef.current?.focus();
  }, [target]);

  useEffect(() => {
    if (target === null) {
      return;
    }

    function onKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape" && !actionModalOpen) {
        inspector.close();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [actionModalOpen, inspector, target]);

  if (target === null || model === null) {
    return null;
  }

  return (
    <div className="inspector-backdrop" onMouseDown={inspector.close}>
      <aside
        className="inspector-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`${model.label} inspector`}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="inspector-header">
          <div className="inspector-title">
            <span className="eyebrow">{model.label}</span>
            <span className="inspector-id" title={target.id}>
              {shortId(target.id)}
            </span>
            <Tag>{model.reliability}</Tag>
          </div>
          <div className="inspector-header-actions">
            {inspector.canBack ? (
              <button
                type="button"
                className="btn sm ghost"
                onClick={inspector.back}
                aria-label="back"
              >
                back
              </button>
            ) : null}
            <button
              type="button"
              className="btn sm ghost"
              onClick={() => {
                void copyText(target.id);
              }}
            >
              copy id
            </button>
            <button type="button" className="btn sm" onClick={inspector.openInSourceScreen}>
              open source
            </button>
            <button
              type="button"
              className="btn sm ghost"
              onClick={inspector.close}
              aria-label="close inspector"
              ref={closeButtonRef}
            >
              close
            </button>
          </div>
        </div>

        <div className="inspector-tabs" role="tablist" aria-label="inspector tabs">
          {visibleTabs.map((tab) => (
            <button
              key={tab}
              type="button"
              role="tab"
              aria-selected={activeTab === tab}
              className={`inspector-tab ${activeTab === tab ? "active" : ""}`}
              onClick={() => setActiveTab(tab)}
            >
              {tabLabel(tab)}
            </button>
          ))}
        </div>

        <div className="inspector-body">
          <InspectorBody
            target={target}
            activeTab={activeTab}
            model={model}
            onActionModalOpenChange={setActionModalOpen}
          />
        </div>
      </aside>
    </div>
  );
}
