import { BudgetExceededError } from "../../util/errors.js";
import type { AnnotationTarget } from "../identity/annotations.js";
import type { MemoryDisclosureLabelMetadata } from "./disclosure-label.js";

export const MAX_EVIDENCE_ANNOTATIONS_PER_RECORD = 3;
export const MAX_EVIDENCE_ANNOTATION_CHARS = 1_200;
export const MAX_EVIDENCE_AUTHORITY_CHARS = 200;
export const MAX_EVIDENCE_LINEAGE_EPISODES = 4;
export const MAX_EVIDENCE_LINEAGE_EXCERPT_CHARS = 1_200;
export const MAX_EVIDENCE_RECORD_CONTEXT_CHARS = 8_000;
export const MAX_EVIDENCE_PROMPT_CONTEXT_CHARS = 24_000;
export const MEMORY_EVIDENCE_GUIDANCE =
  "Annotations are audited corrections attached after a record was written; lifecycle=archived marks a record retired from recall.";
export const SELF_CORRECTION_GUIDANCE =
  "Annotations are corrections attached to a record after it was written. Those with author=borg are my own and unaudited; the others are audited.";
// A structural slot, not text matching: ordinary prompts omit it altogether.
export const EVIDENCE_GUIDANCE = Symbol("evidence guidance");
const CONTEXT_FIELD = "__borg_evidence_context";
const TEXT_FIELD = "__borg_evidence_text";

type AnnotationPromptRow = {
  text: string;
  written_at?: string;
  authority: string | null;
  author?: "borg";
  seq?: number;
  targets_correction_seq?: number;
  original_record?: AnnotationTarget;
  target_changed?: boolean | null;
  quoted_span_present?: boolean | null;
  disclosure?: string;
  disclosure_label?: MemoryDisclosureLabelMetadata;
};

export type AnnotationPromptFields = {
  annotations?: AnnotationPromptRow[];
  annotations_omitted?: number;
  self_correction_count?: number;
};
export type EpisodeEvidenceContext = AnnotationPromptFields & {
  lifecycle?: "archived" | "inactive" | "unknown";
  visible_lineage?: (Record<string, unknown> & AnnotationPromptFields)[];
  lineage_omitted?: number;
  member_annotations?: Record<string, unknown>[];
};

/** Try the selected order, while allowing another whole annotation to fit after a self correction. */
export function admitAnnotationRows(
  candidates: readonly AnnotationPromptRow[],
  tryAdmit: (rows: AnnotationPromptRow[]) => boolean,
): void {
  const hasSelfCorrection = candidates.some((row) => row.author === "borg");
  let kept: AnnotationPromptRow[] = [];
  for (const candidate of candidates) {
    const next = [...kept, candidate];
    if (tryAdmit(next)) kept = next;
    else if (!hasSelfCorrection) break;
  }
}

/** Internal marker survives the existing JSON-safe serializers; only this renderer expands it. */
export function evidencePromptFields(
  context: EpisodeEvidenceContext | undefined,
): Record<string, unknown> {
  return context === undefined || Object.keys(context).length === 0
    ? {}
    : { [CONTEXT_FIELD]: context };
}

/** Preserve a legacy text row verbatim, appending only its bounded, exact-record envelope. */
export function evidencePromptText(
  text: string,
  context: EpisodeEvidenceContext | undefined,
): unknown {
  return context === undefined ? text : { [TEXT_FIELD]: text, ...evidencePromptFields(context) };
}

function renderLine(part: unknown): string {
  if (typeof part === "string") return part;
  if (part !== null && typeof part === "object" && TEXT_FIELD in part) {
    const { [TEXT_FIELD]: text, ...envelope } = part as Record<string, unknown>;
    return `${text}${Object.keys(envelope).length === 0 ? "" : ` ${JSON.stringify(envelope)}`}`;
  }
  return JSON.stringify(part);
}

type Slot = {
  row: Record<string, unknown>;
  context: EpisodeEvidenceContext;
  parent?: Slot;
  depth: number;
};
const ENVELOPE_FIELDS = [
  "lifecycle",
  "annotations",
  "annotations_omitted",
  "self_correction_count",
  "visible_lineage",
  "lineage_omitted",
  "member_annotations",
] as const;

/** Allocate only added context. Cited text is never changed, and actual serialized growth is counted. */
function renderEvidencePrompt(
  payload: unknown,
  format: "json" | "compact_json" | "lines",
  guidanceText = MEMORY_EVIDENCE_GUIDANCE,
): string {
  const slots: Slot[] = [];
  const clone = (value: unknown, parent?: Slot, depth = 0): unknown => {
    if (Array.isArray(value)) return value.map((entry) => clone(entry, parent, depth + 1));
    if (value === null || typeof value !== "object") return value;
    const input = value as Record<string, unknown>;
    const row: Record<string, unknown> = {};
    const context = input[CONTEXT_FIELD] as EpisodeEvidenceContext | undefined;
    const slot = context === undefined ? undefined : { row, context, parent, depth };
    if (slot !== undefined) slots.push(slot);
    for (const [key, entry] of Object.entries(input)) {
      if (key !== CONTEXT_FIELD) row[key] = clone(entry, parent, depth + 1);
    }
    if (slot !== undefined && context?.member_annotations !== undefined) {
      slot.context = {
        ...context,
        member_annotations: context.member_annotations.map(
          (member) => clone(member, slot, depth + 2) as Record<string, unknown>,
        ),
      };
    }
    return row;
  };
  const packet = clone(payload);
  const selfCorrectionGuidance =
    guidanceText === MEMORY_EVIDENCE_GUIDANCE
      ? `${SELF_CORRECTION_GUIDANCE} lifecycle=archived marks a record retired from recall.`
      : SELF_CORRECTION_GUIDANCE;
  const retainedGuidance = () =>
    slots.some(({ row }) => {
      const retained = row as EpisodeEvidenceContext;
      return (
        retained.annotations?.some((annotation) => annotation.author === "borg") ||
        retained.visible_lineage?.some((member) =>
          member.annotations?.some((annotation) => annotation.author === "borg"),
        )
      );
    })
      ? selfCorrectionGuidance
      : guidanceText;
  const serialize = (guidance?: string): string => {
    const replacer = (_key: string, value: unknown) =>
      value === EVIDENCE_GUIDANCE ? guidance : value;
    return format !== "lines"
      ? JSON.stringify(packet, replacer, format === "json" ? 2 : undefined)
      : (packet as unknown[])
          .flatMap((part) =>
            part === EVIDENCE_GUIDANCE
              ? guidance === undefined
                ? []
                : [guidance]
              : [renderLine(part)],
          )
          .join("\n");
  };
  const ordinary = serialize();
  if (slots.length === 0) return ordinary;
  const setCount = (row: Record<string, unknown>, key: string, count: number) => {
    if (count > 0) row[key] = count;
    else delete row[key];
  };
  // Reserve labels and omission counts for every target before spending on text.
  for (const { row, context } of slots) {
    if (context.lifecycle !== undefined) row.lifecycle = context.lifecycle;
    setCount(
      row,
      "annotations_omitted",
      (context.annotations?.length ?? 0) + (context.annotations_omitted ?? 0),
    );
    setCount(row, "self_correction_count", context.self_correction_count ?? 0);
    setCount(
      row,
      "lineage_omitted",
      (context.visible_lineage?.length ?? 0) + (context.lineage_omitted ?? 0),
    );
    if ((context.member_annotations?.length ?? 0) > 0)
      row.member_annotations = context.member_annotations;
  }
  const fitsRecord = (slot: Slot): boolean => {
    const envelope = Object.fromEntries(
      ENVELOPE_FIELDS.filter((key) => slot.row[key] !== undefined).map((key) => [
        key,
        slot.row[key],
      ]),
    );
    const serialized = JSON.stringify(envelope, null, format === "json" ? 2 : undefined);
    const addedIndent =
      format === "json" ? slot.depth * 2 * (serialized.split("\n").length - 1) : 0;
    return (
      serialized.length + addedIndent <= MAX_EVIDENCE_RECORD_CONTEXT_CHARS &&
      (slot.parent === undefined || fitsRecord(slot.parent))
    );
  };
  const fitsPrompt = () =>
    serialize(retainedGuidance()).length - ordinary.length <= MAX_EVIDENCE_PROMPT_CONTEXT_CHARS;
  if (!slots.every(fitsRecord) || !fitsPrompt()) {
    // Even the mandatory per-target labels/counts exceed the budget. Do not send an oversized call.
    throw new BudgetExceededError("Evidence labels exceed the prompt context budget", {
      code: "OFFLINE_BUDGET_EXCEEDED",
    });
  }
  const tryAppend = (
    slot: Slot,
    row: Record<string, unknown>,
    field: "annotations" | "visible_lineage",
    total: number,
    rows: unknown[],
  ): boolean => {
    const omittedField = field === "annotations" ? "annotations_omitted" : "lineage_omitted";
    const previous = row[field];
    row[field] = rows;
    setCount(row, omittedField, total - rows.length);
    if (fitsRecord(slot) && fitsPrompt()) return true;
    if (previous === undefined) delete row[field];
    else row[field] = previous;
    setCount(row, omittedField, total - rows.length + 1);
    return false;
  };
  const admitAnnotations = (
    slot: Slot,
    row: Record<string, unknown>,
    context: AnnotationPromptFields,
  ) => {
    const annotations = context.annotations ?? [];
    const total = annotations.length + (context.annotations_omitted ?? 0);
    admitAnnotationRows(annotations, (rows) => tryAppend(slot, row, "annotations", total, rows));
  };
  for (const slot of slots) {
    admitAnnotations(slot, slot.row, slot.context);
    const lineage = slot.context.visible_lineage ?? [];
    const total = lineage.length + (slot.context.lineage_omitted ?? 0);
    const kept: typeof lineage = [];
    const deferred: { row: Record<string, unknown>; source: AnnotationPromptFields }[] = [];
    for (const source of lineage) {
      const annotations = source.annotations ?? [];
      const deferAnnotations = annotations.some((annotation) => annotation.author === "borg");
      const row = deferAnnotations ? { ...source } : source;
      if (deferAnnotations) {
        delete row.annotations;
        setCount(
          row,
          "annotations_omitted",
          annotations.length + (source.annotations_omitted ?? 0),
        );
      }
      if (!tryAppend(slot, slot.row, "visible_lineage", total, [...kept, row])) break;
      kept.push(row);
      if (deferAnnotations) deferred.push({ row, source });
    }
    for (const { row, source } of deferred) {
      admitAnnotations(slot, row, source);
    }
  }
  return serialize(retainedGuidance());
}

export function renderEvidencePromptJson(
  payload: Record<string, unknown>,
  options: { compact?: boolean; guidance?: string } = {},
): string {
  return renderEvidencePrompt(payload, options.compact ? "compact_json" : "json", options.guidance);
}
export function renderEvidencePromptLines(parts: readonly unknown[], guidance?: string): string {
  return renderEvidencePrompt(parts, "lines", guidance);
}
