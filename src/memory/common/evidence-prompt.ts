import { BudgetExceededError } from "../../util/errors.js";

export const MAX_EVIDENCE_ANNOTATIONS_PER_RECORD = 3;
export const MAX_EVIDENCE_ANNOTATION_CHARS = 1_200;
export const MAX_EVIDENCE_AUTHORITY_CHARS = 200;
export const MAX_EVIDENCE_LINEAGE_EPISODES = 4;
export const MAX_EVIDENCE_LINEAGE_EXCERPT_CHARS = 1_200;
export const MAX_EVIDENCE_RECORD_CONTEXT_CHARS = 8_000;
export const MAX_EVIDENCE_PROMPT_CONTEXT_CHARS = 24_000;
export const MEMORY_EVIDENCE_GUIDANCE =
  "Annotations are audited corrections attached after a record was written; lifecycle=archived marks a record retired from recall.";
// A structural slot, not text matching: ordinary prompts omit it altogether.
export const EVIDENCE_GUIDANCE = Symbol("evidence guidance");
const CONTEXT_FIELD = "__borg_evidence_context";

export type AnnotationPromptFields = {
  annotations?: Array<{ text: string; written_at?: string; authority: string | null }>;
  annotations_omitted?: number;
};
export type EpisodeEvidenceContext = AnnotationPromptFields & {
  lifecycle?: "archived" | "inactive" | "unknown";
  visible_lineage?: Record<string, unknown>[];
  lineage_omitted?: number;
  member_annotations?: Record<string, unknown>[];
};

/** Internal marker survives the existing JSON-safe serializers; only this renderer expands it. */
export function evidencePromptFields(
  context: EpisodeEvidenceContext | undefined,
): Record<string, unknown> {
  return context === undefined || Object.keys(context).length === 0
    ? {}
    : { [CONTEXT_FIELD]: context };
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
  "visible_lineage",
  "lineage_omitted",
  "member_annotations",
] as const;

/** Allocate only added context. Cited text is never changed, and actual serialized growth is counted. */
function renderEvidencePrompt(payload: unknown, format: "json" | "lines"): string {
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
  const serialize = (guidance: boolean): string => {
    const replacer = (_key: string, value: unknown) =>
      value === EVIDENCE_GUIDANCE ? (guidance ? MEMORY_EVIDENCE_GUIDANCE : undefined) : value;
    return format === "json"
      ? JSON.stringify(packet, replacer, 2)
      : (packet as unknown[])
          .flatMap((part) =>
            part === EVIDENCE_GUIDANCE
              ? guidance
                ? [MEMORY_EVIDENCE_GUIDANCE]
                : []
              : [typeof part === "string" ? part : JSON.stringify(part)],
          )
          .join("\n");
  };
  const ordinary = serialize(false);
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
    serialize(true).length - ordinary.length <= MAX_EVIDENCE_PROMPT_CONTEXT_CHARS;
  if (!slots.every(fitsRecord) || !fitsPrompt()) {
    // Even the mandatory per-target labels/counts exceed the budget. Do not send an oversized call.
    throw new BudgetExceededError("Evidence labels exceed the prompt context budget", {
      code: "OFFLINE_BUDGET_EXCEEDED",
    });
  }
  for (const slot of slots) {
    for (const [field, omittedField] of [
      ["annotations", "annotations_omitted"],
      ["visible_lineage", "lineage_omitted"],
    ] as const) {
      const candidates = slot.context[field] ?? [];
      const total = candidates.length + (slot.context[omittedField] ?? 0);
      const kept: unknown[] = [];
      for (const candidate of candidates) {
        slot.row[field] = [...kept, candidate];
        setCount(slot.row, omittedField, total - kept.length - 1);
        if (!fitsRecord(slot) || !fitsPrompt()) {
          if (kept.length === 0) delete slot.row[field];
          else slot.row[field] = kept;
          setCount(slot.row, omittedField, total - kept.length);
          break;
        }
        kept.push(candidate);
      }
    }
  }
  return serialize(true);
}

export function renderEvidencePromptJson(payload: Record<string, unknown>): string {
  return renderEvidencePrompt(payload, "json");
}
export function renderEvidencePromptLines(parts: readonly unknown[]): string {
  return renderEvidencePrompt(parts, "lines");
}
