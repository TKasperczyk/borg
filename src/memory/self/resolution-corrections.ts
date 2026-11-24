import { z } from "zod";
import { isPlainRecord } from "../../util/guards.js";
import type { OpenQuestion } from "./types.js";
import type { IdentityEvent } from "../identity/types.js";

import { turnOriginSchema } from "../../contracts/cognitive-contracts.js";
import { streamEntryIdSchema } from "../../util/id-schemas.js";
import { episodeIdSchema } from "../episodic/types.js";
import { provenanceSchema } from "../common/provenance.js";
import {
  memoryDisclosureLabelSchema,
  renderMemoryDisclosureLabelForModel,
} from "../common/disclosure-label.js";

export const MAX_RESOLUTION_CORRECTIONS = 20;
export const MAX_RESOLUTION_CORRECTION_CHARS = 2_000;
export const MAX_CORRECTION_REPORTER_CHARS = 200;
export const RESOLUTION_CORRECTION_PAGE_MAX_CHARS = 4_000;
export const RESOLUTION_CORRECTION_TOOL_NAME = "tool.openQuestions.correctResolution";
export const RESOLUTION_CORRECTION_ACTION = "correct_resolution";
export const RESOLUTION_CHANGED_CODE = "resolution_changed";

export type OpenQuestionResolutionIdentity = {
  resolution_version: number | null;
  resolved_at: OpenQuestion["resolved_at"];
  resolution_note: OpenQuestion["resolution_note"];
};

/** Host-held values, copied before any await; never part of model input. */
export function openQuestionResolutionIdentity(
  question: Pick<OpenQuestion, "resolution_version" | "resolved_at" | "resolution_note">,
): OpenQuestionResolutionIdentity {
  return {
    resolution_version: question.resolution_version ?? null,
    resolved_at: question.resolved_at,
    resolution_note: question.resolution_note,
  };
}

export function isSameOpenQuestionResolution(
  current: OpenQuestion,
  target: OpenQuestionResolutionIdentity,
): boolean {
  return (
    current.status === "resolved" &&
    (current.resolution_version ?? null) === target.resolution_version &&
    current.resolved_at === target.resolved_at &&
    current.resolution_note === target.resolution_note
  );
}

const evidenceIdSchema = z.union([episodeIdSchema, streamEntryIdSchema]);
export const resolutionCorrectionEvidenceSchema = z.discriminatedUnion("basis", [
  z.object({ id: evidenceIdSchema, basis: z.literal("own_read") }).strict(),
  z
    .object({
      id: evidenceIdSchema,
      basis: z.literal("reported"),
      reported_by: z.string().trim().min(1).max(MAX_CORRECTION_REPORTER_CHARS),
    })
    .strict(),
]);

export const resolutionCorrectionFieldsSchema = z
  .object({
    correction_note: z.string().trim().min(1).max(MAX_RESOLUTION_CORRECTION_CHARS),
    corrects_span: z.string().trim().min(1).optional(),
    targets_correction_seq: z.number().int().positive().nullable().optional(),
    evidence: z.array(resolutionCorrectionEvidenceSchema).min(1),
  })
  .strict();

export const resolutionCorrectionWriterSchema = z
  .object({
    author: z.literal("borg"),
    turn_id: z.string().min(1).nullable(),
    turn_kind: turnOriginSchema,
    path: z.enum(["tool", "reflector"]),
    provenance: provenanceSchema,
  })
  .strict();

export const resolutionCorrectionSchema = z.object({
  seq: z.number().int().positive(),
  created_at: z.number().int().nonnegative(),
  text: z.string().trim().min(1).max(MAX_RESOLUTION_CORRECTION_CHARS),
  corrects_span: z.string().min(1).optional(),
  targets_correction_seq: z.number().int().positive().nullable(),
  evidence: z.array(resolutionCorrectionEvidenceSchema).min(1),
  writer: resolutionCorrectionWriterSchema,
  disclosure_label: memoryDisclosureLabelSchema,
});

export type ResolutionCorrection = z.infer<typeof resolutionCorrectionSchema>;
export type ResolutionCorrectionFields = z.infer<typeof resolutionCorrectionFieldsSchema>;
export type ResolutionCorrectionWriter = z.infer<typeof resolutionCorrectionWriterSchema>;

export function correctionEvidenceKey(evidence: ResolutionCorrectionFields["evidence"]): string {
  return JSON.stringify([...new Set(evidence.map((item) => JSON.stringify(item)))].sort());
}

export const resolutionCorrectionPresentationSchema = z.object({
  resolution_correction_count: z.number().int().positive().optional(),
  resolution_corrections: z
    .array(
      resolutionCorrectionSchema.extend({
        created_at_iso: z.string(),
        disclosure: z.string(),
      }),
    )
    .optional(),
  resolution_corrections_omitted: z
    .object({
      count: z.number().int().positive(),
      total: z.number().int().positive(),
      reason: z.literal("budget"),
    })
    .optional(),
});

export type ResolutionCorrectionPresentation = z.infer<
  typeof resolutionCorrectionPresentationSchema
>;
export type OpenQuestionPresentation = Omit<OpenQuestion, keyof ResolutionCorrectionPresentation> &
  ResolutionCorrectionPresentation;

/** Keep this sibling of the note/excerpt, never inside text that a caller truncates. */
export function formatResolutionCorrections<
  T extends {
    resolution_corrections?: readonly ResolutionCorrection[];
    resolution_correction_count?: number;
    resolution_corrections_omitted?: { count: number; total: number; reason: "budget" };
  },
>(
  question: T,
  options: { maxChars?: number } = {},
): Omit<T, keyof ResolutionCorrectionPresentation> & ResolutionCorrectionPresentation {
  const {
    resolution_corrections,
    resolution_correction_count,
    resolution_corrections_omitted: _priorOmission,
    ...record
  } = question;
  const corrections = resolution_corrections ?? [];
  const total = resolution_correction_count ?? corrections.length;
  if (total === 0) return record;
  const rendered = [];
  let remaining = options.maxChars ?? Number.POSITIVE_INFINITY;
  for (const correction of [...corrections].sort((a, b) => b.seq - a.seq)) {
    const row = {
      ...correction,
      created_at_iso: new Date(correction.created_at).toISOString(),
      disclosure: renderMemoryDisclosureLabelForModel(correction.disclosure_label),
    };
    const cost = JSON.stringify(row).length;
    if (cost > remaining) break;
    rendered.push(row);
    remaining -= cost;
  }
  const omitted = total - rendered.length;
  return {
    ...record,
    resolution_correction_count: total,
    resolution_corrections: rendered,
    ...(omitted === 0
      ? {}
      : {
          resolution_corrections_omitted: {
            count: omitted,
            total,
            reason: "budget" as const,
          },
        }),
  };
}

// Audit snapshots are historical values, not a lookup of today's question. Keep
// their correction counts next to their own notes when returning full snapshots.
export function formatOpenQuestionIdentityEvent(event: IdentityEvent): IdentityEvent {
  if (event.record_type !== "open_question") return event;
  const snapshot = (value: unknown): unknown => {
    if (!isPlainRecord(value)) return value;
    const corrections = z
      .array(resolutionCorrectionSchema)
      .parse(value.resolution_corrections ?? []);
    return formatResolutionCorrections({
      ...value,
      resolution_corrections: corrections,
    });
  };
  return { ...event, old_value: snapshot(event.old_value), new_value: snapshot(event.new_value) };
}
