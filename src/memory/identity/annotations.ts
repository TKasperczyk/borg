import { createHash } from "node:crypto";
import { z } from "zod";

import { streamEntryIdSchema } from "../../util/id-schemas.js";
import { toWellFormedUtf16 } from "../../util/utf16-boundary.js";
import { episodeIdSchema } from "../episodic/types.js";
import { semanticNodeIdSchema } from "../semantic/types.js";
import {
  MAX_CORRECTION_REPORTER_CHARS,
  resolutionCorrectionEvidenceSchema,
  resolutionCorrectionFieldsSchema,
  resolutionCorrectionSchema,
  resolutionCorrectionWriterSchema,
} from "../self/resolution-corrections.js";
import type { IdentityRecordType } from "./types.js";

export const RECORD_CORRECTION_TOOL_NAME = "tool.records.correct";
export const MAX_LEDGER_ANNOTATIONS_PER_RECORD = 2;
export const MAX_LEDGER_ANNOTATION_CHARS = 600;
export const LEDGER_ANNOTATION_OPTIONS = {
  maxAnnotations: MAX_LEDGER_ANNOTATIONS_PER_RECORD,
  maxChars: MAX_LEDGER_ANNOTATION_CHARS,
  wholeTextOnly: true,
} as const;

const nonblankText = z
  .string()
  .min(1)
  .refine((text) => text.trim().length > 0, {
    message: "Text cannot be blank",
  });
const wellFormedText = (text: string) => toWellFormedUtf16(text) === text;
const unicodeMessage = { message: "Text must be well-formed UTF-16" };
const recordCorrectionTextSchema = nonblankText
  .max(MAX_LEDGER_ANNOTATION_CHARS)
  .refine(wellFormedText, unicodeMessage);
const recordCorrectionSpanSchema = z
  .string()
  .min(1)
  .refine(wellFormedText, unicodeMessage)
  .optional();
const [ownReadEvidence, reportedEvidence] = resolutionCorrectionEvidenceSchema.options;
const recordCorrectionEvidence = z
  .array(
    z.discriminatedUnion("basis", [
      ownReadEvidence,
      reportedEvidence.extend({ reported_by: nonblankText.max(MAX_CORRECTION_REPORTER_CHARS) }),
    ]),
  )
  .min(1);

export const recordCorrectionInputSchema = resolutionCorrectionFieldsSchema.safeExtend({
  target_id: z.union([episodeIdSchema, semanticNodeIdSchema]),
  correction_note: recordCorrectionTextSchema,
  corrects_span: recordCorrectionSpanSchema,
  evidence: recordCorrectionEvidence,
});

export const recordCorrectionTargetSchema = z.object({
  record_type: z.enum(["episode", "semantic_node"]),
  record_id: z.union([episodeIdSchema, semanticNodeIdSchema]),
});

export const recordCorrectionMetadataSchema = resolutionCorrectionSchema
  .pick({
    seq: true,
    corrects_span: true,
    targets_correction_seq: true,
    evidence: true,
    disclosure_label: true,
  })
  .extend({
    writer: resolutionCorrectionWriterSchema.extend({ turn_id: z.string().min(1) }),
    evidence: recordCorrectionEvidence,
    target_text_hash: z.hash("sha256"),
    quoted_text_hash: z.hash("sha256").optional(),
  });

export const recordCorrectionResultSchema = z.object({
  correction: recordCorrectionMetadataSchema.extend({
    original_record: recordCorrectionTargetSchema,
    stream_entry_id: streamEntryIdSchema,
    text: recordCorrectionTextSchema,
    created_at: z.number().finite(),
    created_at_iso: z.string(),
    disclosure: z.string(),
  }),
  seq: z.number().int().positive(),
  total_count: z.number().int().nonnegative(),
  shown_count: z.number().int().nonnegative(),
});

export type RecordCorrectionInput = z.infer<typeof recordCorrectionInputSchema>;
export type RecordCorrectionResult = z.infer<typeof recordCorrectionResultSchema>;
export type RecordCorrectionMetadata = z.infer<typeof recordCorrectionMetadataSchema>;

export function recordCorrectionTextHash(text: string): string {
  return createHash("sha256").update(text, "utf16le").digest("hex");
}

export type AnnotationTarget = { record_type: IdentityRecordType; record_id: string };
export type AuditedAnnotation = {
  stream_entry_id: string;
  text: string;
  written_at?: number;
  authority: string | null;
  self_correction?: RecordCorrectionMetadata & { original_record: AnnotationTarget };
  quoted_correction_text?: string;
  retracted?: true;
};
export type TargetAnnotations = AnnotationTarget & { annotations: AuditedAnnotation[] };

// These are durable event protocol fields, independent of the repair's action name.
export const auditedAnnotationPayloadSchema = z.object({
  stream_entry_id: z.string().min(1),
  annotation: z.discriminatedUnion("event", [
    z.object({
      event: z.literal("episode_annotation"),
      target_ids: z.array(z.string()).optional(),
      episode_id: z.string().optional(),
      correction: z.string().optional(),
      note: z.string().optional(),
      authority: z.string().nullable().optional(),
      self_correction: recordCorrectionMetadataSchema.optional(),
    }),
    z.object({
      event: z.literal("episode_annotation_retracted"),
      reverses_stream_entry_id: z.string().min(1),
      target_ids: z.array(z.string()),
    }),
  ]),
});

export function annotationTargetKey(target: AnnotationTarget): string {
  return `${target.record_type}:${target.record_id}`;
}
