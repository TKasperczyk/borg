import { z } from "zod";

import type { IdentityRecordType } from "./types.js";

export type AnnotationTarget = { record_type: IdentityRecordType; record_id: string };
export type AuditedAnnotation = {
  stream_entry_id: string;
  text: string;
  written_at?: number;
  authority: string | null;
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
