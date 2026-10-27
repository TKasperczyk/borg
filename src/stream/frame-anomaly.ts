import { readFileSync, statSync } from "node:fs";
import { z } from "zod";
import { memoryDisclosureLabelMetadataSchema } from "../memory/common/disclosure-label.js";
import { sessionIdSchema, streamEntryIdSchema } from "../util/id-schemas.js";
import { isPlainRecord } from "../util/guards.js";
import { streamEntryReadSchema, type StreamEntry } from "./types.js";
import { quarantinedUserEntrySourceIds } from "./turn-status.js";

export const FRAME_ANOMALY_ANNOTATION_EVENT = "frame_anomaly_annotation";

// A source annotation, not an authorization grant or a privacy classification.
// Restoration names the exact quarantine markers it reverses; aborts are never reversed.
export const frameAnomalyAnnotationEventSchema = z.object({
  event: z.literal(FRAME_ANOMALY_ANNOTATION_EVENT),
  version: z.literal(1),
  source_stream_entry_id: streamEntryIdSchema,
  source_session_id: sessionIdSchema,
  source_type: z.string(),
  source_stream_entry_ids: z.array(streamEntryIdSchema),
  kind: z.string(),
  confidence: z.number().min(0).max(1),
  rationale: z.string(),
  classifier_model: z.string().nullable(),
  classifier_turn_id: z.string().nullable(),
  classifier_source_event_id: streamEntryIdSchema.nullable(),
  restores_quarantine_marker_ids: z.array(streamEntryIdSchema),
  audience: z.string().nullable(),
  sender_entity_id: z.string().nullable(),
  disclosure: z.string(),
  disclosure_label: memoryDisclosureLabelMetadataSchema,
  source_disclosure_label: memoryDisclosureLabelMetadataSchema,
});

export const frameAnomalyAnnotationSchema = frameAnomalyAnnotationEventSchema.extend({
  annotation_entry_id: streamEntryIdSchema,
  timestamp: z.number(),
  label: z.string(),
});
export type FrameAnomalyAnnotation = z.infer<typeof frameAnomalyAnnotationSchema>;

export function frameAnomalyAnnotationFromEntry(
  entry: Pick<StreamEntry, "kind" | "content" | "id" | "timestamp" | "session_id">,
): FrameAnomalyAnnotation | null {
  if (
    entry.kind !== "internal_event" ||
    !isPlainRecord(entry.content) ||
    entry.content.event !== FRAME_ANOMALY_ANNOTATION_EVENT
  )
    return null;
  const parsed = frameAnomalyAnnotationEventSchema.safeParse(entry.content);
  if (!parsed.success || parsed.data.source_session_id !== entry.session_id) return null;
  return {
    ...parsed.data,
    annotation_entry_id: entry.id,
    timestamp: entry.timestamp,
    label: `frame anomaly suspected: ${parsed.data.kind} (${parsed.data.confidence})`,
  };
}

export function renderFrameAnomalyAnnotations(
  annotations: readonly FrameAnomalyAnnotation[] | undefined,
): string {
  return (annotations ?? [])
    .map(
      (annotation) =>
        `${annotation.label}; rationale=${JSON.stringify(annotation.rationale)}; annotation_id=${annotation.annotation_entry_id}; ${annotation.disclosure}`,
    )
    .join("\n");
}

export function frameAnomalyAnnotationFields(
  annotations: readonly FrameAnomalyAnnotation[] | undefined,
): {
  frame_anomalies?: FrameAnomalyAnnotation[];
} {
  return annotations === undefined || annotations.length === 0
    ? {}
    : {
        frame_anomalies: [
          ...new Map(
            annotations.map((annotation) => [annotation.annotation_entry_id, annotation]),
          ).values(),
        ],
      };
}

// No-index harnesses must see the same projection even when tail/offset reads omit
// the later annotation event. Production uses the indexed projection instead.
// Like StreamReader, tolerate malformed JSONL lines, not malformed annotations.
export function readFrameAnomalyAnnotations(
  streamPath: string,
): Map<string, FrameAnomalyAnnotation[]> {
  const projection = new Map<string, FrameAnomalyAnnotation[]>();
  for (const line of readFileSync(streamPath, "utf8").split("\n")) {
    if (line.trim() === "") continue;
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch {
      continue;
    } // StreamReader reports malformed source lines during its scan.
    const envelope = streamEntryReadSchema.safeParse(raw);
    if (!envelope.success) continue;
    const annotation = frameAnomalyAnnotationFromEntry(envelope.data);
    if (annotation === null) continue;
    const list = projection.get(annotation.source_stream_entry_id) ?? [];
    list.push(annotation);
    projection.set(annotation.source_stream_entry_id, list);
  }
  return projection;
}

export class StreamFrameAnomalyReader {
  private stamp: string | undefined;
  private projection = new Map<string, FrameAnomalyAnnotation[]>();

  annotate(entry: StreamEntry, streamPath: string): StreamEntry {
    const stat = statSync(streamPath);
    const stamp = `${stat.ino}:${stat.size}:${stat.mtimeMs}`;
    if (stamp !== this.stamp) {
      this.projection = readFrameAnomalyAnnotations(streamPath);
      this.stamp = stamp;
    }
    return {
      ...entry,
      ...frameAnomalyAnnotationFields([
        ...(this.projection.get(entry.id) ?? []),
        ...quarantinedUserEntrySourceIds(entry).flatMap((id) =>
          (this.projection.get(id) ?? []).filter((annotation) =>
            annotation.restores_quarantine_marker_ids.includes(entry.id),
          ),
        ),
      ]),
    };
  }
}
