import { z } from "zod";
import type { EpisodeId } from "../../util/ids.js";
import { annotationTargetKey, type AnnotationTarget } from "../identity/annotations.js";
import {
  annotationPromptFields,
  type EpisodeEvidenceReader,
  type EpisodeEvidenceContext,
} from "./episode-evidence.js";
import { renderEvidencePromptJson } from "./evidence-prompt.js";

export type RecordEvidenceReader = Pick<EpisodeEvidenceReader, "identityEventRepository"> & {
  episodicRepository?: Pick<EpisodeEvidenceReader["episodicRepository"], "getLifecycleMany">;
};
export const RECORD_ANNOTATION_GUIDANCE =
  "Annotations are audited corrections attached to a record after it was written.";
export const recordEvidenceShape = {
  lifecycle: z.enum(["archived", "inactive", "unknown"]).optional(),
  annotations: z
    .array(
      z.object({
        text: z.string(),
        written_at: z.string().optional(),
        authority: z.string().nullable(),
      }),
    )
    .optional(),
  annotations_omitted: z.number().int().nonnegative().optional(),
};

/** Tool schemas validate the fully bounded model-facing object, never the internal marker. */
export function renderRecordEvidenceObject<T extends Record<string, unknown>>(
  payload: T,
): Omit<T, "evidence_guidance"> & { evidence_guidance?: string } {
  return JSON.parse(
    renderEvidencePromptJson(payload, { compact: true, guidance: RECORD_ANNOTATION_GUIDANCE }),
  );
}

/** Exact typed bindings only: source/provenance ids never confer an annotation on a derived record. */
export function readRecordEvidenceContexts(
  reader: RecordEvidenceReader | undefined,
  targets: readonly AnnotationTarget[],
  annotationOptions: Parameters<typeof annotationPromptFields>[1] = {},
): Map<string, EpisodeEvidenceContext> {
  if (reader === undefined || targets.length === 0) return new Map();
  const episodes = targets.filter((target) => target.record_type === "episode");
  const lifecycle = reader.episodicRepository?.getLifecycleMany(
    episodes.map((target) => target.record_id as EpisodeId),
  );
  return new Map(
    reader.identityEventRepository.listAnnotationsForTargets(targets).flatMap((target) => {
      const status =
        target.record_type === "episode"
          ? lifecycle?.get(target.record_id as EpisodeId)
          : undefined;
      const context: EpisodeEvidenceContext = {
        ...(status === undefined || status === "active" ? {} : { lifecycle: status }),
        ...annotationPromptFields(target.annotations, annotationOptions),
      };
      return Object.keys(context).length === 0 ? [] : [[annotationTargetKey(target), context]];
    }),
  );
}

export function recordEvidenceContext(
  contexts: ReadonlyMap<string, EpisodeEvidenceContext> | undefined,
  target: AnnotationTarget | undefined,
): EpisodeEvidenceContext | undefined {
  return target === undefined ? undefined : contexts?.get(annotationTargetKey(target));
}
