import { z } from "zod";
import type { EpisodeId, SemanticNodeId } from "../../util/ids.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import {
  annotationTargetKey,
  recordCorrectionTargetSchema,
  type AnnotationTarget,
} from "../identity/annotations.js";
import { memoryDisclosureLabelMetadataSchema } from "./disclosure-label.js";
import {
  renderSelectedAnnotations,
  selectAnnotationPromptRows,
  type AnnotationPromptOptions,
  type EpisodeEvidenceReader,
  type EpisodeEvidenceContext,
} from "./episode-evidence.js";
import type { SemanticNodeRepository } from "../semantic/repository.js";
import { renderEvidencePromptJson } from "./evidence-prompt.js";

export type RecordEvidenceReader = Pick<EpisodeEvidenceReader, "identityEventRepository"> & {
  episodicRepository?: Pick<EpisodeEvidenceReader["episodicRepository"], "getLifecycleMany"> &
    Partial<Pick<EpisodeEvidenceReader["episodicRepository"], "getMany">>;
  semanticNodeRepository?: Pick<SemanticNodeRepository, "getMany">;
  tracer?: TurnTracer;
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
        author: z.literal("borg").optional(),
        seq: z.number().int().positive().optional(),
        targets_correction_seq: z.number().int().positive().optional(),
        original_record: recordCorrectionTargetSchema.optional(),
        target_changed: z.boolean().nullable().optional(),
        quoted_span_present: z.boolean().nullable().optional(),
        disclosure: z.string().optional(),
        disclosure_label: memoryDisclosureLabelMetadataSchema.optional(),
      }),
    )
    .optional(),
  annotations_omitted: z.number().int().nonnegative().optional(),
  self_correction_count: z.number().int().positive().optional(),
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
export async function readRecordEvidenceContexts(
  reader: RecordEvidenceReader | undefined,
  targets: readonly AnnotationTarget[],
  options: AnnotationPromptOptions & {
    readLoadedText?: (target: AnnotationTarget) => string | undefined;
  } = {},
): Promise<Map<string, EpisodeEvidenceContext>> {
  if (reader === undefined || targets.length === 0) return new Map();
  const selections = reader.identityEventRepository
    .listAnnotationsForTargets(targets)
    .map(({ annotations, ...target }) => ({
      target,
      selection: selectAnnotationPromptRows(annotations, options),
    }));
  const episodes = targets.filter((target) => target.record_type === "episode");
  const lifecycle = reader.episodicRepository?.getLifecycleMany(
    episodes.map((target) => target.record_id as EpisodeId),
  );
  const texts = new Map<string, string>();
  const episodeIds: EpisodeId[] = [];
  const nodeIds: SemanticNodeId[] = [];
  for (const { target, selection } of selections) {
    if (!selection.selected.some((row) => row.self_correction !== undefined)) continue;
    const loaded = options.readLoadedText?.(target);
    if (loaded !== undefined) texts.set(annotationTargetKey(target), loaded);
    else if (target.record_type === "episode") episodeIds.push(target.record_id as EpisodeId);
    else if (target.record_type === "semantic_node")
      nodeIds.push(target.record_id as SemanticNodeId);
  }
  if (episodeIds.length > 0 || nodeIds.length > 0) {
    const [episodeRows, nodeRows] = await Promise.allSettled([
      (async () =>
        episodeIds.length === 0 ? [] : (reader.episodicRepository?.getMany?.(episodeIds) ?? []))(),
      (async () =>
        nodeIds.length === 0
          ? []
          : (reader.semanticNodeRepository?.getMany(nodeIds, { includeArchived: true }) ?? []))(),
    ]);
    for (const episode of episodeRows.status === "fulfilled" ? episodeRows.value : []) {
      texts.set(
        annotationTargetKey({ record_type: "episode", record_id: episode.id }),
        episode.narrative,
      );
    }
    for (const node of nodeRows.status === "fulfilled" ? nodeRows.value : []) {
      if (node !== null)
        texts.set(
          annotationTargetKey({ record_type: "semantic_node", record_id: node.id }),
          node.description,
        );
    }
    if (
      reader.tracer?.enabled === true &&
      (episodeRows.status === "rejected" || nodeRows.status === "rejected")
    ) {
      reader.tracer.emit("memory.annotations.body_lookup_failed", {
        turnId: "memory-annotation-read",
        counts_by_reason: {
          ...(episodeRows.status === "rejected" ? { episode: episodeIds.length } : {}),
          ...(nodeRows.status === "rejected" ? { semantic_node: nodeIds.length } : {}),
        },
      });
    }
  }
  return new Map(
    selections.flatMap(({ target, selection }) => {
      const key = annotationTargetKey(target);
      const status =
        target.record_type === "episode"
          ? lifecycle?.get(target.record_id as EpisodeId)
          : undefined;
      const context: EpisodeEvidenceContext = {
        ...(status === undefined || status === "active" ? {} : { lifecycle: status }),
        ...renderSelectedAnnotations(selection, texts.get(key)),
      };
      return Object.keys(context).length === 0 ? [] : [[key, context]];
    }),
  );
}

export function recordEvidenceContext(
  contexts: ReadonlyMap<string, EpisodeEvidenceContext> | undefined,
  target: AnnotationTarget | undefined,
): EpisodeEvidenceContext | undefined {
  return target === undefined ? undefined : contexts?.get(annotationTargetKey(target));
}
