import type { Episode } from "../episodic/types.js";
import type { EpisodicRepository } from "../episodic/repository.js";
import type { IdentityEventRepository } from "../identity/repository.js";
import { recordCorrectionTextHash, type AuditedAnnotation } from "../identity/annotations.js";
import { utf16SafePrefixEnd } from "../../util/utf16-boundary.js";
import { memoryDisclosurePayloadFields } from "./disclosure-serializers.js";
import { memoryDisclosureLabelFromEpisodeAccess } from "./disclosure-label.js";

import {
  MAX_EVIDENCE_ANNOTATIONS_PER_RECORD,
  MAX_EVIDENCE_ANNOTATION_CHARS,
  MAX_EVIDENCE_AUTHORITY_CHARS,
  MAX_EVIDENCE_LINEAGE_EPISODES,
  MAX_EVIDENCE_LINEAGE_EXCERPT_CHARS,
  evidencePromptFields,
  type AnnotationPromptFields,
  type EpisodeEvidenceContext,
} from "./evidence-prompt.js";
export * from "./evidence-prompt.js";

export type EpisodeWithEvidenceContext = Episode & { evidence_context?: EpisodeEvidenceContext };
export type EpisodeEvidenceReader = {
  episodicRepository: Pick<EpisodicRepository, "getLifecycleMany" | "getMany">;
  identityEventRepository: Pick<IdentityEventRepository, "listAnnotationsForTargets">;
};

function excerpt(text: string, limit: number): string {
  return text.length <= limit ? text : `${text.slice(0, utf16SafePrefixEnd(text, limit - 1))}…`;
}

export type AnnotationPromptOptions = {
  maxAnnotations?: number;
  maxChars?: number;
  wholeTextOnly?: boolean;
};

export function selectAnnotationPromptRows(
  annotations: readonly AuditedAnnotation[],
  options: AnnotationPromptOptions = {},
) {
  const maxAnnotations = options.maxAnnotations ?? MAX_EVIDENCE_ANNOTATIONS_PER_RECORD;
  const maxChars = options.maxChars ?? MAX_EVIDENCE_ANNOTATION_CHARS;
  const eligible = annotations.filter(
    (annotation) =>
      (annotation.self_correction === undefined && options.wholeTextOnly !== true) ||
      annotation.text.length <= maxChars,
  );
  const selfCorrectionCount = annotations.filter((row) => row.self_correction !== undefined).length;
  const newestSelfCorrection = eligible.reduce<AuditedAnnotation | undefined>(
    (newest, row) =>
      (row.self_correction?.seq ?? 0) > (newest?.self_correction?.seq ?? 0) ? row : newest,
    undefined,
  );
  const selected = (
    newestSelfCorrection === undefined
      ? eligible
      : [newestSelfCorrection, ...eligible.filter((row) => row !== newestSelfCorrection)]
  ).slice(0, maxAnnotations);
  return { selected, maxChars, selfCorrectionCount, totalCount: annotations.length };
}

export function annotationPromptFields(
  annotations: readonly AuditedAnnotation[],
  options: AnnotationPromptOptions & { currentText?: string } = {},
): AnnotationPromptFields {
  return renderSelectedAnnotations(
    selectAnnotationPromptRows(annotations, options),
    options.currentText,
  );
}

export function renderSelectedAnnotations(
  {
    selected,
    maxChars,
    selfCorrectionCount,
    totalCount,
  }: ReturnType<typeof selectAnnotationPromptRows>,
  currentText?: string,
): AnnotationPromptFields {
  // target_changed compares the stored correction hash with the record text read for this page
  // build. A loaded body is reused, otherwise the reader batches the missing bodies for selected
  // self-written corrections. A failed lookup leaves the status null and is traced by the reader.
  // Quote status uses the same body or the earlier correction's text. A rewrite between retrieval
  // and annotation read can leave an older excerpt beside the status for the newer text. Each turn
  // reloads its records, including warm recall.
  const currentTextHash =
    currentText !== undefined && selected.some((row) => row.self_correction !== undefined)
      ? recordCorrectionTextHash(currentText)
      : undefined;
  return {
    ...(selected.length === 0
      ? {}
      : {
          annotations: selected.map((annotation) => {
            const date =
              annotation.written_at === undefined ? undefined : new Date(annotation.written_at);
            const correction = annotation.self_correction;
            const quotedText =
              correction?.targets_correction_seq == null
                ? currentText
                : annotation.quoted_correction_text;
            const targetChanged =
              currentTextHash === undefined
                ? null
                : currentTextHash !== correction?.target_text_hash;
            return {
              text: correction === undefined ? excerpt(annotation.text, maxChars) : annotation.text,
              ...(date === undefined || !Number.isFinite(date.getTime())
                ? {}
                : { written_at: date.toISOString() }),
              authority:
                annotation.authority === null || correction !== undefined
                  ? null
                  : excerpt(annotation.authority, MAX_EVIDENCE_AUTHORITY_CHARS),
              ...(correction === undefined
                ? {}
                : {
                    author: correction.writer.author,
                    seq: correction.seq,
                    original_record: correction.original_record,
                    target_changed: targetChanged,
                    ...(correction.targets_correction_seq === null
                      ? {}
                      : { targets_correction_seq: correction.targets_correction_seq }),
                    ...(targetChanged !== false && correction.corrects_span !== undefined
                      ? {
                          quoted_span_present:
                            targetChanged === null || quotedText === undefined
                              ? null
                              : quotedText.includes(correction.corrects_span),
                        }
                      : {}),
                    ...memoryDisclosurePayloadFields(correction.disclosure_label),
                  }),
            };
          }),
        }),
    ...(totalCount > selected.length ? { annotations_omitted: totalCount - selected.length } : {}),
    ...(selfCorrectionCount === 0 ? {} : { self_correction_count: selfCorrectionCount }),
  };
}

export function withSemanticNodeEvidenceContext<T extends { id: string; description?: string }>(
  reader: Pick<EpisodeEvidenceReader, "identityEventRepository">,
  nodes: readonly T[],
): Array<T & { evidence_context?: AnnotationPromptFields }> {
  const results = reader.identityEventRepository.listAnnotationsForTargets(
    nodes.map((node) => ({ record_type: "semantic_node", record_id: node.id })),
  );
  const annotations = new Map(results.map((target) => [target.record_id, target.annotations]));
  return nodes.map((node) => {
    const context = annotationPromptFields(annotations.get(node.id) ?? [], {
      currentText: node.description,
    });
    return Object.keys(context).length === 0 ? node : { ...node, evidence_context: context };
  });
}

function lineagePromptRow(
  episode: EpisodeWithEvidenceContext,
): Record<string, unknown> & AnnotationPromptFields {
  return {
    id: episode.id,
    kind: episode.episode_kind ?? "raw",
    span: { start_time: episode.start_time, end_time: episode.end_time },
    excerpt: excerpt(episode.narrative, MAX_EVIDENCE_LINEAGE_EXCERPT_CHARS),
    ...(episode.narrative.length > MAX_EVIDENCE_LINEAGE_EXCERPT_CHARS
      ? { excerpt_truncated: true }
      : {}),
    ...episode.evidence_context,
    ...memoryDisclosurePayloadFields(memoryDisclosureLabelFromEpisodeAccess(episode)),
  };
}

/** Opt-in read projection: never persisted, inherited by a derived record, or used to gate recall. */
export async function withEpisodeEvidenceContext(
  reader: EpisodeEvidenceReader,
  episodes: readonly Episode[],
  options: { includeVisibleLineage?: boolean } = {},
): Promise<EpisodeWithEvidenceContext[]> {
  const lineageIds =
    options.includeVisibleLineage === true
      ? episodes
          .filter((episode) => episode.episode_kind === "consolidation_version")
          .flatMap((episode) => episode.lineage.derived_from)
      : [];
  const lifecycle = reader.episodicRepository.getLifecycleMany([
    ...episodes.map((episode) => episode.id),
    ...lineageIds,
  ]);
  const visibleLineageById = new Map(
    episodes.map((episode) => [
      episode.id,
      options.includeVisibleLineage === true &&
      episode.episode_kind === "consolidation_version" &&
      lifecycle.get(episode.id) === "archived"
        ? [...new Set(episode.lineage.derived_from)].filter((id) => lifecycle.get(id) === "active")
        : [],
    ]),
  );
  const selectedLineageIds = [
    ...new Set(
      [...visibleLineageById.values()].flatMap((ids) =>
        ids.slice(0, MAX_EVIDENCE_LINEAGE_EPISODES),
      ),
    ),
  ];
  const lineage =
    selectedLineageIds.length === 0
      ? []
      : await reader.episodicRepository.getMany(selectedLineageIds);
  const annotationTargets = reader.identityEventRepository.listAnnotationsForTargets(
    [...episodes, ...lineage].map((episode) => ({ record_type: "episode", record_id: episode.id })),
  );
  const annotations = new Map(
    annotationTargets.map((target) => [target.record_id, target.annotations]),
  );
  const decorate = (episode: Episode): EpisodeWithEvidenceContext => {
    const status = lifecycle.get(episode.id) ?? "unknown";
    const context: EpisodeEvidenceContext = {
      ...(status === "active" ? {} : { lifecycle: status }),
      ...annotationPromptFields(annotations.get(episode.id) ?? [], {
        currentText: episode.narrative,
      }),
    };
    return Object.keys(context).length === 0 ? episode : { ...episode, evidence_context: context };
  };
  const lineageById = new Map(lineage.map((episode) => [episode.id, decorate(episode)]));
  return episodes.map((episode) => {
    const decorated = decorate(episode);
    const ids = visibleLineageById.get(episode.id) ?? [];
    if (ids.length > 0) {
      const rows = ids.slice(0, MAX_EVIDENCE_LINEAGE_EPISODES).flatMap((id) => {
        const source = lineageById.get(id);
        return source === undefined ? [] : [lineagePromptRow(source)];
      });
      decorated.evidence_context = {
        ...decorated.evidence_context!,
        visible_lineage: rows,
        ...(ids.length > rows.length ? { lineage_omitted: ids.length - rows.length } : {}),
      };
    }
    return decorated;
  });
}

export function episodeEvidencePromptRow(
  episode: EpisodeWithEvidenceContext,
  extra: Record<string, unknown> = {},
  extraContext: Pick<EpisodeEvidenceContext, "member_annotations"> = {},
): Record<string, unknown> {
  return {
    id: episode.id,
    title: episode.title,
    narrative: episode.narrative,
    ...extra,
    ...evidencePromptFields({ ...episode.evidence_context, ...extraContext }),
    ...memoryDisclosurePayloadFields(memoryDisclosureLabelFromEpisodeAccess(episode)),
  };
}
