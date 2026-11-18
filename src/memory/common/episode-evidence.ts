import type { Episode } from "../episodic/types.js";
import type { EpisodicRepository } from "../episodic/repository.js";
import type { IdentityEventRepository } from "../identity/repository.js";
import type { AuditedAnnotation } from "../identity/annotations.js";
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

export function annotationPromptFields(
  annotations: readonly AuditedAnnotation[],
): AnnotationPromptFields {
  if (annotations.length === 0) return {};
  return {
    annotations: annotations.slice(0, MAX_EVIDENCE_ANNOTATIONS_PER_RECORD).map((annotation) => {
      const date =
        annotation.written_at === undefined ? undefined : new Date(annotation.written_at);
      return {
        text: excerpt(annotation.text, MAX_EVIDENCE_ANNOTATION_CHARS),
        ...(date === undefined || !Number.isFinite(date.getTime())
          ? {}
          : { written_at: date.toISOString() }),
        authority:
          annotation.authority === null
            ? null
            : excerpt(annotation.authority, MAX_EVIDENCE_AUTHORITY_CHARS),
      };
    }),
    ...(annotations.length > MAX_EVIDENCE_ANNOTATIONS_PER_RECORD
      ? { annotations_omitted: annotations.length - MAX_EVIDENCE_ANNOTATIONS_PER_RECORD }
      : {}),
  };
}

export function withSemanticNodeEvidenceContext<T extends { id: string }>(
  reader: Pick<EpisodeEvidenceReader, "identityEventRepository">,
  nodes: readonly T[],
): Array<T & { evidence_context?: AnnotationPromptFields }> {
  const results = reader.identityEventRepository.listAnnotationsForTargets(
    nodes.map((node) => ({ record_type: "semantic_node", record_id: node.id })),
  );
  const annotations = new Map(results.map((target) => [target.record_id, target.annotations]));
  return nodes.map((node) => {
    const context = annotationPromptFields(annotations.get(node.id) ?? []);
    return Object.keys(context).length === 0 ? node : { ...node, evidence_context: context };
  });
}

function lineagePromptRow(episode: EpisodeWithEvidenceContext): Record<string, unknown> {
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
      ...annotationPromptFields(annotations.get(episode.id) ?? []),
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
