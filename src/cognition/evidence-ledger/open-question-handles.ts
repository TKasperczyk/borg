import type { OpenQuestion, OpenQuestionStatus } from "../../memory/self/index.js";
import type { EvidenceLedgerBuildInput } from "./builder-types.js";
import {
  combineScopes,
  scopeFromEpisodeIds,
  scopeFromStreamIds,
  type ScopeResolver,
} from "./scope-resolver.js";
import type { EvidenceLedgerSessionScope } from "./types.js";

export const LIFECYCLE_OPEN_QUESTION_STATUSES = [
  "resolved",
  "abandoned",
] as const satisfies readonly OpenQuestionStatus[];

export function openQuestionScope(
  question: OpenQuestion,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return combineScopes([
    scopeFromStreamIds(openQuestionStreamEntryIds(question), resolver),
    scopeFromEpisodeIds(openQuestionEpisodeIds(question), resolver),
  ]);
}

function openQuestionProvenanceEpisodeIds(question: OpenQuestion): readonly string[] {
  if (question.provenance?.kind === "episodes") {
    return question.provenance.episode_ids;
  }

  if (question.provenance?.kind === "online_reflector") {
    return question.provenance.evidence_episode_ids;
  }

  return [];
}

function openQuestionProvenanceStreamEntryIds(question: OpenQuestion): readonly string[] {
  return question.provenance?.kind === "online_reflector"
    ? question.provenance.evidence_stream_entry_ids
    : [];
}

export function openQuestionStreamEntryIds(question: OpenQuestion): readonly string[] {
  return [
    ...question.resolution_evidence_stream_entry_ids,
    ...openQuestionProvenanceStreamEntryIds(question),
  ];
}

export function openQuestionEpisodeIds(question: OpenQuestion): readonly string[] {
  return [
    ...question.related_episode_ids,
    ...question.resolution_evidence_episode_ids,
    ...openQuestionProvenanceEpisodeIds(question),
  ];
}

export function relevantOpenQuestionStreamIds(
  input: EvidenceLedgerBuildInput,
  resolver: ScopeResolver,
): Set<string> {
  const streamIds = new Set<string>();

  for (const entryId of resolver.streamEntriesById.keys()) {
    streamIds.add(entryId);
  }

  if (input.currentUserEntry !== undefined) {
    streamIds.add(input.currentUserEntry.id);
  }

  for (const item of input.retrievedEvidence) {
    for (const streamId of item.provenance?.streamIds ?? []) {
      streamIds.add(streamId);
    }
  }

  for (const result of input.retrievedEpisodes) {
    for (const streamId of result.episode.source_stream_ids) {
      streamIds.add(streamId);
    }

    for (const entry of result.citationChain) {
      streamIds.add(entry.id);
    }
  }

  return streamIds;
}

export function relevantOpenQuestionEpisodeIds(input: EvidenceLedgerBuildInput): Set<string> {
  const episodeIds = new Set<string>();

  for (const item of input.retrievedEvidence) {
    if (item.provenance?.episodeId !== undefined) {
      episodeIds.add(item.provenance.episodeId);
    }

    if (item.provenance?.parentEpisodeId !== undefined) {
      episodeIds.add(item.provenance.parentEpisodeId);
    }
  }

  for (const result of input.retrievedEpisodes) {
    episodeIds.add(result.episode.id);
  }

  return episodeIds;
}

export function openQuestionStateMetadata(
  question: OpenQuestion,
): Record<string, unknown> | undefined {
  if (question.status === "resolved") {
    return {
      resolution_note: question.resolution_note,
      resolved_at: question.resolved_at,
      resolution_evidence_episode_ids: question.resolution_evidence_episode_ids,
      resolution_evidence_stream_entry_ids: question.resolution_evidence_stream_entry_ids,
    };
  }

  if (question.status === "abandoned") {
    return {
      abandoned_reason: question.abandoned_reason,
      abandoned_at: question.abandoned_at,
    };
  }

  return undefined;
}
