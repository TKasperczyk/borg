import type { OpenQuestion, OpenQuestionStatus } from "../../memory/self/index.js";
import { formatRelativeAge } from "../../util/relative-time.js";
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

  for (const entry of input.currentUserEntries ?? []) {
    streamIds.add(entry.id);
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
  nowMs?: number,
  staleNoTractionTicks?: number,
): Record<string, unknown> | undefined {
  if (question.status === "open") {
    const ticks = question.unresolved_rumination_ticks;
    return {
      created_at: new Date(question.created_at).toISOString(),
      last_touched: new Date(question.last_touched).toISOString(),
      ...(nowMs === undefined
        ? {}
        : {
            created_relative_age: formatRelativeAge(question.created_at, nowMs),
            last_touched_relative_age: formatRelativeAge(question.last_touched, nowMs),
          }),
      unresolved_rumination_ticks: ticks,
      last_ruminated_at:
        question.last_ruminated_at === null
          ? null
          : new Date(question.last_ruminated_at).toISOString(),
      ...(nowMs === undefined || question.last_ruminated_at === null
        ? {}
        : { last_ruminated_relative_age: formatRelativeAge(question.last_ruminated_at, nowMs) }),
      ...(staleNoTractionTicks === undefined
        ? {}
        : { dismissal_threshold_ticks: staleNoTractionTicks }),
      // The count is inert at zero and self-describing there, so the sentence is spent only on
      // rows where the number is live and could be read as sufficient on its own.
      ...(ticks === 0
        ? {}
        : {
            unresolved_rumination_ticks_note:
              "Recorded rumination notes from offline passes that ended with this question still open; a pass that recorded no note stamps the schedule without advancing this count. The count on its own does not close it, because that dismissal also requires no episode created after the question citing it and no active action against it. Zero on another row means no note has been recorded against that question, not that it is fresh.",
          }),
    };
  }

  // Closing a question zeroes unresolved_rumination_ticks and nulls last_ruminated_at in the same
  // write, so on a closed row those two fields record the closure rather than the work, and reading
  // a null there as "never ruminated" is reading the write's own echo. The rumination notes are the
  // durable record: they are append-only and status-blind, so they outlive the row's closure.
  const closedRuminationNote =
    "The rumination counter and last-ruminated stamp were zeroed and nulled by the write that closed this question, so they are not shown here: on a closed row they would describe the closure, not whether the loop ever worked the question. Absence of a stamp is therefore not evidence of absence of rumination. The passes that recorded one persist as rumination notes, which are append-only and survive the question closing; tool.openQuestions.ruminations reads them.";

  if (question.status === "resolved") {
    return {
      resolution_note: question.resolution_note,
      resolved_at: question.resolved_at,
      resolution_evidence_episode_ids: question.resolution_evidence_episode_ids,
      resolution_evidence_stream_entry_ids: question.resolution_evidence_stream_entry_ids,
      rumination_record_note: closedRuminationNote,
    };
  }

  if (question.status === "abandoned") {
    return {
      abandoned_reason: question.abandoned_reason,
      abandoned_at: question.abandoned_at,
      rumination_record_note: closedRuminationNote,
    };
  }

  return undefined;
}
