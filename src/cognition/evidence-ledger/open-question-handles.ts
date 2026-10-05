import type { OpenQuestion, OpenQuestionStatus } from "../../memory/self/index.js";
import { formatRelativeAge } from "../../util/relative-time.js";
import type { EvidenceLedgerBuildInput } from "./builder-types.js";
import type { EpisodeId } from "../../util/ids.js";
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
  semanticSourceEpisodeIdsById?: ReadonlyMap<string, readonly EpisodeId[]>,
): EvidenceLedgerSessionScope {
  return combineScopes([
    scopeFromStreamIds(openQuestionStreamEntryIds(question), resolver),
    scopeFromEpisodeIds(openQuestionEpisodeIds(question, semanticSourceEpisodeIdsById), resolver),
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

export function openQuestionEpisodeIds(
  question: OpenQuestion,
  semanticSourceEpisodeIdsById?: ReadonlyMap<string, readonly EpisodeId[]>,
): readonly string[] {
  return [
    ...question.related_episode_ids,
    ...question.resolution_evidence_episode_ids,
    ...openQuestionProvenanceEpisodeIds(question),
    ...question.related_semantic_node_ids.flatMap(
      (nodeId) => semanticSourceEpisodeIdsById?.get(nodeId) ?? [],
    ),
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

export function relevantOpenQuestionSemanticNodeIds(input: EvidenceLedgerBuildInput): Set<string> {
  return new Set(input.retrievedSemantic?.matched_node_ids ?? []);
}

// A zero is self-describing only while nothing has stamped the row. Zero beside a live
// last_ruminated_at is a third state -- the loop selected this question and the pass recorded
// nothing -- and it renders identically to never-selected, so that case gets its own sentence
// rather than the suppression the other zero earns. Above zero the sentence carries the whole
// predicate, because the number there could be read as sufficient on its own.
function ruminationTicksNoteFor(
  ticks: number,
  lastRuminatedAt: number | null,
): Record<string, string> {
  if (ticks === 0) {
    if (lastRuminatedAt === null) {
      return {};
    }

    return {
      unresolved_rumination_ticks_note:
        "This count stands at zero while last_ruminated_at is set: an offline pass selected it and produced no note, and such a pass advances the stamp without advancing the count. A zero beside a live stamp is therefore a pass that recorded nothing, not a question the loop has never reached -- a null stamp is what the latter looks like.",
    };
  }

  return {
    unresolved_rumination_ticks_note:
      "Note-bearing offline passes and completed dormant-question wakes counted on this row while open, which is not recorded_rumination_notes beside it: a wake writes no note, and a merge hands a duplicate's notes over without its ticks, so either can be larger. An offline pass that recorded no note stamps the schedule without advancing it. The count on its own does not close it, and a count already past the threshold does not mean the other conditions are what kept it open: the dismissal is not evaluated at all on a pass that called the model for this question, so the model's still-open judgment stands and the count advances unchecked. Where it is evaluated, it also requires no cited episode created after the question; action and conversational progress are not measured by this predicate.",
  };
}

export function openQuestionStateMetadata(
  question: OpenQuestion,
  nowMs?: number,
  staleNoTractionTicks?: number,
  recordedRuminationNotes?: number,
): Record<string, unknown> | undefined {
  const recordedNotes =
    recordedRuminationNotes === undefined
      ? {}
      : { recorded_rumination_notes: recordedRuminationNotes };

  if (question.status === "open") {
    const ticks = question.unresolved_rumination_ticks;
    // A row with no ticks, no stamp and no notes carries no rumination signal, and a zero is
    // self-describing there -- which is why the ticks note is suppressed on the same condition.
    // Everywhere else the count is shown: it and the ticks are different quantities, and a merge
    // moves notes onto a row without moving the ticks that were counted for them.
    const silentOnRumination =
      ticks === 0 && question.last_ruminated_at === null && recordedRuminationNotes === 0;
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
      ...(silentOnRumination ? {} : recordedNotes),
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
      ...ruminationTicksNoteFor(ticks, question.last_ruminated_at),
    };
  }

  // Resolving a question zeroes unresolved_rumination_ticks and nulls last_ruminated_at in the same
  // write, so on a resolved row those two fields record the closure rather than the work, and reading
  // a null there as "never ruminated" is reading the write's own echo. The rumination notes are the
  // durable record: they are append-only and status-blind, so they outlive the row's closure, and
  // recorded_rumination_notes is their count where the repository was available to take it.
  const closedRuminationNote =
    "The rumination counter and last-ruminated stamp were zeroed and nulled by the write that closed this question, so they are not shown here: on a closed row they would describe the closure, not whether the loop ever worked the question. The notes those passes recorded outlive it; recorded_rumination_notes is the store's count of them, and tool.openQuestions.ruminations returns a capped tail rather than the whole record.";

  if (question.status === "resolved") {
    return {
      resolution_note: question.resolution_note,
      resolved_at: question.resolved_at,
      resolution_evidence_episode_ids: question.resolution_evidence_episode_ids,
      resolution_evidence_stream_entry_ids: question.resolution_evidence_stream_entry_ids,
      ...recordedNotes,
      rumination_record_note: closedRuminationNote,
    };
  }

  if (question.status === "abandoned") {
    return {
      abandoned_reason: question.abandoned_reason,
      abandoned_at: question.abandoned_at,
      unresolved_rumination_ticks: question.unresolved_rumination_ticks,
      last_ruminated_at:
        question.last_ruminated_at === null
          ? null
          : new Date(question.last_ruminated_at).toISOString(),
      ...recordedNotes,
      rumination_record_note:
        "Abandonment retains the rumination counter and last-pass stamp. Legacy closures may have cleared them; a zero or null on an abandoned row does not prove no rumination occurred. stale_no_traction means the tick threshold was met without a newer cited episode, not that action or conversational progress was assessed.",
    };
  }

  return undefined;
}
