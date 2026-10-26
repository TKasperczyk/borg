import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "../entry-metadata.js";
import {
  LIFECYCLE_OPEN_QUESTION_STATUSES,
  openQuestionEpisodeIds,
  openQuestionScope,
  openQuestionStateMetadata,
  openQuestionStreamEntryIds,
  relevantOpenQuestionEpisodeIds,
  relevantOpenQuestionSemanticNodeIds,
  relevantOpenQuestionStreamIds,
} from "../open-question-handles.js";
import {
  OPEN_QUESTION_TRUST_RANK,
  addEntry,
  cappedTrustRank,
  setSectionFraming,
} from "../section-buckets.js";
import { persistenceClassFromProvenance, scopeFromStreamIds } from "../scope-resolver.js";
import type { EpisodeId } from "../../../util/ids.js";
import { openQuestionMemoryDisclosureLabel } from "../../../memory/common/disclosure-serializers.js";
import { OPEN_QUESTION_STATUSES } from "../../../memory/self/index.js";

// Both admissions below are small: the handle match asks only about this turn's ids, and the draw
// that carries open rows is a scored top-N. So the rows here have never been the register, and a
// reader holding only them cannot tell a short draw from a short store -- the same page is what a
// register of three and a register of hundreds both look like. The count is taken at build time and
// says so, and the statuses are the enum's, so widening it cannot leave this line quietly partial.
function storePopulationFraming(counts: Record<string, number>): string {
  const parts = OPEN_QUESTION_STATUSES.map((status) => `${status} ${counts[status] ?? 0}`);

  return `my open_questions store at this read: ${parts.join(", ")}. Rows here are this turn's admissions out of that, so a status with rows in the store can render none here.`;
}

export async function addOpenQuestionsSection(context: BuilderSectionContext): Promise<void> {
  const storeCounts = context.repos.openQuestions?.countByStatus();

  if (storeCounts !== undefined) {
    setSectionFraming(context.buckets, "open_questions", {
      text: storePopulationFraming(storeCounts),
    });
  }

  const questionsById = new Map(
    context.input.openQuestions.map((question) => [question.id, question]),
  );

  if (context.repos.openQuestions !== undefined) {
    const streamIds = relevantOpenQuestionStreamIds(context.input, context.resolver);
    const episodeIds = relevantOpenQuestionEpisodeIds(context.input);

    for (const question of context.repos.openQuestions.findByHandles({
      streamEntryIds: [...streamIds],
      episodeIds: [...episodeIds],
      semanticNodeIds: [...relevantOpenQuestionSemanticNodeIds(context.input)],
      statuses: LIFECYCLE_OPEN_QUESTION_STATUSES,
    })) {
      questionsById.set(question.id, question);
    }
  }

  // Resolve existing provenance in batches. Semantic IDs stay in the machinery;
  // the prompt uses the same stream citation handles as other ledger evidence.
  const nodeIds = [
    ...new Set(
      [...questionsById.values()].flatMap((question) => question.related_semantic_node_ids),
    ),
  ];
  const nodes =
    nodeIds.length === 0
      ? []
      : ((await context.repos.semanticNodes?.getMany?.(nodeIds, { includeArchived: true })) ?? []);
  const semanticSourceEpisodeIdsById = new Map(
    nodes.flatMap((node) => (node === null ? [] : [[node.id, node.source_episode_ids] as const])),
  );
  const episodeIds = [
    ...new Set(
      [...questionsById.values()].flatMap((question) =>
        openQuestionEpisodeIds(question, semanticSourceEpisodeIdsById),
      ),
    ),
  ];
  const episodeSourceStreamIdsById = new Map(context.resolver.episodeSourceStreamIdsById);
  const episodeScopesById = new Map(context.resolver.episodeScopesById);
  const missingEpisodeIds = episodeIds.filter((id) => !episodeSourceStreamIdsById.has(id));
  const episodes =
    missingEpisodeIds.length === 0
      ? []
      : ((await context.repos.episodes?.getMany(missingEpisodeIds as EpisodeId[])) ?? []);
  for (const episode of episodes) {
    episodeSourceStreamIdsById.set(episode.id, episode.source_stream_ids);
    episodeScopesById.set(
      episode.id,
      scopeFromStreamIds(episode.source_stream_ids, context.resolver),
    );
  }
  const resolver = { ...context.resolver, episodeSourceStreamIdsById, episodeScopesById };

  for (const question of questionsById.values()) {
    const sourceEpisodeIds = openQuestionEpisodeIds(question, semanticSourceEpisodeIdsById);
    const sourceStreamIds = [
      ...new Set([
        ...openQuestionStreamEntryIds(question),
        ...sourceEpisodeIds.flatMap((id) => episodeSourceStreamIdsById.get(id) ?? []),
      ]),
    ];
    const disclosureLabel = openQuestionMemoryDisclosureLabel(question);
    addEntry(
      context.buckets,
      "open_questions",
      cappedTrustRank({
        id: `open_question:${question.id}`,
        source_type: "system_metadata",
        session_scope: openQuestionScope(question, resolver, semanticSourceEpisodeIdsById),
        actor: "memory",
        trust_rank: OPEN_QUESTION_TRUST_RANK,
        text: question.question,
        ...(sourceStreamIds.length === 0 ? {} : { citations: sourceStreamIds }),
        value: question.source,
        state: appendMemoryDisclosureState({
          state: question.status,
          disclosureLabel,
        }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: openQuestionStateMetadata(
            question,
            context.nowMs,
            context.options.openQuestionStaleNoTractionTicks,
            context.repos.openQuestions?.countRuminations(question.id),
          ),
          disclosureLabel,
        }),
        taint: "none",
        ...persistenceClassFromProvenance(
          {
            streamEntryIds: sourceStreamIds,
            episodeIds: sourceEpisodeIds,
          },
          resolver,
        ),
      }),
    );
  }
}
