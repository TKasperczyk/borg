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
  relevantOpenQuestionStreamIds,
} from "../open-question-handles.js";
import { OPEN_QUESTION_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import { persistenceClassFromProvenance } from "../scope-resolver.js";
import { openQuestionMemoryDisclosureLabel } from "../../../memory/common/disclosure-serializers.js";

export function addOpenQuestionsSection(context: BuilderSectionContext): void {
  const questionsById = new Map(
    context.input.openQuestions.map((question) => [question.id, question]),
  );

  if (context.repos.openQuestions !== undefined) {
    const streamIds = relevantOpenQuestionStreamIds(context.input, context.resolver);
    const episodeIds = relevantOpenQuestionEpisodeIds(context.input);

    for (const question of context.repos.openQuestions.findByHandles({
      streamEntryIds: [...streamIds],
      episodeIds: [...episodeIds],
      statuses: LIFECYCLE_OPEN_QUESTION_STATUSES,
    })) {
      questionsById.set(question.id, question);
    }
  }

  for (const question of questionsById.values()) {
    const disclosureLabel = openQuestionMemoryDisclosureLabel(question);
    addEntry(
      context.buckets,
      "open_questions",
      cappedTrustRank({
        id: `open_question:${question.id}`,
        source_type: "system_metadata",
        session_scope: openQuestionScope(question, context.resolver),
        actor: "memory",
        trust_rank: OPEN_QUESTION_TRUST_RANK,
        text: question.question,
        value: question.source,
        state: appendMemoryDisclosureState({
          state: question.status,
          disclosureLabel,
        }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: openQuestionStateMetadata(question, context.nowMs),
          disclosureLabel,
        }),
        taint: "none",
        ...persistenceClassFromProvenance(
          {
            streamEntryIds: openQuestionStreamEntryIds(question),
            episodeIds: openQuestionEpisodeIds(question),
          },
          context.resolver,
        ),
      }),
    );
  }
}
