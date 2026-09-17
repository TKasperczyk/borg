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
import {
  OPEN_QUESTION_TRUST_RANK,
  addEntry,
  cappedTrustRank,
  setSectionFraming,
} from "../section-buckets.js";
import { persistenceClassFromProvenance } from "../scope-resolver.js";
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

export function addOpenQuestionsSection(context: BuilderSectionContext): void {
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
            streamEntryIds: openQuestionStreamEntryIds(question),
            episodeIds: openQuestionEpisodeIds(question),
          },
          context.resolver,
        ),
      }),
    );
  }
}
