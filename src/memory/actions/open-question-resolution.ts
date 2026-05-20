import type { IdentityService } from "../identity/index.js";
import { resolveOpenQuestionThroughIdentityService } from "../lifecycle-ops/index.js";
import type { OpenQuestion, OpenQuestionsRepository } from "../self/index.js";

import type { ActionRecord } from "./types.js";

type ResolveCompletedActionOpenQuestionsOptions = {
  action: ActionRecord;
  openQuestionsRepository: Pick<OpenQuestionsRepository, "get" | "listByGoal">;
  identityService: Pick<IdentityService, "resolveOpenQuestion">;
};

const ACTION_OPEN_QUESTION_PROVENANCE = {
  kind: "online",
  process: "action_state",
} as const;

function resolutionEvidence(action: ActionRecord): {
  resolution_evidence_episode_ids: ActionRecord["provenance_episode_ids"];
  resolution_evidence_stream_entry_ids: ActionRecord["provenance_stream_entry_ids"];
} {
  return {
    resolution_evidence_episode_ids: action.provenance_episode_ids,
    resolution_evidence_stream_entry_ids: action.provenance_stream_entry_ids,
  };
}

function resolutionNote(action: ActionRecord): string {
  return `Resolved by completed action: ${action.description}`;
}

function appendOpenQuestion(
  questions: OpenQuestion[],
  seen: Set<OpenQuestion["id"]>,
  question: OpenQuestion | null,
): void {
  if (question === null || question.status !== "open" || seen.has(question.id)) {
    return;
  }

  seen.add(question.id);
  questions.push(question);
}

export function resolveOpenQuestionsForCompletedAction(
  options: ResolveCompletedActionOpenQuestionsOptions,
): OpenQuestion[] {
  const { action } = options;
  const questions: OpenQuestion[] = [];
  const seen = new Set<OpenQuestion["id"]>();

  if (action.open_question_id !== null) {
    appendOpenQuestion(
      questions,
      seen,
      options.openQuestionsRepository.get(action.open_question_id),
    );
  }

  if (action.goal_id !== null) {
    for (const question of options.openQuestionsRepository.listByGoal({
      goalId: action.goal_id,
      statuses: ["open"],
      limit: 100,
    })) {
      appendOpenQuestion(questions, seen, question);
    }
  }

  const resolved: OpenQuestion[] = [];

  for (const question of questions) {
    const result = resolveOpenQuestionThroughIdentityService({
      openQuestionId: question.id,
      identityService: options.identityService,
      resolution: {
        ...resolutionEvidence(action),
        resolution_note: resolutionNote(action),
      },
      provenance: ACTION_OPEN_QUESTION_PROVENANCE,
      options: {
        throughReview: true,
        reason: "completed_action",
      },
    });

    if (result.status === "conflict") {
      throw result.error;
    }

    if (result.status === "success" && result.value.result.status === "applied") {
      resolved.push(result.value.result.record);
    }
  }

  return resolved;
}
