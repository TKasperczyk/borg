import type { SharedStateEntry } from "../../memory/decision-artifacts/index.js";
import type { OpenQuestionsRepository } from "../../memory/self/index.js";
import type { OpenQuestionId } from "../../util/ids.js";
import { errorMessage, type SharedStateReconciliationResult } from "./reconciliation-summary.js";

export function isTerminalOpenQuestionStatus(status: string): boolean {
  return status === "resolved" || status === "abandoned";
}

export function reconcileOpenQuestionCanonicalizations(input: {
  entry: SharedStateEntry;
  openQuestionIds: readonly OpenQuestionId[];
  repository:
    | (Pick<OpenQuestionsRepository, "resolve"> & Partial<Pick<OpenQuestionsRepository, "get">>)
    | undefined;
  retiredOpenQuestions: Set<OpenQuestionId>;
  result: SharedStateReconciliationResult;
}): void {
  for (const openQuestionId of input.openQuestionIds) {
    input.result.open_questions_resolved_attempted += 1;

    if (input.retiredOpenQuestions.has(openQuestionId)) {
      input.result.open_questions_resolved_skipped += 1;
      continue;
    }

    if (input.repository === undefined) {
      input.result.open_questions_resolved_skipped += 1;
      continue;
    }

    try {
      const openQuestion = input.repository.get?.(openQuestionId) ?? null;

      if (openQuestion !== null && isTerminalOpenQuestionStatus(openQuestion.status)) {
        input.result.open_questions_resolved_skipped += 1;
        continue;
      }

      input.repository.resolve(
        openQuestionId,
        {
          resolution_evidence_stream_entry_ids: input.entry.last_updated_stream_entry_ids,
          resolution_note: `resolved_by_artifact_entry_id=${input.entry.id}`,
        },
        {
          resolvedByArtifactEntryId: input.entry.id,
        },
      );
      input.retiredOpenQuestions.add(openQuestionId);
      input.result.open_questions_retired += 1;
      input.result.open_questions_resolved_succeeded += 1;
    } catch (error) {
      input.result.errors.push({
        channel: "open_question",
        id: openQuestionId,
        artifactEntryId: input.entry.id,
        message: errorMessage(error),
      });
    }
  }
}
