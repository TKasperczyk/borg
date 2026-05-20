import type { SharedStateEntry } from "../../memory/decision-artifacts/index.js";
import {
  canonicalizeOpenQuestionWithSharedStateEntry,
  type LifecycleTracer,
} from "../../memory/lifecycle-ops/index.js";
import type { OpenQuestionsRepository } from "../../memory/self/index.js";
import type { OpenQuestionId } from "../../util/ids.js";
import { errorMessage, type SharedStateReconciliationResult } from "./reconciliation-summary.js";

export { isTerminalOpenQuestionStatus } from "../../memory/lifecycle-ops/index.js";

export function reconcileOpenQuestionCanonicalizations(input: {
  entry: SharedStateEntry;
  openQuestionIds: readonly OpenQuestionId[];
  repository:
    | (Pick<OpenQuestionsRepository, "resolve"> & Partial<Pick<OpenQuestionsRepository, "get">>)
    | undefined;
  retiredOpenQuestions: Set<OpenQuestionId>;
  result: SharedStateReconciliationResult;
  tracer?: LifecycleTracer;
  turnId?: string;
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
      const result = canonicalizeOpenQuestionWithSharedStateEntry({
        openQuestionId,
        entry: input.entry,
        repository: input.repository,
        tracer: input.tracer,
        turnId: input.turnId,
      });

      if (result.status === "no_op" && result.reason === "missing") {
        input.result.open_questions_resolved_skipped += 1;
        input.result.errors.push({
          channel: "open_question",
          id: openQuestionId,
          artifactEntryId: input.entry.id,
          message: `Unknown open question id: ${openQuestionId}`,
        });
        continue;
      }

      if (result.status === "no_op") {
        input.result.open_questions_resolved_skipped += 1;
        continue;
      }

      if (result.status === "conflict") {
        input.result.open_questions_resolved_skipped += 1;
        input.result.errors.push({
          channel: "open_question",
          id: openQuestionId,
          artifactEntryId: input.entry.id,
          message: errorMessage(result.error),
        });
        continue;
      }

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
