import { describe, expect, it, vi } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import { createActionId, createOpenQuestionId, createStreamEntryId } from "../../util/ids.js";
import type { OpenQuestion } from "../self/index.js";
import { resolveOpenQuestionsForCompletedAction } from "./open-question-resolution.js";
import type { ActionRecord } from "./types.js";

describe("completed action open-question resolution", () => {
  it("resolves a directly linked open question when an action completes", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const question = harness.openQuestionsRepository.add({
        question: "Did the Atlas follow-up get sent?",
        urgency: 0.7,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      const streamEntryId = createStreamEntryId();
      const actionId = createActionId();

      harness.actionRepository.add({
        id: actionId,
        description: "Send the Atlas follow-up",
        actor: "borg",
        audience_entity_id: null,
        goal_id: null,
        open_question_id: question.id,
        state: "committed_to_do",
        confidence: 0.9,
        provenance_episode_ids: [],
        provenance_stream_entry_ids: [streamEntryId],
        created_at: harness.clock.now(),
        updated_at: harness.clock.now(),
        considering_at: null,
        committed_at: harness.clock.now(),
        scheduled_at: null,
        completed_at: null,
        not_done_at: null,
        unknown_at: null,
      });

      harness.actionRepository.update(actionId, {
        state: "completed",
      });

      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "resolved",
        resolution_evidence_stream_entry_ids: [streamEntryId],
        resolution_note: "Resolved by completed action: Send the Atlas follow-up",
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("resolves open questions linked through the completed action goal", async () => {
    const harness = await createOfflineTestHarness();

    try {
      const goal = harness.goalsRepository.add({
        description: "Finish the Madrid prep action",
        priority: 0.8,
        provenance: { kind: "manual" },
      });
      const linked = harness.openQuestionsRepository.add({
        question: "Was the Madrid prep action completed?",
        urgency: 0.6,
        goal_id: goal.id,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      const unrelated = harness.openQuestionsRepository.add({
        question: "Is a separate note still unresolved?",
        urgency: 0.6,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      const streamEntryId = createStreamEntryId();

      harness.actionRepository.add({
        id: createActionId(),
        description: "Complete the Madrid prep action",
        actor: "borg",
        audience_entity_id: null,
        goal_id: goal.id,
        open_question_id: null,
        state: "completed",
        confidence: 0.9,
        provenance_episode_ids: [],
        provenance_stream_entry_ids: [streamEntryId],
        created_at: harness.clock.now(),
        updated_at: harness.clock.now(),
        considering_at: null,
        committed_at: null,
        scheduled_at: null,
        completed_at: harness.clock.now(),
        not_done_at: null,
        unknown_at: null,
      });

      expect(harness.openQuestionsRepository.get(linked.id)?.status).toBe("resolved");
      expect(harness.openQuestionsRepository.get(unrelated.id)?.status).toBe("open");
    } finally {
      await harness.cleanup();
    }
  });

  it("throws CAS conflicts from the identity resolver", () => {
    const questionId = createOpenQuestionId();
    const streamEntryId = createStreamEntryId();
    const error = new IdentityCasMismatchError({
      recordType: "open_question",
      recordId: questionId,
      expectedVersion: 2,
    });
    const action = {
      id: createActionId(),
      description: "Complete the linked action",
      actor: "borg",
      audience_entity_id: null,
      open_question_id: questionId,
      goal_id: null,
      state: "completed",
      confidence: 1,
      provenance_episode_ids: [],
      provenance_stream_entry_ids: [streamEntryId],
      created_at: 1_000,
      updated_at: 1_000,
      considering_at: null,
      committed_at: null,
      scheduled_at: null,
      completed_at: 1_000,
      not_done_at: null,
      unknown_at: null,
      canonicalized_by_artifact_entry_id: null,
    } as ActionRecord;

    expect(() =>
      resolveOpenQuestionsForCompletedAction({
        action,
        openQuestionsRepository: {
          get: vi.fn(() => ({ id: questionId, status: "open" }) as OpenQuestion),
          listByGoal: vi.fn(() => []),
        },
        identityService: {
          resolveOpenQuestion: vi.fn(() => {
            throw error;
          }),
        },
      }),
    ).toThrow(error);
  });
});
