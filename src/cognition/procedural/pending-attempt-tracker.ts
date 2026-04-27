import type { ActionResult } from "../action/index.js";
import type { PerceptionResult } from "../types.js";
import type { SkillSelectionResult } from "../../memory/procedural/index.js";
import {
  PENDING_PROCEDURAL_ATTEMPT_TTL_TURNS,
  PENDING_PROCEDURAL_ATTEMPTS_LIMIT,
  type PendingProceduralAttempt,
  type WorkingMemory,
} from "../../memory/working/index.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";

function compactTurnText(text: string, maxLength: number): string {
  const compacted = text.replace(/\s+/g, " ").trim();

  return compacted.length <= maxLength ? compacted : compacted.slice(0, maxLength).trimEnd();
}

export type PendingProceduralAttemptTrackerInput = {
  isUserTurn: boolean;
  userMessage: string;
  perception: PerceptionResult;
  actionResult: ActionResult;
  selectedSkill: SkillSelectionResult | null;
  reflectedWorkingMemory: WorkingMemory;
  persistedUserEntryId: StreamEntryId;
  persistedAgentEntryId: StreamEntryId;
  audienceEntityId: EntityId | null;
};

export class PendingProceduralAttemptTracker {
  update(input: PendingProceduralAttemptTrackerInput): PendingProceduralAttempt[] {
    // Sprint 53: bounded list of pending procedural attempts.
    // Reflection retires only grounded success/failure outcomes;
    // unclear ones survive here until they age out (TTL) or get
    // graded on a later turn.
    const carriedAttempts = input.reflectedWorkingMemory.pending_procedural_attempts.filter(
      (attempt) =>
        input.reflectedWorkingMemory.turn_counter - attempt.turn_counter <
        PENDING_PROCEDURAL_ATTEMPT_TTL_TURNS,
    );
    const shouldAppendNewAttempt = input.isUserTurn && input.perception.mode === "problem_solving";
    const newAttempt = shouldAppendNewAttempt
      ? {
          problem_text: compactTurnText(input.userMessage, 1_000),
          approach_summary:
            input.selectedSkill?.skill.approach ??
            (compactTurnText(input.actionResult.response, 1_000) || "No explicit approach stated."),
          selected_skill_id: input.selectedSkill?.skill.id ?? null,
          source_stream_ids: [input.persistedUserEntryId, input.persistedAgentEntryId],
          turn_counter: input.reflectedWorkingMemory.turn_counter,
          audience_entity_id: input.audienceEntityId,
        }
      : null;
    const combinedAttempts =
      newAttempt === null ? carriedAttempts : [...carriedAttempts, newAttempt];
    // Cap by dropping oldest if needed.
    return combinedAttempts.length > PENDING_PROCEDURAL_ATTEMPTS_LIMIT
      ? combinedAttempts.slice(-PENDING_PROCEDURAL_ATTEMPTS_LIMIT)
      : combinedAttempts;
  }
}
