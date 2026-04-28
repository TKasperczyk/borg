import type { AuditLog } from "../audit-log.js";
import type {
  SkillContextStatsRecord,
  SkillRepository,
  SkillSplitPartInput,
} from "../../memory/procedural/index.js";
import type {
  ReviewQueueItem,
  SkillSplitReviewApplyResult,
  SkillSplitReviewHandler,
  SkillSplitReviewPayload,
} from "../../memory/semantic/index.js";
import type { WorkingMemoryStore } from "../../memory/working/index.js";
import type { Clock } from "../../util/clock.js";
import { createMaintenanceRunId, DEFAULT_SESSION_ID } from "../../util/ids.js";

export type SkillSplitReviewHandlerOptions = {
  skillRepository: SkillRepository;
  auditLog: AuditLog;
  clock: Clock;
  workingMemoryStore?: WorkingMemoryStore;
};

function normalizedDistinct(values: readonly string[]): boolean {
  return new Set(values.map((value) => value.trim().toLowerCase())).size === values.length;
}

function splitPartsFromPayload(payload: SkillSplitReviewPayload): SkillSplitPartInput[] {
  return payload.proposed_children.map((child) => ({
    applies_when: child.problem.trim(),
    approach: child.approach.trim(),
    target_contexts: child.context_stats.map((stats) => stats.context_key),
  }));
}

function movedContextStatsFromPayload(
  payload: SkillSplitReviewPayload,
): SkillContextStatsRecord[] {
  return payload.proposed_children.flatMap((child) => child.context_stats);
}

function clearPendingSkillReferences(
  workingMemoryStore: WorkingMemoryStore | undefined,
  skillId: SkillSplitReviewPayload["original_skill_id"],
  clock: Clock,
): void {
  if (workingMemoryStore === undefined) {
    return;
  }

  try {
    const workingMemory = workingMemoryStore.load(DEFAULT_SESSION_ID);
    let changed = false;
    const pendingProceduralAttempts = workingMemory.pending_procedural_attempts.map((attempt) => {
      if (attempt.selected_skill_id !== skillId) {
        return attempt;
      }

      changed = true;
      return {
        ...attempt,
        selected_skill_id: null,
      };
    });

    if (!changed) {
      return;
    }

    workingMemoryStore.save({
      ...workingMemory,
      pending_procedural_attempts: pendingProceduralAttempts,
      updated_at: clock.now(),
    });
  } catch {
    // The split itself is authoritative. Pending working-memory references are
    // best-effort cleanup and will naturally age out if the store is unavailable.
  }
}

function recordReviewedSplitAttempt(
  options: SkillSplitReviewHandlerOptions,
  payload: SkillSplitReviewPayload,
): void {
  options.skillRepository.recordSplitAttemptAndClearClaim({
    skillId: payload.original_skill_id,
    attemptedAt: options.clock.now(),
    claimedAt: payload.cooldown.claimed_at,
  });
}

export function createSkillSplitReviewHandler(
  options: SkillSplitReviewHandlerOptions,
): SkillSplitReviewHandler {
  return {
    async accept(
      _item: ReviewQueueItem,
      payload: SkillSplitReviewPayload,
    ): Promise<SkillSplitReviewApplyResult> {
      const current = options.skillRepository.get(payload.original_skill_id);

      if (current === null) {
        return {
          status: "rejected",
          reason: `Unknown skill id: ${payload.original_skill_id}`,
        };
      }

      if (current.status !== "active") {
        recordReviewedSplitAttempt(options, payload);
        return {
          status: "rejected",
          reason: `Skill already superseded: ${payload.original_skill_id}`,
        };
      }

      if (!normalizedDistinct(payload.proposed_children.map((child) => child.label))) {
        recordReviewedSplitAttempt(options, payload);
        return {
          status: "rejected",
          reason: "Skill split child labels must be distinct",
        };
      }

      const parts = splitPartsFromPayload(payload);
      const movedContextStats = movedContextStatsFromPayload(payload);
      const split = await options.skillRepository.supersedeWithSplits({
        skillId: payload.original_skill_id,
        parts,
        supersededAt: options.clock.now(),
      });

      if (split === null) {
        recordReviewedSplitAttempt(options, payload);
        return {
          status: "rejected",
          reason: `Skill split no longer applies: ${payload.original_skill_id}`,
        };
      }

      clearPendingSkillReferences(
        options.workingMemoryStore,
        payload.original_skill_id,
        options.clock,
      );
      options.auditLog.record({
        run_id: createMaintenanceRunId(),
        process: "procedural-synthesizer",
        action: "skill_split",
        targets: {
          originalSkillId: payload.original_skill_id,
          newSkillIds: split.created.map((skill) => skill.id),
        },
        reversal: {
          originalSkill: split.previous,
          createdSkills: split.created,
          movedContextStats,
        },
      });

      return {
        status: "applied",
        newSkillIds: split.created.map((skill) => skill.id),
      };
    },

    reject(_item: ReviewQueueItem, payload: SkillSplitReviewPayload): void {
      recordReviewedSplitAttempt(options, payload);
    },
  };
}
