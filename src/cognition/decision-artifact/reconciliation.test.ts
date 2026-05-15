import { describe, expect, it, vi } from "vitest";

import { ActionRepository, actionMigrations } from "../../memory/actions/index.js";
import { CommitmentRepository, commitmentMigrations } from "../../memory/commitments/index.js";
import type { DecisionArtifactEntry } from "../../memory/decision-artifacts/index.js";
import { OpenQuestionsRepository, selfMigrations } from "../../memory/self/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createActionId,
  createCommitmentId,
  createDecisionArtifactEntryId,
  createEntityId,
  createGoalId,
  createOpenQuestionId,
  createStreamEntryId,
} from "../../util/ids.js";
import { reconcileDecisionArtifactCanonicalizations } from "./reconciliation.js";

function lockedEntry(overrides: Partial<DecisionArtifactEntry> = {}): DecisionArtifactEntry {
  const streamEntryId = createStreamEntryId();

  return {
    id: overrides.id ?? createDecisionArtifactEntryId(),
    audience_entity_id: overrides.audience_entity_id ?? createEntityId(),
    kind: overrides.kind ?? "locked",
    text: overrides.text ?? "Granada is locked for 3 nights",
    owner_entity_id: overrides.owner_entity_id ?? null,
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [streamEntryId],
    last_updated_stream_entry_ids: overrides.last_updated_stream_entry_ids ?? [streamEntryId],
    created_at: overrides.created_at ?? 1_000,
    last_updated_at: overrides.last_updated_at ?? 1_000,
    superseded_by_id: overrides.superseded_by_id ?? null,
    rank: overrides.rank ?? 0,
    canonicalizes: overrides.canonicalizes ?? {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [],
      open_question_ids: [],
    },
  };
}

describe("reconcileDecisionArtifactCanonicalizations", () => {
  it("retires canonicalized state through existing repository APIs", () => {
    const goalId = createGoalId();
    const commitmentId = createCommitmentId();
    const actionId = createActionId();
    const openQuestionId = createOpenQuestionId();
    const entry = lockedEntry({
      canonicalizes: {
        goal_ids: [goalId],
        commitment_ids: [commitmentId],
        action_ids: [actionId],
        open_question_ids: [openQuestionId],
      },
    });
    const goalsRepository = {
      updateStatus: vi.fn(),
    };
    const commitmentRepository = {
      revoke: vi.fn(() => ({ id: commitmentId }) as never),
    };
    const actionRepository = {
      update: vi.fn(),
    };
    const openQuestionsRepository = {
      resolve: vi.fn(),
    };

    const result = reconcileDecisionArtifactCanonicalizations({
      entries: [entry],
      repositories: {
        goalsRepository,
        commitmentRepository,
        actionRepository,
        openQuestionsRepository,
      },
    });

    expect(result).toMatchObject({
      goals_retired: 1,
      commitments_retired: 1,
      actions_retired: 1,
      open_questions_retired: 1,
      errors: [],
    });
    expect(goalsRepository.updateStatus).toHaveBeenCalledWith(
      goalId,
      "done",
      {
        kind: "online",
        process: "decision_artifact_reconciliation",
      },
      {
        canonicalizedByArtifactEntryId: entry.id,
      },
    );
    expect(commitmentRepository.revoke).toHaveBeenCalledWith(
      commitmentId,
      `canonicalized_by_artifact_entry_id=${entry.id}`,
      {
        kind: "online",
        process: "decision_artifact_reconciliation",
      },
      undefined,
      {
        canonicalizedByArtifactEntryId: entry.id,
      },
    );
    expect(actionRepository.update).toHaveBeenCalledWith(
      actionId,
      {
        state: "completed",
        canonicalized_by_artifact_entry_id: entry.id,
      },
      {
        skipSideEffects: true,
      },
    );
    expect(openQuestionsRepository.resolve).toHaveBeenCalledWith(
      openQuestionId,
      {
        resolution_evidence_stream_entry_ids: entry.last_updated_stream_entry_ids,
        resolution_note: `resolved_by_artifact_entry_id=${entry.id}`,
      },
      {
        resolvedByArtifactEntryId: entry.id,
      },
    );
  });

  it("suppresses action completion side effects when explicitly canonicalizing linked open questions", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(selfMigrations, actionMigrations),
    });
    const clock = new FixedClock(10_000);
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    let actionCompletionHookCalls = 0;
    const actionRepository = new ActionRepository({
      db,
      clock,
      onCompleted: (record) => {
        actionCompletionHookCalls += 1;
        if (record.open_question_id !== null) {
          openQuestionsRepository.resolve(record.open_question_id, {
            resolution_evidence_stream_entry_ids: record.provenance_stream_entry_ids,
            resolution_note: "resolved by action hook",
          });
        }
      },
    });

    try {
      const source = createStreamEntryId();
      const question = openQuestionsRepository.add({
        question: "Is Granada locked?",
        urgency: 0.6,
        provenance: {
          kind: "system",
        },
        source: "reflection",
      });
      const actionId = createActionId();
      actionRepository.add({
        id: actionId,
        description: "Track Granada decision",
        actor: "borg",
        audience_entity_id: null,
        goal_id: null,
        open_question_id: question.id,
        state: "committed_to_do",
        confidence: 0.9,
        provenance_episode_ids: [],
        provenance_stream_entry_ids: [source],
        created_at: clock.now(),
        updated_at: clock.now(),
        considering_at: null,
        committed_at: clock.now(),
        scheduled_at: null,
        completed_at: null,
        not_done_at: null,
        unknown_at: null,
        canonicalized_by_artifact_entry_id: null,
      });
      const entry = lockedEntry({
        last_updated_stream_entry_ids: [source],
        canonicalizes: {
          goal_ids: [],
          commitment_ids: [],
          action_ids: [actionId],
          open_question_ids: [question.id],
        },
      });

      const result = reconcileDecisionArtifactCanonicalizations({
        entries: [entry],
        repositories: {
          actionRepository,
          openQuestionsRepository,
        },
      });

      expect(result).toMatchObject({
        actions_retired: 1,
        open_questions_retired: 1,
        errors: [],
      });
      expect(actionCompletionHookCalls).toBe(0);
      expect(openQuestionsRepository.get(question.id)).toMatchObject({
        status: "resolved",
        resolution_note: `resolved_by_artifact_entry_id=${entry.id}`,
        resolved_by_artifact_entry_id: entry.id,
      });
    } finally {
      db.close();
    }
  });

  it("reports a missing canonicalized commitment as a reconciliation error", () => {
    const commitmentId = createCommitmentId();
    const entry = lockedEntry({
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [commitmentId],
        action_ids: [],
        open_question_ids: [],
      },
    });
    const commitmentRepository = {
      revoke: vi.fn(() => null),
    };

    const result = reconcileDecisionArtifactCanonicalizations({
      entries: [entry],
      repositories: {
        commitmentRepository,
      },
    });

    expect(result.commitments_retired).toBe(0);
    expect(result.errors).toEqual([
      {
        channel: "commitment",
        id: commitmentId,
        artifactEntryId: entry.id,
        message: `Unknown commitment id: ${commitmentId}`,
      },
    ]);
  });

  it("counts already terminal action canonicalizations as skipped", () => {
    const actionId = createActionId();
    const entry = lockedEntry({
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [],
        action_ids: [actionId],
        open_question_ids: [],
      },
    });
    const actionRepository = {
      get: vi.fn(() => ({ id: actionId, state: "completed" }) as never),
      update: vi.fn(),
    };

    const result = reconcileDecisionArtifactCanonicalizations({
      entries: [entry],
      repositories: {
        actionRepository,
      },
    });

    expect(result).toMatchObject({
      actions_retired: 0,
      actions_completed_attempted: 1,
      actions_completed_succeeded: 0,
      actions_completed_skipped: 1,
      errors: [],
    });
    expect(actionRepository.update).not.toHaveBeenCalled();
  });

  it("skips unmaterialized expired commitment canonicalizations", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const clock = new FixedClock(1_000);
    const commitmentRepository = new CommitmentRepository({
      db,
      clock,
    });

    try {
      const expired = commitmentRepository.add({
        type: "promise",
        directiveFamily: "expired artifact fixture",
        directive: "Use the expired artifact fixture.",
        priority: 5,
        provenance: { kind: "manual" },
        createdAt: 500,
        expiresAt: 900,
      });
      const revoke = vi.spyOn(commitmentRepository, "revoke");
      const entry = lockedEntry({
        canonicalizes: {
          goal_ids: [],
          commitment_ids: [expired.id],
          action_ids: [],
          open_question_ids: [],
        },
      });

      const result = reconcileDecisionArtifactCanonicalizations({
        entries: [entry],
        repositories: {
          commitmentRepository,
        },
        nowMs: clock.now(),
      });

      expect(result).toMatchObject({
        commitments_retired: 0,
        commitments_revoked_attempted: 1,
        commitments_revoked_succeeded: 0,
        commitments_revoked_skipped: 1,
        errors: [],
      });
      expect(revoke).not.toHaveBeenCalled();
      expect(commitmentRepository.get(expired.id)).toMatchObject({
        expired_at: null,
        revoked_at: null,
      });
    } finally {
      db.close();
    }
  });

  it("ignores non-locked entries", () => {
    const goalId = createGoalId();
    const goalsRepository = {
      updateStatus: vi.fn(),
    };

    const result = reconcileDecisionArtifactCanonicalizations({
      entries: [
        lockedEntry({
          kind: "live",
          canonicalizes: {
            goal_ids: [goalId],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        }),
      ],
      repositories: {
        goalsRepository,
      },
    });

    expect(result.goals_retired).toBe(0);
    expect(goalsRepository.updateStatus).not.toHaveBeenCalled();
  });

  it.todo(
    "retires stale plan-branch commitments when a replacement branch is operationalized: lock 5-city itinerary, pivot to 3-anchor route skipping a city, operationalize 3-anchor via booked Renfe legs canonicalized into the artifact, assert old 5-city commitments transition to superseded/revoked without explicit canonicalizes references",
  );
});
