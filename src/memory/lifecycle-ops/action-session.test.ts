import { describe, expect, it, vi } from "vitest";

import type { ActionRecord, ActionRecordListFilter, ActionRecordPatch } from "../actions/index.js";
import { createActionId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import {
  archiveStaleAction,
  expireSessionScopedActions,
  rolloverNextSessionActions,
} from "./index.js";

function makeAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createActionId(),
    description: overrides.description ?? "Finish the session agenda",
    actor: overrides.actor ?? "user",
    audience_entity_id: overrides.audience_entity_id ?? null,
    goal_id: overrides.goal_id ?? null,
    open_question_id: overrides.open_question_id ?? null,
    state: overrides.state ?? "committed_to_do",
    confidence: overrides.confidence ?? 0.9,
    provenance_episode_ids: overrides.provenance_episode_ids ?? [],
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [createStreamEntryId()],
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
    considering_at: overrides.considering_at ?? null,
    committed_at: overrides.committed_at ?? nowMs,
    scheduled_at: overrides.scheduled_at ?? null,
    completed_at: overrides.completed_at ?? null,
    not_done_at: overrides.not_done_at ?? null,
    expired_at: overrides.expired_at ?? null,
    archived_at: overrides.archived_at ?? null,
    unknown_at: overrides.unknown_at ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    session_scope: overrides.session_scope ?? null,
    session_anchor_id: overrides.session_anchor_id ?? null,
    last_referenced_at_ms: overrides.last_referenced_at_ms ?? nowMs,
    last_referenced_turn_counter: overrides.last_referenced_turn_counter ?? null,
  };
}

function makeRepository(records: ActionRecord[]) {
  return {
    list: vi.fn((filter: ActionRecordListFilter = {}) =>
      records.filter((record) => {
        if (filter.states !== undefined && !filter.states.includes(record.state)) {
          return false;
        }

        if ("sessionScope" in filter && record.session_scope !== filter.sessionScope) {
          return false;
        }

        if ("sessionAnchorId" in filter && record.session_anchor_id !== filter.sessionAnchorId) {
          return false;
        }

        return true;
      }),
    ),
    get: vi.fn((id: ActionRecord["id"]) => records.find((record) => record.id === id) ?? null),
    update: vi.fn((id: ActionRecord["id"], patch: ActionRecordPatch) => {
      const index = records.findIndex((record) => record.id === id);
      const current = records[index];

      if (current === undefined) {
        throw new Error(`Unknown action ${id}`);
      }

      records[index] = { ...current, ...patch };
    }),
    records,
  };
}

describe("action session lifecycle operations", () => {
  it("expires only current-session actions anchored to the closing session", () => {
    const sessionId = createSessionId();
    const otherSessionId = createSessionId();
    const scoped = makeAction({ session_scope: "current_session", session_anchor_id: sessionId });
    const otherSession = makeAction({
      session_scope: "current_session",
      session_anchor_id: otherSessionId,
    });
    const durable = makeAction({ session_scope: null });
    const repository = makeRepository([scoped, otherSession, durable]);
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];

    const result = expireSessionScopedActions({
      sessionId,
      repository,
      nowMs: 5_000,
      tracer: {
        enabled: true,
        emit: (event, data) => events.push({ event, data }),
      },
    });

    expect(result.status).toBe("success");
    expect(repository.records).toContainEqual(
      expect.objectContaining({
        id: scoped.id,
        state: "expired",
        expired_at: 5_000,
      }),
    );
    expect(repository.records).toContainEqual(
      expect.objectContaining({
        id: otherSession.id,
        state: "committed_to_do",
      }),
    );
    expect(repository.records).toContainEqual(
      expect.objectContaining({
        id: durable.id,
        state: "committed_to_do",
      }),
    );
    expect(events).toContainEqual({
      event: "action_session_scope.expired",
      data: expect.objectContaining({
        actions_expired_at_session_close: 1,
      }),
    });
  });

  it("rolls next-session actions forward before the old session expires", () => {
    const oldSessionId = createSessionId();
    const newSessionId = createSessionId();
    const nextSession = makeAction({
      session_scope: "next_session",
      session_anchor_id: oldSessionId,
    });
    const durable = makeAction({ session_scope: null });
    const repository = makeRepository([nextSession, durable]);

    const result = rolloverNextSessionActions({
      fromSessionId: oldSessionId,
      toSessionId: newSessionId,
      repository,
      nowMs: 6_000,
    });

    expect(result.status).toBe("success");
    expect(repository.records).toContainEqual(
      expect.objectContaining({
        id: nextSession.id,
        session_scope: "current_session",
        session_anchor_id: newSessionId,
        updated_at: 6_000,
      }),
    );
    expect(repository.records).toContainEqual(
      expect.objectContaining({
        id: durable.id,
        session_scope: null,
        session_anchor_id: null,
      }),
    );
  });

  it("returns a typed conflict result when expiration hits a CAS conflict", () => {
    const sessionId = createSessionId();
    const action = makeAction({ session_scope: "current_session", session_anchor_id: sessionId });
    const repository = makeRepository([action]);
    const conflict = new IdentityCasMismatchError({
      recordType: "action",
      recordId: action.id,
      expectedVersion: 2,
    });
    repository.update.mockImplementationOnce(() => {
      throw conflict;
    });

    const result = expireSessionScopedActions({
      sessionId,
      repository,
      nowMs: 7_000,
    });

    expect(result.status).toBe("conflict");
    if (result.status !== "conflict") {
      throw new Error(`Expected conflict, got ${result.status}`);
    }
    expect(result.error).toBe(conflict);
    expect(result.value).toMatchObject({
      expiredActionIds: [],
      conflictedActionIds: [action.id],
      skippedActionIds: [],
    });
    expect(repository.records[0]).toMatchObject({
      state: "committed_to_do",
      expired_at: null,
    });
  });

  it("archives a stale action as a recoverable terminal state", () => {
    const action = makeAction({ last_referenced_turn_counter: 1 });
    const repository = makeRepository([action]);

    const result = archiveStaleAction({
      actionId: action.id,
      repository,
      nowMs: 8_000,
      turnId: "turn_archive",
    });

    expect(result.status).toBe("success");
    expect(repository.records[0]).toMatchObject({
      state: "archived",
      archived_at: 8_000,
      updated_at: 8_000,
    });
  });
});
