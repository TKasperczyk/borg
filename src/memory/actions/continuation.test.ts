import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase } from "../../storage/sqlite/index.js";
import { createEntityId } from "../../util/ids.js";
import { makeActionRecord } from "../../test-support/factories/memory.js";
import { makeLockedSharedStateEntry } from "../../test-support/factories/shared-state.js";
import { canonicalizeActionWithSharedStateEntry } from "../lifecycle-ops/index.js";
import {
  reconcileSharedStateCanonicalizations,
  findUnsettledSharedStateReconciliation,
} from "../../cognition/shared-state/reconciliation.js";
import { runSharedStateArtifactRetryOnlyReconciliation } from "../../cognition/lifecycle/turn-phase/reconciliation-phase.js";
import { ActionRepository, actionMigrations, createActionRecordsTableSchema } from "./index.js";
import { SharedStateRepository, sharedStateMigrations } from "../shared-state/index.js";

const cleanups: Array<() => void | Promise<void>> = [];
afterEach(async () => {
  while (cleanups.length) await cleanups.pop()!();
});
function fixture(actor: "borg" | "participant" = "borg") {
  const db = openDatabase(":memory:", { migrations: actionMigrations });
  cleanups.push(() => db.close());
  const self = createEntityId(),
    owner = actor === "borg" ? self : createEntityId();
  const onCompleted = vi.fn();
  const repository = new ActionRepository({ db, onCompleted });
  const action = makeActionRecord({ actor: actor === "borg" ? "borg" : owner });
  repository.add(action);
  const entry = makeLockedSharedStateEntry({
    owner_entity_id: owner,
    text: "Nouvelle formulation — 完了",
    last_updated_at: 2_000,
    canonicalizes: {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [action.id],
      open_question_ids: [],
      action_selections: [
        { action_id: action.id, candidate_ids: [action.id], self_entity_id: self },
      ],
    },
  });
  return { db, repository, action, entry, onCompleted };
}

describe("shared-state semantic action continuations", () => {
  it.each(["borg", "participant"] as const)(
    "keeps wording, provenance and offered-id audit for %s; replay is a no-op",
    (actor) => {
      const f = fixture(actor);
      const input = {
        actionId: f.action.id,
        entry: f.entry,
        repository: f.repository,
        nowMs: 2_000,
        turnId: "canonicalization",
      };
      expect(canonicalizeActionWithSharedStateEntry(input).status).toBe("success");
      expect(canonicalizeActionWithSharedStateEntry(input)).toMatchObject({
        status: "no_op",
        reason: "already_applied",
      });
      expect(f.repository.get(f.action.id)).toMatchObject({
        state: "completed",
        description: f.action.description,
        goal_id: f.action.goal_id,
        canonicalized_by_artifact_entry_id: f.entry.id,
      });
      const history = f.repository.readUpdateHistory(f.action.id, 5);
      expect(history.total).toBe(1);
      expect(history.updates[0]).toMatchObject({
        text: f.entry.text,
        source_stream_entry_ids: [
          ...new Set([
            ...f.entry.provenance_stream_entry_ids,
            ...f.entry.last_updated_stream_entry_ids,
          ]),
        ],
        selection_audit: {
          shared_state_entry_id: f.entry.id,
          candidate_ids: [f.action.id],
          chosen_action_id: f.action.id,
        },
      });
      expect(f.onCompleted).not.toHaveBeenCalled();
    },
  );

  it.each(["actor_mismatch", "not_offered", "terminal"] as const)(
    "rejects %s without completing or changing the row",
    (reason) => {
      const f = fixture();
      if (reason === "actor_mismatch") f.entry.owner_entity_id = createEntityId();
      if (reason === "not_offered") f.entry.canonicalizes.action_selections = undefined;
      if (reason === "terminal") f.repository.update(f.action.id, { state: "declined" });
      const before = f.repository.get(f.action.id);
      const emit = vi.fn();
      expect(
        canonicalizeActionWithSharedStateEntry({
          actionId: f.action.id,
          entry: f.entry,
          repository: f.repository,
          nowMs: 2_000,
          turnId: "reject",
          tracer: { enabled: true, emit },
        }),
      ).toMatchObject({ status: "no_op", reason });
      expect(f.repository.get(f.action.id)).toEqual(before);
      expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(0);
      expect(emit).toHaveBeenCalledWith(
        "action_state.continuation_rejected",
        expect.objectContaining({ reason }),
      );
    },
  );

  it("returns an applied continuation and its callback failure after commit", () => {
    const f = fixture();
    const error = new Error("question resolution failed");
    f.onCompleted.mockImplementation(() => {
      throw error;
    });
    const result = f.repository.appendExtractionUpdate(
      f.action.id,
      {
        ...f.action,
        state: "completed",
        updated_at: 2_000,
        selection_audit: {
          turn_id: "direct",
          candidate_index: 0,
          candidate_ids: [f.action.id],
          candidate_omitted_count: 0,
          chosen_action_id: f.action.id,
          rejection_reason: null,
          supplied_goal_ids: [],
          chosen_goal_id: null,
          goal_rejection_reason: null,
        },
      },
      "direct",
    );
    expect(result).toMatchObject({
      status: "applied",
      record: { id: f.action.id, state: "completed" },
      completionFailures: [{ actionId: f.action.id, error }],
    });
    expect(f.repository.get(f.action.id)?.state).toBe("completed");
    expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(1);
  });

  it("refreshes the mirror after shared-state continuation while suppressing only the completion hook", async () => {
    const f = fixture();
    const tempDir = mkdtempSync(join(tmpdir(), "borg-continuation-"));
    const store = new LanceDbStore({ uri: join(tempDir, "lancedb") });
    const table = await store.openTable({
      name: "action_records",
      schema: createActionRecordsTableSchema(4),
    });
    const embed = vi.fn(async () => Float32Array.from([1, 0, 0, 0]));
    const repository = new ActionRepository({
      db: f.db,
      table,
      embeddingClient: {
        embed,
        embedBatch: async (texts) => Promise.all(texts.map(() => embed())),
      },
      onCompleted: f.onCompleted,
    });
    cleanups.push(async () => {
      await repository.waitForPendingEmbeddings();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });
    repository.add(f.action);
    await repository.waitForPendingEmbeddings();
    embed.mockClear();
    const audit = {
      turn_id: "rollback",
      candidate_index: 0,
      candidate_ids: [f.action.id],
      candidate_omitted_count: 0,
      chosen_action_id: f.action.id,
      rejection_reason: null,
      supplied_goal_ids: [],
      chosen_goal_id: null,
      goal_rejection_reason: null,
    };
    // A failure after the continuation write must not schedule a mirror write from the inner path.
    f.db.exec(
      "CREATE TRIGGER fail_decision BEFORE INSERT ON action_extraction_decisions BEGIN SELECT RAISE(ABORT, 'audit failed'); END;",
    );
    expect(() =>
      repository.applyExtractionDecision({
        records: [{ ...f.action, state: "completed", updated_at: 2_000, selection_audit: audit }],
        audit: {
          turn_id: "rollback",
          candidate_ids: [f.action.id],
          candidate_omitted_count: 0,
          referenced_action_ids: [],
          applied_reference_ids: [],
          ignored_reference_ids: [],
          decisions: [],
        },
        nowMs: 2_000,
        turnCounter: 1,
      }),
    ).toThrow("audit failed");
    await repository.waitForPendingEmbeddings();
    expect(embed).not.toHaveBeenCalled();
    expect(repository.get(f.action.id)?.state).toBe(f.action.state);
    expect((await table.list())[0]?.state).toBe(f.action.state);
    f.db.exec("DROP TRIGGER fail_decision");
    expect(
      canonicalizeActionWithSharedStateEntry({
        actionId: f.action.id,
        entry: f.entry,
        repository,
        nowMs: 2_000,
        turnId: "shared",
      }).status,
    ).toBe("success");
    await repository.waitForPendingEmbeddings();
    const sql = repository.get(f.action.id)!;
    const mirror = (await table.list())[0]!;
    expect(sql.state).toBe("completed");
    expect(mirror).toMatchObject({
      id: sql.id,
      description: sql.description,
      actor: sql.actor,
      state: sql.state,
      updated_at: sql.updated_at,
    });
    expect(embed).toHaveBeenCalledOnce();
    expect(f.onCompleted).not.toHaveBeenCalled();
  });

  it("carries the stored offer proof through retry-only reconciliation", () => {
    const f = fixture();
    const sharedDb = openDatabase(":memory:", { migrations: sharedStateMigrations });
    cleanups.push(() => sharedDb.close());
    const shared = new SharedStateRepository({ db: sharedDb });
    shared.upsert(
      f.entry.audience_entity_id,
      [{ ...f.entry, type: "add", state_key: f.entry.state_key! }],
      { now: 2_000 },
    );
    shared.upsert(
      f.entry.audience_entity_id,
      [
        {
          type: "update",
          id: f.entry.id,
          state_key: f.entry.state_key!,
          text: f.entry.text,
          last_updated_stream_entry_ids: f.entry.last_updated_stream_entry_ids,
          canonicalizes: {
            goal_ids: [],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        },
      ],
      { now: 2_001 },
    );
    const previousArtifact = new SharedStateRepository({ db: sharedDb }).get(
      f.entry.audience_entity_id,
    )!;
    expect(previousArtifact.entries[0]?.canonicalizes.action_selections).toEqual(
      f.entry.canonicalizes.action_selections,
    );
    const repositories = { actionRepository: f.repository };
    const unsettled = findUnsettledSharedStateReconciliation({
      previousArtifact,
      repositories,
      nowMs: 2_000,
    })!;
    expect(unsettled.entries[0]?.canonicalizes.action_selections).toEqual(
      f.entry.canonicalizes.action_selections,
    );
    runSharedStateArtifactRetryOnlyReconciliation({
      unsettledReconciliation: unsettled,
      repositories,
      sourceTrustValidator: () => ({ allowed: true }),
      nowMs: 2_000,
      tracer: { enabled: false, includePayloads: false, emit: () => {} },
      turnId: "retry",
    });
    expect(f.repository.readUpdateHistory(f.action.id, 5).total).toBe(1);
    expect(
      reconcileSharedStateCanonicalizations({ entries: [f.entry], repositories }).actions_retired,
    ).toBe(0);
  });
});
