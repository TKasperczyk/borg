import { existsSync } from "node:fs";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ActionRepository,
  createActionRecordsTableSchema,
} from "../src/memory/actions/repository.js";
import { actionRecordSchema } from "../src/memory/actions/types.js";
import { EntityRepository } from "../src/memory/commitments/repository.js";
import { LanceDbStore } from "../src/storage/lancedb/index.js";
import {
  createActionId,
  createEntityId,
  createStreamEntryId,
  createSessionId,
  createSharedStateEntryId,
} from "../src/util/ids.js";
import { quoteSqlString } from "../src/storage/codecs.js";
import { repairFixture } from "./audited-data-repair-test-support.js";
import { loadRepairPlan, repairPendingPath } from "./audited-data-repair.js";
import {
  applyActionActorRepair,
  planActionActorRepair,
  type ActorRepairPlan,
} from "./repair-action-actor.js";

const cleanups: Array<() => void | Promise<void>> = [];
afterEach(async () => {
  vi.restoreAllMocks();
  while (cleanups.length) await cleanups.pop()!();
});
async function fixture(withVectors = true) {
  const f = repairFixture();
  cleanups.push(f.cleanup);
  const entities = new EntityRepository(f);
  const to = entities.add({ canonicalName: "Destination" }).id,
    from = createEntityId();
  const repo = new ActionRepository(f);
  const action = actionRecordSchema.parse({
    id: createActionId(),
    description: "Opaque historical action",
    actor: from,
    audience_entity_id: null,
    state: "completed",
    confidence: 0.8,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [createStreamEntryId()],
    created_at: 100,
    updated_at: 200,
    considering_at: null,
    committed_at: null,
    scheduled_at: null,
    completed_at: 200,
    not_done_at: null,
    expired_at: null,
    archived_at: null,
    unknown_at: null,
  });
  repo.add(action);
  const other = { ...action, id: createActionId(), actor: to };
  repo.add(other);
  let table;
  if (withVectors) {
    const store = new LanceDbStore({ uri: join(f.dataDir, "lancedb") });
    cleanups.push(async () => {
      await store.close();
    });
    table = await store.openTable({
      name: "action_records",
      schema: createActionRecordsTableSchema(3),
    });
    await table.upsert(
      [action, other].map((a) => ({
        id: a.id,
        description: a.description,
        actor: a.actor,
        state: a.state,
        audience_entity_id: null,
        updated_at: a.updated_at,
        embedding: [0.125, -0.5, 0.75],
      })),
      { on: "id" },
    );
  }
  return {
    ...f,
    entities,
    repo,
    from,
    to,
    action: repo.get(action.id)!,
    other: repo.get(other.id)!,
    table,
  };
}
describe("orphan action actor repair", () => {
  it("lists only exact orphan actors and applies audited repository/mirror updates without re-embedding", async () => {
    const f = await fixture();
    const plan = await planActionActorRepair(f, 2000);
    expect(plan.rows).toHaveLength(1);
    expect(plan.rows[0]?.before).toEqual(f.action);
    expect(f.repo.get(f.action.id)).toEqual(f.action);
    expect(f.audit.list()).toEqual([]);
    await applyActionActorRepair(plan);
    expect(f.repo.get(f.action.id)).toEqual({ ...f.action, actor: f.to });
    expect(f.repo.get(f.other.id)).toEqual(f.other);
    await f.table!.checkoutLatest();
    const native = await f.table!.list();
    const mirror = native.find((r) => r.id === f.action.id)!;
    expect(mirror.actor).toBe(f.to);
    expect(Array.from(mirror.embedding as Iterable<number>)).toEqual([0.125, -0.5, 0.75]);
    expect(f.audit.list()).toHaveLength(2);
    expect(f.audit.list().find((a) => a.action.endsWith(f.action.id))?.reversal).toMatchObject({
      actor: f.from,
      before: f.action,
    });
    const rerun = await planActionActorRepair(f);
    expect(rerun.runId).toBeNull();
    await applyActionActorRepair(rerun);
    expect(f.audit.list()).toHaveLength(2);
  });
  it("refuses a source entity row, including a redirect", async () => {
    const f = await fixture(false);
    f.entities.add({ id: f.from, canonicalName: "Existing" });
    await expect(planActionActorRepair(f)).rejects.toThrow("entity-graph redirect");
    f.db.prepare("UPDATE entities SET merged_into = ? WHERE id = ?").run(f.to, f.from);
    await expect(planActionActorRepair(f)).rejects.toThrow("entity-graph redirect");
  });
  it("refuses missing destination and invalid handles", async () => {
    const f = await fixture(false);
    await expect(planActionActorRepair({ ...f, to: createEntityId() })).rejects.toThrow(
      "--to has no entities row",
    );
    await expect(planActionActorRepair({ ...f, from: "typo" })).rejects.toThrow();
  });
  it("works for a SQLite-only fixture without inventing a vector store", async () => {
    const f = await fixture(false);
    const plan = await planActionActorRepair(f);
    expect(plan.mirror_table_present).toBe(false);
    await applyActionActorRepair(plan);
    expect(f.repo.get(f.action.id)?.actor).toBe(f.to);
    expect(existsSync(join(f.dataDir, "lancedb"))).toBe(false);
  });
  it("repairs all five matching rows and reports absent mirrors", async () => {
    const f = await fixture();
    for (let i = 0; i < 4; i++) f.repo.add({ ...f.action, id: createActionId() });
    const plan = await planActionActorRepair(f);
    expect(plan.summary).toEqual({ changed: 5, mirror_updates: 1 });
    await applyActionActorRepair(plan);
    expect(f.repo.list({ actor: f.from })).toEqual([]);
    expect(f.audit.list()).toHaveLength(6);
  });
  it.each(["apply", "after_journal", "after_sqlite"])(
    "preserves five rich rows with sibling metadata changes (%s), including CLI --resume",
    async (mode) => {
      const f = await fixture();
      for (let i = 0; i < 5; i++) {
        f.repo.add({
          ...f.action,
          id: i === 0 ? f.action.id : createActionId(),
          state: i < 4 ? "archived" : "scheduled",
          updated_at: 500 + i,
          archived_at: i < 4 ? 400 + i : null,
          expired_at: i < 4 ? 300 + i : null,
          session_scope: "current_session",
          session_anchor_id: createSessionId(),
          last_referenced_at_ms: 500 + i,
          last_referenced_turn_counter: 40 + i,
          last_referenced_turn_global: 90 + i,
          canonicalized_by_artifact_entry_id: createSharedStateEntryId(),
        });
      }
      const records = f.repo.list({ actor: f.from });
      await f.table!.upsert(
        records.map((a) => ({
          id: a.id,
          description: a.description,
          actor: a.actor,
          state: a.state,
          audience_entity_id: a.audience_entity_id,
          updated_at: a.updated_at,
          embedding: [0.125, -0.5, 0.75],
        })),
        { on: "id" },
      );
      const plan = await planActionActorRepair(f, 2000);
      const first = plan.rows[0]!.before,
        sibling = plan.rows[4]!.before;
      // Fixture-only trigger models metadata changes to a later row from an earlier write.
      // The real rehearsal failure was omitted-field defaults, not a production trigger.
      f.db.exec(`CREATE TRIGGER fixture_touch_sibling AFTER UPDATE OF actor ON action_records
      WHEN OLD.id = ${quoteSqlString(first.id)} AND NEW.actor != OLD.actor
      BEGIN UPDATE action_records SET updated_at = updated_at + 10,
        last_referenced_turn_global = last_referenced_turn_global + 1
        WHERE id = ${quoteSqlString(sibling.id)}; END`);
      if (mode === "apply") await applyActionActorRepair(plan);
      else {
        await expect(
          applyActionActorRepair(plan, {
            checkpoint: (point) => {
              if (point === mode) throw new Error("interrupted");
            },
          }),
        ).rejects.toThrow("interrupted");
        const result = spawnSync(
          process.execPath,
          [
            "--import",
            "tsx",
            "scripts/repair-action-actor.ts",
            "--data-dir",
            f.dataDir,
            "--apply",
            "--resume",
          ],
          { cwd: process.cwd(), encoding: "utf8", timeout: 20_000 },
        );
        expect(result.status, result.stderr).toBe(0);
      }
      for (const before of records) {
        expect(f.repo.get(before.id)).toEqual({
          ...before,
          actor: f.to,
          ...(before.id === sibling.id
            ? {
                updated_at: before.updated_at + 10,
                last_referenced_turn_global: before.last_referenced_turn_global! + 1,
              }
            : {}),
        });
      }
      const audit = f.audit.list().find((a) => a.action.endsWith(sibling.id))!;
      expect(audit.targets.before).toMatchObject({
        actor: f.from,
        updated_at: sibling.updated_at + 10,
        archived_at: sibling.archived_at,
        last_referenced_turn_global: sibling.last_referenced_turn_global! + 1,
      });
      expect(audit.reversal.before).toEqual(audit.targets.before);
      expect(f.audit.list()).toHaveLength(6);
      await f.table!.checkoutLatest();
      const mirrors = await f.table!.list();
      for (const row of plan.rows) {
        const mirror = mirrors.find((r) => r.id === row.before.id)!;
        expect({ ...mirror, embedding: Array.from(mirror.embedding as Iterable<number>) }).toEqual(
          row.mirror_after,
        );
      }
      const repeated = await planActionActorRepair(f);
      expect(repeated.summary.changed).toBe(0);
      await applyActionActorRepair(repeated);
      expect(f.audit.list()).toHaveLength(6);
    },
    30_000,
  );
  it.each(["before_apply", "inside_transaction"])(
    "refuses a genuinely changed actor %s",
    async (when) => {
      const f = await fixture(false);
      for (let i = 0; i < 4; i++) f.repo.add({ ...f.action, id: createActionId() });
      const plan = await planActionActorRepair(f);
      const first = plan.rows[0]!.before,
        last = plan.rows[4]!.before;
      const changedActor = f.entities.add({ canonicalName: "Different actor" }).id;
      if (when === "before_apply") f.repo.add({ ...last, actor: changedActor });
      else
        f.db.exec(`CREATE TRIGGER fixture_change_actor AFTER UPDATE OF actor ON action_records
      WHEN OLD.id = ${quoteSqlString(first.id)} AND NEW.actor != OLD.actor
      BEGIN UPDATE action_records SET actor = ${quoteSqlString(changedActor)} WHERE id = ${quoteSqlString(last.id)}; END`);
      await expect(applyActionActorRepair(plan)).rejects.toThrow("action identity/actor");
      expect(f.audit.list()).toEqual([]);
      for (const row of plan.rows)
        expect(f.repo.get(row.before.id)).toEqual({
          ...row.before,
          ...(when === "before_apply" && row.before.id === last.id ? { actor: changedActor } : {}),
        });
    },
  );
  it.each(["after_journal", "after_sqlite", "after_mirror", "after_completion", "after_archive"])(
    "resumes %s without duplicate audits or vector changes",
    async (point) => {
      const f = await fixture();
      const plan = await planActionActorRepair(f, 2000);
      await expect(
        applyActionActorRepair(plan, {
          checkpoint: (at) => {
            if (at.split(":")[0] === point) throw new Error("interrupted");
          },
        }),
      ).rejects.toThrow("interrupted");
      await applyActionActorRepair(loadRepairPlan<ActorRepairPlan>(f.dataDir, plan.kind), {
        resume: true,
      });
      expect(f.audit.list()).toHaveLength(2);
      expect(f.repo.get(f.action.id)?.actor).toBe(f.to);
      await f.table!.checkoutLatest();
      const mirror = (await f.table!.list()).find((r) => r.id === f.action.id)!;
      expect(mirror.actor).toBe(f.to);
      expect(Array.from(mirror.embedding as Iterable<number>)).toEqual([0.125, -0.5, 0.75]);
    },
  );
  it("rejects stale vectors before SQLite or journaling", async () => {
    const f = await fixture();
    const plan = await planActionActorRepair(f);
    const row = (await f.table!.list()).find((r) => r.id === f.action.id)!;
    await f.table!.upsert([{ ...row, embedding: [1, 0, 0] }], { on: "id" });
    await expect(applyActionActorRepair(plan)).rejects.toThrow("Changed since planning");
    expect(f.repo.get(f.action.id)).toEqual(f.action);
    expect(f.audit.list()).toEqual([]);
    expect(existsSync(repairPendingPath(f.dataDir, plan.kind))).toBe(false);
  });
  it("propagates mirror write failures and resumes the journaled SQL commit", async () => {
    const f = await fixture();
    const plan = await planActionActorRepair(f, 2000);
    const spy = vi
      .spyOn(ActionRepository.prototype, "syncRepairedActorMirror")
      .mockRejectedValueOnce(new Error("mirror offline"));
    await expect(applyActionActorRepair(plan)).rejects.toThrow("mirror offline");
    expect(f.audit.list()).toHaveLength(1);
    spy.mockRestore();
    await applyActionActorRepair(loadRepairPlan<ActorRepairPlan>(f.dataDir, plan.kind), {
      resume: true,
    });
    expect(f.audit.list()).toHaveLength(2);
  });
  it("rechecks source entity absence before apply", async () => {
    const f = await fixture(false);
    const plan = await planActionActorRepair(f);
    f.entities.add({ id: f.from, canonicalName: "Created later" });
    await expect(applyActionActorRepair(plan)).rejects.toThrow("entity-graph redirect");
    expect(f.repo.get(f.action.id)).toEqual(f.action);
  });
});
