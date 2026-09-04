import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import { FixedClock, ManualClock } from "../../util/clock.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { IdentityCasMismatchError, ProvenanceError } from "../../util/errors.js";
import {
  createEntityId,
  createGoalId,
  createStreamEntryId,
  createTraitId,
  type EpisodeId,
} from "../../util/ids.js";
import { expectedRecordVersion } from "../common/cas.js";
import { identityMigrations, IdentityEventRepository } from "../identity/index.js";
import { selfMigrations } from "./migrations.js";
import {
  GOAL_TURN_ROLLBACK_REASON,
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
} from "./repository.js";

describe("self repositories", () => {
  const manualProvenance = { kind: "manual" } as const;
  const episodeProvenance = {
    kind: "episodes" as const,
    episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
  };
  const mergeEpisodeIds = [
    "ep_aaaaaaaaaaaaaaaa",
    "ep_bbbbbbbbbbbbbbbb",
    "ep_cccccccccccccccc",
    "ep_dddddddddddddddd",
    "ep_eeeeeeeeeeeeeeee",
    "ep_ffffffffffffffff",
    "ep_1111111111111111",
    "ep_2222222222222222",
    "ep_3333333333333333",
    "ep_4444444444444444",
  ] as EpisodeId[];

  function reinforceTraitWithEpisodes(
    traits: TraitsRepository,
    label: string,
    episodeIds: readonly EpisodeId[],
    options: { delta?: number; firstTimestamp?: number } = {},
  ) {
    let record: ReturnType<TraitsRepository["reinforce"]> | null = null;
    const delta = options.delta ?? 0.1;
    const firstTimestamp = options.firstTimestamp ?? 1_000;

    for (const [index, episodeId] of episodeIds.entries()) {
      record = traits.reinforce({
        label,
        delta,
        provenance: {
          kind: "episodes",
          episode_ids: [episodeId],
        },
        timestamp: firstTimestamp + index,
      });
    }

    if (record === null) {
      throw new Error("reinforceTraitWithEpisodes requires at least one episode id");
    }

    return record;
  }

  function distinctEpisodeCount(
    events: ReturnType<TraitsRepository["listReinforcementEvents"]>,
  ): number {
    const episodeIds = new Set<EpisodeId>();
    for (const event of events) {
      if (event.provenance.kind !== "episodes") {
        continue;
      }
      for (const episodeId of event.provenance.episode_ids) {
        episodeIds.add(episodeId);
      }
    }
    return episodeIds.size;
  }

  function countRows(db: ReturnType<typeof openDatabase>, sql: string, value: string): number {
    const row = db.prepare(sql).get(value) as { count: number } | undefined;
    return Number(row?.count ?? 0);
  }

  it("manages values and episode bindings", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const values = new ValuesRepository({
      db,
      clock: new FixedClock(100),
    });

    try {
      const value = values.add({
        label: "curiosity",
        description: "Prefer learning over stasis.",
        priority: 10,
        provenance: manualProvenance,
      });

      values.bindToEpisode(value.id, "ep_aaaaaaaaaaaaaaaa" as never);
      values.affirm(value.id, 200);

      expect(values.list()).toEqual([
        expect.objectContaining({
          id: value.id,
          label: "curiosity",
          last_affirmed: 200,
          provenance: episodeProvenance,
        }),
      ]);

      expect(values.remove(value.id)).toBe(true);
      expect(values.list()).toEqual([]);
    } finally {
      db.close();
    }
  });

  it("CAS-protects value, goal, and trait removals", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const values = new ValuesRepository({
      db,
      clock: new FixedClock(100),
    });
    const goals = new GoalsRepository({
      db,
      clock: new FixedClock(100),
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(100),
    });

    try {
      const value = values.add({
        label: "stability",
        description: "Keep state changes explicit.",
        priority: 8,
        provenance: manualProvenance,
      });
      values.reinforce(value.id, manualProvenance, 200, {
        expectedVersion: expectedRecordVersion(value),
      });

      expect(() =>
        values.remove(value.id, {
          expectedVersion: expectedRecordVersion(value),
        }),
      ).toThrow(IdentityCasMismatchError);
      expect(values.get(value.id)).not.toBeNull();

      const goal = goals.add({
        description: "Guard deletes with CAS",
        priority: 5,
        provenance: manualProvenance,
      });
      goals.updateProgress(goal.id, "Concurrent progress.", manualProvenance, {
        expectedVersion: expectedRecordVersion(goal),
      });

      expect(() =>
        goals.remove(goal.id, {
          expectedVersion: expectedRecordVersion(goal),
          auditContext: {
            reason: "test goal removal",
            provenance: manualProvenance,
          },
        }),
      ).toThrow(IdentityCasMismatchError);
      expect(goals.get(goal.id)).not.toBeNull();

      const trait = traits.reinforce({
        label: "careful",
        delta: 0.4,
        provenance: manualProvenance,
        timestamp: 100,
      });
      traits.reinforce({
        label: trait.label,
        delta: 0.1,
        provenance: manualProvenance,
        timestamp: 200,
        expectedVersion: expectedRecordVersion(trait),
      });

      expect(() =>
        traits.remove(trait.id, {
          expectedVersion: expectedRecordVersion(trait),
        }),
      ).toThrow(IdentityCasMismatchError);
      expect(traits.get(trait.id)).not.toBeNull();
    } finally {
      db.close();
    }
  });

  it("audits rollback goal restores and deletes with persisted records", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(selfMigrations, identityMigrations),
    });
    const clock = new ManualClock(100);
    const identityEvents = new IdentityEventRepository({ db, clock });
    const goals = new GoalsRepository({ db, clock, identityEventRepository: identityEvents });

    try {
      const original = goals.add({
        description: "Keep rollback goal audits complete",
        priority: 8,
        provenance: manualProvenance,
      });
      goals.updateStatus(original.id, "done", manualProvenance);
      const changed = goals.get(original.id)!;
      const restored = goals.restore(original);

      expect(restored).toMatchObject({
        id: original.id,
        record_version: changed.record_version,
        status: "active",
      });
      expect(
        goals.remove(original.id, {
          auditContext: {
            reason: GOAL_TURN_ROLLBACK_REASON,
            provenance: { kind: "system" },
          },
        }),
      ).toBe(true);

      const events = identityEvents.list({ recordType: "goal", recordId: original.id, limit: 10 });
      expect(events.map((event) => event.action)).toEqual(["delete", "update", "update", "create"]);
      expect(events[0]).toMatchObject({
        action: "delete",
        old_value: restored,
        new_value: null,
        reason: GOAL_TURN_ROLLBACK_REASON,
        provenance: { kind: "system" },
      });
      expect(events[1]).toMatchObject({
        action: "update",
        old_value: changed,
        new_value: restored,
        reason: GOAL_TURN_ROLLBACK_REASON,
      });
    } finally {
      db.close();
    }
  });

  it("rolls back goal restore and delete mutations when their audit write fails", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(selfMigrations, identityMigrations),
    });
    const clock = new ManualClock(100);
    const identityEvents = new IdentityEventRepository({ db, clock });
    const goals = new GoalsRepository({ db, clock, identityEventRepository: identityEvents });

    try {
      const original = goals.add({
        description: "Keep goal and audit in one transaction",
        priority: 7,
        provenance: manualProvenance,
      });
      goals.updateStatus(original.id, "done", manualProvenance);
      const changed = goals.get(original.id)!;
      vi.spyOn(identityEvents, "record").mockImplementation(() => {
        throw new Error("identity event unavailable");
      });

      expect(() => goals.restore(original)).toThrow("identity event unavailable");
      expect(goals.get(original.id)).toEqual(changed);
      expect(() =>
        goals.remove(original.id, {
          auditContext: {
            reason: GOAL_TURN_ROLLBACK_REASON,
            provenance: { kind: "system" },
          },
        }),
      ).toThrow("identity event unavailable");
      expect(goals.get(original.id)).toEqual(changed);
    } finally {
      db.close();
    }
  });

  it("manages hierarchical goals and progress", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const clock = new ManualClock(100);
    const goals = new GoalsRepository({
      db,
      clock,
    });

    try {
      const seeded = goals.add({
        description: "Keep the migration honest",
        priority: 2,
        progressNotes: "Started already",
        provenance: manualProvenance,
      });
      const parent = goals.add({
        description: "Ship Sprint 2",
        terminalCondition: "Sprint 2 is shipped",
        priority: 10,
        provenance: manualProvenance,
      });
      const child = goals.add({
        description: "Write extractor tests",
        priority: 8,
        parentId: parent.id,
        provenance: manualProvenance,
      });

      goals.updateProgress(child.id, "Covered happy path and dedup.", manualProvenance);
      clock.advance(50);
      goals.updateStatus(child.id, "done", manualProvenance);
      clock.advance(25);
      goals.update(
        child.id,
        {
          progress_notes: "Covered happy path, dedup, and follow-up fixes.",
        },
        manualProvenance,
      );

      expect(goals.list()).toEqual([
        expect.objectContaining({
          id: parent.id,
          terminal_condition: "Sprint 2 is shipped",
          children: [
            expect.objectContaining({
              id: child.id,
              status: "done",
              progress_notes: "Covered happy path, dedup, and follow-up fixes.",
              last_progress_ts: 175,
              provenance: manualProvenance,
            }),
          ],
        }),
        expect.objectContaining({
          id: seeded.id,
          last_progress_ts: 100,
        }),
      ]);
      expect(goals.list({ status: "done" })).toEqual([
        expect.objectContaining({
          id: child.id,
        }),
      ]);
    } finally {
      db.close();
    }
  });

  it("scopes goals by audience and stores source stream anchors", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const goals = new GoalsRepository({
      db,
      clock: new FixedClock(100),
    });
    const alice = createEntityId();
    const bob = createEntityId();
    const streamEntryId = createStreamEntryId();

    try {
      const globalGoal = goals.add({
        description: "Keep shared planning visible",
        priority: 10,
        provenance: manualProvenance,
      });
      const aliceGoal = goals.add({
        description: "Help Alice track italki options",
        priority: 9,
        audienceEntityId: alice,
        ownerEntityId: null,
        counterpartyEntityId: bob,
        sourceStreamEntryIds: [streamEntryId],
        provenance: {
          kind: "online",
          process: "goal-promotion-extractor",
        },
      });
      const bobGoal = goals.add({
        description: "Help Bob track posture audit",
        priority: 8,
        audienceEntityId: bob,
        provenance: manualProvenance,
      });

      expect(goals.get(aliceGoal.id)).toMatchObject({
        audience_entity_id: alice,
        owner_entity_id: null,
        counterparty_entity_id: bob,
        source_stream_entry_ids: [streamEntryId],
      });
      expect(
        goals.list({ status: "active", visibleToAudienceEntityId: alice }).map((goal) => goal.id),
      ).toEqual([globalGoal.id, aliceGoal.id]);
      expect(
        goals.list({ status: "active", visibleToAudienceEntityId: bob }).map((goal) => goal.id),
      ).toEqual([globalGoal.id, bobGoal.id]);
      expect(
        goals.list({ status: "active", visibleToAudienceEntityId: null }).map((goal) => goal.id),
      ).toEqual([globalGoal.id]);
    } finally {
      db.close();
    }
  });

  it("persists audience-scoped active goals across repository reopen", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-goals-"));
    const dbPath = join(tempDir, "borg.db");
    const audienceEntityId = createEntityId();
    const counterpartyEntityId = createEntityId();

    try {
      const firstDb = openDatabase(dbPath, {
        migrations: [...selfMigrations],
      });
      const firstGoals = new GoalsRepository({
        db: firstDb,
        clock: new FixedClock(100),
      });
      const goal = firstGoals.add({
        description: "Help track italki shortlist",
        terminalCondition: "The italki shortlist reaches a selected tutor",
        priority: 8,
        audienceEntityId,
        counterpartyEntityId,
        provenance: manualProvenance,
      });
      firstDb.close();

      const secondDb = openDatabase(dbPath, {
        migrations: [...selfMigrations],
      });
      const secondGoals = new GoalsRepository({
        db: secondDb,
        clock: new FixedClock(200),
      });

      try {
        expect(
          secondGoals
            .list({ status: "active", visibleToAudienceEntityId: audienceEntityId })
            .map((item) => [item.id, item.terminal_condition, item.counterparty_entity_id]),
        ).toEqual([
          [goal.id, "The italki shortlist reaches a selected tutor", counterpartyEntityId],
        ]);
      } finally {
        secondDb.close();
      }
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("adds goal nullable columns to existing goals with additive migrations", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-goal-terminal-migration-"));
    const dbPath = join(tempDir, "borg.db");
    const goalId = createGoalId();
    const oldGoalBaseline = {
      id: 1,
      name: "old_goal_baseline",
      up: `
        CREATE TABLE goals (
          id TEXT PRIMARY KEY,
          record_version INTEGER NOT NULL DEFAULT 1,
          description TEXT NOT NULL,
          priority REAL NOT NULL,
          parent_goal_id TEXT,
          status TEXT NOT NULL CHECK (status IN ('active', 'done', 'abandoned', 'blocked')),
          progress_notes TEXT,
          created_at INTEGER NOT NULL,
          target_at INTEGER,
          provenance_kind TEXT,
          provenance_episode_ids TEXT,
          provenance_process TEXT,
          last_progress_ts INTEGER,
          audience_entity_id TEXT,
          owner_entity_id TEXT,
          source_stream_entry_ids TEXT,
          canonicalized_by_artifact_entry_id TEXT NULL,
          FOREIGN KEY (parent_goal_id) REFERENCES goals(id) ON DELETE SET NULL
        )
      `,
    };
    const terminalConditionMigration = selfMigrations.find((migration) => migration.id === 6);
    const streamProvenanceMigration = selfMigrations.find((migration) => migration.id === 7);
    const counterpartyMigration = selfMigrations.find(
      (migration) => migration.name === "goal_counterparty_entity_id",
    );

    expect(terminalConditionMigration).toBeDefined();
    expect(streamProvenanceMigration).toBeDefined();
    expect(counterpartyMigration).toBeDefined();

    try {
      const oldDb = openDatabase(dbPath, {
        migrations: [oldGoalBaseline],
      });
      oldDb
        .prepare(
          `
            INSERT INTO goals (
              id, description, priority, parent_goal_id, status, progress_notes, created_at,
              target_at, provenance_kind, provenance_episode_ids, provenance_process,
              last_progress_ts, audience_entity_id, owner_entity_id, source_stream_entry_ids,
              canonicalized_by_artifact_entry_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          goalId,
          "Keep old rows readable",
          4,
          null,
          "active",
          null,
          100,
          null,
          "manual",
          null,
          null,
          null,
          null,
          null,
          null,
          null,
        );
      oldDb.close();

      const migratedDb = openDatabase(dbPath, {
        migrations: [
          oldGoalBaseline,
          terminalConditionMigration!,
          streamProvenanceMigration!,
          counterpartyMigration!,
        ],
      });
      const goals = new GoalsRepository({
        db: migratedDb,
        clock: new FixedClock(200),
      });

      try {
        const columns = migratedDb.pragma("table_info(goals)") as Array<{ name: string }>;
        expect(columns.map((column) => column.name)).toContain("terminal_condition");
        expect(columns.map((column) => column.name)).toContain("provenance_stream_entry_ids");
        expect(columns.map((column) => column.name)).toContain("counterparty_entity_id");
        expect(() => counterpartyMigration!.up(migratedDb)).not.toThrow();
        expect(
          (migratedDb.pragma("table_info(goals)") as Array<{ name: string }>).filter(
            (column) => column.name === "counterparty_entity_id",
          ),
        ).toHaveLength(1);
        expect(goals.get(goalId)).toEqual(
          expect.objectContaining({
            id: goalId,
            terminal_condition: null,
            counterparty_entity_id: null,
          }),
        );
      } finally {
        migratedDb.close();
      }
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("reinforces, decays, and culls traits", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const clock = new ManualClock(0);
    const traits = new TraitsRepository({
      db,
      clock,
    });

    try {
      traits.reinforce({
        label: "patient",
        delta: 0.8,
        provenance: manualProvenance,
        timestamp: 0,
      });
      clock.advance(24 * 3_600_000);
      traits.decay(24, clock.now());
      traits.reinforce({
        label: "decisive",
        delta: 0.2,
        provenance: episodeProvenance,
        timestamp: clock.now(),
      });

      const listed = traits.list();
      expect(listed[0]).toEqual(
        expect.objectContaining({
          label: "patient",
          provenance: manualProvenance,
        }),
      );
      expect(listed.find((trait) => trait.label === "patient")?.strength).toBeLessThan(0.8);
      expect(traits.cull(0.3)).toBe(1);
      expect(traits.list()).toEqual([
        expect.objectContaining({
          label: "patient",
        }),
      ]);
    } finally {
      db.close();
    }
  });

  it("promotes traits only after five distinct episode-backed reinforcements", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    const episodeIds = [
      "ep_aaaaaaaaaaaaaaaa" as never,
      "ep_bbbbbbbbbbbbbbbb" as never,
      "ep_cccccccccccccccc" as never,
      "ep_dddddddddddddddd" as never,
      "ep_eeeeeeeeeeeeeeee" as never,
    ] as const;

    try {
      for (const episodeId of episodeIds.slice(0, 4)) {
        traits.reinforce({
          label: "engaged",
          delta: 0.05,
          provenance: {
            kind: "episodes",
            episode_ids: [episodeId],
          },
        });
      }

      traits.reinforce({
        label: "engaged",
        delta: 0.05,
        provenance: {
          kind: "offline",
          process: "reflector",
        },
      });

      expect(traits.list()[0]).toEqual(
        expect.objectContaining({
          label: "engaged",
          state: "candidate",
          established_at: null,
        }),
      );

      traits.reinforce({
        label: "engaged",
        delta: 0.05,
        provenance: {
          kind: "episodes",
          episode_ids: [episodeIds[4]],
        },
      });

      expect(traits.list()[0]).toEqual(
        expect.objectContaining({
          label: "engaged",
          state: "established",
        }),
      );
      expect(traits.list()[0]?.established_at).not.toBeNull();
    } finally {
      db.close();
    }
  });

  it("merges candidate traits and promotes from the union of distinct reinforcement episodes", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const canonical = reinforceTraitWithEpisodes(
        traits,
        "intellectual_honesty",
        mergeEpisodeIds.slice(0, 3),
        { firstTimestamp: 1_000 },
      );
      const source = reinforceTraitWithEpisodes(
        traits,
        "intellectual honesty",
        mergeEpisodeIds.slice(3, 6),
        { firstTimestamp: 2_000 },
      );

      expect(canonical.state).toBe("candidate");
      expect(source.state).toBe("candidate");

      const merged = traits.mergeInto({
        sourceId: source.id,
        canonicalId: canonical.id,
        expectedSourceVersion: expectedRecordVersion(source),
        expectedCanonicalVersion: expectedRecordVersion(canonical),
        provenance: { kind: "offline", process: "trait-consolidation-test" },
      });

      expect(merged).toMatchObject({
        id: canonical.id,
        state: "established",
        support_count: 6,
      });
      expect(merged.established_at).not.toBeNull();
      expect(merged.evidence_episode_ids).toHaveLength(3);
      expect(distinctEpisodeCount(traits.listReinforcementEvents(canonical.id))).toBe(6);
      expect(traits.get(source.id)).toBeNull();
    } finally {
      db.close();
    }
  });

  it("clamps combined trait strength during merge", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const canonical = reinforceTraitWithEpisodes(
        traits,
        "careful_synthesis",
        [mergeEpisodeIds[0]!],
        { delta: 0.7 },
      );
      const source = reinforceTraitWithEpisodes(
        traits,
        "careful synthesis",
        [mergeEpisodeIds[1]!],
        { delta: 0.6, firstTimestamp: 2_000 },
      );

      const merged = traits.mergeInto({
        sourceId: source.id,
        canonicalId: canonical.id,
        expectedSourceVersion: expectedRecordVersion(source),
        expectedCanonicalVersion: expectedRecordVersion(canonical),
        provenance: { kind: "offline", process: "trait-consolidation-test" },
      });

      expect(merged.strength).toBe(1);
    } finally {
      db.close();
    }
  });

  it("merges established traits into an established canonical without demotion", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const canonical = reinforceTraitWithEpisodes(
        traits,
        "truth_seeking",
        mergeEpisodeIds.slice(0, 5),
        { firstTimestamp: 1_000 },
      );
      const source = reinforceTraitWithEpisodes(
        traits,
        "truth seeking",
        mergeEpisodeIds.slice(5, 10),
        { firstTimestamp: 2_000 },
      );

      expect(canonical.state).toBe("established");
      expect(source.state).toBe("established");

      const merged = traits.mergeInto({
        sourceId: source.id,
        canonicalId: canonical.id,
        expectedSourceVersion: expectedRecordVersion(source),
        expectedCanonicalVersion: expectedRecordVersion(canonical),
        provenance: { kind: "offline", process: "trait-consolidation-test" },
      });

      expect(merged.state).toBe("established");
      expect(merged.established_at).toBe(canonical.established_at);
      expect(traits.get(source.id)).toBeNull();
      expect(distinctEpisodeCount(traits.listReinforcementEvents(canonical.id))).toBe(10);
    } finally {
      db.close();
    }
  });

  it("refuses to merge an established source into a candidate canonical", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const canonical = reinforceTraitWithEpisodes(traits, "deliberate", [mergeEpisodeIds[0]!], {
        firstTimestamp: 1_000,
      });
      const source = reinforceTraitWithEpisodes(
        traits,
        "deliberateness",
        mergeEpisodeIds.slice(1, 6),
        { firstTimestamp: 2_000 },
      );

      expect(canonical.state).toBe("candidate");
      expect(source.state).toBe("established");

      expect(() =>
        traits.mergeInto({
          sourceId: source.id,
          canonicalId: canonical.id,
          expectedSourceVersion: expectedRecordVersion(source),
          expectedCanonicalVersion: expectedRecordVersion(canonical),
          provenance: { kind: "offline", process: "trait-consolidation-test" },
        }),
      ).toThrow(/established trait into a candidate canonical/);

      expect(traits.get(source.id)).not.toBeNull();
      expect(traits.listReinforcementEvents(source.id)).toHaveLength(5);
      expect(traits.listReinforcementEvents(canonical.id)).toHaveLength(1);
    } finally {
      db.close();
    }
  });

  it("CAS-protects trait merges", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const canonical = reinforceTraitWithEpisodes(
        traits,
        "steady_reasoning",
        [mergeEpisodeIds[0]!],
        { firstTimestamp: 1_000 },
      );
      const source = reinforceTraitWithEpisodes(traits, "steady reasoning", [mergeEpisodeIds[1]!], {
        firstTimestamp: 2_000,
      });
      const staleSourceVersion = expectedRecordVersion(source);

      const changedSource = traits.reinforce({
        label: source.label,
        delta: 0.1,
        provenance: {
          kind: "episodes",
          episode_ids: [mergeEpisodeIds[2]!],
        },
        timestamp: 3_000,
        expectedVersion: staleSourceVersion,
      });

      expect(() =>
        traits.mergeInto({
          sourceId: changedSource.id,
          canonicalId: canonical.id,
          expectedSourceVersion: staleSourceVersion,
          expectedCanonicalVersion: expectedRecordVersion(canonical),
          provenance: { kind: "offline", process: "trait-consolidation-test" },
        }),
      ).toThrow(IdentityCasMismatchError);
      expect(traits.get(changedSource.id)).not.toBeNull();
      expect(traits.listReinforcementEvents(changedSource.id)).toHaveLength(2);
    } finally {
      db.close();
    }
  });

  it("CAS-protects trait merges when the canonical version is stale", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const canonical = reinforceTraitWithEpisodes(
        traits,
        "adaptive_reasoning",
        [mergeEpisodeIds[0]!],
        { firstTimestamp: 1_000 },
      );
      const source = reinforceTraitWithEpisodes(
        traits,
        "adaptive reasoning",
        [mergeEpisodeIds[1]!],
        { firstTimestamp: 2_000 },
      );
      const staleCanonicalVersion = expectedRecordVersion(canonical);

      const changedCanonical = traits.reinforce({
        label: canonical.label,
        delta: 0.1,
        provenance: {
          kind: "episodes",
          episode_ids: [mergeEpisodeIds[2]!],
        },
        timestamp: 3_000,
        expectedVersion: staleCanonicalVersion,
      });

      expect(() =>
        traits.mergeInto({
          sourceId: source.id,
          canonicalId: changedCanonical.id,
          expectedSourceVersion: expectedRecordVersion(source),
          expectedCanonicalVersion: staleCanonicalVersion,
          provenance: { kind: "offline", process: "trait-consolidation-test" },
        }),
      ).toThrow(IdentityCasMismatchError);
      expect(traits.get(source.id)).not.toBeNull();
      expect(traits.listReinforcementEvents(source.id)).toHaveLength(1);
      expect(traits.listReinforcementEvents(changedCanonical.id)).toHaveLength(2);
    } finally {
      db.close();
    }
  });

  it("refuses to merge a trait into itself", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const trait = reinforceTraitWithEpisodes(traits, "self_consistency", [mergeEpisodeIds[0]!], {
        firstTimestamp: 1_000,
      });

      expect(() =>
        traits.mergeInto({
          sourceId: trait.id,
          canonicalId: trait.id,
          expectedSourceVersion: expectedRecordVersion(trait),
          expectedCanonicalVersion: expectedRecordVersion(trait),
          provenance: { kind: "offline", process: "trait-consolidation-test" },
        }),
      ).toThrow(/Cannot merge a trait into itself/);
      expect(traits.get(trait.id)).not.toBeNull();
      expect(traits.listReinforcementEvents(trait.id)).toHaveLength(1);
    } finally {
      db.close();
    }
  });

  it("rejects trait merges with an unknown source id", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const canonical = reinforceTraitWithEpisodes(
        traits,
        "known_canonical",
        [mergeEpisodeIds[0]!],
        { firstTimestamp: 1_000 },
      );
      const unknownSourceId = createTraitId();

      expect(() =>
        traits.mergeInto({
          sourceId: unknownSourceId,
          canonicalId: canonical.id,
          expectedSourceVersion: 1,
          expectedCanonicalVersion: expectedRecordVersion(canonical),
          provenance: { kind: "offline", process: "trait-consolidation-test" },
        }),
      ).toThrow(/Unknown source trait id/);
      expect(traits.get(canonical.id)).not.toBeNull();
      expect(traits.listReinforcementEvents(canonical.id)).toHaveLength(1);
    } finally {
      db.close();
    }
  });

  it("rejects trait merges with an unknown canonical id", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const source = reinforceTraitWithEpisodes(traits, "known_source", [mergeEpisodeIds[0]!], {
        firstTimestamp: 1_000,
      });
      const unknownCanonicalId = createTraitId();

      expect(() =>
        traits.mergeInto({
          sourceId: source.id,
          canonicalId: unknownCanonicalId,
          expectedSourceVersion: expectedRecordVersion(source),
          expectedCanonicalVersion: 1,
          provenance: { kind: "offline", process: "trait-consolidation-test" },
        }),
      ).toThrow(/Unknown canonical trait id/);
      expect(traits.get(source.id)).not.toBeNull();
      expect(traits.listReinforcementEvents(source.id)).toHaveLength(1);
    } finally {
      db.close();
    }
  });

  it("deletes the source trait and re-points reinforcement and contradiction events", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const traits = new TraitsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const canonical = reinforceTraitWithEpisodes(
        traits,
        "contextual_precision",
        [mergeEpisodeIds[0]!],
        { firstTimestamp: 1_000 },
      );
      const source = reinforceTraitWithEpisodes(
        traits,
        "contextual precision",
        [mergeEpisodeIds[1]!],
        { firstTimestamp: 2_000 },
      );

      traits.recordContradiction({
        label: source.label,
        provenance: { kind: "manual" },
        timestamp: 3_000,
        expectedVersion: expectedRecordVersion(source),
      });
      const sourceAfterContradiction = traits.get(source.id)!;

      const merged = traits.mergeInto({
        sourceId: sourceAfterContradiction.id,
        canonicalId: canonical.id,
        expectedSourceVersion: expectedRecordVersion(sourceAfterContradiction),
        expectedCanonicalVersion: expectedRecordVersion(canonical),
        provenance: { kind: "offline", process: "trait-consolidation-test" },
      });

      expect(traits.get(source.id)).toBeNull();
      expect(traits.listReinforcementEvents(source.id)).toEqual([]);
      expect(traits.listContradictionEvents(source.id)).toEqual([]);
      expect(traits.listReinforcementEvents(canonical.id)).toHaveLength(2);
      expect(traits.listContradictionEvents(canonical.id)).toHaveLength(1);
      expect(merged).toMatchObject({
        support_count: 2,
        contradiction_count: 1,
        last_contradicted_at: 3_000,
      });
      expect(
        countRows(
          db,
          "SELECT COUNT(*) AS count FROM trait_reinforcement_events WHERE trait_id = ?",
          source.id,
        ),
      ).toBe(0);
      expect(
        countRows(
          db,
          "SELECT COUNT(*) AS count FROM trait_contradiction_events WHERE trait_id = ?",
          source.id,
        ),
      ).toBe(0);
    } finally {
      db.close();
    }
  });

  it("tracks value candidate and established state transitions", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const values = new ValuesRepository({
      db,
      clock: new FixedClock(1_000),
    });
    const episodeIds = [
      "ep_aaaaaaaaaaaaaaaa" as never,
      "ep_bbbbbbbbbbbbbbbb" as never,
      "ep_cccccccccccccccc" as never,
    ] as const;

    try {
      const manual = values.add({
        label: "clarity",
        description: "Prefer explicit state.",
        priority: 5,
        provenance: manualProvenance,
      });
      const candidate = values.add({
        label: "patience",
        description: "Stay steady under pressure.",
        priority: 4,
        provenance: {
          kind: "episodes",
          episode_ids: [episodeIds[0]],
        },
      });

      expect(manual).toEqual(
        expect.objectContaining({
          state: "candidate",
          established_at: null,
        }),
      );
      expect(candidate.state).toBe("candidate");

      values.reinforce(candidate.id, {
        kind: "episodes",
        episode_ids: [episodeIds[1]],
      });

      expect(values.get(candidate.id)?.state).toBe("candidate");

      values.reinforce(candidate.id, {
        kind: "episodes",
        episode_ids: [episodeIds[2]],
      });

      expect(values.get(candidate.id)).toEqual(
        expect.objectContaining({
          state: "established",
          provenance: {
            kind: "episodes",
            episode_ids: [
              "ep_aaaaaaaaaaaaaaaa" as never,
              "ep_bbbbbbbbbbbbbbbb" as never,
              "ep_cccccccccccccccc" as never,
            ],
          },
        }),
      );
      expect(values.get(candidate.id)?.established_at).not.toBeNull();
    } finally {
      db.close();
    }
  });

  it("rejects invalid stored value provenance episode ids", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const values = new ValuesRepository({
      db,
      clock: new FixedClock(100),
    });

    try {
      const value = values.add({
        label: "clarity",
        description: "Prefer explicit state.",
        priority: 1,
        provenance: manualProvenance,
      });

      db.prepare(
        `
          UPDATE "values"
          SET provenance_kind = 'episodes', provenance_episode_ids = ?
          WHERE id = ?
        `,
      ).run('["not-an-episode-id"]', value.id);

      expect(() => values.list()).toThrow();
    } finally {
      db.close();
    }
  });

  it("rejects provenance-less creates and updates", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const values = new ValuesRepository({ db, clock: new FixedClock(100) });
    const goals = new GoalsRepository({ db, clock: new FixedClock(100) });
    const traits = new TraitsRepository({ db, clock: new FixedClock(100) });

    try {
      expect(() =>
        values.add({
          label: "clarity",
          description: "Prefer explicit state.",
          priority: 1,
          provenance: undefined as never,
        }),
      ).toThrow(ProvenanceError);

      const goal = goals.add({
        description: "Ship Sprint 6",
        priority: 1,
        provenance: manualProvenance,
      });

      expect(() => goals.updateProgress(goal.id, "Updated", undefined as never)).toThrow(
        ProvenanceError,
      );
      expect(() => goals.updateStatus(goal.id, "done", undefined as never)).toThrow(
        ProvenanceError,
      );
      expect(() =>
        traits.reinforce({
          label: "patient",
          delta: 0.2,
          provenance: undefined as never,
        }),
      ).toThrow(ProvenanceError);
    } finally {
      db.close();
    }
  });

  it("tracks evidence-backed confidence and contradictions for values and traits", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const clock = new ManualClock(1_000);
    const values = new ValuesRepository({ db, clock });
    const traits = new TraitsRepository({ db, clock });

    try {
      const zeroEvidenceValue = values.add({
        label: "stability",
        description: "Prefer calm, predictable responses.",
        priority: 3,
        provenance: manualProvenance,
        createdAt: 500,
      });

      expect(zeroEvidenceValue.confidence).toBeCloseTo(2 / 3, 6);

      const value = values.add({
        label: "clarity",
        description: "Prefer explicit state.",
        priority: 5,
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
        },
        createdAt: 1_000,
      });

      values.reinforce(
        value.id,
        {
          kind: "episodes",
          episode_ids: ["ep_bbbbbbbbbbbbbbbb" as never],
        },
        2_000,
      );
      const establishedValue = values.reinforce(
        value.id,
        {
          kind: "episodes",
          episode_ids: ["ep_cccccccccccccccc" as never],
        },
        3_000,
      );

      expect(establishedValue).toMatchObject({
        state: "established",
        support_count: 3,
        contradiction_count: 0,
        last_tested_at: 3_000,
        evidence_episode_ids: [
          "ep_cccccccccccccccc" as never,
          "ep_bbbbbbbbbbbbbbbb" as never,
          "ep_aaaaaaaaaaaaaaaa" as never,
        ],
      });
      expect(establishedValue.confidence).toBeCloseTo(5 / 6, 6);

      const contradictedValue = values.recordContradiction({
        valueId: value.id,
        provenance: { kind: "manual" },
        timestamp: 4_000,
      });

      expect(contradictedValue).toMatchObject({
        contradiction_count: 1,
        last_contradicted_at: 4_000,
      });
      expect(contradictedValue.confidence).toBeCloseTo(5 / 7, 6);

      const reinforcedTrait = traits.reinforce({
        label: "introspective",
        delta: 0.2,
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_dddddddddddddddd" as never],
        },
        timestamp: 5_000,
      });

      expect(reinforcedTrait).toMatchObject({
        support_count: 1,
        contradiction_count: 0,
        last_tested_at: 5_000,
        evidence_episode_ids: ["ep_dddddddddddddddd" as never],
      });
      expect(reinforcedTrait.confidence).toBeCloseTo(0.75, 6);

      const contradictedTrait = traits.recordContradiction({
        label: "introspective",
        provenance: { kind: "offline", process: "reflector" },
        timestamp: 6_000,
      });

      expect(contradictedTrait).toMatchObject({
        contradiction_count: 1,
        last_contradicted_at: 6_000,
      });
      expect(contradictedTrait.confidence).toBeCloseTo(0.6, 6);
    } finally {
      db.close();
    }
  });
});
