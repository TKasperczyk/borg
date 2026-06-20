import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createSessionId,
  createStreamEntryId,
  type SelfDecisionEventId,
} from "../../util/ids.js";
import { selfDecisionMigrations } from "./migrations.js";
import { SelfDecisionRepository } from "./repository.js";

const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

describe("SelfDecisionRepository", () => {
  it("round-trips self decision events and keeps fire-event writes idempotent", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const sessionId = createSessionId();
    const wakeEntryId = createStreamEntryId();
    const actionEntryId = createStreamEntryId();
    const retryWakeEntryId = createStreamEntryId();

    const created = repository.record({
      occurredAt: 1_000,
      sessionId,
      triggerName: "scheduled_reflection",
      triggerType: "trigger",
      sourceEventId: "scheduled-reflection:1000",
      fireEventId: actionEntryId,
      decisionSummary: "Sol reflected on recent changes.",
      decisionRationale: "Reflection found enough pending state to review.",
      turnResultId: "strm_agent_first",
      sourceStreamEntryIds: [wakeEntryId, actionEntryId, wakeEntryId],
    });
    const duplicate = repository.record({
      occurredAt: 1_500,
      sessionId,
      triggerName: "scheduled_reflection",
      triggerType: "trigger",
      sourceEventId: "scheduled-reflection:1000",
      fireEventId: actionEntryId,
      decisionSummary: "This retry should not overwrite the original.",
      turnResultId: "strm_agent_second",
      sourceStreamEntryIds: [retryWakeEntryId, actionEntryId],
      now: 3_000,
    });

    expect(repository.get(created.id as SelfDecisionEventId)).toEqual(created);
    expect(duplicate).toEqual(created);
    expect(created).toMatchObject({
      occurred_at: 1_000,
      session_id: sessionId,
      trigger_name: "scheduled_reflection",
      trigger_type: "trigger",
      source_event_id: "scheduled-reflection:1000",
      fire_event_id: actionEntryId,
      origin: "autonomous",
      decision_summary: "Sol reflected on recent changes.",
      decision_rationale: "Reflection found enough pending state to review.",
      turn_result_id: "strm_agent_first",
      source_stream_entry_ids: [wakeEntryId, actionEntryId],
      disclosure_class: "self_private",
      created_at: 2_000,
      updated_at: 2_000,
    });
    expect(db.prepare("SELECT COUNT(*) AS count FROM self_decision_events").get()).toEqual({
      count: 1,
    });

    db.close();
  });

  it("keeps genuine recurring fires with the same source event id as separate rows", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-refire-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const sessionId = createSessionId();
    const sourceEventId = "default:1000";
    const firstActionEntryId = createStreamEntryId();
    const secondActionEntryId = createStreamEntryId();

    repository.record({
      occurredAt: 1_000,
      sessionId,
      triggerName: "mood_valence_drop",
      triggerType: "condition",
      sourceEventId,
      fireEventId: firstActionEntryId,
      decisionSummary: "First committed fire.",
      turnResultId: "strm_agent_first",
      sourceStreamEntryIds: [createStreamEntryId(), firstActionEntryId],
    });
    repository.record({
      occurredAt: 5_000,
      sessionId,
      triggerName: "mood_valence_drop",
      triggerType: "condition",
      sourceEventId,
      fireEventId: secondActionEntryId,
      decisionSummary: "Second committed fire.",
      turnResultId: "strm_agent_second",
      sourceStreamEntryIds: [createStreamEntryId(), secondActionEntryId],
    });

    expect(db.prepare("SELECT COUNT(*) AS count FROM self_decision_events").get()).toEqual({
      count: 2,
    });
    expect(
      repository
        .listRecentAutonomousSelfPrivate({
          sinceMs: 0,
          limit: 10,
        })
        .map((row) => row.decisionSummary),
    ).toEqual(["Second committed fire.", "First committed fire."]);

    db.close();
  });

  it("lists autonomous self-private decisions across sessions for self recall", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-cross-session-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({
      db,
      clock: new FixedClock(2_000),
    });

    repository.record({
      occurredAt: 1_000,
      sessionId: DEFAULT_SESSION_ID,
      triggerName: "scheduled_reflection",
      triggerType: "trigger",
      sourceEventId: "scheduled-reflection:1000",
      fireEventId: createStreamEntryId(),
      decisionSummary: "Default-session autonomous decision.",
      turnResultId: "strm_agent_default",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    expect(
      repository.listRecentAutonomousSelfPrivate({
        sinceMs: 0,
        limit: 10,
      }),
    ).toEqual([
      expect.objectContaining({
        decisionSummary: "Default-session autonomous decision.",
        triggerName: "scheduled_reflection",
      }),
    ]);

    db.close();
  });

  it("aggregates autonomous self-private decisions by UTC day", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-density-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const sessionId = createSessionId();
    const firstAt = Date.UTC(2026, 5, 15, 2, 0, 0);
    const secondAt = Date.UTC(2026, 5, 15, 23, 30, 0);
    const nextDayAt = Date.UTC(2026, 5, 16, 1, 0, 0);

    for (const [index, occurredAt] of [firstAt, secondAt, nextDayAt].entries()) {
      repository.record({
        occurredAt,
        sessionId,
        triggerName: `scheduled_reflection_${index}`,
        triggerType: "trigger",
        sourceEventId: `scheduled-reflection:${index}`,
        fireEventId: createStreamEntryId(),
        decisionSummary: `Reflection ${index}.`,
        sourceStreamEntryIds: [createStreamEntryId()],
      });
    }

    expect(
      repository.listDailyAutonomousSelfPrivateDensity({
        sinceMs: firstAt - 1,
        untilMs: nextDayAt - 1,
        limit: 10,
      }),
    ).toEqual([
      {
        dayKey: "2026-06-15",
        dayStartMs: Date.UTC(2026, 5, 15),
        decisionCount: 2,
        distinctDecisionShapeCount: 1,
        firstOccurredAt: firstAt,
        lastOccurredAt: secondAt,
      },
    ]);

    db.close();
  });

  it("lists autonomous self-private decision summaries inside an explicit range", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-range-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const sessionId = createSessionId();
    const firstAt = Date.UTC(2026, 5, 15, 2, 0, 0);
    const inRangeSourceId = createStreamEntryId();

    repository.record({
      occurredAt: firstAt - 1,
      sessionId,
      triggerName: "before",
      triggerType: "trigger",
      sourceEventId: "before",
      fireEventId: createStreamEntryId(),
      decisionSummary: "Before range.",
      sourceStreamEntryIds: [createStreamEntryId()],
    });
    repository.record({
      occurredAt: firstAt,
      sessionId,
      triggerName: "inside",
      triggerType: "trigger",
      sourceEventId: "inside",
      fireEventId: createStreamEntryId(),
      decisionSummary: "Inside range.",
      decisionRationale: "The range read keeps rationale.",
      sourceStreamEntryIds: [inRangeSourceId],
    });

    expect(
      repository.listAutonomousSelfPrivateForRange({
        sinceMs: firstAt,
        untilMs: firstAt,
        limit: 10,
      }),
    ).toEqual([
      {
        occurredAt: firstAt,
        triggerName: "inside",
        triggerType: "trigger",
        decisionSummary: "Inside range.",
        decisionRationale: "The range read keeps rationale.",
        sourceStreamEntryIds: [inRangeSourceId],
      },
    ]);

    db.close();
  });

  it("counts full-window daily density and collapses repeated structural emission shapes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-density-flood-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const sessionId = createSessionId();
    const firstAt = Date.UTC(2026, 5, 15, 2, 0, 0);

    for (let index = 0; index < 80; index += 1) {
      repository.record({
        occurredAt: firstAt + index * 60_000,
        sessionId,
        triggerName: `wake_${index}`,
        triggerType: "trigger",
        sourceEventId: `wake:${index}`,
        fireEventId: createStreamEntryId(),
        decisionSummary: `Reflection ${index}.`,
        sourceStreamEntryIds: [createStreamEntryId()],
      });
    }
    repository.record({
      occurredAt: firstAt + 90 * 60_000,
      sessionId,
      triggerName: "condition_wake",
      triggerType: "condition",
      sourceEventId: "condition:1",
      fireEventId: createStreamEntryId(),
      decisionSummary: "Different structural shape.",
      decisionRationale: "Different shape because rationale is present.",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    expect(
      repository.listDailyAutonomousSelfPrivateDensity({
        sinceMs: firstAt - 1,
        untilMs: firstAt + 100 * 60_000,
        limit: 1,
      }),
    ).toEqual([
      expect.objectContaining({
        dayKey: "2026-06-15",
        decisionCount: 81,
        distinctDecisionShapeCount: 2,
        firstOccurredAt: firstAt,
        lastOccurredAt: firstAt + 90 * 60_000,
      }),
    ]);

    db.close();
  });

  it("validates rows at the read boundary", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-invalid-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const created = repository.record({
      occurredAt: 1_000,
      sessionId: createSessionId(),
      triggerName: "scheduled_reflection",
      triggerType: "trigger",
      sourceEventId: "scheduled-reflection:1000",
      fireEventId: createStreamEntryId(),
      decisionSummary: "Valid row before corruption.",
      turnResultId: "strm_agent_valid",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    db.prepare("UPDATE self_decision_events SET source_stream_entry_ids = ? WHERE id = ?").run(
      "not json",
      created.id,
    );

    expect(() => repository.get(created.id)).toThrow(/self decision event/i);

    db.close();
  });
});
