import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createSelfDecisionEventId,
  createSessionId,
  createStreamEntryId,
} from "../../util/ids.js";
import { selfDecisionMigrations } from "./migrations.js";
import {
  DEFAULT_SELF_DECISION_INTROSPECTION_CAP,
  DEFAULT_SELF_DECISION_INTROSPECTION_RECENCY_WINDOW_MS,
  selectSelfDecisionIntrospection,
} from "./projection.js";
import { SelfDecisionRepository } from "./repository.js";

const NOW_MS = 1_000_000_000;
const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

describe("selectSelfDecisionIntrospection", () => {
  it("returns rows for operator context or private self-cognition and preserves multilingual summaries", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({ db, clock: new FixedClock(NOW_MS) });
    const sessionId = createSessionId();
    const decisionSummary = "Decidí revisar los objetivos pendientes sin enviar mensajes.";

    repository.record({
      occurredAt: NOW_MS - 2 * 60 * 60_000,
      sessionId,
      triggerName: "goal_followup_due",
      triggerType: "trigger",
      sourceEventId: "goal_aaaaaaaaaaaaaaaa:no-target:900",
      fireEventId: createStreamEntryId(),
      decisionSummary,
      turnResultId: "strm_agent_spanish",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const visible = selectSelfDecisionIntrospection({
      repository,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: "creator",
      isPrivateSelfCognition: false,
      nowMs: NOW_MS,
    });
    const privateSelfVisible = selectSelfDecisionIntrospection({
      repository,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: null,
      isPrivateSelfCognition: true,
      nowMs: NOW_MS,
    });
    const participantHidden = selectSelfDecisionIntrospection({
      repository,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: "creator",
      isPrivateSelfCognition: false,
      nowMs: NOW_MS,
    });
    const nonCreatorHidden = selectSelfDecisionIntrospection({
      repository,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: null,
      isPrivateSelfCognition: false,
      nowMs: NOW_MS,
    });

    expect(visible).toEqual([
      expect.objectContaining({
        decisionSummary,
        triggerName: "goal_followup_due",
        triggerType: "trigger",
        relativeAge: "2h ago",
      }),
    ]);
    expect(visible[0]?.text).toContain(decisionSummary);
    expect(privateSelfVisible.map((row) => row.decisionSummary)).toEqual([decisionSummary]);
    expect(participantHidden).toEqual([]);
    expect(nonCreatorHidden).toEqual([]);

    db.close();
  });

  it("renders rationale after the structural decision token when present", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-rationale-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({ db, clock: new FixedClock(NOW_MS) });
    const decisionSummary = "Stayed silent (deliberate silence): low value echo";
    const decisionRationale = "Nie pojawiło się nic nowego, więc odpowiedź byłaby tylko echem.";

    repository.record({
      occurredAt: NOW_MS - 60_000,
      sessionId: DEFAULT_SESSION_ID,
      triggerName: "scheduled_reflection",
      triggerType: "trigger",
      sourceEventId: "scheduled-reflection:rationale",
      fireEventId: createStreamEntryId(),
      decisionSummary,
      decisionRationale,
      turnResultId: "strm_agent_rationale",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const rows = selectSelfDecisionIntrospection({
      repository,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: null,
      isPrivateSelfCognition: true,
      nowMs: NOW_MS,
    });

    expect(rows).toEqual([
      expect.objectContaining({
        decisionSummary,
        decisionRationale,
        text: `Autonomous trigger scheduled_reflection completed 1m ago: ${decisionSummary} because ${decisionRationale}`,
      }),
    ]);

    db.close();
  });

  it("migrates legacy rows with NULL rationale and renders token-only text", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-legacy-rationale-"));
    tempDirs.push(tempDir);
    const dbPath = join(tempDir, "self-decisions.db");
    const legacyDb = openDatabase(dbPath, {
      migrations: [selfDecisionMigrations[0]!],
    });
    const legacyId = createSelfDecisionEventId();
    const legacyFireEventId = createStreamEntryId();
    const legacySourceEntryId = createStreamEntryId();
    const decisionSummary = "Stayed silent (deliberate silence): low value echo";

    legacyDb
      .prepare(
        `
          INSERT INTO self_decision_events (
            id, occurred_at, session_id, trigger_name, trigger_type, source_event_id,
            fire_event_id, origin, decision_summary, turn_result_id, source_stream_entry_ids,
            disclosure_class, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, 'autonomous', ?, ?, ?, 'self_private', ?, ?)
        `,
      )
      .run(
        legacyId,
        NOW_MS - 60_000,
        DEFAULT_SESSION_ID,
        "scheduled_reflection",
        "trigger",
        "scheduled-reflection:legacy",
        legacyFireEventId,
        decisionSummary,
        "strm_agent_legacy",
        JSON.stringify([legacySourceEntryId]),
        NOW_MS - 60_000,
        NOW_MS - 60_000,
      );
    legacyDb.close();

    const migratedDb = openDatabase(dbPath, {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({
      db: migratedDb,
      clock: new FixedClock(NOW_MS),
    });

    expect(
      migratedDb
        .prepare("SELECT decision_rationale FROM self_decision_events WHERE id = ?")
        .get(legacyId),
    ).toEqual({ decision_rationale: null });
    expect(
      repository.listRecentAutonomousSelfPrivate({
        sinceMs: 0,
        limit: 10,
      }),
    ).toEqual([
      expect.objectContaining({
        decisionSummary,
        decisionRationale: null,
      }),
    ]);

    const rows = selectSelfDecisionIntrospection({
      repository,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: null,
      isPrivateSelfCognition: true,
      nowMs: NOW_MS,
    });

    expect(rows).toEqual([
      expect.objectContaining({
        decisionSummary,
        decisionRationale: null,
        text: `Autonomous trigger scheduled_reflection completed 1m ago: ${decisionSummary}`,
      }),
    ]);

    migratedDb.close();
  });

  it("recalls default-session autonomous decisions during private self-cognition", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-private-self-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({ db, clock: new FixedClock(NOW_MS) });
    const currentSessionId = createSessionId();

    repository.record({
      occurredAt: NOW_MS - 60_000,
      sessionId: DEFAULT_SESSION_ID,
      triggerName: "scheduled_reflection",
      triggerType: "trigger",
      sourceEventId: "scheduled-reflection:private-self",
      fireEventId: createStreamEntryId(),
      decisionSummary: "Sol reviewed a pending autonomous choice.",
      turnResultId: "strm_agent_private_self",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const rows = selectSelfDecisionIntrospection({
      repository,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: null,
      isPrivateSelfCognition: true,
      nowMs: NOW_MS,
    });

    expect(currentSessionId).not.toBe(DEFAULT_SESSION_ID);
    expect(rows.map((row) => row.decisionSummary)).toEqual([
      "Sol reviewed a pending autonomous choice.",
    ]);

    db.close();
  });

  it("does not return self-decision rows on ordinary non-operator participant turns", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-no-leak-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({ db, clock: new FixedClock(NOW_MS) });

    repository.record({
      occurredAt: NOW_MS - 60_000,
      sessionId: DEFAULT_SESSION_ID,
      triggerName: "scheduled_reflection",
      triggerType: "trigger",
      sourceEventId: "scheduled-reflection:no-leak",
      fireEventId: createStreamEntryId(),
      decisionSummary: "This self-private decision must not reach a participant turn.",
      turnResultId: "strm_agent_no_leak",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    expect(
      selectSelfDecisionIntrospection({
        repository,
        sessionAudienceRole: "participant",
        currentSenderBorgRole: null,
        isPrivateSelfCognition: false,
        nowMs: NOW_MS,
      }),
    ).toEqual([]);

    db.close();
  });

  it("uses a multi-day default recency window and caps newest rows", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-self-decisions-window-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "self-decisions.db"), {
      migrations: selfDecisionMigrations,
    });
    const repository = new SelfDecisionRepository({ db, clock: new FixedClock(NOW_MS) });
    const sessionId = createSessionId();
    const withinDefaultWindow = NOW_MS - DEFAULT_SELF_DECISION_INTROSPECTION_RECENCY_WINDOW_MS + 1;

    for (let index = 0; index < DEFAULT_SELF_DECISION_INTROSPECTION_CAP + 2; index += 1) {
      repository.record({
        occurredAt: withinDefaultWindow + index,
        sessionId,
        triggerName: "scheduled_reflection",
        triggerType: "trigger",
        sourceEventId: `scheduled-reflection:${index}`,
        fireEventId: createStreamEntryId(),
        decisionSummary: `Decision ${index}`,
        turnResultId: `strm_agent_${index}`,
        sourceStreamEntryIds: [createStreamEntryId()],
      });
    }
    repository.record({
      occurredAt: NOW_MS - DEFAULT_SELF_DECISION_INTROSPECTION_RECENCY_WINDOW_MS - 1,
      sessionId,
      triggerName: "scheduled_reflection",
      triggerType: "trigger",
      sourceEventId: "scheduled-reflection:old",
      fireEventId: createStreamEntryId(),
      decisionSummary: "Too old",
      turnResultId: "strm_agent_old",
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const rows = selectSelfDecisionIntrospection({
      repository,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: "creator",
      isPrivateSelfCognition: false,
      nowMs: NOW_MS,
    });

    expect(rows).toHaveLength(DEFAULT_SELF_DECISION_INTROSPECTION_CAP);
    expect(rows.map((row) => row.decisionSummary)).toEqual([
      "Decision 9",
      "Decision 8",
      "Decision 7",
      "Decision 6",
      "Decision 5",
      "Decision 4",
      "Decision 3",
      "Decision 2",
    ]);
    expect(rows.map((row) => row.decisionSummary)).not.toContain("Too old");

    db.close();
  });
});
