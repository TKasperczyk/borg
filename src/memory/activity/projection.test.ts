import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { EntityRepository, commitmentMigrations } from "../commitments/index.js";
import { SessionsRepository, sessionMigrations } from "../../sessions/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import { activityMigrations } from "./migrations.js";
import { selectCrossSessionSelfActivity } from "./projection.js";
import { ActivityRepository } from "./repository.js";

const NOW_MS = 1_000_000;
const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

describe("selectCrossSessionSelfActivity", () => {
  it("shows other active-session contact only to operator sessions from the creator", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations, commitmentMigrations),
    });
    const sessions = new SessionsRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });
    const activity = new ActivityRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });
    const currentSessionId = createSessionId();
    const otherSessionId = createSessionId();
    const operatorEntityId = createEntityId();
    const aliceEntityId = createEntityId();

    sessions.ensure({
      session_id: currentSessionId,
      source_type: "demo",
      label: "Operator control",
      audience_label: "Tom",
      audience_entity_id: operatorEntityId,
      conversation_kind: "demo",
      audience_role: "operator",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS,
    });
    sessions.ensure({
      session_id: otherSessionId,
      source_type: "demo",
      label: "Alice demo",
      audience_label: "Alice",
      audience_entity_id: null,
      conversation_kind: "demo",
      audience_role: "participant",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS - 41_000,
    });
    activity.record({
      kind: "user_contact",
      occurredAt: NOW_MS - 41_000,
      sessionId: otherSessionId,
      turnId: "alice-turn",
      speakerEntityId: aliceEntityId,
      actorEntityId: aliceEntityId,
      audienceEntityId: aliceEntityId,
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const visible = selectCrossSessionSelfActivity({
      repository: activity,
      currentSessionId,
      currentAudienceEntityId: operatorEntityId,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: "creator",
      isPrivateSelfCognition: false,
      nowMs: NOW_MS,
      recencyWindowMs: 60_000,
      cap: 5,
    });
    const hidden = selectCrossSessionSelfActivity({
      repository: activity,
      currentSessionId,
      currentAudienceEntityId: aliceEntityId,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: null,
      isPrivateSelfCognition: false,
      nowMs: NOW_MS,
      recencyWindowMs: 60_000,
      cap: 5,
    });

    expect(visible).toEqual([
      {
        kind: "user_contact",
        occurredAt: NOW_MS - 41_000,
        relativeAge: "~41s ago",
        text: "Alice contacted Borg ~41s ago in another active session.",
      },
    ]);
    expect(hidden).toEqual([]);

    db.close();
  });

  it("shows other active-session activity during private self-cognition", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-private-self-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations, commitmentMigrations),
    });
    const sessions = new SessionsRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });
    const activity = new ActivityRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });
    const currentSessionId = createSessionId();
    const otherSessionId = createSessionId();
    const aliceEntityId = createEntityId();

    sessions.ensure({
      session_id: currentSessionId,
      source_type: "demo",
      label: "Autonomous wake",
      audience_label: "self",
      audience_entity_id: null,
      conversation_kind: "demo",
      audience_role: "participant",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS,
    });
    sessions.ensure({
      session_id: otherSessionId,
      source_type: "demo",
      label: "Alice demo",
      audience_label: "Alice",
      audience_entity_id: aliceEntityId,
      conversation_kind: "demo",
      audience_role: "participant",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS - 30_000,
    });
    activity.record({
      kind: "turn_completed",
      occurredAt: NOW_MS - 30_000,
      sessionId: otherSessionId,
      turnId: "alice-turn",
      speakerEntityId: aliceEntityId,
      actorEntityId: aliceEntityId,
      audienceEntityId: aliceEntityId,
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const visible = selectCrossSessionSelfActivity({
      repository: activity,
      currentSessionId,
      currentAudienceEntityId: null,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: null,
      isPrivateSelfCognition: true,
      nowMs: NOW_MS,
      recencyWindowMs: 60_000,
      cap: 5,
    });

    expect(visible).toEqual([
      {
        kind: "turn_completed",
        occurredAt: NOW_MS - 30_000,
        relativeAge: "~30s ago",
        text: "Borg completed a turn with Alice ~30s ago in another active session.",
      },
    ]);

    db.close();
  });

  it("does not return cross-session activity on ordinary non-operator participant turns", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-no-leak-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations, commitmentMigrations),
    });
    const sessions = new SessionsRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });
    const activity = new ActivityRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });
    const currentSessionId = createSessionId();
    const otherSessionId = createSessionId();
    const participantEntityId = createEntityId();

    sessions.ensure({
      session_id: currentSessionId,
      source_type: "demo",
      label: "Participant room",
      audience_label: "Sam",
      audience_entity_id: participantEntityId,
      conversation_kind: "demo",
      audience_role: "participant",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS,
    });
    sessions.ensure({
      session_id: otherSessionId,
      source_type: "demo",
      label: "Other private room",
      audience_label: "Alice",
      audience_entity_id: createEntityId(),
      conversation_kind: "demo",
      audience_role: "participant",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS - 30_000,
    });
    activity.record({
      kind: "user_contact",
      occurredAt: NOW_MS - 30_000,
      sessionId: otherSessionId,
      turnId: "other-turn",
      speakerEntityId: createEntityId(),
      actorEntityId: createEntityId(),
      audienceEntityId: createEntityId(),
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    expect(
      selectCrossSessionSelfActivity({
        repository: activity,
        currentSessionId,
        currentAudienceEntityId: participantEntityId,
        sessionAudienceRole: "participant",
        currentSenderBorgRole: null,
        isPrivateSelfCognition: false,
        nowMs: NOW_MS,
        recencyWindowMs: 60_000,
        cap: 5,
      }),
    ).toEqual([]);

    db.close();
  });

  it("labels a group-session contact by the speaker, not the group audience", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-activity-projection-group-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(sessionMigrations, activityMigrations, commitmentMigrations),
    });
    const sessions = new SessionsRepository({ db, clock: new FixedClock(NOW_MS) });
    const activity = new ActivityRepository({ db, clock: new FixedClock(NOW_MS) });
    const entities = new EntityRepository({ db, clock: new FixedClock(NOW_MS) });

    const currentSessionId = createSessionId();
    const groupSessionId = createSessionId();
    const operatorEntityId = createEntityId();
    const groupEntityId = entities.resolve("Planning Room", { kind: "group" });
    const bobEntityId = entities.resolve("Bob", { kind: "person" });

    sessions.ensure({
      session_id: currentSessionId,
      source_type: "demo",
      label: "Operator control",
      audience_label: "Tom",
      audience_entity_id: operatorEntityId,
      conversation_kind: "demo",
      audience_role: "operator",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS,
    });
    sessions.ensure({
      session_id: groupSessionId,
      source_type: "demo",
      label: "Planning Room",
      audience_label: "Planning Room",
      audience_entity_id: groupEntityId,
      conversation_kind: "demo",
      audience_role: "participant",
      status: "active",
      created_at: NOW_MS - 120_000,
      last_activity_at: NOW_MS - 30_000,
    });
    activity.record({
      kind: "user_contact",
      occurredAt: NOW_MS - 30_000,
      sessionId: groupSessionId,
      turnId: "bob-turn",
      speakerEntityId: bobEntityId,
      actorEntityId: bobEntityId,
      audienceEntityId: groupEntityId,
      sourceStreamEntryIds: [createStreamEntryId()],
    });

    const visible = selectCrossSessionSelfActivity({
      repository: activity,
      currentSessionId,
      currentAudienceEntityId: operatorEntityId,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: "creator",
      isPrivateSelfCognition: false,
      nowMs: NOW_MS,
      recencyWindowMs: 60_000,
      cap: 5,
    });

    expect(visible).toEqual([
      {
        kind: "user_contact",
        occurredAt: NOW_MS - 30_000,
        relativeAge: "~30s ago",
        text: "Bob contacted Borg ~30s ago in another active session.",
      },
    ]);

    db.close();
  });
});
