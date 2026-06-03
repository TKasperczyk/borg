import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSessionId, createStreamEntryId } from "../../util/ids.js";
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
  it("returns rows only for creator-in-operator context and preserves multilingual summaries", () => {
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
      sessionId,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: "creator",
      nowMs: NOW_MS,
    });
    const participantHidden = selectSelfDecisionIntrospection({
      repository,
      sessionId,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: "creator",
      nowMs: NOW_MS,
    });
    const nonCreatorHidden = selectSelfDecisionIntrospection({
      repository,
      sessionId,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: null,
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
    expect(participantHidden).toEqual([]);
    expect(nonCreatorHidden).toEqual([]);

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
      sessionId,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: "creator",
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
