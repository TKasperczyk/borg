import { describe, expect, it } from "vitest";

import {
  createAutobiographicalPeriodId,
  createEntityId,
  createSessionId,
  createStreamEntryId,
} from "../../util/ids.js";
import { unknownMemoryDisclosureLabel } from "../common/disclosure-label.js";
import { selectRecentLivedExperienceRows } from "./lived-experience.js";

describe("selectRecentLivedExperienceRows", () => {
  it("keeps recent individual rows and collapses older days into density rows", () => {
    const nowMs = Date.UTC(2026, 5, 17, 12, 0, 0);
    const oldActivityAt = Date.UTC(2026, 5, 15, 10, 0, 0);
    const oldDecisionAt = Date.UTC(2026, 5, 15, 11, 0, 0);
    const recentActivityAt = Date.UTC(2026, 5, 17, 10, 0, 0);
    const audienceEntityId = createEntityId();
    const sessionId = createSessionId();
    const oldSourceId = createStreamEntryId();
    const recentSourceId = createStreamEntryId();
    const rows = selectRecentLivedExperienceRows({
      nowMs,
      crossSessionSelfActivity: [
        {
          kind: "user_contact",
          occurredAt: oldActivityAt,
          sessionId,
          relativeAge: "2d ago",
          text: "Old detailed activity should be collapsed.",
          originAudienceEntityIds: [audienceEntityId],
          sourceStreamEntryIds: [oldSourceId],
        },
        {
          kind: "borg_replied",
          occurredAt: recentActivityAt,
          sessionId,
          relativeAge: "2h ago",
          text: "Borg replied to BotArena group 2h ago in another active session.",
          originAudienceEntityIds: [audienceEntityId],
          sourceStreamEntryIds: [recentSourceId],
        },
      ],
      selfDecisionIntrospection: [
        {
          occurredAt: oldDecisionAt,
          relativeAge: "2d ago",
          triggerName: "scheduled_reflection",
          triggerType: "trigger",
          decisionSummary: "Old decision should be collapsed.",
          decisionRationale: null,
          sourceStreamEntryIds: [createStreamEntryId()],
          text: "Old detailed reflection should be collapsed.",
        },
      ],
      activityDensity: [
        {
          dayKey: "2026-06-15",
          dayStartMs: Date.UTC(2026, 5, 15),
          sessionId,
          sessionLabel: "Arena thread",
          audienceLabel: "BotArena group",
          audienceEntityId,
          eventCount: 51,
          conversationTurnCount: 20,
          kindCounts: {
            userContact: 20,
            borgReplied: 20,
            turnCompleted: 11,
          },
          firstOccurredAt: oldActivityAt,
          lastOccurredAt: Date.UTC(2026, 5, 15, 20, 0, 0),
        },
      ],
      selfDecisionDensity: [
        {
          dayKey: "2026-06-15",
          dayStartMs: Date.UTC(2026, 5, 15),
          decisionCount: 12,
          distinctDecisionShapeCount: 3,
          firstOccurredAt: oldDecisionAt,
          lastOccurredAt: Date.UTC(2026, 5, 15, 21, 0, 0),
        },
      ],
    });

    expect(rows.map((row) => row.kind)).toEqual([
      "cross_session_activity_density",
      "self_decision_density",
      "cross_session_activity",
    ]);
    expect(rows.map((row) => row.text).join("\n")).toContain(
      "[Jun 15] 20 conversation turns with BotArena group",
    );
    expect(rows.map((row) => row.text).join("\n")).toContain(
      "[Jun 15] 12 autonomous reflections; 3 distinct structural patterns",
    );
    expect(rows.map((row) => row.text).join("\n")).not.toContain("Old detailed");
    expect(rows.at(-1)).toMatchObject({
      kind: "cross_session_activity",
      sourceStreamEntryIds: [recentSourceId],
    });
  });

  it("collapses older activity detail by UTC day and session, not day alone", () => {
    const nowMs = Date.UTC(2026, 5, 17, 12, 0, 0);
    const oldActivityAt = Date.UTC(2026, 5, 15, 10, 0, 0);
    const audienceEntityId = createEntityId();
    const collapsedSessionId = createSessionId();
    const retainedSessionId = createSessionId();
    const retainedSourceId = createStreamEntryId();
    const rows = selectRecentLivedExperienceRows({
      nowMs,
      crossSessionSelfActivity: [
        {
          kind: "user_contact",
          occurredAt: oldActivityAt,
          sessionId: collapsedSessionId,
          relativeAge: "2d ago",
          text: "Collapsed session detail should be removed.",
          originAudienceEntityIds: [audienceEntityId],
          sourceStreamEntryIds: [createStreamEntryId()],
        },
        {
          kind: "user_contact",
          occurredAt: oldActivityAt + 60_000,
          sessionId: retainedSessionId,
          relativeAge: "2d ago",
          text: "Retained same-day different-session detail.",
          originAudienceEntityIds: [audienceEntityId],
          sourceStreamEntryIds: [retainedSourceId],
        },
      ],
      selfDecisionIntrospection: [],
      activityDensity: [
        {
          dayKey: "2026-06-15",
          dayStartMs: Date.UTC(2026, 5, 15),
          sessionId: collapsedSessionId,
          sessionLabel: "Arena thread",
          audienceLabel: "BotArena group",
          audienceEntityId,
          eventCount: 3,
          conversationTurnCount: 1,
          kindCounts: {
            userContact: 1,
            borgReplied: 1,
            turnCompleted: 1,
          },
          firstOccurredAt: oldActivityAt,
          lastOccurredAt: oldActivityAt + 120_000,
        },
      ],
      selfDecisionDensity: [],
    });
    const text = rows.map((row) => row.text).join("\n");

    expect(text).toContain("1 conversation turn with BotArena group");
    expect(text).toContain("Retained same-day different-session detail.");
    expect(text).not.toContain("Collapsed session detail should be removed.");
    expect(rows.find((row) => row.text.includes("Retained"))?.sourceStreamEntryIds).toEqual([
      retainedSourceId,
    ]);
  });

  it("adds a return-silence spine anchor from the current-session previous turn", () => {
    const nowMs = Date.UTC(2026, 5, 19, 18, 0, 0);
    const previousTurnAt = Date.UTC(2026, 5, 16, 17, 36, 0);
    const rows = selectRecentLivedExperienceRows({
      nowMs,
      crossSessionSelfActivity: [],
      selfDecisionIntrospection: [],
      activityDensity: [],
      selfDecisionDensity: [],
      returnSilence: {
        currentAudienceLabel: "Tom",
        currentSessionPreviousTurnAt: previousTurnAt,
      },
    });

    expect(rows).toEqual([
      expect.objectContaining({
        kind: "return_silence_delta",
        occurredAt: previousTurnAt,
        relativeAge: "3d ago",
        text: "Returned to Tom; last engaged this current session 3d ago.",
        metadata: expect.objectContaining({
          current_session_previous_turn_at: previousTurnAt,
          returned_at: nowMs,
          disclosure_class: "self_private",
        }),
      }),
    ]);
  });

  it("telescopes only period-covered older spine days and clamps periods before retained recent days", () => {
    const nowMs = Date.UTC(2026, 5, 20, 12, 0, 0);
    const oldActivityAt = Date.UTC(2026, 5, 10, 10, 0, 0);
    const oldDecisionAt = Date.UTC(2026, 5, 10, 11, 0, 0);
    const retainedRecentAt = Date.UTC(2026, 5, 15, 10, 0, 0);
    const audienceEntityId = createEntityId();
    const sessionId = createSessionId();
    const rows = selectRecentLivedExperienceRows({
      nowMs,
      crossSessionSelfActivity: [],
      selfDecisionIntrospection: [],
      activityDensity: [
        {
          dayKey: "2026-06-10",
          dayStartMs: Date.UTC(2026, 5, 10),
          sessionId,
          sessionLabel: "Arena thread",
          audienceLabel: "BotArena group",
          audienceEntityId,
          eventCount: 30,
          conversationTurnCount: 10,
          kindCounts: {
            userContact: 10,
            borgReplied: 10,
            turnCompleted: 10,
          },
          firstOccurredAt: oldActivityAt,
          lastOccurredAt: Date.UTC(2026, 5, 10, 20, 0, 0),
        },
        {
          dayKey: "2026-06-15",
          dayStartMs: Date.UTC(2026, 5, 15),
          sessionId,
          sessionLabel: "Arena thread",
          audienceLabel: "BotArena group",
          audienceEntityId,
          eventCount: 15,
          conversationTurnCount: 5,
          kindCounts: {
            userContact: 5,
            borgReplied: 5,
            turnCompleted: 5,
          },
          firstOccurredAt: retainedRecentAt,
          lastOccurredAt: Date.UTC(2026, 5, 15, 12, 0, 0),
        },
      ],
      selfDecisionDensity: [
        {
          dayKey: "2026-06-10",
          dayStartMs: Date.UTC(2026, 5, 10),
          decisionCount: 28,
          distinctDecisionShapeCount: 1,
          firstOccurredAt: oldDecisionAt,
          lastOccurredAt: Date.UTC(2026, 5, 10, 21, 0, 0),
        },
      ],
      autobiographicalPeriods: [
        {
          id: createAutobiographicalPeriodId(),
          record_version: 1,
          label: "June integration period",
          start_ts: Date.UTC(2026, 5, 1, 0, 0, 0),
          end_ts: null,
          narrative: "Narrative text must not be rendered by this deterministic surface.",
          key_episode_ids: [],
          disclosure_label: unknownMemoryDisclosureLabel(),
          themes: [],
          provenance: { kind: "manual" },
          created_at: nowMs,
          last_updated: nowMs,
        },
      ],
    });
    const text = rows.map((row) => row.text).join("\n");

    expect(rows.map((row) => row.kind)).toEqual([
      "autobiographical_period",
      "cross_session_activity_density",
    ]);
    expect(text).toContain("[Jun 10-12] autobiographical period: June integration period.");
    expect(text).toContain("[Jun 15] 5 conversation turns with BotArena group");
    expect(text).not.toContain("10 conversation turns");
    expect(text).not.toContain("28 autonomous reflections");
    expect(text).not.toContain("Narrative text");
    expect(rows[0]).toMatchObject({
      disclosureLabel: expect.objectContaining({ disclosureClass: "unknown" }),
      metadata: expect.objectContaining({
        clamped_period_start_ts: oldActivityAt,
        clamped_period_end_ts: Date.UTC(2026, 5, 12, 23, 59, 59, 999),
      }),
    });
  });

  it("keeps older density rows when no emitted period covers their day", () => {
    const nowMs = Date.UTC(2026, 5, 20, 12, 0, 0);
    const oldActivityAt = Date.UTC(2026, 5, 10, 10, 0, 0);
    const audienceEntityId = createEntityId();
    const sessionId = createSessionId();
    const densityRow = {
      dayKey: "2026-06-10",
      dayStartMs: Date.UTC(2026, 5, 10),
      sessionId,
      sessionLabel: "Arena thread",
      audienceLabel: "BotArena group",
      audienceEntityId,
      eventCount: 30,
      conversationTurnCount: 10,
      kindCounts: {
        userContact: 10,
        borgReplied: 10,
        turnCompleted: 10,
      },
      firstOccurredAt: oldActivityAt,
      lastOccurredAt: Date.UTC(2026, 5, 10, 20, 0, 0),
    };
    const period = {
      id: createAutobiographicalPeriodId(),
      record_version: 1,
      label: "Unrelated period",
      start_ts: Date.UTC(2026, 4, 1, 0, 0, 0),
      end_ts: Date.UTC(2026, 4, 5, 0, 0, 0),
      narrative: "Unrelated narrative.",
      key_episode_ids: [],
      disclosure_label: unknownMemoryDisclosureLabel(),
      themes: [],
      provenance: { kind: "manual" as const },
      created_at: nowMs,
      last_updated: nowMs,
    };
    const rowsWithoutCoveredPeriod = selectRecentLivedExperienceRows({
      nowMs,
      crossSessionSelfActivity: [],
      selfDecisionIntrospection: [],
      activityDensity: [densityRow],
      selfDecisionDensity: [],
      autobiographicalPeriods: [period],
    });
    const rowsWithPeriodCapZero = selectRecentLivedExperienceRows({
      nowMs,
      crossSessionSelfActivity: [],
      selfDecisionIntrospection: [],
      activityDensity: [densityRow],
      selfDecisionDensity: [],
      autobiographicalPeriods: [
        {
          ...period,
          start_ts: Date.UTC(2026, 5, 1, 0, 0, 0),
          end_ts: Date.UTC(2026, 5, 12, 0, 0, 0),
        },
      ],
      periodCap: 0,
    });

    expect(rowsWithoutCoveredPeriod.map((row) => row.kind)).toEqual([
      "cross_session_activity_density",
    ]);
    expect(rowsWithPeriodCapZero.map((row) => row.kind)).toEqual([
      "cross_session_activity_density",
    ]);
  });
});
