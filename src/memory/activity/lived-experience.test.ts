import { describe, expect, it } from "vitest";

import { createEntityId, createSessionId, createStreamEntryId } from "../../util/ids.js";
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
    expect(rows.map((row) => row.text).join("\n")).toContain("[Jun 15] 12 autonomous reflections");
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
});
