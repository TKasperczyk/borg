import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { SessionsRepository } from "../../sessions/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSessionId, createStreamEntryId, type StreamEntryId } from "../../util/ids.js";
import { selfPrivateMemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  type OfflineTestHarness,
} from "../test-support.js";
import { LIVED_EXPERIENCE_DAY_SUMMARY_TOOL, LivedExperienceDaySummarizerProcess } from "./index.js";

function createSummaryToolResponse(input: {
  utc_day: string;
  gist: string;
  cited_source_stream_entry_ids?: readonly StreamEntryId[];
}) {
  return {
    text: "",
    input_tokens: 20,
    output_tokens: 10,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_lived_day_summary",
        name: LIVED_EXPERIENCE_DAY_SUMMARY_TOOL.name,
        input: {
          utc_day: input.utc_day,
          gist: input.gist,
          salience: 0.8,
          cited_episode_ids: [],
          cited_source_stream_entry_ids: input.cited_source_stream_entry_ids ?? [],
        },
      },
    ],
  };
}

function createProcess(harness: OfflineTestHarness): LivedExperienceDaySummarizerProcess {
  return new LivedExperienceDaySummarizerProcess({
    livedExperienceDaySummaryRepository: harness.livedExperienceDaySummaryRepository,
    registry: harness.registry,
  });
}

describe("LivedExperienceDaySummarizerProcess", () => {
  it("summarizes only closed unsummarized UTC days and persists one disclosure-labeled gist", async () => {
    const nowMs = Date.UTC(2026, 5, 17, 12, 0, 0);
    const targetDayStart = Date.UTC(2026, 5, 15);
    const skippedDayStart = Date.UTC(2026, 5, 16);
    const openDayStart = Date.UTC(2026, 5, 17);
    const targetDecisionSourceId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        createSummaryToolResponse({
          utc_day: "2026-06-15",
          gist: "I held the same restraint across about 2 wakes; the distinct event was a recurring session pattern becoming visible.",
          cited_source_stream_entry_ids: [targetDecisionSourceId],
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: new FixedClock(nowMs),
      configOverrides: {
        offline: {
          livedExperienceDaySummarizer: {
            windowDays: 3,
            maxDaysPerRun: 5,
            maxSelfDecisionEventsPerDay: 10,
            maxActivityEventsPerDay: 10,
            maxEpisodesPerDay: 2,
          },
        },
      },
    });

    try {
      const self = harness.entityRepository.add({
        canonicalName: "Borg self",
        kind: "self",
      });
      const audience = harness.entityRepository.add({
        canonicalName: "Arena audience",
      });
      const sessions = new SessionsRepository({
        db: harness.db,
        clock: harness.clock,
      });
      const sessionId = createSessionId();

      sessions.ensure({
        session_id: sessionId,
        source_type: "botarena",
        label: "Arena thread",
        audience_label: "Arena audience",
        audience_entity_id: audience.id,
        conversation_kind: "thread",
        audience_role: "participant",
        status: "active",
        created_at: targetDayStart,
        last_activity_at: openDayStart,
      });
      harness.selfDecisionRepository.record({
        occurredAt: targetDayStart + 60_000,
        sessionId,
        triggerName: "scheduled_reflection",
        triggerType: "trigger",
        sourceEventId: "scheduled-reflection:2026-06-15",
        fireEventId: createStreamEntryId(),
        decisionSummary: "I held restraint while a repeated pattern recurred.",
        decisionRationale: "The recurring structure was stable enough to note.",
        sourceStreamEntryIds: [targetDecisionSourceId],
      });
      harness.selfDecisionRepository.record({
        occurredAt: skippedDayStart + 60_000,
        sessionId,
        triggerName: "scheduled_reflection",
        triggerType: "trigger",
        sourceEventId: "scheduled-reflection:2026-06-16",
        fireEventId: createStreamEntryId(),
        decisionSummary: "This closed day is already summarized and should be skipped.",
        sourceStreamEntryIds: [createStreamEntryId()],
      });
      harness.selfDecisionRepository.record({
        occurredAt: openDayStart + 60_000,
        sessionId,
        triggerName: "scheduled_reflection",
        triggerType: "trigger",
        sourceEventId: "scheduled-reflection:2026-06-17",
        fireEventId: createStreamEntryId(),
        decisionSummary: "This open day must not be summarized yet.",
        sourceStreamEntryIds: [createStreamEntryId()],
      });
      harness.activityRepository.record({
        kind: "user_contact",
        occurredAt: targetDayStart + 90_000,
        sessionId,
        turnId: "target-turn",
        audienceEntityId: audience.id,
        participantEntityIds: [audience.id],
        sourceStreamEntryIds: [createStreamEntryId()],
      });
      harness.activityRepository.record({
        kind: "borg_replied",
        occurredAt: targetDayStart + 120_000,
        sessionId,
        turnId: "target-turn",
        audienceEntityId: audience.id,
        participantEntityIds: [audience.id],
        sourceStreamEntryIds: [createStreamEntryId()],
      });
      const uncitedEpisode = createEpisodeFixture({
        title: "Closed-day private consolidation evidence",
        narrative: "A prior summary of the same closed day carried unknown disclosure access.",
        start_time: targetDayStart + 30_000,
        end_time: targetDayStart + 45_000,
        created_at: targetDayStart + 45_000,
        updated_at: targetDayStart + 45_000,
        source_stream_ids: [createStreamEntryId()],
        origin_audience_entity_ids: [],
        shared: false,
      });
      await harness.episodicRepository.createEpisode(uncitedEpisode);
      harness.livedExperienceDaySummaryRepository.upsert({
        selfEntityId: self.id,
        utcDay: "2026-06-16",
        dayStartMs: skippedDayStart,
        dayEndMs: skippedDayStart + 24 * 60 * 60_000 - 1,
        gist: "Existing gist.",
        countsSnapshot: {},
        disclosureLabel: selfPrivateMemoryDisclosureLabel([audience.id]),
        provenance: { kind: "offline", process: "lived-experience-day-summarizer" },
      });

      const process = createProcess(harness);
      const plan = await process.plan(harness.createContext(), {});

      expect(plan.errors).toEqual([]);
      expect(plan.items).toHaveLength(1);
      expect(plan.items[0]).toMatchObject({
        action: "upsert_day_summary",
        summary: {
          self_entity_id: self.id,
          utc_day: "2026-06-15",
          disclosure_label: expect.objectContaining({
            disclosureClass: "unknown",
            originAudienceEntityIds: [audience.id],
          }),
        },
        previous: null,
      });
      expect(llm.requests).toHaveLength(1);
      expect(llm.requests[0]?.tool_choice).toEqual({
        type: "tool",
        name: LIVED_EXPERIENCE_DAY_SUMMARY_TOOL.name,
      });
      expect(llm.requests[0]?.budget).toBe("offline-lived-experience-day-summarizer");
      expect(String(llm.requests[0]?.system ?? "")).toContain("experiential narrative only");
      expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
        "I do not decide whether any wake",
      );
      expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
        "I held restraint while a repeated pattern recurred.",
      );

      const result = await process.apply(harness.createContext(), plan);

      expect(result.changes).toHaveLength(1);
      expect(
        harness.livedExperienceDaySummaryRepository.listForWindow({
          selfEntityId: self.id,
          fromMs: targetDayStart,
          toMs: skippedDayStart + 24 * 60 * 60_000 - 1,
          limit: 10,
        }),
      ).toEqual([
        expect.objectContaining({
          utc_day: "2026-06-15",
          gist: "I held the same restraint across about 2 wakes; the distinct event was a recurring session pattern becoming visible.",
          source_episode_ids: [uncitedEpisode.id],
          source_stream_entry_ids: [targetDecisionSourceId],
          disclosure_label: expect.objectContaining({
            disclosureClass: "unknown",
          }),
        }),
        expect.objectContaining({
          utc_day: "2026-06-16",
          gist: "Existing gist.",
        }),
      ]);
    } finally {
      await harness.cleanup();
    }
  });
});
