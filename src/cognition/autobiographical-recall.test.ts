import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { ObservedEventRepository } from "../memory/observed-events/index.js";
import type { SelfDecisionRepository } from "../memory/self-decisions/index.js";
import type { SessionRecord } from "../sessions/index.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import { FixedClock, ManualClock } from "../util/clock.js";
import {
  createEntityId,
  createObservedEventId,
  createSessionId,
  createStreamEntryId,
  type SessionId,
} from "../util/ids.js";
import { AutobiographicalRecallService } from "./autobiographical-recall.js";
import { createSectionBuckets, finalSections } from "./evidence-ledger/section-buckets.js";
import { renderSection } from "./evidence-ledger/section-rendering.js";
import { addAutobiographicalRecallSection } from "./evidence-ledger/sections/autobiographical-recall.js";

const NOW_MS = 10_000;
const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

function sessionRecord(sessionId: SessionId, lastActivityAt: number): SessionRecord {
  return {
    session_id: sessionId,
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: "demo session",
    audience_label: "operator",
    audience_entity_id: createEntityId(),
    conversation_kind: "demo",
    created_at: 0,
    last_activity_at: lastActivityAt,
    last_turn_id: null,
    message_count: 1,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "operator",
  };
}

describe("AutobiographicalRecallService", () => {
  it("preserves self-decision and observed-event source stream anchors", async () => {
    const selfDecisionSourceEntryId = createStreamEntryId();
    const observedEventSourceEntryId = createStreamEntryId();
    const selfDecisionRepository: Pick<SelfDecisionRepository, "listRecentAutonomousSelfPrivate"> =
      {
        listRecentAutonomousSelfPrivate: () => [
          {
            occurredAt: 2_000,
            triggerName: "scheduled_reflection",
            triggerType: "trigger",
            decisionSummary: "Sol reviewed the last arena exchange.",
            decisionRationale: "The recent exchange left an unresolved calibration question.",
            sourceStreamEntryIds: [selfDecisionSourceEntryId],
          },
        ],
      };
    const observedEventRepository: Pick<ObservedEventRepository, "listRecentGlobal"> = {
      listRecentGlobal: (input) =>
        input.disclosureClass === "social_observed"
          ? [
              {
                id: createObservedEventId(),
                occurredAt: 2_100,
                lastSeenAt: 2_100,
                stance: "rejected_frame",
                taint: "quarantined",
                beliefEffect: "unchanged",
                disclosureClass: "social_observed",
                interactionText: "Someone pushed a rejected frame during arena review.",
                recurrenceCount: 1,
                speakerEntityId: createEntityId(),
                audienceEntityId: createEntityId(),
                sourceStreamEntryIds: [observedEventSourceEntryId],
              },
            ]
          : [],
    };
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      selfDecisionRepository,
      observedEventRepository,
      sourceCap: 5,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId: createSessionId(),
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 3_000,
        label: "recent activity",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    expect(result?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "self_decision",
          sourceStreamEntryIds: [selfDecisionSourceEntryId],
        }),
        expect.objectContaining({
          kind: "observed_social_event",
          sourceStreamEntryIds: [observedEventSourceEntryId],
        }),
      ]),
    );

    const buckets = createSectionBuckets();
    addAutobiographicalRecallSection({
      input: {
        autobiographicalRecall: result,
        audienceEntityId: null,
      } as never,
      buckets,
      resolver: {} as never,
      options: {} as never,
      transcript: {} as never,
      streamEntries: [],
      repos: {} as never,
    });
    const section = finalSections(buckets).find((item) => item.id === "autobiographical_recall");

    expect(section?.framing).toEqual({
      text: expect.stringContaining("past evidence"),
      counts: {
        self_decision: 1,
      },
    });
    expect(section?.entries).toHaveLength(2);
    expect(section?.entries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          value: "self_decisions/self_decision",
          state_metadata: expect.objectContaining({
            source_stream_ids: [selfDecisionSourceEntryId],
          }),
        }),
        expect.objectContaining({
          value: "observed_social_events/observed_social_event",
          state_metadata: expect.objectContaining({
            source_stream_ids: [observedEventSourceEntryId],
          }),
        }),
      ]),
    );
    expect(renderSection(section!)).toContain('framing_counts: {"self_decision":1}');
  });

  it("caps stream autobiographical evidence after selecting the most recent in-window entries", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autobiographical-recall-"));
    tempDirs.push(tempDir);
    const sessionId = createSessionId();
    const clock = new ManualClock(1_000);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId,
      clock,
    });

    for (let index = 1; index <= 5; index += 1) {
      clock.set(1_000 + index * 1_000);
      await writer.append({
        kind: "thought",
        content: `thought-${index}`,
      });
    }

    writer.close();

    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      sessionsRepository: {
        list: () => [sessionRecord(sessionId, 6_000)],
      },
      createStreamReader: (readerSessionId) =>
        new StreamReader({ dataDir: tempDir, sessionId: readerSessionId }),
      sourceCap: 3,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId,
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 7_000,
        label: "busy recent window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });
    const streamEvidence = result?.evidence.filter((item) => item.kind === "stream_reflection");

    expect(streamEvidence?.map((item) => item.text)).toEqual([
      expect.stringContaining("thought-5"),
      expect.stringContaining("thought-4"),
      expect.stringContaining("thought-3"),
    ]);
    expect(streamEvidence?.map((item) => item.text).join("\n")).not.toContain("thought-1");
    expect(streamEvidence?.map((item) => item.text).join("\n")).not.toContain("thought-2");
  });

  it("labels self-memory extraction epoch in autobiographical metadata", async () => {
    const beforeInversion = Date.parse("2026-06-04T14:43:26Z");
    const afterInversion = Date.parse("2026-06-04T14:43:28Z");
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(afterInversion + 1_000),
      goalsRepository: {
        list: () => [
          {
            id: "goal_aaaaaaaaaaaaaaaa",
            description: "Review recall-inversion provenance",
            priority: 6,
            parent_goal_id: null,
            status: "active",
            progress_notes: null,
            last_progress_ts: null,
            created_at: afterInversion,
            target_at: null,
            audience_entity_id: null,
            owner_entity_id: null,
            source_stream_entry_ids: [createStreamEntryId()],
            provenance: { kind: "manual" },
            children: [],
          },
        ],
      } as never,
      autobiographicalRepository: {
        listPeriods: () => [
          {
            id: "abp_aaaaaaaaaaaaaaaa",
            label: "Before inversion",
            start_ts: beforeInversion,
            end_ts: null,
            narrative: "A period extracted before global self-memory recall.",
            key_episode_ids: [],
            themes: ["recall"],
            provenance: { kind: "manual" },
            created_at: beforeInversion,
            last_updated: beforeInversion,
          },
        ],
      } as never,
      sourceCap: 5,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId: createSessionId(),
      temporalCue: {
        sinceTs: beforeInversion - 1_000,
        untilTs: afterInversion + 1_000,
        label: "recall inversion boundary",
      },
      isSelfAudience: true,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    expect(result?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "goal",
          metadata: expect.objectContaining({
            self_memory_provenance: "extraction_epoch=extracted_after_recall_inversion",
          }),
        }),
        expect.objectContaining({
          kind: "autobiographical_period",
          metadata: expect.objectContaining({
            self_memory_provenance: "extraction_epoch=extracted_before_recall_inversion",
          }),
        }),
      ]),
    );
  });
});
