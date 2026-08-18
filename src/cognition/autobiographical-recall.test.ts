import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { ActionRecord } from "../memory/actions/index.js";
import type { EpisodeStats } from "../memory/episodic/types.js";
import type { ObservedEventRepository } from "../memory/observed-events/index.js";
import type { GoalTreeNode, OpenQuestion } from "../memory/self/index.js";
import type {
  SelfDecisionProjectionSourceEvent,
  SelfDecisionRepository,
} from "../memory/self-decisions/index.js";
import { createWorkingMemory } from "../memory/working/index.js";
import { createEpisodeFixture } from "../offline/test-support.js";
import { selfPrivateMemoryDisclosureLabel } from "../retrieval/index.js";
import type { SessionRecord } from "../sessions/index.js";
import { StreamReader, StreamWriter } from "../stream/index.js";
import { OUTBOUND_POST_TOOL_NAME } from "../tools/internal/outbound-post-name.js";
import { FixedClock, ManualClock } from "../util/clock.js";
import {
  createActionId,
  createEntityId,
  createGoalId,
  createObservedEventId,
  createOpenQuestionId,
  createSessionId,
  createStreamEntryId,
  type EpisodeId,
  type SessionId,
} from "../util/ids.js";
import {
  AutobiographicalRecallService,
  type AutobiographicalRecallResult,
} from "./autobiographical-recall.js";
import { EvidenceLedgerBuilder } from "./evidence-ledger/builder.js";
import { compactEvidenceLedger, renderEvidenceLedger } from "./evidence-ledger/renderer.js";
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

function goalNode(index: number, overrides: Partial<GoalTreeNode> = {}): GoalTreeNode {
  return {
    id: createGoalId(),
    record_version: 1,
    description: `Goal ${index}`,
    terminal_condition: null,
    priority: 1,
    parent_goal_id: null,
    status: "active",
    progress_notes: null,
    last_progress_ts: 2_000 + index,
    created_at: 1_000 + index,
    target_at: null,
    audience_entity_id: null,
    owner_entity_id: null,
    canonicalized_by_artifact_entry_id: null,
    provenance: { kind: "manual" },
    children: [],
    ...overrides,
  };
}

function selfDecisionEvents(count: number): SelfDecisionProjectionSourceEvent[] {
  return Array.from({ length: count }, (_, index) => ({
    occurredAt: 2_000 + index,
    sourceEventId: `structural-outcome-${index}`,
    triggerName: `trigger-${index}`,
    triggerType: "trigger",
    decisionSummary: `Decision ${index}`,
    decisionRationale: null,
    sourceStreamEntryIds: [createStreamEntryId()],
  }));
}

function openQuestion(index: number, lastTouched: number, urgency: number): OpenQuestion {
  return {
    id: createOpenQuestionId(),
    record_version: 1,
    question: `Question ${index}`,
    urgency,
    status: "open",
    goal_id: null,
    audience_entity_id: null,
    related_episode_ids: [],
    related_semantic_node_ids: [],
    provenance: { kind: "manual" },
    source: "user",
    created_at: lastTouched,
    last_touched: lastTouched,
    resolution_evidence_episode_ids: [],
    resolution_evidence_stream_entry_ids: [],
    resolution_note: null,
    resolved_at: null,
    abandoned_reason: null,
    abandoned_at: null,
    resolved_by_artifact_entry_id: null,
    unresolved_rumination_ticks: 0,
    last_ruminated_at: null,
  };
}

function completedAction(index: number, updatedAt: number, completedAt: number): ActionRecord {
  return {
    id: createActionId(),
    description: `Action ${index}`,
    actor: "borg",
    audience_entity_id: null,
    goal_id: null,
    open_question_id: null,
    state: "completed",
    confidence: 0.9,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [createStreamEntryId()],
    created_at: 100,
    updated_at: updatedAt,
    considering_at: null,
    committed_at: null,
    scheduled_at: null,
    completed_at: completedAt,
    not_done_at: null,
    expired_at: null,
    archived_at: null,
    unknown_at: null,
    canonicalized_by_artifact_entry_id: null,
    session_scope: null,
    session_anchor_id: null,
    last_referenced_at_ms: updatedAt,
    last_referenced_turn_counter: null,
  };
}

function autobiographicalSection(recall: AutobiographicalRecallResult) {
  const buckets = createSectionBuckets();
  addAutobiographicalRecallSection({
    input: {
      autobiographicalRecall: recall,
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

  if (section === undefined) {
    throw new Error("Expected autobiographical recall section");
  }

  return section;
}

function hasAutobiographicalRecallCapMetadata(entry: {
  state_metadata?: Record<string, unknown>;
}): boolean {
  return "autobiographical_recall_cap" in (entry.state_metadata ?? {});
}

async function recallGoals(input: {
  goals: readonly GoalTreeNode[];
  sourceCap: number;
  totalCap: number;
  temporalCue?: {
    sinceTs: number;
    untilTs: number;
    label: string;
  };
}): Promise<AutobiographicalRecallResult> {
  const service = new AutobiographicalRecallService({
    clock: new FixedClock(NOW_MS),
    goalsRepository: {
      list: () => [...input.goals],
    },
    sourceCap: input.sourceCap,
    totalCap: input.totalCap,
  });
  const result = await service.recall({
    sessionId: createSessionId(),
    temporalCue: input.temporalCue ?? {
      sinceTs: 1_000,
      untilTs: 9_000,
      label: "goal window",
    },
    isSelfAudience: false,
    sessionAudienceRole: "operator",
    perceptionMode: "reflective",
  });

  if (result === null) {
    throw new Error("Expected autobiographical recall result");
  }

  return result;
}

function episodeStatsFixture(episodeId: EpisodeId): EpisodeStats {
  return {
    episode_id: episodeId,
    retrieval_count: 0,
    use_count: 0,
    last_retrieved: null,
    win_rate: 0,
    tier: "T2",
    promoted_at: 0,
    promoted_from: null,
    gist: null,
    gist_generated_at: null,
    last_decayed_at: null,
    heat_multiplier: 1,
    valence_mean: 0,
    archived: false,
  };
}

describe("AutobiographicalRecallService", () => {
  // DELIBERATE: autobiographical recall is a second, independent path that puts an episode into the
  // turn prompt. It reads through listRecentForCognition and records nothing, while the retrieval
  // pipeline's projection loop is the only caller of recordRetrieval. So retrieval_count/last_retrieved
  // count pipeline exposures, not prompt exposures, and an episode surfaced only through this path
  // stays at count 0 however often the entity actually saw it. That undercount is an input to episodic
  // heat, decay half-life, curator tiering/archival and associator anchor scoring -- do not read the
  // counter as "times this was in front of the entity". This test pins the asymmetry so a later
  // refactor cannot quietly make it look like one path.
  it("renders episodes into recall evidence without recording a retrieval", async () => {
    const episode = createEpisodeFixture({
      title: "Arena exchange",
      narrative: "Sol and the operator reviewed the last arena exchange.",
      created_at: 2_000,
      start_time: 2_000,
      end_time: 2_500,
    });
    const recordedRetrievals: EpisodeId[] = [];
    const episodicRepository = {
      listRecentForCognition: () =>
        Promise.resolve([{ episode, stats: episodeStatsFixture(episode.id), similarity: 0.9 }]),
      recordRetrieval: (episodeId: EpisodeId) => {
        recordedRetrievals.push(episodeId);
      },
    };
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      episodicRepository,
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
          kind: "episode",
          sourceEpisodeIds: [episode.id],
        }),
      ]),
    );
    expect(recordedRetrievals).toEqual([]);
  });

  it("preserves self-decision and observed-event source stream anchors", async () => {
    const selfDecisionSourceEntryId = createStreamEntryId();
    const observedEventSourceEntryId = createStreamEntryId();
    const selfDecisionRepository: Pick<SelfDecisionRepository, "listRecentAutonomousSelfPrivate"> =
      {
        listRecentAutonomousSelfPrivate: () => [
          {
            occurredAt: 2_000,
            sourceEventId: "scheduled-reflection:arena-exchange",
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

  it("renders goal terminal conditions on the autobiographical surface", async () => {
    const goal = goalNode(0, {
      description: "Track the release readiness decision",
      terminal_condition: "The release readiness decision is made",
      priority: 5,
      status: "active",
      progress_notes: "Compared rollback and launch options.",
    });
    const result = await recallGoals({
      goals: [goal],
      sourceCap: 5,
      totalCap: 10,
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 3_000,
        label: "release readiness window",
      },
    });

    expect(result?.evidence).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "goal",
          text: expect.stringContaining(
            "terminal_condition=The release readiness decision is made",
          ),
        }),
      ]),
    );
  });

  it("retains the newest progress notes when the goal's append-only log exceeds the prompt budget", async () => {
    const oldestNote = `[1] ${"o".repeat(800)}`;
    const newestNote = "[3] I retired the tracker myself once its question was answered.";
    const goal = goalNode(0, {
      description: "Track the release readiness decision",
      terminal_condition: null,
      priority: 5,
      status: "abandoned",
      progress_notes: [oldestNote, `[2] ${"m".repeat(400)}`, newestNote].join("\n"),
    });
    const result = await recallGoals({
      goals: [goal],
      sourceCap: 5,
      totalCap: 10,
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 3_000,
        label: "release readiness window",
      },
    });

    const goalEvidence = result?.evidence.find((item) => item.kind === "goal");

    expect(goalEvidence?.text).toContain(newestNote);
    expect(goalEvidence?.text).toContain("older progress_notes elided");
    expect(goalEvidence?.text).not.toContain(oldestNote);
  });

  it("does not begin a retained progress-note tail with half an astral character", async () => {
    const progressNotes = "😀".repeat(1_000);
    const goal = goalNode(0, {
      description: "Preserve a Unicode-safe progress tail",
      terminal_condition: null,
      priority: 5,
      status: "active",
      progress_notes: progressNotes,
    });
    const result = await recallGoals({
      goals: [goal],
      sourceCap: 5,
      totalCap: 10,
      temporalCue: { sinceTs: 1_000, untilTs: 3_000, label: "Unicode boundary window" },
    });
    const text = result?.evidence.find((item) => item.kind === "goal")?.text ?? "";
    const retainedTail = text.match(/retained_tail_chars=\d+\] ([\s\S]*)$/)?.[1] ?? "";

    expect(retainedTail.length).toBeGreaterThan(0);
    expect(Array.from(retainedTail)[0]).toBe("😀");
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
    expect(streamEvidence?.filter((item) => item.capMetadata !== undefined)).toEqual([
      expect.objectContaining({
        capMetadata: {
          sourceGroup: {
            candidateCount: 5,
            candidateScope: "scanned_sessions",
            renderedCount: 3,
          },
        },
      }),
    ]);
  });

  it("joins outbound tool results so failed attempts stay distinct from skipped calls", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autobiographical-outbound-"));
    tempDirs.push(tempDir);
    const sessionId = createSessionId();
    const clock = new ManualClock(2_000);
    const writer = new StreamWriter({ dataDir: tempDir, sessionId, clock });
    const failedCallId = createStreamEntryId();
    const failedCall = await writer.append({
      kind: "tool_call",
      content: {
        call_id: failedCallId,
        tool_name: OUTBOUND_POST_TOOL_NAME,
        input: { channel: "release-room" },
        origin: "autonomy",
        turn_origin: "autonomous",
      },
    });
    clock.set(2_100);
    const failedResult = await writer.append({
      kind: "tool_result",
      content: {
        call_id: failedCallId,
        ok: false,
        error: "delivery failed",
        duration_ms: 100,
      },
    });
    clock.set(3_000);
    const skippedCallId = createStreamEntryId();
    const skippedCall = await writer.append({
      kind: "tool_call",
      content: {
        call_id: skippedCallId,
        tool_name: OUTBOUND_POST_TOOL_NAME,
        input: { channel: "release-room" },
        origin: "autonomy",
        turn_origin: "autonomous",
        skipped: true,
        skip_reason: "tool_not_available_in_context",
      },
    });
    clock.set(3_100);
    const skippedResult = await writer.append({
      kind: "tool_result",
      content: {
        call_id: skippedCallId,
        ok: false,
        error: "tool_not_available_in_context",
        duration_ms: 0,
      },
    });
    writer.close();

    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      sessionsRepository: {
        list: () => [sessionRecord(sessionId, 3_100)],
      },
      createStreamReader: (readerSessionId) =>
        new StreamReader({ dataDir: tempDir, sessionId: readerSessionId }),
      sourceCap: 10,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId,
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 4_000,
        label: "outbound delivery window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });
    const outbound = result?.evidence.filter((item) => item.kind === "outbound_attempt");

    expect(outbound).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          sourceStreamEntryIds: [failedCall.id, failedResult.id],
          metadata: expect.objectContaining({
            call_id: failedCallId,
            status: "attempted",
            outcome: "failed",
            tool_result_stream_id: failedResult.id,
          }),
        }),
        expect.objectContaining({
          sourceStreamEntryIds: [skippedCall.id, skippedResult.id],
          metadata: expect.objectContaining({
            call_id: skippedCallId,
            status: "not_attempted",
            outcome: "skipped",
            skip_reason: "tool_not_available_in_context",
          }),
        }),
      ]),
    );
  });

  it("surfaces delivery fields so a transported attempt stays distinct from a failed transport", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autobiographical-delivery-"));
    tempDirs.push(tempDir);
    const sessionId = createSessionId();
    const clock = new ManualClock(2_000);
    const writer = new StreamWriter({ dataDir: tempDir, sessionId, clock });

    const appendAttempt = async (outbound: Record<string, unknown>) => {
      const callId = createStreamEntryId();
      const call = await writer.append({
        kind: "tool_call",
        content: {
          call_id: callId,
          tool_name: OUTBOUND_POST_TOOL_NAME,
          input: { channel: "release-room" },
          origin: "autonomy",
          turn_origin: "autonomous",
        },
      });
      clock.set(clock.now() + 100);
      const result = await writer.append({
        kind: "tool_result",
        content: { call_id: callId, ok: true, output: { outbound }, duration_ms: 100 },
      });
      clock.set(clock.now() + 100);
      return { callId, call, result };
    };

    const transported = await appendAttempt({
      status: "completed",
      emitted: true,
      delivery: { status: "transported", source_type: "demo" },
      delivery_outcome: { state: "delivered" },
    });
    const failed = await appendAttempt({
      status: "completed",
      emitted: true,
      delivery: { status: "transport_failed", source_type: "demo" },
      delivery_outcome: { state: "transport_failed" },
    });
    // Results written before `delivery_outcome` existed still join; absence is a schema generation.
    const legacy = await appendAttempt({ status: "target_busy", emitted: false });
    writer.close();

    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      sessionsRepository: {
        list: () => [sessionRecord(sessionId, clock.now())],
      },
      createStreamReader: (readerSessionId) =>
        new StreamReader({ dataDir: tempDir, sessionId: readerSessionId }),
      sourceCap: 10,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId,
      temporalCue: { sinceTs: 1_000, untilTs: 4_000, label: "outbound delivery window" },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });
    const outbound = result?.evidence.filter((item) => item.kind === "outbound_attempt");

    expect(outbound).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          metadata: expect.objectContaining({
            call_id: transported.callId,
            outcome: "succeeded",
            emitted: true,
            delivery_outcome_state: "delivered",
            delivery_status: "transported",
          }),
        }),
        expect.objectContaining({
          metadata: expect.objectContaining({
            call_id: failed.callId,
            outcome: "succeeded",
            emitted: true,
            delivery_outcome_state: "transport_failed",
            delivery_status: "transport_failed",
          }),
        }),
        expect.objectContaining({
          metadata: expect.objectContaining({
            call_id: legacy.callId,
            outcome: "succeeded",
            emitted: false,
          }),
        }),
      ]),
    );

    const legacyItem = outbound?.find((item) => item.metadata.call_id === legacy.callId);
    expect(legacyItem?.metadata).not.toHaveProperty("delivery_outcome_state");
    expect(legacyItem?.metadata).not.toHaveProperty("delivery_status");
  });

  it("marks stream candidate counts as lower bounds when the session fetch saturates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autobiographical-session-cap-"));
    tempDirs.push(tempDir);
    const selectedSessionId = createSessionId();
    const unscannedSessionId = createSessionId();
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: selectedSessionId,
      clock: new FixedClock(2_000),
    });
    await writer.append({
      kind: "thought",
      content: "thought from the scanned session",
    });
    writer.close();
    let requestedSessionLimit: number | null = null;
    const sessions = [
      sessionRecord(selectedSessionId, 2_500),
      sessionRecord(unscannedSessionId, 2_400),
    ];
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      sessionsRepository: {
        list: (input) => {
          requestedSessionLimit = input?.limit ?? null;
          return sessions.slice(0, input?.limit);
        },
      },
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      sessionCap: 1,
      sourceCap: 3,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId: selectedSessionId,
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 3_000,
        label: "session-bound window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    expect(requestedSessionLimit).toBe(2);
    expect(result?.evidence).toHaveLength(1);
    expect(result?.evidence[0]?.capMetadata).toEqual({
      sourceGroup: {
        candidateCountLowerBound: 1,
        candidateScope: "scanned_sessions",
        renderedCount: 1,
      },
    });
  });

  it("filters window eligibility before slicing the per-source cap", async () => {
    const postWindow = {
      ...selfDecisionEvents(1)[0]!,
      occurredAt: 9_000,
      triggerName: "post-window",
    };
    const inWindow = {
      ...selfDecisionEvents(1)[0]!,
      occurredAt: 2_000,
      triggerName: "in-window",
    };
    let requestedLimit: number | null = null;
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      selfDecisionRepository: {
        listRecentAutonomousSelfPrivate: (input) => {
          requestedLimit = input.limit;
          return [postWindow, inWindow].slice(0, input.limit);
        },
        countAutonomousSelfPrivateDecisions: () => 1,
      },
      sourceCap: 1,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId: createSessionId(),
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 3_000,
        label: "bounded historical window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    expect(requestedLimit).toBe(2);
    expect(result?.evidence).toEqual([
      expect.objectContaining({
        kind: "self_decision",
        text: expect.stringContaining("trigger_name=in-window"),
      }),
    ]);
    expect(result?.evidence[0]?.capMetadata).toBeUndefined();
  });

  it("annotates a saturated open-question fetch when sampled eligible items fit the cap", async () => {
    const fetchedQuestions = [
      openQuestion(1, 500, 1),
      openQuestion(2, 2_000, 0.9),
      openQuestion(3, 600, 0.8),
      openQuestion(4, 700, 0.7),
    ];
    let requestedLimit: number | null = null;
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      openQuestionsRepository: {
        list: (input) => {
          requestedLimit = input?.limit ?? null;
          return fetchedQuestions.slice(0, input?.limit);
        },
      },
      sourceCap: 1,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId: createSessionId(),
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 3_000,
        label: "question window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    if (result === null) {
      throw new Error("Expected autobiographical recall result");
    }

    const section = autobiographicalSection(result);

    expect(requestedLimit).toBe(4);
    expect(result.evidence).toHaveLength(1);
    expect(result.evidence[0]).toEqual(
      expect.objectContaining({
        kind: "open_question",
        capMetadata: {
          sourceGroup: {
            candidateCountLowerBound: 1,
            renderedCount: 1,
          },
        },
      }),
    );
    expect(section.entries[0]?.state_metadata).toEqual(
      expect.objectContaining({
        autobiographical_recall_cap: {
          source_group: {
            candidate_count_lower_bound: 1,
            rendered_count: 1,
          },
        },
      }),
    );
  });

  it("annotates a saturated action fetch when sampled eligible items fit the cap", async () => {
    const fetchedActions = [
      completedAction(1, 9_500, 500),
      completedAction(2, 9_000, 2_000),
      completedAction(3, 8_500, 600),
    ];
    let requestedLimit: number | null = null;
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      actionRepository: {
        list: (input) => {
          requestedLimit = input?.limit ?? null;
          return fetchedActions.slice(0, input?.limit);
        },
      },
      sourceCap: 1,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId: createSessionId(),
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 3_000,
        label: "action window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    expect(requestedLimit).toBe(3);
    expect(result?.evidence).toHaveLength(1);
    expect(result?.evidence[0]).toEqual(
      expect.objectContaining({
        kind: "action",
        capMetadata: {
          sourceGroup: {
            candidateCountLowerBound: 1,
            renderedCount: 1,
          },
        },
      }),
    );
  });

  it("sums goal and action candidates into the shared goals source group", async () => {
    // collectGoals and collectActions both file under groupId "goals", so the group's candidate
    // count is a sum across two record kinds. The lower-bound precision comes from the saturated
    // action fetch alone: the goal side is counted exactly and fetched without a limit.
    const fetchedActions = [
      completedAction(1, 2_500, 2_500),
      completedAction(2, 2_400, 2_400),
      completedAction(3, 500, 500),
    ];
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      goalsRepository: {
        list: () => [goalNode(1), goalNode(2), goalNode(3)],
      },
      actionRepository: {
        list: (input) => fetchedActions.slice(0, input?.limit),
      },
      sourceCap: 1,
      totalCap: 10,
    });

    const result = await service.recall({
      sessionId: createSessionId(),
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 3_000,
        label: "mixed window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    expect(result?.evidence.map((item) => item.kind)).toEqual(["goal", "action"]);
    expect(result?.evidence.every((item) => item.groupId === "goals")).toBe(true);
    expect(result?.evidence[0]?.capMetadata).toEqual({
      sourceGroup: {
        // 3 in-window goals (exact) + 2 in-window actions off a saturated fetch (lower bound).
        candidateCountLowerBound: 5,
        renderedCount: 2,
      },
    });
    expect(result?.evidence[1]?.capMetadata).toBeUndefined();
  });

  it("omits cap metadata when a source group fits within its cap", async () => {
    const result = await recallGoals({
      goals: [goalNode(1), goalNode(2)],
      sourceCap: 3,
      totalCap: 10,
    });
    const section = autobiographicalSection(result);

    expect(result.evidence).toHaveLength(2);
    expect(result.evidence.every((item) => item.capMetadata === undefined)).toBe(true);
    expect(section.entries.every((entry) => !hasAutobiographicalRecallCapMetadata(entry))).toBe(
      true,
    );
    expect(renderSection(section)).not.toContain("autobiographical_recall_cap");
  });

  it("renders exact shown and candidate counts when a source-group cap bites", async () => {
    const result = await recallGoals({
      goals: [goalNode(1), goalNode(2), goalNode(3), goalNode(4)],
      sourceCap: 3,
      totalCap: 10,
    });
    const section = autobiographicalSection(result);
    const annotatedEvidence = result.evidence.filter((item) => item.capMetadata !== undefined);
    const annotatedEntries = section.entries.filter(hasAutobiographicalRecallCapMetadata);

    expect(result.evidence).toHaveLength(3);
    expect(annotatedEvidence).toEqual([
      expect.objectContaining({
        groupId: "goals",
        capMetadata: {
          sourceGroup: {
            candidateCount: 4,
            renderedCount: 3,
          },
        },
      }),
    ]);
    expect(annotatedEntries).toHaveLength(1);
    expect(annotatedEntries[0]?.state_metadata).toEqual(
      expect.objectContaining({
        group_id: "goals",
        group_label: "Goals touched",
        autobiographical_recall_cap: {
          source_group: {
            candidate_count: 4,
            rendered_count: 3,
          },
        },
      }),
    );
  });

  it("uses an exact repository count for a capped self-decision group", async () => {
    const fetchedEvents = selfDecisionEvents(4);
    let requestedLimit: number | null = null;
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      selfDecisionRepository: {
        listRecentAutonomousSelfPrivate: (input) => {
          requestedLimit = input.limit;
          return fetchedEvents.slice(0, input.limit);
        },
        countAutonomousSelfPrivateDecisions: () => 9,
      },
      sourceCap: 3,
      totalCap: 10,
    });
    const result = await service.recall({
      sessionId: createSessionId(),
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 9_000,
        label: "decision window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    expect(requestedLimit).toBe(4);
    expect(result?.evidence).toHaveLength(3);
    expect(result?.evidence[0]?.capMetadata).toEqual({
      sourceGroup: {
        candidateCount: 9,
        renderedCount: 3,
      },
    });
  });

  it("labels a bounded-fetch candidate count as a lower bound", async () => {
    const fetchedEvents = selfDecisionEvents(4);
    const service = new AutobiographicalRecallService({
      clock: new FixedClock(NOW_MS),
      selfDecisionRepository: {
        listRecentAutonomousSelfPrivate: (input) => fetchedEvents.slice(0, input.limit),
      },
      sourceCap: 3,
      totalCap: 10,
    });
    const result = await service.recall({
      sessionId: createSessionId(),
      temporalCue: {
        sinceTs: 1_000,
        untilTs: 9_000,
        label: "decision window",
      },
      isSelfAudience: false,
      sessionAudienceRole: "operator",
      perceptionMode: "reflective",
    });

    if (result === null) {
      throw new Error("Expected autobiographical recall result");
    }

    const section = autobiographicalSection(result);

    expect(result.evidence[0]?.capMetadata).toEqual({
      sourceGroup: {
        candidateCountLowerBound: 4,
        renderedCount: 3,
      },
    });
    expect(section.entries[0]?.state_metadata).toEqual(
      expect.objectContaining({
        autobiographical_recall_cap: {
          source_group: {
            candidate_count_lower_bound: 4,
            rendered_count: 3,
          },
        },
      }),
    );
  });

  it("renders exact candidate and survivor counts when the total cap bites", async () => {
    const result = await recallGoals({
      goals: [goalNode(1), goalNode(2), goalNode(3), goalNode(4)],
      sourceCap: 10,
      totalCap: 2,
    });
    const section = autobiographicalSection(result);
    const annotatedEntries = section.entries.filter(hasAutobiographicalRecallCapMetadata);

    expect(result.evidence).toHaveLength(2);
    expect(result.evidence[0]?.capMetadata).toEqual({
      total: {
        candidateCount: 4,
        candidateScope: "post_source_caps",
        renderedCount: 2,
      },
    });
    expect(result.evidence[1]?.capMetadata).toBeUndefined();
    expect(annotatedEntries).toHaveLength(1);
    expect(annotatedEntries[0]?.state_metadata).toEqual(
      expect.objectContaining({
        autobiographical_recall_cap: {
          total: {
            candidate_count: 4,
            candidate_scope: "post_source_caps",
            rendered_count: 2,
          },
        },
      }),
    );
  });

  it("keeps cap counts in structural metadata rather than evidence-item prose", async () => {
    const result = await recallGoals({
      goals: [goalNode(1), goalNode(2), goalNode(3)],
      sourceCap: 2,
      totalCap: 10,
    });
    const rendered = renderSection(autobiographicalSection(result));

    for (const item of result.evidence) {
      expect(item.text).not.toContain("autobiographical_recall_cap");
      expect(item.text).not.toContain("candidate_count");
      expect(item.text).not.toContain("rendered_count");
    }

    expect(rendered).toContain('"autobiographical_recall_cap"');
    expect(rendered).toContain('"candidate_count":3');
    expect(rendered).toContain('"rendered_count":2');
  });

  it("reassigns and recomputes cap metadata through build, provenance dedupe, compaction, and render", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-autobiographical-cap-render-"));
    tempDirs.push(tempDir);
    const sessionId = createSessionId();
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId,
      clock: new FixedClock(2_000),
    });
    const currentUserEntry = await writer.append({
      kind: "user_msg",
      content: "Inspect the recalled horizon.",
    });
    writer.close();
    const disclosureLabel = selfPrivateMemoryDisclosureLabel();
    const survivingIds = [createStreamEntryId(), createStreamEntryId()];
    const recall: AutobiographicalRecallResult = {
      window: {
        startMs: 1_000,
        endMs: 3_000,
        label: "rendered horizon",
        source: "perception_temporal_cue",
      },
      evidence: [
        {
          id: `stream:${currentUserEntry.id}`,
          kind: "stream_reflection",
          groupId: "stream_reflection",
          groupLabel: "Reflection and silence markers",
          occurredAt: 2_000,
          relativeAge: "now",
          score: 0.9,
          text: "Deduped cap carrier",
          disclosureLabel,
          sourceStreamEntryIds: [currentUserEntry.id],
          sourceEpisodeIds: [],
          metadata: {},
          capMetadata: {
            sourceGroup: {
              candidateCount: 5,
              renderedCount: 3,
              candidateScope: "scanned_sessions",
            },
            total: {
              candidateCount: 6,
              renderedCount: 3,
              candidateScope: "post_source_caps",
            },
          },
        },
        ...survivingIds.map((id, index) => ({
          id: `stream:${id}`,
          kind: "stream_reflection" as const,
          groupId: "stream_reflection",
          groupLabel: "Reflection and silence markers",
          occurredAt: 1_900 - index,
          relativeAge: "moments ago",
          score: 0.8 - index * 0.1,
          text: `Surviving reflection ${index + 1}`,
          disclosureLabel,
          sourceStreamEntryIds: [] as const,
          sourceEpisodeIds: [] as const,
          metadata: {},
        })),
      ],
    };
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (readerSessionId) =>
        new StreamReader({ dataDir: tempDir, sessionId: readerSessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });
    const ledger = await builder.build({
      sessionId,
      audienceEntityId: null,
      currentUserMessage: String(currentUserEntry.content),
      currentUserEntry,
      workingMemory: createWorkingMemory(sessionId, 2_000),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      openQuestions: [],
      pendingCorrections: [],
      autobiographicalRecall: recall,
    });
    const compacted = compactEvidenceLedger(ledger, {
      sectionOptions: {
        autobiographical_recall: {
          maxEntries: 1,
          maxTokens: 5_000,
        },
      },
    });
    const section = compacted.ledger.sections.find((item) => item.id === "autobiographical_recall");
    const survivingRecallEntries =
      section?.entries.filter((entry) => typeof entry.state_metadata?.group_id === "string") ?? [];
    const rendered = renderEvidenceLedger(compacted.ledger) ?? "";

    expect(survivingRecallEntries).toHaveLength(1);
    expect(survivingRecallEntries[0]?.text).toBe("Surviving reflection 1");
    expect(survivingRecallEntries[0]?.state_metadata?.autobiographical_recall_cap).toEqual({
      source_group: {
        candidate_count: 5,
        candidate_scope: "scanned_sessions",
        rendered_count: 1,
      },
      total: {
        candidate_count: 6,
        candidate_scope: "post_source_caps",
        rendered_count: 1,
      },
    });
    expect(rendered).toContain("Surviving reflection 1");
    expect(rendered).not.toContain("Deduped cap carrier");
    expect(rendered).toContain('"rendered_count":1');
    expect(rendered).not.toContain('"rendered_count":3');
  });
});
