import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  ACTION_CANDIDATE_CLASSIFICATIONS,
  ACTION_STATES,
  RELATIONAL_SLOT_STATES,
  REVIEW_KINDS,
  ManualClock,
  createSessionId,
  type ActionRecord,
  type Borg,
  type SessionId,
} from "../src/index.js";
import type { EmbeddingClient } from "../src/embeddings/index.js";
import { ActionRepository } from "../src/memory/actions/index.js";
import { actionMigrations } from "../src/memory/actions/migrations.js";
import { CommitmentRepository, commitmentMigrations } from "../src/memory/commitments/index.js";
import {
  IdentityEventRepository,
  IdentityService,
  identityMigrations,
} from "../src/memory/identity/index.js";
import {
  RelationalSlotRepository,
  relationalSlotMigrations,
} from "../src/memory/relational-slots/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  GrowthMarkersRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
  selfMigrations,
} from "../src/memory/self/index.js";
import { ReviewQueueRepository, semanticMigrations } from "../src/memory/semantic/index.js";
import { WorkingMemoryStore } from "../src/memory/working/index.js";
import { composeMigrations, openDatabase } from "../src/storage/sqlite/index.js";
import { ABORTED_TURN_EVENT, type StreamEntry } from "../src/stream/index.js";
import {
  createActionId,
  createDecisionArtifactEntryId,
  createEntityId,
  createStreamEntryId,
} from "../src/util/ids.js";

import { MetricsCapture } from "./metrics.js";
import type { MetricsRow } from "./types.js";

const tempDirs: string[] = [];
const OPEN_QUESTION_OPEN_STATUS = "open";
const OPEN_QUESTION_RESOLVED_STATUS = "resolved";
const TURN_METRICS_KEY_ORDER = [
  "event",
  "ts",
  "turn_counter",
  "turnId",
  "transport_chat_attempts",
  "episode_count",
  "semantic_node_count",
  "semantic_node_count_by_status",
  "semantic_edge_count",
  "semantic_nodes_added_since_last_check",
  "semantic_edges_added_since_last_check",
  "open_question_count",
  "active_goal_count",
  "generation_suppression_count",
  "mood_valence",
  "mood_arousal",
  "retrieval_latency_ms",
  "deliberation_latency_ms",
  "borg_input_tokens",
  "borg_output_tokens",
  "open_question_resolved_count",
  "action_record_count_total",
  "action_record_count_by_state",
  "action_record_count_committed_to_do",
  "action_record_count_canonicalized",
  "action_record_count_active",
  "action_record_creation_source_per_turn",
  "action_record_creation_count_this_turn",
  "action_candidate_classifications_per_turn",
  "action_candidate_rejected_classification",
  "action_persistence_dedup_skipped_embedding",
  "action_persistence_dedup_degraded",
  "recent_completed_action_count",
  "commitment_count_active",
  "commitment_count_superseded",
  "commitment_count_revoked",
  "commitment_count_expired",
  "commitment_count_canonicalized",
  "pending_action_count",
  "pending_action_merge_count",
  "relational_slot_count_by_state",
  "review_queue_open_count_by_type",
  "review_resolver_attempted",
  "review_resolver_accepted",
  "review_resolver_dismissed",
  "review_resolver_rejected",
  "review_resolver_needs_manual",
  "review_queue_enqueued_this_turn",
  "review_queue_resolved_this_turn",
  "review_queue_drain_rate",
  "frame_anomaly_classifier_calls",
  "frame_anomaly_classified_normal_count",
  "frame_anomaly_actual_anomaly_count",
  "frame_anomaly_degraded_count",
  "frame_anomaly_degraded_fallback_match_count",
  "quarantined_user_entry_count",
  "early_extractors_skipped_frame_anomaly_count",
  "goal_promotion_salvaged_promotions",
  "goal_promotion_skipped_promotions",
  "goal_promotion_initial_step_downgraded",
  "goal_promotion_dedup_skipped_extractor_signal",
  "goal_promotion_dedup_skipped_embedding",
  "goal_promotion_dedup_degraded",
  "goal_promotion_classifications_per_turn",
  "goal_promotion_rejected_classification",
  "goal_promotion_cap_rejections",
  "decision_artifact_semantic_revisions_attempted",
  "decision_artifact_semantic_revisions_completed_succeeded",
  "decision_artifact_semantic_nodes_marked_superseded",
  "decision_artifact_semantic_nodes_marked_contradicted",
  "overseer_due_on_suppressed_turn",
] as const;

class SameVectorEmbeddingClient implements EmbeddingClient {
  async embed(): Promise<Float32Array> {
    return Float32Array.from([1, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map(() => Float32Array.from([1, 0]));
  }
}

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-simulator-metrics-"));
  tempDirs.push(dir);
  return dir;
}

function zeroCounts<K extends string>(keys: readonly K[]): Record<K, number> {
  return Object.fromEntries(keys.map((key) => [key, 0])) as Record<K, number>;
}

function makeAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createActionId(),
    description: overrides.description ?? "Review metrics fixture",
    actor: overrides.actor ?? "borg",
    audience_entity_id: overrides.audience_entity_id ?? null,
    goal_id: overrides.goal_id ?? null,
    open_question_id: overrides.open_question_id ?? null,
    state: overrides.state ?? "committed_to_do",
    confidence: overrides.confidence ?? 0.8,
    provenance_episode_ids: overrides.provenance_episode_ids ?? [],
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [createStreamEntryId()],
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
    considering_at: overrides.considering_at ?? null,
    committed_at: overrides.committed_at ?? null,
    scheduled_at: overrides.scheduled_at ?? null,
    completed_at: overrides.completed_at ?? null,
    not_done_at: overrides.not_done_at ?? null,
    unknown_at: overrides.unknown_at ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
  };
}

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function fakeBorg(
  counts: {
    semanticNodes?: number;
    semanticEdges?: number;
    activeGoals?: number;
    suppressedSessions?: readonly SessionId[];
    streamEntriesBySession?: ReadonlyMap<SessionId, readonly StreamEntry[]>;
  } = {},
  observed: { moodSessions?: SessionId[]; tailSessions?: SessionId[] } = {},
): Borg {
  const semanticNodeCount = counts.semanticNodes ?? 1;
  const semanticEdgeCount = counts.semanticEdges ?? 2;
  const activeGoalCount = counts.activeGoals ?? 2;
  const suppressedSessions = new Set(counts.suppressedSessions ?? []);
  const streamEntriesBySession = counts.streamEntriesBySession ?? new Map();

  return {
    mood: {
      current: (sessionId: SessionId) => {
        observed.moodSessions?.push(sessionId);
        return { valence: -0.2, arousal: 0.4 };
      },
    },
    episodic: {
      list: async () => ({ items: [{ id: "episode_1" }, { id: "episode_2" }] }),
    },
    semantic: {
      nodes: {
        list: async () =>
          Array.from({ length: semanticNodeCount }, (_, index) => ({
            id: `node_${index}`,
            status: "active",
          })),
      },
      edges: {
        list: () =>
          Array.from({ length: semanticEdgeCount }, (_, index) => ({ id: `edge_${index}` })),
      },
    },
    actions: {
      count: () => 0,
      countByState: () => zeroCounts(ACTION_STATES),
      countCanonicalized: () => 0,
      countActive: () => 0,
      getCreationCountsBySource: () => ({
        extractor: 0,
        reflector: 0,
        api: 0,
        unknown: 0,
      }),
      countCompletedSince: () => 0,
      latestCompletedAt: () => null,
      listCompletedIds: () => [],
    },
    self: {
      openQuestions: {
        list: () => [{ id: "question_1" }],
      },
      goals: {
        list: () =>
          Array.from({ length: activeGoalCount }, (_, index) => ({
            id: `goal_${index}`,
            children: [],
          })),
      },
    },
    commitments: {
      list: () => [],
      countActive: () => 0,
      countSuperseded: () => 0,
      countRevoked: () => 0,
      countExpired: () => 0,
      countCanonicalized: () => 0,
    },
    relationalSlots: {
      countByState: () => zeroCounts(RELATIONAL_SLOT_STATES),
    },
    review: {
      list: () => [],
    },
    identity: {
      listEvents: () => [],
    },
    workmem: {
      load: () => ({ pending_actions: [] }),
      getPendingActionMergeCount: () => 0,
    },
    stream: {
      tail: (_limit: number, options?: { session?: SessionId }) => {
        if (options?.session !== undefined) {
          observed.tailSessions?.push(options.session);
        }

        if (options?.session !== undefined && streamEntriesBySession.has(options.session)) {
          return [...(streamEntriesBySession.get(options.session) ?? [])];
        }

        return options?.session !== undefined && suppressedSessions.has(options.session)
          ? [{ kind: "agent_suppressed" }]
          : [];
      },
    },
  } as unknown as Borg;
}

function createIdentityHarness(db: ReturnType<typeof openDatabase>, clock: ManualClock) {
  const identityEvents = new IdentityEventRepository({ db, clock });
  const valuesRepository = new ValuesRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const goalsRepository = new GoalsRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const traitsRepository = new TraitsRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const autobiographicalRepository = new AutobiographicalRepository({ db, clock });
  const growthMarkersRepository = new GrowthMarkersRepository({ db, clock });
  const openQuestionsRepository = new OpenQuestionsRepository({ db, clock });
  const commitmentRepository = new CommitmentRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const identity = new IdentityService({
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    growthMarkersRepository,
    openQuestionsRepository,
    commitmentRepository,
    identityEventRepository: identityEvents,
  });

  return {
    identity,
    openQuestionsRepository,
  };
}

describe("MetricsCapture", () => {
  it("captures Borg state, trace latency, and token usage to JSONL", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");

    writeFileSync(
      tracePath,
      [
        { ts: 100, turnId: "turn-1", event: "retrieval_started" },
        { ts: 125, turnId: "turn-1", event: "retrieval_completed" },
        { ts: 130, turnId: "turn-1", event: "llm_call_started" },
        {
          ts: 190,
          turnId: "turn-1",
          event: "llm_call_response",
          usage: { inputTokens: 11, outputTokens: 7 },
        },
        {
          ts: 191,
          turnId: "turn-1",
          event: "decision_artifact_semantic_revision_completed",
          artifact_entry_id: "dart_metrics_completed",
          superseded_count: 2,
          contradicted_count: 1,
        },
        {
          ts: 192,
          turnId: "turn-1",
          event: "decision_artifact_semantic_revision_degraded",
          artifact_entry_id: "dart_metrics_degraded",
          reason: "judge unavailable",
        },
        {
          ts: 193,
          turnId: "turn-1",
          event: "decision_artifact_semantic_revision_degraded",
          artifact_entry_id: "dart_metrics_completed",
          reason: "mark failed after partial apply",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const sessionId = createSessionId();
    const otherSessionId = createSessionId();
    const observed: { moodSessions: SessionId[]; tailSessions: SessionId[] } = {
      moodSessions: [],
      tailSessions: [],
    };
    const capture = new MetricsCapture(metricsPath, { tracePath });
    const row = await capture.capture(
      fakeBorg({ suppressedSessions: [otherSessionId] }, observed),
      "turn-1",
      3,
      {
        sessionId,
        sessionIds: [sessionId, otherSessionId],
        transportChatAttempts: 2,
      },
    );
    const written = JSON.parse(readFileSync(metricsPath, "utf8").trim()) as MetricsRow;

    expect(row.turn_counter).toBe(3);
    expect(row.event).toBe("turn_metrics");
    expect(row.transport_chat_attempts).toBe(2);
    expect(row.episode_count).toBe(2);
    expect(row.semantic_node_count).toBe(1);
    expect(row.semantic_node_count_by_status).toEqual({
      active: 1,
      superseded: 0,
      contradicted: 0,
      quarantined: 0,
    });
    expect(row.semantic_edge_count).toBe(2);
    expect(row.semantic_nodes_added_since_last_check).toBe(0);
    expect(row.semantic_edges_added_since_last_check).toBe(0);
    expect(row.open_question_count).toBe(1);
    expect(row.active_goal_count).toBe(2);
    expect(row.generation_suppression_count).toBe(1);
    expect(row.retrieval_latency_ms).toBe(25);
    expect(row.deliberation_latency_ms).toBe(60);
    expect(row.borg_input_tokens).toBe(11);
    expect(row.borg_output_tokens).toBe(7);
    expect(row.decision_artifact_semantic_revisions_attempted).toBe(2);
    expect(row.decision_artifact_semantic_revisions_completed_succeeded).toBe(1);
    expect(row.decision_artifact_semantic_nodes_marked_superseded).toBe(2);
    expect(row.decision_artifact_semantic_nodes_marked_contradicted).toBe(1);
    expect(observed.moodSessions).toEqual([sessionId]);
    expect(observed.tailSessions).toEqual([sessionId, otherSessionId, sessionId, otherSessionId]);
    expect(written).toEqual(row);
  });

  it("writes turn metric keys in v21 order with new fields appended", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    await new MetricsCapture(metricsPath).capture(fakeBorg(), "turn-ordered", 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    const written = JSON.parse(readFileSync(metricsPath, "utf8").trim()) as MetricsRow;

    expect(Object.keys(written)).toEqual([...TURN_METRICS_KEY_ORDER]);
  });

  it("counts action candidate classification and embedding-dedup traces", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-action",
          event: "action_state_extractor_completed",
          classification_counts: {
            concrete_action: 2,
            conversational_acknowledgment: 1,
            decision_or_preference: 0,
            already_represented: 0,
            none: 0,
            invalid_classification: 1,
          },
        },
        {
          ts: 101,
          turnId: "turn-action",
          event: "action_candidate_classification_rejected",
          classification: "conversational_acknowledgment",
          reason: "non_concrete_classification",
        },
        {
          ts: 102,
          turnId: "turn-action",
          event: "action_candidate_classification_rejected",
          classification: "concrete_action",
          reason: "embedding_dedup",
        },
        {
          ts: 103,
          turnId: "turn-action",
          event: "action_persistence_dedup_skipped_embedding",
          reason: "embedding_dedup",
        },
        {
          ts: 104,
          turnId: "turn-action",
          event: "action_persistence_dedup_degraded",
          reason: "candidate_embedding_failed",
        },
        {
          ts: 200,
          turnId: "other-turn",
          event: "action_persistence_dedup_skipped_embedding",
          reason: "embedding_dedup",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-action",
      1,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row).toMatchObject({
      action_candidate_classifications_per_turn: {
        ...zeroCounts(ACTION_CANDIDATE_CLASSIFICATIONS),
        concrete_action: 2,
        conversational_acknowledgment: 1,
        invalid_classification: 1,
      },
      action_candidate_rejected_classification: 1,
      action_persistence_dedup_skipped_embedding: 1,
      action_persistence_dedup_degraded: 1,
    });
  });

  it("counts goal-promotion salvage and initial-step downgrade traces", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: "turn-goal",
          event: "goal_promotion_extractor_completed",
          salvaged_promotion_count: 2,
          skipped_promotion_count: 1,
          classification_counts: {
            durable_borg_goal: 2,
            one_off: 1,
            not_borg_responsibility: 0,
            already_represented: 0,
            none: 0,
            invalid_classification: 1,
          },
        },
        {
          ts: 100.5,
          turnId: "turn-goal",
          event: "goal_promotion_classification_rejected",
          classification: "one_off",
          reason: "non_durable_classification",
        },
        {
          ts: 100.6,
          turnId: "turn-goal",
          event: "goal_promotion_classification_rejected",
          classification: "durable_borg_goal",
          reason: "cap_exceeded",
        },
        {
          ts: 101,
          turnId: "turn-goal",
          event: "goal_promotion_initial_step_downgraded",
          reason: "wait_without_due_at",
        },
        {
          ts: 102,
          turnId: "turn-goal",
          event: "goal_promotion_skipped_as_duplicate",
          reason: "extractor_signal",
        },
        {
          ts: 103,
          turnId: "turn-goal",
          event: "goal_promotion_skipped_as_duplicate",
          reason: "embedding",
        },
        {
          ts: 104,
          turnId: "turn-goal",
          event: "goal_promotion_dedup_degraded",
          reason: "candidate_embedding_failed",
        },
        {
          ts: 200,
          turnId: "other-turn",
          event: "goal_promotion_extractor_completed",
          salvaged_promotion_count: 1,
          skipped_promotion_count: 1,
          classification_counts: {
            durable_borg_goal: 1,
            one_off: 0,
            not_borg_responsibility: 0,
            already_represented: 0,
            none: 0,
            invalid_classification: 0,
          },
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg(),
      "turn-goal",
      1,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row).toMatchObject({
      goal_promotion_salvaged_promotions: 2,
      goal_promotion_skipped_promotions: 1,
      goal_promotion_initial_step_downgraded: 1,
      goal_promotion_dedup_skipped_extractor_signal: 1,
      goal_promotion_dedup_skipped_embedding: 1,
      goal_promotion_dedup_degraded: 1,
      goal_promotion_classifications_per_turn: {
        durable_borg_goal: 2,
        one_off: 1,
        not_borg_responsibility: 0,
        already_represented: 0,
        none: 0,
        invalid_classification: 1,
      },
      goal_promotion_rejected_classification: 1,
      goal_promotion_cap_rejections: 1,
    });
  });

  it("emits simulator health warning traces when active goals are high", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();

    const row = await new MetricsCapture(metricsPath, { tracePath }).capture(
      fakeBorg({ activeGoals: 26 }),
      "turn-health-high",
      5,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );
    const trace = readFileSync(tracePath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(row.active_goal_count).toBe(26);
    expect(trace).toContainEqual(
      expect.objectContaining({
        event: "simulator_health_warning",
        warning_kind: "active_goals_high",
        turn_counter: 5,
        threshold: 25,
        observed_value: 26,
      }),
    );
  });

  it("emits active-goals-high warnings only on rising edges", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const capture = new MetricsCapture(metricsPath, { tracePath });

    await capture.capture(fakeBorg({ activeGoals: 26 }), "turn-warning-rise-1", 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    await capture.capture(fakeBorg({ activeGoals: 27 }), "turn-warning-still-high", 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    await capture.capture(fakeBorg({ activeGoals: 24 }), "turn-warning-cleared", 3, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    await capture.capture(fakeBorg({ activeGoals: 28 }), "turn-warning-rise-2", 4, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    const warningEvents = readFileSync(tracePath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>)
      .filter(
        (record) =>
          record.event === "simulator_health_warning" &&
          record.warning_kind === "active_goals_high",
      );

    expect(warningEvents).toEqual([
      expect.objectContaining({
        turnId: "turn-warning-rise-1",
        observed_value: 26,
      }),
      expect.objectContaining({
        turnId: "turn-warning-rise-2",
        observed_value: 28,
      }),
    ]);
    expect(capture.listHealthWarnings().map((warning) => warning.turnId)).toEqual([
      "turn-warning-rise-1",
      "turn-warning-rise-2",
    ]);
  });

  it("emits simulator health warning traces when active goal growth is high after turn twenty", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const capture = new MetricsCapture(metricsPath, { tracePath });

    for (let turn = 21; turn <= 30; turn += 1) {
      await capture.capture(fakeBorg({ activeGoals: turn - 20 }), `turn-growth-${turn}`, turn, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });
    }

    const trace = readFileSync(tracePath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(trace).toContainEqual(
      expect.objectContaining({
        event: "simulator_health_warning",
        warning_kind: "active_goals_growth_high",
        turn_counter: 30,
        threshold: 0.5,
        observed_value: 1,
        window_start_turn: 21,
        window_turns: 9,
      }),
    );
  });

  it("counts frame-anomaly classifier, fallback, and durable quarantine markers", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const anomalyTurnId = "turn-frame-anomaly";
    const degradedTurnId = "turn-frame-degraded";
    const normalTurnId = "turn-frame-normal";
    const quarantinedUserEntryId = createStreamEntryId();
    const streamEntriesBySession = new Map<SessionId, StreamEntry[]>([
      [
        sessionId,
        [
          {
            id: createStreamEntryId(),
            timestamp: 1,
            kind: "internal_event",
            content: {
              event: "quarantined_user_entry",
              turn_id: anomalyTurnId,
              source_stream_entry_id: quarantinedUserEntryId,
              kind: "frame_assignment_claim",
            },
            turn_id: anomalyTurnId,
            session_id: sessionId,
            compressed: false,
            sender_entity_id: null,
            reply_target_entity_id: null,
          },
        ],
      ],
    ]);

    writeFileSync(
      tracePath,
      [
        {
          ts: 100,
          turnId: anomalyTurnId,
          event: "llm_call_started",
          label: "frame_anomaly_classifier",
        },
        {
          ts: 101,
          turnId: anomalyTurnId,
          event: "frame_anomaly_classified",
          status: "ok",
          kind: "frame_assignment_claim",
        },
        {
          ts: 200,
          turnId: degradedTurnId,
          event: "llm_call_started",
          label: "frame_anomaly_classifier",
        },
        {
          ts: 201,
          turnId: degradedTurnId,
          event: "frame_anomaly_classified",
          status: "degraded",
          reason: "invalid_payload",
        },
        {
          ts: 202,
          turnId: degradedTurnId,
          event: "frame_anomaly_degraded_fallback_match",
          pattern: "i'm claude",
          kind: "assistant_self_claim_in_user_role",
        },
        {
          ts: 300,
          turnId: normalTurnId,
          event: "llm_call_started",
          label: "frame_anomaly_classifier",
        },
        {
          ts: 301,
          turnId: normalTurnId,
          event: "frame_anomaly_classified",
          status: "ok",
          kind: "normal",
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const capture = new MetricsCapture(metricsPath, { tracePath });
    const borg = fakeBorg({ streamEntriesBySession });
    const anomaly = await capture.capture(borg, anomalyTurnId, 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    const degraded = await capture.capture(borg, degradedTurnId, 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    const normal = await capture.capture(borg, normalTurnId, 3, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    expect(anomaly).toMatchObject({
      frame_anomaly_classifier_calls: 1,
      frame_anomaly_classified_normal_count: 0,
      frame_anomaly_actual_anomaly_count: 1,
      frame_anomaly_degraded_count: 0,
      frame_anomaly_degraded_fallback_match_count: 0,
      quarantined_user_entry_count: 1,
      early_extractors_skipped_frame_anomaly_count: 1,
    });
    expect(degraded).toMatchObject({
      frame_anomaly_classifier_calls: 1,
      frame_anomaly_classified_normal_count: 0,
      frame_anomaly_actual_anomaly_count: 0,
      frame_anomaly_degraded_count: 1,
      frame_anomaly_degraded_fallback_match_count: 1,
      quarantined_user_entry_count: 0,
      early_extractors_skipped_frame_anomaly_count: 1,
    });
    expect(normal).toMatchObject({
      frame_anomaly_classifier_calls: 1,
      frame_anomaly_classified_normal_count: 1,
      frame_anomaly_actual_anomaly_count: 0,
      frame_anomaly_degraded_count: 0,
      frame_anomaly_degraded_fallback_match_count: 0,
      quarantined_user_entry_count: 0,
      early_extractors_skipped_frame_anomaly_count: 0,
    });
  });

  it("records semantic graph growth since the previous capture", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();

    await capture.capture(fakeBorg({ semanticNodes: 1, semanticEdges: 2 }), "turn-1", 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    const row = await capture.capture(
      fakeBorg({ semanticNodes: 4, semanticEdges: 5 }),
      "turn-2",
      2,
      {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      },
    );

    expect(row.semantic_nodes_added_since_last_check).toBe(3);
    expect(row.semantic_edges_added_since_last_check).toBe(3);
  });

  it("counts backdated completed actions as newly completed by id", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const db = openDatabase(join(dir, "borg.db"), {
      migrations: actionMigrations,
    });
    const clock = new ManualClock(1_000);
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();
    const actions = new ActionRepository({ db, clock });
    const borg = {
      ...fakeBorg(),
      actions,
    } as unknown as Borg;

    try {
      actions.add(
        makeAction({
          description: "First completed action",
          state: "completed",
          created_at: 100,
          updated_at: 100,
          completed_at: 100,
        }),
      );

      const firstRow = await capture.capture(borg, "turn-complete-first", 1, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });

      actions.add(
        makeAction({
          description: "Backdated completed action",
          state: "completed",
          created_at: 99,
          updated_at: 101,
          completed_at: 99,
        }),
      );

      const secondRow = await capture.capture(borg, "turn-complete-second", 2, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });

      expect(firstRow.recent_completed_action_count).toBe(1);
      expect(secondRow.recent_completed_action_count).toBe(1);
    } finally {
      db.close();
    }
  });

  it("counts open questions resolved through the identity update path", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const db = openDatabase(join(dir, "borg.db"), {
      migrations: composeMigrations(selfMigrations, commitmentMigrations, identityMigrations),
    });
    const clock = new ManualClock(1_000);
    const sessionId = createSessionId();
    const { identity, openQuestionsRepository } = createIdentityHarness(db, clock);
    const provenance = { kind: "manual" } as const;

    try {
      const question = identity.addOpenQuestion({
        question: "Which metrics path resolves this?",
        urgency: 0.7,
        related_episode_ids: [],
        related_semantic_node_ids: [],
        provenance,
        source: "user",
      });
      const result = identity.updateOpenQuestion(
        question.id,
        {
          status: OPEN_QUESTION_RESOLVED_STATUS,
          resolution_evidence_stream_entry_ids: [createStreamEntryId()],
          resolution_note: "The metrics update path resolved it.",
          resolved_at: clock.now(),
        },
        provenance,
        {
          throughReview: true,
        },
      );
      const borg = {
        ...fakeBorg(),
        identity: {
          listEvents: (...args: Parameters<IdentityService["listEvents"]>) =>
            identity.listEvents(...args),
        },
        self: {
          openQuestions: {
            list: (...args: Parameters<OpenQuestionsRepository["list"]>) =>
              openQuestionsRepository.list(...args),
          },
          goals: {
            list: () => [],
          },
        },
      } as unknown as Borg;

      const row = await new MetricsCapture(metricsPath).capture(borg, "turn-oq-update", 1, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });

      expect(result.status).toBe("applied");
      expect(row.open_question_resolved_count).toBe(1);
    } finally {
      db.close();
    }
  });

  it("captures simulator metrics for action, commitment, working-memory, relational slot, review, and open-question bands", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const db = openDatabase(join(dir, "borg.db"), {
      migrations: composeMigrations(
        actionMigrations,
        commitmentMigrations,
        relationalSlotMigrations,
        semanticMigrations,
        selfMigrations,
        identityMigrations,
      ),
    });
    const clock = new ManualClock(1_000);
    const sessionId = createSessionId();

    try {
      const actions = new ActionRepository({ db, clock });
      actions.add(
        makeAction({
          state: "considering",
          considering_at: 1_000,
        }),
      );
      actions.add(
        makeAction({
          description: "Send the metrics report",
          state: "completed",
          created_at: 1_100,
          updated_at: 1_100,
          completed_at: 1_100,
        }),
      );
      actions.add(
        makeAction({
          description: "Close the sprint notes",
          state: "completed",
          created_at: 1_200,
          updated_at: 1_200,
          completed_at: 1_200,
          canonicalized_by_artifact_entry_id: createDecisionArtifactEntryId(),
        }),
      );

      const commitments = new CommitmentRepository({ db, clock });
      const activeCommitment = commitments.add({
        type: "promise",
        directiveFamily: "metrics active one",
        directive: "Keep the metrics visible.",
        priority: 5,
        provenance: { kind: "manual" },
      });
      commitments.add({
        type: "rule",
        directiveFamily: "metrics active two",
        directive: "Prefer count-only reads.",
        priority: 4,
        provenance: { kind: "manual" },
      });
      const supersededCommitment = commitments.add({
        type: "preference",
        directiveFamily: "metrics superseded",
        directive: "Use the older metrics wording.",
        priority: 3,
        provenance: { kind: "manual" },
      });
      const revokedCommitment = commitments.add({
        type: "promise",
        directiveFamily: "metrics revoked",
        directive: "Retire the old metric commitment.",
        priority: 3,
        provenance: { kind: "manual" },
      });
      commitments.add({
        type: "promise",
        directiveFamily: "metrics expired",
        directive: "Expire the old metric commitment.",
        priority: 3,
        provenance: { kind: "manual" },
        createdAt: 500,
        expiresAt: 900,
      });
      const canonicalizedCommitment = commitments.add({
        type: "promise",
        directiveFamily: "metrics canonicalized",
        directive: "Canonicalize the old metric commitment.",
        priority: 3,
        provenance: { kind: "manual" },
      });
      commitments.supersede(supersededCommitment.id, activeCommitment.id);
      commitments.revoke(revokedCommitment.id, "metrics test revocation", { kind: "manual" });
      commitments.revoke(
        canonicalizedCommitment.id,
        "metrics test canonicalization",
        { kind: "manual" },
        undefined,
        {
          canonicalizedByArtifactEntryId: createDecisionArtifactEntryId(),
        },
      );

      const workingMemoryStore = new WorkingMemoryStore({ dataDir: dir, clock });
      const embeddingClient = new SameVectorEmbeddingClient();
      await workingMemoryStore.addPendingAction({
        sessionId,
        action: {
          description: "Follow up on metric output",
          next_action: "inspect the metrics JSONL row",
        },
        embeddingClient,
      });
      await workingMemoryStore.addPendingAction({
        sessionId,
        action: {
          description: "Check the simulator metrics artifact",
          next_action: "review the metrics JSONL output",
        },
        embeddingClient,
      });

      const relationalSlots = new RelationalSlotRepository({ db, clock });
      relationalSlots.applyAssertion({
        subject_entity_id: createEntityId(),
        slot_key: "partner.name",
        asserted_value: "Ari",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      const contestedSubject = createEntityId();
      relationalSlots.applyAssertion({
        subject_entity_id: contestedSubject,
        slot_key: "partner.name",
        asserted_value: "Bo",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      relationalSlots.applyAssertion({
        subject_entity_id: contestedSubject,
        slot_key: "partner.name",
        asserted_value: "Cam",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      const quarantinedSubject = createEntityId();
      relationalSlots.applyAssertion({
        subject_entity_id: quarantinedSubject,
        slot_key: "partner.name",
        asserted_value: "Dee",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      relationalSlots.applyAssertion({
        subject_entity_id: quarantinedSubject,
        slot_key: "partner.name",
        asserted_value: "Eli",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      relationalSlots.applyAssertion({
        subject_entity_id: quarantinedSubject,
        slot_key: "partner.name",
        asserted_value: "Finn",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      const revokedSlot = relationalSlots.applyAssertion({
        subject_entity_id: createEntityId(),
        slot_key: "partner.name",
        asserted_value: "Grey",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      relationalSlots.setState(revokedSlot.slot.id, "revoked");

      const reviewQueue = new ReviewQueueRepository({ db, clock });
      reviewQueue.enqueue({
        kind: "new_insight",
        refs: {},
        reason: "First metrics fixture.",
      });
      reviewQueue.enqueue({
        kind: "new_insight",
        refs: {},
        reason: "Second metrics fixture.",
      });
      reviewQueue.enqueue({
        kind: "contradiction",
        refs: {},
        reason: "Contradiction metrics fixture.",
      });

      const identityEvents = new IdentityEventRepository({ db, clock });
      identityEvents.record({
        record_type: "open_question",
        record_id: "open_question_metrics_1",
        action: "resolve",
        old_value: {
          status: OPEN_QUESTION_OPEN_STATUS,
        },
        new_value: {
          status: OPEN_QUESTION_RESOLVED_STATUS,
        },
        provenance: { kind: "manual" },
      });

      const borg = {
        ...fakeBorg(),
        actions,
        commitments: {
          list: (options = {}) => commitments.list(options),
          countActive: () => commitments.countActive(),
          countSuperseded: () => commitments.countSuperseded(),
          countRevoked: () => commitments.countRevoked(),
          countExpired: () => commitments.countExpired(),
          countCanonicalized: () => commitments.countCanonicalized(),
        },
        relationalSlots: {
          countByState: () => relationalSlots.countByState(),
        },
        review: {
          list: (options = {}) => reviewQueue.list(options),
        },
        identity: {
          listEvents: (...args: Parameters<IdentityEventRepository["list"]>) =>
            identityEvents.list(...args),
        },
        workmem: {
          load: (id = sessionId) => workingMemoryStore.load(id),
          getPendingActionMergeCount: () => workingMemoryStore.getPendingActionMergeCount(),
        },
      } as unknown as Borg;
      const row = await new MetricsCapture(metricsPath).capture(borg, "turn-memory-bands", 1, {
        sessionId,
        sessionIds: [sessionId],
        transportChatAttempts: 1,
      });

      expect(row.action_record_count_total).toBe(3);
      expect(row.action_record_count_by_state).toEqual({
        ...zeroCounts(ACTION_STATES),
        considering: 1,
        completed: 2,
      });
      expect(row.action_record_count_committed_to_do).toBe(0);
      expect(row.action_record_count_canonicalized).toBe(1);
      expect(row.action_record_count_active).toBe(1);
      expect(row.action_record_creation_source_per_turn).toEqual({
        extractor: 0,
        reflector: 0,
        api: 0,
        unknown: 3,
      });
      expect(row.action_record_creation_count_this_turn).toBe(3);
      expect(row.recent_completed_action_count).toBe(2);
      expect(row.commitment_count_active).toBe(2);
      expect(row.commitment_count_superseded).toBe(1);
      expect(row.commitment_count_revoked).toBe(2);
      expect(row.commitment_count_expired).toBe(1);
      expect(row.commitment_count_canonicalized).toBe(1);
      expect(row.pending_action_count).toBe(1);
      expect(row.pending_action_merge_count).toBe(1);
      expect(row.relational_slot_count_by_state).toEqual({
        ...zeroCounts(RELATIONAL_SLOT_STATES),
        established: 1,
        contested: 1,
        quarantined: 1,
        revoked: 1,
      });
      expect(row.review_queue_open_count_by_type).toEqual({
        ...zeroCounts(REVIEW_KINDS),
        contradiction: 1,
        new_insight: 2,
      });
      expect(row.open_question_resolved_count).toBe(1);
    } finally {
      db.close();
    }
  });

  it("emits checkpoint duplicate-pressure traces without merging action records", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");
    const sessionId = createSessionId();
    const first = makeAction({
      description: "Review Atlas rollout",
      state: "committed_to_do",
    });
    const second = makeAction({
      description: "Check Atlas deployment",
      state: "scheduled",
    });
    const third = makeAction({
      description: "Draft billing follow-up",
      state: "considering",
    });
    const records = [first, second, third];
    const actions = {
      count: () => records.length,
      countByState: () => ({
        ...zeroCounts(ACTION_STATES),
        considering: 1,
        committed_to_do: 1,
        scheduled: 1,
      }),
      countCanonicalized: () => 0,
      countActive: () => records.length,
      getCreationCountsBySource: () => ({
        extractor: 0,
        reflector: 0,
        api: 0,
        unknown: 0,
      }),
      countCompletedSince: () => 0,
      latestCompletedAt: () => null,
      listCompletedIds: () => [],
      list: () => records,
      findSimilarDescriptionPairs: async () => [
        {
          leftId: first.id,
          rightId: second.id,
          similarity: 0.9,
        },
      ],
    };
    const borg = {
      ...fakeBorg(),
      actions,
    } as unknown as Borg;

    await new MetricsCapture(metricsPath, { tracePath }).capture(borg, "turn-duplicate", 10, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });

    const trace = readFileSync(tracePath, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);

    expect(records).toHaveLength(3);
    expect(trace).toContainEqual(
      expect.objectContaining({
        event: "action_duplicate_pressure_observed",
        turnId: "turn-duplicate",
        cluster_count: 1,
        max_cluster_size: 2,
        total_actions_in_clusters: 2,
        threshold_used: 0.85,
      }),
    );
  });

  it("captures aborted turns with a failure reason", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();
    const failureReason = "transport failed";

    const row = await capture.captureAborted(fakeBorg(), 4, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 3,
      failureReason,
    });

    expect(row.event).toBe("aborted_turn");
    expect(row.turn_counter).toBe(4);
    expect(row.transport_chat_attempts).toBe(3);
    expect(row.failure_reason).toBe(failureReason);
  });

  it("excludes aborted suppressions from generation_suppression_count", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();
    const activeTurnId = "turn-active-suppression";
    const abortedTurnId = "turn-aborted-suppression";
    const streamEntriesBySession = new Map<SessionId, StreamEntry[]>([
      [
        sessionId,
        [
          {
            id: createStreamEntryId(),
            timestamp: 1,
            kind: "agent_suppressed",
            content: { reason: "generation_gate" },
            turn_id: activeTurnId,
            session_id: sessionId,
            compressed: false,
            sender_entity_id: null,
            reply_target_entity_id: null,
          },
          {
            id: createStreamEntryId(),
            timestamp: 2,
            kind: "agent_suppressed",
            content: { reason: "generation_gate" },
            turn_id: abortedTurnId,
            session_id: sessionId,
            compressed: false,
            sender_entity_id: null,
            reply_target_entity_id: null,
          },
          {
            id: createStreamEntryId(),
            timestamp: 3,
            kind: "internal_event",
            content: {
              event: ABORTED_TURN_EVENT,
              turn_id: abortedTurnId,
              reason: "turn failed",
            },
            turn_id: abortedTurnId,
            turn_status: "aborted",
            session_id: sessionId,
            compressed: false,
            sender_entity_id: null,
            reply_target_entity_id: null,
          },
        ],
      ],
    ]);
    const borg = fakeBorg({ streamEntriesBySession });

    const completed = await capture.capture(borg, activeTurnId, 1, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
    });
    const aborted = await capture.captureAborted(borg, 2, {
      sessionId,
      sessionIds: [sessionId],
      transportChatAttempts: 1,
      failureReason: "turn failed",
      turnId: abortedTurnId,
    });

    expect(completed.generation_suppression_count).toBe(1);
    expect(aborted.generation_suppression_count).toBe(1);
  });
});
