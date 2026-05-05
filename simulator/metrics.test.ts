import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
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
import { createActionId, createEntityId, createStreamEntryId } from "../src/util/ids.js";

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
  "recent_completed_action_count",
  "commitment_count_active",
  "commitment_count_superseded",
  "pending_action_count",
  "pending_action_merge_count",
  "relational_slot_count_by_state",
  "review_queue_open_count_by_type",
  "frame_anomaly_classifier_calls",
  "frame_anomaly_classified_normal_count",
  "frame_anomaly_actual_anomaly_count",
  "frame_anomaly_degraded_count",
  "frame_anomaly_degraded_fallback_match_count",
  "quarantined_user_entry_count",
  "early_extractors_skipped_frame_anomaly_count",
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
    suppressedSessions?: readonly SessionId[];
    streamEntriesBySession?: ReadonlyMap<SessionId, readonly StreamEntry[]>;
  } = {},
  observed: { moodSessions?: SessionId[]; tailSessions?: SessionId[] } = {},
): Borg {
  const semanticNodeCount = counts.semanticNodes ?? 1;
  const semanticEdgeCount = counts.semanticEdges ?? 2;
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
          Array.from({ length: semanticNodeCount }, (_, index) => ({ id: `node_${index}` })),
      },
      edges: {
        list: () =>
          Array.from({ length: semanticEdgeCount }, (_, index) => ({ id: `edge_${index}` })),
      },
    },
    actions: {
      count: () => 0,
      countByState: () => zeroCounts(ACTION_STATES),
      countCompletedSince: () => 0,
      latestCompletedAt: () => null,
      listCompletedIds: () => [],
    },
    self: {
      openQuestions: {
        list: () => [{ id: "question_1" }],
      },
      goals: {
        list: () => [{ id: "goal_1", children: [{ id: "goal_2" }] }],
      },
    },
    commitments: {
      list: () => [],
      countActive: () => 0,
      countSuperseded: () => 0,
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
      commitments.supersede(supersededCommitment.id, activeCommitment.id);

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
      expect(row.recent_completed_action_count).toBe(2);
      expect(row.commitment_count_active).toBe(2);
      expect(row.commitment_count_superseded).toBe(1);
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
          },
          {
            id: createStreamEntryId(),
            timestamp: 2,
            kind: "agent_suppressed",
            content: { reason: "generation_gate" },
            turn_id: abortedTurnId,
            session_id: sessionId,
            compressed: false,
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
