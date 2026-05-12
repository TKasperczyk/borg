import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { ActionRecord } from "../../memory/actions/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type { RelationalSlot } from "../../memory/relational-slots/index.js";
import { OpenQuestionsRepository, type OpenQuestion } from "../../memory/self/index.js";
import { selfMigrations } from "../../memory/self/migrations.js";
import type { RetrievedEpisode, RetrievedSemantic } from "../../retrieval/index.js";
import { createEpisodeFixture, createRetrievalScoreFixture } from "../../offline/test-support.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter, type StreamEntry } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createActionId,
  createCommitmentId,
  createEntityId,
  createEpisodeId,
  createGoalId,
  createOpenQuestionId,
  createRelationalSlotId,
  createSessionId,
  createSemanticEdgeId,
  createSemanticNodeId,
  createStreamEntryId,
  type EntityId,
} from "../../util/ids.js";
import { EvidenceLedgerBuilder } from "./builder.js";

const NOW_MS = 1_800_000_000_000;

function makeWorkingMemory() {
  return {
    session_id: DEFAULT_SESSION_ID,
    turn_counter: 4,
    hot_entities: [],
    pending_actions: [],
    pending_social_attribution: null,
    pending_trait_attribution: null,
    mood: null,
    pending_procedural_attempts: [],
    discourse_state: {
      stop_until_substantive_content: null,
    },
    suppressed: [],
    mode: "problem_solving" as const,
    updated_at: NOW_MS,
  };
}

function makeRetrievedEpisode(input: {
  id: ReturnType<typeof createEpisodeId>;
  narrative: string;
  sourceStreamIds: StreamEntry["id"][];
  citationChain: StreamEntry[];
}): RetrievedEpisode {
  return {
    episode: createEpisodeFixture({
      id: input.id,
      title: `${input.id} title`,
      narrative: input.narrative,
      source_stream_ids: input.sourceStreamIds,
      created_at: NOW_MS,
      updated_at: NOW_MS,
    }),
    score: 0.9,
    scoreBreakdown: createRetrievalScoreFixture({ similarity: 0.9 }),
    citationChain: input.citationChain,
  };
}

function makeSemanticNode(input: {
  episodeId: ReturnType<typeof createEpisodeId>;
  label?: string;
}): RetrievedSemantic["matched_nodes"][number] {
  return {
    id: createSemanticNodeId(),
    kind: "proposition",
    label: input.label ?? "Qualia proof proposition",
    description: "A semantic proposition derived from an assistant self-report episode.",
    domain: null,
    aliases: [],
    confidence: 0.7,
    source_episode_ids: [input.episodeId],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    last_verified_at: NOW_MS,
    embedding: new Float32Array([0, 1, 0, 0]),
    archived: false,
    superseded_by: null,
  };
}

function makeSemanticEdge(input: {
  fromNodeId: ReturnType<typeof createSemanticNodeId>;
  toNodeId: ReturnType<typeof createSemanticNodeId>;
  episodeId: ReturnType<typeof createEpisodeId>;
}): RetrievedSemantic["support_hits"][number]["edgePath"][number] {
  return {
    id: createSemanticEdgeId(),
    from_node_id: input.fromNodeId,
    to_node_id: input.toNodeId,
    relation: "supports",
    confidence: 0.6,
    evidence_episode_ids: [input.episodeId],
    created_at: NOW_MS,
    last_verified_at: NOW_MS,
    valid_from: NOW_MS,
    valid_to: null,
    invalidated_at: null,
    invalidated_by_edge_id: null,
    invalidated_by_review_id: null,
    invalidated_by_process: null,
    invalidated_reason: null,
  };
}

function makeAction(
  streamEntryId: StreamEntry["id"],
  overrides: Partial<ActionRecord> = {},
): ActionRecord {
  const state = overrides.state ?? "scheduled";

  return {
    id: createActionId(),
    description: "File the Barcelona callback note",
    actor: "borg",
    audience_entity_id: null,
    goal_id: null,
    open_question_id: null,
    state,
    confidence: 0.86,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [streamEntryId],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    considering_at: null,
    committed_at: null,
    scheduled_at: state === "scheduled" ? NOW_MS : null,
    completed_at: null,
    not_done_at: null,
    unknown_at: null,
    ...overrides,
  };
}

function makeSlot(
  streamEntryId: StreamEntry["id"],
  overrides: Partial<RelationalSlot> = {},
): RelationalSlot {
  return {
    id: createRelationalSlotId(),
    subject_entity_id: "ent_aaaaaaaaaaaaaaaa" as RelationalSlot["subject_entity_id"],
    slot_key: "tutor.name",
    value: "Marta",
    state: "established",
    evidence_stream_entry_ids: [streamEntryId],
    contradicted_by_stream_entry_ids: [],
    alternate_values: [],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    ...overrides,
  };
}

function makeCommitment(streamEntryId: StreamEntry["id"]): CommitmentRecord {
  return {
    id: createCommitmentId(),
    type: "preference",
    directive_family: "current_session_primacy",
    closure_pressure_relevance: "neutral",
    directive: "Use the current session before prior summaries.",
    priority: 80,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    provenance: {
      kind: "online",
      process: "test",
    },
    source_stream_entry_ids: [streamEntryId],
    created_at: NOW_MS,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    last_reinforced_at: NOW_MS,
  };
}

function makeOpenQuestion(episodeId: ReturnType<typeof createEpisodeId>): OpenQuestion {
  return {
    id: createOpenQuestionId(),
    question: "Should the callback be attributed to this session or an older one?",
    urgency: 0.7,
    status: "open",
    goal_id: null,
    audience_entity_id: null,
    related_episode_ids: [episodeId],
    related_semantic_node_ids: [],
    provenance: null,
    source: "deliberator",
    created_at: NOW_MS,
    last_touched: NOW_MS,
    resolution_evidence_episode_ids: [],
    resolution_evidence_stream_entry_ids: [],
    resolution_note: null,
    resolved_at: null,
    abandoned_reason: null,
    abandoned_at: null,
    unresolved_rumination_ticks: 0,
    last_ruminated_at: null,
  };
}

describe("EvidenceLedgerBuilder", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("orders sections, derives current/prior scope from handles, and includes transcript under budget", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Marta is the tutor for the Barcelona callback.",
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "I will keep Marta tied to the current callback.",
      persistence_class: "assistant_self_report",
    });
    const priorEntry: StreamEntry = {
      id: createStreamEntryId(),
      timestamp: NOW_MS - 60_000,
      kind: "user_msg",
      content: "An older session mentioned Barcelona without Marta.",
      turn_status: "active",
      session_id: createSessionId(),
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    };
    const priorEpisodeId = createEpisodeId();
    const action = makeAction(userEntry.id);
    const slot = makeSlot(userEntry.id);
    const commitment = makeCommitment(userEntry.id);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [slot],
      },
      actionRepository: {
        list: () => [action],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-1",
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [commitment],
      retrievedEvidence: [
        {
          id: "raw-current",
          source: "recent_raw_stream",
          text: String(assistantEntry.content),
          provenance: {
            streamIds: [assistantEntry.id],
          },
          recallIntentId: "intent-1",
          matchedTerms: [],
          score: 0.8,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [
        makeRetrievedEpisode({
          id: createEpisodeId(),
          narrative: "Current callback narrative.",
          sourceStreamIds: [userEntry.id],
          citationChain: [userEntry],
        }),
        makeRetrievedEpisode({
          id: priorEpisodeId,
          narrative: "Prior Barcelona narrative.",
          sourceStreamIds: [priorEntry.id],
          citationChain: [priorEntry],
        }),
      ],
      retrievedSemantic: null,
      openQuestions: [makeOpenQuestion(priorEpisodeId)],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(ledger.sections.map((section) => section.id)).toEqual([
      "current_user_message",
      "current_session_transcript",
      "commitments_and_constraints",
      "closure_discourse_state",
      "contradictions_quarantines",
      "action_states",
      "relational_slots",
      "retrieved_raw_stream_evidence",
      "retrieved_memory_evidence",
      "episodes",
      "semantic_graph",
      "open_questions",
      "prior_session_memory",
    ]);
    expect(ledger.transcriptIncluded).toBe(true);
    expect(
      ledger.sections.find((section) => section.id === "current_user_message")?.entries[0],
    ).toMatchObject({
      id: `current_user_message:${userEntry.id}`,
      stream_index: 0,
    });
    expect(
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries,
    ).toEqual([
      expect.objectContaining({
        id: `current_session_stream:${userEntry.id}`,
        stream_index: 0,
      }),
      expect.objectContaining({
        id: `current_session_stream:${assistantEntry.id}`,
        stream_index: 1,
        persistence_class: "assistant_self_report",
      }),
    ]);
    // Sprint 8d.6.3: the retrieved raw stream item points at a stream id
    // already covered by the current_session_transcript section, so the
    // duplicate retrieved_raw_stream_evidence row is dropped. The
    // underlying assistantEntry is rendered exactly once (in the
    // transcript section above), with persistence_class preserved.
    expect(
      ledger.sections
        .find((section) => section.id === "retrieved_raw_stream_evidence")
        ?.entries.find((entry) => entry.id === "retrieved_stream:raw-current"),
    ).toBeUndefined();
    expect(
      ledger.sections.find((section) => section.id === "action_states")?.entries[0],
    ).toMatchObject({
      source_type: "action_record",
      session_scope: "current_session",
      state: "scheduled",
    });
    expect(
      ledger.sections.find((section) => section.id === "relational_slots")?.entries[0],
    ).toMatchObject({
      session_scope: "current_session",
      value: "tutor.name=Marta",
      state: "established",
    });
    expect(ledger.sections.find((section) => section.id === "episodes")?.entries).toEqual([
      expect.objectContaining({
        session_scope: "current_session",
        text: "Current callback narrative.",
      }),
    ]);
    expect(
      ledger.sections.find((section) => section.id === "prior_session_memory")?.entries,
    ).toEqual([
      expect.objectContaining({
        source_type: "episode",
        session_scope: "prior_session",
        text: "Prior Barcelona narrative.",
      }),
      expect.objectContaining({
        source_type: "system_metadata",
        session_scope: "prior_session",
      }),
    ]);
  });

  it("renders relational slots scoped and ordered by active participant", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const bob = createEntityId();
    const unseen = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice gives her tutor update.",
      sender_entity_id: alice,
    });
    const bobEntry = await writer.append({
      kind: "user_msg",
      content: "Bob gives his dog update.",
      sender_entity_id: bob,
    });
    const slots = [
      makeSlot(aliceEntry.id, {
        subject_entity_id: alice,
        slot_key: "tutor.name",
        value: "Marta",
      }),
      makeSlot(bobEntry.id, {
        subject_entity_id: bob,
        slot_key: "dog.name",
        value: "Niko",
      }),
      makeSlot(aliceEntry.id, {
        subject_entity_id: unseen,
        slot_key: "partner.name",
        value: "Lee",
      }),
    ];
    const listSlots = (
      options: {
        subjectEntityId?: EntityId;
        states?: readonly RelationalSlot["state"][];
        limit?: number;
      } = {},
    ) =>
      slots
        .filter(
          (slot) =>
            (options.subjectEntityId === undefined ||
              slot.subject_entity_id === options.subjectEntityId) &&
            (options.states === undefined || options.states.some((state) => state === slot.state)),
        )
        .slice(0, options.limit ?? slots.length);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: listSlots,
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(bobEntry.content),
      currentUserEntry: bobEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [
        {
          entityId: bob,
          displayName: "Bob",
          role: "speaker",
        },
        {
          entityId: alice,
          displayName: "Alice",
          role: "participant",
        },
      ],
    });
    const relationalEntries =
      ledger.sections.find((section) => section.id === "relational_slots")?.entries ?? [];

    expect(relationalEntries.map((entry) => entry.value)).toEqual([
      "dog.name=Niko",
      "tutor.name=Marta",
    ]);
    expect(relationalEntries.map((entry) => entry.state_metadata)).toEqual([
      {
        subject_entity_id: bob,
        subject_display_name: "Bob",
        subject_role: "speaker",
      },
      {
        subject_entity_id: alice,
        subject_display_name: "Alice",
        subject_role: "participant",
      },
    ]);
  });

  it("renders legacy global relational slots when active participant set is empty", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const alice = createEntityId();
    const bob = createEntityId();
    const aliceEntry = await writer.append({
      kind: "user_msg",
      content: "Alice gives her tutor update.",
      sender_entity_id: null,
    });
    const bobEntry = await writer.append({
      kind: "user_msg",
      content: "Bob gives his dog update.",
      sender_entity_id: null,
    });
    const slots = [
      makeSlot(aliceEntry.id, {
        subject_entity_id: alice,
        slot_key: "tutor.name",
        value: "Marta",
      }),
      makeSlot(bobEntry.id, {
        subject_entity_id: bob,
        slot_key: "dog.name",
        value: "Niko",
      }),
    ];
    const listSlots = (
      options: {
        subjectEntityId?: EntityId;
        states?: readonly RelationalSlot["state"][];
        limit?: number;
      } = {},
    ) =>
      slots
        .filter(
          (slot) =>
            (options.subjectEntityId === undefined ||
              slot.subject_entity_id === options.subjectEntityId) &&
            (options.states === undefined || options.states.some((state) => state === slot.state)),
        )
        .slice(0, options.limit ?? slots.length);
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: listSlots,
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(bobEntry.content),
      currentUserEntry: bobEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
      activeParticipants: [],
    });
    const relationalEntries =
      ledger.sections.find((section) => section.id === "relational_slots")?.entries ?? [];

    expect(relationalEntries.map((entry) => entry.value)).toEqual([
      "tutor.name=Marta",
      "dog.name=Niko",
    ]);
    expect(relationalEntries.map((entry) => entry.state_metadata)).toEqual([undefined, undefined]);
  });

  it("surfaces the most recent speaker when the current user entry has a sender", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const senderEntityId = createEntityId();
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Atlas needs a rollback plan.",
      sender_entity_id: senderEntityId,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      entityRepository: {
        get: (id: EntityId) =>
          id === senderEntityId
            ? {
                id: senderEntityId,
                canonical_name: "Alice",
                aliases: [],
                kind: "person",
                created_at: NOW_MS,
              }
            : null,
      },
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-speaker",
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    const currentUserEntry = ledger.sections.find(
      (section) => section.id === "current_user_message",
    )?.entries[0];

    expect(currentUserEntry?.text).toBe("Atlas needs a rollback plan.");
    expect(currentUserEntry?.state_metadata).toEqual({
      sender_entity_id: senderEntityId,
      sender_display_name: "Alice",
    });
  });

  it("surfaces agent reply targets in current-session transcript metadata", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const aliceEntityId = createEntityId();
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    await writer.append({
      kind: "user_msg",
      content: "Can someone pick the train dates?",
      sender_entity_id: aliceEntityId,
      audience: "Planning Channel",
    });
    await writer.append({
      kind: "agent_msg",
      content: "Alice, can you own the train dates?",
      audience: "Planning Channel",
      reply_target_entity_id: aliceEntityId,
    });
    await writer.append({
      kind: "agent_msg",
      content: "For the channel, keep budget and rest days together.",
      audience: "Planning Channel",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      entityRepository: {
        get: (id: EntityId) =>
          id === aliceEntityId
            ? {
                id: aliceEntityId,
                canonical_name: "Alice",
                aliases: [],
                kind: "person",
                created_at: NOW_MS,
              }
            : null,
      },
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-reply-target",
      audienceEntityId: null,
      currentUserMessage: "Next message",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const transcriptEntries =
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries ?? [];

    expect(transcriptEntries.map((entry) => entry.state_metadata)).toEqual([
      {
        sender_entity_id: aliceEntityId,
        sender_display_name: "Alice",
      },
      {
        reply_target_kind: "entity",
        reply_target_entity_id: aliceEntityId,
        reply_target_display_name: "Alice",
      },
      undefined,
    ]);
  });

  it("preserves legacy single-persona agent transcript metadata shape", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Can you keep this on the rollout list?",
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "I will keep it on the rollout list.",
      reply_target_entity_id: null,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-legacy-agent-metadata",
      audienceEntityId: null,
      currentUserMessage: "Next message",
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const transcriptEntries =
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries ?? [];
    const agentEntry = transcriptEntries.find(
      (entry) => entry.id === `current_session_stream:${assistantEntry.id}`,
    );

    expect(transcriptEntries.map((entry) => entry.id)).toContain(
      `current_session_stream:${userEntry.id}`,
    );
    expect(agentEntry).not.toHaveProperty("state_metadata");
  });

  it("keeps current user message rendering unchanged when sender is omitted", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Atlas needs a rollback plan.",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      entityRepository: {
        get: () => {
          throw new Error("sender lookup should not run for omitted sender ids");
        },
      },
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn-legacy-speaker",
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(
      ledger.sections.find((section) => section.id === "current_user_message")?.entries[0]?.text,
    ).toBe("Atlas needs a rollback plan.");
  });

  it("renders one action thread for same-goal similar action transitions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "I need to write the harness doc.",
    });
    const goalId = createGoalId();
    const considering = makeAction(userEntry.id, {
      description: "consider writing the harness presentation",
      actor: "user",
      goal_id: goalId,
      state: "considering",
      created_at: NOW_MS,
      updated_at: NOW_MS,
      considering_at: NOW_MS,
      scheduled_at: null,
    });
    const committed = makeAction(userEntry.id, {
      description: "write the harness presentation",
      actor: "user",
      goal_id: goalId,
      state: "committed_to_do",
      created_at: NOW_MS + 1,
      updated_at: NOW_MS + 10,
      committed_at: NOW_MS + 10,
      scheduled_at: null,
    });
    const completed = makeAction(userEntry.id, {
      description: "finished writing the harness presentation",
      actor: "user",
      goal_id: goalId,
      state: "completed",
      created_at: NOW_MS + 2,
      updated_at: NOW_MS + 20,
      completed_at: NOW_MS + 20,
      scheduled_at: null,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: {
        list: () => [completed, committed, considering],
        findSimilarDescriptionPairs: async () => [
          { leftId: considering.id, rightId: committed.id, similarity: 0.91 },
          { leftId: committed.id, rightId: completed.id, similarity: 0.92 },
        ],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      actionThreadSimilarityThreshold: 0.85,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const actionEntries =
      ledger.sections.find((section) => section.id === "action_states")?.entries ?? [];

    expect(actionEntries).toHaveLength(1);
    expect(actionEntries[0]).toMatchObject({
      id: expect.stringMatching(/^action_thread:/),
      state: "completed",
      state_metadata: expect.objectContaining({
        transitions: 3,
        current_action_id: completed.id,
        goal_id: goalId,
      }),
    });
    expect(actionEntries[0]?.text).toContain(
      "originating_intent: consider writing the harness presentation",
    );
    expect(actionEntries[0]?.text).toContain("transitions: 3, current: completed");
    expect(actionEntries[0]?.text).toContain(
      "current_intent: finished writing the harness presentation",
    );
  });

  it("renders non-raw retrieved evidence sources into ledger sections", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Use the ledger-only evidence.",
    });
    const commitmentId = createCommitmentId();
    const warmEpisodeId = createEpisodeId();
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "warm-prior",
          source: "warm_recall",
          text: "Warm recall narrative from a prior session.",
          provenance: { episodeId: warmEpisodeId },
          recallIntentId: "warm_recall",
          matchedTerms: ["harness"],
          score: 0.31,
          scoreBreakdown: {},
        },
        {
          id: "commitment-boundary",
          source: "commitment",
          text: "boundary: Do not add terminal closures.",
          provenance: { commitmentId },
          recallIntentId: "intent-commitment",
          matchedTerms: [],
          score: 0.72,
          scoreBreakdown: {},
        },
        {
          id: "working-focus",
          source: "working_state",
          text: "Working state focus is the harness presentation.",
          recallIntentId: "intent-working",
          matchedTerms: [],
          score: 0.44,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(
      ledger.sections
        .find((section) => section.id === "retrieved_memory_evidence")
        ?.entries.find((entry) => entry.id === "retrieved_evidence:warm-prior"),
    ).toMatchObject({
      source_type: "episode",
      value: "warm_recall",
      text: "Warm recall narrative from a prior session.",
      state: expect.stringContaining("intent=warm_recall"),
      via_retrieval: true,
    });
    expect(
      ledger.sections
        .find((section) => section.id === "commitments_and_constraints")
        ?.entries.find((entry) => entry.id === "retrieved_evidence:commitment-boundary"),
    ).toMatchObject({
      source_type: "commitment",
      value: "commitment",
      text: "boundary: Do not add terminal closures.",
      state_metadata: expect.objectContaining({ commitment_id: commitmentId }),
      via_retrieval: true,
    });
    expect(
      ledger.sections
        .find((section) => section.id === "closure_discourse_state")
        ?.entries.find((entry) => entry.id === "retrieved_evidence:working-focus"),
    ).toMatchObject({
      source_type: "system_metadata",
      actor: "system",
      value: "working_state",
      text: "Working state focus is the harness presentation.",
      via_retrieval: true,
    });
  });

  it("does not collapse action threads across distinct goals or low similarity", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "I need to write and then book flights.",
    });
    const docGoalId = createGoalId();
    const travelGoalId = createGoalId();
    const docAction = makeAction(userEntry.id, {
      description: "write the harness presentation",
      actor: "user",
      goal_id: docGoalId,
      state: "committed_to_do",
      committed_at: NOW_MS + 10,
      updated_at: NOW_MS + 10,
      scheduled_at: null,
    });
    const docDifferentIntent = makeAction(userEntry.id, {
      description: "ask lead for platform budget",
      actor: "user",
      goal_id: docGoalId,
      state: "scheduled",
      updated_at: NOW_MS + 20,
      scheduled_at: NOW_MS + 20,
    });
    const travelAction = makeAction(userEntry.id, {
      description: "book Spain flights",
      actor: "user",
      goal_id: travelGoalId,
      state: "scheduled",
      updated_at: NOW_MS + 30,
      scheduled_at: NOW_MS + 30,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: {
        list: () => [travelAction, docDifferentIntent, docAction],
        findSimilarDescriptionPairs: async () => [
          { leftId: docAction.id, rightId: docDifferentIntent.id, similarity: 0.7 },
          { leftId: docAction.id, rightId: travelAction.id, similarity: 0.95 },
        ],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      actionThreadSimilarityThreshold: 0.85,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const actionEntries =
      ledger.sections.find((section) => section.id === "action_states")?.entries ?? [];

    expect(actionEntries).toHaveLength(3);
    expect(actionEntries.map((entry) => entry.state_metadata?.["transitions"])).toEqual([1, 1, 1]);
  });

  it("summarizes omitted null-goal action threads with state counts and bounded samples", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Track several independent action notes.",
    });
    const actions = [
      makeAction(userEntry.id, {
        description: "newest visible action",
        state: "scheduled",
        updated_at: NOW_MS + 60,
        scheduled_at: NOW_MS + 60,
      }),
      makeAction(userEntry.id, {
        description: "second visible action",
        state: "considering",
        updated_at: NOW_MS + 50,
        considering_at: NOW_MS + 50,
        scheduled_at: null,
      }),
      makeAction(userEntry.id, {
        description:
          "completed omitted thread with enough extra context to require a short bounded sample tail should not appear",
        state: "completed",
        updated_at: NOW_MS + 40,
        completed_at: NOW_MS + 40,
        scheduled_at: null,
      }),
      makeAction(userEntry.id, {
        description: "scheduled omitted thread",
        state: "scheduled",
        updated_at: NOW_MS + 30,
        scheduled_at: NOW_MS + 30,
      }),
      makeAction(userEntry.id, {
        description: "committed omitted thread",
        state: "committed_to_do",
        updated_at: NOW_MS + 20,
        committed_at: NOW_MS + 20,
        scheduled_at: null,
      }),
      makeAction(userEntry.id, {
        description: "completed older omitted thread",
        state: "completed",
        updated_at: NOW_MS + 10,
        completed_at: NOW_MS + 10,
        scheduled_at: null,
      }),
    ];
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: {
        list: () => actions,
        findSimilarDescriptionPairs: async () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
      actionThreadRenderLimit: 2,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const actionEntries =
      ledger.sections.find((section) => section.id === "action_states")?.entries ?? [];
    const summary = actionEntries.find((entry) => entry.id === "action_threads:older_summary");

    expect(actionEntries).toHaveLength(3);
    expect(summary?.text).toContain("threads=4");
    expect(summary?.text).toContain("records=4");
    expect(summary?.text).toContain("committed_to_do=1");
    expect(summary?.text).toContain("scheduled=1");
    expect(summary?.text).toContain("completed=2");
    expect(summary?.text).toContain("completed omitted thread with enough extra context");
    expect(summary?.text).not.toContain("tail should not appear");
    expect(summary?.text).toContain("scheduled omitted thread");
    expect(summary?.text).toContain("committed omitted thread");
  });

  it("renders relevant resolved open questions from the repository with state metadata", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const currentEntry = await writer.append({
      kind: "user_msg",
      content: "What should I ask about dinner now?",
    });
    const resolvedEntry = await writer.append({
      kind: "user_msg",
      content: "The mushroom dish worked out well.",
    });
    const resolvedQuestion: OpenQuestion = {
      ...makeOpenQuestion(createEpisodeId()),
      question: "Did the mushroom dish work out?",
      status: "resolved",
      resolution_evidence_stream_entry_ids: [resolvedEntry.id],
      resolution_note: "The user explicitly said the dish worked out well.",
      resolved_at: NOW_MS,
      last_touched: NOW_MS,
    };
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        findByHandles: () => [resolvedQuestion],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(currentEntry.content),
      currentUserEntry: currentEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(ledger.sections.find((section) => section.id === "open_questions")?.entries).toEqual([
      expect.objectContaining({
        id: `open_question:${resolvedQuestion.id}`,
        state: "resolved",
        state_metadata: expect.objectContaining({
          resolution_note: "The user explicitly said the dish worked out well.",
          resolved_at: NOW_MS,
          resolution_evidence_stream_entry_ids: [resolvedEntry.id],
        }),
      }),
    ]);
  });

  it("includes old resolved open questions by handle before repository limits", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const currentEntry = await writer.append({
      kind: "user_msg",
      content: "What should I ask about dinner now?",
    });
    const resolvedEntry = await writer.append({
      kind: "user_msg",
      content: "The old mushroom question resolved positively.",
    });
    const db = openDatabase(join(tempDir, "open-questions.db"), {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(NOW_MS),
    });

    try {
      for (let index = 0; index < 40; index += 1) {
        const decoy = repository.add({
          question: `Decoy resolved question ${index}?`,
          urgency: 1,
          provenance: { kind: "manual" },
          source: "user",
          created_at: NOW_MS - index,
          last_touched: NOW_MS - index,
        });

        repository.resolve(decoy.id, {
          resolution_evidence_stream_entry_ids: [createStreamEntryId()],
          resolution_note: "High-urgency decoy resolution.",
        });
      }

      const target = repository.add({
        question: "Did the old mushroom dish work out?",
        urgency: 0.01,
        provenance: { kind: "manual" },
        source: "user",
        created_at: NOW_MS - 100_000,
        last_touched: NOW_MS - 100_000,
      });
      const resolvedTarget = repository.resolve(target.id, {
        resolution_evidence_stream_entry_ids: [resolvedEntry.id],
        resolution_note: "The user said the old mushroom question resolved positively.",
      });
      const builder = new EvidenceLedgerBuilder({
        createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
        relationalSlotRepository: {
          list: () => [],
        },
        actionRepository: {
          list: () => [],
        },
        openQuestionsRepository: repository,
        currentSessionTranscriptTokenBudget: 50_000,
      });

      const ledger = await builder.build({
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: null,
        currentUserMessage: String(currentEntry.content),
        currentUserEntry: currentEntry,
        workingMemory: makeWorkingMemory(),
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        retrievedSemantic: null,
        openQuestions: [],
        pendingCorrections: [],
        frameAnomaly: null,
      });

      expect(ledger.sections.find((section) => section.id === "open_questions")?.entries).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: `open_question:${resolvedTarget.id}`,
            state: "resolved",
          }),
        ]),
      );
    } finally {
      db.close();
    }
  });

  it("compacts older assistant transcript entries without dropping user-authored facts", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const factEntry = await writer.append({
      kind: "user_msg",
      content: "The launch window is Tuesday and the reviewer is Priya.",
    });
    for (let index = 0; index < 10; index += 1) {
      await writer.append({
        kind: "agent_msg",
        content: `Assistant planning response ${index} with implementation details repeated for budget pressure.`,
      });
    }
    const currentEntry = await writer.append({
      kind: "user_msg",
      content: "Current question should be rendered above, not duplicated in full here.",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 1,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(currentEntry.content),
      currentUserEntry: currentEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const transcriptEntries =
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries ?? [];

    expect(ledger.transcriptIncluded).toBe(true);
    expect(ledger.transcriptCompacted).toBe(true);
    expect(ledger.transcriptOmittedReason).toBeUndefined();
    expect(ledger.originalTranscriptTokenEstimate).toBeGreaterThan(1);
    expect(ledger.compactedTranscriptEntryCount).toBe(3);
    expect(ledger.rawPreservedUserTranscriptEntryCount).toBe(1);
    expect(transcriptEntries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: `current_session_stream:${factEntry.id}`,
          text: "The launch window is Tuesday and the reviewer is Priya.",
        }),
        expect.objectContaining({
          source_type: "system_metadata",
          state: "compacted",
          text: expect.stringContaining("Earlier assistant/system transcript entries compacted"),
        }),
        expect.objectContaining({
          id: `current_session_compacted_current_user:${currentEntry.id}`,
          text: expect.stringContaining("full text is rendered in section 1"),
        }),
      ]),
    );
  });

  it("propagates assistant self-report persistence through episode and semantic ledger entries", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Does that prove you have qualia?",
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "The gap feels like a discontinuity with a remembered edge.",
      persistence_class: "assistant_self_report",
    });
    const episodeId = createEpisodeId();
    const matchedNode = makeSemanticNode({
      episodeId,
      label: "Verified qualia claim",
    });
    const supportNode = makeSemanticNode({
      episodeId,
      label: "Self-report support",
    });
    const supportEdge = makeSemanticEdge({
      fromNodeId: matchedNode.id,
      toNodeId: supportNode.id,
      episodeId,
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [],
      retrievedEpisodes: [
        makeRetrievedEpisode({
          id: episodeId,
          narrative: "The earlier assistant self-report does not establish verified qualia.",
          sourceStreamIds: [assistantEntry.id],
          citationChain: [assistantEntry],
        }),
      ],
      retrievedSemantic: {
        supports: [supportNode],
        contradicts: [],
        categories: [],
        matched_node_ids: [matchedNode.id],
        matched_nodes: [matchedNode],
        support_hits: [
          {
            root_node_id: matchedNode.id,
            node: supportNode,
            edgePath: [supportEdge],
          },
        ],
        causal_hits: [],
        contradiction_hits: [],
        category_hits: [],
      },
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const allEntries = ledger.sections.flatMap((section) => section.entries);

    expect(allEntries.find((entry) => entry.id === `episode:${episodeId}`)).toMatchObject({
      persistence_class: "assistant_self_report",
    });
    expect(
      allEntries.find((entry) => entry.id === `semantic_node:${matchedNode.id}`),
    ).toMatchObject({
      persistence_class: "assistant_self_report",
    });
    expect(
      allEntries.find((entry) => entry.id === `semantic_node:${supportNode.id}`),
    ).toMatchObject({
      persistence_class: "assistant_self_report",
    });
    expect(
      allEntries.find((entry) => entry.id === `semantic_edge:${supportEdge.id}`),
    ).toMatchObject({
      persistence_class: "assistant_self_report",
    });

    expect(userEntry.kind).toBe("user_msg");
  });

  it("labels retrieved assistant self-report raw stream evidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const assistantEntry = await writer.append({
      kind: "agent_msg",
      content: "The gap feels like a discontinuity with a remembered edge.",
      persistence_class: "assistant_self_report",
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "What did you say earlier?",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 1,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "raw-self-report",
          source: "raw_stream",
          text: String(assistantEntry.content),
          provenance: {
            streamIds: [assistantEntry.id],
          },
          recallIntentId: "intent-self-report",
          matchedTerms: [],
          score: 0.9,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const rawEntry = ledger.sections
      .find((section) => section.id === "retrieved_raw_stream_evidence")
      ?.entries.find((entry) => entry.id === "retrieved_stream:raw-self-report");
    const transcriptEntry = ledger.sections
      .find((section) => section.id === "current_session_transcript")
      ?.entries.find((entry) => entry.id === `current_session_stream:${assistantEntry.id}`);

    expect(ledger.transcriptIncluded).toBe(true);
    expect(ledger.transcriptCompacted).toBe(true);
    expect(transcriptEntry).toMatchObject({
      source_type: "current_session_stream",
      actor: "assistant",
      persistence_class: "assistant_self_report",
    });
    expect(rawEntry).toBeUndefined();

    expect(userEntry.kind).toBe("user_msg");
  });

  it("labels raw stream evidence scope only when every provenance handle resolves consistently", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const currentEntry = await writer.append({
      kind: "user_msg",
      content: "Current-session raw evidence.",
    });
    const priorEntry: StreamEntry = {
      id: createStreamEntryId(),
      timestamp: NOW_MS - 60_000,
      kind: "user_msg",
      content: "Prior-session raw evidence.",
      turn_status: "active",
      session_id: createSessionId(),
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    };
    const unresolvedEntryId = createStreamEntryId();
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository: {
        list: () => [],
      },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(currentEntry.content),
      currentUserEntry: currentEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "raw-all-current",
          source: "raw_stream",
          text: "all current",
          provenance: {
            streamIds: [currentEntry.id],
          },
          recallIntentId: "intent-current",
          matchedTerms: [],
          score: 0.9,
          scoreBreakdown: {},
        },
        {
          id: "raw-all-prior",
          source: "raw_stream",
          text: "all prior",
          provenance: {
            streamIds: [priorEntry.id],
          },
          recallIntentId: "intent-prior",
          matchedTerms: [],
          score: 0.8,
          scoreBreakdown: {},
        },
        {
          id: "raw-mixed",
          source: "raw_stream",
          text: "mixed",
          provenance: {
            streamIds: [currentEntry.id, priorEntry.id],
          },
          recallIntentId: "intent-mixed",
          matchedTerms: [],
          score: 0.7,
          scoreBreakdown: {},
        },
        {
          id: "raw-unresolved",
          source: "raw_stream",
          text: "unresolved",
          provenance: {
            streamIds: [unresolvedEntryId],
          },
          recallIntentId: "intent-unresolved",
          matchedTerms: [],
          score: 0.6,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [
        makeRetrievedEpisode({
          id: createEpisodeId(),
          narrative: "Prior source bridge.",
          sourceStreamIds: [priorEntry.id],
          citationChain: [priorEntry],
        }),
      ],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });
    const allEntries = ledger.sections.flatMap((section) => section.entries);

    // Sprint 8d.6.3: raw-all-current points at currentEntry.id which is
    // already in the current_session_transcript section, so it is deduped.
    // Its scope info still appears -- on the transcript entry itself.
    expect(
      allEntries.find((entry) => entry.id === "retrieved_stream:raw-all-current"),
    ).toBeUndefined();
    expect(
      allEntries.find((entry) => entry.id === `current_session_stream:${currentEntry.id}`),
    ).toMatchObject({
      source_type: "current_session_stream",
      session_scope: "current_session",
      stream_index: 0,
    });
    expect(allEntries.find((entry) => entry.id === "retrieved_stream:raw-all-prior")).toMatchObject(
      {
        source_type: "prior_session_stream",
        session_scope: "prior_session",
      },
    );
    expect(allEntries.find((entry) => entry.id === "retrieved_stream:raw-mixed")).toMatchObject({
      source_type: "system_metadata",
      session_scope: "global",
    });
    expect(
      allEntries.find((entry) => entry.id === "retrieved_stream:raw-unresolved"),
    ).toMatchObject({
      source_type: "system_metadata",
      session_scope: "global",
    });
  });

  it("dedupes retrieved raw stream evidence against current_session_transcript by stream id", async () => {
    // Sprint 8d.6.3 regression: same underlying stream entry must not
    // appear twice (once in transcript, once in retrieved_raw_stream_evidence).
    // v36/v37 finalizer prompts had ~25k duplicate tokens from this class.
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Transcript-covered evidence.",
    });
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "raw-duplicate",
          source: "raw_stream",
          text: String(userEntry.content),
          provenance: { streamIds: [userEntry.id] },
          recallIntentId: "intent-dup",
          matchedTerms: [],
          score: 0.9,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    const transcriptEntry = ledger.sections
      .find((section) => section.id === "current_session_transcript")
      ?.entries.find((entry) => entry.id === `current_session_stream:${userEntry.id}`);
    const retrievedEntry = ledger.sections
      .find((section) => section.id === "retrieved_raw_stream_evidence")
      ?.entries.find((entry) => entry.id === "retrieved_stream:raw-duplicate");

    expect(transcriptEntry).toBeDefined();
    expect(retrievedEntry).toBeUndefined();
  });

  it("keeps retrieved raw stream evidence whose stream id is not in the transcript", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Transcript message.",
    });
    const otherStreamId = createStreamEntryId();
    const builder = new EvidenceLedgerBuilder({
      createStreamReader: (sessionId) => new StreamReader({ dataDir: tempDir, sessionId }),
      relationalSlotRepository: { list: () => [] },
      actionRepository: { list: () => [] },
      currentSessionTranscriptTokenBudget: 50_000,
    });

    const ledger = await builder.build({
      sessionId: DEFAULT_SESSION_ID,
      audienceEntityId: null,
      currentUserMessage: String(userEntry.content),
      currentUserEntry: userEntry,
      workingMemory: makeWorkingMemory(),
      applicableCommitments: [],
      retrievedEvidence: [
        {
          id: "raw-other",
          source: "raw_stream",
          text: "non-transcript text",
          provenance: { streamIds: [otherStreamId] },
          recallIntentId: "intent-other",
          matchedTerms: [],
          score: 0.7,
          scoreBreakdown: {},
        },
      ],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(
      ledger.sections
        .find((section) => section.id === "retrieved_raw_stream_evidence")
        ?.entries.find((entry) => entry.id === "retrieved_stream:raw-other"),
    ).toBeDefined();
  });
});
