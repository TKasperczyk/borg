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
  createEpisodeId,
  createOpenQuestionId,
  createRelationalSlotId,
  createSessionId,
  createSemanticEdgeId,
  createSemanticNodeId,
  createStreamEntryId,
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

function makeAction(streamEntryId: StreamEntry["id"]): ActionRecord {
  return {
    id: createActionId(),
    description: "File the Barcelona callback note",
    actor: "borg",
    audience_entity_id: null,
    state: "scheduled",
    confidence: 0.86,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [streamEntryId],
    created_at: NOW_MS,
    updated_at: NOW_MS,
    considering_at: null,
    committed_at: null,
    scheduled_at: NOW_MS,
    completed_at: null,
    not_done_at: null,
    unknown_at: null,
  };
}

function makeSlot(streamEntryId: StreamEntry["id"]): RelationalSlot {
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

  it("omits the whole transcript when it exceeds the configured budget", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(NOW_MS),
    });
    const userEntry = await writer.append({
      kind: "user_msg",
      content: "This transcript is intentionally longer than a one-token budget.",
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
      retrievedEvidence: [],
      retrievedEpisodes: [],
      retrievedSemantic: null,
      openQuestions: [],
      pendingCorrections: [],
      frameAnomaly: null,
    });

    expect(ledger.transcriptIncluded).toBe(false);
    expect(ledger.transcriptOmittedReason).toBe("over_budget");
    expect(
      ledger.sections.find((section) => section.id === "current_session_transcript")?.entries,
    ).toEqual([]);
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

    expect(ledger.transcriptIncluded).toBe(false);
    expect(rawEntry).toMatchObject({
      source_type: "current_session_stream",
      actor: "assistant",
      persistence_class: "assistant_self_report",
      via_retrieval: true,
    });

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
