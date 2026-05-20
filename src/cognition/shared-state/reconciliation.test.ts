import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import { ActionRepository, actionMigrations } from "../../memory/actions/index.js";
import {
  CommitmentRepository,
  commitmentMigrations,
  type CommitmentRecord,
  type CommitmentType,
} from "../../memory/commitments/index.js";
import type {
  SharedStateArtifact,
  SharedStateEntry,
} from "../../memory/decision-artifacts/index.js";
import type {
  SemanticNodeCorrectionRef,
  SemanticNodeSearchCandidate,
  SemanticNodeStatusTransition,
} from "../../memory/semantic/index.js";
import {
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "../../memory/semantic/repository.js";
import { semanticMigrations } from "../../memory/semantic/migrations.js";
import { OpenQuestionsRepository, selfMigrations } from "../../memory/self/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import {
  createActionId,
  createCommitmentId,
  createSharedStateEntryId,
  createEntityId,
  createEpisodeId,
  createGoalId,
  createOpenQuestionId,
  createSemanticNodeId,
  createStreamEntryId,
  type EpisodeId,
  type SemanticNodeId,
} from "../../util/ids.js";
import { createEpisodeFixture, createSemanticNodeFixture } from "../../offline/test-support.js";
import {
  findUnsettledSharedStateReconciliation,
  reconcileSharedStateCanonicalizations,
  reconcileSemanticBeliefRevision,
  SemanticRevisionVerdictCache,
} from "./reconciliation.js";
import { SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES } from "./commitment-canonicalization.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";

const CANONICALIZABLE_COMMITMENT_TYPES = SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES;
const NON_CANONICALIZABLE_COMMITMENT_TYPES = [
  "preference",
  "boundary",
] as const satisfies readonly CommitmentType[];
const PROMISE_COMMITMENT_TYPE = CANONICALIZABLE_COMMITMENT_TYPES[0];
const PREFERENCE_COMMITMENT_TYPE = NON_CANONICALIZABLE_COMMITMENT_TYPES[0];

function lockedEntry(overrides: Partial<SharedStateEntry> = {}): SharedStateEntry {
  const streamEntryId = createStreamEntryId();

  return {
    id: overrides.id ?? createSharedStateEntryId(),
    audience_entity_id: overrides.audience_entity_id ?? createEntityId(),
    kind: overrides.kind ?? "locked",
    text: overrides.text ?? "Release freeze is locked for the workstream",
    owner_entity_id: overrides.owner_entity_id ?? null,
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [streamEntryId],
    last_updated_stream_entry_ids: overrides.last_updated_stream_entry_ids ?? [streamEntryId],
    created_at: overrides.created_at ?? 1_000,
    last_updated_at: overrides.last_updated_at ?? 1_000,
    superseded_by_id: overrides.superseded_by_id ?? null,
    rank: overrides.rank ?? 0,
    canonicalizes: overrides.canonicalizes ?? {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [],
      open_question_ids: [],
    },
  };
}

function sharedStateArtifact(entries: readonly SharedStateEntry[]): SharedStateArtifact {
  const audienceEntityId = entries[0]?.audience_entity_id ?? createEntityId();

  return {
    audience_entity_id: audienceEntityId,
    record_version: 1,
    created_at: 1_000,
    updated_at: 1_000,
    last_compiled_at: 1_000,
    last_compiled_stream_entry_id: createStreamEntryId(),
    entries: [...entries],
  };
}

function addCommitment(
  repository: CommitmentRepository,
  input: {
    type: CommitmentRecord["type"];
    directiveFamily: string;
    directive: string;
    createdAt?: number;
    expiresAt?: number | null;
  },
): CommitmentRecord {
  return repository.add({
    type: input.type,
    directiveFamily: input.directiveFamily,
    directive: input.directive,
    priority: 5,
    provenance: { kind: "manual" },
    createdAt: input.createdAt,
    expiresAt: input.expiresAt,
    skipDirectiveFamilyMerge: true,
  });
}

function createTraceRecorder(): TurnTracer & {
  events: Array<{ event: TurnTraceEventName; data: TurnTraceData }>;
} {
  const events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];

  return {
    enabled: true,
    includePayloads: true,
    events,
    emit: vi.fn((event: TurnTraceEventName, data: TurnTraceData) => {
      events.push({ event, data });
    }),
  };
}

function semanticRevisionResponse(input: {
  verdicts: Array<{
    node_id: SemanticNodeId;
    verdict: "supersede" | "contradict" | "keep" | "uncertain";
  }>;
}) {
  return {
    text: "",
    input_tokens: 5,
    output_tokens: 5,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_decision_artifact_semantic_revision",
        name: "EmitDecisionArtifactSemanticRevision",
        input: {
          verdicts: input.verdicts,
        },
      },
    ],
  };
}

function semanticRevisionOperation(entry: SharedStateEntry) {
  return {
    type: "add" as const,
    id: entry.id,
    kind: "locked" as const,
    text: entry.text,
    owner_entity_id: entry.owner_entity_id,
    provenance_stream_entry_ids: entry.provenance_stream_entry_ids,
    last_updated_stream_entry_ids: entry.last_updated_stream_entry_ids,
    rank: entry.rank,
    canonicalizes: entry.canonicalizes,
  };
}

function semanticRevisionDependencies(input: {
  nodeId?: SemanticNodeId;
  verdict?: "supersede" | "contradict" | "keep" | "uncertain";
  searchError?: Error;
  llmError?: Error;
  candidateCount?: number;
  verdictCache?: SemanticRevisionVerdictCache;
}) {
  const episodeId = createEpisodeId();
  const node = createSemanticNodeFixture(
    {
      id: input.nodeId ?? createSemanticNodeId(),
      label: "Project runtime is Node 20",
      description: "The project runtime is Node 20.",
      source_episode_ids: [episodeId],
    },
    [1, 0, 0, 0],
  );
  const candidates = Array.from({ length: input.candidateCount ?? 1 }, (_, index) => ({
    node:
      index === 0
        ? node
        : createSemanticNodeFixture(
            {
              id: createSemanticNodeId(),
              label: `Project runtime candidate ${index}`,
              description: `Runtime candidate ${index}.`,
              source_episode_ids: [episodeId],
            },
            [1, 0, 0, 0],
          ),
    similarity: 0.95,
  }));
  const searchByVector = vi.fn(async () => {
    if (input.searchError !== undefined) {
      throw input.searchError;
    }

    return candidates;
  });
  const markSuperseded = vi.fn(
    async (id: SemanticNodeId, correctedBy: SemanticNodeCorrectionRef, supersededAt: number) => ({
      id,
      fromStatus: "active" as const,
      toStatus: "superseded" as const,
      correctedBy,
      supersededAt,
    }),
  );
  const markContradicted = vi.fn(
    async (id: SemanticNodeId, correctedBy: SemanticNodeCorrectionRef, supersededAt: number) => ({
      id,
      fromStatus: "active" as const,
      toStatus: "contradicted" as const,
      correctedBy,
      supersededAt,
    }),
  );
  const complete = vi.fn(async () => {
    if (input.llmError !== undefined) {
      throw input.llmError;
    }

    return semanticRevisionResponse({
      verdicts: candidates.map((candidate) => ({
        node_id: candidate.node.id,
        verdict: input.verdict ?? "keep",
      })),
    });
  });

  return {
    node,
    searchByVector,
    markSuperseded,
    markContradicted,
    complete,
    dependencies: {
      semanticNodeRepository: {
        searchByVector,
        markSuperseded,
        markContradicted,
      },
      episodicRepository: {
        getMany: vi.fn(async (ids: EpisodeId[]) =>
          ids.map((id) =>
            createEpisodeFixture({
              id,
              audience_entity_id: null,
              shared: true,
            }),
          ),
        ),
      },
      embeddingClient: {
        embed: vi.fn(async () => Float32Array.from([1, 0, 0, 0])),
        embedBatch: vi.fn(async (texts: readonly string[]) =>
          texts.map(() => Float32Array.from([1, 0, 0, 0])),
        ),
      },
      llmClient: {
        complete,
        converse: vi.fn(),
      },
      model: "semantic-revision-test",
      verdictCache: input.verdictCache ?? new SemanticRevisionVerdictCache(),
    },
  };
}

function semanticStatusTransition(input: {
  id: SemanticNodeId;
  toStatus: "superseded" | "contradicted";
  correctedBy: SemanticNodeCorrectionRef;
  supersededAt: number;
}): SemanticNodeStatusTransition {
  return {
    id: input.id,
    fromStatus: "active",
    toStatus: input.toStatus,
    correctedBy: input.correctedBy,
    supersededAt: input.supersededAt,
  };
}

describe("findUnsettledSharedStateReconciliation", () => {
  it("does not flag durable commitment canonicalizations as unsettled", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const clock = new FixedClock(1_000);
    const commitmentRepository = new CommitmentRepository({
      db,
      clock,
    });

    try {
      const commitment = addCommitment(commitmentRepository, {
        type: PREFERENCE_COMMITMENT_TYPE,
        directiveFamily: "work update style",
        directive: "Prefer concise work updates.",
        createdAt: 500,
      });
      const entry = lockedEntry({
        canonicalizes: {
          goal_ids: [],
          commitment_ids: [commitment.id],
          action_ids: [],
          open_question_ids: [],
        },
      });

      const unsettledReconciliation = findUnsettledSharedStateReconciliation({
        previousArtifact: sharedStateArtifact([entry]),
        repositories: {
          commitmentRepository,
        },
        nowMs: clock.now(),
      });

      expect(unsettledReconciliation).toBeNull();
    } finally {
      db.close();
    }
  });
});

describe("reconcileSemanticBeliefRevision", () => {
  it("keeps active candidate nodes when the judge returns keep", async () => {
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const deps = semanticRevisionDependencies({
      verdict: "keep",
    });
    const trace = createTraceRecorder();

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: deps.dependencies,
      nowMs: 2_000,
      tracer: trace,
      turnId: "turn_semantic_keep",
    });

    expect(deps.searchByVector).toHaveBeenCalledTimes(1);
    expect(deps.complete).toHaveBeenCalledTimes(1);
    expect(deps.markSuperseded).not.toHaveBeenCalled();
    expect(deps.markContradicted).not.toHaveBeenCalled();
    expect(result).toMatchObject({
      semantic_nodes_reviewed_attempted: 1,
      semantic_nodes_marked_superseded: 0,
      semantic_nodes_marked_contradicted: 0,
    });
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.completed",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          candidates_enumerated: 1,
          kept_count: 1,
        }),
      }),
    );
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "llm_call.started",
        data: expect.objectContaining({
          label: "decision_artifact_semantic_revision",
        }),
      }),
    );
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "llm_call.completed",
        data: expect.objectContaining({
          label: "decision_artifact_semantic_revision",
          usage: {
            inputTokens: 5,
            outputTokens: 5,
          },
        }),
      }),
    );
  });

  it("reuses cached keep verdicts without calling the judge again for unchanged entry and candidate status", async () => {
    const cache = new SemanticRevisionVerdictCache();
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const deps = semanticRevisionDependencies({
      verdict: "keep",
      verdictCache: cache,
    });
    const firstTrace = createTraceRecorder();
    const secondTrace = createTraceRecorder();

    await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: deps.dependencies,
      nowMs: 2_000,
      tracer: firstTrace,
      turnId: "turn_semantic_cache_seed",
      turnCounter: 1,
    });

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: deps.dependencies,
      nowMs: 3_000,
      tracer: secondTrace,
      turnId: "turn_semantic_cache_hit",
      turnCounter: 2,
    });

    expect(deps.complete).toHaveBeenCalledTimes(1);
    expect(result.semantic_nodes_reviewed_attempted).toBe(1);
    expect(secondTrace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.cache.completed",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          candidate_node_id: deps.node.id,
          cached_verdict: "keep",
          age_turns: 1,
        }),
      }),
    );
    expect(secondTrace.events.filter((event) => event.event === "llm_call.started")).toHaveLength(
      0,
    );
    expect(secondTrace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.completed",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          candidates_enumerated: 1,
          kept_count: 1,
        }),
      }),
    );
  });

  it("re-judges a cached pair when the artifact entry text changes", async () => {
    const cache = new SemanticRevisionVerdictCache();
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const updatedEntry = {
      ...entry,
      text: "Project runtime is Node 24",
    };
    const deps = semanticRevisionDependencies({
      verdict: "keep",
      verdictCache: cache,
    });

    await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: deps.dependencies,
      nowMs: 2_000,
      turnId: "turn_semantic_cache_text_seed",
      turnCounter: 1,
    });

    await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([updatedEntry]),
      operations: [semanticRevisionOperation(updatedEntry)],
      dependencies: deps.dependencies,
      nowMs: 3_000,
      turnId: "turn_semantic_cache_text_changed",
      turnCounter: 2,
    });

    expect(deps.complete).toHaveBeenCalledTimes(2);
  });

  it("re-judges a cached pair when the active candidate node is updated", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-semantic-revision-cache-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new ManualClock(2_000);
    const semanticNodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const cache = new SemanticRevisionVerdictCache();
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const episodeId = createEpisodeId();
    const nodeId = createSemanticNodeId();

    try {
      await semanticNodeRepository.insert({
        id: nodeId,
        kind: "proposition",
        label: "Project runtime is Node 20",
        description: "The project runtime is Node 20.",
        domain: null,
        aliases: [],
        confidence: 0.7,
        source_episode_ids: [episodeId],
        created_at: 2_000,
        updated_at: 2_000,
        last_verified_at: 2_000,
        embedding: Float32Array.from([1, 0, 0, 0]),
        archived: false,
        superseded_by: null,
        status: "active",
        corrected_by: null,
        superseded_at: null,
      });

      const complete = vi.fn(async () =>
        semanticRevisionResponse({
          verdicts: [
            {
              node_id: nodeId,
              verdict: "keep",
            },
          ],
        }),
      );
      const trace = createTraceRecorder();

      const dependencies = {
        semanticNodeRepository,
        episodicRepository: {
          getMany: vi.fn(async (ids: EpisodeId[]) =>
            ids.map((id) =>
              createEpisodeFixture({
                id,
                source_stream_ids: [createStreamEntryId()],
                shared: true,
              }),
            ),
          ),
        },
        embeddingClient: {
          embed: vi.fn(async () => Float32Array.from([1, 0, 0, 0])),
          embedBatch: vi.fn(),
        },
        llmClient: {
          complete,
          converse: vi.fn(),
        },
        model: "semantic-revision-test",
        verdictCache: cache,
      };

      await reconcileSemanticBeliefRevision({
        artifact: sharedStateArtifact([entry]),
        operations: [semanticRevisionOperation(entry)],
        dependencies,
        nowMs: 2_000,
        turnId: "turn_semantic_cache_node_update_seed",
        turnCounter: 1,
      });

      clock.set(3_000);
      const updatedNode = await semanticNodeRepository.update(nodeId, {
        description: "The project runtime is Node 20 with revised support context.",
      });

      expect(updatedNode?.status).toBe("active");
      expect(updatedNode?.updated_at).toBe(3_000);

      await reconcileSemanticBeliefRevision({
        artifact: sharedStateArtifact([entry]),
        operations: [semanticRevisionOperation(entry)],
        dependencies,
        nowMs: 3_000,
        tracer: trace,
        turnId: "turn_semantic_cache_node_updated",
        turnCounter: 2,
      });

      expect(complete).toHaveBeenCalledTimes(2);
      expect(
        trace.events.filter(
          (event) => event.event === "semantic_revision.cache.completed",
        ),
      ).toHaveLength(0);
    } finally {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("filters a cached candidate that is no longer active before cache reuse", async () => {
    const cache = new SemanticRevisionVerdictCache();
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const episodeId = createEpisodeId();
    const nodeId = createSemanticNodeId();
    const activeNode = createSemanticNodeFixture(
      {
        id: nodeId,
        label: "Project runtime is Node 20",
        description: "The project runtime is Node 20.",
        source_episode_ids: [episodeId],
      },
      [1, 0, 0, 0],
    );
    const supersededNode = {
      ...activeNode,
      status: "superseded" as const,
    };
    let currentNode = activeNode;
    const searchByVector = vi.fn(async () => [{ node: currentNode, similarity: 0.95 }]);
    const complete = vi.fn(async () =>
      semanticRevisionResponse({
        verdicts: [
          {
            node_id: nodeId,
            verdict: "keep",
          },
        ],
      }),
    );
    const trace = createTraceRecorder();

    const dependencies = {
      semanticNodeRepository: {
        searchByVector,
        markSuperseded: vi.fn(),
        markContradicted: vi.fn(),
      },
      episodicRepository: {
        getMany: vi.fn(async (ids: EpisodeId[]) =>
          ids.map((id) =>
            createEpisodeFixture({
              id,
              shared: true,
            }),
          ),
        ),
      },
      embeddingClient: {
        embed: vi.fn(async () => Float32Array.from([1, 0, 0, 0])),
        embedBatch: vi.fn(),
      },
      llmClient: {
        complete,
        converse: vi.fn(),
      },
      model: "semantic-revision-test",
      verdictCache: cache,
    };

    await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies,
      nowMs: 2_000,
      turnId: "turn_semantic_cache_status_seed",
      turnCounter: 1,
    });

    currentNode = supersededNode;

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies,
      nowMs: 3_000,
      tracer: trace,
      turnId: "turn_semantic_cache_status_changed",
      turnCounter: 2,
    });

    expect(complete).toHaveBeenCalledTimes(1);
    expect(result.semantic_nodes_reviewed_attempted).toBe(0);
    expect(
      trace.events.filter(
        (event) => event.event === "semantic_revision.cache.completed",
      ),
    ).toHaveLength(0);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.completed",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          candidates_enumerated: 0,
        }),
      }),
    );
  });

  it("evicts the oldest semantic revision verdict cache entry after the 1000-entry cap", () => {
    const cache = new SemanticRevisionVerdictCache();

    for (let index = 0; index < 1_001; index += 1) {
      cache.set({
        artifactEntryId: `artifact_entry_${index}`,
        candidateNodeId: `semantic_node_${index}`,
        value: {
          verdict: "keep",
          entry_text_hash: `hash_${index}`,
          candidate_status_at_review: "active",
          candidate_updated_at_at_review: index,
          last_reviewed_at_turn: index,
        },
      });
    }

    expect(cache.size).toBe(1_000);
    expect(
      cache.get({
        artifactEntryId: "artifact_entry_0",
        candidateNodeId: "semantic_node_0",
      }),
    ).toBeNull();
    expect(
      cache.get({
        artifactEntryId: "artifact_entry_1",
        candidateNodeId: "semantic_node_1",
      }),
    ).toMatchObject({
      verdict: "keep",
      entry_text_hash: "hash_1",
    });
  });

  it("succeeds without calling the judge when embedding search returns no candidates", async () => {
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const deps = semanticRevisionDependencies({
      candidateCount: 0,
    });
    const trace = createTraceRecorder();

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: deps.dependencies,
      nowMs: 2_000,
      tracer: trace,
      turnId: "turn_semantic_empty",
    });

    expect(deps.searchByVector).toHaveBeenCalledTimes(1);
    expect(deps.complete).not.toHaveBeenCalled();
    expect(result.semantic_nodes_reviewed_attempted).toBe(0);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.completed",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          candidates_enumerated: 0,
        }),
      }),
    );
  });

  it("degrades without marking nodes when embedding search fails", async () => {
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const deps = semanticRevisionDependencies({
      searchError: new Error("vector index unavailable"),
    });
    const trace = createTraceRecorder();

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: deps.dependencies,
      nowMs: 2_000,
      tracer: trace,
      turnId: "turn_semantic_search_error",
    });

    expect(deps.complete).not.toHaveBeenCalled();
    expect(deps.markSuperseded).not.toHaveBeenCalled();
    expect(result.semantic_nodes_marked_superseded).toBe(0);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.degraded",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          reason: "vector index unavailable",
        }),
      }),
    );
  });

  it("degrades without marking nodes when the LLM judge fails", async () => {
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const deps = semanticRevisionDependencies({
      llmError: new Error("judge unavailable"),
    });
    const trace = createTraceRecorder();

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: deps.dependencies,
      nowMs: 2_000,
      tracer: trace,
      turnId: "turn_semantic_llm_error",
    });

    expect(deps.searchByVector).toHaveBeenCalledTimes(1);
    expect(deps.markSuperseded).not.toHaveBeenCalled();
    expect(result.semantic_nodes_marked_superseded).toBe(0);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.degraded",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          reason: "judge unavailable",
        }),
      }),
    );
  });

  it("does not process locked entries with contaminated provenance", async () => {
    const source = createStreamEntryId();
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
      provenance_stream_entry_ids: [source],
      last_updated_stream_entry_ids: [source],
    });
    const deps = semanticRevisionDependencies({
      verdict: "supersede",
    });

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: deps.dependencies,
      nowMs: 2_000,
      sourceTrustValidator: () => ({ allowed: false, reason: "quarantined" }),
      turnId: "turn_semantic_contaminated",
    });

    expect(deps.searchByVector).not.toHaveBeenCalled();
    expect(deps.complete).not.toHaveBeenCalled();
    expect(deps.markSuperseded).not.toHaveBeenCalled();
    expect(result.semantic_nodes_reviewed_attempted).toBe(0);
  });

  it("overfetches vector candidates before filtering active and audience visibility", async () => {
    const audience = createEntityId();
    const otherAudience = createEntityId();
    const visibleEpisodeId = createEpisodeId();
    const hiddenEpisodeId = createEpisodeId();
    const entry = lockedEntry({
      audience_entity_id: audience,
      text: "Project runtime is Node 22",
    });
    const inactiveNode = createSemanticNodeFixture({
      label: "Project runtime is Node 18",
      description: "The project runtime is Node 18.",
      source_episode_ids: [visibleEpisodeId],
      status: "superseded",
    });
    const crossAudienceNode = createSemanticNodeFixture({
      label: "Project runtime is Node 19",
      description: "The project runtime is Node 19.",
      source_episode_ids: [hiddenEpisodeId],
    });
    const archivedNode = createSemanticNodeFixture({
      label: "Project runtime is Node 21",
      description: "The project runtime is Node 21.",
      source_episode_ids: [visibleEpisodeId],
      archived: true,
    });
    const targetNode = createSemanticNodeFixture({
      label: "Project runtime is Node 20",
      description: "The project runtime is Node 20.",
      source_episode_ids: [visibleEpisodeId],
    });
    const extraNode = createSemanticNodeFixture({
      label: "Project runtime was planned for Node 20",
      description: "A prior plan mentioned Node 20.",
      source_episode_ids: [visibleEpisodeId],
    });
    const candidates: SemanticNodeSearchCandidate[] = [
      inactiveNode,
      crossAudienceNode,
      archivedNode,
      targetNode,
      extraNode,
    ].map((node, index) => ({
      node,
      similarity: 0.99 - index * 0.01,
    }));
    const searchByVector = vi.fn(async (_embedding: Float32Array, options: { limit?: number }) =>
      candidates.slice(0, options.limit ?? candidates.length),
    );
    const markSuperseded = vi.fn(
      async (id: SemanticNodeId, correctedBy: SemanticNodeCorrectionRef, supersededAt: number) =>
        semanticStatusTransition({
          id,
          toStatus: "superseded",
          correctedBy,
          supersededAt,
        }),
    );
    const complete = vi.fn(async () =>
      semanticRevisionResponse({
        verdicts: [
          {
            node_id: targetNode.id,
            verdict: "supersede",
          },
        ],
      }),
    );
    const trace = createTraceRecorder();

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: {
        semanticNodeRepository: {
          searchByVector,
          markSuperseded,
          markContradicted: vi.fn(),
        },
        episodicRepository: {
          getMany: vi.fn(async (ids: EpisodeId[]) =>
            ids.map((id) =>
              createEpisodeFixture({
                id,
                audience_entity_id: id === hiddenEpisodeId ? otherAudience : null,
                shared: id === hiddenEpisodeId ? false : true,
              }),
            ),
          ),
        },
        embeddingClient: {
          embed: vi.fn(async () => Float32Array.from([1, 0, 0, 0])),
          embedBatch: vi.fn(),
        },
        llmClient: {
          complete,
          converse: vi.fn(),
        },
        model: "semantic-revision-test",
        candidateLimit: 3,
      },
      nowMs: 2_000,
      tracer: trace,
      turnId: "turn_semantic_overfetch",
    });

    expect(searchByVector).toHaveBeenCalledWith(
      expect.any(Float32Array),
      expect.objectContaining({ limit: 9 }),
    );
    expect(complete).toHaveBeenCalledTimes(1);
    expect(markSuperseded).toHaveBeenCalledWith(
      targetNode.id,
      entry.last_updated_stream_entry_ids[0],
      2_000,
    );
    expect(result.semantic_nodes_reviewed_attempted).toBe(2);
    expect(result.semantic_nodes_marked_superseded).toBe(1);
  });

  it("does not revise semantic nodes extracted from the same source stream as the artifact entry", async () => {
    const source = createStreamEntryId();
    const episodeId = createEpisodeId();
    const node = createSemanticNodeFixture({
      label: "Project runtime is Node 20",
      description: "The project runtime is Node 20.",
      source_episode_ids: [episodeId],
    });
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
      provenance_stream_entry_ids: [source],
      last_updated_stream_entry_ids: [source],
    });
    const searchByVector = vi.fn(async () => [{ node, similarity: 0.98 }]);
    const markSuperseded = vi.fn(
      async (id: SemanticNodeId, correctedBy: SemanticNodeCorrectionRef, supersededAt: number) =>
        semanticStatusTransition({
          id,
          toStatus: "superseded",
          correctedBy,
          supersededAt,
        }),
    );
    const complete = vi.fn(async () =>
      semanticRevisionResponse({
        verdicts: [
          {
            node_id: node.id,
            verdict: "supersede",
          },
        ],
      }),
    );
    const trace = createTraceRecorder();

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: {
        semanticNodeRepository: {
          searchByVector,
          markSuperseded,
          markContradicted: vi.fn(),
        },
        episodicRepository: {
          getMany: vi.fn(async () => [
            createEpisodeFixture({
              id: episodeId,
              source_stream_ids: [source],
              shared: true,
            }),
          ]),
        },
        embeddingClient: {
          embed: vi.fn(async () => Float32Array.from([1, 0, 0, 0])),
          embedBatch: vi.fn(),
        },
        llmClient: {
          complete,
          converse: vi.fn(),
        },
        model: "semantic-revision-test",
      },
      nowMs: 2_000,
      tracer: trace,
      turnId: "turn_semantic_same_source",
    });

    expect(complete).not.toHaveBeenCalled();
    expect(markSuperseded).not.toHaveBeenCalled();
    expect(result.semantic_nodes_reviewed_attempted).toBe(0);
    expect(result.semantic_nodes_marked_superseded).toBe(0);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.completed",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          candidates_enumerated: 0,
        }),
      }),
    );
  });

  it("counts missing nodes from mark calls as skipped while processing the remaining verdicts", async () => {
    const episodeId = createEpisodeId();
    const entry = lockedEntry({
      text: "Project runtime is Node 22",
    });
    const nodes = Array.from({ length: 3 }, (_, index) =>
      createSemanticNodeFixture({
        label: `Project runtime stale candidate ${index}`,
        description: `The project runtime stale candidate ${index}.`,
        source_episode_ids: [episodeId],
      }),
    );
    const candidates = nodes.map((node) => ({ node, similarity: 0.97 }));
    const searchByVector = vi.fn(async () => candidates);
    const markSuperseded = vi.fn(
      async (id: SemanticNodeId, correctedBy: SemanticNodeCorrectionRef, supersededAt: number) =>
        id === nodes[0]?.id
          ? null
          : semanticStatusTransition({
              id,
              toStatus: "superseded",
              correctedBy,
              supersededAt,
            }),
    );
    const complete = vi.fn(async () =>
      semanticRevisionResponse({
        verdicts: nodes.map((node) => ({
          node_id: node.id,
          verdict: "supersede",
        })),
      }),
    );
    const trace = createTraceRecorder();

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact([entry]),
      operations: [semanticRevisionOperation(entry)],
      dependencies: {
        semanticNodeRepository: {
          searchByVector,
          markSuperseded,
          markContradicted: vi.fn(),
        },
        episodicRepository: {
          getMany: vi.fn(async (ids: EpisodeId[]) =>
            ids.map((id) =>
              createEpisodeFixture({
                id,
                shared: true,
              }),
            ),
          ),
        },
        embeddingClient: {
          embed: vi.fn(async () => Float32Array.from([1, 0, 0, 0])),
          embedBatch: vi.fn(),
        },
        llmClient: {
          complete,
          converse: vi.fn(),
        },
        model: "semantic-revision-test",
      },
      nowMs: 2_000,
      tracer: trace,
      turnId: "turn_semantic_node_missing",
    });

    expect(markSuperseded).toHaveBeenCalledTimes(3);
    expect(result.semantic_nodes_marked_superseded).toBe(2);
    expect(result.semantic_nodes_skipped).toBe(1);
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.degraded",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          node_id: nodes[0]?.id,
          verdict: "supersede",
          reason: "node_missing",
        }),
      }),
    );
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "semantic_revision.completed",
        data: expect.objectContaining({
          artifact_entry_id: entry.id,
          superseded_count: 2,
          skipped_count: 1,
          skipped_count_by_reason: expect.objectContaining({
            node_missing: 1,
          }),
        }),
      }),
    );
  });

  it("caps semantic revision to the first three locked artifact entries per turn", async () => {
    const entries = Array.from({ length: 7 }, (_, index) =>
      lockedEntry({
        text: `Workstream runtime decision ${index} is locked`,
        rank: index,
      }),
    );
    const searchByVector = vi.fn(async () => []);
    const trace = createTraceRecorder();

    const result = await reconcileSemanticBeliefRevision({
      artifact: sharedStateArtifact(entries),
      operations: entries.map((entry) => semanticRevisionOperation(entry)),
      dependencies: {
        semanticNodeRepository: {
          searchByVector,
          markSuperseded: vi.fn(),
          markContradicted: vi.fn(),
        },
        episodicRepository: {
          getMany: vi.fn(),
        },
        embeddingClient: {
          embed: vi.fn(async () => Float32Array.from([1, 0, 0, 0])),
          embedBatch: vi.fn(),
        },
        llmClient: {
          complete: vi.fn(),
          converse: vi.fn(),
        },
        model: "semantic-revision-test",
      },
      nowMs: 2_000,
      tracer: trace,
      turnId: "turn_semantic_entry_cap",
    });

    expect(searchByVector).toHaveBeenCalledTimes(3);
    expect(result.semantic_nodes_reviewed_attempted).toBe(0);
    expect(
      trace.events.filter(
        (event) => event.event === "semantic_revision.completed",
      ),
    ).toHaveLength(3);
    expect(
      trace.events.filter(
        (event) =>
          event.event === "semantic_revision.degraded" &&
          event.data.reason === "skipped_over_cap",
      ),
    ).toEqual([
      expect.objectContaining({
        data: expect.objectContaining({
          artifact_entry_id: entries[3]?.id,
          reason: "skipped_over_cap",
        }),
      }),
      expect.objectContaining({
        data: expect.objectContaining({
          artifact_entry_id: entries[4]?.id,
          reason: "skipped_over_cap",
        }),
      }),
      expect.objectContaining({
        data: expect.objectContaining({
          artifact_entry_id: entries[5]?.id,
          reason: "skipped_over_cap",
        }),
      }),
      expect.objectContaining({
        data: expect.objectContaining({
          artifact_entry_id: entries[6]?.id,
          reason: "skipped_over_cap",
        }),
      }),
    ]);
  });
});

describe("reconcileSharedStateCanonicalizations", () => {
  it("skips contaminated locked entries and traces the skipped canonicalization", () => {
    const quarantinedSource = createStreamEntryId();
    const goalId = createGoalId();
    const commitmentId = createCommitmentId();
    const actionId = createActionId();
    const openQuestionId = createOpenQuestionId();
    const entry = lockedEntry({
      provenance_stream_entry_ids: [quarantinedSource],
      last_updated_stream_entry_ids: [quarantinedSource],
      canonicalizes: {
        goal_ids: [goalId],
        commitment_ids: [commitmentId],
        action_ids: [actionId],
        open_question_ids: [openQuestionId],
      },
    });
    const goalsRepository = {
      updateStatus: vi.fn(),
    };
    const commitmentRepository = {
      get: vi.fn(),
      revoke: vi.fn(),
    };
    const actionRepository = {
      get: vi.fn(),
      update: vi.fn(),
    };
    const openQuestionsRepository = {
      get: vi.fn(),
      resolve: vi.fn(),
    };
    const trace = createTraceRecorder();

    const result = reconcileSharedStateCanonicalizations({
      entries: [entry],
      repositories: {
        goalsRepository,
        commitmentRepository,
        actionRepository,
        openQuestionsRepository,
      },
      sourceTrustValidator: (streamEntryId) =>
        streamEntryId === quarantinedSource
          ? { allowed: false, reason: "quarantined" }
          : { allowed: true },
      tracer: trace,
      turnId: "turn_reconcile_trust",
    });
    const skipEvents = trace.events.filter(
      (event) => event.event === "shared_state.reconcile.skipped",
    );

    expect(goalsRepository.updateStatus).not.toHaveBeenCalled();
    expect(commitmentRepository.get).not.toHaveBeenCalled();
    expect(commitmentRepository.revoke).not.toHaveBeenCalled();
    expect(actionRepository.get).not.toHaveBeenCalled();
    expect(actionRepository.update).not.toHaveBeenCalled();
    expect(openQuestionsRepository.get).not.toHaveBeenCalled();
    expect(openQuestionsRepository.resolve).not.toHaveBeenCalled();
    expect(result).toMatchObject({
      goals_retired: 0,
      commitments_retired: 0,
      actions_retired: 0,
      open_questions_retired: 0,
      goals_canonicalized_attempted: 1,
      goals_canonicalized_succeeded: 0,
      goals_canonicalized_skipped: 1,
      commitments_revoked_attempted: 1,
      commitments_revoked_succeeded: 0,
      commitments_revoked_skipped: 1,
      actions_completed_attempted: 1,
      actions_completed_succeeded: 0,
      actions_completed_skipped: 1,
      open_questions_resolved_attempted: 1,
      open_questions_resolved_succeeded: 0,
      open_questions_resolved_skipped: 1,
      errors: [],
    });
    expect(skipEvents).toHaveLength(1);
    expect(skipEvents).toContainEqual(
      expect.objectContaining({
        event: "shared_state.reconcile.skipped",
        data: expect.objectContaining({
          turnId: "turn_reconcile_trust",
          artifact_entry_id: entry.id,
          contaminated_source_id_count: 1,
          quarantined_source_id_count: 1,
          inactive_source_id_count: 0,
        }),
      }),
    );
  });

  it("retires canonicalized state through existing repository APIs", () => {
    const goalId = createGoalId();
    const commitmentId = createCommitmentId();
    const actionId = createActionId();
    const openQuestionId = createOpenQuestionId();
    const entry = lockedEntry({
      canonicalizes: {
        goal_ids: [goalId],
        commitment_ids: [commitmentId],
        action_ids: [actionId],
        open_question_ids: [openQuestionId],
      },
    });
    const goalsRepository = {
      updateStatus: vi.fn(),
    };
    const commitmentRepository = {
      get: vi.fn(
        () =>
          ({
            id: commitmentId,
            type: PROMISE_COMMITMENT_TYPE,
            revoked_at: null,
            expired_at: null,
            expires_at: null,
            superseded_by: null,
          }) as never,
      ),
      revoke: vi.fn(() => ({ id: commitmentId }) as never),
    };
    const actionRepository = {
      update: vi.fn(),
    };
    const openQuestionsRepository = {
      resolve: vi.fn(),
    };

    const result = reconcileSharedStateCanonicalizations({
      entries: [entry],
      repositories: {
        goalsRepository,
        commitmentRepository,
        actionRepository,
        openQuestionsRepository,
      },
    });

    expect(result).toMatchObject({
      goals_retired: 1,
      commitments_retired: 1,
      actions_retired: 1,
      open_questions_retired: 1,
      errors: [],
    });
    expect(goalsRepository.updateStatus).toHaveBeenCalledWith(
      goalId,
      "done",
      {
        kind: "online",
        process: "decision_artifact_reconciliation",
      },
      {
        canonicalizedByArtifactEntryId: entry.id,
      },
    );
    expect(commitmentRepository.revoke).toHaveBeenCalledWith(
      commitmentId,
      `canonicalized_by_artifact_entry_id=${entry.id}`,
      {
        kind: "online",
        process: "decision_artifact_reconciliation",
      },
      undefined,
      {
        canonicalizedByArtifactEntryId: entry.id,
      },
    );
    expect(actionRepository.update).toHaveBeenCalledWith(
      actionId,
      {
        state: "completed",
        canonicalized_by_artifact_entry_id: entry.id,
      },
      {
        skipSideEffects: true,
      },
    );
    expect(openQuestionsRepository.resolve).toHaveBeenCalledWith(
      openQuestionId,
      {
        resolution_evidence_stream_entry_ids: entry.last_updated_stream_entry_ids,
        resolution_note: `resolved_by_artifact_entry_id=${entry.id}`,
      },
      {
        resolvedByArtifactEntryId: entry.id,
      },
    );
  });

  it("suppresses action completion side effects when explicitly canonicalizing linked open questions", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(selfMigrations, actionMigrations),
    });
    const clock = new FixedClock(10_000);
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    let actionCompletionHookCalls = 0;
    const actionRepository = new ActionRepository({
      db,
      clock,
      onCompleted: (record) => {
        actionCompletionHookCalls += 1;
        if (record.open_question_id !== null) {
          openQuestionsRepository.resolve(record.open_question_id, {
            resolution_evidence_stream_entry_ids: record.provenance_stream_entry_ids,
            resolution_note: "resolved by action hook",
          });
        }
      },
    });

    try {
      const source = createStreamEntryId();
      const question = openQuestionsRepository.add({
        question: "Is Granada locked?",
        urgency: 0.6,
        provenance: {
          kind: "system",
        },
        source: "reflection",
      });
      const actionId = createActionId();
      actionRepository.add({
        id: actionId,
        description: "Track Granada decision",
        actor: "borg",
        audience_entity_id: null,
        goal_id: null,
        open_question_id: question.id,
        state: "committed_to_do",
        confidence: 0.9,
        provenance_episode_ids: [],
        provenance_stream_entry_ids: [source],
        created_at: clock.now(),
        updated_at: clock.now(),
        considering_at: null,
        committed_at: clock.now(),
        scheduled_at: null,
        completed_at: null,
        not_done_at: null,
        unknown_at: null,
        canonicalized_by_artifact_entry_id: null,
      });
      const entry = lockedEntry({
        last_updated_stream_entry_ids: [source],
        canonicalizes: {
          goal_ids: [],
          commitment_ids: [],
          action_ids: [actionId],
          open_question_ids: [question.id],
        },
      });

      const result = reconcileSharedStateCanonicalizations({
        entries: [entry],
        repositories: {
          actionRepository,
          openQuestionsRepository,
        },
      });

      expect(result).toMatchObject({
        actions_retired: 1,
        open_questions_retired: 1,
        errors: [],
      });
      expect(actionCompletionHookCalls).toBe(0);
      expect(openQuestionsRepository.get(question.id)).toMatchObject({
        status: "resolved",
        resolution_note: `resolved_by_artifact_entry_id=${entry.id}`,
        resolved_by_artifact_entry_id: entry.id,
      });
    } finally {
      db.close();
    }
  });

  it("reports a missing canonicalized commitment as a reconciliation error", () => {
    const commitmentId = createCommitmentId();
    const entry = lockedEntry({
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [commitmentId],
        action_ids: [],
        open_question_ids: [],
      },
    });
    const commitmentRepository = {
      get: vi.fn(() => null),
      revoke: vi.fn(() => null),
    };

    const result = reconcileSharedStateCanonicalizations({
      entries: [entry],
      repositories: {
        commitmentRepository,
      },
    });

    expect(result.commitments_retired).toBe(0);
    expect(result.errors).toEqual([
      {
        channel: "commitment",
        id: commitmentId,
        artifactEntryId: entry.id,
        message: `Unknown commitment id: ${commitmentId}`,
      },
    ]);
  });

  it("counts already terminal action canonicalizations as skipped", () => {
    const actionId = createActionId();
    const entry = lockedEntry({
      canonicalizes: {
        goal_ids: [],
        commitment_ids: [],
        action_ids: [actionId],
        open_question_ids: [],
      },
    });
    const actionRepository = {
      get: vi.fn(() => ({ id: actionId, state: "completed" }) as never),
      update: vi.fn(),
    };

    const result = reconcileSharedStateCanonicalizations({
      entries: [entry],
      repositories: {
        actionRepository,
      },
    });

    expect(result).toMatchObject({
      actions_retired: 0,
      actions_completed_attempted: 1,
      actions_completed_succeeded: 0,
      actions_completed_skipped: 1,
      errors: [],
    });
    expect(actionRepository.update).not.toHaveBeenCalled();
  });

  it("skips unmaterialized expired commitment canonicalizations", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const clock = new FixedClock(1_000);
    const commitmentRepository = new CommitmentRepository({
      db,
      clock,
    });

    try {
      const expired = commitmentRepository.add({
        type: PROMISE_COMMITMENT_TYPE,
        directiveFamily: "expired artifact fixture",
        directive: "Use the expired artifact fixture.",
        priority: 5,
        provenance: { kind: "manual" },
        createdAt: 500,
        expiresAt: 900,
      });
      const revoke = vi.spyOn(commitmentRepository, "revoke");
      const entry = lockedEntry({
        canonicalizes: {
          goal_ids: [],
          commitment_ids: [expired.id],
          action_ids: [],
          open_question_ids: [],
        },
      });

      const result = reconcileSharedStateCanonicalizations({
        entries: [entry],
        repositories: {
          commitmentRepository,
        },
        nowMs: clock.now(),
      });

      expect(result).toMatchObject({
        commitments_retired: 0,
        commitments_revoked_attempted: 1,
        commitments_revoked_succeeded: 0,
        commitments_revoked_skipped: 1,
        errors: [],
      });
      expect(revoke).not.toHaveBeenCalled();
      expect(commitmentRepository.get(expired.id)).toMatchObject({
        expired_at: null,
        revoked_at: null,
      });
    } finally {
      db.close();
    }
  });

  it.each(NON_CANONICALIZABLE_COMMITMENT_TYPES)(
    "skips %s commitment canonicalizations without revoking",
    (type) => {
      const db = openDatabase(":memory:", {
        migrations: commitmentMigrations,
      });
      const clock = new FixedClock(1_000);
      const commitmentRepository = new CommitmentRepository({
        db,
        clock,
      });

      try {
        const commitment = addCommitment(commitmentRepository, {
          type,
          directiveFamily: `${type} work policy`,
          directive: "Keep the work policy active.",
          createdAt: 500,
        });
        const revoke = vi.spyOn(commitmentRepository, "revoke");
        const entry = lockedEntry({
          canonicalizes: {
            goal_ids: [],
            commitment_ids: [commitment.id],
            action_ids: [],
            open_question_ids: [],
          },
        });

        const result = reconcileSharedStateCanonicalizations({
          entries: [entry],
          repositories: {
            commitmentRepository,
          },
          nowMs: clock.now(),
        });

        expect(result).toMatchObject({
          commitments_retired: 0,
          commitments_revoked_attempted: 1,
          commitments_revoked_succeeded: 0,
          commitments_revoked_skipped: 1,
          errors: [],
          skipped_commitments: [
            {
              channel: "commitment",
              id: commitment.id,
              artifactEntryId: entry.id,
              reason: "non_canonicalizable_commitment_type",
              commitmentType: type,
            },
          ],
        });
        expect(revoke).not.toHaveBeenCalled();
        expect(commitmentRepository.get(commitment.id)).toMatchObject({
          revoked_at: null,
          canonicalized_by_artifact_entry_id: null,
        });
      } finally {
        db.close();
      }
    },
  );

  it.each(CANONICALIZABLE_COMMITMENT_TYPES)(
    "revokes %s commitment canonicalizations with artifact backref",
    (type) => {
      const db = openDatabase(":memory:", {
        migrations: commitmentMigrations,
      });
      const clock = new FixedClock(1_000);
      const commitmentRepository = new CommitmentRepository({
        db,
        clock,
      });

      try {
        const commitment = addCommitment(commitmentRepository, {
          type,
          directiveFamily: `${type} release decision`,
          directive: "Use the locked release decision.",
          createdAt: 500,
        });
        const revoke = vi.spyOn(commitmentRepository, "revoke");
        const entry = lockedEntry({
          canonicalizes: {
            goal_ids: [],
            commitment_ids: [commitment.id],
            action_ids: [],
            open_question_ids: [],
          },
        });

        const result = reconcileSharedStateCanonicalizations({
          entries: [entry],
          repositories: {
            commitmentRepository,
          },
          nowMs: clock.now(),
        });

        expect(result).toMatchObject({
          commitments_retired: 1,
          commitments_revoked_attempted: 1,
          commitments_revoked_succeeded: 1,
          commitments_revoked_skipped: 0,
          errors: [],
          skipped_commitments: [],
        });
        expect(revoke).toHaveBeenCalledWith(
          commitment.id,
          `canonicalized_by_artifact_entry_id=${entry.id}`,
          {
            kind: "online",
            process: "decision_artifact_reconciliation",
          },
          undefined,
          {
            canonicalizedByArtifactEntryId: entry.id,
          },
        );
        expect(commitmentRepository.get(commitment.id)).toMatchObject({
          revoked_at: clock.now(),
          canonicalized_by_artifact_entry_id: entry.id,
        });
      } finally {
        db.close();
      }
    },
  );

  it("ignores non-locked entries", () => {
    const goalId = createGoalId();
    const goalsRepository = {
      updateStatus: vi.fn(),
    };

    const result = reconcileSharedStateCanonicalizations({
      entries: [
        lockedEntry({
          kind: "live",
          canonicalizes: {
            goal_ids: [goalId],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        }),
      ],
      repositories: {
        goalsRepository,
      },
    });

    expect(result.goals_retired).toBe(0);
    expect(goalsRepository.updateStatus).not.toHaveBeenCalled();
  });

  it.todo(
    "retires stale plan-branch commitments when a replacement branch is operationalized: lock 5-city itinerary, pivot to 3-anchor route skipping a city, operationalize 3-anchor via booked Renfe legs canonicalized into the artifact, assert old 5-city commitments transition to superseded/revoked without explicit canonicalizes references",
  );
});
