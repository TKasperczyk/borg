import { vi } from "vitest";

import type {
  SemanticNode,
  SemanticNodeCorrectionRef,
  SemanticNodeSearchCandidate,
  SemanticNodeStatusTransition,
} from "../../memory/semantic/index.js";
import type { LLMCompleteResult } from "../../llm/index.js";
import { createEpisodeFixture, createSemanticNodeFixture } from "../../offline/test-support.js";
import {
  SemanticRevisionVerdictCache,
  type SharedStateSemanticBeliefRevisionDependencies,
} from "../../cognition/shared-state/reconciliation.js";
import {
  createEpisodeId,
  createSemanticNodeId,
  type EpisodeId,
  type SemanticNodeId,
} from "../../util/ids.js";

import { makeToolUseCompleteResult } from "./llm.js";

export type SemanticRevisionVerdict = "supersede" | "contradict" | "keep" | "uncertain";

export type SemanticRevisionDependenciesFixture = {
  node: SemanticNode;
  candidates: SemanticNodeSearchCandidate[];
  searchByVector: SharedStateSemanticBeliefRevisionDependencies["semanticNodeRepository"]["searchByVector"];
  markSuperseded: SharedStateSemanticBeliefRevisionDependencies["semanticNodeRepository"]["markSuperseded"];
  markContradicted: SharedStateSemanticBeliefRevisionDependencies["semanticNodeRepository"]["markContradicted"];
  complete: SharedStateSemanticBeliefRevisionDependencies["llmClient"]["complete"];
  dependencies: SharedStateSemanticBeliefRevisionDependencies;
};

export function makeSemanticRevisionResponse(input: {
  verdicts: Array<{
    node_id: SemanticNodeId;
    verdict: SemanticRevisionVerdict;
  }>;
  inputTokens?: number;
  outputTokens?: number;
}): LLMCompleteResult {
  return makeToolUseCompleteResult({
    toolId: "toolu_shared_state_semantic_revision",
    toolName: "EmitSharedStateSemanticRevision",
    toolInput: {
      verdicts: input.verdicts,
    },
    inputTokens: input.inputTokens ?? 5,
    outputTokens: input.outputTokens ?? 5,
  });
}

export function makeSemanticNodeStatusTransition(input: {
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

export function makeSemanticRevisionDependencies(
  input: {
    nodeId?: SemanticNodeId;
    verdict?: SemanticRevisionVerdict;
    searchError?: Error;
    llmError?: Error;
    candidateCount?: number;
    verdictCache?: SemanticRevisionVerdictCache;
  } = {},
): SemanticRevisionDependenciesFixture {
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
    async (id: SemanticNodeId, correctedBy: SemanticNodeCorrectionRef, supersededAt: number) =>
      makeSemanticNodeStatusTransition({
        id,
        toStatus: "superseded",
        correctedBy,
        supersededAt,
      }),
  );
  const markContradicted = vi.fn(
    async (id: SemanticNodeId, correctedBy: SemanticNodeCorrectionRef, supersededAt: number) =>
      makeSemanticNodeStatusTransition({
        id,
        toStatus: "contradicted",
        correctedBy,
        supersededAt,
      }),
  );
  const complete = vi.fn(async () => {
    if (input.llmError !== undefined) {
      throw input.llmError;
    }

    return makeSemanticRevisionResponse({
      verdicts: candidates.map((candidate) => ({
        node_id: candidate.node.id,
        verdict: input.verdict ?? "keep",
      })),
    });
  });
  const dependencies = {
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
  } satisfies SharedStateSemanticBeliefRevisionDependencies;

  return {
    node,
    candidates,
    searchByVector,
    markSuperseded,
    markContradicted,
    complete,
    dependencies,
  };
}
