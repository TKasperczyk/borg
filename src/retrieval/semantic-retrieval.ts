/* Semantic-band retrieval for label/vector lookup and graph walks. */
import type { EmbeddingClient } from "../embeddings/index.js";
import { isEpisodeVisibleToAudience } from "../memory/episodic/index.js";
import type { EpisodicRepository } from "../memory/episodic/repository.js";
import type { Episode } from "../memory/episodic/types.js";
import type { SemanticGraph } from "../memory/semantic/graph.js";
import type { SemanticNodeRepository } from "../memory/semantic/repository.js";
import type {
  BeliefRevisionTarget,
  OpenBeliefRevisionStatus,
  ReviewQueueRepository,
} from "../memory/semantic/review-queue.js";
import type { SemanticContext, SemanticNode, SemanticWalkStep } from "../memory/semantic/types.js";
import type { EntityId } from "../util/ids.js";

const DEFAULT_UNDER_REVIEW_MULTIPLIER = 0.5;
const DEFAULT_SEMANTIC_NODE_MIN_SIMILARITY = 0.01;

export type RetrievedSemanticUnderReview = {
  review_id: number;
  reason: string;
  reason_code: OpenBeliefRevisionStatus["reason_code"];
  invalidated_edge_id: string;
};

export type RetrievedSemanticNode = SemanticNode & {
  historical?: boolean;
  base_retrieval_score?: number;
  retrieval_score?: number;
  under_review?: RetrievedSemanticUnderReview;
};

export type RetrievedSemanticHit = {
  root_node_id: SemanticNode["id"];
  node: RetrievedSemanticNode;
  edgePath: SemanticWalkStep["edgePath"];
};

export type RetrievedSemantic = SemanticContext & {
  as_of?: number | null;
  matched_node_ids: SemanticNode["id"][];
  matched_nodes: RetrievedSemanticNode[];
  support_hits: RetrievedSemanticHit[];
  causal_hits: RetrievedSemanticHit[];
  contradiction_hits: RetrievedSemanticHit[];
  category_hits: RetrievedSemanticHit[];
};

export type SemanticRetrievalOptions = {
  audienceEntityId?: EntityId | null;
  crossAudience?: boolean;
  graphWalkDepth?: number;
  maxGraphNodes?: number;
  asOf?: number;
  underReviewMultiplier?: number;
  queryVector?: Float32Array;
  exactTerms?: readonly string[];
};

export type SemanticRetrievalDependencies = {
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository?: SemanticNodeRepository;
  semanticGraph?: SemanticGraph;
  reviewQueueRepository?: Pick<ReviewQueueRepository, "listOpenBeliefRevisionsByTarget">;
};

export type ResolvedSemanticRetrieval = {
  context: SemanticContext;
  contradictionPresent: boolean;
  matchedNodeIds: SemanticNode["id"][];
  matchedNodes: RetrievedSemanticNode[];
  supportHits: RetrievedSemanticHit[];
  causalHits: RetrievedSemanticHit[];
  contradictionHits: RetrievedSemanticHit[];
  categoryHits: RetrievedSemanticHit[];
  asOf?: number;
};

type MatchedNodeCandidate = {
  node: SemanticNode;
  baseScore: number;
};

function emptySemanticRetrieval(): ResolvedSemanticRetrieval {
  return {
    context: {
      supports: [],
      contradicts: [],
      categories: [],
    },
    contradictionPresent: false,
    matchedNodeIds: [],
    matchedNodes: [],
    supportHits: [],
    causalHits: [],
    contradictionHits: [],
    categoryHits: [],
  };
}

async function resolveVisibleEpisodeIds(
  episodicRepository: EpisodicRepository,
  episodeIds: readonly Episode["id"][],
  visibility: Pick<SemanticRetrievalOptions, "audienceEntityId" | "crossAudience">,
): Promise<Set<string> | null> {
  if (visibility.crossAudience === true) {
    return null;
  }

  const uniqueEpisodeIds = [...new Set(episodeIds)];

  if (uniqueEpisodeIds.length === 0) {
    return new Set<string>();
  }

  const episodes = await episodicRepository.getMany(uniqueEpisodeIds);

  return new Set(
    episodes
      .filter((episode) =>
        isEpisodeVisibleToAudience(episode, visibility.audienceEntityId, {
          crossAudience: visibility.crossAudience,
        }),
      )
      .map((episode) => episode.id),
  );
}

function isSemanticNodeVisible(
  node: SemanticNode,
  visibleEpisodeIds: ReadonlySet<string> | null,
): boolean {
  return (
    visibleEpisodeIds === null ||
    node.source_episode_ids.every((episodeId) => visibleEpisodeIds.has(episodeId))
  );
}

export async function isSemanticNodeVisibleToAudience(
  node: SemanticNode,
  visibility: Pick<SemanticRetrievalOptions, "audienceEntityId" | "crossAudience">,
  dependencies: Pick<SemanticRetrievalDependencies, "episodicRepository">,
): Promise<boolean> {
  const visibleEpisodeIds = await resolveVisibleEpisodeIds(
    dependencies.episodicRepository,
    node.source_episode_ids,
    visibility,
  );

  return isSemanticNodeVisible(node, visibleEpisodeIds);
}

function isSemanticWalkStepVisible(
  step: SemanticWalkStep,
  visibleEpisodeIds: ReadonlySet<string> | null,
): boolean {
  return (
    isSemanticNodeVisible(step.node, visibleEpisodeIds) &&
    (visibleEpisodeIds === null ||
      step.edgePath.every((edge) =>
        edge.evidence_episode_ids.every((episodeId) => visibleEpisodeIds.has(episodeId)),
      ))
  );
}

export async function filterSemanticWalkStepsByAudience(
  steps: readonly SemanticWalkStep[],
  visibility: Pick<SemanticRetrievalOptions, "audienceEntityId" | "crossAudience">,
  dependencies: Pick<SemanticRetrievalDependencies, "episodicRepository">,
): Promise<SemanticWalkStep[]> {
  const visibleEpisodeIds = await resolveVisibleEpisodeIds(
    dependencies.episodicRepository,
    steps.flatMap((step) => [
      ...step.node.source_episode_ids,
      ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
    ]),
    visibility,
  );

  return steps.filter((step) => isSemanticWalkStepVisible(step, visibleEpisodeIds));
}

function normalizeUnderReviewMultiplier(value: number | undefined): number {
  const multiplier = value ?? DEFAULT_UNDER_REVIEW_MULTIPLIER;

  if (!Number.isFinite(multiplier) || multiplier < 0 || multiplier > 1) {
    throw new TypeError("underReviewMultiplier must be between 0 and 1");
  }

  return multiplier;
}

function semanticNodeTargetKey(nodeId: SemanticNode["id"]): string {
  return JSON.stringify(["semantic_node", nodeId]);
}

function recordMatchedNode(
  candidatesById: Map<SemanticNode["id"], MatchedNodeCandidate>,
  node: SemanticNode,
  baseScore: number,
): void {
  const existing = candidatesById.get(node.id);

  if (existing === undefined || baseScore > existing.baseScore) {
    candidatesById.set(node.id, {
      node,
      baseScore,
    });
  }
}

function buildUnderReviewStatus(
  status: OpenBeliefRevisionStatus | undefined,
): RetrievedSemanticUnderReview | undefined {
  if (status === undefined) {
    return undefined;
  }

  return {
    review_id: status.review_id,
    reason: status.reason,
    reason_code: status.reason_code,
    invalidated_edge_id: status.invalidated_edge_id,
  };
}

function annotateSemanticNode(
  node: SemanticNode,
  input: {
    baseScore?: number;
    underReviewByNodeId: ReadonlyMap<string, OpenBeliefRevisionStatus>;
    underReviewMultiplier: number;
  },
): RetrievedSemanticNode {
  const status = buildUnderReviewStatus(
    input.underReviewByNodeId.get(semanticNodeTargetKey(node.id)),
  );

  if (input.baseScore === undefined) {
    return status === undefined
      ? node
      : {
          ...node,
          under_review: status,
        };
  }

  const retrievalScore =
    status === undefined ? input.baseScore : input.baseScore * input.underReviewMultiplier;

  return {
    ...node,
    base_retrieval_score: input.baseScore,
    retrieval_score: retrievalScore,
    ...(status === undefined ? {} : { under_review: status }),
  };
}

async function collectUnderReviewStatuses(
  nodes: readonly SemanticNode[],
  dependencies: SemanticRetrievalDependencies,
  options: Pick<SemanticRetrievalOptions, "audienceEntityId" | "crossAudience">,
): Promise<Map<string, OpenBeliefRevisionStatus>> {
  if (dependencies.reviewQueueRepository === undefined || nodes.length === 0) {
    return new Map();
  }

  const targets = [...new Set(nodes.map((node) => node.id))].map(
    (nodeId): BeliefRevisionTarget => ({
      target_type: "semantic_node",
      target_id: nodeId,
    }),
  );

  return dependencies.reviewQueueRepository.listOpenBeliefRevisionsByTarget(targets, {
    audienceEntityId: options.audienceEntityId,
    crossAudience: options.crossAudience,
    episodicRepository: dependencies.episodicRepository,
  });
}

async function isHistoricalPropositionMatch(
  node: SemanticNode,
  semanticGraph: SemanticGraph,
  asOf: number | undefined,
): Promise<boolean> {
  if (node.kind !== "proposition") {
    return false;
  }

  const supportNeighbors = await semanticGraph.neighbors(node.id, {
    relations: ["supports"],
    direction: "in",
    includeInvalid: true,
  });

  if (supportNeighbors.length === 0) {
    return false;
  }

  if (!supportNeighbors.some(({ edge }) => edge.valid_to !== null)) {
    return false;
  }

  if (asOf === undefined) {
    const currentSupportNeighbors = await semanticGraph.neighbors(node.id, {
      relations: ["supports"],
      direction: "in",
    });

    return (
      currentSupportNeighbors.length === 0 &&
      supportNeighbors.every(({ edge }) => edge.valid_to !== null)
    );
  }

  return supportNeighbors.every(({ edge }) => edge.valid_to !== null && edge.valid_to <= asOf);
}

export async function resolveSemanticContext(
  query: string,
  options: SemanticRetrievalOptions,
  dependencies: SemanticRetrievalDependencies,
): Promise<ResolvedSemanticRetrieval> {
  const { embeddingClient, episodicRepository, semanticNodeRepository, semanticGraph } =
    dependencies;

  if (semanticNodeRepository === undefined || semanticGraph === undefined) {
    return emptySemanticRetrieval();
  }

  const underReviewMultiplier = normalizeUnderReviewMultiplier(options.underReviewMultiplier);
  const matchedNodeCandidatesById = new Map<SemanticNode["id"], MatchedNodeCandidate>();

  const queryVector = options.queryVector ?? (await embeddingClient.embed(query));
  const byVector = await semanticNodeRepository.searchByVector(queryVector, {
    limit: 3,
    minSimilarity: DEFAULT_SEMANTIC_NODE_MIN_SIMILARITY,
    includeArchived: false,
  });

  for (const item of byVector) {
    recordMatchedNode(matchedNodeCandidatesById, item.node, item.similarity);
  }

  for (const term of options.exactTerms ?? []) {
    const exactMatches = await semanticNodeRepository.findByExactLabelOrAlias(term, 5, {
      includeArchived: false,
    });

    for (const node of exactMatches) {
      recordMatchedNode(matchedNodeCandidatesById, node, 1);
    }
  }

  const matchedNodeCandidates = [...matchedNodeCandidatesById.values()];
  const matchedNodeVisibility = await resolveVisibleEpisodeIds(
    episodicRepository,
    matchedNodeCandidates.flatMap(({ node }) => node.source_episode_ids),
    options,
  );
  const uniqueNodes = new Map(
    matchedNodeCandidates
      .filter(({ node }) => isSemanticNodeVisible(node, matchedNodeVisibility))
      .map((candidate) => [candidate.node.id, candidate] as const),
  );
  const supports = new Map<string, SemanticNode>();
  const contradicts = new Map<string, SemanticNode>();
  const categories = new Map<string, SemanticNode>();
  const walkDepth = options.graphWalkDepth ?? 2;
  const maxGraphNodes = options.maxGraphNodes ?? 16;
  const supportNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> = [];
  const causalNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> = [];
  const contradictionNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> =
    [];
  const categoryNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> = [];
  const supportHits: RetrievedSemanticHit[] = [];
  const causalHits: RetrievedSemanticHit[] = [];
  const contradictionHits: RetrievedSemanticHit[] = [];
  const categoryHits: RetrievedSemanticHit[] = [];

  for (const node of uniqueNodes.values()) {
    const walkedSupports = await semanticGraph.walk(node.node.id, {
      relations: ["supports"],
      direction: "out",
      depth: walkDepth,
      maxNodes: maxGraphNodes,
      asOf: options.asOf,
    });
    const walkedCausals = await semanticGraph.walk(node.node.id, {
      relations: ["causes", "prevents"],
      direction: "out",
      depth: walkDepth,
      maxNodes: maxGraphNodes,
      asOf: options.asOf,
    });
    const walkedContradictions = await semanticGraph.walk(node.node.id, {
      relations: ["contradicts"],
      direction: "both",
      depth: walkDepth,
      maxNodes: maxGraphNodes,
      asOf: options.asOf,
    });
    const walkedCategories = await semanticGraph.walk(node.node.id, {
      relations: ["is_a"],
      direction: "out",
      depth: walkDepth,
      maxNodes: maxGraphNodes,
      asOf: options.asOf,
    });

    supportNeighbors.push(...walkedSupports.map((step) => ({ rootNodeId: node.node.id, step })));
    causalNeighbors.push(...walkedCausals.map((step) => ({ rootNodeId: node.node.id, step })));
    contradictionNeighbors.push(
      ...walkedContradictions.map((step) => ({ rootNodeId: node.node.id, step })),
    );
    categoryNeighbors.push(...walkedCategories.map((step) => ({ rootNodeId: node.node.id, step })));
  }

  const semanticVisibility = await resolveVisibleEpisodeIds(
    episodicRepository,
    [
      ...[...uniqueNodes.values()].flatMap(({ node }) => node.source_episode_ids),
      ...supportNeighbors.flatMap(({ step }) => [
        ...step.node.source_episode_ids,
        ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
      ]),
      ...causalNeighbors.flatMap(({ step }) => [
        ...step.node.source_episode_ids,
        ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
      ]),
      ...contradictionNeighbors.flatMap(({ step }) => [
        ...step.node.source_episode_ids,
        ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
      ]),
      ...categoryNeighbors.flatMap(({ step }) => [
        ...step.node.source_episode_ids,
        ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
      ]),
    ],
    options,
  );

  const visibleSupportNeighbors = supportNeighbors.filter(({ step }) =>
    isSemanticWalkStepVisible(step, semanticVisibility),
  );
  const visibleCausalNeighbors = causalNeighbors.filter(({ step }) =>
    isSemanticWalkStepVisible(step, semanticVisibility),
  );
  const visibleContradictionNeighbors = contradictionNeighbors.filter(({ step }) =>
    isSemanticWalkStepVisible(step, semanticVisibility),
  );
  const visibleCategoryNeighbors = categoryNeighbors.filter(({ step }) =>
    isSemanticWalkStepVisible(step, semanticVisibility),
  );
  const underReviewByNodeId = await collectUnderReviewStatuses(
    [
      ...[...uniqueNodes.values()].map(({ node }) => node),
      ...visibleSupportNeighbors.map(({ step }) => step.node),
      ...visibleCausalNeighbors.map(({ step }) => step.node),
      ...visibleContradictionNeighbors.map(({ step }) => step.node),
      ...visibleCategoryNeighbors.map(({ step }) => step.node),
    ],
    dependencies,
    options,
  );
  const visibleMatchedNodes = await Promise.all(
    [...uniqueNodes.values()]
      .filter(({ node }) => isSemanticNodeVisible(node, semanticVisibility))
      .map(async (candidate): Promise<RetrievedSemanticNode> => {
        const annotated = annotateSemanticNode(candidate.node, {
          baseScore: candidate.baseScore,
          underReviewByNodeId,
          underReviewMultiplier,
        });

        if (await isHistoricalPropositionMatch(candidate.node, semanticGraph, options.asOf)) {
          return {
            ...annotated,
            historical: true,
          };
        }

        return annotated;
      }),
  );
  visibleMatchedNodes.sort(
    (left, right) =>
      (right.retrieval_score ?? 0) - (left.retrieval_score ?? 0) ||
      (right.base_retrieval_score ?? 0) - (left.base_retrieval_score ?? 0) ||
      right.updated_at - left.updated_at ||
      left.id.localeCompare(right.id),
  );

  for (const item of visibleSupportNeighbors) {
    const node = annotateSemanticNode(item.step.node, {
      underReviewByNodeId,
      underReviewMultiplier,
    });
    supports.set(item.step.node.id, node);
    supportHits.push({
      root_node_id: item.rootNodeId,
      node,
      edgePath: item.step.edgePath,
    });
  }

  for (const item of visibleCausalNeighbors) {
    causalHits.push({
      root_node_id: item.rootNodeId,
      node: annotateSemanticNode(item.step.node, {
        underReviewByNodeId,
        underReviewMultiplier,
      }),
      edgePath: item.step.edgePath,
    });
  }

  for (const item of visibleContradictionNeighbors) {
    const node = annotateSemanticNode(item.step.node, {
      underReviewByNodeId,
      underReviewMultiplier,
    });
    contradicts.set(item.step.node.id, node);
    contradictionHits.push({
      root_node_id: item.rootNodeId,
      node,
      edgePath: item.step.edgePath,
    });
  }

  for (const item of visibleCategoryNeighbors) {
    const node = annotateSemanticNode(item.step.node, {
      underReviewByNodeId,
      underReviewMultiplier,
    });
    categories.set(item.step.node.id, node);
    categoryHits.push({
      root_node_id: item.rootNodeId,
      node,
      edgePath: item.step.edgePath,
    });
  }

  return {
    context: {
      supports: [...supports.values()],
      contradicts: [...contradicts.values()],
      categories: [...categories.values()],
    },
    contradictionPresent: contradicts.size > 0,
    matchedNodeIds: visibleMatchedNodes.map((node) => node.id),
    matchedNodes: visibleMatchedNodes,
    supportHits,
    causalHits,
    contradictionHits,
    categoryHits,
    asOf: options.asOf,
  };
}

export function toRetrievedSemantic(resolved: ResolvedSemanticRetrieval): RetrievedSemantic {
  return {
    as_of: resolved.asOf ?? null,
    supports: resolved.context.supports,
    contradicts: resolved.context.contradicts,
    categories: resolved.context.categories,
    matched_node_ids: resolved.matchedNodeIds,
    matched_nodes: resolved.matchedNodes,
    support_hits: resolved.supportHits,
    causal_hits: resolved.causalHits,
    contradiction_hits: resolved.contradictionHits,
    category_hits: resolved.categoryHits,
  };
}
