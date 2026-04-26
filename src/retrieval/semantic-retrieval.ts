/* Semantic-band retrieval for label/vector lookup and graph walks. */
import { extractEntitiesHeuristically } from "../cognition/perception/entity-extractor.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { isEpisodeVisibleToAudience } from "../memory/episodic/index.js";
import type { EpisodicRepository } from "../memory/episodic/repository.js";
import type { Episode } from "../memory/episodic/types.js";
import type { SemanticGraph } from "../memory/semantic/graph.js";
import type { SemanticNodeRepository } from "../memory/semantic/repository.js";
import type { SemanticContext, SemanticNode, SemanticWalkStep } from "../memory/semantic/types.js";
import type { EntityId } from "../util/ids.js";

export type RetrievedSemanticNode = SemanticNode & {
  historical?: boolean;
};

export type RetrievedSemanticHit = {
  root_node_id: SemanticNode["id"];
  node: SemanticNode;
  edgePath: SemanticWalkStep["edgePath"];
};

export type RetrievedSemantic = SemanticContext & {
  as_of?: number | null;
  matched_node_ids: SemanticNode["id"][];
  matched_nodes: RetrievedSemanticNode[];
  support_hits: RetrievedSemanticHit[];
  contradiction_hits: RetrievedSemanticHit[];
  category_hits: RetrievedSemanticHit[];
};

export type SemanticRetrievalOptions = {
  audienceEntityId?: EntityId | null;
  crossAudience?: boolean;
  graphWalkDepth?: number;
  maxGraphNodes?: number;
  asOf?: number;
};

export type SemanticRetrievalDependencies = {
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository?: SemanticNodeRepository;
  semanticGraph?: SemanticGraph;
};

export type ResolvedSemanticRetrieval = {
  context: SemanticContext;
  contradictionPresent: boolean;
  matchedNodeIds: SemanticNode["id"][];
  matchedNodes: RetrievedSemanticNode[];
  supportHits: RetrievedSemanticHit[];
  contradictionHits: RetrievedSemanticHit[];
  categoryHits: RetrievedSemanticHit[];
  asOf?: number;
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

  const labels = [...extractEntitiesHeuristically(query), ...query.split(/[,\n]+/)]
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
  const matchedNodes: SemanticNode[] = [];

  for (const label of labels) {
    const matches = await semanticNodeRepository.findByLabelOrAlias(label, 3);
    matchedNodes.push(...matches);
  }

  if (matchedNodes.length === 0) {
    const queryVector = await embeddingClient.embed(query);
    const byVector = await semanticNodeRepository.searchByVector(queryVector, {
      limit: 3,
      includeArchived: false,
    });
    matchedNodes.push(...byVector.map((item) => item.node));
  }

  const matchedNodeCandidates = [...new Map(matchedNodes.map((node) => [node.id, node])).values()];
  const matchedNodeVisibility = await resolveVisibleEpisodeIds(
    episodicRepository,
    matchedNodeCandidates.flatMap((node) => node.source_episode_ids),
    options,
  );
  const uniqueNodes = new Map(
    matchedNodeCandidates
      .filter((node) => isSemanticNodeVisible(node, matchedNodeVisibility))
      .map((node) => [node.id, node] as const),
  );
  const supports = new Map<string, SemanticNode>();
  const contradicts = new Map<string, SemanticNode>();
  const categories = new Map<string, SemanticNode>();
  const walkDepth = options.graphWalkDepth ?? 2;
  const maxGraphNodes = options.maxGraphNodes ?? 16;
  const supportNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> = [];
  const contradictionNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> =
    [];
  const categoryNeighbors: Array<{ rootNodeId: SemanticNode["id"]; step: SemanticWalkStep }> = [];
  const supportHits: RetrievedSemanticHit[] = [];
  const contradictionHits: RetrievedSemanticHit[] = [];
  const categoryHits: RetrievedSemanticHit[] = [];

  for (const node of uniqueNodes.values()) {
    const walkedSupports = await semanticGraph.walk(node.id, {
      relations: ["supports"],
      direction: "out",
      depth: walkDepth,
      maxNodes: maxGraphNodes,
      asOf: options.asOf,
    });
    const walkedContradictions = await semanticGraph.walk(node.id, {
      relations: ["contradicts"],
      direction: "both",
      depth: walkDepth,
      maxNodes: maxGraphNodes,
      asOf: options.asOf,
    });
    const walkedCategories = await semanticGraph.walk(node.id, {
      relations: ["is_a"],
      direction: "out",
      depth: walkDepth,
      maxNodes: maxGraphNodes,
      asOf: options.asOf,
    });

    supportNeighbors.push(...walkedSupports.map((step) => ({ rootNodeId: node.id, step })));
    contradictionNeighbors.push(
      ...walkedContradictions.map((step) => ({ rootNodeId: node.id, step })),
    );
    categoryNeighbors.push(...walkedCategories.map((step) => ({ rootNodeId: node.id, step })));
  }

  const semanticVisibility = await resolveVisibleEpisodeIds(
    episodicRepository,
    [
      ...[...uniqueNodes.values()].flatMap((node) => node.source_episode_ids),
      ...supportNeighbors.flatMap(({ step }) => [
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

  const visibleMatchedNodes = await Promise.all(
    [...uniqueNodes.values()]
      .filter((node) => isSemanticNodeVisible(node, semanticVisibility))
      .map(async (node): Promise<RetrievedSemanticNode> => {
        if (await isHistoricalPropositionMatch(node, semanticGraph, options.asOf)) {
          return {
            ...node,
            historical: true,
          };
        }

        return node;
      }),
  );

  for (const item of supportNeighbors) {
    if (!isSemanticWalkStepVisible(item.step, semanticVisibility)) {
      continue;
    }

    supports.set(item.step.node.id, item.step.node);
    supportHits.push({
      root_node_id: item.rootNodeId,
      node: item.step.node,
      edgePath: item.step.edgePath,
    });
  }

  for (const item of contradictionNeighbors) {
    if (!isSemanticWalkStepVisible(item.step, semanticVisibility)) {
      continue;
    }

    contradicts.set(item.step.node.id, item.step.node);
    contradictionHits.push({
      root_node_id: item.rootNodeId,
      node: item.step.node,
      edgePath: item.step.edgePath,
    });
  }

  for (const item of categoryNeighbors) {
    if (!isSemanticWalkStepVisible(item.step, semanticVisibility)) {
      continue;
    }

    categories.set(item.step.node.id, item.step.node);
    categoryHits.push({
      root_node_id: item.rootNodeId,
      node: item.step.node,
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
    contradiction_hits: resolved.contradictionHits,
    category_hits: resolved.categoryHits,
  };
}
