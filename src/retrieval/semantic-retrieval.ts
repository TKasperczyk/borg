/* Semantic-band retrieval for label/vector lookup and graph walks. */
import type { EmbeddingClient } from "../embeddings/index.js";
import {
  isEpisodeVisibleToCapability,
  resolveViewerCapability,
} from "../memory/episodic/access.js";
import type { EpisodicRepository } from "../memory/episodic/repository.js";
import type { Episode } from "../memory/episodic/types.js";
import type { SemanticGraph } from "../memory/semantic/graph.js";
import type { SemanticNodeRepository } from "../memory/semantic/repository.js";
import type {
  BeliefRevisionTarget,
  OpenBeliefRevisionStatus,
  ReviewQueueRepository,
} from "../memory/review-queue/review-queue.js";
import type {
  SemanticContext,
  SemanticEdge,
  SemanticNode,
  SemanticNodeStatus,
  SemanticWalkStep,
} from "../memory/semantic/types.js";
import { SEMANTIC_NODE_STATUSES } from "../memory/semantic/types.js";
import type { EntityId } from "../util/ids.js";
import {
  combineDisclosureLabelForEpisodeIds,
  combineMemoryDisclosureLabels,
  resolveDisclosureLabelsByEpisodeId,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../memory/common/disclosure-label.js";

const DEFAULT_UNDER_REVIEW_MULTIPLIER = 0.5;
const DEFAULT_SEMANTIC_OVERFETCH_MULTIPLIER = 3;
const MAX_SEMANTIC_OVERFETCH_MULTIPLIER = 10;
const MAX_SEMANTIC_CANDIDATE_FETCH_LIMIT = 50;
const DEFAULT_VECTOR_MATCH_LIMIT = 3;
const DEFAULT_EXACT_MATCH_LIMIT = 5;
export const DEFAULT_SEMANTIC_STATUS_MULTIPLIERS = {
  active: 1,
  superseded: 0.5,
  contradicted: 0.3,
  quarantined: 0.2,
} as const satisfies Record<SemanticNodeStatus, number>;
export type SemanticStatusMultipliers = Record<SemanticNodeStatus, number>;
const DEFAULT_SEMANTIC_NODE_MIN_SIMILARITY = 0.01;

export type RetrievedSemanticUnderReview = {
  review_id: number;
  reason: string;
  reason_code: OpenBeliefRevisionStatus["reason_code"];
  invalidated_edge_id: string;
  disclosureLabel: MemoryDisclosureLabel;
};

export type RetrievedSemanticNode = SemanticNode & {
  historical?: boolean;
  base_retrieval_score?: number;
  retrieval_score?: number;
  status_retrieval_multiplier?: number;
  under_review?: RetrievedSemanticUnderReview;
  partial_source_visibility?: boolean;
  source_visibility_fraction?: number;
  disclosureLabel?: MemoryDisclosureLabel;
};

export type RetrievedSemanticEdge = SemanticEdge & {
  partial_source_visibility?: boolean;
  source_visibility_fraction?: number;
  disclosureLabel?: MemoryDisclosureLabel;
};

export type RetrievedSemanticHit = {
  root_node_id: SemanticNode["id"];
  node: RetrievedSemanticNode;
  edgePath: RetrievedSemanticEdge[];
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
  // Disclosure/admin-only all-audiences source search. Ignored for cognition source recall.
  crossAudience?: boolean;
  graphWalkDepth?: number;
  maxGraphNodes?: number;
  asOf?: number;
  underReviewMultiplier?: number;
  statusMultipliers?: Partial<SemanticStatusMultipliers>;
  overfetchMultiplier?: number;
  queryVector?: Float32Array;
  exactTerms?: readonly string[];
};

type SemanticVisibilityOptions = Pick<
  SemanticRetrievalOptions,
  "audienceEntityId" | "crossAudience"
>;

type SemanticSourceMode = "cognition" | "disclosure";

export type SemanticSourceAdapter = {
  admittedSourceEpisodeIds: Episode["id"][];
  partial: boolean;
  sourceVisibilityFraction?: number;
  disclosureLabel: MemoryDisclosureLabel;
};

export type SemanticRetrievalDependencies = {
  embeddingClient: EmbeddingClient;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository?: SemanticNodeRepository;
  semanticGraph?: SemanticGraph;
  reviewQueueRepository?: Pick<
    ReviewQueueRepository,
    "listOpenBeliefRevisionsByTarget" | "listOpenBeliefRevisionsByTargetForCognition"
  >;
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

type SemanticNodeSourcesForDisclosure = {
  availableSourceEpisodeIds: SemanticNode["source_episode_ids"];
  partial: boolean;
};

type SemanticEdgeSourcesForDisclosure = {
  availableEvidenceEpisodeIds: SemanticEdge["evidence_episode_ids"];
  partial: boolean;
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

async function resolveSemanticSourceEpisodeIdsForDisclosure(
  episodicRepository: EpisodicRepository,
  episodeIds: readonly Episode["id"][],
  visibility: SemanticVisibilityOptions,
): Promise<Set<string> | null> {
  const viewer = resolveViewerCapability(visibility);

  if (viewer.kind === "unrestricted") {
    return null;
  }

  const uniqueEpisodeIds = [...new Set(episodeIds)];

  if (uniqueEpisodeIds.length === 0) {
    return new Set<string>();
  }

  const episodes = await episodicRepository.getMany(uniqueEpisodeIds);

  return new Set(
    episodes
      .filter((episode) => isEpisodeVisibleToCapability(episode, viewer))
      .map((episode) => episode.id),
  );
}

async function resolveEpisodeDisclosureLabels(
  episodicRepository: EpisodicRepository,
  episodeIds: readonly Episode["id"][],
): Promise<Map<string, MemoryDisclosureLabel>> {
  return new Map<string, MemoryDisclosureLabel>(
    await resolveDisclosureLabelsByEpisodeId(episodeIds, (ids) => episodicRepository.getMany(ids)),
  );
}

function disclosureLabelForEpisodeIds(
  episodeIds: readonly Episode["id"][],
  labelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>,
): MemoryDisclosureLabel {
  return combineMemoryDisclosureLabels(
    episodeIds.map(
      (episodeId) => labelsByEpisodeId.get(episodeId) ?? unknownMemoryDisclosureLabel(),
    ),
  );
}

function adaptSemanticSourceEpisodes(input: {
  sourceEpisodeIds: readonly Episode["id"][];
  admittedSourceEpisodeIds: ReadonlySet<string> | null;
  labelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>;
}): SemanticSourceAdapter {
  const admittedSourceEpisodeIds =
    input.admittedSourceEpisodeIds === null
      ? [...input.sourceEpisodeIds]
      : input.sourceEpisodeIds.filter((episodeId) =>
          input.admittedSourceEpisodeIds?.has(episodeId),
        );
  const partial =
    input.admittedSourceEpisodeIds !== null &&
    admittedSourceEpisodeIds.length > 0 &&
    admittedSourceEpisodeIds.length < input.sourceEpisodeIds.length;

  return {
    admittedSourceEpisodeIds,
    partial,
    ...(partial
      ? {
          sourceVisibilityFraction: admittedSourceEpisodeIds.length / input.sourceEpisodeIds.length,
        }
      : {}),
    disclosureLabel: disclosureLabelForEpisodeIds(input.sourceEpisodeIds, input.labelsByEpisodeId),
  };
}

export async function resolveSemanticDisclosureSourceAdapter(input: {
  episodicRepository: EpisodicRepository;
  sourceEpisodeIds: readonly Episode["id"][];
  mode: SemanticSourceMode;
  visibility: SemanticVisibilityOptions;
}): Promise<SemanticSourceAdapter | null> {
  const admittedSourceEpisodeIds =
    input.mode === "disclosure"
      ? await resolveSemanticSourceEpisodeIdsForDisclosure(
          input.episodicRepository,
          input.sourceEpisodeIds,
          input.visibility,
        )
      : null;
  const adapted = adaptSemanticSourceEpisodes({
    sourceEpisodeIds: input.sourceEpisodeIds,
    admittedSourceEpisodeIds,
    labelsByEpisodeId: await resolveEpisodeDisclosureLabels(
      input.episodicRepository,
      input.sourceEpisodeIds,
    ),
  });

  return input.mode === "disclosure" &&
    admittedSourceEpisodeIds !== null &&
    adapted.admittedSourceEpisodeIds.length === 0
    ? null
    : adapted;
}

export async function resolveMemoryDisclosureLabelForEpisodeIds(
  episodicRepository: EpisodicRepository,
  episodeIds: readonly Episode["id"][],
): Promise<MemoryDisclosureLabel> {
  return combineDisclosureLabelForEpisodeIds(episodeIds, (ids) => episodicRepository.getMany(ids));
}

function semanticSourcePartialFields(
  source: SemanticSourceAdapter,
): Pick<RetrievedSemanticNode, "partial_source_visibility" | "source_visibility_fraction"> {
  return source.partial
    ? {
        partial_source_visibility: true,
        source_visibility_fraction: source.sourceVisibilityFraction,
      }
    : {};
}

function withSemanticNodeSourceAdapter<T extends SemanticNode>(
  node: T,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
  labelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>,
): T &
  Pick<
    RetrievedSemanticNode,
    "partial_source_visibility" | "source_visibility_fraction" | "disclosureLabel"
  > {
  const source = adaptSemanticSourceEpisodes({
    sourceEpisodeIds: node.source_episode_ids,
    admittedSourceEpisodeIds,
    labelsByEpisodeId,
  });

  return {
    ...node,
    ...(admittedSourceEpisodeIds !== null &&
    source.admittedSourceEpisodeIds.length !== node.source_episode_ids.length
      ? { source_episode_ids: source.admittedSourceEpisodeIds }
      : {}),
    ...semanticSourcePartialFields(source),
    disclosureLabel: source.disclosureLabel,
  };
}

function withSemanticEdgeSourceAdapter<T extends SemanticEdge>(
  edge: T,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
  labelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>,
): T &
  Pick<
    RetrievedSemanticEdge,
    "partial_source_visibility" | "source_visibility_fraction" | "disclosureLabel"
  > {
  const source = adaptSemanticSourceEpisodes({
    sourceEpisodeIds: edge.evidence_episode_ids,
    admittedSourceEpisodeIds,
    labelsByEpisodeId,
  });

  return {
    ...edge,
    ...(admittedSourceEpisodeIds !== null &&
    source.admittedSourceEpisodeIds.length !== edge.evidence_episode_ids.length
      ? { evidence_episode_ids: source.admittedSourceEpisodeIds }
      : {}),
    ...semanticSourcePartialFields(source),
    disclosureLabel: source.disclosureLabel,
  };
}

function hasSemanticNodeSourcesForDisclosure(
  node: SemanticNode,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
): boolean {
  return (
    resolveSemanticNodeSourcesForDisclosure(node, admittedSourceEpisodeIds)
      .availableSourceEpisodeIds.length > 0
  );
}

function resolveSemanticNodeSourcesForDisclosure(
  node: SemanticNode,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
): SemanticNodeSourcesForDisclosure {
  if (admittedSourceEpisodeIds === null) {
    return {
      availableSourceEpisodeIds: [...node.source_episode_ids],
      partial: false,
    };
  }

  const availableSourceEpisodeIds = node.source_episode_ids.filter((episodeId) =>
    admittedSourceEpisodeIds.has(episodeId),
  );

  return {
    availableSourceEpisodeIds,
    partial:
      availableSourceEpisodeIds.length > 0 &&
      availableSourceEpisodeIds.length < node.source_episode_ids.length,
  };
}

function withSemanticSourcesForDisclosure<T extends SemanticNode>(
  node: T,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
): T & Pick<RetrievedSemanticNode, "partial_source_visibility" | "source_visibility_fraction"> {
  const sourceVisibility = resolveSemanticNodeSourcesForDisclosure(node, admittedSourceEpisodeIds);

  if (
    admittedSourceEpisodeIds === null ||
    sourceVisibility.availableSourceEpisodeIds.length === node.source_episode_ids.length
  ) {
    return node;
  }

  return {
    ...node,
    source_episode_ids: sourceVisibility.availableSourceEpisodeIds,
    ...(sourceVisibility.partial
      ? {
          partial_source_visibility: true,
          source_visibility_fraction:
            sourceVisibility.availableSourceEpisodeIds.length / node.source_episode_ids.length,
        }
      : {}),
  };
}

function resolveSemanticEdgeSourcesForDisclosure(
  edge: SemanticEdge,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
): SemanticEdgeSourcesForDisclosure {
  if (admittedSourceEpisodeIds === null) {
    return {
      availableEvidenceEpisodeIds: [...edge.evidence_episode_ids],
      partial: false,
    };
  }

  const availableEvidenceEpisodeIds = edge.evidence_episode_ids.filter((episodeId) =>
    admittedSourceEpisodeIds.has(episodeId),
  );

  return {
    availableEvidenceEpisodeIds,
    partial:
      availableEvidenceEpisodeIds.length > 0 &&
      availableEvidenceEpisodeIds.length < edge.evidence_episode_ids.length,
  };
}

function hasSemanticEdgeSourcesForDisclosure(
  edge: SemanticEdge,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
): boolean {
  return (
    resolveSemanticEdgeSourcesForDisclosure(edge, admittedSourceEpisodeIds)
      .availableEvidenceEpisodeIds.length > 0
  );
}

function withSemanticEdgeSourcesForDisclosure<T extends SemanticEdge>(
  edge: T,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
): T & Pick<RetrievedSemanticEdge, "partial_source_visibility" | "source_visibility_fraction"> {
  const sourceVisibility = resolveSemanticEdgeSourcesForDisclosure(edge, admittedSourceEpisodeIds);

  if (
    admittedSourceEpisodeIds === null ||
    sourceVisibility.availableEvidenceEpisodeIds.length === edge.evidence_episode_ids.length
  ) {
    return edge;
  }

  return {
    ...edge,
    evidence_episode_ids: sourceVisibility.availableEvidenceEpisodeIds,
    ...(sourceVisibility.partial
      ? {
          partial_source_visibility: true,
          source_visibility_fraction:
            sourceVisibility.availableEvidenceEpisodeIds.length / edge.evidence_episode_ids.length,
        }
      : {}),
  };
}

function withSemanticWalkStepEdgesForDisclosure(
  step: SemanticWalkStep,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
  disclosureLabelsByEpisodeId?: ReadonlyMap<string, MemoryDisclosureLabel>,
): SemanticWalkStep & { edgePath: RetrievedSemanticEdge[] } {
  return {
    ...step,
    edgePath: step.edgePath.map((edge) => {
      const visibleEdge = withSemanticEdgeSourcesForDisclosure(edge, admittedSourceEpisodeIds);

      return disclosureLabelsByEpisodeId === undefined
        ? visibleEdge
        : withSemanticEdgeSourceAdapter(
            edge,
            admittedSourceEpisodeIds,
            disclosureLabelsByEpisodeId,
          );
    }),
  };
}

export async function isSemanticNodeAvailableForDisclosure(
  node: SemanticNode,
  visibility: SemanticVisibilityOptions,
  dependencies: Pick<SemanticRetrievalDependencies, "episodicRepository">,
): Promise<boolean> {
  const admittedSourceEpisodeIds = await resolveSemanticSourceEpisodeIdsForDisclosure(
    dependencies.episodicRepository,
    node.source_episode_ids,
    visibility,
  );

  return hasSemanticNodeSourcesForDisclosure(node, admittedSourceEpisodeIds);
}

function isSemanticWalkStepAvailableForDisclosure(
  step: SemanticWalkStep,
  admittedSourceEpisodeIds: ReadonlySet<string> | null,
): boolean {
  return (
    hasSemanticNodeSourcesForDisclosure(step.node, admittedSourceEpisodeIds) &&
    step.edgePath.every((edge) =>
      hasSemanticEdgeSourcesForDisclosure(edge, admittedSourceEpisodeIds),
    )
  );
}

export async function filterSemanticWalkStepsForDisclosure(
  steps: readonly SemanticWalkStep[],
  visibility: SemanticVisibilityOptions,
  dependencies: Pick<SemanticRetrievalDependencies, "episodicRepository">,
): Promise<Array<SemanticWalkStep & { edgePath: RetrievedSemanticEdge[] }>> {
  const admittedSourceEpisodeIds = await resolveSemanticSourceEpisodeIdsForDisclosure(
    dependencies.episodicRepository,
    steps.flatMap((step) => [
      ...step.node.source_episode_ids,
      ...step.edgePath.flatMap((edge) => edge.evidence_episode_ids),
    ]),
    visibility,
  );

  return steps
    .filter((step) => isSemanticWalkStepAvailableForDisclosure(step, admittedSourceEpisodeIds))
    .map((step) => ({
      ...withSemanticWalkStepEdgesForDisclosure(step, admittedSourceEpisodeIds),
      node: withSemanticSourcesForDisclosure(step.node, admittedSourceEpisodeIds),
    }));
}

function normalizeUnderReviewMultiplier(value: number | undefined): number {
  const multiplier = value ?? DEFAULT_UNDER_REVIEW_MULTIPLIER;

  if (!Number.isFinite(multiplier) || multiplier < 0 || multiplier > 1) {
    throw new TypeError("underReviewMultiplier must be between 0 and 1");
  }

  return multiplier;
}

function normalizeStatusMultipliers(
  value: Partial<SemanticStatusMultipliers> | undefined,
): SemanticStatusMultipliers {
  const multipliers: SemanticStatusMultipliers = {
    ...DEFAULT_SEMANTIC_STATUS_MULTIPLIERS,
    ...(value ?? {}),
  };

  for (const status of SEMANTIC_NODE_STATUSES) {
    const multiplier = multipliers[status];

    if (!Number.isFinite(multiplier) || multiplier < 0 || multiplier > 1) {
      throw new TypeError(`status multiplier for ${status} must be between 0 and 1`);
    }
  }

  return multipliers;
}

function normalizeOverfetchMultiplier(value: number | undefined): number {
  const multiplier = value ?? DEFAULT_SEMANTIC_OVERFETCH_MULTIPLIER;

  if (!Number.isFinite(multiplier)) {
    throw new TypeError("semantic overfetch multiplier must be a positive integer");
  }

  return Math.min(MAX_SEMANTIC_OVERFETCH_MULTIPLIER, Math.max(1, Math.trunc(multiplier)));
}

function overfetchLimit(baseLimit: number, multiplier: number): number {
  return Math.min(baseLimit * multiplier, MAX_SEMANTIC_CANDIDATE_FETCH_LIMIT);
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
    disclosureLabel: status.disclosureLabel,
  };
}

function semanticCandidateScore(
  candidate: MatchedNodeCandidate,
  input: {
    underReviewByNodeId: ReadonlyMap<string, OpenBeliefRevisionStatus>;
    underReviewMultiplier: number;
    statusMultipliers: SemanticStatusMultipliers;
  },
): number {
  const underReviewMultiplier = input.underReviewByNodeId.has(
    semanticNodeTargetKey(candidate.node.id),
  )
    ? input.underReviewMultiplier
    : 1;

  return (
    candidate.baseScore * input.statusMultipliers[candidate.node.status] * underReviewMultiplier
  );
}

function compareMatchedNodeCandidates(
  left: MatchedNodeCandidate,
  right: MatchedNodeCandidate,
  input: {
    underReviewByNodeId: ReadonlyMap<string, OpenBeliefRevisionStatus>;
    underReviewMultiplier: number;
    statusMultipliers: SemanticStatusMultipliers;
  },
): number {
  return (
    semanticCandidateScore(right, input) - semanticCandidateScore(left, input) ||
    right.baseScore - left.baseScore ||
    right.node.updated_at - left.node.updated_at ||
    left.node.id.localeCompare(right.node.id)
  );
}

function annotateSemanticNode(
  node: SemanticNode,
  input: {
    baseScore?: number;
    underReviewByNodeId: ReadonlyMap<string, OpenBeliefRevisionStatus>;
    underReviewMultiplier: number;
    statusMultipliers: SemanticStatusMultipliers;
    admittedSourceEpisodeIds: ReadonlySet<string> | null;
    disclosureLabelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>;
  },
): RetrievedSemanticNode {
  const status = buildUnderReviewStatus(
    input.underReviewByNodeId.get(semanticNodeTargetKey(node.id)),
  );
  const visibleNode = withSemanticNodeSourceAdapter(
    node,
    input.admittedSourceEpisodeIds,
    input.disclosureLabelsByEpisodeId,
  );
  const statusMultiplier = input.statusMultipliers[node.status];
  const multiplierFields =
    statusMultiplier === 1 ? {} : { status_retrieval_multiplier: statusMultiplier };

  if (input.baseScore === undefined) {
    return status === undefined
      ? {
          ...visibleNode,
          ...multiplierFields,
        }
      : {
          ...visibleNode,
          ...multiplierFields,
          under_review: status,
        };
  }

  const retrievalScore =
    input.baseScore * (status === undefined ? 1 : input.underReviewMultiplier) * statusMultiplier;

  return {
    ...visibleNode,
    ...multiplierFields,
    base_retrieval_score: input.baseScore,
    retrieval_score: retrievalScore,
    ...(status === undefined ? {} : { under_review: status }),
  };
}

async function collectUnderReviewStatuses(
  nodes: readonly SemanticNode[],
  dependencies: SemanticRetrievalDependencies,
  options: SemanticVisibilityOptions,
  sourceMode: SemanticSourceMode,
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

  return sourceMode === "cognition"
    ? dependencies.reviewQueueRepository.listOpenBeliefRevisionsByTargetForCognition(targets, {
        episodicRepository: dependencies.episodicRepository,
      })
    : dependencies.reviewQueueRepository.listOpenBeliefRevisionsByTarget(targets, {
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

async function resolveSemanticContextWithDisclosureSourceMode(
  query: string,
  options: SemanticRetrievalOptions,
  dependencies: SemanticRetrievalDependencies,
  sourceMode: SemanticSourceMode,
): Promise<ResolvedSemanticRetrieval> {
  const { embeddingClient, episodicRepository, semanticNodeRepository, semanticGraph } =
    dependencies;

  if (semanticNodeRepository === undefined || semanticGraph === undefined) {
    return emptySemanticRetrieval();
  }

  const underReviewMultiplier = normalizeUnderReviewMultiplier(options.underReviewMultiplier);
  const statusMultipliers = normalizeStatusMultipliers(options.statusMultipliers);
  const overfetchMultiplier = normalizeOverfetchMultiplier(options.overfetchMultiplier);
  const exactTerms = options.exactTerms ?? [];
  const directMatchLimit =
    exactTerms.length > 0 ? DEFAULT_EXACT_MATCH_LIMIT : DEFAULT_VECTOR_MATCH_LIMIT;
  const matchedNodeCandidatesById = new Map<SemanticNode["id"], MatchedNodeCandidate>();

  const queryVector = options.queryVector ?? (await embeddingClient.embed(query));
  const byVector = await semanticNodeRepository.searchByVector(queryVector, {
    limit: overfetchLimit(DEFAULT_VECTOR_MATCH_LIMIT, overfetchMultiplier),
    minSimilarity: DEFAULT_SEMANTIC_NODE_MIN_SIMILARITY,
    includeArchived: false,
  });

  for (const item of byVector) {
    recordMatchedNode(matchedNodeCandidatesById, item.node, item.similarity);
  }

  for (const term of exactTerms) {
    const exactMatches = await semanticNodeRepository.findByExactLabelOrAlias(
      term,
      overfetchLimit(DEFAULT_EXACT_MATCH_LIMIT, overfetchMultiplier),
      {
        includeArchived: false,
      },
    );

    for (const node of exactMatches) {
      recordMatchedNode(matchedNodeCandidatesById, node, 1);
    }
  }

  const matchedNodeCandidates = [...matchedNodeCandidatesById.values()];
  const matchedNodeVisibility =
    sourceMode === "disclosure"
      ? await resolveSemanticSourceEpisodeIdsForDisclosure(
          episodicRepository,
          matchedNodeCandidates.flatMap(({ node }) => node.source_episode_ids),
          options,
        )
      : null;
  const uniqueNodes = new Map(
    matchedNodeCandidates
      .filter(({ node }) => hasSemanticNodeSourcesForDisclosure(node, matchedNodeVisibility))
      .map((candidate) => [candidate.node.id, candidate] as const),
  );
  const candidateUnderReviewByNodeId = await collectUnderReviewStatuses(
    [...uniqueNodes.values()].map(({ node }) => node),
    dependencies,
    options,
    sourceMode,
  );
  const selectedNodeCandidates = [...uniqueNodes.values()]
    .sort((left, right) =>
      compareMatchedNodeCandidates(left, right, {
        underReviewByNodeId: candidateUnderReviewByNodeId,
        underReviewMultiplier,
        statusMultipliers,
      }),
    )
    .slice(0, directMatchLimit);
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

  for (const node of selectedNodeCandidates) {
    const walkedSupports = await semanticGraph.walk(node.node.id, {
      relations: ["supports"],
      direction: "out",
      depth: walkDepth,
      maxNodes: maxGraphNodes,
      asOf: options.asOf,
    });
    const walkedInboundSupports =
      node.node.kind === "proposition"
        ? await semanticGraph.walk(node.node.id, {
            relations: ["supports"],
            direction: "in",
            depth: walkDepth,
            maxNodes: maxGraphNodes,
            asOf: options.asOf,
          })
        : [];
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
    supportNeighbors.push(
      ...walkedInboundSupports.map((step) => ({ rootNodeId: node.node.id, step })),
    );
    causalNeighbors.push(...walkedCausals.map((step) => ({ rootNodeId: node.node.id, step })));
    contradictionNeighbors.push(
      ...walkedContradictions.map((step) => ({ rootNodeId: node.node.id, step })),
    );
    categoryNeighbors.push(...walkedCategories.map((step) => ({ rootNodeId: node.node.id, step })));
  }

  const semanticSourceEpisodeIds = [
    ...selectedNodeCandidates.flatMap(({ node }) => node.source_episode_ids),
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
  ];
  const semanticVisibility =
    sourceMode === "disclosure"
      ? await resolveSemanticSourceEpisodeIdsForDisclosure(
          episodicRepository,
          semanticSourceEpisodeIds,
          options,
        )
      : null;
  const disclosureLabelsByEpisodeId = await resolveEpisodeDisclosureLabels(
    episodicRepository,
    semanticSourceEpisodeIds,
  );

  const visibleSupportNeighbors = supportNeighbors
    .filter(({ step }) => isSemanticWalkStepAvailableForDisclosure(step, semanticVisibility))
    .map((item) => ({
      ...item,
      step: withSemanticWalkStepEdgesForDisclosure(
        item.step,
        semanticVisibility,
        disclosureLabelsByEpisodeId,
      ),
    }));
  const visibleCausalNeighbors = causalNeighbors
    .filter(({ step }) => isSemanticWalkStepAvailableForDisclosure(step, semanticVisibility))
    .map((item) => ({
      ...item,
      step: withSemanticWalkStepEdgesForDisclosure(
        item.step,
        semanticVisibility,
        disclosureLabelsByEpisodeId,
      ),
    }));
  const visibleContradictionNeighbors = contradictionNeighbors
    .filter(({ step }) => isSemanticWalkStepAvailableForDisclosure(step, semanticVisibility))
    .map((item) => ({
      ...item,
      step: withSemanticWalkStepEdgesForDisclosure(
        item.step,
        semanticVisibility,
        disclosureLabelsByEpisodeId,
      ),
    }));
  const visibleCategoryNeighbors = categoryNeighbors
    .filter(({ step }) => isSemanticWalkStepAvailableForDisclosure(step, semanticVisibility))
    .map((item) => ({
      ...item,
      step: withSemanticWalkStepEdgesForDisclosure(
        item.step,
        semanticVisibility,
        disclosureLabelsByEpisodeId,
      ),
    }));
  const underReviewByNodeId = await collectUnderReviewStatuses(
    [
      ...selectedNodeCandidates.map(({ node }) => node),
      ...visibleSupportNeighbors.map(({ step }) => step.node),
      ...visibleCausalNeighbors.map(({ step }) => step.node),
      ...visibleContradictionNeighbors.map(({ step }) => step.node),
      ...visibleCategoryNeighbors.map(({ step }) => step.node),
    ],
    dependencies,
    options,
    sourceMode,
  );
  const visibleMatchedNodes = await Promise.all(
    selectedNodeCandidates
      .filter(({ node }) => hasSemanticNodeSourcesForDisclosure(node, semanticVisibility))
      .map(async (candidate): Promise<RetrievedSemanticNode> => {
        const annotated = annotateSemanticNode(candidate.node, {
          baseScore: candidate.baseScore,
          underReviewByNodeId,
          underReviewMultiplier,
          statusMultipliers,
          admittedSourceEpisodeIds: semanticVisibility,
          disclosureLabelsByEpisodeId,
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
      statusMultipliers,
      admittedSourceEpisodeIds: semanticVisibility,
      disclosureLabelsByEpisodeId,
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
        statusMultipliers,
        admittedSourceEpisodeIds: semanticVisibility,
        disclosureLabelsByEpisodeId,
      }),
      edgePath: item.step.edgePath,
    });
  }

  for (const item of visibleContradictionNeighbors) {
    const node = annotateSemanticNode(item.step.node, {
      underReviewByNodeId,
      underReviewMultiplier,
      statusMultipliers,
      admittedSourceEpisodeIds: semanticVisibility,
      disclosureLabelsByEpisodeId,
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
      statusMultipliers,
      admittedSourceEpisodeIds: semanticVisibility,
      disclosureLabelsByEpisodeId,
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

export async function resolveSemanticContextForCognition(
  query: string,
  options: SemanticRetrievalOptions,
  dependencies: SemanticRetrievalDependencies,
): Promise<ResolvedSemanticRetrieval> {
  return resolveSemanticContextWithDisclosureSourceMode(query, options, dependencies, "cognition");
}

export async function resolveSemanticContextForDisclosure(
  query: string,
  options: SemanticRetrievalOptions,
  dependencies: SemanticRetrievalDependencies,
): Promise<ResolvedSemanticRetrieval> {
  return resolveSemanticContextWithDisclosureSourceMode(query, options, dependencies, "disclosure");
}

export async function resolveSemanticContext(
  query: string,
  options: SemanticRetrievalOptions,
  dependencies: SemanticRetrievalDependencies,
): Promise<ResolvedSemanticRetrieval> {
  return resolveSemanticContextForCognition(query, options, dependencies);
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
