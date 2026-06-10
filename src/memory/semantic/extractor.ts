import { z } from "zod";

import type { EmbeddingClient } from "../../embeddings/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import {
  renderParticipantRoster,
  type ParticipantRosterForRendering,
} from "../common/participant-roster-rendering.js";
import { memoryDisclosurePayloadFields } from "../common/disclosure-serializers.js";
import {
  HEADCOUNT_SET_GROUNDING_PROMPT,
  RELATIONSHIP_LABEL_WRITE_GROUNDING_PROMPT,
  RELATIONSHIP_LABELS_PROMPT,
} from "../common/relationship-label-prompts.js";
import { checkRelationshipClaimGroundingAsync } from "../common/relationship-claim-grounding.js";
import {
  relationshipClaimSchema,
  type RelationshipClaim,
} from "../common/relationship-claims.js";
import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { LLMError, SemanticError, StorageError } from "../../util/errors.js";
import { createSemanticNodeId, type EntityId, type StreamEntryId } from "../../util/ids.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import { memoryDisclosureLabelFromEpisodeAccess } from "../common/disclosure-label.js";
import {
  normalizeEpisodeAccess,
  type Episode,
  type EpisodicRepository,
} from "../episodic/index.js";
import type {
  RelationshipEvidenceStreamEntryTrustResult,
  RelationshipEvidenceStreamEntryTrustValidator,
} from "../source-trust.js";
import { SemanticEdgeRepository, SemanticNodeRepository } from "./repository.js";
import type { SemanticReviewService } from "./review-service.js";
import type { ReviewQueueInsertInput } from "../review-queue/review-queue.js";
import { canonicalizeDomain } from "./domain.js";
import {
  semanticNodeKindSchema,
  semanticObservationMetadataSchema,
  semanticRelationSchema,
  type SemanticNode,
  type SemanticNodeKind,
  type SemanticObservationMetadata,
} from "./types.js";

const MAX_EXTRACTOR_NODES = 40;
const MAX_EXTRACTOR_EDGES = 60;

const extractorNodeSchema = z.object({
  kind: semanticNodeKindSchema,
  label: z.string().min(1),
  description: z.string().min(1),
  domain: z.string().min(1).nullable().default(null),
  aliases: z.array(z.string().min(1)),
  observation_metadata: semanticObservationMetadataSchema.nullable().default(null),
  relationship_claims: z.array(relationshipClaimSchema).optional().default([]),
  confidence: z.number().min(0).max(1),
  source_episode_ids: z.array(z.string().min(1)).min(1),
});

const extractorEdgeSchema = z.object({
  from_label: z.string().min(1),
  to_label: z.string().min(1),
  relation: semanticRelationSchema,
  confidence: z.number().min(0).max(1),
  evidence_episode_ids: z.array(z.string().min(1)).min(1),
  valid_from_ts: z.number().finite().nullable().default(null),
  valid_to_ts: z.number().finite().nullable().default(null),
});

const extractorResponseSchema = z.object({
  nodes: z.array(extractorNodeSchema).max(MAX_EXTRACTOR_NODES),
  edges: z.array(z.unknown()).max(MAX_EXTRACTOR_EDGES),
});
const extractorResponseToolSchema = z.object({
  nodes: z.array(extractorNodeSchema).max(MAX_EXTRACTOR_NODES),
  edges: z.array(extractorEdgeSchema).max(MAX_EXTRACTOR_EDGES),
});

type ExtractorNode = z.infer<typeof extractorNodeSchema>;
type ExtractorEdge = z.infer<typeof extractorEdgeSchema>;
type ParsedExtractorEdge = {
  candidate: ExtractorEdge;
  candidateIndex: number;
};
type SkippedEdgeTraceDetail = {
  candidate_index: number;
  reason: SemanticInsertSkipReason;
  from_label?: string;
  to_label?: string;
  relation?: ExtractorEdge["relation"];
  evidence_ids?: string[];
};

const DEFAULT_CONFIDENCE_CEILING = 0.7;
const DEDUP_THRESHOLD = 0.88;
const SKIPPED_EDGE_TRACE_DETAIL_LIMIT = 10;
const EXTRACT_SEMANTIC_TOOL_NAME = "EmitSemanticCandidates";
export const EXTRACT_SEMANTIC_TOOL = {
  name: EXTRACT_SEMANTIC_TOOL_NAME,
  description: "Emit grounded semantic nodes and edges extracted from episodes.",
  inputSchema: toToolInputSchema(extractorResponseToolSchema),
} satisfies LLMToolDefinition;

export type SemanticExtractorOptions = {
  nodeRepository: SemanticNodeRepository;
  edgeRepository: SemanticEdgeRepository;
  embeddingClient: EmbeddingClient;
  episodicRepository: Pick<EpisodicRepository, "getMany">;
  llmClient: LLMClient;
  model: string;
  semanticReviewService?: Pick<SemanticReviewService, "queueDuplicateReview">;
  reviewEnqueue?: (input: ReviewQueueInsertInput) => unknown;
  participantRoster?: ParticipantRosterForRendering | null;
  selfEntityId?: EntityId | null;
  relationshipEvidenceStreamEntryTrust?: RelationshipEvidenceStreamEntryTrustValidator;
  clock?: Clock;
  tracer?: TurnTracer;
  traceTurnId?: string;
  dedupThreshold?: number;
  confidenceCeiling?: number;
};

export type SemanticRelationshipEvidenceStreamEntryTrustResult =
  RelationshipEvidenceStreamEntryTrustResult;
export type SemanticRelationshipEvidenceStreamEntryTrustValidator =
  RelationshipEvidenceStreamEntryTrustValidator;

export type ExtractSemanticResult = {
  insertedNodes: number;
  updatedNodes: number;
  skippedNodes: number;
  insertedEdges: number;
  updatedEdges: number;
  skippedEdges: number;
};

type SemanticInsertSkipReason =
  | "dedupe_match"
  | "schema_invalid"
  | "invalid_ref"
  | "invalid_endpoint"
  | "validity_window_conflict"
  | "episode_archived_post_plan"
  | "relationship_claim_ungrounded"
  | "other";

function buildPrompt(input: {
  episodes: readonly Episode[];
  participantRoster?: ParticipantRosterForRendering | null;
  selfEntityId?: EntityId | null;
  knownNodeKinds: readonly SemanticNodeKind[];
}): string {
  const roster = renderParticipantRoster(input.participantRoster);
  const selfEntityGuidance =
    input.selfEntityId === undefined || input.selfEntityId === null
      ? null
      : `Entity ${input.selfEntityId} is yourself; refer to all entities by name, including yourself.`;
  const knownNodeKindGuidance =
    input.knownNodeKinds.length === 0
      ? "Known semantic node kinds already in graph: none."
      : `Known semantic node kinds already in graph: ${input.knownNodeKinds.join(", ")}.`;

  return [
    "Extract semantic knowledge from the provided episodes.",
    `Emit your result by calling the ${EXTRACT_SEMANTIC_TOOL_NAME} tool exactly once.`,
    "Populate the tool arguments directly with arrays and objects. Do not put JSON, XML tags, or parameter wrappers inside string fields.",
    "Extract all salient semantic nodes and edges that are grounded in the provided episodes.",
    "Distinguish temporally bounded events such as trips, visits, conversations, or meetings from permanent or long-term state changes such as moves, relocations, role changes, or life changes.",
    "When choosing labels and aliases, do not collapse event-scoped language into permanent-state language or the reverse.",
    "If the source wording is ambiguous, prefer the narrower event-scoped interpretation.",
    "For observation-type propositions, decide by meaning whether the node records that someone observed something happened. When it does, populate observation_metadata with any directly stated witness, timeframe/date, count_or_intensity, source_kind, confidence, and status. Use null for unknown fields and null observation_metadata for non-observation nodes.",
    "Do not merge multiple observations into one proposition merely because they concern the same topic. Distinct witness, timeframe/date, count_or_intensity, source_kind, confidence, or status belongs in observation_metadata so identity can preserve separate observations.",
    "Each node must cite source_episode_ids from the provided episode ids only.",
    "Each edge must use from_label and to_label values that match node labels exactly.",
    "Only use relation values allowed by the tool schema.",
    "Edges may only reference nodes that already exist or are extracted in this batch.",
    "Set node.kind to a lowercase_slug structural shape label. Reuse one of the known kinds when it fits; coin a new lowercase_slug kind only for a genuinely new information shape.",
    knownNodeKindGuidance,
    'Emit a compact canonical domain string for each node when it helps metadata display or later filtering (examples: "tech", "people", "places", "food", "process"). Use null for broadly general nodes. Domain never decides semantic compatibility; vector meaning and exact labels do.',
    "Temporal validity for edges: set valid_from_ts and valid_to_ts to numeric Unix epoch milliseconds only when the episode wording explicitly says when the relation became true or stopped being true. Resolve relative dates against the episode start time yourself. Use null when unknown. Do not infer validity dates from the episode timestamp alone.",
    RELATIONSHIP_LABELS_PROMPT,
    RELATIONSHIP_LABEL_WRITE_GROUNDING_PROMPT,
    HEADCOUNT_SET_GROUNDING_PROMPT,
    "When a node label or description asserts a sensitive interpersonal relationship, emit relationship_claims with supporting evidence ids. Do not cite assistant output as relationship evidence. If no accepted evidence grounds the claim, rewrite the node neutrally before emitting it.",
    roster === null ? "Thread roster: none supplied." : roster,
    ...(selfEntityGuidance === null ? [] : [selfEntityGuidance]),
    "Keep confidence modest for fresh extractions.",
    "Episodes:",
    ...input.episodes.map((episode) => {
      const access = normalizeEpisodeAccess(episode);
      const disclosureLabel = memoryDisclosureLabelFromEpisodeAccess(episode);

      return JSON.stringify({
        id: episode.id,
        title: episode.title,
        narrative: episode.narrative,
        participants: episode.participants,
        audience_entity_id: access.audience_entity_id,
        origin_audience_entity_ids: access.origin_audience_entity_ids,
        shared: access.shared,
        location: episode.location,
        tags: episode.tags,
        ...memoryDisclosurePayloadFields(disclosureLabel),
      });
    }),
  ].join("\n");
}

function parseResponse(input: unknown): {
  nodes: ExtractorNode[];
  edges: ParsedExtractorEdge[];
  rawEdgeCount: number;
  schemaInvalidEdgeCount: number;
  schemaInvalidEdgeDetails: SkippedEdgeTraceDetail[];
} {
  const parsed = extractorResponseSchema.safeParse(input);

  if (!parsed.success) {
    throw new LLMError("Semantic extractor returned invalid payload", {
      cause: parsed.error,
      code: "SEMANTIC_EXTRACTOR_INVALID",
    });
  }

  const edges: ParsedExtractorEdge[] = [];
  const schemaInvalidEdgeDetails: SkippedEdgeTraceDetail[] = [];
  let schemaInvalidEdgeCount = 0;

  for (const [candidateIndex, edge] of parsed.data.edges.entries()) {
    const parsedEdge = extractorEdgeSchema.safeParse(edge);

    if (!parsedEdge.success) {
      schemaInvalidEdgeCount += 1;
      pushSkippedEdgeTraceDetail(
        schemaInvalidEdgeDetails,
        skippedEdgeTraceDetailFromRaw(candidateIndex, "schema_invalid", edge),
      );
      continue;
    }

    edges.push({
      candidate: parsedEdge.data,
      candidateIndex,
    });
  }

  return {
    nodes: parsed.data.nodes,
    edges,
    rawEdgeCount: parsed.data.edges.length,
    schemaInvalidEdgeCount,
    schemaInvalidEdgeDetails,
  };
}

function pushSkippedEdgeTraceDetail(
  details: SkippedEdgeTraceDetail[],
  detail: SkippedEdgeTraceDetail,
): void {
  if (details.length < SKIPPED_EDGE_TRACE_DETAIL_LIMIT) {
    details.push(detail);
  }
}

function skippedEdgeTraceDetailFromRaw(
  candidateIndex: number,
  reason: SemanticInsertSkipReason,
  raw: unknown,
): SkippedEdgeTraceDetail {
  const detail: SkippedEdgeTraceDetail = {
    candidate_index: candidateIndex,
    reason,
  };
  const parsed = z
    .object({
      from_label: z.string().min(1).optional(),
      to_label: z.string().min(1).optional(),
      relation: semanticRelationSchema.optional(),
      evidence_episode_ids: z.array(z.string().min(1)).optional(),
    })
    .passthrough()
    .safeParse(raw);

  if (!parsed.success) {
    return detail;
  }

  return {
    ...detail,
    ...(parsed.data.from_label === undefined ? {} : { from_label: parsed.data.from_label }),
    ...(parsed.data.to_label === undefined ? {} : { to_label: parsed.data.to_label }),
    ...(parsed.data.relation === undefined ? {} : { relation: parsed.data.relation }),
    ...(parsed.data.evidence_episode_ids === undefined
      ? {}
      : { evidence_ids: parsed.data.evidence_episode_ids }),
  };
}

function skippedEdgeTraceDetailFromCandidate(
  candidateIndex: number,
  reason: SemanticInsertSkipReason,
  candidate: ExtractorEdge,
): SkippedEdgeTraceDetail {
  return {
    candidate_index: candidateIndex,
    reason,
    from_label: candidate.from_label,
    to_label: candidate.to_label,
    relation: candidate.relation,
    evidence_ids: candidate.evidence_episode_ids,
  };
}

function mergeAliases(left: readonly string[], right: readonly string[]): string[] {
  return [
    ...new Set(
      [...left, ...right].map((value) => value.trim()).filter((value) => value.length > 0),
    ),
  ];
}

function buildNodeEmbeddingText(input: {
  label: string;
  description: string;
  aliases: readonly string[];
  observationMetadata: SemanticObservationMetadata | null;
}): string {
  const parts = [input.label, input.description, input.aliases.join(" ")];

  if (input.observationMetadata !== null) {
    parts.push(JSON.stringify(input.observationMetadata));
  }

  return parts.join("\n");
}

function observationMetadataIdentity(input: SemanticObservationMetadata | null): string | null {
  if (input === null) {
    return null;
  }

  return JSON.stringify({
    witness: input.witness,
    timeframe: input.timeframe,
    count_or_intensity: input.count_or_intensity,
    source_kind: input.source_kind,
    status: input.status,
  });
}

function observationMetadataAligns(
  left: SemanticObservationMetadata | null,
  right: SemanticObservationMetadata | null,
): boolean {
  return observationMetadataIdentity(left) === observationMetadataIdentity(right);
}

function uniqueEpisodeIds(ids: readonly Episode["id"][]): Episode["id"][] {
  return [...new Set(ids)];
}

function episodeIdOverlap(
  left: readonly Episode["id"][],
  right: readonly Episode["id"][],
): Episode["id"][] {
  const rightIds = new Set(right);

  return uniqueEpisodeIds(left.filter((id) => rightIds.has(id)));
}

function relationshipClaimsTracePayload(claims: readonly RelationshipClaim[]): RelationshipClaim[] {
  return claims.map((claim) => ({
    ...claim,
    evidence_relational_slot_ids: [...claim.evidence_relational_slot_ids],
    evidence_stream_entry_ids: [...claim.evidence_stream_entry_ids],
  }));
}

function relationshipClaimLabelFamilies(claims: readonly RelationshipClaim[]): string[] {
  return [...new Set(claims.map((claim) => claim.label_family))];
}

export class SemanticExtractor {
  private readonly clock: Clock;
  private readonly dedupThreshold: number;
  private readonly confidenceCeiling: number;

  constructor(private readonly options: SemanticExtractorOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.dedupThreshold = options.dedupThreshold ?? DEDUP_THRESHOLD;
    this.confidenceCeiling = options.confidenceCeiling ?? DEFAULT_CONFIDENCE_CEILING;
  }

  private traceInsertSkipped(input: {
    kind: "node" | "edge";
    reason: SemanticInsertSkipReason;
    relationshipClaims?: readonly RelationshipClaim[];
    ungroundedRelationshipClaims?: readonly RelationshipClaim[];
  }): void {
    if (this.options.tracer?.enabled !== true) {
      return;
    }

    this.options.tracer.emit("semantic_insert.skipped", {
      turnId: this.options.traceTurnId ?? "semantic_extractor",
      kind: input.kind,
      reason: input.reason,
      ...(input.relationshipClaims === undefined
        ? {}
        : {
            relationship_claim_label_families: relationshipClaimLabelFamilies(
              input.relationshipClaims,
            ),
            relationship_claims: relationshipClaimsTracePayload(input.relationshipClaims),
          }),
      ...(input.ungroundedRelationshipClaims === undefined
        ? {}
        : {
            ungrounded_relationship_claims: relationshipClaimsTracePayload(
              input.ungroundedRelationshipClaims,
            ),
          }),
    });
  }

  private traceExtractorInvoked(input: {
    inputEpisodeCount: number;
    parsedNodeCount: number;
    parsedEdgeCount: number;
    acceptedNodeCount: number;
    acceptedEdgeCount: number;
    skipReasons: readonly string[];
  }): void {
    if (this.options.tracer?.enabled !== true) {
      return;
    }

    this.options.tracer.emit("semantic_extractor.started", {
      turnId: this.options.traceTurnId ?? "semantic_extractor",
      input_episode_count: input.inputEpisodeCount,
      prompt_label: "semantic-extraction",
      parsed_node_count: input.parsedNodeCount,
      parsed_edge_count: input.parsedEdgeCount,
      accepted_node_count: input.acceptedNodeCount,
      accepted_edge_count: input.acceptedEdgeCount,
      skip_reasons: [...input.skipReasons],
    });
  }

  private tracePartialFailure(input: {
    inputEpisodeCount: number;
    parsedNodeCount: number;
    parsedEdgeCount: number;
    acceptedNodeCount: number;
    acceptedEdgeCount: number;
    skippedNodeCount: number;
    skippedEdgeCount: number;
    skipReasons: readonly string[];
    skippedEdgeDetails: readonly SkippedEdgeTraceDetail[];
  }): void {
    if (this.options.tracer?.enabled !== true) {
      return;
    }

    this.options.tracer.emit("semantic_extractor.degraded", {
      turnId: this.options.traceTurnId ?? "semantic_extractor",
      input_episode_count: input.inputEpisodeCount,
      parsed_node_count: input.parsedNodeCount,
      parsed_edge_count: input.parsedEdgeCount,
      accepted_node_count: input.acceptedNodeCount,
      accepted_edge_count: input.acceptedEdgeCount,
      skipped_node_count: input.skippedNodeCount,
      skipped_edge_count: input.skippedEdgeCount,
      skip_reasons: [...input.skipReasons],
      skipped_edge_details: input.skippedEdgeDetails.slice(0, SKIPPED_EDGE_TRACE_DETAIL_LIMIT),
    });
  }

  private validateEpisodeRefs(
    candidateIds: readonly string[],
    episodeIds: ReadonlySet<string>,
    label: string,
  ): Episode["id"][] {
    if (!candidateIds.every((value) => episodeIds.has(value))) {
      throw new SemanticError(`Semantic extractor referenced unknown ${label}`, {
        code: "SEMANTIC_EXTRACTOR_INVALID_REF",
      });
    }

    return candidateIds.map((value) => value as Episode["id"]);
  }

  private async relationshipClaimGroundingForCandidate(
    candidate: ExtractorNode,
    sourceEpisodes: readonly Episode[],
  ): ReturnType<typeof checkRelationshipClaimGroundingAsync> {
    const sourceStreamEntryIds = new Set(
      sourceEpisodes.flatMap((episode) => episode.source_stream_ids),
    );

    return checkRelationshipClaimGroundingAsync({
      claims: candidate.relationship_claims,
      participantRoster: this.options.participantRoster,
      allowedRelationshipEvidenceStreamEntryIds: sourceStreamEntryIds,
      relationshipEvidenceStreamEntryTrust: this.options.relationshipEvidenceStreamEntryTrust,
    });
  }

  private async upsertNode(
    candidate: ExtractorNode,
    allowedEpisodeIds: ReadonlySet<string>,
    episodeById: ReadonlyMap<Episode["id"], Episode>,
  ): Promise<{
    status: "inserted" | "updated" | "skipped";
    node?: SemanticNode;
    reason?: SemanticInsertSkipReason;
    ungroundedRelationshipClaims?: RelationshipClaim[];
  }> {
    const sourceEpisodeIds = this.validateEpisodeRefs(
      candidate.source_episode_ids,
      allowedEpisodeIds,
      "source_episode_ids",
    );
    const sourceEpisodes = sourceEpisodeIds
      .map((episodeId) => episodeById.get(episodeId))
      .filter((episode): episode is Episode => episode !== undefined);
    const relationshipClaimGrounding = await this.relationshipClaimGroundingForCandidate(
      candidate,
      sourceEpisodes,
    );

    if (!relationshipClaimGrounding.grounded) {
      return {
        status: "skipped",
        reason: "relationship_claim_ungrounded",
        ungroundedRelationshipClaims: relationshipClaimGrounding.ungroundedClaims,
      };
    }

    try {
      const candidateLabel = candidate.label.trim();
      const candidateDescription = candidate.description.trim();
      const candidateAliases = mergeAliases(candidate.aliases, []);
      const embedding = await this.options.embeddingClient.embed(
        buildNodeEmbeddingText({
          label: candidateLabel,
          description: candidateDescription,
          aliases: candidateAliases,
          observationMetadata: candidate.observation_metadata,
        }),
      );
      const isCompatibleNode = (node: SemanticNode): boolean => {
        if (node.kind !== candidate.kind) {
          return false;
        }

        if (!observationMetadataAligns(node.observation_metadata, candidate.observation_metadata)) {
          return false;
        }

        return cosineSimilarity(node.embedding, embedding) >= this.dedupThreshold;
      };
      const byLabelMatches = await this.options.nodeRepository.findByExactLabelOrAlias(
        candidate.label,
        5,
        {
          includeArchived: true,
        },
      );
      const byLabel: SemanticNode[] = [];

      for (const match of byLabelMatches) {
        if (isCompatibleNode(match)) {
          byLabel.push(match);
        }
      }

      const byVectorMatches = await this.options.nodeRepository.searchByVector(embedding, {
        limit: 3,
        minSimilarity: this.dedupThreshold,
      });
      const byVector: Array<{ node: SemanticNode; similarity: number }> = [];

      for (const match of byVectorMatches) {
        if (isCompatibleNode(match.node)) {
          byVector.push(match);
        }
      }

      const existing = byLabel[0];
      const vectorOnlyMatch = existing === undefined ? byVector[0] : undefined;
      const nowMs = this.clock.now();

      if (existing === undefined) {
        const inserted = await this.options.nodeRepository.insert({
          id: createSemanticNodeId(),
          kind: candidate.kind,
          label: candidateLabel,
          description: candidateDescription,
          domain: canonicalizeDomain(candidate.domain),
          aliases: candidateAliases,
          observation_metadata: candidate.observation_metadata,
          confidence: Math.min(candidate.confidence, this.confidenceCeiling),
          source_episode_ids: sourceEpisodeIds,
          created_at: nowMs,
          updated_at: nowMs,
          last_verified_at: nowMs,
          embedding,
          archived: false,
          superseded_by: null,
        });
        if (vectorOnlyMatch !== undefined) {
          this.queueVectorOnlyDuplicateReview({
            candidate: inserted,
            matched: vectorOnlyMatch.node,
            similarity: vectorOnlyMatch.similarity,
          });
        }
        this.options.semanticReviewService?.queueDuplicateReview(inserted);

        return {
          status: "inserted",
          node: inserted,
        };
      }

      const nextDescription =
        candidate.confidence >= existing.confidence ? candidateDescription : existing.description;
      const nextAliases = mergeAliases(existing.aliases, [candidateLabel, ...candidate.aliases]);
      const updatedEmbedding = await this.options.embeddingClient.embed(
        buildNodeEmbeddingText({
          label: existing.label,
          description: nextDescription,
          aliases: nextAliases,
          observationMetadata: candidate.observation_metadata,
        }),
      );
      const updated = await this.options.nodeRepository.update(existing.id, {
        description: nextDescription,
        domain: existing.domain ?? canonicalizeDomain(candidate.domain),
        aliases: nextAliases,
        observation_metadata: existing.observation_metadata,
        confidence: Math.max(
          existing.confidence * 0.99,
          Math.min(candidate.confidence, this.confidenceCeiling),
        ),
        source_episode_ids: sourceEpisodeIds,
        last_verified_at: nowMs,
        embedding: updatedEmbedding,
        archived: false,
      });

      return updated === null
        ? {
            status: "skipped",
            reason: "other",
          }
        : {
            status: "updated",
            node: updated,
            reason: "dedupe_match",
          };
    } catch (error) {
      if (error instanceof StorageError || error instanceof SemanticError) {
        return {
          status: "skipped",
          reason: "other",
        };
      }

      throw error;
    }
  }

  private queueVectorOnlyDuplicateReview(input: {
    candidate: SemanticNode;
    matched: SemanticNode;
    similarity: number;
  }): void {
    if (this.options.reviewEnqueue === undefined) {
      return;
    }

    const overlappingSourceEpisodeIds = episodeIdOverlap(
      input.candidate.source_episode_ids,
      input.matched.source_episode_ids,
    );

    this.options.reviewEnqueue({
      kind: "duplicate",
      refs: {
        node_ids: [input.candidate.id, input.matched.id],
        node_labels: [input.candidate.label, input.matched.label],
        duplicate_subtype: "vector_only_merge_candidate",
        vector_similarity: input.similarity,
        source_overlap: {
          candidate_source_episode_ids: uniqueEpisodeIds(input.candidate.source_episode_ids),
          matched_source_episode_ids: uniqueEpisodeIds(input.matched.source_episode_ids),
          overlapping_source_episode_ids: overlappingSourceEpisodeIds,
          overlap_count: overlappingSourceEpisodeIds.length,
        },
      },
      reason: `Vector-only semantic merge candidate with similarity ${input.similarity.toFixed(3)}`,
      sourceProcess: "semantic-extractor",
      ...(this.options.traceTurnId === undefined ? {} : { traceTurnId: this.options.traceTurnId }),
    });
  }

  private edgeNodeCanAcceptEvidence(node: SemanticNode): boolean {
    return !node.archived && node.status === "active";
  }

  private async resolveEdgeNode(
    label: string,
    batchNodes: ReadonlyMap<string, SemanticNode>,
  ): Promise<SemanticNode | undefined> {
    const key = label.toLowerCase();
    const localNode = batchNodes.get(key);

    if (localNode !== undefined && this.edgeNodeCanAcceptEvidence(localNode)) {
      return localNode;
    }

    const matches = await this.options.nodeRepository.findByExactLabelOrAlias(label, 10, {
      includeArchived: false,
    });
    const activeMatchesById = new Map<SemanticNode["id"], SemanticNode>();

    for (const matchedNode of matches) {
      if (!this.edgeNodeCanAcceptEvidence(matchedNode)) {
        continue;
      }

      activeMatchesById.set(matchedNode.id, matchedNode);
    }

    const activeMatches = [...activeMatchesById.values()];

    return activeMatches.length === 1 ? activeMatches[0] : undefined;
  }

  private edgeSkipReason(error: SemanticError): SemanticInsertSkipReason {
    if (error.code === "SEMANTIC_EDGE_DUPLICATE") {
      return "dedupe_match";
    }

    if (error.code === "SEMANTIC_EDGE_DANGLING") {
      return "invalid_endpoint";
    }

    return "other";
  }

  async extractFromEpisodes(episodes: readonly Episode[]): Promise<ExtractSemanticResult> {
    if (episodes.length === 0) {
      return {
        insertedNodes: 0,
        updatedNodes: 0,
        skippedNodes: 0,
        insertedEdges: 0,
        updatedEdges: 0,
        skippedEdges: 0,
      };
    }

    const knownNodeKinds = this.options.nodeRepository.listDistinctKinds();
    let parsed: ReturnType<typeof parseResponse>;

    try {
      parsed = (
        await callStructuredTool({
          llmClient: this.options.llmClient,
          request: {
            model: this.options.model,
            system: "Extract semantic nodes and edges grounded only in the provided episodes.",
            messages: [
              {
                role: "user",
                content: buildPrompt({
                  episodes,
                  participantRoster: this.options.participantRoster ?? null,
                  selfEntityId: this.options.selfEntityId ?? null,
                  knownNodeKinds,
                }),
              },
            ],
            tools: [EXTRACT_SEMANTIC_TOOL],
            tool_choice: { type: "tool", name: EXTRACT_SEMANTIC_TOOL_NAME },
            // Opus-class extraction can emit large structured batches. This is the
            // OUTPUT-token ceiling, which is a fixed model limit (~20k for
            // claude-opus-4-6), independent of the much larger input context window.
            // 20_000 is verified within the model's max and well above the prior
            // 12_000, so dense episodes aren't truncated. Do NOT raise past the
            // model's output cap or the request is rejected outright.
            max_tokens: 20_000,
            budget: "semantic-extraction",
          },
          toolName: EXTRACT_SEMANTIC_TOOL_NAME,
          parse: parseResponse,
        })
      ).parsed;
    } catch (error) {
      if (isStructuredToolCallError(error, "llm_failed")) {
        this.traceExtractorInvoked({
          inputEpisodeCount: episodes.length,
          parsedNodeCount: 0,
          parsedEdgeCount: 0,
          acceptedNodeCount: 0,
          acceptedEdgeCount: 0,
          skipReasons: ["llm_failed"],
        });
        throw error.cause ?? error;
      }

      this.traceExtractorInvoked({
        inputEpisodeCount: episodes.length,
        parsedNodeCount: 0,
        parsedEdgeCount: 0,
        acceptedNodeCount: 0,
        acceptedEdgeCount: 0,
        skipReasons: ["parse_failed"],
      });

      if (isStructuredToolCallError(error, "missing_tool_call")) {
        throw new LLMError(`Semantic extractor did not emit tool ${EXTRACT_SEMANTIC_TOOL_NAME}`, {
          code: "SEMANTIC_EXTRACTOR_INVALID",
        });
      }

      throw isStructuredToolCallError(error, "invalid_payload") ? (error.cause ?? error) : error;
    }

    const allowedEpisodeIds = new Set(episodes.map((episode) => episode.id));
    const episodeById = new Map(episodes.map((episode) => [episode.id, episode]));
    const batchNodes = new Map<string, SemanticNode>();
    const skipReasons = new Set<string>();
    let insertedNodes = 0;
    let updatedNodes = 0;
    let skippedNodes = 0;
    let insertedEdges = 0;
    let updatedEdges = 0;
    let skippedEdges = parsed.schemaInvalidEdgeCount;
    let invalidEdgeSkips = parsed.schemaInvalidEdgeCount;
    const skippedEdgeDetails = [...parsed.schemaInvalidEdgeDetails];

    if (parsed.schemaInvalidEdgeCount > 0) {
      skipReasons.add("schema_invalid");

      for (let index = 0; index < parsed.schemaInvalidEdgeCount; index += 1) {
        this.traceInsertSkipped({
          kind: "edge",
          reason: "schema_invalid",
        });
      }
    }

    try {
      for (const candidate of parsed.nodes) {
        const outcome = await this.upsertNode(candidate, allowedEpisodeIds, episodeById);

        if (outcome.status === "inserted") {
          insertedNodes += 1;
        } else if (outcome.status === "updated") {
          updatedNodes += 1;
          if (outcome.reason !== undefined) {
            skipReasons.add(outcome.reason);
            this.traceInsertSkipped({
              kind: "node",
              reason: outcome.reason,
            });
          }
        } else {
          skippedNodes += 1;
          const reason = outcome.reason ?? "other";
          skipReasons.add(reason);
          this.traceInsertSkipped({
            kind: "node",
            reason,
            ...(reason === "relationship_claim_ungrounded"
              ? {
                  relationshipClaims: candidate.relationship_claims,
                  ungroundedRelationshipClaims: outcome.ungroundedRelationshipClaims ?? [],
                }
              : {}),
          });
        }

        if (outcome.node !== undefined) {
          const key = outcome.node.label.toLowerCase();
          batchNodes.set(key, outcome.node);

          for (const alias of outcome.node.aliases) {
            batchNodes.set(alias.toLowerCase(), outcome.node);
          }
        }
      }

      // Insert/update nodes before edges so endpoint validation never sees
      // dangling in-batch references.
      for (const { candidate, candidateIndex } of parsed.edges) {
        let evidenceEpisodeIds: Episode["id"][];

        try {
          evidenceEpisodeIds = this.validateEpisodeRefs(
            candidate.evidence_episode_ids,
            allowedEpisodeIds,
            "evidence_episode_ids",
          );
        } catch (error) {
          if (error instanceof SemanticError && error.code === "SEMANTIC_EXTRACTOR_INVALID_REF") {
            skippedEdges += 1;
            invalidEdgeSkips += 1;
            skipReasons.add("invalid_ref");
            pushSkippedEdgeTraceDetail(
              skippedEdgeDetails,
              skippedEdgeTraceDetailFromCandidate(candidateIndex, "invalid_ref", candidate),
            );
            this.traceInsertSkipped({
              kind: "edge",
              reason: "invalid_ref",
            });
            continue;
          }

          throw error;
        }

        const fromNode = await this.resolveEdgeNode(candidate.from_label, batchNodes);
        const toNode = await this.resolveEdgeNode(candidate.to_label, batchNodes);

        if (fromNode === undefined || toNode === undefined) {
          skippedEdges += 1;
          invalidEdgeSkips += 1;
          skipReasons.add("invalid_endpoint");
          pushSkippedEdgeTraceDetail(
            skippedEdgeDetails,
            skippedEdgeTraceDetailFromCandidate(candidateIndex, "invalid_endpoint", candidate),
          );
          this.traceInsertSkipped({
            kind: "edge",
            reason: "invalid_endpoint",
          });
          continue;
        }

        if (fromNode.id === toNode.id) {
          skippedEdges += 1;
          invalidEdgeSkips += 1;
          skipReasons.add("invalid_endpoint");
          pushSkippedEdgeTraceDetail(
            skippedEdgeDetails,
            skippedEdgeTraceDetailFromCandidate(candidateIndex, "invalid_endpoint", candidate),
          );
          this.traceInsertSkipped({
            kind: "edge",
            reason: "invalid_endpoint",
          });
          continue;
        }

        try {
          const nowMs = this.clock.now();

          this.options.edgeRepository.addEdge(
            {
              from_node_id: fromNode.id,
              to_node_id: toNode.id,
              relation: candidate.relation,
              confidence: Math.min(candidate.confidence, this.confidenceCeiling),
              evidence_episode_ids: evidenceEpisodeIds,
              created_at: nowMs,
              last_verified_at: nowMs,
              ...(candidate.valid_from_ts === null ? {} : { valid_from: candidate.valid_from_ts }),
              ...(candidate.valid_to_ts === null ? {} : { valid_to: candidate.valid_to_ts }),
            },
            this.options.reviewEnqueue === undefined
              ? undefined
              : { enqueueReview: this.options.reviewEnqueue },
          );
          insertedEdges += 1;
        } catch (error) {
          if (error instanceof SemanticError) {
            if (error.code === "SEMANTIC_EDGE_DUPLICATE") {
              const nowMs = this.clock.now();
              const merged = this.options.edgeRepository.mergeOpenEdgeEvidence({
                from_node_id: fromNode.id,
                to_node_id: toNode.id,
                relation: candidate.relation,
                confidence: Math.min(candidate.confidence, this.confidenceCeiling),
                evidence_episode_ids: evidenceEpisodeIds,
                last_verified_at: nowMs,
                ...(candidate.valid_from_ts === null
                  ? {}
                  : { valid_from: candidate.valid_from_ts }),
                ...(candidate.valid_to_ts === null ? {} : { valid_to: candidate.valid_to_ts }),
              });

              if (merged !== null) {
                updatedEdges += 1;
                continue;
              }
            }

            const reason = this.edgeSkipReason(error);
            skippedEdges += 1;
            skipReasons.add(reason);
            pushSkippedEdgeTraceDetail(
              skippedEdgeDetails,
              skippedEdgeTraceDetailFromCandidate(candidateIndex, reason, candidate),
            );
            this.traceInsertSkipped({
              kind: "edge",
              reason,
            });
            continue;
          }

          throw error;
        }
      }
    } catch (error) {
      if (error instanceof SemanticError && error.code === "SEMANTIC_EXTRACTOR_INVALID_REF") {
        skipReasons.add("invalid_ref");
      }

      this.traceExtractorInvoked({
        inputEpisodeCount: episodes.length,
        parsedNodeCount: parsed.nodes.length,
        parsedEdgeCount: parsed.rawEdgeCount,
        acceptedNodeCount: insertedNodes + updatedNodes,
        acceptedEdgeCount: insertedEdges + updatedEdges,
        skipReasons: [...skipReasons],
      });
      throw error;
    }

    if (invalidEdgeSkips > 0 && insertedNodes + updatedNodes + insertedEdges + updatedEdges > 0) {
      this.tracePartialFailure({
        inputEpisodeCount: episodes.length,
        parsedNodeCount: parsed.nodes.length,
        parsedEdgeCount: parsed.rawEdgeCount,
        acceptedNodeCount: insertedNodes + updatedNodes,
        acceptedEdgeCount: insertedEdges + updatedEdges,
        skippedNodeCount: skippedNodes,
        skippedEdgeCount: skippedEdges,
        skipReasons: [...skipReasons],
        skippedEdgeDetails,
      });
    }

    this.traceExtractorInvoked({
      inputEpisodeCount: episodes.length,
      parsedNodeCount: parsed.nodes.length,
      parsedEdgeCount: parsed.rawEdgeCount,
      acceptedNodeCount: insertedNodes + updatedNodes,
      acceptedEdgeCount: insertedEdges + updatedEdges,
      skipReasons: [...skipReasons],
    });

    return {
      insertedNodes,
      updatedNodes,
      skippedNodes,
      insertedEdges,
      updatedEdges,
      skippedEdges,
    };
  }
}
