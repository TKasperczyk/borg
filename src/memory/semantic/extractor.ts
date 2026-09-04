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
import { relationshipClaimSchema, type RelationshipClaim } from "../common/relationship-claims.js";
import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  GENERIC_SELF_ENTITY_VOICE_ANCHOR,
  SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE,
} from "../../util/self-memory-voice.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { LLMError, SemanticError, StorageError } from "../../util/errors.js";
import { createSemanticNodeId, type EntityId, type StreamEntryId } from "../../util/ids.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import { memoryDisclosureLabelFromEpisodeAccess } from "../common/disclosure-label.js";
import {
  episodeParticipantDisplayNames,
  episodeParticipantEntityIds,
  normalizeEpisodeAccess,
  type Episode,
  type EpisodicRepository,
} from "../episodic/index.js";
import type { EntityRecord, EntityRepository } from "../commitments/index.js";
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

const extractorNodeBaseSchema = z.object({
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
const extractorDescriptionPerspectiveSchema = z.enum([
  "first_person",
  "third_person",
  "impersonal",
]);
const extractorNodeSchema = extractorNodeBaseSchema.extend({
  identity_entity_id: z.string().min(1).nullable().default(null),
  description_perspective: z
    .enum(["first_person", "third_person", "impersonal", "unspecified"])
    .default("unspecified"),
});
const extractorNodeToolSchema = extractorNodeBaseSchema.extend({
  identity_entity_id: z
    .string()
    .min(1)
    .nullable()
    .describe(
      "Repository entity id when this node denotes one supplied identity anchor; null otherwise.",
    ),
  description_perspective: extractorDescriptionPerspectiveSchema.describe(
    "Grammatical perspective of the description, used for identity attribution validation.",
  ),
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
  nodes: z.array(z.unknown()),
  edges: z.array(z.unknown()).max(MAX_EXTRACTOR_EDGES),
});
const extractorResponseToolSchema = z.object({
  nodes: z.array(extractorNodeToolSchema).max(MAX_EXTRACTOR_NODES),
  edges: z.array(extractorEdgeSchema).max(MAX_EXTRACTOR_EDGES),
});

type ExtractorNode = z.infer<typeof extractorNodeSchema>;
type ExtractorEdge = z.infer<typeof extractorEdgeSchema>;
type ParsedExtractorEdge = {
  candidate: ExtractorEdge;
  candidateIndex: number;
};
type SemanticIdentityAnchor = Pick<EntityRecord, "id" | "canonical_name" | "aliases" | "kind">;
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
  entityRepository?: Pick<EntityRepository, "get">;
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
  | "over_cap"
  | "invalid_ref"
  | "invalid_endpoint"
  | "identity_guard"
  | "validity_window_conflict"
  | "episode_archived_post_plan"
  | "relationship_claim_ungrounded"
  | "other";

type SkippedNodeTraceDetail = {
  candidate_index: number;
  reason: Extract<SemanticInsertSkipReason, "schema_invalid" | "over_cap">;
};

function buildPrompt(input: {
  episodes: readonly Episode[];
  participantRoster?: ParticipantRosterForRendering | null;
  selfEntityId?: EntityId | null;
  identityAnchors: readonly SemanticIdentityAnchor[];
  knownNodeKinds: readonly SemanticNodeKind[];
}): string {
  const roster = renderParticipantRoster(input.participantRoster);
  const selfEntityGuidance =
    input.selfEntityId === undefined || input.selfEntityId === null
      ? GENERIC_SELF_ENTITY_VOICE_ANCHOR
      : `Entity ${input.selfEntityId} is yourself; write your own actions, statements, and decisions in first person, and refer to every other entity by name or stable handle.`;
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
    "Identity guards: when a node denotes an entity in identity_anchors_by_entity_id, copy that entity's id into identity_entity_id. Use null only when the node does not denote a supplied identity anchor; never invent an entity id.",
    'Set description_perspective to "first_person", "third_person", or "impersonal" according to the description you emit.',
    "A person identity anchor must use kind person and must be described in third person, by name. Never write a non-self person's statements, actions, work, or history as my own.",
    "Every node that denotes the self must use the single supplied self identity_entity_id and canonical self label. Do not coin parallel self, assistant, agent, or first-person identity nodes.",
    "Never emit is_a or instance_of from a person identity node into the self identity node.",
    "When a node label or description asserts a sensitive interpersonal relationship, emit relationship_claims with supporting evidence ids. Do not cite assistant output as relationship evidence. If no accepted evidence grounds the claim, rewrite the node neutrally before emitting it.",
    roster === null ? "Thread roster: none supplied." : roster,
    selfEntityGuidance,
    "Keep confidence modest for fresh extractions.",
    "<identity_anchors_by_entity_id>",
    JSON.stringify(
      Object.fromEntries(
        input.identityAnchors.map((anchor) => [
          anchor.id,
          {
            entity_id: anchor.id,
            display_name: anchor.canonical_name,
            aliases: anchor.aliases,
            kind: anchor.kind,
          },
        ]),
      ),
    ),
    "</identity_anchors_by_entity_id>",
    "Episodes:",
    ...input.episodes.map((episode) => {
      const access = normalizeEpisodeAccess(episode);
      const disclosureLabel = memoryDisclosureLabelFromEpisodeAccess(episode);

      return JSON.stringify({
        id: episode.id,
        title: episode.title,
        narrative: episode.narrative,
        participants: episodeParticipantDisplayNames(episode.participants),
        participant_entities: episodeParticipantEntityIds(episode.participants).map((entityId) => ({
          entity_id: entityId,
          display_name:
            input.identityAnchors.find((anchor) => anchor.id === entityId)?.canonical_name ?? null,
        })),
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
  rawNodeCount: number;
  retainedNodeCount: number;
  schemaInvalidNodeCount: number;
  overCapNodeCount: number;
  skippedNodeDetails: SkippedNodeTraceDetail[];
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

  const nodes: ExtractorNode[] = [];
  const skippedNodeDetails: SkippedNodeTraceDetail[] = [];
  const retainedNodes = parsed.data.nodes.slice(0, MAX_EXTRACTOR_NODES);
  let schemaInvalidNodeCount = 0;

  for (const [candidateIndex, node] of retainedNodes.entries()) {
    const parsedNode = extractorNodeSchema.safeParse(node);

    if (!parsedNode.success) {
      schemaInvalidNodeCount += 1;
      skippedNodeDetails.push({
        candidate_index: candidateIndex,
        reason: "schema_invalid",
      });
      continue;
    }

    nodes.push(parsedNode.data);
  }

  const overCapNodeCount = parsed.data.nodes.length - retainedNodes.length;
  for (
    let candidateIndex = retainedNodes.length;
    candidateIndex < parsed.data.nodes.length;
    candidateIndex += 1
  ) {
    skippedNodeDetails.push({
      candidate_index: candidateIndex,
      reason: "over_cap",
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
    nodes,
    edges,
    rawNodeCount: parsed.data.nodes.length,
    retainedNodeCount: retainedNodes.length,
    schemaInvalidNodeCount,
    overCapNodeCount,
    skippedNodeDetails,
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

function participantRosterEntityIds(
  roster: ParticipantRosterForRendering | null | undefined,
): EntityId[] {
  if (roster === null || roster === undefined) {
    return [];
  }

  return [
    ...roster.participants.map((participant) => participant.entity_id as EntityId),
    ...roster.non_chat_subjects.map((subject) => subject.entity_id as EntityId),
    ...roster.unknown_or_uncertain.flatMap((item) =>
      item.entity_id === null ? [] : [item.entity_id as EntityId],
    ),
  ];
}

function identityAnchorsForExtraction(
  options: Pick<
    SemanticExtractorOptions,
    "entityRepository" | "participantRoster" | "selfEntityId"
  >,
): SemanticIdentityAnchor[] {
  if (options.entityRepository === undefined) {
    return [];
  }

  const ids = new Set<EntityId>(participantRosterEntityIds(options.participantRoster));

  if (options.selfEntityId !== null && options.selfEntityId !== undefined) {
    ids.add(options.selfEntityId);
  }

  return [...ids].flatMap((entityId) => {
    const entity = options.entityRepository?.get(entityId);

    return entity === null || entity === undefined
      ? []
      : [
          {
            id: entity.id,
            canonical_name: entity.canonical_name,
            aliases: [...entity.aliases],
            kind: entity.kind,
          },
        ];
  });
}

function identityAnchorNames(anchor: SemanticIdentityAnchor): string[] {
  return [anchor.canonical_name, ...anchor.aliases].map((name) => name.trim());
}

function identityAnchorForCandidate(
  candidate: ExtractorNode,
  anchors: readonly SemanticIdentityAnchor[],
): {
  anchor: SemanticIdentityAnchor | null;
  invalidExplicitRef: boolean;
  ambiguousKnownLabel: boolean;
} {
  const label = candidate.label.trim();
  const labelMatches = anchors.filter((anchor) =>
    identityAnchorNames(anchor).some((name) => name === label),
  );
  const labelAnchor = labelMatches.length === 1 ? (labelMatches[0] ?? null) : null;

  if (candidate.identity_entity_id !== null) {
    const anchor = anchors.find((item) => item.id === candidate.identity_entity_id);

    return {
      anchor: anchor ?? null,
      invalidExplicitRef:
        anchor === undefined || (labelAnchor !== null && labelAnchor.id !== anchor.id),
      ambiguousKnownLabel: false,
    };
  }

  const selfAnchors = anchors.filter((anchor) => anchor.kind === "self");
  const isStructurallySelfReferential =
    candidate.kind === "self" ||
    (candidate.kind === "agent" && candidate.description_perspective === "first_person");

  return {
    anchor:
      labelAnchor ??
      (isStructurallySelfReferential && selfAnchors.length === 1 ? (selfAnchors[0] ?? null) : null),
    invalidExplicitRef: false,
    ambiguousKnownLabel: labelMatches.length > 1,
  };
}

function identityAnchorForNode(
  node: SemanticNode,
  anchors: readonly SemanticIdentityAnchor[],
): SemanticIdentityAnchor | null {
  const nodeNames = new Set([node.label.trim(), ...node.aliases.map((alias) => alias.trim())]);
  const matches = anchors.filter((anchor) =>
    identityAnchorNames(anchor).some((name) => nodeNames.has(name)),
  );

  return matches.length === 1 ? (matches[0] ?? null) : null;
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
    candidateIndex?: number;
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
      ...(input.candidateIndex === undefined ? {} : { candidate_index: input.candidateIndex }),
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
    rawNodeCount: number;
    retainedNodeCount: number;
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
      raw_node_count: input.rawNodeCount,
      retained_node_count: input.retainedNodeCount,
      parsed_node_count: input.parsedNodeCount,
      parsed_edge_count: input.parsedEdgeCount,
      accepted_node_count: input.acceptedNodeCount,
      accepted_edge_count: input.acceptedEdgeCount,
      skip_reasons: [...input.skipReasons],
    });
  }

  private tracePartialFailure(input: {
    inputEpisodeCount: number;
    rawNodeCount: number;
    retainedNodeCount: number;
    parsedNodeCount: number;
    parsedEdgeCount: number;
    acceptedNodeCount: number;
    acceptedEdgeCount: number;
    skippedNodeCount: number;
    skippedEdgeCount: number;
    skipReasons: readonly string[];
    skippedNodeDetails: readonly SkippedNodeTraceDetail[];
    skippedEdgeDetails: readonly SkippedEdgeTraceDetail[];
  }): void {
    if (this.options.tracer?.enabled !== true) {
      return;
    }

    this.options.tracer.emit("semantic_extractor.degraded", {
      turnId: this.options.traceTurnId ?? "semantic_extractor",
      input_episode_count: input.inputEpisodeCount,
      raw_node_count: input.rawNodeCount,
      retained_node_count: input.retainedNodeCount,
      parsed_node_count: input.parsedNodeCount,
      parsed_edge_count: input.parsedEdgeCount,
      accepted_node_count: input.acceptedNodeCount,
      accepted_edge_count: input.acceptedEdgeCount,
      skipped_node_count: input.skippedNodeCount,
      skipped_edge_count: input.skippedEdgeCount,
      skip_reasons: [...input.skipReasons],
      skipped_node_details: input.skippedNodeDetails.slice(0, SKIPPED_EDGE_TRACE_DETAIL_LIMIT),
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
    identityAnchors: readonly SemanticIdentityAnchor[],
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

    const identityResolution = identityAnchorForCandidate(candidate, identityAnchors);

    if (identityResolution.invalidExplicitRef) {
      return {
        status: "skipped",
        reason: "invalid_ref",
      };
    }

    if (identityResolution.ambiguousKnownLabel) {
      return {
        status: "skipped",
        reason: "identity_guard",
      };
    }

    const identityAnchor = identityResolution.anchor;

    const isFirstPerson = candidate.description_perspective === "first_person";
    const resolvesToSelf = identityAnchor?.kind === "self";
    const isPersonCandidate = candidate.kind === "person" || identityAnchor?.kind === "person";
    const isSelfKindCandidate = candidate.kind === "self" || candidate.kind === "agent";

    if (isFirstPerson && !resolvesToSelf && (isPersonCandidate || isSelfKindCandidate)) {
      return {
        status: "skipped",
        reason: "identity_guard",
      };
    }

    try {
      const candidateLabel = identityAnchor?.canonical_name ?? candidate.label.trim();
      const candidateDescription = candidate.description.trim();
      const candidateAliases = mergeAliases(
        candidate.aliases,
        identityAnchor === null
          ? []
          : [candidate.label, ...identityAnchor.aliases].filter(
              (alias) => alias.trim() !== candidateLabel,
            ),
      );
      const candidateKind =
        identityAnchor?.kind === "person"
          ? "person"
          : identityAnchor?.kind === "self"
            ? "self"
            : candidate.kind;
      const embedding = await this.options.embeddingClient.embed(
        buildNodeEmbeddingText({
          label: candidateLabel,
          description: candidateDescription,
          aliases: candidateAliases,
          observationMetadata: candidate.observation_metadata,
        }),
      );
      const isCompatibleNode = (node: SemanticNode): boolean => {
        if (node.kind !== candidateKind) {
          return false;
        }

        if (identityAnchor !== null) {
          return (
            node.label === candidateLabel || node.aliases.some((alias) => alias === candidateLabel)
          );
        }

        if (!observationMetadataAligns(node.observation_metadata, candidate.observation_metadata)) {
          return false;
        }

        return cosineSimilarity(node.embedding, embedding) >= this.dedupThreshold;
      };
      const byLabelMatches = await this.options.nodeRepository.findByExactLabelOrAlias(
        candidateLabel,
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
          kind: candidateKind,
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
    const identityAnchors = identityAnchorsForExtraction(this.options);
    let parsed: ReturnType<typeof parseResponse>;

    try {
      parsed = (
        await callStructuredTool({
          llmClient: this.options.llmClient,
          request: {
            model: this.options.model,
            system: [
              "Extract semantic nodes and edges grounded only in the provided episodes.",
              `${SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE} Apply this to a node's label and description when the node is about the self entity. Keep nodes about the world or about other participants in their natural third-person form, naming those participants.`,
            ].join(" "),
            messages: [
              {
                role: "user",
                content: buildPrompt({
                  episodes,
                  participantRoster: this.options.participantRoster ?? null,
                  selfEntityId: this.options.selfEntityId ?? null,
                  identityAnchors,
                  knownNodeKinds,
                }),
              },
            ],
            tools: [EXTRACT_SEMANTIC_TOOL],
            tool_choice: { type: "tool", name: EXTRACT_SEMANTIC_TOOL_NAME },
            // Large structured batches benefit from a generous requested output
            // budget. This is a request, not a transport mandate: the LLM client
            // clamps it to the selected model's output ceiling before sending it.
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
          rawNodeCount: 0,
          retainedNodeCount: 0,
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
        rawNodeCount: 0,
        retainedNodeCount: 0,
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
    const parsedNodeSkips = parsed.schemaInvalidNodeCount + parsed.overCapNodeCount;
    let skippedNodes = parsedNodeSkips;
    let insertedEdges = 0;
    let updatedEdges = 0;
    let skippedEdges = parsed.schemaInvalidEdgeCount;
    let invalidEdgeSkips = parsed.schemaInvalidEdgeCount;
    const skippedEdgeDetails = [...parsed.schemaInvalidEdgeDetails];

    for (const skippedNode of parsed.skippedNodeDetails) {
      skipReasons.add(skippedNode.reason);
      this.traceInsertSkipped({
        kind: "node",
        reason: skippedNode.reason,
        candidateIndex: skippedNode.candidate_index,
      });
    }

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
        const outcome = await this.upsertNode(
          candidate,
          allowedEpisodeIds,
          episodeById,
          identityAnchors,
        );

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

        const fromIdentity = identityAnchorForNode(fromNode, identityAnchors);
        const toIdentity = identityAnchorForNode(toNode, identityAnchors);
        const isPersonIntoSelfClassification =
          (fromIdentity?.kind === "person" || fromNode.kind === "person") &&
          (toIdentity?.kind === "self" || toNode.kind === "self") &&
          (candidate.relation === "is_a" || candidate.relation === "instance_of");

        if (isPersonIntoSelfClassification) {
          skippedEdges += 1;
          invalidEdgeSkips += 1;
          skipReasons.add("identity_guard");
          pushSkippedEdgeTraceDetail(
            skippedEdgeDetails,
            skippedEdgeTraceDetailFromCandidate(candidateIndex, "identity_guard", candidate),
          );
          this.traceInsertSkipped({
            kind: "edge",
            reason: "identity_guard",
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
        rawNodeCount: parsed.rawNodeCount,
        retainedNodeCount: parsed.retainedNodeCount,
        parsedNodeCount: parsed.nodes.length,
        parsedEdgeCount: parsed.rawEdgeCount,
        acceptedNodeCount: insertedNodes + updatedNodes,
        acceptedEdgeCount: insertedEdges + updatedEdges,
        skipReasons: [...skipReasons],
      });
      throw error;
    }

    if (
      (parsedNodeSkips > 0 || invalidEdgeSkips > 0) &&
      insertedNodes + updatedNodes + insertedEdges + updatedEdges > 0
    ) {
      this.tracePartialFailure({
        inputEpisodeCount: episodes.length,
        rawNodeCount: parsed.rawNodeCount,
        retainedNodeCount: parsed.retainedNodeCount,
        parsedNodeCount: parsed.nodes.length,
        parsedEdgeCount: parsed.rawEdgeCount,
        acceptedNodeCount: insertedNodes + updatedNodes,
        acceptedEdgeCount: insertedEdges + updatedEdges,
        skippedNodeCount: skippedNodes,
        skippedEdgeCount: skippedEdges,
        skipReasons: [...skipReasons],
        skippedNodeDetails: parsed.skippedNodeDetails,
        skippedEdgeDetails,
      });
    }

    this.traceExtractorInvoked({
      inputEpisodeCount: episodes.length,
      rawNodeCount: parsed.rawNodeCount,
      retainedNodeCount: parsed.retainedNodeCount,
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
