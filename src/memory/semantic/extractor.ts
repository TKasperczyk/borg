import { z } from "zod";

import type { EmbeddingClient } from "../../embeddings/index.js";
import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { LLMError, SemanticError, StorageError } from "../../util/errors.js";
import { createSemanticNodeId } from "../../util/ids.js";
import { episodeAccessScopeKey } from "../episodic/access.js";
import type { Episode, EpisodicRepository } from "../episodic/index.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
} from "./repository.js";
import { canonicalizeDomain } from "./domain.js";
import {
  semanticNodeKindSchema,
  semanticRelationSchema,
  type SemanticEdge,
  type SemanticNode,
} from "./types.js";

const extractorNodeSchema = z.object({
  kind: semanticNodeKindSchema,
  label: z.string().min(1),
  description: z.string().min(1),
  domain: z.string().min(1).nullable().default(null),
  aliases: z.array(z.string().min(1)),
  confidence: z.number().min(0).max(1),
  source_episode_ids: z.array(z.string().min(1)).min(1),
});

const extractorEdgeSchema = z.object({
  from_label: z.string().min(1),
  to_label: z.string().min(1),
  relation: semanticRelationSchema,
  confidence: z.number().min(0).max(1),
  evidence_episode_ids: z.array(z.string().min(1)).min(1),
});

const extractorResponseSchema = z.object({
  nodes: z.array(extractorNodeSchema).max(6),
  edges: z.array(extractorEdgeSchema).max(8),
});

type ExtractorNode = z.infer<typeof extractorNodeSchema>;
type ExtractorEdge = z.infer<typeof extractorEdgeSchema>;

const DEFAULT_CONFIDENCE_CEILING = 0.7;
const DEDUP_THRESHOLD = 0.88;
const EXTRACT_SEMANTIC_TOOL_NAME = "EmitSemanticCandidates";
export const EXTRACT_SEMANTIC_TOOL = {
  name: EXTRACT_SEMANTIC_TOOL_NAME,
  description: "Emit grounded semantic nodes and edges extracted from episodes.",
  inputSchema: toToolInputSchema(extractorResponseSchema),
} satisfies LLMToolDefinition;

export type SemanticExtractorOptions = {
  nodeRepository: SemanticNodeRepository;
  edgeRepository: SemanticEdgeRepository;
  embeddingClient: EmbeddingClient;
  episodicRepository: Pick<EpisodicRepository, "getMany">;
  llmClient: LLMClient;
  model: string;
  clock?: Clock;
  dedupThreshold?: number;
  confidenceCeiling?: number;
};

export type ExtractSemanticResult = {
  insertedNodes: number;
  updatedNodes: number;
  skippedNodes: number;
  insertedEdges: number;
  skippedEdges: number;
};

function buildPrompt(episodes: readonly Episode[]): string {
  return [
    "Extract semantic knowledge from the provided episodes.",
    `Emit your result by calling the ${EXTRACT_SEMANTIC_TOOL_NAME} tool exactly once.`,
    "Populate the tool arguments directly with arrays and objects. Do not put JSON, XML tags, or parameter wrappers inside string fields.",
    "Return at most 6 nodes and 8 edges. Prefer the most central concepts and claims over peripheral details.",
    "Each node must cite source_episode_ids from the provided episode ids only.",
    "Each edge must use from_label and to_label values that match node labels exactly.",
    "Only use relation values allowed by the tool schema.",
    "Edges may only reference nodes that already exist or are extracted in this batch.",
    'Emit a free-form domain string for each node when it helps disambiguate homonyms. Prefer stable canonical buckets when they fit (examples: "tech", "people", "places", "food", "process"). Use null for broadly general nodes.',
    "Keep confidence modest for fresh extractions.",
    "Episodes:",
    ...episodes.map((episode) =>
      JSON.stringify({
        id: episode.id,
        title: episode.title,
        narrative: episode.narrative,
        participants: episode.participants,
        audience_entity_id: episode.audience_entity_id ?? null,
        shared: episode.shared ?? (episode.audience_entity_id ?? null) === null,
        location: episode.location,
        tags: episode.tags,
      }),
    ),
  ].join("\n");
}

function parseResponse(result: LLMCompleteResult): {
  nodes: ExtractorNode[];
  edges: ExtractorEdge[];
} {
  const call = result.tool_calls.find((toolCall) => toolCall.name === EXTRACT_SEMANTIC_TOOL_NAME);

  if (call === undefined) {
    throw new LLMError(`Semantic extractor did not emit tool ${EXTRACT_SEMANTIC_TOOL_NAME}`, {
      code: "SEMANTIC_EXTRACTOR_INVALID",
    });
  }

  const parsed = extractorResponseSchema.safeParse(normalizeSemanticToolInput(call.input));

  if (!parsed.success) {
    throw new LLMError("Semantic extractor returned invalid payload", {
      cause: parsed.error,
      code: "SEMANTIC_EXTRACTOR_INVALID",
    });
  }

  return parsed.data;
}

function parseJsonArrayString(value: string): unknown {
  const trimmed = value.trim();

  if (!trimmed.startsWith("[") && !trimmed.startsWith("{")) {
    return value;
  }

  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return value;
  }
}

function normalizeSemanticToolInput(input: unknown): unknown {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    return input;
  }

  const record = input as Record<string, unknown>;
  const normalized: Record<string, unknown> = { ...record };
  const rawNodes = normalized.nodes;

  if (typeof rawNodes === "string") {
    const parts = rawNodes.split(/\s*<parameter name="edges">\s*/);

    if (parts.length === 2) {
      normalized.nodes = parseJsonArrayString(parts[0] ?? "");
      normalized.edges = parseJsonArrayString(parts[1] ?? "");
      return normalized;
    }

    normalized.nodes = parseJsonArrayString(rawNodes);
  }

  if (typeof normalized.edges === "string") {
    normalized.edges = parseJsonArrayString(normalized.edges);
  }

  if (normalized.nodes === undefined) {
    normalized.nodes = [];
  }

  if (normalized.edges === undefined) {
    normalized.edges = [];
  }

  return normalized;
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
}): string {
  return `${input.label}\n${input.description}\n${input.aliases.join(" ")}`;
}

function hasCompatibleDomain(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  const normalizedLeft = canonicalizeDomain(left);
  const normalizedRight = canonicalizeDomain(right);

  return normalizedLeft !== null && normalizedRight !== null && normalizedLeft === normalizedRight;
}

function resolveEpisodeScopeKeys(episodes: readonly Episode[]): Set<string> {
  return new Set(episodes.map((episode) => episodeAccessScopeKey(episode)));
}

function haveSameScopeKeys(
  left: ReadonlySet<string>,
  right: ReadonlySet<string>,
): boolean {
  return (
    left.size === right.size &&
    [...left].every((value) => right.has(value))
  );
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

  private async upsertNode(
    candidate: ExtractorNode,
    allowedEpisodeIds: ReadonlySet<string>,
    episodeById: ReadonlyMap<Episode["id"], Episode>,
  ): Promise<{ status: "inserted" | "updated" | "skipped"; node?: SemanticNode }> {
    const sourceEpisodeIds = this.validateEpisodeRefs(
      candidate.source_episode_ids,
      allowedEpisodeIds,
      "source_episode_ids",
    );
    const sourceEpisodes = sourceEpisodeIds
      .map((episodeId) => episodeById.get(episodeId))
      .filter((episode): episode is Episode => episode !== undefined);
    const candidateScopeKeys = resolveEpisodeScopeKeys(sourceEpisodes);

    try {
      const candidateLabel = candidate.label.trim();
      const candidateDescription = candidate.description.trim();
      const candidateAliases = mergeAliases(candidate.aliases, []);
      const embedding = await this.options.embeddingClient.embed(
        buildNodeEmbeddingText({
          label: candidateLabel,
          description: candidateDescription,
          aliases: candidateAliases,
        }),
      );
      const isCompatibleNode = async (node: SemanticNode): Promise<boolean> => {
        if (node.kind !== candidate.kind) {
          return false;
        }

        if (!hasCompatibleDomain(node.domain, candidate.domain)) {
          return false;
        }

        const nodeSourceEpisodes = await this.options.episodicRepository.getMany(node.source_episode_ids);
        const nodeScopeKeys = resolveEpisodeScopeKeys(nodeSourceEpisodes);

        return (
          nodeSourceEpisodes.length === node.source_episode_ids.length &&
          haveSameScopeKeys(nodeScopeKeys, candidateScopeKeys)
        );
      };
      const byLabelMatches = await this.options.nodeRepository.findByLabelOrAlias(candidate.label, 5, {
        includeArchived: true,
      });
      const byLabel: SemanticNode[] = [];

      for (const match of byLabelMatches) {
        if (await isCompatibleNode(match)) {
          byLabel.push(match);
        }
      }

      const byVectorMatches = await this.options.nodeRepository.searchByVector(embedding, {
        limit: 3,
        minSimilarity: this.dedupThreshold,
      });
      const byVector: Array<{ node: SemanticNode; similarity: number }> = [];

      for (const match of byVectorMatches) {
        if (await isCompatibleNode(match.node)) {
          byVector.push(match);
        }
      }

      const existing = byLabel[0] ?? byVector[0]?.node;
      const nowMs = this.clock.now();

      if (existing === undefined) {
        const inserted = await this.options.nodeRepository.insert({
          id: createSemanticNodeId(),
          kind: candidate.kind,
          label: candidateLabel,
          description: candidateDescription,
          domain: canonicalizeDomain(candidate.domain),
          aliases: candidateAliases,
          confidence: Math.min(candidate.confidence, this.confidenceCeiling),
          source_episode_ids: sourceEpisodeIds,
          created_at: nowMs,
          updated_at: nowMs,
          last_verified_at: nowMs,
          embedding,
          archived: false,
          superseded_by: null,
        });

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
        }),
      );
      const updated = await this.options.nodeRepository.update(existing.id, {
        description: nextDescription,
        domain: existing.domain ?? canonicalizeDomain(candidate.domain),
        aliases: nextAliases,
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
          }
        : {
            status: "updated",
            node: updated,
          };
    } catch (error) {
      if (error instanceof StorageError || error instanceof SemanticError) {
        return {
          status: "skipped",
        };
      }

      throw error;
    }
  }

  private resolveEdgeNode(
    label: string,
    batchNodes: ReadonlyMap<string, SemanticNode>,
    existingNodes: ReadonlyMap<string, SemanticNode>,
  ): SemanticNode | undefined {
    return batchNodes.get(label.toLowerCase()) ?? existingNodes.get(label.toLowerCase());
  }

  async extractFromEpisodes(episodes: readonly Episode[]): Promise<ExtractSemanticResult> {
    if (episodes.length === 0) {
      return {
        insertedNodes: 0,
        updatedNodes: 0,
        skippedNodes: 0,
        insertedEdges: 0,
        skippedEdges: 0,
      };
    }

    const result = await this.options.llmClient.complete({
      model: this.options.model,
      system: "Extract semantic nodes and edges grounded only in the provided episodes.",
      messages: [
        {
          role: "user",
          content: buildPrompt(episodes),
        },
      ],
      tools: [EXTRACT_SEMANTIC_TOOL],
      tool_choice: { type: "tool", name: EXTRACT_SEMANTIC_TOOL_NAME },
      max_tokens: 12_000,
      budget: "semantic-extraction",
    });
    const parsed = parseResponse(result);
    const allowedEpisodeIds = new Set(episodes.map((episode) => episode.id));
    const episodeById = new Map(episodes.map((episode) => [episode.id, episode]));
    const existingNodes = new Map<string, SemanticNode>();
    const batchNodes = new Map<string, SemanticNode>();
    let insertedNodes = 0;
    let updatedNodes = 0;
    let skippedNodes = 0;
    let insertedEdges = 0;
    let skippedEdges = 0;

    for (const candidate of parsed.nodes) {
      const outcome = await this.upsertNode(candidate, allowedEpisodeIds, episodeById);

      if (outcome.status === "inserted") {
        insertedNodes += 1;
      } else if (outcome.status === "updated") {
        updatedNodes += 1;
      } else {
        skippedNodes += 1;
      }

      if (outcome.node !== undefined) {
        const key = outcome.node.label.toLowerCase();
        batchNodes.set(key, outcome.node);

        for (const alias of outcome.node.aliases) {
          batchNodes.set(alias.toLowerCase(), outcome.node);
        }
      }
    }

    for (const candidate of parsed.nodes) {
      const matches = await this.options.nodeRepository.findByLabelOrAlias(candidate.label, 3, {
        includeArchived: true,
      });

      for (const match of matches) {
        if (!existingNodes.has(match.label.toLowerCase())) {
          existingNodes.set(match.label.toLowerCase(), match);
        }

        for (const alias of match.aliases) {
          if (!existingNodes.has(alias.toLowerCase())) {
            existingNodes.set(alias.toLowerCase(), match);
          }
        }
      }
    }

    // Insert/update nodes before edges so endpoint validation never sees
    // dangling in-batch references.
    for (const candidate of parsed.edges) {
      const fromNode = this.resolveEdgeNode(candidate.from_label, batchNodes, existingNodes);
      const toNode = this.resolveEdgeNode(candidate.to_label, batchNodes, existingNodes);

      if (fromNode === undefined || toNode === undefined) {
        throw new SemanticError("Semantic extractor referenced an unknown edge node", {
          code: "SEMANTIC_EXTRACTOR_INVALID_REF",
        });
      }

      if (fromNode.id === toNode.id) {
        skippedEdges += 1;
        continue;
      }

      const evidenceEpisodeIds = this.validateEpisodeRefs(
        candidate.evidence_episode_ids,
        allowedEpisodeIds,
        "evidence_episode_ids",
      );

      try {
        this.options.edgeRepository.addEdge({
          from_node_id: fromNode.id,
          to_node_id: toNode.id,
          relation: candidate.relation,
          confidence: Math.min(candidate.confidence, this.confidenceCeiling),
          evidence_episode_ids: evidenceEpisodeIds,
          created_at: this.clock.now(),
          last_verified_at: this.clock.now(),
        });
        insertedEdges += 1;
      } catch (error) {
        if (error instanceof SemanticError) {
          skippedEdges += 1;
          continue;
        }

        throw error;
      }
    }

    return {
      insertedNodes,
      updatedNodes,
      skippedNodes,
      insertedEdges,
      skippedEdges,
    };
  }
}
