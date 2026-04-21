import { z } from "zod";

import type { EmbeddingClient } from "../../embeddings/index.js";
import type { LLMClient } from "../../llm/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { LLMError, SemanticError, StorageError } from "../../util/errors.js";
import { createSemanticNodeId } from "../../util/ids.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  type SemanticNodeRepositoryOptions,
} from "./repository.js";
import {
  semanticNodeKindSchema,
  semanticRelationSchema,
  type SemanticEdge,
  type SemanticNode,
} from "./types.js";
import type { Episode } from "../episodic/index.js";

const extractorNodeSchema = z.object({
  kind: semanticNodeKindSchema,
  label: z.string().min(1),
  description: z.string().min(1),
  aliases: z.array(z.string().min(1)).default([]),
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
  nodes: z.array(extractorNodeSchema).default([]),
  edges: z.array(extractorEdgeSchema).default([]),
});

type ExtractorNode = z.infer<typeof extractorNodeSchema>;
type ExtractorEdge = z.infer<typeof extractorEdgeSchema>;

const DEFAULT_CONFIDENCE_CEILING = 0.7;
const DEDUP_THRESHOLD = 0.88;

export type SemanticExtractorOptions = {
  nodeRepository: SemanticNodeRepository;
  edgeRepository: SemanticEdgeRepository;
  embeddingClient: EmbeddingClient;
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
    'Return strict JSON with shape {"nodes":[...],"edges":[...]} and no surrounding prose.',
    "Each node must cite source_episode_ids from the provided episode ids only.",
    "Each edge must use from_label and to_label values that match node labels exactly.",
    "Edges may only reference nodes that already exist or are extracted in this batch.",
    "Keep confidence modest for fresh extractions.",
    "Episodes:",
    ...episodes.map((episode) =>
      JSON.stringify({
        id: episode.id,
        title: episode.title,
        narrative: episode.narrative,
        participants: episode.participants,
        tags: episode.tags,
      }),
    ),
  ].join("\n");
}

function parseResponse(text: string): {
  nodes: ExtractorNode[];
  edges: ExtractorEdge[];
} {
  let raw: unknown;

  try {
    raw = JSON.parse(text) as unknown;
  } catch (error) {
    throw new LLMError("Semantic extractor returned non-JSON output", {
      cause: error,
      code: "SEMANTIC_EXTRACTOR_INVALID",
    });
  }

  const parsed = extractorResponseSchema.safeParse(raw);

  if (!parsed.success) {
    throw new LLMError("Semantic extractor returned invalid payload", {
      cause: parsed.error,
      code: "SEMANTIC_EXTRACTOR_INVALID",
    });
  }

  return parsed.data;
}

function mergeAliases(left: readonly string[], right: readonly string[]): string[] {
  return [
    ...new Set(
      [...left, ...right].map((value) => value.trim()).filter((value) => value.length > 0),
    ),
  ];
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
  ): Promise<{ status: "inserted" | "updated" | "skipped"; node?: SemanticNode }> {
    const sourceEpisodeIds = this.validateEpisodeRefs(
      candidate.source_episode_ids,
      allowedEpisodeIds,
      "source_episode_ids",
    );

    try {
      const embedding = await this.options.embeddingClient.embed(
        `${candidate.label}\n${candidate.description}\n${candidate.aliases.join(" ")}`,
      );
      const byLabel = await this.options.nodeRepository.findByLabelOrAlias(candidate.label, 5);
      const byVector = await this.options.nodeRepository.searchByVector(embedding, {
        limit: 3,
        minSimilarity: this.dedupThreshold,
      });
      const existing = byLabel[0] ?? byVector[0]?.node;
      const nowMs = this.clock.now();

      if (existing === undefined) {
        const inserted = await this.options.nodeRepository.insert({
          id: createSemanticNodeId(),
          kind: candidate.kind,
          label: candidate.label.trim(),
          description: candidate.description.trim(),
          aliases: mergeAliases(candidate.aliases, []),
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

      const updated = await this.options.nodeRepository.update(existing.id, {
        description:
          candidate.confidence >= existing.confidence
            ? candidate.description.trim()
            : existing.description,
        aliases: mergeAliases(existing.aliases, [candidate.label, ...candidate.aliases]),
        confidence: Math.max(
          existing.confidence * 0.99,
          Math.min(candidate.confidence, this.confidenceCeiling),
        ),
        source_episode_ids: sourceEpisodeIds,
        last_verified_at: nowMs,
        embedding,
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
      max_tokens: 1_200,
      budget: "semantic-extraction",
    });
    const parsed = parseResponse(result.text);
    const allowedEpisodeIds = new Set(episodes.map((episode) => episode.id));
    const existingNodes = new Map<string, SemanticNode>();
    const batchNodes = new Map<string, SemanticNode>();
    let insertedNodes = 0;
    let updatedNodes = 0;
    let skippedNodes = 0;
    let insertedEdges = 0;
    let skippedEdges = 0;

    for (const candidate of parsed.nodes) {
      const outcome = await this.upsertNode(candidate, allowedEpisodeIds);

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
      const matches = await this.options.nodeRepository.findByLabelOrAlias(candidate.label, 3);

      for (const match of matches) {
        existingNodes.set(match.label.toLowerCase(), match);

        for (const alias of match.aliases) {
          existingNodes.set(alias.toLowerCase(), match);
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
