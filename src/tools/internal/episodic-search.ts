import { z } from "zod";

import type { RetrievedEpisode } from "../../retrieval/index.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const DEFAULT_EPISODIC_SEARCH_LIMIT = 5;
const MAX_EPISODIC_SEARCH_LIMIT = 5;
const MAX_NARRATIVE_CHARS = 400;
const MAX_CITATION_CONTENT_CHARS = 180;
const MAX_CITATIONS_PER_EPISODE = 3;

const episodicSearchInputSchema = z.object({
  query: z.string().min(1),
  limit: z.number().int().positive().max(MAX_EPISODIC_SEARCH_LIMIT).optional(),
});

const episodicSearchOutputSchema = z.object({
  episodes: z.array(
    z.object({
      id: z.string().min(1),
      title: z.string().min(1),
      narrative: z.string(),
      participants: z.array(z.string()),
      tags: z.array(z.string()),
      start_time: z.number().finite(),
      end_time: z.number().finite(),
      source_stream_ids: z.array(z.string().min(1)),
      score: z.number().finite(),
      score_breakdown: z.object({
        similarity: z.number().finite(),
        decayed_salience: z.number().finite(),
        time_relevance: z.number().finite(),
      }),
      citation_chain: z.array(
        z.object({
          id: z.string().min(1),
          kind: z.string().min(1),
          timestamp: z.number().finite(),
          content: z.string(),
        }),
      ),
    }),
  ),
});

export type EpisodicSearchToolOptions = {
  searchEpisodes: (
    query: string,
    limit: number | undefined,
    context: ToolInvocationContext,
  ) => Promise<RetrievedEpisode[]>;
};

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function truncateText(text: string, maxChars: number): string {
  const normalized = normalizeWhitespace(text);

  if (normalized.length <= maxChars) {
    return normalized;
  }

  return `${normalized.slice(0, maxChars - 3).trimEnd()}...`;
}

function summarizeCitationContent(content: unknown): string {
  const serialized = JSON.stringify(content ?? null);
  const text = typeof content === "string" ? content : serialized ?? String(content);

  return truncateText(text, MAX_CITATION_CONTENT_CHARS);
}

export function createEpisodicSearchTool(
  options: EpisodicSearchToolOptions,
): ToolDefinition<
  z.infer<typeof episodicSearchInputSchema>,
  z.infer<typeof episodicSearchOutputSchema>
> {
  return {
    name: "tool.episodic.search",
    description: "Search episodic memory for relevant episodes.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: episodicSearchInputSchema,
    outputSchema: episodicSearchOutputSchema,
    async invoke(input, context) {
      const results = await options.searchEpisodes(
        input.query,
        Math.min(input.limit ?? DEFAULT_EPISODIC_SEARCH_LIMIT, MAX_EPISODIC_SEARCH_LIMIT),
        context,
      );

      return {
        episodes: results.map((result) => ({
          id: result.episode.id,
          title: result.episode.title,
          narrative: truncateText(result.episode.narrative, MAX_NARRATIVE_CHARS),
          participants: result.episode.participants,
          tags: result.episode.tags,
          start_time: result.episode.start_time,
          end_time: result.episode.end_time,
          source_stream_ids: result.episode.source_stream_ids,
          score: result.score,
          score_breakdown: {
            similarity: result.scoreBreakdown.similarity,
            decayed_salience: result.scoreBreakdown.decayedSalience,
            time_relevance: result.scoreBreakdown.timeRelevance,
          },
          citation_chain: result.citationChain.slice(0, MAX_CITATIONS_PER_EPISODE).map((entry) => ({
            id: entry.id,
            kind: entry.kind,
            timestamp: entry.timestamp,
            content: summarizeCitationContent(entry.content),
          })),
        })),
      };
    },
  };
}
