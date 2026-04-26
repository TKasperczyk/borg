import { z } from "zod";

import type { RetrievedEpisode } from "../../retrieval/index.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const episodicSearchInputSchema = z.object({
  query: z.string().min(1),
  limit: z.number().int().positive().max(10).optional(),
});

const episodicSearchOutputSchema = z.object({
  episodes: z.array(
    z.object({
      id: z.string().min(1),
      title: z.string().min(1),
      score: z.number().finite(),
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
      const results = await options.searchEpisodes(input.query, input.limit ?? 5, context);

      return {
        episodes: results.map((result) => ({
          id: result.episode.id,
          title: result.episode.title,
          score: result.score,
        })),
      };
    },
  };
}
