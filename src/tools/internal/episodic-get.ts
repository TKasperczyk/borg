import { z } from "zod";
import {
  episodeIdSchema,
  episodeLineageSchema,
  type EpisodicRepository,
} from "../../memory/episodic/index.js";
import {
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelMetadataSchema,
} from "../../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import type { ToolDefinition } from "../dispatcher.js";

const inputSchema = z.object({ episode_id: episodeIdSchema }).strict();
const outputSchema = z.object({
  episode: z
    .object({
      id: episodeIdSchema,
      title: z.string(),
      narrative: z.string(),
      participants: z.array(z.string()),
      tags: z.array(z.string()),
      start_time: z.number(),
      end_time: z.number(),
      created_at: z.number(),
      updated_at: z.number(),
      source_stream_ids: z.array(z.string()),
      confidence: z.number(),
      lineage: episodeLineageSchema,
      episode_kind: z.string(),
      consolidation_family_id: z.string().nullable(),
      archived: z.boolean(),
      lifecycle_status: z.enum(["active", "archived", "inactive"]),
      tier: z.string().nullable(),
      disclosure: z.string(),
      disclosure_label: memoryDisclosureLabelMetadataSchema,
    })
    .nullable(),
});

export function createEpisodicGetTool(options: {
  episodicRepository: Pick<EpisodicRepository, "get" | "getStats" | "isEpisodeEffectivelyVisible">;
}): ToolDefinition<z.infer<typeof inputSchema>, z.infer<typeof outputSchema>> {
  return {
    name: "tool.episodic.get",
    description:
      "Read the full text of one episodic record by id; use when an excerpt was cut. Includes provenance, lifecycle, and disclosure labels.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema,
    outputSchema,
    async invoke(input) {
      const episode = await options.episodicRepository.get(input.episode_id, {
        includeArchived: true,
      });
      if (episode === null) return { episode: null };
      const stats = options.episodicRepository.getStats(episode.id);
      return {
        episode: {
          id: episode.id,
          title: episode.title,
          narrative: episode.narrative,
          participants: episode.participants,
          tags: episode.tags,
          start_time: episode.start_time,
          end_time: episode.end_time,
          created_at: episode.created_at,
          updated_at: episode.updated_at,
          source_stream_ids: episode.source_stream_ids,
          confidence: episode.confidence,
          lineage: episode.lineage,
          episode_kind: episode.episode_kind ?? "raw",
          consolidation_family_id: episode.consolidation_family_id ?? null,
          archived: stats?.archived ?? false,
          lifecycle_status:
            stats?.archived === true
              ? "archived"
              : options.episodicRepository.isEpisodeEffectivelyVisible(episode.id)
                ? "active"
                : "inactive",
          tier: stats?.tier ?? null,
          ...memoryDisclosurePayloadFields(memoryDisclosureLabelFromEpisodeAccess(episode)),
        },
      };
    },
  };
}
