import { episodeIdSchema, type Episode } from "../memory/episodic/index.js";

import type { OfflineContext } from "./types.js";

async function representedEpisodeIds(ctx: OfflineContext): Promise<Set<Episode["id"]>> {
  const represented = new Set<Episode["id"]>();
  const nodes = await ctx.semanticNodeRepository.list({
    includeArchived: true,
    limit: 100_000,
  });

  for (const node of nodes) {
    for (const episodeId of node.source_episode_ids) {
      represented.add(episodeId);
    }
  }

  for (const edge of ctx.semanticEdgeRepository.listEdges({ includeInvalid: true })) {
    for (const episodeId of edge.evidence_episode_ids) {
      represented.add(episodeId);
    }
  }

  return represented;
}

function auditedExtractionEpisodeIds(ctx: OfflineContext): Set<Episode["id"]> {
  const audited = new Set<Episode["id"]>();

  for (const audit of ctx.auditLog.list({ process: "semantic-extractor", reverted: false })) {
    const episodeIds = audit.targets.episode_ids;

    if (!Array.isArray(episodeIds)) {
      continue;
    }

    for (const episodeId of episodeIds) {
      const parsed = episodeIdSchema.safeParse(episodeId);

      if (parsed.success) {
        audited.add(parsed.data);
      }
    }
  }

  return audited;
}

export async function extractedEpisodeIds(ctx: OfflineContext): Promise<Set<Episode["id"]>> {
  return new Set([...(await representedEpisodeIds(ctx)), ...auditedExtractionEpisodeIds(ctx)]);
}

export async function isEpisodeExtracted(
  ctx: OfflineContext,
  episodeId: Episode["id"],
): Promise<boolean> {
  return (await extractedEpisodeIds(ctx)).has(episodeId);
}
