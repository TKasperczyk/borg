import { episodeIdSchema, type Episode } from "../memory/episodic/index.js";

import type { OfflineContext } from "./types.js";

function representedEpisodeIds(ctx: OfflineContext): Set<Episode["id"]> {
  return new Set([
    ...ctx.semanticNodeRepository.listAllSourceEpisodeIds(),
    ...ctx.semanticEdgeRepository.listAllEvidenceEpisodeIds(),
  ]);
}

function auditedExtractionEpisodeIds(ctx: OfflineContext): Set<Episode["id"]> {
  const audited = new Set<Episode["id"]>();

  for (const targets of ctx.auditLog.listTargets({
    process: "semantic-extractor",
    reverted: false,
  })) {
    const episodeIds = targets.episode_ids;

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
  return new Set([...representedEpisodeIds(ctx), ...auditedExtractionEpisodeIds(ctx)]);
}

export async function isEpisodeExtracted(
  ctx: OfflineContext,
  episodeId: Episode["id"],
): Promise<boolean> {
  return (await extractedEpisodeIds(ctx)).has(episodeId);
}
