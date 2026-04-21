import type { Episode } from "../../memory/episodic/types.js";
import { tokenizeText } from "../../util/text/tokenize.js";

export function computeGoalRelevance(
  goalDescriptions: readonly string[],
  episode: Pick<Episode, "title" | "narrative" | "tags">,
): number {
  if (goalDescriptions.length === 0) {
    return 0;
  }

  const episodeTokens = tokenizeText(
    `${episode.title} ${episode.narrative} ${episode.tags.join(" ")}`,
  );
  let bestOverlap = 0;

  for (const description of goalDescriptions) {
    const goalTokens = tokenizeText(description);

    if (goalTokens.size === 0) {
      continue;
    }

    let overlap = 0;

    for (const token of goalTokens) {
      if (episodeTokens.has(token)) {
        overlap += 1;
      }
    }

    bestOverlap = Math.max(bestOverlap, overlap / goalTokens.size);
  }

  return bestOverlap;
}
