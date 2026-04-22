import type { ValueRecord } from "../../memory/self/index.js";
import type { Episode } from "../../memory/episodic/types.js";
import { tokenizeText } from "../../util/text/tokenize.js";

export function computeValueAlignment(
  activeValues: readonly Pick<ValueRecord, "label" | "description">[],
  episode: Pick<Episode, "title" | "narrative" | "tags">,
): number {
  if (activeValues.length === 0) {
    return 0;
  }

  const episodeTokens = tokenizeText(
    `${episode.title} ${episode.narrative} ${episode.tags.join(" ")}`,
  );
  let bestOverlap = 0;

  for (const value of activeValues) {
    const valueTokens = tokenizeText(`${value.label} ${value.description}`);

    if (valueTokens.size === 0) {
      continue;
    }

    let overlap = 0;

    for (const token of valueTokens) {
      if (episodeTokens.has(token)) {
        overlap += 1;
      }
    }

    bestOverlap = Math.max(bestOverlap, overlap / valueTokens.size);
  }

  return bestOverlap;
}
