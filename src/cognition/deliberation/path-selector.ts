// Chooses the S1/S2 deliberation path from perception, stakes, and retrieval signals.
import type { RetrievalConfidence, RetrievedEpisode } from "../../retrieval/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { tokenizeText } from "../../util/text/tokenize.js";
import type { CognitiveMode } from "../types.js";
import type { TurnStakes } from "./types.js";

export type DeliberationPathDecision = {
  path: "system_1" | "system_2";
  reason: string;
};

export type DeliberationPathTrace = {
  tracer: TurnTracer;
  turnId: string;
};

// When no RetrievalConfidence is supplied, fall back to the old score-average.
// This keeps test harnesses and other callers that don't pass confidence working;
// production paths plumb RetrievalConfidence through pipeline -> deliberator.
function fallbackConfidence(results: readonly RetrievedEpisode[]): number {
  if (results.length === 0) {
    return 0;
  }

  const total = results.reduce((sum, result) => sum + result.score, 0);
  return total / results.length;
}

function hasContradictionSignal(retrievedEpisodes: readonly RetrievedEpisode[]): boolean {
  // Contradiction in retrieved context: a "warning"-tagged episode and a
  // "recommended"-tagged episode sharing topic tokens. This is matching on
  // extractor-generated structured tags (schema-meaningful), NOT on the raw
  // user message. The previous regex pattern on the user message ("but",
  // "however", "actually", ...) was a same-class overfit to what mode
  // detection was doing and has been removed; the semantic-graph
  // contradictionPresent flag from retrieval carries the genuine case.
  const warnings = retrievedEpisodes.filter((result) => result.episode.tags.includes("warning"));
  const recommendations = retrievedEpisodes.filter((result) =>
    result.episode.tags.includes("recommended"),
  );

  for (const warning of warnings) {
    const warningTokens = tokenizeText(
      `${warning.episode.title} ${warning.episode.tags.join(" ")}`,
    );

    for (const recommendation of recommendations) {
      const recommendationTokens = tokenizeText(
        `${recommendation.episode.title} ${recommendation.episode.tags.join(" ")}`,
      );
      const overlap = [...warningTokens].some((token) => recommendationTokens.has(token));

      if (overlap) {
        return true;
      }
    }
  }

  return false;
}

export function chooseDeliberationPath(
  mode: CognitiveMode,
  stakes: TurnStakes,
  retrievedEpisodes: readonly RetrievedEpisode[],
  contradictionPresent = false,
  retrievalConfidence?: RetrievalConfidence | null,
  trace?: DeliberationPathTrace,
): DeliberationPathDecision {
  const confidence =
    retrievalConfidence !== undefined && retrievalConfidence !== null
      ? retrievalConfidence.overall
      : fallbackConfidence(retrievedEpisodes);
  const contextContradiction =
    contradictionPresent || retrievalConfidence?.contradictionPresent === true;

  const select = (
    path: DeliberationPathDecision["path"],
    reason: string,
    effectiveContradiction = contextContradiction,
  ): DeliberationPathDecision => {
    if (trace?.tracer.enabled === true) {
      trace.tracer.emit("path_selected", {
        turnId: trace.turnId,
        path,
        reason,
        confidenceOverall: confidence,
        contradictionPresent: effectiveContradiction,
      });
    }

    return {
      path,
      reason,
    };
  };

  if (mode === "idle") {
    return select("system_1", "Idle mode keeps the response on the cheap path.");
  }

  if (mode === "reflective") {
    return select("system_2", "Reflective mode always takes the deeper reasoning path.");
  }

  const episodeContradiction = hasContradictionSignal(retrievedEpisodes);

  if (contextContradiction || episodeContradiction) {
    return select(
      "system_2",
      "Retrieved-context contradiction triggered deeper reasoning.",
      true,
    );
  }

  if (stakes === "high") {
    return select("system_2", "High-stakes request requires explicit planning.");
  }

  if (confidence < 0.45) {
    return select("system_2", "Low retrieval confidence triggered deeper reasoning.");
  }

  return select("system_1", "Retrieval confidence is strong enough for a direct response.");
}
