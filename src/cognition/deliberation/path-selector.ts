// Chooses the S1/S2 deliberation path from perception, stakes, and retrieval signals.
import type { RetrievalConfidence, RetrievedEpisode } from "../../retrieval/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
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

export function chooseDeliberationPath(
  mode: CognitiveMode,
  stakes: TurnStakes,
  _retrievedEpisodes: readonly RetrievedEpisode[],
  contradictionPresent = false,
  retrievalConfidence: RetrievalConfidence,
  trace?: DeliberationPathTrace,
): DeliberationPathDecision {
  const confidence = retrievalConfidence.overall;
  const contextContradiction =
    contradictionPresent || retrievalConfidence.contradictionPresent === true;

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
    return select("system_1", "Idle mode keeps the response on the direct path.");
  }

  if (mode === "reflective") {
    return select("system_2", "Reflective mode always takes the deeper reasoning path.");
  }

  if (contextContradiction) {
    return select("system_2", "Retrieved-context contradiction triggered deeper reasoning.");
  }

  if (stakes === "high") {
    return select("system_2", "High-stakes request requires explicit planning.");
  }

  if (confidence < 0.45) {
    return select("system_2", "Low retrieval confidence triggered deeper reasoning.");
  }

  return select("system_1", "Retrieval confidence is strong enough for a direct response.");
}
