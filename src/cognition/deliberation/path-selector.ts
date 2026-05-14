// Chooses the S1/S2 deliberation path from perception, stakes, and retrieval signals.
import type { RetrievalConfidence, RetrievedEpisode } from "../../retrieval/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { CognitiveMode } from "../types.js";
import type {
  DeliberationRoutingForcedBy,
  DeliberationRoutingOverride,
  TurnStakes,
} from "./types.js";

export type DeliberationPathDecision = {
  path: "system_1" | "system_2";
  reason: string;
  forced_by?: DeliberationRoutingForcedBy | null;
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
  routingOverride?: DeliberationRoutingOverride | null,
): DeliberationPathDecision {
  const confidence = retrievalConfidence.overall;
  const contextContradiction =
    contradictionPresent || retrievalConfidence.contradictionPresent === true;

  const select = (
    path: DeliberationPathDecision["path"],
    reason: string,
    effectiveContradiction = contextContradiction,
    forcedBy: DeliberationRoutingForcedBy | null = null,
  ): DeliberationPathDecision => {
    if (trace?.tracer.enabled === true) {
      trace.tracer.emit("path_selected", {
        turnId: trace.turnId,
        path,
        reason,
        confidenceOverall: confidence,
        contradictionPresent: effectiveContradiction,
        forced_by: forcedBy,
      });
    }

    return {
      path,
      reason,
      forced_by: forcedBy,
    };
  };

  const naturalDecision = (): DeliberationPathDecision => {
    // Reflective always wins -- it's an explicit request for deeper thought.
    if (mode === "reflective") {
      return {
        path: "system_2",
        reason: "Reflective mode always takes the deeper reasoning path.",
        forced_by: null,
      };
    }

    // High-stakes and contradiction must escalate even in idle mode -- a
    // misclassified high-stakes idle turn can't be allowed to skip S2.
    if (contextContradiction) {
      return {
        path: "system_2",
        reason: "Retrieved-context contradiction triggered deeper reasoning.",
        forced_by: null,
      };
    }

    if (stakes === "high") {
      return {
        path: "system_2",
        reason: "High-stakes request requires explicit planning.",
        forced_by: null,
      };
    }

    if (mode === "idle") {
      return {
        path: "system_1",
        reason: "Idle mode keeps the response on the direct path.",
        forced_by: null,
      };
    }

    if (confidence < 0.45) {
      return {
        path: "system_2",
        reason: "Low retrieval confidence triggered deeper reasoning.",
        forced_by: null,
      };
    }

    return {
      path: "system_1",
      reason: "Retrieval confidence is strong enough for a direct response.",
      forced_by: null,
    };
  };

  if (routingOverride?.forceSystem2 === true) {
    const baseDecision = naturalDecision();
    const openQuestionLocalHandleMap = Object.fromEntries(
      (routingOverride.openQuestions ?? []).map((question, index) => [
        question.localHandle ?? `contradiction_${index + 1}`,
        question.id,
      ]),
    );

    if (trace?.tracer.enabled === true) {
      trace.tracer.emit("s2_routing_forced_by_contradiction", {
        turnId: trace.turnId,
        perceptionMode: mode,
        isOperational: routingOverride.isOperational === true,
        audienceEntityId: routingOverride.audienceEntityId ?? null,
        openQuestionIds: [...routingOverride.oqIds],
        openQuestionSources: [
          ...new Set((routingOverride.openQuestions ?? []).map((question) => question.source)),
        ],
        openQuestionLocalHandleMap,
        basePath: baseDecision.path,
        baseReason: baseDecision.reason,
        forcedPath: "system_2",
      });
    }

    if (baseDecision.path === "system_2") {
      return select(baseDecision.path, baseDecision.reason, contextContradiction);
    }

    return select(
      "system_2",
      routingOverride.reason,
      contextContradiction,
      routingOverride.forcedBy,
    );
  }

  const baseDecision = naturalDecision();
  return select(baseDecision.path, baseDecision.reason);
}
