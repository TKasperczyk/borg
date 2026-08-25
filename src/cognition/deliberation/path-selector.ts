// Chooses the S1/S2 deliberation path from perception, stakes, and retrieval signals.
import type {
  RetrievalConfidence,
  RetrievedContradictionRouting,
  RetrievedEpisode,
} from "../../retrieval/index.js";
import type { SessionId } from "../../util/ids.js";
import { DELIBERATION_S2_CONFIDENCE_FLOOR } from "./constants.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import type { CognitiveMode } from "../types.js";
import type { ContradictionRoutingCooldown } from "./contradiction-routing-cooldown.js";
import type {
  ContradictionRoutingTier,
  DeliberationRoutingForcedBy,
  DeliberationRoutingOverride,
  TurnStakes,
} from "./types.js";

export type DeliberationPathDecision = {
  path: "system_1" | "system_2";
  reason: string;
  forced_by?: DeliberationRoutingForcedBy | null;
  contradiction_tier: ContradictionRoutingTier;
  contradiction_fingerprints: string[];
  contradiction_cooldown_demoted: boolean;
};

type NaturalDeliberationPathDecision = Pick<
  DeliberationPathDecision,
  "path" | "reason" | "forced_by"
>;

export type DeliberationPathTrace = {
  tracer: TurnTracer;
  turnId: string;
  sessionId?: SessionId;
};

export type DeliberationPathContradictionRoutingOptions = {
  routing?: RetrievedContradictionRouting | null;
  cooldown?: ContradictionRoutingCooldown;
  audienceKey?: string | null;
  currentTurn?: number;
  cooldownTurns?: number;
  enabled?: boolean;
};

export function chooseDeliberationPath(
  mode: CognitiveMode,
  stakes: TurnStakes,
  _retrievedEpisodes: readonly RetrievedEpisode[],
  contradictionPresent = false,
  retrievalConfidence: RetrievalConfidence,
  trace?: DeliberationPathTrace,
  routingOverride?: DeliberationRoutingOverride | null,
  contradictionRoutingOptions?: DeliberationPathContradictionRoutingOptions,
): DeliberationPathDecision {
  const confidence = retrievalConfidence.overall;
  const contextContradiction =
    contradictionPresent || retrievalConfidence.contradictionPresent === true;
  const contradictionRoutingEnabled = contradictionRoutingOptions?.enabled !== false;
  const contradictionFingerprints = contradictionRoutingEnabled
    ? uniqueSorted(
        contradictionRoutingOptions?.routing?.contradictions.map(
          (contradiction) => contradiction.fingerprint,
        ) ?? [],
      )
    : [];
  const baseContradictionClassification = classifyContradictionRouting({
    contradictionPresent: contextContradiction,
    retrievalConfidenceContradiction: retrievalConfidence.contradictionPresent === true,
    contradictionFingerprints,
    enabled: contradictionRoutingEnabled,
  });
  const effectiveRoutingOverride = contradictionRoutingEnabled ? routingOverride : null;

  const select = (
    path: DeliberationPathDecision["path"],
    reason: string,
    classification = baseContradictionClassification,
    forcedBy: DeliberationRoutingForcedBy | null = null,
    cooldownDemoted = false,
    selectedFingerprints = contradictionFingerprints,
  ): DeliberationPathDecision => {
    if (trace?.tracer.enabled === true) {
      if (contradictionRoutingEnabled) {
        trace.tracer.emit("deliberation.contradiction_routing.completed", {
          turnId: trace.turnId,
          ...(trace.sessionId === undefined ? {} : { session_id: trace.sessionId }),
          fingerprints: selectedFingerprints,
          tier: classification.tier,
          reason: classification.reason,
        });
      }
      trace.tracer.emit("deliberation.path.completed", {
        turnId: trace.turnId,
        ...(trace.sessionId === undefined ? {} : { session_id: trace.sessionId }),
        path,
        reason,
        confidenceOverall: confidence,
        contradictionPresent: contextContradiction,
        forced_by: forcedBy,
        contradiction_tier: classification.tier,
        contradiction_fingerprints: selectedFingerprints,
        contradiction_cooldown_demoted: cooldownDemoted,
      });
    }

    return {
      path,
      reason,
      forced_by: forcedBy,
      contradiction_tier: classification.tier,
      contradiction_fingerprints: selectedFingerprints,
      contradiction_cooldown_demoted: cooldownDemoted,
    };
  };

  const naturalDecision = (): NaturalDeliberationPathDecision => {
    // Reflective always wins -- it's an explicit request for deeper thought.
    if (mode === "reflective") {
      return {
        path: "system_2",
        reason: "Reflective mode always takes the deeper reasoning path.",
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

    if (confidence < DELIBERATION_S2_CONFIDENCE_FLOOR) {
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

  if (effectiveRoutingOverride?.forceSystem2 === true) {
    const baseDecision = naturalDecision();
    const overrideFingerprints = fingerprintsForRoutingOverride(
      effectiveRoutingOverride,
      contradictionRoutingOptions?.routing ?? null,
    );
    const cooldownHits = cooldownHitsForOverride({
      fingerprints: overrideFingerprints,
      contradictionRoutingOptions,
    });
    const cooldownDemoted =
      overrideFingerprints.length > 0 && cooldownHits.length === overrideFingerprints.length;
    const forcedClassification = {
      tier: "s2_forced" as const,
      reason: "Operational contradiction open question forces S2.",
    };
    const demotedClassification = {
      tier: "s2_recommended" as const,
      reason: "Operational contradiction open question force demoted by fingerprint cooldown.",
    };
    const openQuestionLocalHandleMap = Object.fromEntries(
      (effectiveRoutingOverride.openQuestions ?? []).map((question, index) => [
        question.localHandle ?? `contradiction_${index + 1}`,
        question.id,
      ]),
    );

    if (trace?.tracer.enabled === true) {
      trace.tracer.emit("deliberation.path.transitioned", {
        turnId: trace.turnId,
        ...(trace.sessionId === undefined ? {} : { session_id: trace.sessionId }),
        perceptionMode: mode,
        isOperational: effectiveRoutingOverride.isOperational === true,
        audienceEntityId: effectiveRoutingOverride.audienceEntityId ?? null,
        openQuestionIds: [...effectiveRoutingOverride.oqIds],
        openQuestionSources: [
          ...new Set(
            (effectiveRoutingOverride.openQuestions ?? []).map((question) => question.source),
          ),
        ],
        openQuestionLocalHandleMap,
        contradictionFingerprints: overrideFingerprints,
        basePath: baseDecision.path,
        baseReason: baseDecision.reason,
        forcedPath: cooldownDemoted ? baseDecision.path : "system_2",
      });
    }

    if (cooldownDemoted && trace?.tracer.enabled === true) {
      for (const hit of cooldownHits) {
        trace.tracer.emit("deliberation.contradiction_routing.transitioned", {
          turnId: trace.turnId,
          ...(trace.sessionId === undefined ? {} : { session_id: trace.sessionId }),
          fingerprint: hit.fingerprint,
          last_forced_turn: hit.lastForcedTurn,
          current_turn: hit.currentTurn,
        });
      }
    }

    if (cooldownDemoted) {
      return select(
        baseDecision.path,
        baseDecision.reason,
        demotedClassification,
        null,
        true,
        overrideFingerprints,
      );
    }

    if (baseDecision.path === "system_2") {
      return select(
        baseDecision.path,
        baseDecision.reason,
        forcedClassification,
        null,
        false,
        overrideFingerprints,
      );
    }

    recordForcedContradictions({
      fingerprints: overrideFingerprints,
      contradictionRoutingOptions,
    });

    return select(
      "system_2",
      effectiveRoutingOverride.reason,
      forcedClassification,
      effectiveRoutingOverride.forcedBy,
      false,
      overrideFingerprints,
    );
  }

  const baseDecision = naturalDecision();
  return select(baseDecision.path, baseDecision.reason);
}

function classifyContradictionRouting(input: {
  contradictionPresent: boolean;
  retrievalConfidenceContradiction: boolean;
  contradictionFingerprints: readonly string[];
  enabled: boolean;
}): { tier: ContradictionRoutingTier; reason: string } {
  if (
    !input.enabled ||
    (!input.contradictionPresent && input.contradictionFingerprints.length === 0)
  ) {
    return {
      tier: "none",
      reason: "No retrieved contradiction signal.",
    };
  }

  if (input.retrievalConfidenceContradiction) {
    return {
      tier: "confidence_penalty",
      reason: "Retrieved contradiction is applied as a confidence penalty and prompt annotation.",
    };
  }

  return {
    tier: "annotation_only",
    reason: "Retrieved contradiction is surfaced as a prompt annotation only.",
  };
}

function fingerprintsForRoutingOverride(
  routingOverride: DeliberationRoutingOverride,
  routing: RetrievedContradictionRouting | null,
): string[] {
  if (
    routingOverride.contradictionFingerprints !== undefined &&
    routingOverride.contradictionFingerprints.length > 0
  ) {
    return uniqueSorted(routingOverride.contradictionFingerprints);
  }

  if (
    routing === null ||
    routing.contradictions.length === 0 ||
    routingOverride.oqIds.length === 0
  ) {
    return [];
  }

  const oqIds = new Set(routingOverride.oqIds);

  return uniqueSorted(
    routing.contradictions
      .filter((contradiction) =>
        contradiction.linkedOpenQuestionIds.some((openQuestionId) => oqIds.has(openQuestionId)),
      )
      .map((contradiction) => contradiction.fingerprint),
  );
}

function cooldownHitsForOverride(input: {
  fingerprints: readonly string[];
  contradictionRoutingOptions?: DeliberationPathContradictionRoutingOptions;
}) {
  const options = input.contradictionRoutingOptions;

  if (
    options?.enabled === false ||
    options?.cooldown === undefined ||
    options.audienceKey === null ||
    options.audienceKey === undefined ||
    options.currentTurn === undefined ||
    options.cooldownTurns === undefined ||
    input.fingerprints.length === 0
  ) {
    return [];
  }

  return options.cooldown.getCoolingFingerprints({
    audience: options.audienceKey,
    fingerprints: input.fingerprints,
    currentTurn: options.currentTurn,
    cooldownTurns: options.cooldownTurns,
  });
}

function recordForcedContradictions(input: {
  fingerprints: readonly string[];
  contradictionRoutingOptions?: DeliberationPathContradictionRoutingOptions;
}): void {
  const options = input.contradictionRoutingOptions;

  if (
    options?.enabled === false ||
    options?.cooldown === undefined ||
    options.audienceKey === null ||
    options.audienceKey === undefined ||
    options.currentTurn === undefined ||
    input.fingerprints.length === 0
  ) {
    return;
  }

  options.cooldown.recordForced({
    audience: options.audienceKey,
    fingerprints: input.fingerprints,
    currentTurn: options.currentTurn,
  });
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}
