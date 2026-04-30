import { SystemClock, type Clock } from "../../util/clock.js";
import type { LLMClient } from "../../llm/index.js";
import type { AffectiveExtractorDegradedReason } from "../../memory/affective/index.js";
import { NOOP_TRACER, type TurnTracer } from "../tracing/tracer.js";
import { perceptionResultSchema, type CognitiveMode, type PerceptionResult } from "../types.js";
import {
  createNeutralAffectiveSignal,
  type AffectiveSignal,
} from "../../memory/affective/index.js";
import { detectAffectiveSignal } from "./affective-signal.js";
import { EntityExtractor } from "./entity-extractor.js";
import { detectFactualChallenge } from "./factual-challenge.js";
import { ModeDetector } from "./mode-detector.js";
import { detectTemporalCue } from "./temporal-cue.js";

export type PerceptionClassifierName =
  | "entity_extractor"
  | "mode_detector"
  | "affective_signal"
  | "temporal_cue"
  | "factual_challenge";

export type PerceptionClassifierFailure = {
  classifier: PerceptionClassifierName;
  error: unknown;
};

export type PerceptionClassifierFailureObserver = (
  failure: PerceptionClassifierFailure,
) => Promise<void> | void;

export async function runPerceptionClassifierSafely<T>(input: {
  classifier: PerceptionClassifierName;
  run: () => Promise<T> | T;
  fallback: () => Promise<T> | T;
  onFailure?: PerceptionClassifierFailureObserver;
}): Promise<T> {
  try {
    return await input.run();
  } catch (error) {
    try {
      await input.onFailure?.({
        classifier: input.classifier,
        error,
      });
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return input.fallback();
  }
}

export type PerceiverOptions = {
  llmClient?: LLMClient;
  model?: string;
  useLlmFallback?: boolean;
  affectiveUseLlmFallback?: boolean;
  /**
   * If true (default) and an LLM client + model are configured, temporal
   * cue extraction runs an LLM classifier every turn. Previously this was
   * a hardcoded 6-phrase regex list that missed most real temporal
   * phrasings ("last Tuesday", "earlier today", "a few days ago"). With
   * fakes or no LLM, degrades to "no temporal filter".
   */
  temporalCueUseLlmFallback?: boolean;
  /**
   * If true (default) and an LLM client + model are configured, detect
   * user challenges to remembered/stored facts as a separate perception
   * signal. This is intentionally not a cognitive mode; it opens a
   * verification retrieval lane without changing coarse routing.
   */
  factualChallengeUseLlmFallback?: boolean;
  clock?: Clock;
  detectAffectiveSignal?: typeof detectAffectiveSignal;
  onAffectiveError?: (error: unknown) => Promise<void> | void;
  onClassifierFailure?: PerceptionClassifierFailureObserver;
  tracer?: TurnTracer;
  turnId?: string;
  /**
   * Mode returned when LLM-based mode detection isn't firing (either the
   * fallback is disabled or no LLM client is configured). Test harnesses
   * that run with fake LLMs typically use this to pick the mode they want
   * to exercise without scripting a tool-call response.
   */
  modeWhenLlmAbsent?: CognitiveMode;
};

export class Perceiver {
  private readonly clock: Clock;
  private readonly entityExtractor: EntityExtractor;
  private readonly modeDetector: ModeDetector;
  private readonly llmClient?: LLMClient;
  private readonly model?: string;
  private readonly affectiveUseLlmFallback: boolean;
  private readonly temporalCueUseLlmFallback: boolean;
  private readonly factualChallengeUseLlmFallback: boolean;
  private readonly detectAffectiveSignal: typeof detectAffectiveSignal;
  private readonly onAffectiveError?: (error: unknown) => Promise<void> | void;
  private readonly onClassifierFailure?: PerceptionClassifierFailureObserver;
  private readonly tracer: TurnTracer;
  private readonly turnId?: string;
  private readonly modeWhenLlmAbsent: CognitiveMode;

  constructor(options: PerceiverOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
    this.llmClient = options.llmClient;
    this.model = options.model;
    this.affectiveUseLlmFallback = options.affectiveUseLlmFallback ?? true;
    this.temporalCueUseLlmFallback = options.temporalCueUseLlmFallback ?? true;
    this.factualChallengeUseLlmFallback = options.factualChallengeUseLlmFallback ?? true;
    this.detectAffectiveSignal = options.detectAffectiveSignal ?? detectAffectiveSignal;
    this.onAffectiveError = options.onAffectiveError;
    this.onClassifierFailure = options.onClassifierFailure;
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.turnId = options.turnId;
    this.modeWhenLlmAbsent = options.modeWhenLlmAbsent ?? "idle";
    this.entityExtractor = new EntityExtractor({
      llmClient: options.llmClient,
      model: options.model,
    });
    this.modeDetector = new ModeDetector({
      llmClient: options.llmClient,
      model: options.model,
      useLlmFallback: options.useLlmFallback,
      defaultMode: this.modeWhenLlmAbsent,
    });
  }

  private async detectAffectiveSignalSafely(
    text: string,
    recentHistory: readonly string[],
  ): Promise<{ signal: AffectiveSignal; degraded: boolean }> {
    let degraded = false;
    const markDegraded = async (
      reason: AffectiveExtractorDegradedReason | "classifier_error",
      error?: unknown,
    ): Promise<void> => {
      degraded = true;

      if (this.tracer.enabled && this.turnId !== undefined) {
        this.tracer.emit("perception_classifier_degraded", {
          turnId: this.turnId,
          classifier: "affective_signal",
          reason,
        });
      }

      if (error !== undefined) {
        try {
          await this.onAffectiveError?.(error);
        } catch {
          // Best-effort hook logging only.
        }
      }
    };

    try {
      const signal = await this.detectAffectiveSignal(text, recentHistory, {
        llmClient: this.llmClient,
        model: this.model,
        useLlmFallback: this.affectiveUseLlmFallback,
        onDegraded: markDegraded,
      });

      return { signal, degraded };
    } catch (error) {
      await markDegraded("classifier_error", error);

      return { signal: createNeutralAffectiveSignal(), degraded: true };
    }
  }

  async perceive(text: string, recentHistory: readonly string[] = []): Promise<PerceptionResult> {
    if (this.tracer.enabled && this.turnId !== undefined) {
      this.tracer.emit("perception_started", {
        turnId: this.turnId,
        inputCharCount: text.length,
        recentHistoryCount: recentHistory.length,
      });
    }

    const nowMs = this.clock.now();
    const [entities, mode, affective, temporalCue, factualChallenge] = await Promise.all([
      runPerceptionClassifierSafely({
        classifier: "entity_extractor",
        run: () => this.entityExtractor.extractEntities(text),
        // Fallback returns empty entities. Previous regex-heuristic
        // fallback produced false-positive entities at high rates
        // ('Good', 'If', '[End.]') that poisoned downstream
        // retrieval. Empty is the honest signal when the LLM call
        // fails -- the turn proceeds with no entities for this turn.
        fallback: () => [],
        onFailure: this.onClassifierFailure,
      }),
      runPerceptionClassifierSafely({
        classifier: "mode_detector",
        run: () => this.modeDetector.detectMode(text, recentHistory),
        fallback: () => this.modeWhenLlmAbsent,
        onFailure: this.onClassifierFailure,
      }),
      this.detectAffectiveSignalSafely(text, recentHistory),
      // Temporal cue extraction is now LLM-backed; degrades to null when
      // no LLM client is configured. Runs in parallel with the rest of
      // perception so it doesn't add serial latency.
      detectTemporalCue(text, nowMs, {
        llmClient: this.temporalCueUseLlmFallback ? this.llmClient : undefined,
        model: this.temporalCueUseLlmFallback ? this.model : undefined,
        onDegraded: async (reason, error) => {
          if (this.tracer.enabled && this.turnId !== undefined) {
            this.tracer.emit("perception_classifier_degraded", {
              turnId: this.turnId,
              classifier: "temporal_cue",
              reason,
            });
          }

          if (error !== undefined) {
            try {
              await this.onClassifierFailure?.({
                classifier: "temporal_cue",
                error,
              });
            } catch {
              // Best-effort hook logging only.
            }
          }
        },
      }),
      detectFactualChallenge(text, recentHistory, {
        llmClient: this.factualChallengeUseLlmFallback ? this.llmClient : undefined,
        model: this.factualChallengeUseLlmFallback ? this.model : undefined,
        onDegraded: async (reason, error) => {
          if (this.tracer.enabled && this.turnId !== undefined) {
            this.tracer.emit("perception_classifier_degraded", {
              turnId: this.turnId,
              classifier: "factual_challenge",
              reason,
            });
          }

          if (error !== undefined) {
            try {
              await this.onClassifierFailure?.({
                classifier: "factual_challenge",
                error,
              });
            } catch {
              // Best-effort hook logging only.
            }
          }
        },
      }),
    ]);

    const perception = perceptionResultSchema.parse({
      entities,
      mode,
      affectiveSignal: affective.signal,
      affectiveSignalDegraded: affective.degraded,
      temporalCue,
      factualChallenge,
    });

    if (this.tracer.enabled && this.turnId !== undefined) {
      this.tracer.emit("perception_completed", {
        turnId: this.turnId,
        mode: perception.mode,
        entities: perception.entities,
        temporalCue: perception.temporalCue,
      });
    }

    return perception;
  }
}

export async function perceive(
  text: string,
  options: PerceiverOptions = {},
  recentHistory: readonly string[] = [],
): Promise<PerceptionResult> {
  return new Perceiver(options).perceive(text, recentHistory);
}
