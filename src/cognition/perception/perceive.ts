import { SystemClock, type Clock } from "../../util/clock.js";
import type { LLMClient } from "../../llm/index.js";
import { NOOP_TRACER, type TurnTracer } from "../tracing/tracer.js";
import { perceptionResultSchema, type CognitiveMode, type PerceptionResult } from "../types.js";
import {
  createNeutralAffectiveSignal,
  type AffectiveSignal,
} from "../../memory/affective/index.js";
import { detectAffectiveSignal } from "./affective-signal.js";
import { EntityExtractor } from "./entity-extractor.js";
import { ModeDetector } from "./mode-detector.js";
import { detectTemporalCue } from "./temporal-cue.js";

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
  clock?: Clock;
  detectAffectiveSignal?: typeof detectAffectiveSignal;
  onAffectiveError?: (error: unknown) => Promise<void> | void;
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
  private readonly detectAffectiveSignal: typeof detectAffectiveSignal;
  private readonly onAffectiveError?: (error: unknown) => Promise<void> | void;
  private readonly tracer: TurnTracer;
  private readonly turnId?: string;

  constructor(options: PerceiverOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
    this.llmClient = options.llmClient;
    this.model = options.model;
    this.affectiveUseLlmFallback = options.affectiveUseLlmFallback ?? false;
    this.temporalCueUseLlmFallback = options.temporalCueUseLlmFallback ?? true;
    this.detectAffectiveSignal = options.detectAffectiveSignal ?? detectAffectiveSignal;
    this.onAffectiveError = options.onAffectiveError;
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.turnId = options.turnId;
    this.entityExtractor = new EntityExtractor({
      llmClient: options.llmClient,
      model: options.model,
      useLlmFallback: options.useLlmFallback,
    });
    this.modeDetector = new ModeDetector({
      llmClient: options.llmClient,
      model: options.model,
      useLlmFallback: options.useLlmFallback,
      defaultMode: options.modeWhenLlmAbsent,
    });
  }

  private async detectAffectiveSignalSafely(
    text: string,
    recentHistory: readonly string[],
  ): Promise<AffectiveSignal> {
    try {
      return await this.detectAffectiveSignal(text, recentHistory, {
        llmClient: this.llmClient,
        model: this.model,
        useLlmFallback: this.affectiveUseLlmFallback,
      });
    } catch (error) {
      try {
        await this.onAffectiveError?.(error);
      } catch {
        // Best-effort hook logging only.
      }
      return createNeutralAffectiveSignal();
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
    const [entities, mode, affectiveSignal, temporalCue] = await Promise.all([
      this.entityExtractor.extractEntities(text),
      this.modeDetector.detectMode(text, recentHistory),
      this.detectAffectiveSignalSafely(text, recentHistory),
      // Temporal cue extraction is now LLM-backed; degrades to null when
      // no LLM client is configured. Runs in parallel with the rest of
      // perception so it doesn't add serial latency.
      detectTemporalCue(text, nowMs, {
        llmClient: this.temporalCueUseLlmFallback ? this.llmClient : undefined,
        model: this.temporalCueUseLlmFallback ? this.model : undefined,
      }),
    ]);

    const perception = perceptionResultSchema.parse({
      entities,
      mode,
      affectiveSignal,
      temporalCue,
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
