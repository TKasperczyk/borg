import { SystemClock, type Clock } from "../../util/clock.js";
import type { LLMClient } from "../../llm/index.js";
import { perceptionResultSchema, type PerceptionResult } from "../types.js";
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
  clock?: Clock;
  detectAffectiveSignal?: typeof detectAffectiveSignal;
  onAffectiveError?: (error: unknown) => Promise<void> | void;
};

export class Perceiver {
  private readonly clock: Clock;
  private readonly entityExtractor: EntityExtractor;
  private readonly modeDetector: ModeDetector;
  private readonly llmClient?: LLMClient;
  private readonly model?: string;
  private readonly affectiveUseLlmFallback: boolean;
  private readonly detectAffectiveSignal: typeof detectAffectiveSignal;
  private readonly onAffectiveError?: (error: unknown) => Promise<void> | void;

  constructor(options: PerceiverOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
    this.llmClient = options.llmClient;
    this.model = options.model;
    this.affectiveUseLlmFallback = options.affectiveUseLlmFallback ?? false;
    this.detectAffectiveSignal = options.detectAffectiveSignal ?? detectAffectiveSignal;
    this.onAffectiveError = options.onAffectiveError;
    this.entityExtractor = new EntityExtractor({
      llmClient: options.llmClient,
      model: options.model,
      useLlmFallback: options.useLlmFallback,
    });
    this.modeDetector = new ModeDetector({
      llmClient: options.llmClient,
      model: options.model,
      useLlmFallback: options.useLlmFallback,
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
    const [entities, mode, affectiveSignal] = await Promise.all([
      this.entityExtractor.extractEntities(text),
      this.modeDetector.detectMode(text, recentHistory),
      this.detectAffectiveSignalSafely(text, recentHistory),
    ]);

    return perceptionResultSchema.parse({
      entities,
      mode,
      affectiveSignal,
      temporalCue: detectTemporalCue(text, this.clock.now()),
    });
  }
}

export async function perceive(
  text: string,
  options: PerceiverOptions = {},
  recentHistory: readonly string[] = [],
): Promise<PerceptionResult> {
  return new Perceiver(options).perceive(text, recentHistory);
}
