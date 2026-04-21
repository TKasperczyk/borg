import { SystemClock, type Clock } from "../../util/clock.js";
import type { LLMClient } from "../../llm/index.js";
import { perceptionResultSchema, type PerceptionResult } from "../types.js";
import { detectAffectiveSignal } from "./affective-signal.js";
import { EntityExtractor } from "./entity-extractor.js";
import { ModeDetector } from "./mode-detector.js";
import { detectTemporalCue } from "./temporal-cue.js";

export type PerceiverOptions = {
  llmClient?: LLMClient;
  model?: string;
  useLlmFallback?: boolean;
  clock?: Clock;
};

export class Perceiver {
  private readonly clock: Clock;
  private readonly entityExtractor: EntityExtractor;
  private readonly modeDetector: ModeDetector;

  constructor(options: PerceiverOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
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

  async perceive(text: string, recentHistory: readonly string[] = []): Promise<PerceptionResult> {
    const [entities, mode] = await Promise.all([
      this.entityExtractor.extractEntities(text),
      this.modeDetector.detectMode(text, recentHistory),
    ]);

    return perceptionResultSchema.parse({
      entities,
      mode,
      affectiveSignal: detectAffectiveSignal(),
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
