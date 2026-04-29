import type { LLMClient } from "../../llm/index.js";
import { AffectiveExtractor, type AffectiveSignal } from "../../memory/affective/index.js";
import type { AffectiveExtractorDegradedReason } from "../../memory/affective/index.js";

export type DetectAffectiveSignalOptions = {
  llmClient?: LLMClient;
  model?: string;
  useLlmFallback?: boolean;
  onDegraded?: (reason: AffectiveExtractorDegradedReason, error?: unknown) => Promise<void> | void;
};

export async function detectAffectiveSignal(
  text: string,
  recentHistory: readonly string[] = [],
  options: DetectAffectiveSignalOptions = {},
): Promise<AffectiveSignal> {
  const extractor = new AffectiveExtractor({
    llmClient: options.llmClient,
    model: options.model,
    useLlmFallback: options.useLlmFallback,
    onDegraded: options.onDegraded,
  });

  return extractor.analyze(text, recentHistory);
}
