import type { LLMClient } from "../../llm/index.js";
import {
  AffectiveExtractor,
  analyzeAffectiveSignalHeuristically,
  type AffectiveSignal,
} from "../../memory/affective/index.js";

export type DetectAffectiveSignalOptions = {
  llmClient?: LLMClient;
  model?: string;
  useLlmFallback?: boolean;
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
  });

  return extractor.analyze(text, recentHistory);
}

export function detectAffectiveSignalHeuristically(text: string): AffectiveSignal {
  return analyzeAffectiveSignalHeuristically(text);
}
