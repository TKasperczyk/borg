import type { LLMClient } from "../../llm/index.js";
import type { ProceduralContext } from "../../memory/procedural/index.js";
import {
  ProceduralContextExtractor,
  type ProceduralContextDegradedReason,
  type ExtractProceduralContextInput,
} from "./context-extractor.js";

export type DeriveProceduralContextInput = ExtractProceduralContextInput;

export type DeriveProceduralContextOptions = {
  llmClient?: LLMClient;
  model?: string;
  onDegraded?: (reason: ProceduralContextDegradedReason, error?: unknown) => Promise<void> | void;
};

export async function deriveProceduralContext(
  input: DeriveProceduralContextInput,
  options: DeriveProceduralContextOptions = {},
): Promise<ProceduralContext | null> {
  return new ProceduralContextExtractor(options).extract(input);
}
