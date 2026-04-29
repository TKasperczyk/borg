import type { LLMClient } from "../../llm/index.js";
import type { ProceduralContext } from "../../memory/procedural/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import type { EntityId } from "../../util/ids.js";
import type { PerceptionResult } from "../types.js";
import {
  ProceduralContextExtractor,
  type ProceduralContextDegradedReason,
} from "./context-extractor.js";

export type DeriveProceduralContextInput = {
  userMessage: string;
  perception: Pick<PerceptionResult, "mode" | "entities">;
  isSelfAudience: boolean;
  audienceEntityId: EntityId | null;
  audienceProfile?: SocialProfile | null;
  inputAudience?: string;
};

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
