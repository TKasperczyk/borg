export { mkdtempSync, rmSync } from "node:fs";
export { join } from "node:path";
export { tmpdir } from "node:os";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { REVIEW_OPEN_QUESTION_TOOL, selfMigrations } from "../../memory/self/index.js";
import { Borg } from "../../borg.js";

export { DEFAULT_CONFIG } from "../../config/index.js";
export { Reflector } from "../../cognition/index.js";
export type { ReflectorOptions } from "../../cognition/index.js";
export {
  FakeLLMClient,
  createFakeEmitAnswerResponse as createEmitAnswerResponse,
} from "../../llm/test-support/fake-client.js";
export type { LLMClient } from "../../llm/index.js";
export { EntityRepository, commitmentMigrations } from "../../memory/commitments/index.js";
export { episodicMigrations } from "../../memory/episodic/index.js";
export { EpisodicRepository, createEpisodesTableSchema } from "../../memory/episodic/repository.js";
export { REVIEW_OPEN_QUESTION_TOOL, selfMigrations };
export { retrievalMigrations } from "../../retrieval/index.js";
export { LanceDbStore } from "../../storage/lancedb/index.js";
export { composeMigrations, openDatabase, SqliteDatabase } from "../../storage/sqlite/index.js";
export { ManualClock } from "../../util/clock.js";
export {
  createEntityId,
  createSemanticEdgeId,
  createEpisodeId,
  createSessionId,
  createStreamEntryId,
} from "../../util/ids.js";
export { createTestConfig } from "../../offline/test-support.js";
export { createMigrations as createBorgMigrations, resolveBorgConfig } from "../storage-setup.js";
export { Borg };

export function borgInternals<T>(borg: Borg): T {
  // TODO(Sprint 9.11): These facade tests still need private dependency access
  // for fault injection and lifecycle race assertions. Replace this with public
  // test hooks if Borg grows stable fault-injection seams.
  return borg as unknown as T;
}

export const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";
export const ENTITY_TOOL_NAME = "EmitEntityExtraction";
export const MODE_TOOL_NAME = "EmitModeDetection";
export const TEMPORAL_TOOL_NAME = "EmitTemporalCue";

export function createTurnPlanResponse() {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_plan",
        name: "EmitTurnPlan",
        input: {
          uncertainty: "",
          verification_steps: [],
          tensions: [],
          voice_note: "",
          intents: [],
        },
      },
    ],
  };
}

export function createGenerationGateResponse(input: {
  decision: "proceed" | "suppress";
  substantive: boolean;
  reason?: string;
}) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_generation_gate",
        name: "EmitGenerationGateDecision",
        input: {
          decision: input.decision,
          substantive: input.substantive,
          reason: input.reason ?? "classified by generation gate",
          confidence: 0.9,
        },
      },
    ],
  };
}

export function createTraitReflectionResponse(input: {
  traitLabel: string;
  evidence: string;
  strengthDelta?: number;
  advancedGoals?: Array<{ goal_id: string; evidence: string }>;
}) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: input.advancedGoals ?? [],
          procedural_outcomes: [],
          trait_demonstrations: [
            {
              trait_label: input.traitLabel,
              evidence: input.evidence,
              strength_delta: input.strengthDelta ?? 0.05,
            },
          ],
          intent_updates: [],
        },
      },
    ],
  };
}

export function createEmptyReflectionResponse(
  openQuestions: Array<{
    question: string;
    urgency: number;
    related_episode_ids: string[];
  }> = [],
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
          open_questions: openQuestions,
        },
      },
    ],
  };
}

export function createReviewOpenQuestionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_review_open_question",
        name: REVIEW_OPEN_QUESTION_TOOL.name,
        input: {
          question: "¿Qué atribución debería revisar?",
          urgency: 0.68,
          related_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          related_semantic_node_ids: [],
        },
      },
    ],
  };
}

export function createInvalidEntityClassifierResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_entity",
        name: ENTITY_TOOL_NAME,
        input: { entities: [1] },
      },
    ],
  };
}

export function createInvalidModeClassifierResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_mode",
        name: MODE_TOOL_NAME,
        input: { mode: "unknown", is_operational: false },
      },
    ],
  };
}

export function createNoTemporalCueResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_temporal",
        name: TEMPORAL_TOOL_NAME,
        input: { has_cue: false },
      },
    ],
  };
}

export class ScriptedEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/Planning sync|planning|Atlas|atlas|pnpm|deploy|rollback/.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}
