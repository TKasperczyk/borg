import type { Borg, Clock, Config, EmbeddingClient, LLMClient } from "../../src/index.js";
import { Borg as BorgFacade, DEFAULT_CONFIG, FixedClock } from "../../src/index.js";

import { DeterministicEmbeddingClient } from "./embedding.js";

export const DEFAULT_EVAL_EMBEDDING_DIMS = 64;

type EvalConfigOverrides = {
  dataDir?: string;
  perception?: Partial<Config["perception"]>;
  affective?: Partial<Config["affective"]>;
  embedding?: Partial<Config["embedding"]>;
  anthropic?: Partial<Omit<Config["anthropic"], "models">> & {
    models?: Partial<Config["anthropic"]["models"]>;
  };
  self?: Partial<Config["self"]>;
  offline?: Partial<
    Omit<
      Config["offline"],
      "consolidator" | "reflector" | "curator" | "overseer" | "ruminator" | "selfNarrator"
    >
  > & {
    consolidator?: Partial<Config["offline"]["consolidator"]>;
    reflector?: Partial<Config["offline"]["reflector"]>;
    curator?: Partial<Config["offline"]["curator"]>;
    overseer?: Partial<Config["offline"]["overseer"]>;
    ruminator?: Partial<Config["offline"]["ruminator"]>;
    selfNarrator?: Partial<Config["offline"]["selfNarrator"]>;
  };
  autonomy?: Partial<Omit<Config["autonomy"], "triggers">> & {
    triggers?: Partial<Config["autonomy"]["triggers"]> & {
      commitmentExpiring?: Partial<Config["autonomy"]["triggers"]["commitmentExpiring"]>;
      openQuestionDormant?: Partial<Config["autonomy"]["triggers"]["openQuestionDormant"]>;
      scheduledReflection?: Partial<Config["autonomy"]["triggers"]["scheduledReflection"]>;
    };
  };
};

export type CreateEvalBorgOptions = {
  tempDir: string;
  llm: LLMClient;
  clock?: Clock;
  embeddingClient?: EmbeddingClient;
  embeddingDimensions?: number;
  config?: EvalConfigOverrides;
};

export async function createEvalBorg(options: CreateEvalBorgOptions): Promise<Borg> {
  const embeddingDimensions = options.embeddingDimensions ?? DEFAULT_EVAL_EMBEDDING_DIMS;
  const config: Config = {
    dataDir: options.tempDir,
    perception: {
      ...DEFAULT_CONFIG.perception,
      ...options.config?.perception,
      useLlmFallback: false,
      modeWhenLlmAbsent: options.config?.perception?.modeWhenLlmAbsent ?? "problem_solving",
    },
    affective: {
      ...DEFAULT_CONFIG.affective,
      ...options.config?.affective,
    },
    embedding: {
      ...DEFAULT_CONFIG.embedding,
      ...options.config?.embedding,
      baseUrl: "http://localhost:1234/v1",
      apiKey: "test",
      model: "eval-deterministic",
      dims: embeddingDimensions,
    },
    anthropic: {
      ...DEFAULT_CONFIG.anthropic,
      ...options.config?.anthropic,
      auth: "api-key",
      apiKey: "test",
      models: {
        ...DEFAULT_CONFIG.anthropic.models,
        ...options.config?.anthropic?.models,
        cognition: "eval-cognition",
        background: "eval-background",
        extraction: "eval-extraction",
      },
    },
    self: {
      ...DEFAULT_CONFIG.self,
      ...options.config?.self,
    },
    offline: {
      ...DEFAULT_CONFIG.offline,
      ...options.config?.offline,
      consolidator: {
        ...DEFAULT_CONFIG.offline.consolidator,
        ...options.config?.offline?.consolidator,
      },
      reflector: {
        ...DEFAULT_CONFIG.offline.reflector,
        ...options.config?.offline?.reflector,
      },
      curator: {
        ...DEFAULT_CONFIG.offline.curator,
        ...options.config?.offline?.curator,
      },
      overseer: {
        ...DEFAULT_CONFIG.offline.overseer,
        ...options.config?.offline?.overseer,
      },
      ruminator: {
        ...DEFAULT_CONFIG.offline.ruminator,
        ...options.config?.offline?.ruminator,
      },
      selfNarrator: {
        ...DEFAULT_CONFIG.offline.selfNarrator,
        ...options.config?.offline?.selfNarrator,
      },
    },
    autonomy: {
      ...DEFAULT_CONFIG.autonomy,
      ...options.config?.autonomy,
      triggers: {
        ...DEFAULT_CONFIG.autonomy.triggers,
        ...options.config?.autonomy?.triggers,
        commitmentExpiring: {
          ...DEFAULT_CONFIG.autonomy.triggers.commitmentExpiring,
          ...options.config?.autonomy?.triggers?.commitmentExpiring,
        },
        openQuestionDormant: {
          ...DEFAULT_CONFIG.autonomy.triggers.openQuestionDormant,
          ...options.config?.autonomy?.triggers?.openQuestionDormant,
        },
        scheduledReflection: {
          ...DEFAULT_CONFIG.autonomy.triggers.scheduledReflection,
          ...options.config?.autonomy?.triggers?.scheduledReflection,
        },
      },
    },
  };

  return BorgFacade.open({
    config,
    clock: options.clock ?? new FixedClock(1_000_000),
    embeddingDimensions,
    embeddingClient: options.embeddingClient ?? new DeterministicEmbeddingClient(embeddingDimensions),
    llmClient: options.llm,
    liveExtraction: false,
  });
}
