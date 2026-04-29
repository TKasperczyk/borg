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
  procedural?: Partial<Config["procedural"]>;
  retrieval?: Partial<Omit<Config["retrieval"], "semantic">> & {
    semantic?: Partial<Config["retrieval"]["semantic"]>;
  };
  generation?: Partial<Config["generation"]>;
  streamIngestion?: Partial<Omit<Config["streamIngestion"], "preTurnCatchup">> & {
    preTurnCatchup?: Partial<Config["streamIngestion"]["preTurnCatchup"]>;
  };
  executive?: Partial<Config["executive"]>;
  offline?: Partial<
    Omit<
      Config["offline"],
      | "consolidator"
      | "reflector"
      | "proceduralSynthesizer"
      | "curator"
      | "overseer"
      | "ruminator"
      | "selfNarrator"
      | "beliefReviser"
    >
  > & {
    consolidator?: Partial<Config["offline"]["consolidator"]>;
    reflector?: Partial<Config["offline"]["reflector"]>;
    proceduralSynthesizer?: Partial<Config["offline"]["proceduralSynthesizer"]>;
    curator?: Partial<Config["offline"]["curator"]>;
    overseer?: Partial<Config["offline"]["overseer"]>;
    ruminator?: Partial<Config["offline"]["ruminator"]>;
    selfNarrator?: Partial<Config["offline"]["selfNarrator"]>;
    beliefReviser?: Partial<Config["offline"]["beliefReviser"]>;
  };
  maintenance?: Partial<Config["maintenance"]>;
  autonomy?: Partial<Omit<Config["autonomy"], "triggers">> & {
    triggers?: Partial<Config["autonomy"]["triggers"]> & {
      commitmentExpiring?: Partial<Config["autonomy"]["triggers"]["commitmentExpiring"]>;
      openQuestionDormant?: Partial<Config["autonomy"]["triggers"]["openQuestionDormant"]>;
      scheduledReflection?: Partial<Config["autonomy"]["triggers"]["scheduledReflection"]>;
      goalFollowupDue?: Partial<Config["autonomy"]["triggers"]["goalFollowupDue"]>;
    };
    conditions?: Partial<Config["autonomy"]["conditions"]> & {
      commitmentRevoked?: Partial<Config["autonomy"]["conditions"]["commitmentRevoked"]>;
      moodValenceDrop?: Partial<Config["autonomy"]["conditions"]["moodValenceDrop"]>;
      openQuestionUrgencyBump?: Partial<
        Config["autonomy"]["conditions"]["openQuestionUrgencyBump"]
      >;
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
  tracerPath?: string;
  env?: NodeJS.ProcessEnv;
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
    procedural: {
      ...DEFAULT_CONFIG.procedural,
      ...options.config?.procedural,
    },
    retrieval: {
      ...DEFAULT_CONFIG.retrieval,
      ...options.config?.retrieval,
      semantic: {
        ...DEFAULT_CONFIG.retrieval.semantic,
        ...options.config?.retrieval?.semantic,
      },
    },
    generation: {
      ...DEFAULT_CONFIG.generation,
      ...options.config?.generation,
    },
    streamIngestion: {
      ...DEFAULT_CONFIG.streamIngestion,
      ...options.config?.streamIngestion,
      preTurnCatchup: {
        ...DEFAULT_CONFIG.streamIngestion.preTurnCatchup,
        ...options.config?.streamIngestion?.preTurnCatchup,
      },
    },
    executive: {
      ...DEFAULT_CONFIG.executive,
      ...options.config?.executive,
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
      proceduralSynthesizer: {
        ...DEFAULT_CONFIG.offline.proceduralSynthesizer,
        ...options.config?.offline?.proceduralSynthesizer,
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
      beliefReviser: {
        ...DEFAULT_CONFIG.offline.beliefReviser,
        ...options.config?.offline?.beliefReviser,
      },
    },
    maintenance: {
      ...DEFAULT_CONFIG.maintenance,
      ...options.config?.maintenance,
      lightProcesses: [
        ...(options.config?.maintenance?.lightProcesses ??
          DEFAULT_CONFIG.maintenance.lightProcesses),
      ],
      heavyProcesses: [
        ...(options.config?.maintenance?.heavyProcesses ??
          DEFAULT_CONFIG.maintenance.heavyProcesses),
      ],
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
        goalFollowupDue: {
          ...DEFAULT_CONFIG.autonomy.triggers.goalFollowupDue,
          ...options.config?.autonomy?.triggers?.goalFollowupDue,
        },
      },
      conditions: {
        ...DEFAULT_CONFIG.autonomy.conditions,
        ...options.config?.autonomy?.conditions,
        commitmentRevoked: {
          ...DEFAULT_CONFIG.autonomy.conditions.commitmentRevoked,
          ...options.config?.autonomy?.conditions?.commitmentRevoked,
        },
        moodValenceDrop: {
          ...DEFAULT_CONFIG.autonomy.conditions.moodValenceDrop,
          ...options.config?.autonomy?.conditions?.moodValenceDrop,
        },
        openQuestionUrgencyBump: {
          ...DEFAULT_CONFIG.autonomy.conditions.openQuestionUrgencyBump,
          ...options.config?.autonomy?.conditions?.openQuestionUrgencyBump,
        },
      },
    },
  };

  return BorgFacade.open({
    config,
    clock: options.clock ?? new FixedClock(1_000_000),
    embeddingDimensions,
    embeddingClient:
      options.embeddingClient ?? new DeterministicEmbeddingClient(embeddingDimensions),
    llmClient: options.llm,
    tracerPath: options.tracerPath,
    env: options.env,
    liveExtraction: false,
  });
}
