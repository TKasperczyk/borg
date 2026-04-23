import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { z } from "zod";

import { readJsonFile } from "../util/atomic-write.js";
import { ConfigError } from "../util/errors.js";

const DEFAULT_DATA_DIR = "~/.borg";
const anthropicAuthModeSchema = z.enum(["auto", "oauth", "api-key"]);

const configFileSchema = z
  .object({
    dataDir: z.string().min(1).optional(),
    defaultUser: z.string().min(1).optional(),
    perception: z
      .object({
        useLlmFallback: z.boolean().optional(),
        modeWhenLlmAbsent: z
          .enum(["problem_solving", "relational", "reflective", "idle"])
          .optional(),
      })
      .partial()
      .optional(),
    affective: z
      .object({
        useLlmFallback: z.boolean().optional(),
        incomingMoodWeight: z.number().min(0).max(1).optional(),
        moodHistoryRetentionDays: z.number().positive().optional(),
        moodHalfLifeHours: z.number().positive().optional(),
      })
      .partial()
      .optional(),
    embedding: z
      .object({
        baseUrl: z.string().url().optional(),
        apiKey: z.string().min(1).optional(),
        model: z.string().min(1).optional(),
        dims: z.number().int().positive().optional(),
      })
      .partial()
      .optional(),
    anthropic: z
      .object({
        auth: anthropicAuthModeSchema.optional(),
        apiKey: z.string().min(1).optional(),
        models: z
          .object({
            cognition: z.string().min(1).optional(),
            background: z.string().min(1).optional(),
            extraction: z.string().min(1).optional(),
          })
          .partial()
          .optional(),
      })
      .partial()
      .optional(),
    self: z
      .object({
        autoBootstrapPeriod: z.boolean().optional(),
      })
      .partial()
      .optional(),
    offline: z
      .object({
        consolidator: z
          .object({
            enabled: z.boolean().optional(),
            similarityThreshold: z.number().positive().optional(),
            minClusterSize: z.number().int().positive().optional(),
            maxClustersPerRun: z.number().int().positive().optional(),
            budget: z.number().int().positive().optional(),
          })
          .partial()
          .optional(),
        reflector: z
          .object({
            enabled: z.boolean().optional(),
            minSupport: z.number().int().positive().optional(),
            ceilingConfidence: z.number().positive().max(0.5).optional(),
            maxInsightsPerRun: z.number().int().positive().optional(),
            budget: z.number().int().positive().optional(),
          })
          .partial()
          .optional(),
        curator: z
          .object({
            enabled: z.boolean().optional(),
            t1Heat: z.number().positive().optional(),
            t2Heat: z.number().positive().optional(),
            t3DemoteHeat: z.number().positive().optional(),
            archiveAgeDays: z.number().positive().optional(),
            archiveMinHeat: z.number().nonnegative().optional(),
            traitHalfLifeDays: z.number().positive().optional(),
          })
          .partial()
          .optional(),
        overseer: z
          .object({
            enabled: z.boolean().optional(),
            lookbackHours: z.number().positive().optional(),
            maxChecksPerRun: z.number().int().positive().optional(),
            budget: z.number().int().positive().optional(),
          })
          .partial()
          .optional(),
        ruminator: z
          .object({
            enabled: z.boolean().optional(),
            maxQuestionsPerRun: z.number().int().positive().optional(),
            resolveConfidenceThreshold: z.number().min(0).max(1).optional(),
            stalenessDays: z.number().positive().optional(),
            budget: z.number().int().positive().optional(),
            perQuestionBudget: z.number().int().positive().optional(),
          })
          .partial()
          .optional(),
        selfNarrator: z
          .object({
            enabled: z.boolean().optional(),
            budget: z.number().int().positive().optional(),
            maxObservationsPerRun: z.number().int().positive().optional(),
            minSupportEpisodes: z.number().int().positive().optional(),
            cadenceHintDays: z.number().positive().optional(),
          })
          .partial()
          .optional(),
      })
      .partial()
      .optional(),
    autonomy: z
      .object({
        enabled: z.boolean().optional(),
        intervalMs: z.number().int().positive().optional(),
        maxWakesPerHour: z.number().int().positive().optional(),
        triggers: z
          .object({
            commitmentExpiring: z
              .object({
                enabled: z.boolean().optional(),
                lookaheadMs: z.number().int().positive().optional(),
              })
              .partial()
              .optional(),
            openQuestionDormant: z
              .object({
                enabled: z.boolean().optional(),
                dormantMs: z.number().int().positive().optional(),
              })
              .partial()
              .optional(),
            scheduledReflection: z
              .object({
                enabled: z.boolean().optional(),
                intervalMs: z.number().int().positive().optional(),
              })
              .partial()
              .optional(),
            goalFollowupDue: z
              .object({
                enabled: z.boolean().optional(),
                lookaheadMs: z.number().int().positive().optional(),
                staleMs: z.number().int().positive().optional(),
              })
              .partial()
              .optional(),
          })
          .partial()
          .optional(),
        conditions: z
          .object({
            commitmentRevoked: z
              .object({
                enabled: z.boolean().optional(),
              })
              .partial()
              .optional(),
            moodValenceDrop: z
              .object({
                enabled: z.boolean().optional(),
                threshold: z.number().min(-1).max(1).optional(),
                windowN: z.number().int().positive().optional(),
                activationPeriodMs: z.number().int().positive().optional(),
              })
              .partial()
              .optional(),
            openQuestionUrgencyBump: z
              .object({
                enabled: z.boolean().optional(),
                threshold: z.number().min(0).max(1).optional(),
              })
              .partial()
              .optional(),
          })
          .partial()
          .optional(),
      })
      .partial()
      .optional(),
  })
  .partial();

export const configSchema = z.object({
  dataDir: z.string().min(1),
  defaultUser: z.string().min(1).optional(),
  perception: z.object({
    useLlmFallback: z.boolean(),
    modeWhenLlmAbsent: z
      .enum(["problem_solving", "relational", "reflective", "idle"])
      .optional(),
  }),
  affective: z.object({
    useLlmFallback: z.boolean(),
    incomingMoodWeight: z.number().min(0).max(1),
    moodHistoryRetentionDays: z.number().positive(),
    moodHalfLifeHours: z.number().positive(),
  }),
  embedding: z.object({
    baseUrl: z.string().url(),
    apiKey: z.string().min(1).optional(),
    model: z.string().min(1),
    dims: z.number().int().positive(),
  }),
  anthropic: z
    .object({
      auth: anthropicAuthModeSchema,
      apiKey: z.string().min(1).optional(),
      models: z.object({
        cognition: z.string().min(1),
        background: z.string().min(1),
        extraction: z.string().min(1),
      }),
    })
    .superRefine((value, context) => {
      if (value.auth === "api-key" && value.apiKey === undefined) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Anthropic API key must be configured when anthropic.auth is api-key",
          path: ["apiKey"],
        });
      }
    }),
  self: z.object({
    autoBootstrapPeriod: z.boolean(),
  }),
  offline: z.object({
    consolidator: z.object({
      enabled: z.boolean(),
      similarityThreshold: z.number().positive(),
      minClusterSize: z.number().int().positive(),
      maxClustersPerRun: z.number().int().positive(),
      budget: z.number().int().positive(),
    }),
    reflector: z.object({
      enabled: z.boolean(),
      minSupport: z.number().int().positive(),
      ceilingConfidence: z.number().positive().max(0.5),
      maxInsightsPerRun: z.number().int().positive(),
      budget: z.number().int().positive(),
    }),
    curator: z.object({
      enabled: z.boolean(),
      t1Heat: z.number().positive(),
      t2Heat: z.number().positive(),
      t3DemoteHeat: z.number().positive(),
      archiveAgeDays: z.number().positive(),
      archiveMinHeat: z.number().nonnegative(),
      traitHalfLifeDays: z.number().positive(),
    }),
    overseer: z.object({
      enabled: z.boolean(),
      lookbackHours: z.number().positive(),
      maxChecksPerRun: z.number().int().positive(),
      budget: z.number().int().positive(),
    }),
    ruminator: z.object({
      enabled: z.boolean(),
      maxQuestionsPerRun: z.number().int().positive(),
      resolveConfidenceThreshold: z.number().min(0).max(1),
      stalenessDays: z.number().positive(),
      budget: z.number().int().positive(),
      perQuestionBudget: z.number().int().positive(),
    }),
    selfNarrator: z.object({
      enabled: z.boolean(),
      budget: z.number().int().positive(),
      maxObservationsPerRun: z.number().int().positive(),
      minSupportEpisodes: z.number().int().positive(),
      cadenceHintDays: z.number().positive(),
    }),
  }),
  autonomy: z.object({
    enabled: z.boolean(),
    intervalMs: z.number().int().positive(),
    maxWakesPerHour: z.number().int().positive(),
    triggers: z.object({
      commitmentExpiring: z.object({
        enabled: z.boolean(),
        lookaheadMs: z.number().int().positive(),
      }),
      openQuestionDormant: z.object({
        enabled: z.boolean(),
        dormantMs: z.number().int().positive(),
      }),
      scheduledReflection: z.object({
        enabled: z.boolean(),
        intervalMs: z.number().int().positive(),
      }),
      goalFollowupDue: z.object({
        enabled: z.boolean(),
        lookaheadMs: z.number().int().positive(),
        staleMs: z.number().int().positive(),
      }),
    }),
    conditions: z.object({
      commitmentRevoked: z.object({
        enabled: z.boolean(),
      }),
      moodValenceDrop: z.object({
        enabled: z.boolean(),
        threshold: z.number().min(-1).max(1),
        windowN: z.number().int().positive(),
        activationPeriodMs: z.number().int().positive(),
      }),
      openQuestionUrgencyBump: z.object({
        enabled: z.boolean(),
        threshold: z.number().min(0).max(1),
      }),
    }),
  }),
});

export type Config = z.infer<typeof configSchema>;

export const DEFAULT_CONFIG: Config = {
  dataDir: expandPath(DEFAULT_DATA_DIR),
  defaultUser: undefined,
  perception: {
    useLlmFallback: true,
  },
  affective: {
    useLlmFallback: false,
    incomingMoodWeight: 0.3,
    moodHistoryRetentionDays: 90,
    moodHalfLifeHours: 24,
  },
  embedding: {
    baseUrl: "http://localhost:1234/v1",
    apiKey: "lm-studio",
    model: "text-embedding-qwen3-embedding-8b",
    dims: 4096,
  },
  anthropic: {
    auth: "auto",
    apiKey: undefined,
    models: {
      cognition: "claude-opus-4-7",
      background: "claude-haiku-4-5-20251001",
      extraction: "claude-sonnet-4-6",
    },
  },
  self: {
    autoBootstrapPeriod: true,
  },
  offline: {
    consolidator: {
      enabled: true,
      similarityThreshold: 0.82,
      minClusterSize: 2,
      maxClustersPerRun: 2,
      budget: 60_000,
    },
    reflector: {
      enabled: true,
      minSupport: 3,
      ceilingConfidence: 0.5,
      maxInsightsPerRun: 2,
      budget: 60_000,
    },
    curator: {
      enabled: true,
      t1Heat: 5,
      t2Heat: 15,
      t3DemoteHeat: 3,
      archiveAgeDays: 45,
      archiveMinHeat: 1,
      traitHalfLifeDays: 30,
    },
    overseer: {
      enabled: true,
      lookbackHours: 24,
      maxChecksPerRun: 8,
      budget: 40_000,
    },
    ruminator: {
      enabled: true,
      maxQuestionsPerRun: 3,
      resolveConfidenceThreshold: 0.65,
      stalenessDays: 30,
      budget: 6_000,
      perQuestionBudget: 8_000,
    },
    selfNarrator: {
      enabled: true,
      budget: 80_000,
      maxObservationsPerRun: 4,
      minSupportEpisodes: 2,
      cadenceHintDays: 7,
    },
  },
  autonomy: {
    enabled: false,
    intervalMs: 60_000,
    maxWakesPerHour: 6,
    triggers: {
      commitmentExpiring: {
        enabled: true,
        lookaheadMs: 86_400_000,
      },
      openQuestionDormant: {
        enabled: true,
        dormantMs: 604_800_000,
      },
      scheduledReflection: {
        enabled: false,
        intervalMs: 14_400_000,
      },
      goalFollowupDue: {
        enabled: true,
        lookaheadMs: 604_800_000,
        staleMs: 1_209_600_000,
      },
    },
    conditions: {
      commitmentRevoked: {
        enabled: true,
      },
      moodValenceDrop: {
        enabled: false,
        threshold: -0.5,
        windowN: 5,
        activationPeriodMs: 86_400_000,
      },
      openQuestionUrgencyBump: {
        enabled: true,
        threshold: 0.9,
      },
    },
  },
};

export type LoadConfigOptions = {
  dataDir?: string;
  env?: NodeJS.ProcessEnv;
};

export function expandPath(pathLike: string): string {
  if (pathLike === "~") {
    return homedir();
  }

  if (pathLike.startsWith("~/")) {
    return join(homedir(), pathLike.slice(2));
  }

  return isAbsolute(pathLike) ? pathLike : resolve(pathLike);
}

function readOptionalEnvString(env: NodeJS.ProcessEnv, name: string): string | undefined {
  const value = env[name]?.trim();
  return value ? value : undefined;
}

function readOptionalEnvNumber(env: NodeJS.ProcessEnv, name: string): number | undefined {
  const raw = readOptionalEnvString(env, name);

  if (raw === undefined) {
    return undefined;
  }

  const value = Number(raw);

  if (!Number.isInteger(value) || value <= 0) {
    throw new ConfigError(`Environment variable ${name} must be a positive integer`);
  }

  return value;
}

function readOptionalEnvFloat(env: NodeJS.ProcessEnv, name: string): number | undefined {
  const raw = readOptionalEnvString(env, name);

  if (raw === undefined) {
    return undefined;
  }

  const value = Number(raw);

  if (!Number.isFinite(value) || value <= 0) {
    throw new ConfigError(`Environment variable ${name} must be a positive number`);
  }

  return value;
}

function readOptionalEnvUnitInterval(env: NodeJS.ProcessEnv, name: string): number | undefined {
  const raw = readOptionalEnvString(env, name);

  if (raw === undefined) {
    return undefined;
  }

  const value = Number(raw);

  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new ConfigError(`Environment variable ${name} must be between 0 and 1`);
  }

  return value;
}

function readOptionalEnvBoolean(env: NodeJS.ProcessEnv, name: string): boolean | undefined {
  const raw = readOptionalEnvString(env, name);

  if (raw === undefined) {
    return undefined;
  }

  if (raw === "true" || raw === "1") {
    return true;
  }

  if (raw === "false" || raw === "0") {
    return false;
  }

  throw new ConfigError(`Environment variable ${name} must be true/false or 1/0`);
}

function readOptionalEnvAnthropicAuthMode(
  env: NodeJS.ProcessEnv,
  name: string,
): z.infer<typeof anthropicAuthModeSchema> | undefined {
  const raw = readOptionalEnvString(env, name);

  if (raw === undefined) {
    return undefined;
  }

  const parsed = anthropicAuthModeSchema.safeParse(raw);

  if (!parsed.success) {
    throw new ConfigError(
      `Environment variable ${name} must be one of: ${anthropicAuthModeSchema.options.join(", ")}`,
    );
  }

  return parsed.data;
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException & { code: string } {
  return error instanceof Error && typeof (error as NodeJS.ErrnoException).code === "string";
}

function parseConfigFile(dataDir: string): z.infer<typeof configFileSchema> {
  const configPath = join(dataDir, "config.json");

  try {
    const rawConfig = readJsonFile<unknown>(configPath);

    if (rawConfig === undefined) {
      return {};
    }

    const parsed = configFileSchema.safeParse(rawConfig);

    if (!parsed.success) {
      throw new ConfigError(`Invalid config file at ${configPath}`, {
        cause: parsed.error,
        code: "CONFIG_FILE_INVALID",
      });
    }

    return parsed.data;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return {};
    }

    if (error instanceof ConfigError) {
      throw error;
    }

    throw new ConfigError(`Invalid config file at ${configPath}`, {
      cause: error,
      code: "CONFIG_FILE_INVALID",
    });
  }
}

export function loadConfig(options: LoadConfigOptions = {}): Config {
  const env = options.env ?? process.env;
  const lookupDataDir = expandPath(
    options.dataDir ?? readOptionalEnvString(env, "BORG_DATA_DIR") ?? DEFAULT_DATA_DIR,
  );
  const fileConfig = parseConfigFile(lookupDataDir);

  const candidate = {
    dataDir: expandPath(
      options.dataDir ??
        readOptionalEnvString(env, "BORG_DATA_DIR") ??
        fileConfig.dataDir ??
        DEFAULT_CONFIG.dataDir,
    ),
    defaultUser:
      readOptionalEnvString(env, "BORG_DEFAULT_USER") ??
      fileConfig.defaultUser ??
      DEFAULT_CONFIG.defaultUser,
    perception: {
      useLlmFallback:
        readOptionalEnvBoolean(env, "BORG_PERCEPTION_USE_LLM_FALLBACK") ??
        fileConfig.perception?.useLlmFallback ??
        DEFAULT_CONFIG.perception.useLlmFallback,
      modeWhenLlmAbsent:
        fileConfig.perception?.modeWhenLlmAbsent ??
        DEFAULT_CONFIG.perception.modeWhenLlmAbsent,
    },
    affective: {
      useLlmFallback:
        readOptionalEnvBoolean(env, "BORG_AFFECTIVE_USE_LLM_FALLBACK") ??
        fileConfig.affective?.useLlmFallback ??
        DEFAULT_CONFIG.affective.useLlmFallback,
      incomingMoodWeight:
        readOptionalEnvUnitInterval(env, "BORG_AFFECTIVE_INCOMING_MOOD_WEIGHT") ??
        fileConfig.affective?.incomingMoodWeight ??
        DEFAULT_CONFIG.affective.incomingMoodWeight,
      moodHistoryRetentionDays:
        readOptionalEnvFloat(env, "BORG_AFFECTIVE_MOOD_HISTORY_RETENTION_DAYS") ??
        fileConfig.affective?.moodHistoryRetentionDays ??
        DEFAULT_CONFIG.affective.moodHistoryRetentionDays,
      moodHalfLifeHours:
        readOptionalEnvFloat(env, "BORG_AFFECTIVE_MOOD_HALF_LIFE_HOURS") ??
        fileConfig.affective?.moodHalfLifeHours ??
        DEFAULT_CONFIG.affective.moodHalfLifeHours,
    },
    embedding: {
      baseUrl:
        readOptionalEnvString(env, "BORG_EMBEDDING_BASE_URL") ??
        fileConfig.embedding?.baseUrl ??
        DEFAULT_CONFIG.embedding.baseUrl,
      apiKey:
        readOptionalEnvString(env, "BORG_EMBEDDING_API_KEY") ??
        fileConfig.embedding?.apiKey ??
        DEFAULT_CONFIG.embedding.apiKey,
      model:
        readOptionalEnvString(env, "BORG_EMBEDDING_MODEL") ??
        fileConfig.embedding?.model ??
        DEFAULT_CONFIG.embedding.model,
      dims:
        readOptionalEnvNumber(env, "BORG_EMBEDDING_DIMS") ??
        fileConfig.embedding?.dims ??
        DEFAULT_CONFIG.embedding.dims,
    },
    anthropic: {
      auth:
        readOptionalEnvAnthropicAuthMode(env, "BORG_ANTHROPIC_AUTH") ??
        fileConfig.anthropic?.auth ??
        DEFAULT_CONFIG.anthropic.auth,
      apiKey:
        readOptionalEnvString(env, "ANTHROPIC_API_KEY") ??
        fileConfig.anthropic?.apiKey ??
        DEFAULT_CONFIG.anthropic.apiKey,
      models: {
        cognition:
          readOptionalEnvString(env, "BORG_MODEL_COGNITION") ??
          fileConfig.anthropic?.models?.cognition ??
          DEFAULT_CONFIG.anthropic.models.cognition,
        background:
          readOptionalEnvString(env, "BORG_MODEL_BACKGROUND") ??
          fileConfig.anthropic?.models?.background ??
          DEFAULT_CONFIG.anthropic.models.background,
        extraction:
          readOptionalEnvString(env, "BORG_MODEL_EXTRACTION") ??
          fileConfig.anthropic?.models?.extraction ??
          DEFAULT_CONFIG.anthropic.models.extraction,
      },
    },
    self: {
      autoBootstrapPeriod:
        readOptionalEnvBoolean(env, "BORG_SELF_AUTO_BOOTSTRAP_PERIOD") ??
        fileConfig.self?.autoBootstrapPeriod ??
        DEFAULT_CONFIG.self.autoBootstrapPeriod,
    },
    offline: {
      consolidator: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_CONSOLIDATOR_ENABLED") ??
          fileConfig.offline?.consolidator?.enabled ??
          DEFAULT_CONFIG.offline.consolidator.enabled,
        similarityThreshold:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CONSOLIDATOR_SIMILARITY_THRESHOLD") ??
          fileConfig.offline?.consolidator?.similarityThreshold ??
          DEFAULT_CONFIG.offline.consolidator.similarityThreshold,
        minClusterSize:
          readOptionalEnvNumber(env, "BORG_OFFLINE_CONSOLIDATOR_MIN_CLUSTER_SIZE") ??
          fileConfig.offline?.consolidator?.minClusterSize ??
          DEFAULT_CONFIG.offline.consolidator.minClusterSize,
        maxClustersPerRun:
          readOptionalEnvNumber(env, "BORG_OFFLINE_CONSOLIDATOR_MAX_CLUSTERS_PER_RUN") ??
          fileConfig.offline?.consolidator?.maxClustersPerRun ??
          DEFAULT_CONFIG.offline.consolidator.maxClustersPerRun,
        budget:
          readOptionalEnvNumber(env, "BORG_OFFLINE_CONSOLIDATOR_BUDGET") ??
          fileConfig.offline?.consolidator?.budget ??
          DEFAULT_CONFIG.offline.consolidator.budget,
      },
      reflector: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_REFLECTOR_ENABLED") ??
          fileConfig.offline?.reflector?.enabled ??
          DEFAULT_CONFIG.offline.reflector.enabled,
        minSupport:
          readOptionalEnvNumber(env, "BORG_OFFLINE_REFLECTOR_MIN_SUPPORT") ??
          fileConfig.offline?.reflector?.minSupport ??
          DEFAULT_CONFIG.offline.reflector.minSupport,
        ceilingConfidence:
          readOptionalEnvFloat(env, "BORG_OFFLINE_REFLECTOR_CEILING_CONFIDENCE") ??
          fileConfig.offline?.reflector?.ceilingConfidence ??
          DEFAULT_CONFIG.offline.reflector.ceilingConfidence,
        maxInsightsPerRun:
          readOptionalEnvNumber(env, "BORG_OFFLINE_REFLECTOR_MAX_INSIGHTS_PER_RUN") ??
          fileConfig.offline?.reflector?.maxInsightsPerRun ??
          DEFAULT_CONFIG.offline.reflector.maxInsightsPerRun,
        budget:
          readOptionalEnvNumber(env, "BORG_OFFLINE_REFLECTOR_BUDGET") ??
          fileConfig.offline?.reflector?.budget ??
          DEFAULT_CONFIG.offline.reflector.budget,
      },
      curator: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_CURATOR_ENABLED") ??
          fileConfig.offline?.curator?.enabled ??
          DEFAULT_CONFIG.offline.curator.enabled,
        t1Heat:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_T1_HEAT") ??
          fileConfig.offline?.curator?.t1Heat ??
          DEFAULT_CONFIG.offline.curator.t1Heat,
        t2Heat:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_T2_HEAT") ??
          fileConfig.offline?.curator?.t2Heat ??
          DEFAULT_CONFIG.offline.curator.t2Heat,
        t3DemoteHeat:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_T3_DEMOTE_HEAT") ??
          fileConfig.offline?.curator?.t3DemoteHeat ??
          DEFAULT_CONFIG.offline.curator.t3DemoteHeat,
        archiveAgeDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_ARCHIVE_AGE_DAYS") ??
          fileConfig.offline?.curator?.archiveAgeDays ??
          DEFAULT_CONFIG.offline.curator.archiveAgeDays,
        archiveMinHeat:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_ARCHIVE_MIN_HEAT") ??
          fileConfig.offline?.curator?.archiveMinHeat ??
          DEFAULT_CONFIG.offline.curator.archiveMinHeat,
        traitHalfLifeDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_TRAIT_HALF_LIFE_DAYS") ??
          fileConfig.offline?.curator?.traitHalfLifeDays ??
          DEFAULT_CONFIG.offline.curator.traitHalfLifeDays,
      },
      overseer: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_OVERSEER_ENABLED") ??
          fileConfig.offline?.overseer?.enabled ??
          DEFAULT_CONFIG.offline.overseer.enabled,
        lookbackHours:
          readOptionalEnvFloat(env, "BORG_OFFLINE_OVERSEER_LOOKBACK_HOURS") ??
          fileConfig.offline?.overseer?.lookbackHours ??
          DEFAULT_CONFIG.offline.overseer.lookbackHours,
        maxChecksPerRun:
          readOptionalEnvNumber(env, "BORG_OFFLINE_OVERSEER_MAX_CHECKS_PER_RUN") ??
          fileConfig.offline?.overseer?.maxChecksPerRun ??
          DEFAULT_CONFIG.offline.overseer.maxChecksPerRun,
        budget:
          readOptionalEnvNumber(env, "BORG_OFFLINE_OVERSEER_BUDGET") ??
          fileConfig.offline?.overseer?.budget ??
          DEFAULT_CONFIG.offline.overseer.budget,
      },
      ruminator: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_RUMINATOR_ENABLED") ??
          fileConfig.offline?.ruminator?.enabled ??
          DEFAULT_CONFIG.offline.ruminator.enabled,
        maxQuestionsPerRun:
          readOptionalEnvNumber(env, "BORG_OFFLINE_RUMINATOR_MAX_QUESTIONS_PER_RUN") ??
          fileConfig.offline?.ruminator?.maxQuestionsPerRun ??
          DEFAULT_CONFIG.offline.ruminator.maxQuestionsPerRun,
        resolveConfidenceThreshold:
          readOptionalEnvFloat(env, "BORG_OFFLINE_RUMINATOR_RESOLVE_CONFIDENCE_THRESHOLD") ??
          fileConfig.offline?.ruminator?.resolveConfidenceThreshold ??
          DEFAULT_CONFIG.offline.ruminator.resolveConfidenceThreshold,
        stalenessDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_RUMINATOR_STALENESS_DAYS") ??
          fileConfig.offline?.ruminator?.stalenessDays ??
          DEFAULT_CONFIG.offline.ruminator.stalenessDays,
        budget:
          readOptionalEnvNumber(env, "BORG_OFFLINE_RUMINATOR_BUDGET") ??
          fileConfig.offline?.ruminator?.budget ??
          DEFAULT_CONFIG.offline.ruminator.budget,
        perQuestionBudget:
          readOptionalEnvNumber(env, "BORG_OFFLINE_RUMINATOR_PER_QUESTION_BUDGET") ??
          fileConfig.offline?.ruminator?.perQuestionBudget ??
          DEFAULT_CONFIG.offline.ruminator.perQuestionBudget,
      },
      selfNarrator: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_SELF_NARRATOR_ENABLED") ??
          fileConfig.offline?.selfNarrator?.enabled ??
          DEFAULT_CONFIG.offline.selfNarrator.enabled,
        budget:
          readOptionalEnvNumber(env, "BORG_OFFLINE_SELF_NARRATOR_BUDGET") ??
          fileConfig.offline?.selfNarrator?.budget ??
          DEFAULT_CONFIG.offline.selfNarrator.budget,
        maxObservationsPerRun:
          readOptionalEnvNumber(env, "BORG_OFFLINE_SELF_NARRATOR_MAX_OBSERVATIONS_PER_RUN") ??
          fileConfig.offline?.selfNarrator?.maxObservationsPerRun ??
          DEFAULT_CONFIG.offline.selfNarrator.maxObservationsPerRun,
        minSupportEpisodes:
          readOptionalEnvNumber(env, "BORG_OFFLINE_SELF_NARRATOR_MIN_SUPPORT_EPISODES") ??
          fileConfig.offline?.selfNarrator?.minSupportEpisodes ??
          DEFAULT_CONFIG.offline.selfNarrator.minSupportEpisodes,
        cadenceHintDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_SELF_NARRATOR_CADENCE_HINT_DAYS") ??
          fileConfig.offline?.selfNarrator?.cadenceHintDays ??
          DEFAULT_CONFIG.offline.selfNarrator.cadenceHintDays,
      },
    },
    autonomy: {
      enabled:
        readOptionalEnvBoolean(env, "BORG_AUTONOMY_ENABLED") ??
        fileConfig.autonomy?.enabled ??
        DEFAULT_CONFIG.autonomy.enabled,
      intervalMs:
        readOptionalEnvNumber(env, "BORG_AUTONOMY_INTERVAL_MS") ??
        fileConfig.autonomy?.intervalMs ??
        DEFAULT_CONFIG.autonomy.intervalMs,
      maxWakesPerHour:
        readOptionalEnvNumber(env, "BORG_AUTONOMY_MAX_WAKES_PER_HOUR") ??
        fileConfig.autonomy?.maxWakesPerHour ??
        DEFAULT_CONFIG.autonomy.maxWakesPerHour,
      triggers: {
        commitmentExpiring: {
          enabled:
            readOptionalEnvBoolean(env, "BORG_AUTONOMY_TRIGGER_COMMITMENT_EXPIRING_ENABLED") ??
            fileConfig.autonomy?.triggers?.commitmentExpiring?.enabled ??
            DEFAULT_CONFIG.autonomy.triggers.commitmentExpiring.enabled,
          lookaheadMs:
            readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_COMMITMENT_EXPIRING_LOOKAHEAD_MS") ??
            fileConfig.autonomy?.triggers?.commitmentExpiring?.lookaheadMs ??
            DEFAULT_CONFIG.autonomy.triggers.commitmentExpiring.lookaheadMs,
        },
        openQuestionDormant: {
          enabled:
            readOptionalEnvBoolean(env, "BORG_AUTONOMY_TRIGGER_OPEN_QUESTION_DORMANT_ENABLED") ??
            fileConfig.autonomy?.triggers?.openQuestionDormant?.enabled ??
            DEFAULT_CONFIG.autonomy.triggers.openQuestionDormant.enabled,
          dormantMs:
            readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_OPEN_QUESTION_DORMANT_MS") ??
            fileConfig.autonomy?.triggers?.openQuestionDormant?.dormantMs ??
            DEFAULT_CONFIG.autonomy.triggers.openQuestionDormant.dormantMs,
        },
        scheduledReflection: {
          enabled:
            readOptionalEnvBoolean(env, "BORG_AUTONOMY_TRIGGER_SCHEDULED_REFLECTION_ENABLED") ??
            fileConfig.autonomy?.triggers?.scheduledReflection?.enabled ??
            DEFAULT_CONFIG.autonomy.triggers.scheduledReflection.enabled,
          intervalMs:
            readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_SCHEDULED_REFLECTION_INTERVAL_MS") ??
            fileConfig.autonomy?.triggers?.scheduledReflection?.intervalMs ??
            DEFAULT_CONFIG.autonomy.triggers.scheduledReflection.intervalMs,
        },
        goalFollowupDue: {
          enabled:
            readOptionalEnvBoolean(env, "BORG_AUTONOMY_TRIGGER_GOAL_FOLLOWUP_DUE_ENABLED") ??
            fileConfig.autonomy?.triggers?.goalFollowupDue?.enabled ??
            DEFAULT_CONFIG.autonomy.triggers.goalFollowupDue.enabled,
          lookaheadMs:
            readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_GOAL_FOLLOWUP_DUE_LOOKAHEAD_MS") ??
            fileConfig.autonomy?.triggers?.goalFollowupDue?.lookaheadMs ??
            DEFAULT_CONFIG.autonomy.triggers.goalFollowupDue.lookaheadMs,
          staleMs:
            readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_GOAL_FOLLOWUP_DUE_STALE_MS") ??
            fileConfig.autonomy?.triggers?.goalFollowupDue?.staleMs ??
            DEFAULT_CONFIG.autonomy.triggers.goalFollowupDue.staleMs,
        },
      },
      conditions: {
        commitmentRevoked: {
          enabled:
            readOptionalEnvBoolean(env, "BORG_AUTONOMY_CONDITION_COMMITMENT_REVOKED_ENABLED") ??
            fileConfig.autonomy?.conditions?.commitmentRevoked?.enabled ??
            DEFAULT_CONFIG.autonomy.conditions.commitmentRevoked.enabled,
        },
        moodValenceDrop: {
          enabled:
            readOptionalEnvBoolean(env, "BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_ENABLED") ??
            fileConfig.autonomy?.conditions?.moodValenceDrop?.enabled ??
            DEFAULT_CONFIG.autonomy.conditions.moodValenceDrop.enabled,
          threshold:
            readOptionalEnvFloat(env, "BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_THRESHOLD") ??
            fileConfig.autonomy?.conditions?.moodValenceDrop?.threshold ??
            DEFAULT_CONFIG.autonomy.conditions.moodValenceDrop.threshold,
          windowN:
            readOptionalEnvNumber(env, "BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_WINDOW_N") ??
            fileConfig.autonomy?.conditions?.moodValenceDrop?.windowN ??
            DEFAULT_CONFIG.autonomy.conditions.moodValenceDrop.windowN,
          activationPeriodMs:
            readOptionalEnvNumber(
              env,
              "BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_ACTIVATION_PERIOD_MS",
            ) ??
            fileConfig.autonomy?.conditions?.moodValenceDrop?.activationPeriodMs ??
            DEFAULT_CONFIG.autonomy.conditions.moodValenceDrop.activationPeriodMs,
        },
        openQuestionUrgencyBump: {
          enabled:
            readOptionalEnvBoolean(
              env,
              "BORG_AUTONOMY_CONDITION_OPEN_QUESTION_URGENCY_BUMP_ENABLED",
            ) ??
            fileConfig.autonomy?.conditions?.openQuestionUrgencyBump?.enabled ??
            DEFAULT_CONFIG.autonomy.conditions.openQuestionUrgencyBump.enabled,
          threshold:
            readOptionalEnvFloat(
              env,
              "BORG_AUTONOMY_CONDITION_OPEN_QUESTION_URGENCY_BUMP_THRESHOLD",
            ) ??
            fileConfig.autonomy?.conditions?.openQuestionUrgencyBump?.threshold ??
            DEFAULT_CONFIG.autonomy.conditions.openQuestionUrgencyBump.threshold,
        },
      },
    },
  };

  const parsed = configSchema.safeParse(candidate);

  if (!parsed.success) {
    throw new ConfigError("Invalid borg configuration", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function redactSecret(value: string | undefined): string | undefined {
  return value === undefined ? undefined : "[REDACTED]";
}

export function redactConfig(config: Config): Config {
  return {
    ...config,
    perception: {
      ...config.perception,
    },
    affective: {
      ...config.affective,
    },
    embedding: {
      ...config.embedding,
      apiKey: redactSecret(config.embedding.apiKey),
    },
    anthropic: {
      ...config.anthropic,
      apiKey: redactSecret(config.anthropic.apiKey),
    },
    self: {
      ...config.self,
    },
    offline: {
      ...config.offline,
    },
    autonomy: {
      ...config.autonomy,
      triggers: {
        ...config.autonomy.triggers,
        commitmentExpiring: {
          ...config.autonomy.triggers.commitmentExpiring,
        },
        openQuestionDormant: {
          ...config.autonomy.triggers.openQuestionDormant,
        },
        scheduledReflection: {
          ...config.autonomy.triggers.scheduledReflection,
        },
        goalFollowupDue: {
          ...config.autonomy.triggers.goalFollowupDue,
        },
      },
      conditions: {
        ...config.autonomy.conditions,
        commitmentRevoked: {
          ...config.autonomy.conditions.commitmentRevoked,
        },
        moodValenceDrop: {
          ...config.autonomy.conditions.moodValenceDrop,
        },
        openQuestionUrgencyBump: {
          ...config.autonomy.conditions.openQuestionUrgencyBump,
        },
      },
    },
  };
}
