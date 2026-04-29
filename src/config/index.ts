import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { z } from "zod";

import { DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD } from "../executive/index.js";
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
    procedural: z
      .object({
        skillSelectionMinSimilarity: z.number().min(0).max(1).optional(),
      })
      .partial()
      .optional(),
    retrieval: z
      .object({
        semantic: z
          .object({
            underReviewMultiplier: z.number().min(0).max(1).optional(),
          })
          .partial()
          .optional(),
      })
      .partial()
      .optional(),
    generation: z
      .object({
        discourseStateHardCapTurns: z.number().int().positive().optional(),
      })
      .partial()
      .optional(),
    streamIngestion: z
      .object({
        preTurnCatchup: z
          .object({
            maxEntries: z.number().int().positive().optional(),
          })
          .partial()
          .optional(),
      })
      .partial()
      .optional(),
    executive: z
      .object({
        goalFocusThreshold: z.number().min(0).max(1).optional(),
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
            goalSimilarityThreshold: z.number().min(0).max(1).optional(),
            ceilingConfidence: z.number().positive().max(0.5).optional(),
            maxInsightsPerRun: z.number().int().positive().optional(),
            budget: z.number().int().positive().optional(),
          })
          .partial()
          .optional(),
        proceduralSynthesizer: z
          .object({
            enabled: z.boolean().optional(),
            minSupport: z.number().int().positive().optional(),
            maxSkillsPerRun: z.number().int().positive().optional(),
            dedupThreshold: z.number().min(0).max(1).optional(),
            minContextAttemptsForSplit: z.number().int().positive().optional(),
            minDivergenceForSplit: z.number().min(0).max(1).optional(),
            splitCooldownDays: z.number().positive().optional(),
            splitClaimStaleSec: z.number().int().positive().optional(),
            maxSplitParseFailures: z.number().int().positive().optional(),
            skillSplitDryRun: z.boolean().optional(),
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
            episodeDecayIntervalMs: z.number().positive().optional(),
            episodeSalienceHalfLifeDays: z.number().positive().optional(),
            episodeHeatHalfLifeDays: z.number().positive().optional(),
            traitHalfLifeDays: z.number().positive().optional(),
            retrievalLogRetentionDays: z.number().positive().optional(),
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
        beliefReviser: z
          .object({
            enabled: z.boolean().optional(),
            confidenceDropMultiplier: z.number().min(0).max(1).optional(),
            confidenceFloor: z.number().min(0).max(1).optional(),
            regradeBatchSize: z.number().int().positive().optional(),
            maxEventsPerRun: z.number().int().positive().optional(),
            maxReviewsPerRun: z.number().int().positive().optional(),
            claimStaleSec: z.number().positive().optional(),
            maxParseFailures: z.number().int().positive().optional(),
            maxLlmCalls: z.number().int().positive().optional(),
            consecutiveParseFailureLimit: z.number().int().positive().optional(),
          })
          .partial()
          .optional(),
      })
      .partial()
      .optional(),
    maintenance: z
      .object({
        enabled: z.boolean().optional(),
        lightIntervalMs: z.number().int().positive().optional(),
        heavyIntervalMs: z.number().int().positive().optional(),
        lightProcesses: z
          .array(
            z.enum([
              "consolidator",
              "reflector",
              "curator",
              "overseer",
              "ruminator",
              "self-narrator",
              "procedural-synthesizer",
              "belief-reviser",
            ]),
          )
          .optional(),
        heavyProcesses: z
          .array(
            z.enum([
              "consolidator",
              "reflector",
              "curator",
              "overseer",
              "ruminator",
              "self-narrator",
              "procedural-synthesizer",
              "belief-reviser",
            ]),
          )
          .optional(),
      })
      .partial()
      .optional(),
    autonomy: z
      .object({
        enabled: z.boolean().optional(),
        intervalMs: z.number().int().positive().optional(),
        maxWakesPerWindow: z.number().int().positive().optional(),
        budgetWindowMs: z.number().int().positive().optional(),
        executiveFocus: z
          .object({
            enabled: z.boolean().optional(),
            stalenessSec: z.number().int().positive().optional(),
            dueLeadSec: z.number().int().nonnegative().optional(),
            wakeCooldownSec: z.number().int().nonnegative().optional(),
          })
          .partial()
          .optional(),
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
    modeWhenLlmAbsent: z.enum(["problem_solving", "relational", "reflective", "idle"]).optional(),
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
  procedural: z.object({
    skillSelectionMinSimilarity: z.number().min(0).max(1),
  }),
  retrieval: z.object({
    semantic: z.object({
      underReviewMultiplier: z.number().min(0).max(1),
    }),
  }),
  generation: z.object({
    discourseStateHardCapTurns: z.number().int().positive(),
  }),
  streamIngestion: z.object({
    preTurnCatchup: z.object({
      maxEntries: z.number().int().positive(),
    }),
  }),
  executive: z.object({
    goalFocusThreshold: z.number().min(0).max(1),
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
      goalSimilarityThreshold: z.number().min(0).max(1),
      ceilingConfidence: z.number().positive().max(0.5),
      maxInsightsPerRun: z.number().int().positive(),
      budget: z.number().int().positive(),
    }),
    proceduralSynthesizer: z.object({
      enabled: z.boolean(),
      minSupport: z.number().int().positive(),
      maxSkillsPerRun: z.number().int().positive(),
      dedupThreshold: z.number().min(0).max(1),
      minContextAttemptsForSplit: z.number().int().positive(),
      minDivergenceForSplit: z.number().min(0).max(1),
      splitCooldownDays: z.number().positive(),
      splitClaimStaleSec: z.number().int().positive(),
      maxSplitParseFailures: z.number().int().positive(),
      skillSplitDryRun: z.boolean(),
      budget: z.number().int().positive(),
    }),
    curator: z.object({
      enabled: z.boolean(),
      t1Heat: z.number().positive(),
      t2Heat: z.number().positive(),
      t3DemoteHeat: z.number().positive(),
      archiveAgeDays: z.number().positive(),
      archiveMinHeat: z.number().nonnegative(),
      episodeDecayIntervalMs: z.number().positive(),
      episodeSalienceHalfLifeDays: z.number().positive(),
      episodeHeatHalfLifeDays: z.number().positive(),
      traitHalfLifeDays: z.number().positive(),
      retrievalLogRetentionDays: z.number().positive(),
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
    beliefReviser: z.object({
      enabled: z.boolean(),
      confidenceDropMultiplier: z.number().min(0).max(1),
      confidenceFloor: z.number().min(0).max(1),
      regradeBatchSize: z.number().int().positive(),
      maxEventsPerRun: z.number().int().positive(),
      maxReviewsPerRun: z.number().int().positive(),
      claimStaleSec: z.number().positive(),
      maxParseFailures: z.number().int().positive(),
      maxLlmCalls: z.number().int().positive(),
      consecutiveParseFailureLimit: z.number().int().positive(),
    }),
  }),
  maintenance: z.object({
    enabled: z.boolean(),
    lightIntervalMs: z.number().int().positive(),
    heavyIntervalMs: z.number().int().positive(),
    lightProcesses: z.array(
      z.enum([
        "consolidator",
        "reflector",
        "curator",
        "overseer",
        "ruminator",
        "self-narrator",
        "procedural-synthesizer",
        "belief-reviser",
      ]),
    ),
    heavyProcesses: z.array(
      z.enum([
        "consolidator",
        "reflector",
        "curator",
        "overseer",
        "ruminator",
        "self-narrator",
        "procedural-synthesizer",
        "belief-reviser",
      ]),
    ),
  }),
  autonomy: z.object({
    enabled: z.boolean(),
    intervalMs: z.number().int().positive(),
    maxWakesPerWindow: z.number().int().positive(),
    budgetWindowMs: z.number().int().positive(),
    executiveFocus: z.object({
      enabled: z.boolean(),
      stalenessSec: z.number().int().positive(),
      dueLeadSec: z.number().int().nonnegative(),
      wakeCooldownSec: z.number().int().nonnegative(),
    }),
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
    // Affective perception uses the background model as the primary classifier
    // when configured; heuristics are the offline/test fallback path.
    useLlmFallback: true,
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
      // All slots default to Opus 4.7. borg runs under OAuth subscription
      // (not per-token billing), so cost optimization is not a concern. We
      // accept the latency hit on synchronous classifier calls in exchange
      // for consistent quality across extraction, reflection, and all
      // offline maintenance. The three slots still exist so individual
      // deployments CAN split them via config or env vars if they ever
      // need to.
      cognition: "claude-opus-4-7",
      background: "claude-opus-4-7",
      extraction: "claude-opus-4-7",
    },
  },
  procedural: {
    skillSelectionMinSimilarity: 0.5,
  },
  retrieval: {
    semantic: {
      underReviewMultiplier: 0.5,
    },
  },
  generation: {
    discourseStateHardCapTurns: 50,
  },
  streamIngestion: {
    preTurnCatchup: {
      maxEntries: 100,
    },
  },
  executive: {
    goalFocusThreshold: DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD,
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
      goalSimilarityThreshold: 0.82,
      ceilingConfidence: 0.5,
      maxInsightsPerRun: 2,
      budget: 60_000,
    },
    proceduralSynthesizer: {
      enabled: true,
      minSupport: 2,
      maxSkillsPerRun: 3,
      dedupThreshold: 0.88,
      minContextAttemptsForSplit: 5,
      minDivergenceForSplit: 0.3,
      splitCooldownDays: 7,
      splitClaimStaleSec: 1_800,
      maxSplitParseFailures: 3,
      // Legacy/deprecated: split proposals now always go through the review queue.
      skillSplitDryRun: true,
      budget: 4_000,
    },
    curator: {
      enabled: true,
      t1Heat: 5,
      t2Heat: 15,
      t3DemoteHeat: 3,
      archiveAgeDays: 45,
      archiveMinHeat: 1,
      episodeDecayIntervalMs: 24 * 60 * 60 * 1_000,
      episodeSalienceHalfLifeDays: 30,
      episodeHeatHalfLifeDays: 7,
      traitHalfLifeDays: 30,
      retrievalLogRetentionDays: 90,
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
      // Threshold applies to RetrievalConfidence.overall, a conservative
      // epistemic evidence-quality signal, not the relevance ranking score.
      resolveConfidenceThreshold: 0.55,
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
    beliefReviser: {
      enabled: true,
      confidenceDropMultiplier: 0.5,
      confidenceFloor: 0.05,
      regradeBatchSize: 10,
      maxEventsPerRun: 32,
      maxReviewsPerRun: 128,
      claimStaleSec: 600,
      maxParseFailures: 3,
      // Call-count cap for regrade LLM work; run `budget` remains token-based.
      maxLlmCalls: 20,
      consecutiveParseFailureLimit: 5,
    },
  },
  maintenance: {
    // Maintenance is core to the architecture (cold paths do real work --
    // semantic insight extraction, contradiction sweeps, decay/promotion,
    // belief revision). Default on so a fresh deployment actually runs the
    // dream cycle once a runtime (daemon, etc.) calls scheduler.start().
    enabled: true,
    lightIntervalMs: 14_400_000,
    heavyIntervalMs: 86_400_000,
    lightProcesses: ["consolidator", "curator"],
    heavyProcesses: [
      "reflector",
      "overseer",
      "ruminator",
      "self-narrator",
      "procedural-synthesizer",
      "belief-reviser",
    ],
  },
  autonomy: {
    // Self-initiated cognition is part of the architecture's "autonomous
    // being" framing. Default on so a fresh deployment exercises the
    // wake-source triggers (commitment expiring, open-question dormant,
    // goal follow-up due, executive-focus due) once a runtime (daemon, ...)
    // calls scheduler.start(). Library callers stay in control because
    // start() is still explicit. maxWakesPerWindow caps the cost.
    enabled: true,
    intervalMs: 60_000,
    maxWakesPerWindow: 6,
    budgetWindowMs: 86_400_000,
    executiveFocus: {
      // Default on alongside autonomy so a stale selected goal or due
      // executive step actually causes a self-initiated turn instead of
      // sitting silently until the next user message.
      enabled: true,
      stalenessSec: 86_400,
      dueLeadSec: 0,
      wakeCooldownSec: 3_600,
    },
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

  if (!Number.isFinite(value)) {
    throw new ConfigError(`Environment variable ${name} must be a finite number`);
  }

  return value;
}

function readOptionalEnvFloat(env: NodeJS.ProcessEnv, name: string): number | undefined {
  const raw = readOptionalEnvString(env, name);

  if (raw === undefined) {
    return undefined;
  }

  const value = Number(raw);

  if (!Number.isFinite(value)) {
    throw new ConfigError(`Environment variable ${name} must be a finite number`);
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
        fileConfig.perception?.modeWhenLlmAbsent ?? DEFAULT_CONFIG.perception.modeWhenLlmAbsent,
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
    procedural: {
      skillSelectionMinSimilarity:
        readOptionalEnvUnitInterval(env, "BORG_PROCEDURAL_SKILL_SELECTION_MIN_SIMILARITY") ??
        fileConfig.procedural?.skillSelectionMinSimilarity ??
        DEFAULT_CONFIG.procedural.skillSelectionMinSimilarity,
    },
    retrieval: {
      semantic: {
        underReviewMultiplier:
          readOptionalEnvUnitInterval(env, "BORG_RETRIEVAL_SEMANTIC_UNDER_REVIEW_MULTIPLIER") ??
          fileConfig.retrieval?.semantic?.underReviewMultiplier ??
          DEFAULT_CONFIG.retrieval.semantic.underReviewMultiplier,
      },
    },
    generation: {
      discourseStateHardCapTurns:
        readOptionalEnvNumber(env, "BORG_GENERATION_DISCOURSE_HARD_CAP_TURNS") ??
        fileConfig.generation?.discourseStateHardCapTurns ??
        DEFAULT_CONFIG.generation.discourseStateHardCapTurns,
    },
    streamIngestion: {
      preTurnCatchup: {
        maxEntries:
          readOptionalEnvNumber(env, "BORG_STREAM_INGESTION_PRE_TURN_CATCHUP_MAX_ENTRIES") ??
          fileConfig.streamIngestion?.preTurnCatchup?.maxEntries ??
          DEFAULT_CONFIG.streamIngestion.preTurnCatchup.maxEntries,
      },
    },
    executive: {
      goalFocusThreshold:
        readOptionalEnvUnitInterval(env, "BORG_EXECUTIVE_GOAL_FOCUS_THRESHOLD") ??
        fileConfig.executive?.goalFocusThreshold ??
        DEFAULT_CONFIG.executive.goalFocusThreshold,
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
        goalSimilarityThreshold:
          readOptionalEnvFloat(env, "BORG_OFFLINE_REFLECTOR_GOAL_SIMILARITY_THRESHOLD") ??
          fileConfig.offline?.reflector?.goalSimilarityThreshold ??
          DEFAULT_CONFIG.offline.reflector.goalSimilarityThreshold,
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
      proceduralSynthesizer: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_ENABLED") ??
          fileConfig.offline?.proceduralSynthesizer?.enabled ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.enabled,
        minSupport:
          readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MIN_SUPPORT") ??
          fileConfig.offline?.proceduralSynthesizer?.minSupport ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.minSupport,
        maxSkillsPerRun:
          readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MAX_SKILLS_PER_RUN") ??
          fileConfig.offline?.proceduralSynthesizer?.maxSkillsPerRun ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.maxSkillsPerRun,
        dedupThreshold:
          readOptionalEnvUnitInterval(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_DEDUP_THRESHOLD") ??
          fileConfig.offline?.proceduralSynthesizer?.dedupThreshold ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.dedupThreshold,
        minContextAttemptsForSplit:
          readOptionalEnvNumber(
            env,
            "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MIN_CONTEXT_ATTEMPTS_FOR_SPLIT",
          ) ??
          fileConfig.offline?.proceduralSynthesizer?.minContextAttemptsForSplit ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.minContextAttemptsForSplit,
        minDivergenceForSplit:
          readOptionalEnvUnitInterval(
            env,
            "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MIN_DIVERGENCE_FOR_SPLIT",
          ) ??
          fileConfig.offline?.proceduralSynthesizer?.minDivergenceForSplit ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.minDivergenceForSplit,
        splitCooldownDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_SPLIT_COOLDOWN_DAYS") ??
          fileConfig.offline?.proceduralSynthesizer?.splitCooldownDays ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.splitCooldownDays,
        splitClaimStaleSec:
          readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_SPLIT_CLAIM_STALE_SEC") ??
          fileConfig.offline?.proceduralSynthesizer?.splitClaimStaleSec ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.splitClaimStaleSec,
        maxSplitParseFailures:
          readOptionalEnvNumber(
            env,
            "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MAX_SPLIT_PARSE_FAILURES",
          ) ??
          fileConfig.offline?.proceduralSynthesizer?.maxSplitParseFailures ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.maxSplitParseFailures,
        skillSplitDryRun:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_SKILL_SPLIT_DRY_RUN") ??
          fileConfig.offline?.proceduralSynthesizer?.skillSplitDryRun ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.skillSplitDryRun,
        budget:
          readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_BUDGET") ??
          fileConfig.offline?.proceduralSynthesizer?.budget ??
          DEFAULT_CONFIG.offline.proceduralSynthesizer.budget,
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
        episodeDecayIntervalMs:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_EPISODE_DECAY_INTERVAL_MS") ??
          fileConfig.offline?.curator?.episodeDecayIntervalMs ??
          DEFAULT_CONFIG.offline.curator.episodeDecayIntervalMs,
        episodeSalienceHalfLifeDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_EPISODE_SALIENCE_HALF_LIFE_DAYS") ??
          fileConfig.offline?.curator?.episodeSalienceHalfLifeDays ??
          DEFAULT_CONFIG.offline.curator.episodeSalienceHalfLifeDays,
        episodeHeatHalfLifeDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_EPISODE_HEAT_HALF_LIFE_DAYS") ??
          fileConfig.offline?.curator?.episodeHeatHalfLifeDays ??
          DEFAULT_CONFIG.offline.curator.episodeHeatHalfLifeDays,
        traitHalfLifeDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_TRAIT_HALF_LIFE_DAYS") ??
          fileConfig.offline?.curator?.traitHalfLifeDays ??
          DEFAULT_CONFIG.offline.curator.traitHalfLifeDays,
        retrievalLogRetentionDays:
          readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_RETRIEVAL_LOG_RETENTION_DAYS") ??
          fileConfig.offline?.curator?.retrievalLogRetentionDays ??
          DEFAULT_CONFIG.offline.curator.retrievalLogRetentionDays,
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
      beliefReviser: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_OFFLINE_BELIEF_REVISER_ENABLED") ??
          fileConfig.offline?.beliefReviser?.enabled ??
          DEFAULT_CONFIG.offline.beliefReviser.enabled,
        confidenceDropMultiplier:
          readOptionalEnvUnitInterval(
            env,
            "BORG_OFFLINE_BELIEF_REVISER_CONFIDENCE_DROP_MULTIPLIER",
          ) ??
          fileConfig.offline?.beliefReviser?.confidenceDropMultiplier ??
          DEFAULT_CONFIG.offline.beliefReviser.confidenceDropMultiplier,
        confidenceFloor:
          readOptionalEnvUnitInterval(env, "BORG_OFFLINE_BELIEF_REVISER_CONFIDENCE_FLOOR") ??
          fileConfig.offline?.beliefReviser?.confidenceFloor ??
          DEFAULT_CONFIG.offline.beliefReviser.confidenceFloor,
        regradeBatchSize:
          readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_REGRADE_BATCH_SIZE") ??
          fileConfig.offline?.beliefReviser?.regradeBatchSize ??
          DEFAULT_CONFIG.offline.beliefReviser.regradeBatchSize,
        maxEventsPerRun:
          readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_MAX_EVENTS_PER_RUN") ??
          fileConfig.offline?.beliefReviser?.maxEventsPerRun ??
          DEFAULT_CONFIG.offline.beliefReviser.maxEventsPerRun,
        maxReviewsPerRun:
          readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_MAX_REVIEWS_PER_RUN") ??
          fileConfig.offline?.beliefReviser?.maxReviewsPerRun ??
          DEFAULT_CONFIG.offline.beliefReviser.maxReviewsPerRun,
        claimStaleSec:
          readOptionalEnvFloat(env, "BORG_OFFLINE_BELIEF_REVISER_CLAIM_STALE_SEC") ??
          fileConfig.offline?.beliefReviser?.claimStaleSec ??
          DEFAULT_CONFIG.offline.beliefReviser.claimStaleSec,
        maxParseFailures:
          readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_MAX_PARSE_FAILURES") ??
          fileConfig.offline?.beliefReviser?.maxParseFailures ??
          DEFAULT_CONFIG.offline.beliefReviser.maxParseFailures,
        maxLlmCalls:
          readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_MAX_LLM_CALLS") ??
          fileConfig.offline?.beliefReviser?.maxLlmCalls ??
          DEFAULT_CONFIG.offline.beliefReviser.maxLlmCalls,
        consecutiveParseFailureLimit:
          readOptionalEnvNumber(
            env,
            "BORG_OFFLINE_BELIEF_REVISER_CONSECUTIVE_PARSE_FAILURE_LIMIT",
          ) ??
          fileConfig.offline?.beliefReviser?.consecutiveParseFailureLimit ??
          DEFAULT_CONFIG.offline.beliefReviser.consecutiveParseFailureLimit,
      },
    },
    maintenance: {
      enabled:
        readOptionalEnvBoolean(env, "BORG_MAINTENANCE_ENABLED") ??
        fileConfig.maintenance?.enabled ??
        DEFAULT_CONFIG.maintenance.enabled,
      lightIntervalMs:
        readOptionalEnvNumber(env, "BORG_MAINTENANCE_LIGHT_INTERVAL_MS") ??
        fileConfig.maintenance?.lightIntervalMs ??
        DEFAULT_CONFIG.maintenance.lightIntervalMs,
      heavyIntervalMs:
        readOptionalEnvNumber(env, "BORG_MAINTENANCE_HEAVY_INTERVAL_MS") ??
        fileConfig.maintenance?.heavyIntervalMs ??
        DEFAULT_CONFIG.maintenance.heavyIntervalMs,
      lightProcesses:
        fileConfig.maintenance?.lightProcesses ?? DEFAULT_CONFIG.maintenance.lightProcesses,
      heavyProcesses:
        fileConfig.maintenance?.heavyProcesses ?? DEFAULT_CONFIG.maintenance.heavyProcesses,
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
      maxWakesPerWindow:
        readOptionalEnvNumber(env, "BORG_AUTONOMY_MAX_WAKES_PER_WINDOW") ??
        fileConfig.autonomy?.maxWakesPerWindow ??
        DEFAULT_CONFIG.autonomy.maxWakesPerWindow,
      budgetWindowMs:
        readOptionalEnvNumber(env, "BORG_AUTONOMY_BUDGET_WINDOW_MS") ??
        fileConfig.autonomy?.budgetWindowMs ??
        DEFAULT_CONFIG.autonomy.budgetWindowMs,
      executiveFocus: {
        enabled:
          readOptionalEnvBoolean(env, "BORG_AUTONOMY_EXECUTIVE_FOCUS_ENABLED") ??
          fileConfig.autonomy?.executiveFocus?.enabled ??
          DEFAULT_CONFIG.autonomy.executiveFocus.enabled,
        stalenessSec:
          readOptionalEnvNumber(env, "BORG_AUTONOMY_EXECUTIVE_FOCUS_STALENESS_SEC") ??
          fileConfig.autonomy?.executiveFocus?.stalenessSec ??
          DEFAULT_CONFIG.autonomy.executiveFocus.stalenessSec,
        dueLeadSec:
          readOptionalEnvNumber(env, "BORG_AUTONOMY_EXECUTIVE_FOCUS_DUE_LEAD_SEC") ??
          fileConfig.autonomy?.executiveFocus?.dueLeadSec ??
          DEFAULT_CONFIG.autonomy.executiveFocus.dueLeadSec,
        wakeCooldownSec:
          readOptionalEnvNumber(env, "BORG_AUTONOMY_EXECUTIVE_FOCUS_WAKE_COOLDOWN_SEC") ??
          fileConfig.autonomy?.executiveFocus?.wakeCooldownSec ??
          DEFAULT_CONFIG.autonomy.executiveFocus.wakeCooldownSec,
      },
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
    procedural: {
      ...config.procedural,
    },
    retrieval: {
      semantic: {
        ...config.retrieval.semantic,
      },
    },
    streamIngestion: {
      preTurnCatchup: {
        ...config.streamIngestion.preTurnCatchup,
      },
    },
    executive: {
      ...config.executive,
    },
    offline: {
      ...config.offline,
    },
    maintenance: {
      ...config.maintenance,
      lightProcesses: [...config.maintenance.lightProcesses],
      heavyProcesses: [...config.maintenance.heavyProcesses],
    },
    autonomy: {
      ...config.autonomy,
      executiveFocus: {
        ...config.autonomy.executiveFocus,
      },
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
