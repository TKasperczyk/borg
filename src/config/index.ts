import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { z } from "zod";

import { DEFAULT_HOST_CAPABILITIES_SECTION } from "../cognition/deliberation/constants.js";
import {
  RELATIONAL_CLAIM_KINDS,
  type RelationalClaimKind,
} from "../cognition/generation/relational-claim-kinds.js";
import { DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD } from "../executive/index.js";
import { readJsonFile } from "../util/atomic-write.js";
import { ConfigError } from "../util/errors.js";

const DEFAULT_DATA_DIR = "~/.borg";

export function expandPath(pathLike: string): string {
  if (pathLike === "~") {
    return homedir();
  }

  if (pathLike.startsWith("~/")) {
    return join(homedir(), pathLike.slice(2));
  }

  return isAbsolute(pathLike) ? pathLike : resolve(pathLike);
}

const anthropicAuthModeSchema = z.enum(["auto", "oauth", "api-key"]);
export const postGenerationGuardModeSchema = z.enum(["enforce", "shadow"]);
const postGenerationGuardConfigSchema = z
  .object({
    mode: postGenerationGuardModeSchema.default("enforce"),
  })
  .prefault({});
export const relationalClaimGuardModeSchema = z.union([
  postGenerationGuardModeSchema,
  z
    .object({
      perCategory: z
        .object({
          default: postGenerationGuardModeSchema,
          overrides: z
            .partialRecord(z.enum(RELATIONAL_CLAIM_KINDS), postGenerationGuardModeSchema)
            .optional(),
        })
        .strict(),
    })
    .strict(),
]);
const relationalClaimGuardConfigSchema = z
  .object({
    mode: relationalClaimGuardModeSchema.default("enforce"),
  })
  .prefault({});
const evidenceLedgerConfigSchema = z
  .object({
    enabled: z.boolean().default(false),
    currentSessionTranscriptTokenBudget: z.number().int().positive().default(2_500),
    actionThreadRenderLimit: z.number().int().positive().default(12),
    actionThreadSimilarityThreshold: z.number().min(0).max(1).default(0.85),
    actionThreadSourceRecordLimit: z.number().int().positive().default(256),
  })
  .prefault({});
const cognitionThinkingConfigSchema = z
  .object({
    enabled: z.boolean().default(false),
    budget_tokens: z.number().int().positive().default(4096),
  })
  .prefault({});
const cognitionConfigSchema = z
  .object({
    thinking: cognitionThinkingConfigSchema,
  })
  .prefault({});
const manifestFinalizerConfigSchema = z
  .object({
    enabled: z.boolean().default(false),
  })
  .prefault({});
const postGenerationGuardsConfigSchema = z
  .object({
    commitment: postGenerationGuardConfigSchema,
    relationalClaim: relationalClaimGuardConfigSchema,
    closurePressure: postGenerationGuardConfigSchema,
  })
  .prefault({});
const maintenanceProcessSchema = z.enum([
  "consolidator",
  "reflector",
  "semantic-extractor",
  "curator",
  "overseer",
  "ruminator",
  "self-narrator",
  "procedural-synthesizer",
  "belief-reviser",
]);

export type PostGenerationGuardMode = z.infer<typeof postGenerationGuardModeSchema>;
export type RelationalClaimGuardMode =
  | PostGenerationGuardMode
  | {
      perCategory: {
        default: PostGenerationGuardMode;
        overrides?: Partial<Record<RelationalClaimKind, PostGenerationGuardMode>>;
      };
    };
const anthropicModelsConfigSchema = z
  .object({
    // The main cognition/extraction/background slots default to Opus 4.7.
    // Recall expansion is a small structured fanout task and has its own
    // Haiku slot so it can stay fast without reusing background.
    cognition: z.string().min(1).default("claude-opus-4-7"),
    background: z.string().min(1).default("claude-opus-4-7"),
    extraction: z.string().min(1).default("claude-opus-4-7"),
    recallExpansion: z.string().min(1).default("claude-haiku-4-5-20251001"),
  })
  .prefault({});

const anthropicConfigSchema = z
  .object({
    auth: anthropicAuthModeSchema.default("auto"),
    apiKey: z.string().min(1).optional(),
    models: anthropicModelsConfigSchema,
  })
  .prefault({});

const configBaseSchema = z.object({
  dataDir: z.string().min(1).default(DEFAULT_DATA_DIR).transform(expandPath),
  defaultUser: z.string().min(1).optional(),
  host_capabilities: z.string().min(1).default(DEFAULT_HOST_CAPABILITIES_SECTION),
  perception: z
    .object({
      useLlmFallback: z.boolean().default(true),
      modeWhenLlmAbsent: z.enum(["problem_solving", "relational", "reflective", "idle"]).optional(),
    })
    .prefault({}),
  affective: z
    .object({
      // Affective perception uses the background model as the primary classifier
      // when configured; heuristics are the offline/test fallback path.
      useLlmFallback: z.boolean().default(true),
      incomingMoodWeight: z.number().min(0).max(1).default(0.3),
      moodHistoryRetentionDays: z.number().positive().default(90),
      moodHalfLifeHours: z.number().positive().default(24),
    })
    .prefault({}),
  embedding: z
    .object({
      baseUrl: z.string().url().default("http://localhost:1234/v1"),
      apiKey: z.string().min(1).default("lm-studio"),
      model: z.string().min(1).default("text-embedding-qwen3-embedding-8b"),
      dims: z.number().int().positive().default(4096),
    })
    .prefault({}),
  anthropic: anthropicConfigSchema,
  procedural: z
    .object({
      skillSelectionMinSimilarity: z.number().min(0).max(1).default(0.5),
    })
    .prefault({}),
  retrieval: z
    .object({
      semantic: z
        .object({
          underReviewMultiplier: z.number().min(0).max(1).default(0.5),
        })
        .prefault({}),
    })
    .prefault({}),
  generation: z
    .object({
      discourseStateHardCapTurns: z.number().int().positive().default(50),
      cognition: cognitionConfigSchema,
      evidenceLedger: evidenceLedgerConfigSchema,
      manifestFinalizer: manifestFinalizerConfigSchema,
      postGenerationGuards: postGenerationGuardsConfigSchema,
    })
    .prefault({}),
  streamIngestion: z
    .object({
      preTurnCatchup: z
        .object({
          maxEntries: z.number().int().positive().default(100),
        })
        .prefault({}),
    })
    .prefault({}),
  executive: z
    .object({
      goalFocusThreshold: z.number().min(0).max(1).default(DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD),
    })
    .prefault({}),
  offline: z
    .object({
      consolidator: z
        .object({
          enabled: z.boolean().default(true),
          similarityThreshold: z.number().positive().default(0.82),
          minClusterSize: z.number().int().positive().default(2),
          maxClustersPerRun: z.number().int().positive().default(2),
          budget: z.number().int().positive().default(60_000),
        })
        .prefault({}),
      reflector: z
        .object({
          enabled: z.boolean().default(true),
          minSupport: z.number().int().positive().default(3),
          goalSimilarityThreshold: z.number().min(0).max(1).default(0.82),
          ceilingConfidence: z.number().positive().max(0.5).default(0.5),
          maxInsightsPerRun: z.number().int().positive().default(2),
          budget: z.number().int().positive().default(60_000),
        })
        .prefault({}),
      semanticExtractor: z
        .object({
          enabled: z.boolean().default(true),
          maxEpisodesPerRun: z.number().int().positive().default(8),
          budget: z.number().int().positive().default(60_000),
        })
        .prefault({}),
      proceduralSynthesizer: z
        .object({
          enabled: z.boolean().default(true),
          minSupport: z.number().int().positive().default(2),
          maxSkillsPerRun: z.number().int().positive().default(3),
          dedupThreshold: z.number().min(0).max(1).default(0.88),
          minContextAttemptsForSplit: z.number().int().positive().default(5),
          minDivergenceForSplit: z.number().min(0).max(1).default(0.3),
          splitCooldownDays: z.number().positive().default(7),
          splitClaimStaleSec: z.number().int().positive().default(1_800),
          maxSplitParseFailures: z.number().int().positive().default(3),
          budget: z.number().int().positive().default(4_000),
        })
        .prefault({}),
      curator: z
        .object({
          enabled: z.boolean().default(true),
          t1Heat: z.number().positive().default(5),
          t2Heat: z.number().positive().default(15),
          t3DemoteHeat: z.number().positive().default(3),
          archiveAgeDays: z.number().positive().default(45),
          archiveMinHeat: z.number().nonnegative().default(1),
          episodeDecayIntervalMs: z
            .number()
            .positive()
            .default(24 * 60 * 60 * 1_000),
          episodeSalienceHalfLifeDays: z.number().positive().default(30),
          episodeHeatHalfLifeDays: z.number().positive().default(7),
          traitHalfLifeDays: z.number().positive().default(30),
          retrievalLogRetentionDays: z.number().positive().default(90),
        })
        .prefault({}),
      overseer: z
        .object({
          enabled: z.boolean().default(true),
          lookbackHours: z.number().positive().default(24),
          maxChecksPerRun: z.number().int().positive().default(8),
          budget: z.number().int().positive().nullable().default(null),
        })
        .prefault({}),
      ruminator: z
        .object({
          enabled: z.boolean().default(true),
          maxQuestionsPerRun: z.number().int().positive().default(8),
          // Threshold applies to RetrievalConfidence.overall, a conservative
          // epistemic evidence-quality signal, not the relevance ranking score.
          resolveConfidenceThreshold: z.number().min(0).max(1).default(0.55),
          duplicateSimilarityThreshold: z.number().min(0).max(1).default(0.9),
          stalenessDays: z.number().positive().default(30),
          stalenessTicks: z.number().int().positive().nullable().default(null),
          staleNoTractionTicks: z.number().int().positive().default(4),
          budget: z.number().int().positive().default(40_000),
          perQuestionBudget: z.number().int().positive().default(8_000),
        })
        .prefault({}),
      selfNarrator: z
        .object({
          enabled: z.boolean().default(true),
          budget: z.number().int().positive().default(80_000),
          maxObservationsPerRun: z.number().int().positive().default(4),
          minSupportEpisodes: z.number().int().positive().default(2),
          cadenceHintDays: z.number().positive().default(7),
        })
        .prefault({}),
      beliefReviser: z
        .object({
          enabled: z.boolean().default(true),
          confidenceDropMultiplier: z.number().min(0).max(1).default(0.5),
          confidenceFloor: z.number().min(0).max(1).default(0.05),
          regradeBatchSize: z.number().int().positive().default(10),
          maxEventsPerRun: z.number().int().positive().default(32),
          maxReviewsPerRun: z.number().int().positive().default(128),
          claimStaleSec: z.number().positive().default(600),
          maxParseFailures: z.number().int().positive().default(3),
          // Call-count cap for regrade LLM work; run `budget` remains token-based.
          maxLlmCalls: z.number().int().positive().default(20),
          consecutiveParseFailureLimit: z.number().int().positive().default(5),
        })
        .prefault({}),
    })
    .prefault({}),
  maintenance: z
    .object({
      // Maintenance is core to the architecture (cold paths do real work --
      // semantic insight extraction, contradiction sweeps, decay/promotion,
      // belief revision). Default on so a fresh deployment actually runs the
      // dream cycle once a runtime (daemon, etc.) calls scheduler.start().
      enabled: z.boolean().default(true),
      lightIntervalMs: z.number().int().positive().default(14_400_000),
      heavyIntervalMs: z.number().int().positive().default(86_400_000),
      lightProcesses: z
        .array(maintenanceProcessSchema)
        .default(["consolidator", "semantic-extractor", "curator"]),
      heavyProcesses: z
        .array(maintenanceProcessSchema)
        .default([
          "reflector",
          "overseer",
          "ruminator",
          "self-narrator",
          "procedural-synthesizer",
          "belief-reviser",
        ]),
    })
    .prefault({}),
  autonomy: z
    .object({
      // Self-initiated cognition is part of the architecture's "autonomous
      // being" framing. Default on so a fresh deployment exercises the
      // wake-source triggers (commitment expiring, open-question dormant,
      // goal follow-up due, executive-focus due) once a runtime (daemon, ...)
      // calls scheduler.start(). Library callers stay in control because
      // start() is still explicit. maxWakesPerWindow caps the cost.
      enabled: z.boolean().default(true),
      intervalMs: z.number().int().positive().default(60_000),
      maxWakesPerWindow: z.number().int().positive().default(6),
      budgetWindowMs: z.number().int().positive().default(86_400_000),
      executiveFocus: z
        .object({
          // Default on alongside autonomy so a stale selected goal or due
          // executive step actually causes a self-initiated turn instead of
          // sitting silently until the next user message.
          enabled: z.boolean().default(true),
          stalenessSec: z.number().int().positive().default(86_400),
          dueLeadSec: z.number().int().nonnegative().default(0),
          wakeCooldownSec: z.number().int().nonnegative().default(3_600),
        })
        .prefault({}),
      triggers: z
        .object({
          commitmentExpiring: z
            .object({
              enabled: z.boolean().default(true),
              lookaheadMs: z.number().int().positive().default(86_400_000),
            })
            .prefault({}),
          openQuestionDormant: z
            .object({
              enabled: z.boolean().default(true),
              dormantMs: z.number().int().positive().default(604_800_000),
            })
            .prefault({}),
          scheduledReflection: z
            .object({
              enabled: z.boolean().default(false),
              intervalMs: z.number().int().positive().default(14_400_000),
            })
            .prefault({}),
          goalFollowupDue: z
            .object({
              enabled: z.boolean().default(true),
              lookaheadMs: z.number().int().positive().default(604_800_000),
              staleMs: z.number().int().positive().default(1_209_600_000),
            })
            .prefault({}),
        })
        .prefault({}),
      conditions: z
        .object({
          commitmentRevoked: z
            .object({
              enabled: z.boolean().default(true),
            })
            .prefault({}),
          moodValenceDrop: z
            .object({
              enabled: z.boolean().default(false),
              threshold: z.number().min(-1).max(1).default(-0.5),
              windowN: z.number().int().positive().default(5),
              activationPeriodMs: z.number().int().positive().default(86_400_000),
            })
            .prefault({}),
          openQuestionUrgencyBump: z
            .object({
              enabled: z.boolean().default(true),
              threshold: z.number().min(0).max(1).default(0.9),
            })
            .prefault({}),
        })
        .prefault({}),
    })
    .prefault({}),
});

const configOutputSchema = configBaseSchema.transform(
  (config): z.output<typeof configBaseSchema> => ({
    defaultUser: undefined,
    ...config,
    anthropic: {
      apiKey: undefined,
      ...config.anthropic,
    },
  }),
);

export const configSchema = configOutputSchema.superRefine((value, context) => {
  if (value.anthropic.auth === "api-key" && value.anthropic.apiKey === undefined) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      message: "Anthropic API key must be configured when anthropic.auth is api-key",
      path: ["anthropic", "apiKey"],
    });
  }
});

export type Config = z.infer<typeof configSchema>;
type ConfigInput = z.input<typeof configSchema>;
type DeepPartial<T> = T extends readonly unknown[]
  ? T
  : T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;
type ConfigOverrides = DeepPartial<ConfigInput>;

export const DEFAULT_CONFIG: Config = configSchema.parse({});

export type LoadConfigOptions = {
  dataDir?: string;
  env?: NodeJS.ProcessEnv;
};

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

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function mergeConfigOverrides(base: ConfigOverrides, override: ConfigOverrides): ConfigOverrides {
  const merged: Record<string, unknown> = { ...(base as Record<string, unknown>) };

  for (const [key, value] of Object.entries(override as Record<string, unknown>)) {
    if (value === undefined) {
      continue;
    }

    const existing = merged[key];
    merged[key] =
      isPlainRecord(existing) && isPlainRecord(value)
        ? mergeConfigOverrides(existing as ConfigOverrides, value as ConfigOverrides)
        : value;
  }

  return merged as ConfigOverrides;
}

function setConfigOverride(
  overrides: ConfigOverrides,
  path: readonly [string, ...string[]],
  value: unknown,
): void {
  if (value === undefined) {
    return;
  }

  let cursor = overrides as Record<string, unknown>;

  for (let index = 0; index < path.length - 1; index += 1) {
    const key = path[index] as string;
    const existing = cursor[key];

    if (isPlainRecord(existing)) {
      cursor = existing;
      continue;
    }

    const next: Record<string, unknown> = {};
    cursor[key] = next;
    cursor = next;
  }

  cursor[path[path.length - 1] as string] = value;
}

function loadEnvOverrides(env: NodeJS.ProcessEnv): ConfigOverrides {
  const overrides: ConfigOverrides = {};

  setConfigOverride(overrides, ["dataDir"], readOptionalEnvString(env, "BORG_DATA_DIR"));
  setConfigOverride(overrides, ["defaultUser"], readOptionalEnvString(env, "BORG_DEFAULT_USER"));
  setConfigOverride(
    overrides,
    ["host_capabilities"],
    readOptionalEnvString(env, "BORG_HOST_CAPABILITIES"),
  );
  setConfigOverride(
    overrides,
    ["perception", "useLlmFallback"],
    readOptionalEnvBoolean(env, "BORG_PERCEPTION_USE_LLM_FALLBACK"),
  );
  setConfigOverride(
    overrides,
    ["affective", "useLlmFallback"],
    readOptionalEnvBoolean(env, "BORG_AFFECTIVE_USE_LLM_FALLBACK"),
  );
  setConfigOverride(
    overrides,
    ["affective", "incomingMoodWeight"],
    readOptionalEnvUnitInterval(env, "BORG_AFFECTIVE_INCOMING_MOOD_WEIGHT"),
  );
  setConfigOverride(
    overrides,
    ["affective", "moodHistoryRetentionDays"],
    readOptionalEnvFloat(env, "BORG_AFFECTIVE_MOOD_HISTORY_RETENTION_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["affective", "moodHalfLifeHours"],
    readOptionalEnvFloat(env, "BORG_AFFECTIVE_MOOD_HALF_LIFE_HOURS"),
  );
  setConfigOverride(
    overrides,
    ["embedding", "baseUrl"],
    readOptionalEnvString(env, "BORG_EMBEDDING_BASE_URL"),
  );
  setConfigOverride(
    overrides,
    ["embedding", "apiKey"],
    readOptionalEnvString(env, "BORG_EMBEDDING_API_KEY"),
  );
  setConfigOverride(
    overrides,
    ["embedding", "model"],
    readOptionalEnvString(env, "BORG_EMBEDDING_MODEL"),
  );
  setConfigOverride(
    overrides,
    ["embedding", "dims"],
    readOptionalEnvNumber(env, "BORG_EMBEDDING_DIMS"),
  );
  setConfigOverride(
    overrides,
    ["anthropic", "auth"],
    readOptionalEnvAnthropicAuthMode(env, "BORG_ANTHROPIC_AUTH"),
  );
  setConfigOverride(
    overrides,
    ["anthropic", "apiKey"],
    readOptionalEnvString(env, "ANTHROPIC_API_KEY"),
  );
  setConfigOverride(
    overrides,
    ["anthropic", "models", "cognition"],
    readOptionalEnvString(env, "BORG_MODEL_COGNITION"),
  );
  setConfigOverride(
    overrides,
    ["anthropic", "models", "background"],
    readOptionalEnvString(env, "BORG_MODEL_BACKGROUND"),
  );
  setConfigOverride(
    overrides,
    ["anthropic", "models", "extraction"],
    readOptionalEnvString(env, "BORG_MODEL_EXTRACTION"),
  );
  setConfigOverride(
    overrides,
    ["anthropic", "models", "recallExpansion"],
    readOptionalEnvString(env, "BORG_MODEL_RECALL_EXPANSION"),
  );
  setConfigOverride(
    overrides,
    ["procedural", "skillSelectionMinSimilarity"],
    readOptionalEnvUnitInterval(env, "BORG_PROCEDURAL_SKILL_SELECTION_MIN_SIMILARITY"),
  );
  setConfigOverride(
    overrides,
    ["retrieval", "semantic", "underReviewMultiplier"],
    readOptionalEnvUnitInterval(env, "BORG_RETRIEVAL_SEMANTIC_UNDER_REVIEW_MULTIPLIER"),
  );
  setConfigOverride(
    overrides,
    ["generation", "discourseStateHardCapTurns"],
    readOptionalEnvNumber(env, "BORG_GENERATION_DISCOURSE_HARD_CAP_TURNS"),
  );
  setConfigOverride(
    overrides,
    ["generation", "cognition", "thinking", "enabled"],
    readOptionalEnvBoolean(env, "BORG_GENERATION_COGNITION_THINKING_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["generation", "cognition", "thinking", "budget_tokens"],
    readOptionalEnvNumber(env, "BORG_GENERATION_COGNITION_THINKING_BUDGET_TOKENS"),
  );
  setConfigOverride(
    overrides,
    ["generation", "evidenceLedger", "enabled"],
    readOptionalEnvBoolean(env, "BORG_GENERATION_EVIDENCE_LEDGER_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["generation", "evidenceLedger", "currentSessionTranscriptTokenBudget"],
    readOptionalEnvNumber(
      env,
      "BORG_GENERATION_EVIDENCE_LEDGER_CURRENT_SESSION_TRANSCRIPT_TOKEN_BUDGET",
    ),
  );
  setConfigOverride(
    overrides,
    ["generation", "evidenceLedger", "actionThreadRenderLimit"],
    readOptionalEnvNumber(env, "BORG_GENERATION_EVIDENCE_LEDGER_ACTION_THREAD_RENDER_LIMIT"),
  );
  setConfigOverride(
    overrides,
    ["generation", "evidenceLedger", "actionThreadSimilarityThreshold"],
    readOptionalEnvUnitInterval(
      env,
      "BORG_GENERATION_EVIDENCE_LEDGER_ACTION_THREAD_SIMILARITY_THRESHOLD",
    ),
  );
  setConfigOverride(
    overrides,
    ["generation", "evidenceLedger", "actionThreadSourceRecordLimit"],
    readOptionalEnvNumber(env, "BORG_GENERATION_EVIDENCE_LEDGER_ACTION_THREAD_SOURCE_RECORD_LIMIT"),
  );
  setConfigOverride(
    overrides,
    ["generation", "manifestFinalizer", "enabled"],
    readOptionalEnvBoolean(env, "BORG_GENERATION_MANIFEST_FINALIZER_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["streamIngestion", "preTurnCatchup", "maxEntries"],
    readOptionalEnvNumber(env, "BORG_STREAM_INGESTION_PRE_TURN_CATCHUP_MAX_ENTRIES"),
  );
  setConfigOverride(
    overrides,
    ["executive", "goalFocusThreshold"],
    readOptionalEnvUnitInterval(env, "BORG_EXECUTIVE_GOAL_FOCUS_THRESHOLD"),
  );
  setConfigOverride(
    overrides,
    ["offline", "consolidator", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_CONSOLIDATOR_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "consolidator", "similarityThreshold"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CONSOLIDATOR_SIMILARITY_THRESHOLD"),
  );
  setConfigOverride(
    overrides,
    ["offline", "consolidator", "minClusterSize"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_CONSOLIDATOR_MIN_CLUSTER_SIZE"),
  );
  setConfigOverride(
    overrides,
    ["offline", "consolidator", "maxClustersPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_CONSOLIDATOR_MAX_CLUSTERS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "consolidator", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_CONSOLIDATOR_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "reflector", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_REFLECTOR_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "reflector", "minSupport"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_REFLECTOR_MIN_SUPPORT"),
  );
  setConfigOverride(
    overrides,
    ["offline", "reflector", "goalSimilarityThreshold"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_REFLECTOR_GOAL_SIMILARITY_THRESHOLD"),
  );
  setConfigOverride(
    overrides,
    ["offline", "reflector", "ceilingConfidence"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_REFLECTOR_CEILING_CONFIDENCE"),
  );
  setConfigOverride(
    overrides,
    ["offline", "reflector", "maxInsightsPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_REFLECTOR_MAX_INSIGHTS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "reflector", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_REFLECTOR_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "semanticExtractor", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_SEMANTIC_EXTRACTOR_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "semanticExtractor", "maxEpisodesPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_SEMANTIC_EXTRACTOR_MAX_EPISODES_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "semanticExtractor", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_SEMANTIC_EXTRACTOR_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "minSupport"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MIN_SUPPORT"),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "maxSkillsPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MAX_SKILLS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "dedupThreshold"],
    readOptionalEnvUnitInterval(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_DEDUP_THRESHOLD"),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "minContextAttemptsForSplit"],
    readOptionalEnvNumber(
      env,
      "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MIN_CONTEXT_ATTEMPTS_FOR_SPLIT",
    ),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "minDivergenceForSplit"],
    readOptionalEnvUnitInterval(
      env,
      "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MIN_DIVERGENCE_FOR_SPLIT",
    ),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "splitCooldownDays"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_SPLIT_COOLDOWN_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "splitClaimStaleSec"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_SPLIT_CLAIM_STALE_SEC"),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "maxSplitParseFailures"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_MAX_SPLIT_PARSE_FAILURES"),
  );
  setConfigOverride(
    overrides,
    ["offline", "proceduralSynthesizer", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_PROCEDURAL_SYNTHESIZER_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_CURATOR_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "t1Heat"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_T1_HEAT"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "t2Heat"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_T2_HEAT"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "t3DemoteHeat"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_T3_DEMOTE_HEAT"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "archiveAgeDays"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_ARCHIVE_AGE_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "archiveMinHeat"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_ARCHIVE_MIN_HEAT"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "episodeDecayIntervalMs"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_EPISODE_DECAY_INTERVAL_MS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "episodeSalienceHalfLifeDays"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_EPISODE_SALIENCE_HALF_LIFE_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "episodeHeatHalfLifeDays"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_EPISODE_HEAT_HALF_LIFE_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "traitHalfLifeDays"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_TRAIT_HALF_LIFE_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "curator", "retrievalLogRetentionDays"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_CURATOR_RETRIEVAL_LOG_RETENTION_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "overseer", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_OVERSEER_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "overseer", "lookbackHours"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_OVERSEER_LOOKBACK_HOURS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "overseer", "maxChecksPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_OVERSEER_MAX_CHECKS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "overseer", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_OVERSEER_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "ruminator", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_RUMINATOR_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "ruminator", "maxQuestionsPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_RUMINATOR_MAX_QUESTIONS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "ruminator", "resolveConfidenceThreshold"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_RUMINATOR_RESOLVE_CONFIDENCE_THRESHOLD"),
  );
  setConfigOverride(
    overrides,
    ["offline", "ruminator", "stalenessDays"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_RUMINATOR_STALENESS_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "ruminator", "stalenessTicks"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_RUMINATOR_STALENESS_TICKS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "ruminator", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_RUMINATOR_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "ruminator", "perQuestionBudget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_RUMINATOR_PER_QUESTION_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "selfNarrator", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_SELF_NARRATOR_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "selfNarrator", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_SELF_NARRATOR_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "selfNarrator", "maxObservationsPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_SELF_NARRATOR_MAX_OBSERVATIONS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "selfNarrator", "minSupportEpisodes"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_SELF_NARRATOR_MIN_SUPPORT_EPISODES"),
  );
  setConfigOverride(
    overrides,
    ["offline", "selfNarrator", "cadenceHintDays"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_SELF_NARRATOR_CADENCE_HINT_DAYS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "enabled"],
    readOptionalEnvBoolean(env, "BORG_OFFLINE_BELIEF_REVISER_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "confidenceDropMultiplier"],
    readOptionalEnvUnitInterval(env, "BORG_OFFLINE_BELIEF_REVISER_CONFIDENCE_DROP_MULTIPLIER"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "confidenceFloor"],
    readOptionalEnvUnitInterval(env, "BORG_OFFLINE_BELIEF_REVISER_CONFIDENCE_FLOOR"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "regradeBatchSize"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_REGRADE_BATCH_SIZE"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "maxEventsPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_MAX_EVENTS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "maxReviewsPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_MAX_REVIEWS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "claimStaleSec"],
    readOptionalEnvFloat(env, "BORG_OFFLINE_BELIEF_REVISER_CLAIM_STALE_SEC"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "maxParseFailures"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_MAX_PARSE_FAILURES"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "maxLlmCalls"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_MAX_LLM_CALLS"),
  );
  setConfigOverride(
    overrides,
    ["offline", "beliefReviser", "consecutiveParseFailureLimit"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_BELIEF_REVISER_CONSECUTIVE_PARSE_FAILURE_LIMIT"),
  );
  setConfigOverride(
    overrides,
    ["maintenance", "enabled"],
    readOptionalEnvBoolean(env, "BORG_MAINTENANCE_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["maintenance", "lightIntervalMs"],
    readOptionalEnvNumber(env, "BORG_MAINTENANCE_LIGHT_INTERVAL_MS"),
  );
  setConfigOverride(
    overrides,
    ["maintenance", "heavyIntervalMs"],
    readOptionalEnvNumber(env, "BORG_MAINTENANCE_HEAVY_INTERVAL_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "intervalMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_INTERVAL_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "maxWakesPerWindow"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_MAX_WAKES_PER_WINDOW"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "budgetWindowMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_BUDGET_WINDOW_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "executiveFocus", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_EXECUTIVE_FOCUS_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "executiveFocus", "stalenessSec"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_EXECUTIVE_FOCUS_STALENESS_SEC"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "executiveFocus", "dueLeadSec"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_EXECUTIVE_FOCUS_DUE_LEAD_SEC"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "executiveFocus", "wakeCooldownSec"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_EXECUTIVE_FOCUS_WAKE_COOLDOWN_SEC"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "commitmentExpiring", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_TRIGGER_COMMITMENT_EXPIRING_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "commitmentExpiring", "lookaheadMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_COMMITMENT_EXPIRING_LOOKAHEAD_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "openQuestionDormant", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_TRIGGER_OPEN_QUESTION_DORMANT_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "openQuestionDormant", "dormantMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_OPEN_QUESTION_DORMANT_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "scheduledReflection", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_TRIGGER_SCHEDULED_REFLECTION_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "scheduledReflection", "intervalMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_SCHEDULED_REFLECTION_INTERVAL_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "goalFollowupDue", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_TRIGGER_GOAL_FOLLOWUP_DUE_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "goalFollowupDue", "lookaheadMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_GOAL_FOLLOWUP_DUE_LOOKAHEAD_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "triggers", "goalFollowupDue", "staleMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_TRIGGER_GOAL_FOLLOWUP_DUE_STALE_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "conditions", "commitmentRevoked", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_CONDITION_COMMITMENT_REVOKED_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "conditions", "moodValenceDrop", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "conditions", "moodValenceDrop", "threshold"],
    readOptionalEnvFloat(env, "BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_THRESHOLD"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "conditions", "moodValenceDrop", "windowN"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_WINDOW_N"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "conditions", "moodValenceDrop", "activationPeriodMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_ACTIVATION_PERIOD_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "conditions", "openQuestionUrgencyBump", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_CONDITION_OPEN_QUESTION_URGENCY_BUMP_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "conditions", "openQuestionUrgencyBump", "threshold"],
    readOptionalEnvFloat(env, "BORG_AUTONOMY_CONDITION_OPEN_QUESTION_URGENCY_BUMP_THRESHOLD"),
  );

  return overrides;
}

function parseConfigFile(dataDir: string): ConfigOverrides {
  const configPath = join(dataDir, "config.json");

  try {
    const rawConfig = readJsonFile<unknown>(configPath);

    if (rawConfig === undefined) {
      return {};
    }

    const parsed = configBaseSchema.safeParse(rawConfig);

    if (!parsed.success) {
      throw new ConfigError(`Invalid config file at ${configPath}`, {
        cause: parsed.error,
        code: "CONFIG_FILE_INVALID",
      });
    }

    return rawConfig as ConfigOverrides;
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
  const envDataDir = readOptionalEnvString(env, "BORG_DATA_DIR");
  const lookupDataDir = expandPath(options.dataDir ?? envDataDir ?? DEFAULT_DATA_DIR);
  const fileOverrides = parseConfigFile(lookupDataDir);
  const envOverrides = loadEnvOverrides(env);
  let candidate = mergeConfigOverrides(fileOverrides, envOverrides);

  if (options.dataDir !== undefined) {
    candidate = mergeConfigOverrides(candidate, { dataDir: expandPath(options.dataDir) });
  }

  const parsed = configSchema.safeParse(candidate);

  if (!parsed.success) {
    throw new ConfigError("Invalid borg configuration", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function redactSecret(value: string): string;
function redactSecret(value: string | undefined): string | undefined;
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
