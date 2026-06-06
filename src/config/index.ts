import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { z } from "zod";

import { DEFAULT_HOST_CAPABILITIES_SECTION } from "../cognition/prompts/host-capabilities.js";
import {
  EVIDENCE_LEDGER_SECTION_DEFINITIONS,
  type EvidenceLedgerSectionId,
} from "../cognition/evidence-ledger/types.js";
import { DEFAULT_EXECUTIVE_GOAL_FOCUS_THRESHOLD } from "../executive/index.js";
import { commitmentKindSchema, type CommitmentKind } from "../memory/commitments/types.js";
import { sessionIdSchema, sessionSourceTypeSchema } from "../sessions/index.js";
import { readJsonFile } from "../util/atomic-write.js";
import { ConfigError } from "../util/errors.js";
import { isPlainRecord } from "../util/guards.js";

const DEFAULT_DATA_DIR = "~/.borg";
export const DEFAULT_ACTIVE_PARTICIPANT_LIMIT = 8;

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
function normalizeLlmEnabledAlias(value: unknown): unknown {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }

  const record = { ...(value as Record<string, unknown>) };

  if (record.llmEnabled === undefined && record.useLlmFallback !== undefined) {
    record.llmEnabled = record.useLlmFallback;
  }

  delete record.useLlmFallback;

  return record;
}

const perceptionConfigSchema = z
  .preprocess(
    normalizeLlmEnabledAlias,
    z
      .object({
        llmEnabled: z.boolean().default(true),
      })
      .strict(),
  )
  .prefault({});
const affectiveConfigSchema = z
  .preprocess(
    normalizeLlmEnabledAlias,
    z
      .object({
        // Affective perception uses the background model as the primary classifier
        // when configured; heuristics are the offline/test fallback path.
        llmEnabled: z.boolean().default(true),
        incomingMoodWeight: z.number().min(0).max(1).default(0.3),
        moodHistoryRetentionDays: z.number().positive().default(90),
        moodHalfLifeHours: z.number().positive().default(24),
      })
      .strict(),
  )
  .prefault({});
const postGenerationGuardConfigSchema = z
  .object({
    mode: postGenerationGuardModeSchema.default("enforce"),
  })
  .prefault({});
const commitmentEnforceConfigSchema = z
  .object({
    // Deprecated compatibility setting retained for old configs. Stored/effective
    // enforcement_class controls commitment guard enforcement.
    criticalKinds: z
      .array(commitmentKindSchema)
      .default(["boundary", "audience_rule"] satisfies CommitmentKind[]),
    regenerateBeforeSuppress: z.boolean().default(true),
    rewriteOnViolation: z.boolean().default(false),
  })
  .strict()
  .prefault({});
const commitmentsConfigSchema = z
  .object({
    enforce: commitmentEnforceConfigSchema,
  })
  .strict()
  .prefault({});
const evidenceLedgerSectionIds = EVIDENCE_LEDGER_SECTION_DEFINITIONS.map(
  (definition) => definition.id,
) as [EvidenceLedgerSectionId, ...EvidenceLedgerSectionId[]];
const evidenceLedgerSectionIdSchema = z.enum(evidenceLedgerSectionIds);
const evidenceLedgerSectionOptionsSchema = z
  .object({
    maxEntries: z.number().int().positive().optional(),
    maxTokens: z.number().int().positive().optional(),
  })
  .strict();
const sharedStateKindSoftCapsSchema = z
  .object({
    locked: z.number().int().positive().default(24),
    live: z.number().int().positive().default(10),
    low_salience_live: z.number().int().positive().default(4),
    dormant_live: z.number().int().positive().default(1),
    invalidated: z.number().int().positive().default(4),
    tentative: z.number().int().positive().default(2),
  })
  .strict()
  .prefault({});
const sharedStateRenderReservedSlotsSchema = z
  .object({
    live: z.number().int().nonnegative().default(8),
    invalidated: z.number().int().nonnegative().default(3),
  })
  .strict()
  .prefault({});
const sharedStatePreviousArtifactSummaryMaxEntriesSchema = z
  .object({
    locked: z.number().int().nonnegative().default(14),
    live: z.number().int().nonnegative().default(8),
    low_salience_live: z.number().int().nonnegative().default(2),
    dormant_live: z.number().int().nonnegative().default(0),
    invalidated: z.number().int().nonnegative().default(4),
    tentative: z.number().int().nonnegative().default(2),
  })
  .strict()
  .prefault({});
const sharedStatePreviousArtifactSummaryConfigSchema = z
  .object({
    maxEntries: sharedStatePreviousArtifactSummaryMaxEntriesSchema,
    summaryTokenBudget: z.number().int().positive().default(6_000),
    maxEntryTextTokens: z.number().int().positive().default(1_000),
  })
  .strict()
  .prefault({});
const sharedStateCompilerPrefilterConfigSchema = z
  .object({
    enabled: z.boolean().default(true),
  })
  .strict()
  .prefault({});
const sharedStateLedgerDeltaConfigSchema = z
  .object({
    enabled: z.boolean().default(true),
    minTailPerSection: z.number().int().nonnegative().default(3),
  })
  .strict()
  .prefault({});
const sharedStateConfigSchema = z
  .object({
    maxActiveEntries: z.number().int().positive().default(40),
    maxLiveEntriesPerKey: z.number().int().positive().default(2),
    recentTurnThreshold: z.number().int().positive().default(5),
    dormantTurnThreshold: z.number().int().positive().default(15),
    kindSoftCaps: sharedStateKindSoftCapsSchema,
    renderMaxEntries: z.number().int().positive().default(40),
    renderMaxTokens: z.number().int().positive().default(5_000),
    renderReservedSlots: sharedStateRenderReservedSlotsSchema,
    renderLockedCap: z.number().int().nonnegative().default(14),
    newestStateChangeReservedSlots: z.number().int().nonnegative().default(3),
    previousArtifactSummary: sharedStatePreviousArtifactSummaryConfigSchema,
    compilerPrefilter: sharedStateCompilerPrefilterConfigSchema,
    ledgerDelta: sharedStateLedgerDeltaConfigSchema,
  })
  .strict()
  .prefault({});
const evidenceLedgerConfigSchema = z
  .object({
    enabled: z.boolean().default(true),
    currentSessionTranscriptTokenBudget: z.number().int().positive().default(2_500),
    actionThreadRenderLimit: z.number().int().positive().default(12),
    actionThreadSimilarityThreshold: z.number().min(0).max(1).default(0.85),
    actionThreadSourceRecordLimit: z.number().int().positive().default(256),
    finalizerTargetTokens: z.number().int().positive().default(60_000),
    finalizerHardCapTokens: z.number().int().positive().default(100_000),
    finalizerMaxEntryTextTokens: z.number().int().positive().default(1_200),
    sectionOptions: z
      .partialRecord(evidenceLedgerSectionIdSchema, evidenceLedgerSectionOptionsSchema)
      .default({}),
    decisionArtifact: sharedStateConfigSchema,
  })
  .prefault({});
const cognitionThinkingConfigSchema = z
  .object({
    enabled: z.boolean().default(false),
    budget_tokens: z.number().int().positive().default(4096),
  })
  .prefault({});
const generationCognitionConfigSchema = z
  .object({
    thinking: cognitionThinkingConfigSchema,
  })
  .prefault({});
const actionLifecycleConfigSchema = z
  .object({
    archiveStaleAfterInactiveTurns: z.number().int().nonnegative().default(20),
  })
  .strict()
  .prefault({});
const cognitionConfigSchema = z
  .object({
    actionLifecycle: actionLifecycleConfigSchema,
  })
  .strict()
  .prefault({});
const attachmentsConfigSchema = z
  .object({
    maxBytesPerImage: z
      .number()
      .int()
      .positive()
      .default(10 * 1024 * 1024),
    maxWidth: z.number().int().positive().default(8192),
    maxHeight: z.number().int().positive().default(8192),
    maxImagesPerTurn: z.number().int().positive().default(4),
    maxImagesPerLedger: z.number().int().positive().default(4),
    maxLedgerImageBytes: z
      .number()
      .int()
      .positive()
      .default(8 * 1024 * 1024),
    maxRetrievedImageRefs: z.number().int().positive().default(8),
    imageRenderMaxDimension: z.number().int().positive().default(8192),
    perceptionPromptVersion: z.string().min(1).default("v88-p1-2026-05-25"),
  })
  .strict()
  .prefault({});
const contradictionRoutingConfigSchema = z
  .object({
    enabled: z.boolean().default(true),
    cooldownTurns: z.number().int().nonnegative().default(5),
  })
  .strict()
  .prefault({});
const deliberationConfigSchema = z
  .object({
    contradictionRouting: contradictionRoutingConfigSchema,
  })
  .strict()
  .prefault({});
const postGenerationGuardsConfigSchema = z
  .object({
    commitment: postGenerationGuardConfigSchema,
    closurePressure: postGenerationGuardConfigSchema,
  })
  .strict()
  .prefault({});
const maintenanceProcessSchema = z.enum([
  "consolidator",
  "reflector",
  "semantic-extractor",
  "curator",
  "overseer",
  "review-resolver",
  "ruminator",
  "self-narrator",
  "procedural-synthesizer",
  "belief-reviser",
  "creator-directive-reconciler",
  "commitment-reconciler",
]);

export type PostGenerationGuardMode = z.infer<typeof postGenerationGuardModeSchema>;
const anthropicModelsConfigSchema = z
  .object({
    // The main cognition/extraction/background slots default to Opus 4.6.
    // Recall expansion is a small structured fanout task and has its own
    // Haiku slot so it can stay fast without reusing background.
    // Creator-directive extraction is a nuanced semantic classification
    // (it must split a durable fact from a behavioral rule), which Haiku
    // under-emits; it gets its own Sonnet slot -- stronger than the recall
    // Haiku, cheaper than the Opus cognition slot -- and only fires on
    // creator-in-operator turns, so the cost is bounded.
    cognition: z.string().min(1).default("claude-opus-4-6"),
    background: z.string().min(1).default("claude-opus-4-6"),
    extraction: z.string().min(1).default("claude-opus-4-6"),
    recallExpansion: z.string().min(1).default("claude-haiku-4-5-20251001"),
    creatorDirective: z.string().min(1).default("claude-sonnet-4-6"),
    imagePerception: z.string().min(1).default("claude-haiku-4-5-20251001"),
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
  perception: perceptionConfigSchema,
  affective: affectiveConfigSchema,
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
      semanticOverfetchMultiplier: z.number().int().min(1).max(10).default(3),
      semantic: z
        .object({
          underReviewMultiplier: z.number().min(0).max(1).default(0.5),
          statusMultipliers: z
            .object({
              active: z.number().min(0).max(1).default(1),
              superseded: z.number().min(0).max(1).default(0.5),
              contradicted: z.number().min(0).max(1).default(0.3),
              quarantined: z.number().min(0).max(1).default(0.2),
            })
            .strict()
            .prefault({}),
        })
        .prefault({}),
    })
    .prefault({}),
  commitments: commitmentsConfigSchema,
  attachments: attachmentsConfigSchema,
  cognition: cognitionConfigSchema,
  deliberation: deliberationConfigSchema,
  generation: z
    .object({
      discourseStateHardCapTurns: z.number().int().positive().default(50),
      activeParticipantLimit: z.number().int().positive().default(DEFAULT_ACTIVE_PARTICIPANT_LIMIT),
      cognition: generationCognitionConfigSchema,
      evidenceLedger: evidenceLedgerConfigSchema,
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
          similarityThreshold: z.number().positive().default(0.82),
          maxClusterDiameter: z.number().min(0).max(2).default(0.18),
          temporalProximityMs: z
            .number()
            .int()
            .nonnegative()
            .default(30 * 24 * 60 * 60 * 1_000),
          highSimilarityTemporalBypassThreshold: z.number().min(0).max(1).default(0.97),
          highSimilarityTemporalBypassMaxGapMs: z
            .number()
            .int()
            .nonnegative()
            .default(180 * 24 * 60 * 60 * 1_000),
          minClusterSize: z.number().int().positive().default(2),
          maxClustersPerRun: z.number().int().positive().default(2),
          maxFamilyRawMembers: z.number().int().positive().default(64),
          budget: z.number().int().positive().default(60_000),
        })
        .prefault({}),
      reflector: z
        .object({
          minSupport: z.number().int().positive().default(3),
          goalSimilarityThreshold: z.number().min(0).max(1).default(0.82),
          ceilingConfidence: z.number().positive().max(0.5).default(0.5),
          maxInsightsPerRun: z.number().int().positive().default(2),
          budget: z.number().int().positive().default(60_000),
        })
        .prefault({}),
      semanticExtractor: z
        .object({
          maxEpisodesPerRun: z.number().int().positive().default(8),
          maxInputTokensPerRun: z.number().int().positive().default(150_000),
          budget: z.number().int().positive().default(60_000),
        })
        .prefault({}),
      proceduralSynthesizer: z
        .object({
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
          lookbackHours: z.number().positive().default(24),
          maxChecksPerRun: z.number().int().positive().default(8),
          budget: z.number().int().positive().nullable().default(null),
        })
        .prefault({}),
      reviewResolver: z
        .object({
          maxItemsPerPass: z.number().int().positive().default(3),
          budget: z.number().int().positive().nullable().default(null),
        })
        .prefault({}),
      ruminator: z
        .object({
          maxQuestionsPerRun: z.number().int().positive().default(8),
          // Threshold applies to RetrievalConfidence.overall, a conservative
          // epistemic evidence-quality signal, not the relevance ranking score.
          resolveConfidenceThreshold: z.number().min(0).max(1).default(0.55),
          duplicateSimilarityThreshold: z.number().min(0).max(1).default(0.9),
          stalenessDays: z.number().positive().default(30),
          staleNoTractionTicks: z.number().int().positive().default(4),
          budget: z.number().int().positive().default(40_000),
        })
        .prefault({}),
      selfNarrator: z
        .object({
          budget: z.number().int().positive().default(80_000),
          maxObservationsPerRun: z.number().int().positive().default(4),
          minSupportEpisodes: z.number().int().positive().default(2),
          cadenceHintDays: z.number().positive().default(7),
        })
        .prefault({}),
      beliefReviser: z
        .object({
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
      creatorDirectiveReconciler: z
        .object({
          maxFamiliesPerRun: z.number().int().positive().default(8),
          budget: z.number().int().positive().default(60_000),
        })
        .prefault({}),
      commitmentReconciler: z
        .object({
          maxGroupsPerRun: z.number().int().positive().default(8),
          budget: z.number().int().positive().default(60_000),
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
      // These cadence lists are the single authority for offline process
      // enablement. Remove a process from both lists to disable it.
      lightProcesses: z
        .array(maintenanceProcessSchema)
        .default(["consolidator", "semantic-extractor", "curator"]),
      heavyProcesses: z
        .array(maintenanceProcessSchema)
        .default([
          "reflector",
          "overseer",
          "review-resolver",
          "ruminator",
          "self-narrator",
          "procedural-synthesizer",
          "belief-reviser",
          "creator-directive-reconciler",
          "commitment-reconciler",
        ]),
    })
    .prefault({}),
  autonomy: z
    .object({
      // Self-initiated cognition is part of the architecture's "autonomous
      // being" framing. The scheduler skeleton is default on (a runtime calls
      // scheduler.start(); library callers stay in control because start() is
      // explicit, and maxWakesPerWindow caps the cost). But only the wake
      // sources that can actually fire in the current regime are enabled by
      // default: the event-driven conditions (commitment_revoked,
      // open_question_urgency_bump), executive-focus-due, and the deliberate
      // scheduled-wake lever. The time-threshold triggers (commitment_expiring,
      // open_question_dormant, goal_followup_due) are default OFF -- they are
      // structurally inert until the underlying data carries the signal they
      // key on (commitments rarely set expires_at; dormancy/staleness windows
      // are 7-14 days against memory that is currently days old). Their modules
      // are retained; flip them on per-deployment once that data matures.
      enabled: z.boolean().default(true),
      intervalMs: z.number().int().positive().default(60_000),
      maxWakesPerWindow: z.number().int().positive().default(6),
      budgetWindowMs: z.number().int().positive().default(86_400_000),
      proactiveOutbound: z
        .object({
          enabled: z.boolean().default(false),
          maxPostsPerWindow: z.number().int().positive().default(2),
          maxPostsPerTargetPerWindow: z.number().int().positive().default(1),
          windowMs: z.number().int().positive().default(86_400_000),
          maxAuthorizedTargets: z.number().int().positive().default(20),
          allowByCreatorDirective: z.boolean().default(true),
          allowByConfig: z
            .object({
              sessionIds: z.array(sessionIdSchema).default([]),
              sourceTypes: z.array(sessionSourceTypeSchema).default([]),
            })
            .prefault({}),
        })
        .prefault({}),
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
              // Default off: commitments rarely carry expires_at, so this
              // trigger is structurally inert. See the autonomy comment above.
              enabled: z.boolean().default(false),
              lookaheadMs: z.number().int().positive().default(86_400_000),
            })
            .prefault({}),
          openQuestionDormant: z
            .object({
              // Default off: the 7-day dormancy window is inert against
              // memory that is currently days old. See the comment above.
              enabled: z.boolean().default(false),
              dormantMs: z.number().int().positive().default(604_800_000),
            })
            .prefault({}),
          scheduledReflection: z
            .object({
              enabled: z.boolean().default(false),
              intervalMs: z.number().int().positive().default(14_400_000),
            })
            .prefault({}),
          scheduledWake: z
            .object({
              // On by default but inert unless Borg actually schedules a wake
              // via tool.scheduledWakes.create -- the entity's deliberate,
              // one-time self-invocation lever.
              enabled: z.boolean().default(true),
            })
            .prefault({}),
          goalFollowupDue: z
            .object({
              // Default off: the 7-14 day follow-up/stale windows are inert
              // against memory that is currently days old. See comment above.
              enabled: z.boolean().default(false),
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

function readOptionalEnvBooleanAlias(
  env: NodeJS.ProcessEnv,
  primary: string,
  deprecated: string,
): boolean | undefined {
  return readOptionalEnvBoolean(env, primary) ?? readOptionalEnvBoolean(env, deprecated);
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
    ["perception", "llmEnabled"],
    readOptionalEnvBooleanAlias(
      env,
      "BORG_PERCEPTION_LLM_ENABLED",
      "BORG_PERCEPTION_USE_LLM_FALLBACK",
    ),
  );
  setConfigOverride(
    overrides,
    ["affective", "llmEnabled"],
    readOptionalEnvBooleanAlias(
      env,
      "BORG_AFFECTIVE_LLM_ENABLED",
      "BORG_AFFECTIVE_USE_LLM_FALLBACK",
    ),
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
    ["anthropic", "models", "creatorDirective"],
    readOptionalEnvString(env, "BORG_MODEL_CREATOR_DIRECTIVE"),
  );
  setConfigOverride(
    overrides,
    ["procedural", "skillSelectionMinSimilarity"],
    readOptionalEnvUnitInterval(env, "BORG_PROCEDURAL_SKILL_SELECTION_MIN_SIMILARITY"),
  );
  setConfigOverride(
    overrides,
    ["attachments", "maxBytesPerImage"],
    readOptionalEnvNumber(env, "BORG_ATTACHMENTS_MAX_BYTES_PER_IMAGE"),
  );
  setConfigOverride(
    overrides,
    ["attachments", "maxWidth"],
    readOptionalEnvNumber(env, "BORG_ATTACHMENTS_MAX_WIDTH"),
  );
  setConfigOverride(
    overrides,
    ["attachments", "maxHeight"],
    readOptionalEnvNumber(env, "BORG_ATTACHMENTS_MAX_HEIGHT"),
  );
  setConfigOverride(
    overrides,
    ["attachments", "maxImagesPerTurn"],
    readOptionalEnvNumber(env, "BORG_ATTACHMENTS_MAX_IMAGES_PER_TURN"),
  );
  setConfigOverride(
    overrides,
    ["attachments", "maxImagesPerLedger"],
    readOptionalEnvNumber(env, "BORG_ATTACHMENTS_MAX_IMAGES_PER_LEDGER"),
  );
  setConfigOverride(
    overrides,
    ["attachments", "maxLedgerImageBytes"],
    readOptionalEnvNumber(env, "BORG_ATTACHMENTS_MAX_LEDGER_IMAGE_BYTES"),
  );
  setConfigOverride(
    overrides,
    ["attachments", "maxRetrievedImageRefs"],
    readOptionalEnvNumber(env, "BORG_ATTACHMENTS_MAX_RETRIEVED_IMAGE_REFS"),
  );
  setConfigOverride(
    overrides,
    ["attachments", "imageRenderMaxDimension"],
    readOptionalEnvNumber(env, "BORG_ATTACHMENTS_IMAGE_RENDER_MAX_DIMENSION"),
  );
  setConfigOverride(
    overrides,
    ["retrieval", "semantic", "underReviewMultiplier"],
    readOptionalEnvUnitInterval(env, "BORG_RETRIEVAL_SEMANTIC_UNDER_REVIEW_MULTIPLIER"),
  );
  setConfigOverride(
    overrides,
    ["deliberation", "contradictionRouting", "enabled"],
    readOptionalEnvBoolean(env, "BORG_DELIBERATION_CONTRADICTION_ROUTING_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["deliberation", "contradictionRouting", "cooldownTurns"],
    readOptionalEnvNumber(env, "BORG_DELIBERATION_CONTRADICTION_ROUTING_COOLDOWN_TURNS"),
  );
  setConfigOverride(
    overrides,
    ["cognition", "actionLifecycle", "archiveStaleAfterInactiveTurns"],
    readOptionalEnvNumber(
      env,
      "BORG_COGNITION_ACTION_LIFECYCLE_ARCHIVE_STALE_AFTER_INACTIVE_TURNS",
    ),
  );
  setConfigOverride(
    overrides,
    ["generation", "discourseStateHardCapTurns"],
    readOptionalEnvNumber(env, "BORG_GENERATION_DISCOURSE_HARD_CAP_TURNS"),
  );
  setConfigOverride(
    overrides,
    ["generation", "activeParticipantLimit"],
    readOptionalEnvNumber(env, "BORG_GENERATION_ACTIVE_PARTICIPANT_LIMIT"),
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
    ["generation", "evidenceLedger", "finalizerTargetTokens"],
    readOptionalEnvNumber(env, "BORG_GENERATION_EVIDENCE_LEDGER_FINALIZER_TARGET_TOKENS"),
  );
  setConfigOverride(
    overrides,
    ["generation", "evidenceLedger", "finalizerHardCapTokens"],
    readOptionalEnvNumber(env, "BORG_GENERATION_EVIDENCE_LEDGER_FINALIZER_HARD_CAP_TOKENS"),
  );
  setConfigOverride(
    overrides,
    ["generation", "evidenceLedger", "finalizerMaxEntryTextTokens"],
    readOptionalEnvNumber(env, "BORG_GENERATION_EVIDENCE_LEDGER_FINALIZER_MAX_ENTRY_TEXT_TOKENS"),
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
    ["offline", "semanticExtractor", "maxEpisodesPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_SEMANTIC_EXTRACTOR_MAX_EPISODES_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "semanticExtractor", "maxInputTokensPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_SEMANTIC_EXTRACTOR_MAX_INPUT_TOKENS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "semanticExtractor", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_SEMANTIC_EXTRACTOR_BUDGET"),
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
    ["offline", "ruminator", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_RUMINATOR_BUDGET"),
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
    ["offline", "creatorDirectiveReconciler", "maxFamiliesPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_CREATOR_DIRECTIVE_RECONCILER_MAX_FAMILIES_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "creatorDirectiveReconciler", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_CREATOR_DIRECTIVE_RECONCILER_BUDGET"),
  );
  setConfigOverride(
    overrides,
    ["offline", "commitmentReconciler", "maxGroupsPerRun"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_COMMITMENT_RECONCILER_MAX_GROUPS_PER_RUN"),
  );
  setConfigOverride(
    overrides,
    ["offline", "commitmentReconciler", "budget"],
    readOptionalEnvNumber(env, "BORG_OFFLINE_COMMITMENT_RECONCILER_BUDGET"),
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
    ["autonomy", "proactiveOutbound", "enabled"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_PROACTIVE_OUTBOUND_ENABLED"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "proactiveOutbound", "maxPostsPerWindow"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_PROACTIVE_OUTBOUND_MAX_POSTS_PER_WINDOW"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "proactiveOutbound", "maxPostsPerTargetPerWindow"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_PROACTIVE_OUTBOUND_MAX_POSTS_PER_TARGET_PER_WINDOW"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "proactiveOutbound", "windowMs"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_PROACTIVE_OUTBOUND_WINDOW_MS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "proactiveOutbound", "maxAuthorizedTargets"],
    readOptionalEnvNumber(env, "BORG_AUTONOMY_PROACTIVE_OUTBOUND_MAX_AUTHORIZED_TARGETS"),
  );
  setConfigOverride(
    overrides,
    ["autonomy", "proactiveOutbound", "allowByCreatorDirective"],
    readOptionalEnvBoolean(env, "BORG_AUTONOMY_PROACTIVE_OUTBOUND_ALLOW_BY_CREATOR_DIRECTIVE"),
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
      semanticOverfetchMultiplier: config.retrieval.semanticOverfetchMultiplier,
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
        scheduledWake: {
          ...config.autonomy.triggers.scheduledWake,
        },
        goalFollowupDue: {
          ...config.autonomy.triggers.goalFollowupDue,
        },
      },
      proactiveOutbound: {
        ...config.autonomy.proactiveOutbound,
        allowByConfig: {
          sessionIds: [...config.autonomy.proactiveOutbound.allowByConfig.sessionIds],
          sourceTypes: [...config.autonomy.proactiveOutbound.allowByConfig.sourceTypes],
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
