import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { OFFLINE_PROCESS_NAMES } from "../contracts/offline-process.js";
import { writeJsonFileAtomic } from "../util/atomic-write.js";
import { ConfigError } from "../util/errors.js";
import { DEFAULT_CONFIG, configSchema, loadConfig, redactConfig } from "./index.js";

describe("config", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("loads defaults without requiring API keys", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {},
    });

    expect(config.dataDir).toBe(tempDir);
    expect(config.embedding.baseUrl).toBe("http://localhost:1234/v1");
    expect(config.embedding.model).toBe("text-embedding-qwen3-embedding-8b");
    expect(config.anthropic.auth).toBe("auto");
    expect(config.anthropic.apiKey).toBeUndefined();
    expect(config.anthropic.models).toEqual({
      cognition: "claude-opus-5",
      background: "claude-opus-5",
      extraction: "claude-opus-5",
      recallExpansion: "claude-haiku-4-5-20251001",
      creatorDirective: "claude-sonnet-4-6",
      imagePerception: "claude-haiku-4-5-20251001",
    });
    expect(config.anthropic).toMatchObject({
      oauthSseInactivityTimeoutMs: 120_000,
      oauthSseFirstMessageEventTimeoutMs: 240_000,
      oauthSseMessageEventGapTimeoutMs: 180_000,
      oauthFetchHeadersTimeoutMs: 120_000,
      oauthUnaryBodyTimeoutMs: 120_000,
      unaryCallTimeoutMs: 360_000,
      streamingCallTimeoutMs: 720_000,
    });
    expect(config.host_capabilities).toContain("Inputs available to me");
    expect(config.host_capabilities).toContain("Proactive outbound messaging");
    expect(config.perception.llmEnabled).toBe(true);
    expect(config.frameAnomaly.peerChannelSourceTypes).toEqual(["kira"]);
    expect(config.affective.llmEnabled).toBe(true);
    expect(config.episodic.salienceGateEnabled).toBe(true);
    expect(config.offline.curator.episodeDecayIntervalMs).toBe(24 * 60 * 60 * 1_000);
    expect(config.offline.curator.episodeSalienceHalfLifeDays).toBe(30);
    expect(config.offline.curator.episodeHeatHalfLifeDays).toBe(7);
    expect(config.offline.curator.traitHalfLifeDays).toBe(30);
    expect(config.offline.curator.retrievalLogRetentionDays).toBe(90);
    expect(config.offline.semanticExtractor).toEqual({
      maxEpisodesPerRun: 8,
      maxInputTokensPerRun: 150_000,
      budget: 60_000,
    });
    expect(config.offline.associator).toEqual({
      episodesPerSample: 8,
      maxSamplesPerRun: 2,
      maxFindingsPerRun: 4,
      ceilingConfidence: 0.5,
      budget: 60_000,
    });
    expect(config.offline.creatorDirectiveReconciler).toEqual({
      maxFamiliesPerRun: 8,
      budget: 60_000,
    });
    expect(config.offline.commitmentReconciler).toEqual({
      maxGroupsPerRun: 8,
      budget: 60_000,
    });
    expect(config.offline.livedExperienceDaySummarizer).toEqual({
      budget: 160_000,
      windowDays: 7,
      maxDaysPerRun: 3,
      maxSelfDecisionEventsPerDay: 96,
      maxActivityEventsPerDay: 256,
      maxEpisodesPerDay: 12,
    });
    expect(config.offline.overseer.budget).toBeNull();
    expect(config.maintenance.lightProcesses).toEqual([
      "consolidator",
      "semantic-extractor",
      "curator",
    ]);
    expect(config.maintenance.optimizeStorage).toBe(true);
    expect(config.maintenance.lightBudget).toBeNull();
    expect(config.maintenance.heavyBudget).toBeNull();
    expect(config.maintenance.startupGraceMs).toBe(30_000);
    expect(config.maintenance.busyRetryBaseMs).toBe(60_000);
    expect(config.maintenance.busyRetryMaxMs).toBe(900_000);
    expect(config.maintenance.heavyProcesses).toEqual([
      "reflector",
      "overseer",
      "associator",
      "review-resolver",
      "ruminator",
      "self-narrator",
      "lived-experience-day-summarizer",
      "procedural-synthesizer",
      "belief-reviser",
      "creator-directive-reconciler",
      "commitment-reconciler",
    ]);
    expect([
      ...new Set([...config.maintenance.lightProcesses, ...config.maintenance.heavyProcesses]),
    ]).toEqual(expect.arrayContaining([...OFFLINE_PROCESS_NAMES]));
    expect(
      new Set([...config.maintenance.lightProcesses, ...config.maintenance.heavyProcesses]).size,
    ).toBe(OFFLINE_PROCESS_NAMES.length);
    expect(config.executive.goalFocusThreshold).toBe(0.45);
    expect(config.autonomy.maxWakesPerWindow).toBe(6);
    expect(config.autonomy.budgetWindowMs).toBe(24 * 60 * 60 * 1_000);
    expect(config.autonomy.reservedContemplativeWakesPerWindow).toBe(1);
    expect(config.autonomy.proactiveOutbound).toEqual({
      enabled: false,
      maxPostsPerWindow: 2,
      maxPostsPerTargetPerWindow: 1,
      windowMs: 24 * 60 * 60 * 1_000,
      maxAuthorizedTargets: 20,
      allowByCreatorDirective: true,
      allowByConfig: {
        sessionIds: [],
        sourceTypes: [],
      },
    });
    expect(config.autonomy.fleetBrake).toEqual({
      enabled: true,
      emptyStreakThreshold: 5,
      baseCooldownMs: 30 * 60 * 1_000,
      cooldownMultiplier: 2,
      maxCooldownMs: 6 * 60 * 60 * 1_000,
      errorStreakThreshold: 3,
      errorBasePauseMs: 5 * 60 * 1_000,
      errorMaxPauseMs: 30 * 60 * 1_000,
      freshnessBypassCap: 3,
    });
    expect(config.autonomy.executiveFocus.wakeCooldownSec).toBe(3_600);
    expect(config.autonomy.executiveFocus.emptyWakeBackoffMultiplier).toBe(2);
    expect(config.autonomy.executiveFocus.wakeCooldownMaxSec).toBe(86_400);
    expect(config.autonomy.executiveFocus.emptyWakeDormancyCount).toBe(3);
    expect(config.autonomy.triggers.goalFollowupDue.respectStaleBackoff).toBe(true);
    expect(config.streamIngestion.settle).toEqual({
      settleMs: 0,
      maxSettleMs: 30_000,
    });
    expect(config.streamIngestion.preTurnCatchup.maxEntries).toBe(100);
    expect(config.retrieval.semanticOverfetchMultiplier).toBe(3);
    expect(config.deliberation.contradictionRouting).toEqual({
      enabled: true,
      cooldownTurns: 5,
    });
    expect(config.commitments).toEqual({
      enforce: {
        regenerateBeforeSuppress: true,
        rewriteOnViolation: false,
      },
    });
    expect(config.generation.evidenceLedger).toEqual({
      enabled: true,
      currentSessionTranscriptTokenBudget: 2_500,
      actionThreadRenderLimit: 12,
      actionThreadSimilarityThreshold: 0.85,
      actionThreadSourceRecordLimit: 256,
      finalizerTargetTokens: 60_000,
      finalizerHardCapTokens: 100_000,
      finalizerMaxEntryTextTokens: 1_200,
      recentLivedExperience: {
        recencyWindowMs: 3 * 24 * 60 * 60_000,
        cap: 64,
        densityCap: 48,
        gapThresholdMs: 3 * 60 * 60_000,
      },
      sectionOptions: {},
      decisionArtifact: {
        maxActiveEntries: 40,
        maxLiveEntriesPerKey: 2,
        kindSoftCaps: {
          locked: 24,
          live: 10,
          low_salience_live: 4,
          dormant_live: 1,
          invalidated: 4,
          tentative: 2,
        },
        recentTurnThreshold: 5,
        dormantTurnThreshold: 15,
        renderMaxEntries: 40,
        renderMaxTokens: 5_000,
        renderReservedSlots: {
          live: 8,
          invalidated: 3,
        },
        renderLockedCap: 14,
        newestStateChangeReservedSlots: 3,
        previousArtifactSummary: {
          maxEntries: {
            locked: 14,
            live: 8,
            low_salience_live: 2,
            dormant_live: 0,
            invalidated: 4,
            tentative: 2,
          },
          summaryTokenBudget: 6_000,
          maxEntryTextTokens: 1_000,
        },
        compilerPrefilter: {
          enabled: true,
        },
        ledgerDelta: {
          enabled: true,
          minTailPerSection: 3,
        },
      },
    });
    expect(config.generation.cognition).toEqual({
      thinking: {
        enabled: false,
        mode: "adaptive",
        effort: "high",
        budget_tokens: 4096,
      },
    });
    expect(config.cognition.actionLifecycle).toEqual({
      archiveStaleAfterInactiveTurns: 20,
    });
    expect(config.generation.activeParticipantLimit).toBe(8);
    expect(config.generation.postGenerationGuards).toEqual({
      commitment: {
        mode: "enforce",
      },
      closurePressure: {
        mode: "enforce",
      },
    });
  });

  it("derives exported defaults from schema defaults", () => {
    expect(configSchema.parse({})).toEqual(DEFAULT_CONFIG);
  });

  it("loads frame-anomaly peer channel source types from config", () => {
    const config = configSchema.parse({
      frameAnomaly: {
        peerChannelSourceTypes: ["kira", "peerlink"],
      },
    });

    expect(config.frameAnomaly.peerChannelSourceTypes).toEqual(["kira", "peerlink"]);
  });

  it("accepts deprecated llm fallback aliases as llmEnabled config", () => {
    const config = configSchema.parse({
      perception: {
        useLlmFallback: false,
      },
      affective: {
        useLlmFallback: false,
      },
    });

    expect(config.perception.llmEnabled).toBe(false);
    expect(config.affective.llmEnabled).toBe(false);
    expect("useLlmFallback" in config.perception).toBe(false);
    expect("useLlmFallback" in config.affective).toBe(false);
  });

  it("caps semantic retrieval overfetch multiplier in config", () => {
    expect(
      configSchema.parse({
        retrieval: {
          semanticOverfetchMultiplier: 10,
        },
      }).retrieval.semanticOverfetchMultiplier,
    ).toBe(10);

    expect(() =>
      configSchema.parse({
        retrieval: {
          semanticOverfetchMultiplier: 11,
        },
      }),
    ).toThrow();
  });

  it("treats associator volume settings as hard caps", () => {
    expect(
      configSchema.parse({
        offline: {
          associator: {
            episodesPerSample: 8,
            maxSamplesPerRun: 2,
            maxFindingsPerRun: 4,
          },
        },
      }).offline.associator,
    ).toMatchObject({
      episodesPerSample: 8,
      maxSamplesPerRun: 2,
      maxFindingsPerRun: 4,
    });

    expect(() =>
      configSchema.parse({
        offline: {
          associator: {
            episodesPerSample: 9,
          },
        },
      }),
    ).toThrow();
    expect(() =>
      configSchema.parse({
        offline: {
          associator: {
            maxSamplesPerRun: 3,
          },
        },
      }),
    ).toThrow();
    expect(() =>
      configSchema.parse({
        offline: {
          associator: {
            maxFindingsPerRun: 5,
          },
        },
      }),
    ).toThrow();
  });

  it("keeps schema defaults when only one env var is set", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_OFFLINE_RUMINATOR_MAX_QUESTIONS_PER_RUN: "5",
      },
    });

    expect(config.offline.ruminator.maxQuestionsPerRun).toBe(5);
    expect(config.offline.overseer).toEqual(DEFAULT_CONFIG.offline.overseer);
    expect(config.generation.cognition).toEqual(DEFAULT_CONFIG.generation.cognition);
    expect(config.maintenance).toEqual(DEFAULT_CONFIG.maintenance);
  });

  it("accepts deprecated llm fallback env aliases", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_PERCEPTION_USE_LLM_FALLBACK: "false",
        BORG_AFFECTIVE_USE_LLM_FALLBACK: "false",
      },
    });

    expect(config.perception.llmEnabled).toBe(false);
    expect(config.affective.llmEnabled).toBe(false);
  });

  it("names the autonomy wake cap for the configured rolling window", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_AUTONOMY_MAX_WAKES_PER_WINDOW: "9",
        BORG_AUTONOMY_BUDGET_WINDOW_MS: "7200000",
        BORG_AUTONOMY_RESERVED_CONTEMPLATIVE_WAKES_PER_WINDOW: "2",
      },
    });

    expect(config.autonomy.maxWakesPerWindow).toBe(9);
    expect(config.autonomy.budgetWindowMs).toBe(7_200_000);
    expect(config.autonomy.reservedContemplativeWakesPerWindow).toBe(2);
  });

  it("loads executive focus empty-wake backoff config from env", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_AUTONOMY_EXECUTIVE_FOCUS_EMPTY_WAKE_BACKOFF_MULTIPLIER: "3.5",
        BORG_AUTONOMY_EXECUTIVE_FOCUS_WAKE_COOLDOWN_MAX_SEC: "7200",
        BORG_AUTONOMY_EXECUTIVE_FOCUS_EMPTY_WAKE_DORMANCY_COUNT: "5",
      },
    });

    expect(config.autonomy.executiveFocus.emptyWakeBackoffMultiplier).toBe(3.5);
    expect(config.autonomy.executiveFocus.wakeCooldownMaxSec).toBe(7_200);
    expect(config.autonomy.executiveFocus.emptyWakeDormancyCount).toBe(5);
  });

  it("loads fleet-brake and followup rollback controls from env", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_AUTONOMY_FLEET_BRAKE_ENABLED: "false",
        BORG_AUTONOMY_FLEET_BRAKE_EMPTY_STREAK_THRESHOLD: "7",
        BORG_AUTONOMY_FLEET_BRAKE_BASE_COOLDOWN_MS: "120000",
        BORG_AUTONOMY_FLEET_BRAKE_COOLDOWN_MULTIPLIER: "3.5",
        BORG_AUTONOMY_FLEET_BRAKE_MAX_COOLDOWN_MS: "7200000",
        BORG_AUTONOMY_FLEET_BRAKE_ERROR_STREAK_THRESHOLD: "4",
        BORG_AUTONOMY_FLEET_BRAKE_ERROR_BASE_PAUSE_MS: "60000",
        BORG_AUTONOMY_FLEET_BRAKE_ERROR_MAX_PAUSE_MS: "900000",
        BORG_AUTONOMY_FLEET_BRAKE_FRESHNESS_BYPASS_CAP: "2",
        BORG_AUTONOMY_TRIGGER_GOAL_FOLLOWUP_DUE_RESPECT_STALE_BACKOFF: "false",
      },
    });

    expect(config.autonomy.fleetBrake).toEqual({
      enabled: false,
      emptyStreakThreshold: 7,
      baseCooldownMs: 120_000,
      cooldownMultiplier: 3.5,
      maxCooldownMs: 7_200_000,
      errorStreakThreshold: 4,
      errorBasePauseMs: 60_000,
      errorMaxPauseMs: 900_000,
      freshnessBypassCap: 2,
    });
    expect(config.autonomy.triggers.goalFollowupDue.respectStaleBackoff).toBe(false);
  });

  it("loads autonomous proactive outbound gates from config and env", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    writeJsonFileAtomic(join(tempDir, "config.json"), {
      autonomy: {
        proactiveOutbound: {
          enabled: true,
          maxPostsPerWindow: 3,
          maxPostsPerTargetPerWindow: 2,
          windowMs: 3_600_000,
          allowByConfig: {
            sessionIds: ["default"],
            sourceTypes: ["demo"],
          },
        },
      },
    });

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_AUTONOMY_PROACTIVE_OUTBOUND_MAX_POSTS_PER_WINDOW: "4",
        BORG_AUTONOMY_PROACTIVE_OUTBOUND_MAX_POSTS_PER_TARGET_PER_WINDOW: "3",
        BORG_AUTONOMY_PROACTIVE_OUTBOUND_MAX_AUTHORIZED_TARGETS: "9",
        BORG_AUTONOMY_PROACTIVE_OUTBOUND_ALLOW_BY_CREATOR_DIRECTIVE: "false",
      },
    });

    expect(config.autonomy.proactiveOutbound).toEqual({
      enabled: true,
      maxPostsPerWindow: 4,
      maxPostsPerTargetPerWindow: 3,
      windowMs: 3_600_000,
      maxAuthorizedTargets: 9,
      allowByCreatorDirective: false,
      allowByConfig: {
        sessionIds: ["default"],
        sourceTypes: ["demo"],
      },
    });
  });

  it("loads host capabilities from the config file", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const hostCapabilities = [
      "Inputs available to me:",
      "- host-owned project tracker",
      "",
      "Output channels available now:",
      "- EmitAnswer: respond to the user",
      "- NotifyUserLater: send a user-visible follow-up",
    ].join("\n");

    writeJsonFileAtomic(join(tempDir, "config.json"), {
      host_capabilities: hostCapabilities,
    });

    const config = loadConfig({
      dataDir: tempDir,
      env: {},
    });

    expect(config.host_capabilities).toBe(hostCapabilities);
    expect(config.host_capabilities).not.toContain("Proactive outbound messaging");
  });

  it("merges config file values with environment overrides", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    writeJsonFileAtomic(join(tempDir, "config.json"), {
      embedding: {
        model: "file-model",
        dims: 2048,
      },
      anthropic: {
        models: {
          cognition: "file-cognition",
          recallExpansion: "file-recall",
        },
      },
      offline: {
        curator: {
          retrievalLogRetentionDays: 60,
        },
        beliefReviser: {
          enabled: false,
          maxLlmCalls: 4,
        },
        creatorDirectiveReconciler: {
          maxFamiliesPerRun: 4,
        },
        associator: {
          maxFindingsPerRun: 2,
        },
        semanticExtractor: {
          maxEpisodesPerRun: 3,
        },
      },
      executive: {
        goalFocusThreshold: 0.4,
      },
      streamIngestion: {
        settle: {
          settleMs: 1_000,
          maxSettleMs: 10_000,
        },
        preTurnCatchup: {
          maxEntries: 12,
        },
      },
      cognition: {
        actionLifecycle: {
          archiveStaleAfterInactiveTurns: 24,
        },
      },
      generation: {
        evidenceLedger: {
          enabled: true,
          currentSessionTranscriptTokenBudget: 12_000,
          actionThreadRenderLimit: 10,
          sectionOptions: {
            current_session_transcript: {
              maxEntries: 24,
            },
            prior_session_memory: {
              maxTokens: 2_500,
            },
          },
        },
        cognition: {
          thinking: {
            enabled: false,
            budget_tokens: 2048,
          },
        },
      },
    });

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_EMBEDDING_MODEL: "env-model",
        BORG_EMBEDDING_DIMS: "1024",
        BORG_PERCEPTION_LLM_ENABLED: "false",
        BORG_EPISODIC_SALIENCE_GATE_ENABLED: "false",
        BORG_OFFLINE_CURATOR_RETRIEVAL_LOG_RETENTION_DAYS: "45",
        BORG_OFFLINE_BELIEF_REVISER_MAX_LLM_CALLS: "7",
        BORG_OFFLINE_CREATOR_DIRECTIVE_RECONCILER_MAX_FAMILIES_PER_RUN: "6",
        BORG_OFFLINE_CREATOR_DIRECTIVE_RECONCILER_BUDGET: "14000",
        BORG_OFFLINE_ASSOCIATOR_MAX_FINDINGS_PER_RUN: "3",
        BORG_OFFLINE_ASSOCIATOR_BUDGET: "13000",
        BORG_OFFLINE_SEMANTIC_EXTRACTOR_MAX_INPUT_TOKENS_PER_RUN: "90000",
        BORG_OFFLINE_SEMANTIC_EXTRACTOR_BUDGET: "12000",
        BORG_OFFLINE_LIVED_EXPERIENCE_DAY_SUMMARIZER_BUDGET: "123000",
        BORG_EXECUTIVE_GOAL_FOCUS_THRESHOLD: "0.6",
        BORG_STREAM_INGESTION_PRE_TURN_CATCHUP_MAX_ENTRIES: "8",
        BORG_GENERATION_EVIDENCE_LEDGER_CURRENT_SESSION_TRANSCRIPT_TOKEN_BUDGET: "16000",
        BORG_GENERATION_EVIDENCE_LEDGER_ACTION_THREAD_RENDER_LIMIT: "8",
        BORG_GENERATION_EVIDENCE_LEDGER_ACTION_THREAD_SIMILARITY_THRESHOLD: "0.9",
        BORG_GENERATION_EVIDENCE_LEDGER_ACTION_THREAD_SOURCE_RECORD_LIMIT: "128",
        BORG_GENERATION_EVIDENCE_LEDGER_FINALIZER_TARGET_TOKENS: "60000",
        BORG_GENERATION_EVIDENCE_LEDGER_FINALIZER_HARD_CAP_TOKENS: "100000",
        BORG_GENERATION_EVIDENCE_LEDGER_FINALIZER_MAX_ENTRY_TEXT_TOKENS: "900",
        BORG_GENERATION_EVIDENCE_LEDGER_RECENT_LIVED_EXPERIENCE_RECENCY_WINDOW_MS: "172800000",
        BORG_GENERATION_EVIDENCE_LEDGER_RECENT_LIVED_EXPERIENCE_CAP: "72",
        BORG_GENERATION_EVIDENCE_LEDGER_RECENT_LIVED_EXPERIENCE_DENSITY_CAP: "24",
        BORG_GENERATION_EVIDENCE_LEDGER_RECENT_LIVED_EXPERIENCE_GAP_THRESHOLD_MS: "60000",
        BORG_COGNITION_ACTION_LIFECYCLE_ARCHIVE_STALE_AFTER_INACTIVE_TURNS: "18",
        BORG_DELIBERATION_CONTRADICTION_ROUTING_ENABLED: "false",
        BORG_DELIBERATION_CONTRADICTION_ROUTING_COOLDOWN_TURNS: "3",
        BORG_STREAM_INGESTION_SETTLE_MS: "3000",
        BORG_STREAM_INGESTION_MAX_SETTLE_MS: "30000",
        BORG_GENERATION_COGNITION_THINKING_ENABLED: "true",
        BORG_GENERATION_COGNITION_THINKING_BUDGET_TOKENS: "8192",
        BORG_MAINTENANCE_OPTIMIZE_STORAGE: "false",
        BORG_MAINTENANCE_STARTUP_GRACE_MS: "1234",
        BORG_MAINTENANCE_BUSY_RETRY_BASE_MS: "2345",
        BORG_MAINTENANCE_BUSY_RETRY_MAX_MS: "3456",
        BORG_MODEL_RECALL_EXPANSION: "env-recall",
        BORG_ANTHROPIC_OAUTH_SSE_INACTIVITY_TIMEOUT_MS: "111",
        BORG_ANTHROPIC_OAUTH_SSE_FIRST_MESSAGE_EVENT_TIMEOUT_MS: "222",
        BORG_ANTHROPIC_OAUTH_SSE_MESSAGE_EVENT_GAP_TIMEOUT_MS: "333",
        BORG_ANTHROPIC_OAUTH_FETCH_HEADERS_TIMEOUT_MS: "444",
        BORG_ANTHROPIC_OAUTH_UNARY_BODY_TIMEOUT_MS: "555",
        BORG_ANTHROPIC_UNARY_CALL_TIMEOUT_MS: "666",
        BORG_ANTHROPIC_STREAMING_CALL_TIMEOUT_MS: "777",
        ANTHROPIC_API_KEY: "secret",
      },
    });

    expect(config.embedding.model).toBe("env-model");
    expect(config.embedding.dims).toBe(1024);
    expect(config.perception.llmEnabled).toBe(false);
    expect(config.episodic.salienceGateEnabled).toBe(false);
    expect(config.anthropic.auth).toBe("auto");
    expect(config.anthropic.apiKey).toBe("secret");
    expect(config.anthropic.models.cognition).toBe("file-cognition");
    expect(config.anthropic.models.recallExpansion).toBe("env-recall");
    expect(config.anthropic.oauthSseInactivityTimeoutMs).toBe(111);
    expect(config.anthropic.oauthSseFirstMessageEventTimeoutMs).toBe(222);
    expect(config.anthropic.oauthSseMessageEventGapTimeoutMs).toBe(333);
    expect(config.anthropic.oauthFetchHeadersTimeoutMs).toBe(444);
    expect(config.anthropic.oauthUnaryBodyTimeoutMs).toBe(555);
    expect(config.anthropic.unaryCallTimeoutMs).toBe(666);
    expect(config.anthropic.streamingCallTimeoutMs).toBe(777);
    expect(config.executive.goalFocusThreshold).toBe(0.6);
    expect(config.streamIngestion.settle).toEqual({
      settleMs: 3000,
      maxSettleMs: 30000,
    });
    expect(config.streamIngestion.preTurnCatchup.maxEntries).toBe(8);
    expect(config.deliberation.contradictionRouting).toEqual({
      enabled: false,
      cooldownTurns: 3,
    });
    expect(config.commitments.enforce).toEqual({
      regenerateBeforeSuppress: true,
      rewriteOnViolation: false,
    });
    expect(config.generation.postGenerationGuards.commitment.mode).toBe("enforce");
    expect(config.generation.postGenerationGuards.closurePressure.mode).toBe("enforce");
    expect(config.generation.evidenceLedger.enabled).toBe(true);
    expect(config.generation.evidenceLedger.currentSessionTranscriptTokenBudget).toBe(16_000);
    expect(config.generation.evidenceLedger.actionThreadRenderLimit).toBe(8);
    expect(config.generation.evidenceLedger.actionThreadSimilarityThreshold).toBe(0.9);
    expect(config.generation.evidenceLedger.actionThreadSourceRecordLimit).toBe(128);
    expect(config.generation.evidenceLedger.finalizerTargetTokens).toBe(60_000);
    expect(config.generation.evidenceLedger.finalizerHardCapTokens).toBe(100_000);
    expect(config.generation.evidenceLedger.finalizerMaxEntryTextTokens).toBe(900);
    expect(config.generation.evidenceLedger.recentLivedExperience).toEqual({
      recencyWindowMs: 172_800_000,
      cap: 72,
      densityCap: 24,
      gapThresholdMs: 60_000,
    });
    expect(config.generation.evidenceLedger.decisionArtifact).toMatchObject({
      maxActiveEntries: 40,
      maxLiveEntriesPerKey: 2,
      renderMaxEntries: 40,
      renderLockedCap: 14,
    });
    expect(config.generation.evidenceLedger.sectionOptions).toEqual({
      current_session_transcript: {
        maxEntries: 24,
      },
      prior_session_memory: {
        maxTokens: 2_500,
      },
    });
    expect(config.generation.cognition.thinking).toEqual({
      enabled: true,
      mode: "adaptive",
      effort: "high",
      budget_tokens: 8192,
    });
    expect(config.maintenance.optimizeStorage).toBe(false);
    expect(config.maintenance.startupGraceMs).toBe(1234);
    expect(config.maintenance.busyRetryBaseMs).toBe(2345);
    expect(config.maintenance.busyRetryMaxMs).toBe(3456);
    expect(config.cognition.actionLifecycle.archiveStaleAfterInactiveTurns).toBe(18);
    expect(config.offline.curator.retrievalLogRetentionDays).toBe(45);
    expect(config.offline.beliefReviser.maxLlmCalls).toBe(7);
    expect(config.offline.creatorDirectiveReconciler.maxFamiliesPerRun).toBe(6);
    expect(config.offline.creatorDirectiveReconciler.budget).toBe(14_000);
    expect(config.offline.associator.maxFindingsPerRun).toBe(3);
    expect(config.offline.associator.budget).toBe(13_000);
    expect(config.offline.semanticExtractor.maxEpisodesPerRun).toBe(3);
    expect(config.offline.semanticExtractor.maxInputTokensPerRun).toBe(90_000);
    expect(config.offline.semanticExtractor.budget).toBe(12_000);
    expect(config.offline.livedExperienceDaySummarizer.budget).toBe(123_000);
  });

  it("accepts post-generation guard simple modes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    writeJsonFileAtomic(join(tempDir, "config.json"), {
      generation: {
        postGenerationGuards: {
          commitment: {
            mode: "enforce",
          },
          closurePressure: {
            mode: "shadow",
          },
        },
      },
    });

    const config = loadConfig({
      dataDir: tempDir,
      env: {},
    });

    expect(config.generation.postGenerationGuards.commitment.mode).toBe("enforce");
    expect(config.generation.postGenerationGuards.closurePressure.mode).toBe("shadow");
  });

  it("accepts commitment enforce guard options", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    writeJsonFileAtomic(join(tempDir, "config.json"), {
      commitments: {
        enforce: {
          rewriteOnViolation: true,
        },
      },
    });

    const config = loadConfig({
      dataDir: tempDir,
      env: {},
    });

    expect(config.commitments.enforce).toEqual({
      regenerateBeforeSuppress: true,
      rewriteOnViolation: true,
    });
  });

  it("defaults recall expansion to the dedicated Haiku slot", () => {
    expect(DEFAULT_CONFIG.anthropic.models).toEqual({
      cognition: "claude-opus-5",
      background: "claude-opus-5",
      extraction: "claude-opus-5",
      recallExpansion: "claude-haiku-4-5-20251001",
      creatorDirective: "claude-sonnet-4-6",
      imagePerception: "claude-haiku-4-5-20251001",
    });
  });

  it("requires an api key when anthropic auth mode is api-key", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: {
          BORG_ANTHROPIC_AUTH: "api-key",
        },
      }),
    ).toThrow(ConfigError);
  });

  it("throws config errors for invalid numeric environment values", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: {
          BORG_EMBEDDING_DIMS: "nope",
        },
      }),
    ).toThrow(ConfigError);
  });

  it("loads maintenance process lists and token budgets from the environment", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_MAINTENANCE_LIGHT_PROCESSES: " consolidator, curator ",
        BORG_MAINTENANCE_HEAVY_PROCESSES: "review-resolver,belief-reviser",
        BORG_MAINTENANCE_LIGHT_BUDGET: "1200",
        BORG_MAINTENANCE_HEAVY_BUDGET: "3400",
        BORG_OFFLINE_REVIEW_RESOLVER_BUDGET: "560",
      },
    });

    expect(config.maintenance.lightProcesses).toEqual(["consolidator", "curator"]);
    expect(config.maintenance.heavyProcesses).toEqual(["review-resolver", "belief-reviser"]);
    expect(config.maintenance.lightBudget).toBe(1_200);
    expect(config.maintenance.heavyBudget).toBe(3_400);
    expect(config.offline.reviewResolver.budget).toBe(560);
  });

  it.each([
    ["BORG_MAINTENANCE_LIGHT_BUDGET", "0"],
    ["BORG_MAINTENANCE_HEAVY_BUDGET", "-1"],
    ["BORG_OFFLINE_REVIEW_RESOLVER_BUDGET", "1.5"],
  ])("rejects invalid positive integer budget %s=%s", (name, value) => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() => loadConfig({ dataDir: tempDir, env: { [name]: value } })).toThrow(ConfigError);
  });

  it.each([
    ["unknown process", "consolidator,not-a-process"],
    ["blank entry", "consolidator,,curator"],
    ["leading blank", ",consolidator"],
    ["trailing blank", "consolidator,"],
    ["empty list", "   "],
    ["duplicate process", "consolidator,consolidator"],
  ])("rejects a maintenance CSV with %s", (_label, value) => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: { BORG_MAINTENANCE_LIGHT_PROCESSES: value },
      }),
    ).toThrow(ConfigError);
  });

  it("accepts negative and zero env numbers when the schema allows them", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        BORG_AUTONOMY_CONDITION_MOOD_VALENCE_DROP_THRESHOLD: "-0.5",
        BORG_OFFLINE_CURATOR_ARCHIVE_MIN_HEAT: "0",
        BORG_AUTONOMY_CONDITION_OPEN_QUESTION_URGENCY_BUMP_THRESHOLD: "0",
      },
    });

    expect(config.autonomy.conditions.moodValenceDrop.threshold).toBe(-0.5);
    expect(config.offline.curator.archiveMinHeat).toBe(0);
    expect(config.autonomy.conditions.openQuestionUrgencyBump.threshold).toBe(0);
  });

  it("rejects non-finite env numbers", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: {
          BORG_OFFLINE_CURATOR_ARCHIVE_MIN_HEAT: "NaN",
        },
      }),
    ).toThrow(ConfigError);
  });

  it("rejects reflector confidence ceilings above the hard cap", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    expect(() =>
      loadConfig({
        dataDir: tempDir,
        env: {
          BORG_OFFLINE_REFLECTOR_CEILING_CONFIDENCE: "0.9",
        },
      }),
    ).toThrow(ConfigError);
  });

  it("wraps invalid config file JSON in a typed config error", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const configPath = join(tempDir, "config.json");
    writeFileSync(configPath, '{"broken"', "utf8");

    try {
      loadConfig({
        dataDir: tempDir,
        env: {},
      });
      expect.unreachable("loadConfig should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(ConfigError);
      expect((error as ConfigError).code).toBe("CONFIG_FILE_INVALID");
      expect((error as ConfigError).message).toContain(configPath);
    }
  });

  it("redacts secrets for display", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const config = loadConfig({
      dataDir: tempDir,
      env: {
        ANTHROPIC_API_KEY: "secret",
      },
    });

    expect(redactConfig(config)).toMatchObject({
      frameAnomaly: {
        peerChannelSourceTypes: ["kira"],
      },
      embedding: {
        apiKey: "[REDACTED]",
      },
      anthropic: {
        apiKey: "[REDACTED]",
      },
    });
  });
});
