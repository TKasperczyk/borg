import { afterEach, describe, expect, it, vi } from "vitest";

import {
  FakeLLMClient,
  ManualClock,
  createTestConfig,
  Borg,
  ScriptedEmbeddingClient,
  borgInternals,
  createEmptyReflectionResponse,
  createEmitAnswerResponse,
  createInvalidEntityClassifierResponse,
  createInvalidModeClassifierResponse,
  createNoTemporalCueResponse,
  createTurnPlanResponse,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

describe("Borg", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("keeps a turn running when mood update fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("Try the rollback plan.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
        ],
      }),
      liveExtraction: false,
    });

    try {
      const internal = borgInternals<{
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: () => Promise<unknown>;
              moodRepository: {
                update: (sessionId: string, update: unknown) => unknown;
              };
            };
          };
        };
      }>(borg);
      vi.spyOn(internal.deps.turnOrchestrator.options.moodRepository, "update").mockImplementation(
        () => {
          throw new Error("mood exploded");
        },
      );
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async () => ({
        valence: -0.7,
        arousal: 0.4,
        dominant_emotion: "fear",
      });

      const result = await borg.turn({
        userMessage: "Atlas deploy failed again.",
      });

      expect(result.response).toContain("rollback plan");
      expect(borg.stream.tail(4)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "mood_update",
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("feeds current-turn perceived mood into retrieval before mood persistence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      dataDir: tempDir,
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("We can slow down and inspect the failure.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
        ],
      }),
      liveExtraction: false,
    });

    try {
      const internal = borgInternals<{
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: () => Promise<unknown>;
              retrievalPipeline: {
                searchWithContext: (
                  query: string,
                  options?: Record<string, unknown>,
                ) => Promise<unknown>;
              };
            };
          };
        };
      }>(borg);
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async () => ({
        valence: -0.9,
        arousal: 0.85,
        dominant_emotion: "fear",
      });
      const searchSpy = vi.spyOn(
        internal.deps.turnOrchestrator.options.retrievalPipeline,
        "searchWithContext",
      );

      await borg.turn({
        userMessage: "Atlas deploy failed and I am panicking.",
      });

      expect(searchSpy.mock.calls[0]?.[1]).toMatchObject({
        moodState: {
          valence: -0.9,
          arousal: 0.85,
          dominant_emotion: "fear",
        },
      });
    } finally {
      await borg.close();
    }
  });

  it("keeps a turn running when social update fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("Focus on the audience and clarify the tone first.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("I'll keep this short for Sam.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
        ],
      }),
    });

    try {
      const internal = borgInternals<{
        deps: {
          turnOrchestrator: {
            options: {
              socialRepository: {
                recordInteractionWithId: (entityId: string, interaction: unknown) => unknown;
              };
            };
          };
        };
      }>(borg);
      vi.spyOn(
        internal.deps.turnOrchestrator.options.socialRepository,
        "recordInteractionWithId",
      ).mockImplementation(() => {
        throw new Error("social exploded");
      });

      const result = await borg.turn({
        userMessage: "Can you phrase this carefully for Sam?",
        audience: "Sam",
      });

      expect(result.response).toContain("clarify the tone first");
      expect(borg.stream.tail(4)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "social_update",
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("attributes social sentiment from the next user turn instead of the agent response", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createTurnPlanResponse(),
          createEmitAnswerResponse("Warm, supportive reply for Sam.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createTurnPlanResponse(),
          createEmitAnswerResponse("I hear that landed badly.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("I hear that landed badly.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("I hear that landed badly.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
        ],
      }),
      liveExtraction: false,
    });

    try {
      const internal = borgInternals<{
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: (text: string) => Promise<unknown>;
            };
          };
        };
      }>(borg);
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async (text) =>
        text.includes("frustrated")
          ? {
              valence: -1,
              arousal: 0.6,
              dominant_emotion: "anger",
            }
          : {
              valence: 0,
              arousal: 0,
              dominant_emotion: null,
            };

      await borg.turn({
        userMessage: "Can you phrase this carefully for Sam?",
        audience: "Sam",
      });

      const profileAfterFirst = borg.social.getProfile("Sam");
      const pendingAfterFirst = borg.workmem.load().pending_social_attribution;

      expect(profileAfterFirst?.interaction_count).toBe(1);
      expect(profileAfterFirst?.sentiment_history).toEqual([]);
      expect(pendingAfterFirst).not.toBeNull();
      expect(pendingAfterFirst?.interaction_id).toBeGreaterThan(0);

      clock.advance(1_000);
      await borg.turn({
        userMessage: "I'm frustrated and upset with how that landed.",
        audience: "Sam",
      });

      const profileAfterSecond = borg.social.getProfile("Sam");
      const pendingAfterSecond = borg.workmem.load().pending_social_attribution;

      expect(profileAfterSecond?.interaction_count).toBe(2);
      expect(profileAfterSecond?.sentiment_history).toEqual([
        {
          ts: pendingAfterFirst?.turn_completed_ts ?? 0,
          valence: -1,
        },
      ]);
      expect(profileAfterSecond?.last_interaction_at).toBe(2_000);
      expect(pendingAfterSecond?.turn_completed_ts).toBe(2_000);
    } finally {
      await borg.close();
    }
  });

  it("records group-channel social interactions on the current speaker", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("I will keep Alice's plan separate.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: false,
    });

    try {
      borg.entities.resolve("Planning Room", {
        kind: "group",
      });
      const alice = borg.entities.resolve("Alice", {
        kind: "person",
      });

      await borg.turn({
        userMessage: "I can take the flights part.",
        audience: "Planning Room",
        senderEntityId: alice,
      });

      expect(borg.social.getProfile("Alice")?.interaction_count).toBe(1);
      expect(borg.social.getProfile("Planning Room")).toBeNull();
    } finally {
      await borg.close();
    }
  });

  it("keeps pending social attribution across an autonomous wake until the next user reply", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
        autonomy: {
          enabled: true,
          intervalMs: 60_000,
          maxWakesPerWindow: 6,
          budgetWindowMs: 86_400_000,
          executiveFocus: {
            enabled: false,
            stalenessSec: 86_400,
            dueLeadSec: 0,
          },
          triggers: {
            commitmentExpiring: {
              enabled: false,
              lookaheadMs: 86_400_000,
            },
            openQuestionDormant: {
              enabled: false,
              dormantMs: 604_800_000,
            },
            scheduledReflection: {
              enabled: true,
              intervalMs: 60_000,
            },
            goalFollowupDue: {
              enabled: false,
              lookaheadMs: 604_800_000,
              staleMs: 1_209_600_000,
            },
          },
          conditions: {
            commitmentRevoked: {
              enabled: false,
            },
            moodValenceDrop: {
              enabled: false,
              threshold: -0.5,
              windowN: 5,
              activationPeriodMs: 86_400_000,
            },
            openQuestionUrgencyBump: {
              enabled: false,
              threshold: 0.9,
            },
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("First reply for Sam.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Autonomous reflection.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Follow-up reply for Sam.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
          createEmitAnswerResponse("Extra fallback.", {
            inputTokens: 8,
            outputTokens: 4,
          }),
        ],
      }),
    });

    try {
      const internal = borgInternals<{
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: (text: string) => Promise<unknown>;
            };
          };
        };
      }>(borg);
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async (text) =>
        text.includes("frustrated")
          ? {
              valence: -1,
              arousal: 0.6,
              dominant_emotion: "anger",
            }
          : {
              valence: 0,
              arousal: 0,
              dominant_emotion: null,
            };

      await borg.turn({
        userMessage: "Can you phrase this carefully for Sam?",
        audience: "Sam",
      });

      const pendingAfterFirst = borg.workmem.load().pending_social_attribution;
      expect(pendingAfterFirst).not.toBeNull();

      clock.advance(1_000);
      const wakeResult = await borg.autonomy.scheduler.tick();
      expect(wakeResult.firedEvents).toBe(1);
      expect(borg.workmem.load().pending_social_attribution).toEqual(pendingAfterFirst);
      expect(borg.social.getProfile("Sam")?.interaction_count).toBe(1);

      clock.advance(1_000);
      await borg.turn({
        userMessage: "I'm frustrated and upset with how that landed.",
        audience: "Sam",
      });

      const profileAfterSecond = borg.social.getProfile("Sam");
      expect(profileAfterSecond?.interaction_count).toBe(2);
      expect(profileAfterSecond?.sentiment_history).toEqual([
        {
          ts: pendingAfterFirst?.turn_completed_ts ?? 0,
          valence: -1,
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("keeps mood neutral when only the agent response is enthusiastic", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEmitAnswerResponse(
            "Amazing, great, wonderful progress! I'm thrilled this is working!",
            {
              inputTokens: 8,
              outputTokens: 4,
            },
          ),
        ],
      }),
    });

    try {
      await borg.turn({
        userMessage: "Status update on Atlas build.",
      });

      expect(borg.mood.current("default" as never).valence).toBe(0);
      expect(borg.mood.history("default" as never)).toEqual([]);
    } finally {
      await borg.close();
    }
  });

  it("falls back to neutral affect when affective extraction fails and logs an internal event", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("Let's inspect the deploy state first.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
          createEmptyReflectionResponse(),
        ],
      }),
    });

    try {
      const internal = borgInternals<{
        deps: {
          turnOrchestrator: {
            options: {
              affectiveSignalDetector?: (
                text: string,
                recentHistory?: readonly string[],
                options?: unknown,
              ) => Promise<unknown>;
            };
          };
        };
      }>(borg);
      internal.deps.turnOrchestrator.options.affectiveSignalDetector = async () => {
        throw new Error("affect exploded");
      };

      const result = await borg.turn({
        userMessage: "Atlas deploy failed and I'm upset.",
      });

      expect(result.response).toContain("inspect the deploy state");
      expect(borg.workmem.load().mood).toEqual({
        valence: 0,
        arousal: 0,
        dominant_emotion: null,
      });
      expect(borg.stream.tail(4)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "affective_extraction",
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });

  it("logs internal events when perception classifiers degrade", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: true,
        },
        affective: {
          llmEnabled: false,
          incomingMoodWeight: 0.3,
          moodHalfLifeHours: 24,
          moodHistoryRetentionDays: 90,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
      }),
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          createInvalidEntityClassifierResponse(),
          createInvalidModeClassifierResponse(),
          createNoTemporalCueResponse(),
          createEmitAnswerResponse("The turn still completed.", {
            inputTokens: 10,
            outputTokens: 5,
          }),
          createEmptyReflectionResponse(),
        ],
      }),
    });

    try {
      const result = await borg.turn({
        userMessage: 'Talk to @alice about "Project Atlas".',
      });

      expect(result.response).toContain("still completed");
      // hot_entities is empty when entity-extractor classifier degrades:
      // the regex-heuristic fallback was removed in favor of LLM-only
      // extraction, so a failed LLM call yields empty entities rather
      // than false positives.
      expect(borg.workmem.load()).toMatchObject({
        mode: "idle",
        hot_entities: [],
      });
      expect(borg.stream.tail(10)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "perception_classifier",
              classifier: "entity_extractor",
              error: expect.stringContaining("Entity fallback returned invalid payload"),
            }),
          }),
          expect.objectContaining({
            kind: "internal_event",
            content: expect.objectContaining({
              hook: "perception_classifier",
              classifier: "mode_detector",
              error: expect.stringContaining("Mode fallback returned invalid payload"),
            }),
          }),
        ]),
      );
    } finally {
      await borg.close();
    }
  });
});
