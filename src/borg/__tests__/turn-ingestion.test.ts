import { afterEach, describe, expect, it, vi } from "vitest";

import {
  FakeLLMClient,
  ManualClock,
  createSessionId,
  createTestConfig,
  Borg,
  ScriptedEmbeddingClient,
  borgInternals,
  createEmptyReflectionResponse,
  createReviewOpenQuestionResponse,
  createEntityId,
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

  it("waits for live ingestion to flush before closing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
          {
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: true,
    });

    let closePromise: Promise<void> | undefined;
    let closed = false;
    let resolveExtraction:
      | ((value: { inserted: number; updated: number; skipped: number }) => void)
      | undefined;

    try {
      const internal = borgInternals<{
        deps: {
          streamIngestionCoordinator?: {
            options: {
              extractor: {
                extractFromStream(): Promise<{
                  inserted: number;
                  updated: number;
                  skipped: number;
                }>;
              };
            };
          };
        };
      }>(borg);
      const coordinator = internal.deps.streamIngestionCoordinator;
      expect(coordinator).toBeDefined();

      let notifyExtractionStarted: (() => void) | undefined;
      const extractionStarted = new Promise<void>((resolve) => {
        notifyExtractionStarted = resolve;
      });
      let extractionCalls = 0;
      coordinator!.options.extractor = {
        async extractFromStream(): Promise<{ inserted: number; updated: number; skipped: number }> {
          extractionCalls += 1;
          notifyExtractionStarted?.();

          return await new Promise((resolve) => {
            resolveExtraction = resolve;
          });
        },
      };

      await borg.turn({
        userMessage: "Atlas deploy failed again.",
      });
      await extractionStarted;

      closePromise = borg.close();
      void closePromise.then(() => {
        closed = true;
      });

      await Promise.resolve();
      expect(extractionCalls).toBe(1);
      expect(closed).toBe(false);

      resolveExtraction?.({
        inserted: 1,
        updated: 0,
        skipped: 0,
      });
      await closePromise;
      expect(closed).toBe(true);
    } finally {
      if (!closed) {
        resolveExtraction?.({
          inserted: 1,
          updated: 0,
          skipped: 0,
        });
        await closePromise?.catch(() => undefined);

        if (closePromise === undefined) {
          await borg.close().catch(() => undefined);
        }
      }
    }
  });

  it("drains pending review open-question hooks before closing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    let notifyHookStarted: (() => void) | undefined;
    let releaseHook: (() => void) | undefined;
    const hookStarted = new Promise<void>((resolve) => {
      notifyHookStarted = resolve;
    });
    const hookRelease = new Promise<void>((resolve) => {
      releaseHook = resolve;
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
          async () => {
            notifyHookStarted?.();
            await hookRelease;

            return createReviewOpenQuestionResponse();
          },
        ],
      }),
    });

    let closePromise: Promise<void> | undefined;

    try {
      const internal = borgInternals<{
        deps: {
          reviewQueueRepository: {
            enqueue(input: {
              kind: "misattribution";
              refs: Record<string, unknown>;
              reason: string;
            }): unknown;
          };
        };
      }>(borg);

      internal.deps.reviewQueueRepository.enqueue({
        kind: "misattribution",
        refs: {
          target_type: "episode",
          target_id: "ep_aaaaaaaaaaaaaaaa",
        },
        reason: "La memoria mezcla dos atribuciones.",
      });
      await hookStarted;

      closePromise = borg.close();
      await Promise.resolve();
      releaseHook?.();
      await closePromise;

      const reopened = await Borg.open({
        dataDir: tempDir,
        clock,
        embeddingDimensions: 4,
        embeddingClient: new ScriptedEmbeddingClient(),
        llmClient: new FakeLLMClient(),
      });

      try {
        expect(reopened.self.openQuestions.list({ status: "open" })).toEqual([
          expect.objectContaining({
            question: "¿Qué atribución debería revisar?",
            urgency: 0.68,
            related_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          }),
        ]);
      } finally {
        await reopened.close();
      }
    } finally {
      releaseHook?.();
      if (closePromise === undefined) {
        await borg.close().catch(() => undefined);
      } else {
        await closePromise.catch(() => undefined);
      }
    }
  });

  it("enables live extraction by default", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
          {
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
    });

    try {
      const internal = borgInternals<{
        deps: {
          streamIngestionCoordinator?: {
            options: {
              extractor: {
                extractFromStream(): Promise<{
                  inserted: number;
                  updated: number;
                  skipped: number;
                }>;
              };
            };
          };
        };
      }>(borg);
      const coordinator = internal.deps.streamIngestionCoordinator;
      expect(coordinator).toBeDefined();
      let extractionCalls = 0;
      coordinator!.options.extractor = {
        async extractFromStream(): Promise<{ inserted: number; updated: number; skipped: number }> {
          extractionCalls += 1;
          return {
            inserted: 0,
            updated: 0,
            skipped: 0,
          };
        },
      };

      await borg.turn({
        userMessage: "Atlas deploy failed again.",
      });
      await borg.close();

      expect(extractionCalls).toBe(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("logs perception entries after the user message", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
          {
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Atlas deploy failed again.",
        stakes: "low",
      });

      expect(borg.stream.tail(3).map((entry) => entry.kind)).toEqual([
        "user_msg",
        "perception",
        "agent_msg",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("persists sender entity id from turn input on the user message", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const senderEntityId = createEntityId();
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
          {
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: false,
    });

    try {
      await borg.turn({
        userMessage: "Atlas deploy failed again.",
        senderEntityId,
      });

      const userEntry = borg.stream.tail(3).find((entry) => entry.kind === "user_msg");

      expect(userEntry?.sender_entity_id).toBe(senderEntityId);
    } finally {
      await borg.close();
    }
  });

  it("logs live extraction failures to the triggering session stream", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const sessionId = createSessionId();
    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
          {
            text: "Try the rollback plan.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      liveExtraction: true,
    });

    try {
      const internal = borgInternals<{
        deps: {
          streamIngestionCoordinator?: {
            options: {
              extractor: {
                extractFromStream(): Promise<{
                  inserted: number;
                  updated: number;
                  skipped: number;
                }>;
              };
            };
          };
        };
      }>(borg);
      const coordinator = internal.deps.streamIngestionCoordinator;
      expect(coordinator).toBeDefined();
      coordinator!.options.extractor = {
        async extractFromStream(): Promise<never> {
          throw new Error("boom");
        },
      };

      await borg.turn({
        userMessage: "Atlas deploy failed again.",
        sessionId,
      });
      await borg.close();

      const failedSessionEntries = borg.stream.tail(10, {
        session: sessionId,
      });
      const defaultSessionEntries = borg.stream.tail(10);

      expect(
        failedSessionEntries.some(
          (entry) =>
            entry.kind === "internal_event" &&
            String(entry.content).includes("Live episodic extraction failed: boom"),
        ),
      ).toBe(true);
      expect(
        defaultSessionEntries.some(
          (entry) =>
            entry.kind === "internal_event" &&
            String(entry.content).includes("Live episodic extraction failed: boom"),
        ),
      ).toBe(false);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("continues the turn and logs an internal event when pre-turn catch-up throws", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
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
          {
            text: "The turn still completes.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
          createEmptyReflectionResponse(),
        ],
      }),
      liveExtraction: true,
    });

    try {
      const internal = borgInternals<{
        deps: {
          streamIngestionCoordinator?: {
            catchUp(): Promise<never>;
            ingest(): Promise<{
              ran: boolean;
              processedEntries: number;
            }>;
          };
        };
      }>(borg);
      const coordinator = internal.deps.streamIngestionCoordinator;
      expect(coordinator).toBeDefined();
      coordinator!.catchUp = async (): Promise<never> => {
        throw new Error("catch-up exploded");
      };
      coordinator!.ingest = async () => ({
        ran: false,
        processedEntries: 0,
      });

      const result = await borg.turn({
        userMessage: "Please continue despite ingestion trouble.",
      });

      expect(result.response).toBe("The turn still completes.");
      expect(
        borg.stream.tail(10).some((entry) => {
          if (entry.kind !== "internal_event" || typeof entry.content !== "object") {
            return false;
          }

          return (
            entry.content !== null &&
            "hook" in entry.content &&
            entry.content.hook === "stream_ingestion_pre_turn_catchup" &&
            "error" in entry.content &&
            String(entry.content.error).includes("catch-up exploded")
          );
        }),
      ).toBe(true);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });
});
