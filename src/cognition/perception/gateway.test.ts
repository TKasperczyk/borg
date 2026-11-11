import { describe, expect, it, vi } from "vitest";

import type { Config } from "../../config/index.js";
import type { LLMCompleteOptions } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import type { StreamReader } from "../../stream/index.js";
import { ConfigError, LLMError } from "../../util/errors.js";
import { computeRetrievalLimit } from "../attention/weights.js";
import type { CognitiveMode } from "../types.js";
import { DEFAULT_SESSION_ID, type SessionId, type StreamEntryId } from "../../util/ids.js";
import { ManualClock } from "../../util/clock.js";
import type { RecencyWindow } from "../recency/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { PerceptionGateway } from "./gateway.js";

function makeConfig(llmEnabled: boolean): Config {
  return {
    perception: {
      llmEnabled,
    },
    affective: {
      llmEnabled: false,
    },
    anthropic: {
      models: {
        background: "background-model",
        recallExpansion: "fast-model",
      },
    },
  } as Config;
}

function makeRecencyWindow(): RecencyWindow {
  return {
    messages: [
      {
        role: "user",
        content: "prior question",
        stream_entry_id: "strm_abcdefghijklmnop" as StreamEntryId,
        sender_entity_id: null,
        ts: 900,
      },
      {
        role: "assistant",
        content: "prior answer",
        stream_entry_id: "strm_bcdefghijklmnopa" as StreamEntryId,
        sender_entity_id: null,
        ts: 950,
      },
    ],
    latest_ts: 950,
    total_chars: 27,
  };
}

function makeTracer(): TurnTracer {
  return {
    enabled: true,
    includePayloads: false,
    emit: vi.fn(),
  };
}

describe("PerceptionGateway", () => {
  it("compiles recency before perception and updates user-turn working memory", async () => {
    const recencyWindow = makeRecencyWindow();
    const compile = vi.fn(() => recencyWindow);
    const detectAffectiveSignal = vi.fn(async () => ({
      valence: 0.6,
      arousal: 0.2,
      dominant_emotion: "curiosity" as const,
    }));
    const tracer = makeTracer();
    const gateway = new PerceptionGateway({
      config: makeConfig(false),
      llmFactory: vi.fn(() => {
        throw new Error("should not be called");
      }),
      clock: new ManualClock(1_000),
      tracer,
      getAffectiveSignalDetector: () => detectAffectiveSignal,
      turnContextCompiler: {
        compile,
      },
      createStreamReader: vi.fn(() => ({}) as StreamReader),
    });

    const result = await gateway
      .beginTurn({
        turnId: "turn-1",
        onHookFailure: vi.fn(),
      })
      .perceive({
        sessionId: DEFAULT_SESSION_ID,
        isSelfAudience: true,
        origin: "user",
        cognitionInput: "plain lower text",
        workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 500),
      });

    expect(compile).toHaveBeenCalledWith(expect.anything(), {
      includeSelfTurns: true,
    });
    expect(compile.mock.invocationCallOrder[0]).toBeLessThan(
      detectAffectiveSignal.mock.invocationCallOrder[0]!,
    );
    expect(detectAffectiveSignal).toHaveBeenCalledWith(
      "plain lower text",
      ["user: prior question", "assistant: prior answer"],
      expect.objectContaining({
        model: "background-model",
        llmEnabled: false,
      }),
    );
    expect(tracer.emit).toHaveBeenCalledWith("recency.completed", {
      turnId: "turn-1",
      session_id: DEFAULT_SESSION_ID,
      messageCount: 2,
      sourceEntryIds: ["strm_abcdefghijklmnop", "strm_bcdefghijklmnopa"],
    });
    expect(result.recencyWindow).toBe(recencyWindow);
    expect(result.workingMood).toEqual({
      valence: 0.6,
      arousal: 0.2,
      dominant_emotion: "curiosity",
    });
    expect(result.workingMemory).toMatchObject({
      turn_counter: 1,
      hot_entities: [],
      mood: result.workingMood,
      mode: "idle",
      updated_at: 1_000,
    });
  });

  it("replaces hot entities wholesale instead of merging the previous turn's", async () => {
    const gateway = new PerceptionGateway({
      config: makeConfig(false),
      llmFactory: vi.fn(() => {
        throw new Error("should not be called");
      }),
      clock: new ManualClock(1_000),
      tracer: makeTracer(),
      getAffectiveSignalDetector: () =>
        vi.fn(async () => ({
          valence: 0.6,
          arousal: 0.2,
          dominant_emotion: "curiosity" as const,
        })),
      turnContextCompiler: {
        compile: vi.fn(() => makeRecencyWindow()),
      },
      createStreamReader: vi.fn(() => ({}) as StreamReader),
    });

    const result = await gateway
      .beginTurn({
        turnId: "turn-1",
        onHookFailure: vi.fn(),
      })
      .perceive({
        sessionId: DEFAULT_SESSION_ID,
        isSelfAudience: true,
        origin: "user",
        cognitionInput: "plain lower text",
        workingMemory: {
          ...createWorkingMemory(DEFAULT_SESSION_ID, 500),
          hot_entities: ["Tom", "borg.db"],
        },
      });

    // No LLM, so the extractor yields nothing for this message. The prior
    // entities must not survive: hot_entities is a per-turn readout of the
    // current message, not an accumulating register.
    expect(result.workingMemory.hot_entities).toEqual([]);
  });

  it("preserves existing mood for autonomous turns", async () => {
    const existingMood = {
      valence: -0.2,
      arousal: 0.4,
      dominant_emotion: "sadness" as const,
    };
    const gateway = new PerceptionGateway({
      config: makeConfig(false),
      llmFactory: vi.fn(),
      clock: new ManualClock(2_000),
      tracer: {
        enabled: false,
        includePayloads: false,
        emit: vi.fn(),
      },
      getAffectiveSignalDetector: () =>
        vi.fn(async () => ({
          valence: 0.8,
          arousal: 0.1,
          dominant_emotion: "joy" as const,
        })),
      turnContextCompiler: {
        compile: vi.fn(() => ({
          messages: [],
          latest_ts: null,
          total_chars: 0,
        })),
      },
      createStreamReader: vi.fn(() => ({}) as StreamReader),
    });

    const result = await gateway
      .beginTurn({
        turnId: "turn-2",
        onHookFailure: vi.fn(),
      })
      .perceive({
        sessionId: DEFAULT_SESSION_ID,
        isSelfAudience: false,
        origin: "autonomous",
        cognitionInput: "autonomous lower text",
        workingMemory: {
          ...createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
          mood: existingMood,
        },
      });

    expect(result.workingMood).toBe(existingMood);
    expect(result.workingMemory.mood).toBe(existingMood);
  });

  it("falls back when optional perception LLM config is unavailable", async () => {
    const gateway = new PerceptionGateway({
      config: makeConfig(true),
      llmFactory: vi.fn(() => {
        throw new ConfigError("missing llm config");
      }),
      clock: new ManualClock(3_000),
      tracer: {
        enabled: false,
        includePayloads: false,
        emit: vi.fn(),
      },
      getAffectiveSignalDetector: () =>
        vi.fn(async () => ({
          valence: 0,
          arousal: 0,
          dominant_emotion: null,
        })),
      turnContextCompiler: {
        compile: vi.fn(() => ({
          messages: [],
          latest_ts: null,
          total_chars: 0,
        })),
      },
      createStreamReader: vi.fn((sessionId: SessionId) => {
        expect(sessionId).toBe(DEFAULT_SESSION_ID);
        return {} as StreamReader;
      }),
    });

    const result = await gateway
      .beginTurn({
        turnId: "turn-3",
        onHookFailure: vi.fn(),
      })
      .perceive({
        sessionId: DEFAULT_SESSION_ID,
        isSelfAudience: false,
        cognitionInput: "lowercase fallback",
        workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      });

    expect(result.perception.mode).toBe("idle");
  });

  describe("when the mode classifier fails", () => {
    function refusingModeLlm(): FakeLLMClient {
      const respond = (options: LLMCompleteOptions) => {
        const toolName = options.tools?.[0]?.name ?? "";

        if (toolName === "EmitModeDetection") {
          throw new LLMError("Anthropic refused the request");
        }

        return {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use" as const,
          tool_calls: [
            {
              id: "toolu_other",
              name: toolName,
              input:
                toolName === "EmitEntityExtraction"
                  ? { entities: [], user_identity_names: [] }
                  : { has_cue: false },
            },
          ],
        };
      };

      return new FakeLLMClient({ responses: [respond, respond, respond] });
    }

    async function perceiveWithPreviousMode(previousMode: CognitiveMode | null) {
      const tracer = makeTracer();
      const onHookFailure = vi.fn(async () => undefined);
      const llm = refusingModeLlm();
      const gateway = new PerceptionGateway({
        config: makeConfig(true),
        llmFactory: () => llm,
        clock: new ManualClock(4_000),
        tracer,
        getAffectiveSignalDetector: () =>
          vi.fn(async () => ({
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          })),
        turnContextCompiler: {
          compile: vi.fn(() => makeRecencyWindow()),
        },
        createStreamReader: vi.fn(() => ({}) as StreamReader),
      });

      const result = await gateway
        .beginTurn({ turnId: "turn-4", onHookFailure })
        .perceive({
          sessionId: DEFAULT_SESSION_ID,
          isSelfAudience: false,
          origin: "user",
          cognitionInput: "a substantive technical message",
          workingMemory: {
            ...createWorkingMemory(DEFAULT_SESSION_ID, 3_000),
            mode: previousMode,
          },
        });

      return { result, tracer, onHookFailure };
    }

    it("carries the session's last perceived mode instead of the idle placeholder", async () => {
      const { result, tracer, onHookFailure } = await perceiveWithPreviousMode("problem_solving");

      expect(result.perception.modeDegraded).toBe(true);
      expect(result.perception.mode).toBe("problem_solving");
      expect(result.perception.isOperational).toBe(false);
      expect(result.workingMemory.mode).toBe("problem_solving");
      // The boundary the placeholder crossed: idle caps episodic retrieval at one.
      expect(computeRetrievalLimit(result.perception.mode)).toBe(6);
      expect(tracer.emit).toHaveBeenCalledWith("perception.classifier.degraded", {
        turnId: "turn-4",
        session_id: DEFAULT_SESSION_ID,
        classifier: "mode_detector",
        reason: "classifier_error",
      });
      expect(onHookFailure).toHaveBeenCalledWith("perception_classifier", expect.any(LLMError), {
        classifier: "mode_detector",
      });
    });

    it("keeps the idle placeholder, still marked degraded, when no earlier mode exists", async () => {
      const { result } = await perceiveWithPreviousMode(null);

      expect(result.perception.modeDegraded).toBe(true);
      expect(result.perception.mode).toBe("idle");
      expect(result.workingMemory.mode).toBe("idle");
    });
  });
});
