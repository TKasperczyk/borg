import { describe, expect, it, vi } from "vitest";

import type { Config } from "../../config/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import type { StreamReader } from "../../stream/index.js";
import { ConfigError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID, type SessionId, type StreamEntryId } from "../../util/ids.js";
import { ManualClock } from "../../util/clock.js";
import type { RecencyWindow } from "../recency/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { PerceptionGateway } from "./gateway.js";

function makeConfig(useLlmFallback: boolean): Config {
  return {
    perception: {
      useLlmFallback,
      modeWhenLlmAbsent: "idle",
    },
    affective: {
      useLlmFallback: false,
    },
    anthropic: {
      models: {
        background: "background-model",
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
        ts: 900,
      },
      {
        role: "assistant",
        content: "prior answer",
        stream_entry_id: "strm_bcdefghijklmnopa" as StreamEntryId,
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
        useLlmFallback: false,
      }),
    );
    expect(tracer.emit).toHaveBeenCalledWith("recency_compiled", {
      turnId: "turn-1",
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
});
