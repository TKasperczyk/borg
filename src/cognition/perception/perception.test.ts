import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { FixedClock } from "../../util/clock.js";
import { EntityExtractor } from "./entity-extractor.js";
import { ModeDetector } from "./mode-detector.js";
import { Perceiver, runPerceptionClassifierSafely } from "./perceive.js";
import { detectTemporalCue } from "./temporal-cue.js";

const ENTITY_TOOL_NAME = "EmitEntityExtraction";
const MODE_TOOL_NAME = "EmitModeDetection";

function invalidEntityResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_entity",
        name: ENTITY_TOOL_NAME,
        input: { entities: [1] },
      },
    ],
  };
}

function invalidModeResponse() {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_mode",
        name: MODE_TOOL_NAME,
        input: { mode: "unknown" },
      },
    ],
  };
}

function entityResponse(entities: readonly string[]) {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_entity",
        name: ENTITY_TOOL_NAME,
        input: { entities },
      },
    ],
  };
}

function modeResponse(mode: string) {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_mode",
        name: MODE_TOOL_NAME,
        input: { mode },
      },
    ],
  };
}

describe("perception", () => {
  it("returns classifier results without notifying on the safe-wrapper success path", async () => {
    const onFailure = vi.fn();

    const result = await runPerceptionClassifierSafely({
      classifier: "mode_detector",
      run: async () => "reflective",
      fallback: () => "idle",
      onFailure,
    });

    expect(result).toBe("reflective");
    expect(onFailure).not.toHaveBeenCalled();
  });

  it("returns the fallback and notifies on the safe-wrapper failure path", async () => {
    const error = new Error("classifier exploded");
    const onFailure = vi.fn();

    const result = await runPerceptionClassifierSafely({
      classifier: "entity_extractor",
      run: async () => {
        throw error;
      },
      fallback: () => ["Atlas"],
      onFailure,
    });

    expect(result).toEqual(["Atlas"]);
    expect(onFailure).toHaveBeenCalledWith({
      classifier: "entity_extractor",
      error,
    });
  });

  it("returns empty entities when no LLM client is configured", async () => {
    // Perception entity extraction is LLM-only: the previous regex
    // heuristic produced false-positive entities at high rates
    // ('Good', 'If', '[End.]') that poisoned downstream retrieval.
    // Without an LLM the honest signal is empty.
    const extractor = new EntityExtractor();

    expect(await extractor.extractEntities("Jane Doe said yesterday was rough")).toEqual([]);
  });

  it("keeps language-neutral entity sanitizer checks only", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_entity",
              name: ENTITY_TOOL_NAME,
              input: {
                entities: ["李", "the", "Human:", "[end]", "!!!"],
              },
            },
          ],
        },
      ],
    });
    const extractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });

    expect(await extractor.extractEntities("irrelevant")).toEqual(["李", "the", "Human:", "[end]"]);
  });

  it("defaults to idle when no LLM client is configured", async () => {
    // The heuristic tier was removed; without an LLM, the safe neutral
    // default is "idle" (skips S2 planning, uses default retrieval weights).
    const detector = new ModeDetector({ useLlmFallback: false });
    expect(await detector.detectMode("pnpm build throws an error trace")).toBe("idle");
    expect(await detector.detectMode("Why do I keep avoiding this?")).toBe("idle");
    expect(await detector.detectMode("ok")).toBe("idle");
  });

  it("classifies every message via the LLM when configured", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: ENTITY_TOOL_NAME,
              input: { entities: ["Atlas"] },
            },
          ],
        },
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_2",
              name: MODE_TOOL_NAME,
              input: { mode: "reflective" },
            },
          ],
        },
      ],
    });
    const entityExtractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });
    const modeDetector = new ModeDetector({
      llmClient: llm,
      model: "haiku",
      useLlmFallback: true,
    });

    expect(await entityExtractor.extractEntities("something vague")).toEqual(["Atlas"]);
    expect(await modeDetector.detectMode("maybe this", [])).toBe("reflective");
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: ENTITY_TOOL_NAME,
    });
    expect(llm.requests[1]?.tool_choice).toEqual({
      type: "tool",
      name: MODE_TOOL_NAME,
    });
  });

  it("runs the entity fallback for long zero-hit text and truncates the prompt", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: ENTITY_TOOL_NAME,
              input: { entities: ["pgvector", "qdrant"] },
            },
          ],
        },
      ],
    });
    const extractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
      shortTextThreshold: 40,
    });
    const longLowercaseText = `${"pgvector qdrant ".repeat(180)}borg memory index drift`;

    const entities = await extractor.extractEntities(longLowercaseText);

    expect(entities).toEqual(["pgvector", "qdrant"]);
    expect(String(llm.requests[0]?.messages[0]?.content ?? "").length).toBeLessThanOrEqual(2_000);
  });

  it("returns the LLM entity payload after output sanitization", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_1",
              name: ENTITY_TOOL_NAME,
              input: { entities: ["Sam", "bicycles"] },
            },
          ],
        },
      ],
    });
    const extractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
    });

    expect(await extractor.extractEntities("yesterday Sam mentioned bicycles")).toEqual([
      "Sam",
      "bicycles",
    ]);
    expect(llm.requests).toHaveLength(1);
  });

  it("falls back to the configured neutral mode when mode detection throws", async () => {
    const onClassifierFailure = vi.fn();
    const llm = new FakeLLMClient({
      responses: [entityResponse(["Atlas"]), invalidModeResponse()],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "haiku",
      affectiveUseLlmFallback: false,
      temporalCueUseLlmFallback: false,
      modeWhenLlmAbsent: "relational",
      onClassifierFailure,
    });

    const perceived = await perceiver.perceive("plain lower text");

    expect(perceived.mode).toBe("relational");
    expect(perceived.entities).toEqual(["Atlas"]);
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "mode_detector",
        error: expect.any(Error),
      }),
    );
  });

  it("falls back to empty entities when entity extraction throws", async () => {
    // The previous regex-heuristic fallback was removed -- it
    // produced false-positive entities at high rates that poisoned
    // downstream retrieval. Empty entities is the honest signal on
    // failure. The turn proceeds; mode and other classifiers run
    // independently per Promise.all.
    const onClassifierFailure = vi.fn();
    const llm = new FakeLLMClient({
      responses: [invalidEntityResponse(), modeResponse("problem_solving")],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "haiku",
      affectiveUseLlmFallback: false,
      temporalCueUseLlmFallback: false,
      onClassifierFailure,
    });

    const perceived = await perceiver.perceive(
      'Talk to @alice about "Project Atlas" with Jane Doe.',
    );

    expect(perceived.entities).toEqual([]);
    expect(perceived.mode).toBe("problem_solving");
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "entity_extractor",
        error: expect.any(Error),
      }),
    );
  });

  it("degrades mode and entities independently when both classifiers throw", async () => {
    const onClassifierFailure = vi.fn();
    const llm = new FakeLLMClient({
      responses: [invalidEntityResponse(), invalidModeResponse()],
    });
    const perceiver = new Perceiver({
      llmClient: llm,
      model: "haiku",
      affectiveUseLlmFallback: false,
      temporalCueUseLlmFallback: false,
      modeWhenLlmAbsent: "idle",
      onClassifierFailure,
    });

    const perceived = await perceiver.perceive('Meet @alice about "Project Atlas".');

    expect(perceived.entities).toEqual([]);
    expect(perceived.mode).toBe("idle");
    expect(onClassifierFailure).toHaveBeenCalledTimes(2);
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "entity_extractor",
        error: expect.any(Error),
      }),
    );
    expect(onClassifierFailure).toHaveBeenCalledWith(
      expect.objectContaining({
        classifier: "mode_detector",
        error: expect.any(Error),
      }),
    );
  });

  it("keeps degraded perception structurally identical to successful perception", async () => {
    const successful = await new Perceiver({
      llmClient: new FakeLLMClient({
        responses: [entityResponse(["Atlas"]), modeResponse("reflective")],
      }),
      model: "haiku",
      affectiveUseLlmFallback: false,
      temporalCueUseLlmFallback: false,
    }).perceive("plain lower text");
    const degraded = await new Perceiver({
      llmClient: new FakeLLMClient({
        responses: [invalidEntityResponse(), invalidModeResponse()],
      }),
      model: "haiku",
      affectiveUseLlmFallback: false,
      temporalCueUseLlmFallback: false,
      modeWhenLlmAbsent: "idle",
    }).perceive("plain lower text");

    expect(Object.keys(degraded).sort()).toEqual(Object.keys(successful).sort());
    expect({
      ...degraded,
      entities: successful.entities,
      mode: successful.mode,
    }).toEqual(successful);
  });

  it("produces a perception result with empty entities and null temporal cue when no LLM is configured", async () => {
    // Without an LLM, perception's entity extraction returns []
    // (heuristic was removed -- it produced too many false positives
    // like 'Good', 'If', '[End.]'). Mode degrades to 'idle' and the
    // temporal cue degrades to null. The turn proceeds with an
    // empty perception payload rather than confidently-wrong tags.
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const perceiver = new Perceiver({
      useLlmFallback: false,
      clock: new FixedClock(nowMs),
    });
    const perceived = await perceiver.perceive("Jane Doe said yesterday was rough");

    expect(perceived.entities).toEqual([]);
    expect(perceived.temporalCue).toBeNull();
    expect(perceived.mode).toBe("idle");
    expect(perceived.affectiveSignal).toEqual({
      valence: 0,
      arousal: 0,
      dominant_emotion: null,
    });
    expect(perceived.affectiveSignalDegraded).toBe(true);
  });

  it("extracts a temporal cue via the LLM when one is configured", async () => {
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const sinceTs = nowMs - 24 * 60 * 60 * 1_000;
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_temporal",
              name: "EmitTemporalCue",
              input: {
                has_cue: true,
                since_ts: sinceTs,
                until_ts: nowMs,
                label: "yesterday",
              },
            },
          ],
        },
      ],
    });

    const cue = await detectTemporalCue("Jane said yesterday was rough", nowMs, {
      llmClient: llm,
      model: "haiku",
    });

    expect(cue).toEqual({
      sinceTs,
      untilTs: nowMs,
      label: "yesterday",
    });
  });
});
