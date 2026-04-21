import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { FixedClock } from "../../util/clock.js";
import { EntityExtractor, extractEntitiesHeuristically } from "./entity-extractor.js";
import { ModeDetector, detectModeHeuristically } from "./mode-detector.js";
import { Perceiver } from "./perceive.js";
import { detectTemporalCue } from "./temporal-cue.js";

describe("perception", () => {
  it("extracts entities with heuristics", () => {
    expect(
      extractEntitiesHeuristically(
        'Talk to @alice about "Project Atlas" with Jane Doe at ACME tomorrow.',
      ),
    ).toEqual(["@alice", "Project Atlas", "Jane Doe", "ACME"]);
  });

  it("detects modes heuristically", () => {
    expect(detectModeHeuristically("pnpm build throws an error trace")).toBe("problem_solving");
    expect(detectModeHeuristically("Thanks, can you help me word this for @sam?")).toBe(
      "relational",
    );
    expect(detectModeHeuristically("Why do I keep avoiding this about myself?")).toBe("reflective");
    expect(detectModeHeuristically("ok")).toBe("idle");
  });

  it("uses llm fallback when heuristics yield nothing", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: JSON.stringify({ entities: ["Atlas"] }),
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: JSON.stringify({ mode: "reflective" }),
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const entityExtractor = new EntityExtractor({
      llmClient: llm,
      model: "haiku",
      useLlmFallback: true,
    });
    const modeDetector = new ModeDetector({
      llmClient: llm,
      model: "haiku",
      useLlmFallback: true,
    });

    expect(await entityExtractor.extractEntities("something vague")).toEqual(["Atlas"]);
    expect(await modeDetector.detectMode("maybe this", [])).toBe("reflective");
  });

  it("detects temporal cues and produces a perception result", async () => {
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const perceiver = new Perceiver({
      useLlmFallback: false,
      clock: new FixedClock(nowMs),
    });
    const perceived = await perceiver.perceive("Jane Doe said yesterday was rough");
    const cue = detectTemporalCue("yesterday", nowMs);

    expect(perceived.entities).toContain("Jane Doe");
    expect(perceived.temporalCue).toEqual(cue);
    expect(perceived.mode).toBe("idle");
  });
});
