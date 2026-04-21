import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { FixedClock } from "../../util/clock.js";
import { EntityExtractor, extractEntitiesHeuristically } from "./entity-extractor.js";
import { ModeDetector, detectModeHeuristically } from "./mode-detector.js";
import { Perceiver } from "./perceive.js";
import { detectTemporalCue } from "./temporal-cue.js";

const ENTITY_TOOL_NAME = "EmitEntityExtraction";
const MODE_TOOL_NAME = "EmitModeDetection";

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
      useLlmFallback: true,
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
