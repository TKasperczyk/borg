import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { FixedClock } from "../../util/clock.js";
import { EntityExtractor, extractEntitiesHeuristically } from "./entity-extractor.js";
import { ModeDetector } from "./mode-detector.js";
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

  it("produces a perception result with null temporal cue when no LLM is configured", async () => {
    // Previously this module had a hardcoded "yesterday" -> 24h-window
    // pattern. With the heuristic tier removed, temporal extraction is
    // LLM-only; without an LLM client the cue is null and retrieval
    // simply doesn't get a time filter -- which is the safe default.
    const nowMs = new Date("2026-04-21T12:00:00Z").getTime();
    const perceiver = new Perceiver({
      useLlmFallback: false,
      clock: new FixedClock(nowMs),
    });
    const perceived = await perceiver.perceive("Jane Doe said yesterday was rough");

    expect(perceived.entities).toContain("Jane Doe");
    expect(perceived.temporalCue).toBeNull();
    expect(perceived.mode).toBe("idle");
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
