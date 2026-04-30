import { describe, expect, it } from "vitest";
import { z } from "zod";

import { ENTITY_FALLBACK_TOOL } from "../cognition/perception/entity-extractor.js";
import { MODE_FALLBACK_TOOL } from "../cognition/perception/mode-detector.js";
import { AFFECTIVE_FALLBACK_TOOL } from "../memory/affective/extractor.js";
import { EXTRACT_EPISODES_TOOL } from "../memory/episodic/extractor.js";
import { EXTRACT_SEMANTIC_TOOL } from "../memory/semantic/extractor.js";
import { MERGE_TOOL } from "../offline/consolidator/index.js";
import { OVERSEER_TOOL } from "../offline/overseer/index.js";
import { REFLECTOR_TOOL } from "../offline/reflector/index.js";
import { RUMINATOR_TOOL } from "../offline/ruminator/index.js";
import { SELF_NARRATOR_TOOL } from "../offline/self-narrator/index.js";

type JsonObjectSchema = {
  type: "object";
  properties: Record<string, unknown>;
  required?: string[];
};

function expectObjectSchema(schema: unknown): asserts schema is JsonObjectSchema {
  expect(schema).toMatchObject({
    type: "object",
  });
  expect(schema).toHaveProperty("properties");
  expect(Object.keys((schema as JsonObjectSchema).properties)).not.toHaveLength(0);
}

describe("tool schemas", () => {
  it("supports z.toJSONSchema in this zod version", () => {
    const schema = z.object({
      mode: z.enum(["reflective", "idle"]),
      tags: z.array(z.string()),
    });

    expect(z.toJSONSchema(schema)).toMatchObject({
      type: "object",
      properties: {
        mode: {
          type: "string",
          enum: ["reflective", "idle"],
        },
        tags: {
          type: "array",
        },
      },
      required: ["mode", "tags"],
    });
  });

  it("derives concrete object schemas for every structured tool", () => {
    const cases = [
      { tool: EXTRACT_EPISODES_TOOL, required: ["episodes"] },
      { tool: EXTRACT_SEMANTIC_TOOL, required: ["nodes", "edges"] },
      { tool: MERGE_TOOL, required: ["title", "narrative"] },
      { tool: OVERSEER_TOOL, required: ["flags"] },
      {
        tool: REFLECTOR_TOOL,
        required: ["label", "description", "confidence", "source_episode_ids"],
      },
      { tool: RUMINATOR_TOOL, required: ["resolution_note", "growth_marker"] },
      { tool: SELF_NARRATOR_TOOL, required: ["observations"] },
      { tool: ENTITY_FALLBACK_TOOL, required: ["entities"] },
      { tool: MODE_FALLBACK_TOOL, required: ["mode"] },
      { tool: AFFECTIVE_FALLBACK_TOOL, required: ["valence", "arousal", "dominant_emotion"] },
    ] as const;

    for (const { tool, required } of cases) {
      expectObjectSchema(tool.inputSchema);
      expect(tool.inputSchema.required ?? []).toEqual(expect.arrayContaining([...required]));
    }
  });

  it("preserves enum constraints in derived schemas", () => {
    expectObjectSchema(MODE_FALLBACK_TOOL.inputSchema);
    expect(MODE_FALLBACK_TOOL.inputSchema.properties.mode).toMatchObject({
      type: "string",
      enum: ["problem_solving", "relational", "reflective", "idle"],
    });

    expectObjectSchema(EXTRACT_SEMANTIC_TOOL.inputSchema);
    const nodesSchema = EXTRACT_SEMANTIC_TOOL.inputSchema.properties.nodes as {
      items?: { properties?: Record<string, unknown> };
    };
    expect(nodesSchema.items?.properties?.kind).toMatchObject({
      type: "string",
      enum: ["concept", "entity", "proposition"],
    });
  });
});
