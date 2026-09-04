import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../src/llm/test-support/fake-client.js";

import { JsonlValueCache } from "../embedding-ab/cache.js";
import type { EpisodeDocument } from "../embedding-ab/types.js";
import {
  generateRecallPlannerCases,
  parseCachedGeneratedCase,
  type CachedGeneratedCase,
} from "./llm-tasks.js";

describe("recall planner generated cases", () => {
  const directories: string[] = [];

  afterEach(() => {
    for (const directory of directories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("creates an exactly two-turn Polish referential case and reuses its scratch cache", async () => {
    const outDir = mkdtempSync(join(tmpdir(), "borg-recall-case-generation-"));
    directories.push(outDir);
    const episode: EpisodeDocument = {
      id: "ep_aaaaaaaaaaaaaaaa",
      title: "Decyzja wdrożeniowa Atlas",
      narrative: "Maja Chen porównała dwa warianty i wybrała canary rollback.",
      tags: ["Atlas", "rollback"],
      embedding_text:
        "Decyzja wdrożeniowa Atlas\nMaja Chen porównała dwa warianty.\nAtlas rollback",
      embedding_text_sha256: "episode-content-hash",
    };
    const client = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 10,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_generated_case",
              name: "EmitPolishReferentialRecallCase",
              input: {
                context_turns: [
                  {
                    role: "user",
                    content: "Przypomnij, co Maja mówiła o rollbacku Atlasa.",
                  },
                  {
                    role: "assistant",
                    content: "Maja porównała dwa warianty rollbacku Atlasa.",
                  },
                ],
                focus: "I który z nich wybrała?",
                notes: "Zaimek wymaga obu poprzednich wypowiedzi.",
              },
            },
          ],
        },
      ],
    });
    const cache = new JsonlValueCache<CachedGeneratedCase>(
      join(outDir, "cache", "generated-cases.jsonl"),
      parseCachedGeneratedCase,
      join(outDir, "cache"),
    );

    const first = await generateRecallPlannerCases({
      episodes: [episode],
      count: 1,
      memoryOwnerName: "team-agent",
      llmClient: client,
      model: "fake-generator",
      cache,
    });
    const second = await generateRecallPlannerCases({
      episodes: [episode],
      count: 1,
      memoryOwnerName: "team-agent",
      llmClient: client,
      model: "fake-generator",
      cache,
    });

    expect(first).toEqual([
      expect.objectContaining({
        source_episode_id: episode.id,
        cache_hit: false,
        case: expect.objectContaining({
          id: `generated-${episode.id}`,
          focus: "I który z nich wybrała?",
          context_turns: [
            expect.objectContaining({ role: "user" }),
            expect.objectContaining({ role: "assistant" }),
          ],
          identity: { memory_owner_name: "team-agent" },
          expected_episode_ids: [episode.id],
        }),
      }),
    ]);
    expect(second[0]).toMatchObject({ cache_hit: true, case: first[0]?.case });
    expect(client.requests).toHaveLength(1);
    expect(client.requests[0]?.messages[0]?.content).toContain("episode_json=");
  });
});
