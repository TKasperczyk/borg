import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { Borg, FakeLLMClient, ManualClock } from "../index.js";
import { TestEmbeddingClient } from "../offline/test-support.js";

describe("autonomy integration", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("runs a full commitment-expiring autonomous tick and records stream entries", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "I should either renew this commitment or let it expire deliberately.",
          input_tokens: 12,
          output_tokens: 8,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_judge",
              name: "EmitCommitmentViolations",
              input: {
                violations: [],
              },
            },
          ],
        },
      ],
    });
    const borg = await Borg.open({
      config: {
        dataDir: tempDir,
        perception: {
          useLlmFallback: false,
          modeWhenLlmAbsent: "idle",
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "test-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "test-cognition",
            background: "test-background",
            extraction: "test-extraction",
          },
        },
        autonomy: {
          enabled: true,
          intervalMs: 60_000,
          maxWakesPerHour: 6,
          triggers: {
            commitmentExpiring: {
              enabled: true,
              lookaheadMs: 86_400_000,
            },
            openQuestionDormant: {
              enabled: false,
              dormantMs: 604_800_000,
            },
            scheduledReflection: {
              enabled: false,
              intervalMs: 14_400_000,
            },
          },
        },
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new TestEmbeddingClient(),
      llmClient: llm,
    });

    try {
      borg.commitments.add({
        type: "promise",
        directive: "Review the Sprint 10 autonomy substrate",
        priority: 8,
        provenance: {
          kind: "manual",
        },
        expiresAt: clock.now() + 10_000,
      });

      const result = await borg.autonomy.scheduler.tick();
      expect(result.firedEvents).toBe(1);

      const entries = borg.stream.tail(6);
      expect(entries.map((entry) => entry.kind)).toEqual([
        "internal_event",
        "tool_call",
        "tool_result",
        "user_msg",
        "agent_msg",
        "internal_event",
      ]);
      expect(entries[0]?.content).toMatchObject({
        kind: "autonomous_wake",
        trigger: "commitment_expiring",
      });
      expect(entries[1]?.content).toMatchObject({
        tool_name: "tool.commitments.list",
        origin: "autonomous",
      });
      expect(entries[2]?.content).toMatchObject({
        ok: true,
      });
      expect(entries[3]?.audience).toBe("self");
      expect(entries[4]?.audience).toBe("self");
      expect(entries[5]?.content).toMatchObject({
        kind: "autonomous_action",
        trigger: "commitment_expiring",
      });
    } finally {
      await borg.close();
    }
  });
});
