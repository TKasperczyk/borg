import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { createSessionId, type Borg, type SessionId } from "../src/index.js";

import { MetricsCapture } from "./metrics.js";
import type { MetricsRow } from "./types.js";

const tempDirs: string[] = [];

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-simulator-metrics-"));
  tempDirs.push(dir);
  return dir;
}

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function fakeBorg(
  counts: {
    semanticNodes?: number;
    semanticEdges?: number;
    suppressedSessions?: readonly SessionId[];
  } = {},
  observed: { moodSessions?: SessionId[]; tailSessions?: SessionId[] } = {},
): Borg {
  const semanticNodeCount = counts.semanticNodes ?? 1;
  const semanticEdgeCount = counts.semanticEdges ?? 2;
  const suppressedSessions = new Set(counts.suppressedSessions ?? []);

  return {
    mood: {
      current: (sessionId: SessionId) => {
        observed.moodSessions?.push(sessionId);
        return { valence: -0.2, arousal: 0.4 };
      },
    },
    episodic: {
      list: async () => ({ items: [{ id: "episode_1" }, { id: "episode_2" }] }),
    },
    semantic: {
      nodes: {
        list: async () =>
          Array.from({ length: semanticNodeCount }, (_, index) => ({ id: `node_${index}` })),
      },
      edges: {
        list: () =>
          Array.from({ length: semanticEdgeCount }, (_, index) => ({ id: `edge_${index}` })),
      },
    },
    self: {
      openQuestions: {
        list: () => [{ id: "question_1" }],
      },
      goals: {
        list: () => [{ id: "goal_1", children: [{ id: "goal_2" }] }],
      },
    },
    stream: {
      tail: (_limit: number, options?: { session?: SessionId }) => {
        if (options?.session !== undefined) {
          observed.tailSessions?.push(options.session);
        }

        return options?.session !== undefined && suppressedSessions.has(options.session)
          ? [{ kind: "agent_suppressed" }]
          : [];
      },
    },
  } as unknown as Borg;
}

describe("MetricsCapture", () => {
  it("captures Borg state, trace latency, and token usage to JSONL", async () => {
    const dir = tempDir();
    const tracePath = join(dir, "trace.jsonl");
    const metricsPath = join(dir, "metrics.jsonl");

    writeFileSync(
      tracePath,
      [
        { ts: 100, turnId: "turn-1", event: "retrieval_started" },
        { ts: 125, turnId: "turn-1", event: "retrieval_completed" },
        { ts: 130, turnId: "turn-1", event: "llm_call_started" },
        {
          ts: 190,
          turnId: "turn-1",
          event: "llm_call_response",
          usage: { inputTokens: 11, outputTokens: 7 },
        },
      ]
        .map((record) => JSON.stringify(record))
        .join("\n"),
    );

    const sessionId = createSessionId();
    const otherSessionId = createSessionId();
    const observed: { moodSessions: SessionId[]; tailSessions: SessionId[] } = {
      moodSessions: [],
      tailSessions: [],
    };
    const capture = new MetricsCapture(metricsPath, { tracePath });
    const row = await capture.capture(
      fakeBorg({ suppressedSessions: [otherSessionId] }, observed),
      "turn-1",
      3,
      {
        sessionId,
        sessionIds: [sessionId, otherSessionId],
      },
    );
    const written = JSON.parse(readFileSync(metricsPath, "utf8").trim()) as MetricsRow;

    expect(row.turn_counter).toBe(3);
    expect(row.episode_count).toBe(2);
    expect(row.semantic_node_count).toBe(1);
    expect(row.semantic_edge_count).toBe(2);
    expect(row.semantic_nodes_added_since_last_check).toBe(0);
    expect(row.semantic_edges_added_since_last_check).toBe(0);
    expect(row.open_question_count).toBe(1);
    expect(row.active_goal_count).toBe(2);
    expect(row.generation_suppression_count).toBe(1);
    expect(row.retrieval_latency_ms).toBe(25);
    expect(row.deliberation_latency_ms).toBe(60);
    expect(row.borg_input_tokens).toBe(11);
    expect(row.borg_output_tokens).toBe(7);
    expect(observed.moodSessions).toEqual([sessionId]);
    expect(observed.tailSessions).toEqual([sessionId, otherSessionId]);
    expect(written).toEqual(row);
  });

  it("records semantic graph growth since the previous capture", async () => {
    const dir = tempDir();
    const metricsPath = join(dir, "metrics.jsonl");
    const capture = new MetricsCapture(metricsPath);
    const sessionId = createSessionId();

    await capture.capture(fakeBorg({ semanticNodes: 1, semanticEdges: 2 }), "turn-1", 1, {
      sessionId,
      sessionIds: [sessionId],
    });
    const row = await capture.capture(
      fakeBorg({ semanticNodes: 4, semanticEdges: 5 }),
      "turn-2",
      2,
      {
        sessionId,
        sessionIds: [sessionId],
      },
    );

    expect(row.semantic_nodes_added_since_last_check).toBe(3);
    expect(row.semantic_edges_added_since_last_check).toBe(3);
  });
});
