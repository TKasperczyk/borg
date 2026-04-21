import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import { Deliberator } from "./deliberator.js";

function makeRetrievedEpisode(id: string, score: number, tags: string[] = []): RetrievedEpisode {
  return {
    episode: {
      id: id as RetrievedEpisode["episode"]["id"],
      title: `${id} title`,
      narrative: `${id} narrative`,
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: [
        "strm_aaaaaaaaaaaaaaaa" as RetrievedEpisode["episode"]["source_stream_ids"][number],
      ],
      significance: 0.8,
      tags,
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    },
    score,
    scoreBreakdown: {
      similarity: score,
      decayedSalience: 0.3,
      heat: 1,
      goalRelevance: 0,
      timeRelevance: 0,
      suppressionPenalty: 0,
    },
    citationChain: [],
  };
}

describe("deliberator", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("chooses system 1 when confidence is high and stakes are low", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Direct answer",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const deliberator = new Deliberator({
      llmClient: llm,
      cognitionModel: "sonnet",
      backgroundModel: "haiku",
    });

    const result = await deliberator.run({
      sessionId: DEFAULT_SESSION_ID,
      userMessage: "Help with Project Atlas",
      perception: {
        entities: ["Project Atlas"],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.9)],
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 1,
        scratchpad: "",
        current_focus: null,
        recent_thoughts: [],
        hot_entities: [],
        pending_intents: [],
        suppressed: [],
        mode: "problem_solving",
        updated_at: 0,
      },
      selfSnapshot: {
        values: [],
        goals: [],
        traits: [],
      },
      options: {
        stakes: "low",
      },
    });

    expect(result.path).toBe("system_1");
    expect(result.response).toBe("Direct answer");
    expect(result.thoughts).toEqual([]);
  });

  it("chooses system 2 for reflective mode even with high confidence", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Think about what this reveals about my current pattern.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Reflective answer",
          input_tokens: 12,
          output_tokens: 6,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const deliberator = new Deliberator({
      llmClient: llm,
      cognitionModel: "sonnet",
      backgroundModel: "haiku",
    });

    const result = await deliberator.run({
      sessionId: DEFAULT_SESSION_ID,
      userMessage: "What does this say about me?",
      perception: {
        entities: [],
        mode: "reflective",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.95)],
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 1,
        scratchpad: "",
        current_focus: null,
        recent_thoughts: [],
        hot_entities: [],
        pending_intents: [],
        suppressed: [],
        mode: "reflective",
        updated_at: 0,
      },
      selfSnapshot: {
        values: [],
        goals: [],
        traits: [],
      },
      options: {
        stakes: "low",
      },
    });

    expect(result.path).toBe("system_2");
    expect(result.decision_reason).toContain("Reflective mode");
  });

  it("chooses system 2 when contradiction language appears even at low stakes", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Check the contradiction before answering.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Resolved answer",
          input_tokens: 12,
          output_tokens: 6,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const deliberator = new Deliberator({
      llmClient: llm,
      cognitionModel: "sonnet",
      backgroundModel: "haiku",
    });

    const result = await deliberator.run({
      sessionId: DEFAULT_SESSION_ID,
      userMessage: "This looked fine, but the deploy still failed.",
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.95)],
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 1,
        scratchpad: "",
        current_focus: null,
        recent_thoughts: [],
        hot_entities: [],
        pending_intents: [],
        suppressed: [],
        mode: "problem_solving",
        updated_at: 0,
      },
      selfSnapshot: {
        values: [],
        goals: [],
        traits: [],
      },
      options: {
        stakes: "low",
      },
    });

    expect(result.path).toBe("system_2");
    expect(result.decision_reason).toContain("Contradiction heuristic");
  });

  it("chooses system 2 for high stakes and persists scratchpad thoughts", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Check the failure mode first.",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Careful answer. Next step: rerun the deploy.",
          input_tokens: 20,
          output_tokens: 10,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const deliberator = new Deliberator({
      llmClient: llm,
      cognitionModel: "sonnet",
      backgroundModel: "haiku",
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock: new FixedClock(100),
    });

    try {
      const result = await deliberator.run(
        {
          sessionId: DEFAULT_SESSION_ID,
          userMessage: "High stakes deployment problem",
          perception: {
            entities: [],
            mode: "problem_solving",
            affectiveSignal: { valence: 0, arousal: 0 },
            temporalCue: null,
          },
          retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.2, ["warning"])],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            scratchpad: "",
            current_focus: null,
            recent_thoughts: [],
            hot_entities: [],
            pending_intents: [],
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
          selfSnapshot: {
            values: [],
            goals: [],
            traits: [],
          },
          options: {
            stakes: "high",
          },
          reRetrieve: async () => [makeRetrievedEpisode("ep_bbbbbbbbbbbbbbbb", 0.7)],
        },
        writer,
      );
      const reader = new StreamReader({
        dataDir: tempDir,
        sessionId: DEFAULT_SESSION_ID,
      });
      const thoughtEntries = reader.tail(1);

      expect(result.path).toBe("system_2");
      expect(result.thoughts).toEqual(["Check the failure mode first."]);
      expect(result.thoughtsPersisted).toBe(true);
      expect(result.usage.input_tokens).toBe(30);
      expect(result.retrievedEpisodes.map((episode) => episode.episode.id)).toEqual([
        "ep_aaaaaaaaaaaaaaaa",
        "ep_bbbbbbbbbbbbbbbb",
      ]);
      expect(thoughtEntries[0]?.kind).toBe("thought");
    } finally {
      writer.close();
    }
  });
});
