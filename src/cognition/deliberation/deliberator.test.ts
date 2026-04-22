import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import {
  CommitmentRepository,
  EntityRepository,
  commitmentMigrations,
} from "../../memory/commitments/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import { Deliberator } from "./deliberator.js";

function makeRetrievedEpisode(
  id: string,
  score: number,
  tags: string[] = [],
  semanticContext: RetrievedEpisode["semantic_context"] = {
    supports: [],
    contradicts: [],
    categories: [],
  },
): RetrievedEpisode {
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
      moodBoost: 0,
      socialRelevance: 0,
      suppressionPenalty: 0,
    },
    citationChain: [],
    semantic_context: semanticContext,
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
    expect(llm.requests[1]?.system ?? llm.requests[0]?.system).not.toContain("Skill you might try");
  });

  it("includes related semantic context in the Sonnet prompt", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Context aware answer",
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
      userMessage: "What should I know about Atlas?",
      perception: {
        entities: ["Atlas"],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [
        makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.92, ["atlas"], {
          supports: [
            {
              id: "semn_aaaaaaaaaaaaaaaa" as RetrievedEpisode["semantic_context"]["supports"][number]["id"],
              kind: "concept",
              label: "Atlas Service",
              description: "Primary deployment service",
              aliases: [],
              confidence: 0.72,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
              created_at: 0,
              updated_at: 0,
              last_verified_at: 0,
              embedding: Float32Array.from([1, 0, 0, 0]),
              archived: false,
              superseded_by: null,
            },
          ],
          contradicts: [
            {
              id: "semn_bbbbbbbbbbbbbbbb" as RetrievedEpisode["semantic_context"]["contradicts"][number]["id"],
              kind: "proposition",
              label: "Atlas is stable",
              description: "Claimed stable despite deploy errors",
              aliases: [],
              confidence: 0.61,
              source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
              created_at: 0,
              updated_at: 0,
              last_verified_at: 0,
              embedding: Float32Array.from([1, 0, 0, 0]),
              archived: false,
              superseded_by: null,
            },
          ],
          categories: [],
        }),
      ],
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
    expect(llm.requests[0]?.system).toContain("Related semantic context:");
    expect(llm.requests[0]?.system).toContain(
      "supports: Atlas Service - Primary deployment service (conf 0.72)",
    );
    expect(llm.requests[0]?.system).toContain(
      "contradicts: Atlas is stable - Claimed stable despite deploy errors (conf 0.61)",
    );
    // Identity framing must affirm the being positively and flag block 1 as
    // OAuth plumbing without naming-to-reject (pink-elephant pattern).
    const system = llm.requests[0]?.system as string;
    expect(system).toMatch(/AI being/);
    expect(system).toMatch(/OAuth authentication context/);
    // No pre-assigned name; host (or being over time) fills that in.
    expect(system).not.toMatch(/You are Borg\b/);
  });

  it("includes skill guidance only for problem-solving mode when a candidate exists", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Skill-aware answer",
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
    const selectedSkill = {
      skill: {
        id: "skl_aaaaaaaaaaaaaaaa" as never,
        applies_when: "Rust lifetime debugging",
        approach: "Shrink borrow scopes.",
        alpha: 4,
        beta: 2,
        attempts: 4,
        successes: 3,
        failures: 1,
        alternatives: [],
        source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
        last_used: null,
        last_successful: null,
        created_at: 0,
        updated_at: 0,
      },
      sampledValue: 0.82,
      evaluatedCandidates: [
        {
          skill: {
            id: "skl_aaaaaaaaaaaaaaaa" as never,
            applies_when: "Rust lifetime debugging",
            approach: "Shrink borrow scopes.",
            alpha: 4,
            beta: 2,
            attempts: 4,
            successes: 3,
            failures: 1,
            alternatives: [],
            source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
            last_used: null,
            last_successful: null,
            created_at: 0,
            updated_at: 0,
          },
          similarity: 0.9,
          stats: {
            mean: 0.67,
            mode: 0.75,
            ci_95: [0.4, 0.9] as [number, number],
          },
          sampledValue: 0.82,
        },
      ],
    };

    await deliberator.run({
      sessionId: DEFAULT_SESSION_ID,
      userMessage: "Help with Rust lifetimes",
      perception: {
        entities: ["Rust"],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.9)],
      selectedSkill,
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

    expect(llm.requests[0]?.system).toContain("### Skill you might try");
    expect(llm.requests[0]?.system).toContain("Applies when: Rust lifetime debugging");
    expect(llm.requests[0]?.system).toContain("Approach: Shrink borrow scopes.");
  });

  it("omits the skill section when problem-solving mode has no matching skill", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "No-skill answer",
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

    await deliberator.run({
      sessionId: DEFAULT_SESSION_ID,
      userMessage: "Help with Atlas deploys",
      perception: {
        entities: ["Atlas"],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.9)],
      selectedSkill: null,
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

    expect(llm.requests[0]?.system).not.toContain("Skill you might try");
  });

  it("includes reflective open questions in the prompt", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Think through the uncertainty first.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Reflective answer with open questions in view.",
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

    await deliberator.run({
      sessionId: DEFAULT_SESSION_ID,
      userMessage: "What am I still missing about Atlas?",
      perception: {
        entities: ["Atlas"],
        mode: "reflective",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.8, ["atlas"])],
      openQuestionsContext: [
        {
          id: "oq_aaaaaaaaaaaaaaaa" as never,
          question: "Why does Atlas fail after rollback?",
          urgency: 0.7,
          status: "open",
          related_episode_ids: [],
          related_semantic_node_ids: [],
          source: "reflection",
          created_at: 0,
          last_touched: 0,
          resolution_episode_id: null,
          resolution_note: null,
          resolved_at: null,
          abandoned_reason: null,
          abandoned_at: null,
        },
      ],
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 1,
        scratchpad: "",
        current_focus: "Atlas",
        recent_thoughts: [],
        hot_entities: ["Atlas"],
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

    expect(llm.requests[1]?.system).toContain("Open questions you're carrying:");
    expect(llm.requests[1]?.system).toContain("Why does Atlas fail after rollback?");
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

  it("chooses system 2 when retrieval reports a contradiction even without lexical hints", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Compare the conflicting retrieved facts before answering.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Resolved contradiction answer",
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
      userMessage: "Summarize the deployment guidance.",
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.95)],
      contradictionPresent: true,
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

  it("injects applicable commitments into the system prompt", async () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const clock = new FixedClock(1_000);
    const entities = new EntityRepository({
      db,
      clock,
    });
    const commitments = new CommitmentRepository({
      db,
      clock,
    });
    const sam = entities.resolve("Sam");
    const atlas = entities.resolve("Atlas");
    const commitment = commitments.add({
      type: "boundary",
      directive: "Do not discuss Atlas with Sam",
      priority: 9,
      restrictedAudience: sam,
      aboutEntity: atlas,
    });
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Boundaried answer",
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

    try {
      const result = await deliberator.run({
        sessionId: DEFAULT_SESSION_ID,
        userMessage: "Can you update Sam about Atlas?",
        audience: "Sam",
        perception: {
          entities: ["Atlas", "Sam"],
          mode: "problem_solving",
          affectiveSignal: { valence: 0, arousal: 0 },
          temporalCue: null,
        },
        retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.95)],
        applicableCommitments: [commitment],
        entityRepository: entities,
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
      expect(llm.requests[0]?.system).toContain("Commitments you made to this person:");
      expect(llm.requests[0]?.system).toContain("Do not discuss Atlas with Sam");
      expect(llm.requests[0]?.system).toContain("audience=Sam");
      expect(llm.requests[0]?.system).toContain("about=Atlas");
    } finally {
      db.close();
    }
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
