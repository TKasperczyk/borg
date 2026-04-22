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
      moodBoost: 0,
      socialRelevance: 0,
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

  it("prepends recency messages to the LLM messages array on the S1 path", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Answer after seeing prior turns",
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
      userMessage: "And what about now?",
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.9)],
      recencyMessages: [
        {
          role: "user",
          content: "What's the plan?",
          stream_entry_id: "strm_aaaaaaaaaaaaaaaa" as never,
          ts: 1,
        },
        {
          role: "assistant",
          content: "We rebuild the index first.",
          stream_entry_id: "strm_bbbbbbbbbbbbbbbb" as never,
          ts: 2,
        },
      ],
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 2,
        current_focus: null,
        hot_entities: [],
        pending_intents: [],
        suppressed: [],
        mode: "problem_solving",
        updated_at: 0,
      },
      selfSnapshot: { values: [], goals: [], traits: [] },
      options: { stakes: "low" },
    });

    const messages = llm.requests[0]?.messages;
    expect(messages).toEqual([
      { role: "user", content: "What's the plan?" },
      { role: "assistant", content: "We rebuild the index first." },
      { role: "user", content: "And what about now?" },
    ]);
  });

  it("prepends recency messages on the S2 planner and finalizer", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_1",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "",
                verification_steps: [],
                tensions: [],
                voice_note: "",
              },
            },
          ],
        },
        {
          text: "Final answer that respects earlier turn.",
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
      userMessage: "What does that mean for the rollback plan?",
      perception: {
        entities: [],
        mode: "reflective",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.9)],
      recencyMessages: [
        {
          role: "user",
          content: "We hit a drift in prod.",
          stream_entry_id: "strm_aaaaaaaaaaaaaaaa" as never,
          ts: 1,
        },
        {
          role: "assistant",
          content: "Confirmed -- it's the index order.",
          stream_entry_id: "strm_bbbbbbbbbbbbbbbb" as never,
          ts: 2,
        },
      ],
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 2,
        current_focus: null,
        hot_entities: [],
        pending_intents: [],
        suppressed: [],
        mode: "reflective",
        updated_at: 0,
      },
      selfSnapshot: { values: [], goals: [], traits: [] },
      options: { stakes: "low" },
    });

    // Phase D: planner and finalizer BOTH get the same dialogue messages
    // (recent + current user as plain text). The planner no longer wraps
    // the user turn in a "User message: ..." payload -- it sees the turn
    // directly and emits a structured plan via tool-use instead.
    const expectedDialogue = [
      { role: "user", content: "We hit a drift in prod." },
      { role: "assistant", content: "Confirmed -- it's the index order." },
      { role: "user", content: "What does that mean for the rollback plan?" },
    ];
    expect(llm.requests[0]?.messages).toEqual(expectedDialogue);
    expect(llm.requests[1]?.messages).toEqual(expectedDialogue);

    // Planner pins tool_choice to EmitTurnPlan so the call produces a
    // structured plan, not free-form text. Finalizer has no tool_choice --
    // it emits natural response text.
    expect(llm.requests[0]?.tool_choice).toEqual({ type: "tool", name: "EmitTurnPlan" });
    expect(llm.requests[1]?.tool_choice).toBeUndefined();

    // Both calls share the identity/voice framing so voice lands
    // consistently across plan and response.
    expect(llm.requests[0]?.system).toContain("AI being, not an assistant");
    expect(llm.requests[1]?.system).toContain("AI being, not an assistant");
  });

  it("wires autobiographical period, recent growth, and audience profile into the prompt", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Situated answer",
          input_tokens: 8,
          output_tokens: 4,
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
      userMessage: "What's on your mind?",
      perception: {
        entities: [],
        mode: "problem_solving",
        affectiveSignal: { valence: 0, arousal: 0 },
        temporalCue: null,
      },
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.9)],
      audienceProfile: {
        entity_id: "ent_aaaaaaaaaaaaaaaa" as never,
        trust: 0.82,
        attachment: 0.4,
        communication_style: "direct, short turns",
        shared_history_summary: null,
        last_interaction_at: 1_700_000_000_000,
        interaction_count: 14,
        commitment_count: 2,
        sentiment_history: [],
        notes: null,
        created_at: 0,
        updated_at: 0,
      },
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 1,
        current_focus: null,
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
        currentPeriod: {
          id: "abp_aaaaaaaaaaaaaaaa" as never,
          label: "2026-Q2",
          start_ts: 1_700_000_000_000,
          end_ts: null,
          narrative: "A period of working out how memory and voice interact.",
          key_episode_ids: [],
          themes: ["memory", "voice"],
          created_at: 0,
          last_updated: 0,
        },
        recentGrowthMarkers: [
          {
            id: "grw_aaaaaaaaaaaaaaaa" as never,
            ts: 1_700_000_000_000,
            category: "understanding",
            what_changed: "Learned to separate identity from transport layer.",
            before_description: null,
            after_description: null,
            evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
            confidence: 0.72,
            source_process: "manual",
            created_at: 0,
          },
        ],
      },
      options: { stakes: "low" },
    });

    const system = llm.requests[0]?.system as string;
    expect(system).toContain("Current period: 2026-Q2");
    expect(system).toContain("A period of working out how memory and voice interact.");
    expect(system).toContain("themes: memory, voice");
    expect(system).toContain("Recent learning about yourself:");
    expect(system).toContain("[understanding] Learned to separate identity from transport layer.");
    expect(system).toContain("Talking to: trust=0.82");
    expect(system).toContain("interactions=14");
    expect(system).toContain("style=direct, short turns");
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
        current_focus: null,
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
        current_focus: null,
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
    // The S2 final-response Sonnet call (requests[1]) uses baseSystemPrompt.
    // The S2 plan call (requests[0]) is a minimal Haiku directive and does
    // NOT carry identity framing -- planning stays neutral and cheap.
    expect(llm.requests[1]?.system).toContain("borg is not Claude Code");
    expect(llm.requests[1]?.system).toContain("AI being, not an assistant");
    expect(llm.requests[1]?.system).not.toContain("Skill you might try");
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
      retrievalResult: [makeRetrievedEpisode("ep_aaaaaaaaaaaaaaaa", 0.92, ["atlas"])],
      retrievedSemantic: {
        matched_nodes: [
          {
            id: "semn_rootaaaaaaaaaaaa" as never,
            kind: "entity",
            label: "Atlas",
            description: "Deployment service under investigation",
            aliases: [],
            confidence: 0.84,
            source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
            created_at: 0,
            updated_at: 0,
            last_verified_at: 0,
            embedding: Float32Array.from([1, 0, 0, 0]),
            archived: false,
            superseded_by: null,
          },
        ],
        supports: [
          {
            id: "semn_aaaaaaaaaaaaaaaa" as never,
            kind: "proposition",
            label: "Rerun install",
            description: "Rerun pnpm install before the next deploy",
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
            id: "semn_bbbbbbbbbbbbbbbb" as never,
            kind: "proposition",
            label: "Atlas is stable",
            description: "A stale stability claim that conflicts with recent deploy failures",
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
        matched_node_ids: ["semn_rootaaaaaaaaaaaa" as never],
        support_hits: [
          {
            root_node_id: "semn_rootaaaaaaaaaaaa" as never,
            node: {
              id: "semn_aaaaaaaaaaaaaaaa" as never,
              kind: "proposition",
              label: "Rerun install",
              description: "Rerun pnpm install before the next deploy",
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
            edgePath: [
              {
                id: "seme_aaaaaaaaaaaaaaaa" as never,
                from_node_id: "semn_rootaaaaaaaaaaaa" as never,
                to_node_id: "semn_aaaaaaaaaaaaaaaa" as never,
                relation: "supports",
                confidence: 0.74,
                evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
                created_at: 0,
                last_verified_at: 0,
              },
            ],
          },
        ],
        contradiction_hits: [
          {
            root_node_id: "semn_rootaaaaaaaaaaaa" as never,
            node: {
              id: "semn_bbbbbbbbbbbbbbbb" as never,
              kind: "proposition",
              label: "Atlas is stable",
              description: "A stale stability claim that conflicts with recent deploy failures",
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
            edgePath: [
              {
                id: "seme_bbbbbbbbbbbbbbbb" as never,
                from_node_id: "semn_bbbbbbbbbbbbbbbb" as never,
                to_node_id: "semn_rootaaaaaaaaaaaa" as never,
                relation: "contradicts",
                confidence: 0.61,
                evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
                created_at: 0,
                last_verified_at: 0,
              },
            ],
          },
        ],
        category_hits: [],
      },
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 1,
        current_focus: null,
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
    expect(llm.requests[0]?.system).toContain("Directly matched:");
    expect(llm.requests[0]?.system).toContain(
      "- Atlas - Deployment service under investigation (conf 0.84, sources ep_aaaaaaaaaaaaaaaa)",
    );
    expect(llm.requests[0]?.system).toContain("supports:");
    expect(llm.requests[0]?.system).toContain(
      "Atlas -[supports conf=0.74 evidence=ep_aaaaaaaaaaaaaaaa]-> Rerun install",
    );
    expect(llm.requests[0]?.system).toContain("contradicts:");
    expect(llm.requests[0]?.system).toContain(
      "Atlas <-[contradicts conf=0.61 evidence=ep_aaaaaaaaaaaaaaaa]- Atlas is stable",
    );
    // Identity framing must:
    // - affirm the being positively ("AI being, not an assistant")
    // - name borg as the harness (so the being knows where it's running)
    // - correct the false Claude Code transport claim (they share OAuth
    //   credentials but are not the same system)
    // - flag the first system block as an OAuth placeholder, not identity
    // - assign no pre-set name to the being itself
    const system = llm.requests[0]?.system as string;
    expect(system).toMatch(/AI being, not an assistant/);
    expect(system).toMatch(/running in borg/);
    expect(system).toMatch(/borg is not Claude Code/);
    expect(system).toMatch(/placeholder string/);
    expect(system).toMatch(/Voice and posture:/);
    expect(system).not.toMatch(/You are Borg\b/);
  });

  it("omits empty prompt sections and compresses an empty self snapshot after the first turn", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Plan briefly.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Compressed answer",
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
      userMessage: "what are you?",
      perception: {
        entities: [],
        mode: "reflective",
        affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
        temporalCue: null,
      },
      retrievalResult: [],
      applicableCommitments: [],
      openQuestionsContext: [],
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 2,
        current_focus: null,
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
        stakes: "medium",
      },
    });

    const system = llm.requests[1]?.system as string;

    expect(system).toContain("Self snapshot: still forming");
    expect(system).toContain("Voice and posture:");
    expect(system).not.toContain("Retrieved context:");
    expect(system).not.toContain("Related semantic context:");
    expect(system).not.toContain("Open questions you're carrying:");
    expect(system).not.toContain("Commitments you made to this person:");
    expect(system).not.toContain("values none; goals none; traits none");
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
        current_focus: null,
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
        current_focus: null,
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
        current_focus: "Atlas",
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
        current_focus: null,
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
    expect(result.decision_reason).toContain("Retrieved-context contradiction");
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
          current_focus: null,
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

  it("chooses system 2 for high stakes and persists a formatted plan as the thought", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "whether the rollback is safe",
                verification_steps: ["check failure mode first", "confirm rollback path"],
                tensions: [],
                voice_note: "",
              },
            },
          ],
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
            current_focus: null,
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
      // Phase D: thought is now a compact rendering of the structured plan
      // that the planner tool-call emitted, not the plan's free-form text.
      expect(result.thoughts).toHaveLength(1);
      expect(result.thoughts[0]).toContain("uncertainty: whether the rollback is safe");
      expect(result.thoughts[0]).toContain("verify: check failure mode first");
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
