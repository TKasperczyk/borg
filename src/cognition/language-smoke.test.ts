import { describe, expect, it, vi } from "vitest";

import { SuppressionSet } from "./attention/index.js";
import { Perceiver } from "./perception/index.js";
import { TurnRetrievalCoordinator } from "./retrieval/turn-coordinator.js";
import { computeExecutiveContextFits, selectExecutiveFocus } from "../executive/index.js";
import { FakeLLMClient, type LLMCompleteOptions } from "../llm/index.js";
import { SkillSelector } from "../memory/procedural/index.js";
import {
  buildSelfScoringFeatureSet,
  toRetrievalScoringFeatures,
} from "../retrieval/scoring-features.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import {
  TestEmbeddingClient,
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
  createWorkingMemoryFixture,
} from "../offline/test-support.js";

function toolResponse(name: string, input: Record<string, unknown>) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: `toolu_${name}`,
        name,
        input,
      },
    ],
  };
}

function perceptionToolRouter(options: LLMCompleteOptions) {
  const toolName = options.tool_choice?.type === "tool" ? options.tool_choice.name : "";

  switch (toolName) {
    case "EmitEntityExtraction":
      return toolResponse(toolName, { entities: ["Atlas"] });
    case "EmitModeDetection":
      return toolResponse(toolName, { mode: "problem_solving" });
    case "EmitAffectiveSignal":
      return toolResponse(toolName, { valence: 0, arousal: 0.2, dominant_emotion: null });
    case "EmitTemporalCue":
      return toolResponse(toolName, { has_cue: false });
    default:
      throw new Error(`Unexpected tool request: ${toolName}`);
  }
}

describe("cross-language cognition smoke", () => {
  it("uses LLM structure and paired bilingual vectors for a Chinese turn", async () => {
    const chineseTurn = "请帮我调试 Atlas 部署里的 TypeScript 错误，并提醒我相关问题。";
    const goalText = "Stabilize Atlas deployment";
    const valueLabel = "clarity";
    const valueDescription = "Prefer explicit deployment notes.";
    const valueText = `${valueLabel}\n${valueDescription}`;
    const openQuestionText = "Which Atlas deployment risk still needs an answer?";
    const vector = [1, 0, 0, 0] as const;
    const embeddingClient = new TestEmbeddingClient(
      new Map<string, readonly number[]>([
        [chineseTurn, vector],
        [goalText, vector],
        [valueText, vector],
        [openQuestionText, vector],
      ]),
    );
    const llmClient = new FakeLLMClient({
      responses: [
        perceptionToolRouter,
        perceptionToolRouter,
        perceptionToolRouter,
        perceptionToolRouter,
        toolResponse("EmitProceduralContext", {
          problem_kind: "code_debugging",
          domain_tags: ["typescript", "deployment"],
          confidence: 0.92,
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      embeddingClient,
      llmClient,
    });

    try {
      const goal = harness.goalsRepository.add({
        description: goalText,
        priority: 10,
        provenance: { kind: "manual" },
      });
      const valueRecord = harness.valuesRepository.add({
        label: valueLabel,
        description: valueDescription,
        priority: 10,
        provenance: { kind: "manual" },
      });
      const episode = createEpisodeFixture(
        {
          title: "Atlas TypeScript deployment fix",
          narrative: "The Atlas deployment stabilized after a TypeScript configuration fix.",
          tags: ["atlas"],
        },
        [...vector],
      );
      await harness.episodicRepository.insert(episode);
      const semanticNode = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Atlas deployment TypeScript fix",
            description: "Atlas deployment failures can involve TypeScript configuration.",
            source_episode_ids: [episode.id],
          },
          [...vector],
        ),
      );
      const openQuestion = harness.openQuestionsRepository.add({
        question: openQuestionText,
        urgency: 0.8,
        related_semantic_node_ids: [semanticNode.id],
        source: "reflection",
        provenance: { kind: "manual" },
      });
      await harness.openQuestionsRepository.waitForPendingEmbeddings();

      const classifierFailure = vi.fn();
      const perception = await new Perceiver({
        llmClient,
        model: "claude-opus-4-7",
        onClassifierFailure: classifierFailure,
        clock: harness.clock,
      }).perceive(chineseTurn);
      const selfScoringFeatures = await buildSelfScoringFeatureSet({
        embeddingClient,
        goals: [goal],
        activeValues: [valueRecord],
      });
      const contextFitByGoalId = await computeExecutiveContextFits({
        embeddingClient,
        goalVectors: selfScoringFeatures.goalVectors,
        contextText: chineseTurn,
      });
      const executiveFocus = selectExecutiveFocus({
        goals: [goal],
        cognitionInput: chineseTurn,
        nowMs: harness.clock.now(),
        threshold: 0,
        deadlineLookaheadMs: 7 * 24 * 60 * 60 * 1_000,
        staleMs: 14 * 24 * 60 * 60 * 1_000,
        contextFitByGoalId,
      });
      const retrievalScoringFeatures = toRetrievalScoringFeatures({
        selfFeatures: selfScoringFeatures,
        primaryGoalId: executiveFocus.selected_goal?.id ?? null,
      });
      const coordinator = new TurnRetrievalCoordinator({
        commitmentRepository: harness.commitmentRepository,
        reviewQueueRepository: harness.reviewQueueRepository,
        moodRepository: harness.moodRepository,
        retrievalPipeline: harness.retrievalPipeline,
        skillSelector: new SkillSelector({
          repository: harness.skillRepository,
          contextStatsRepository: harness.proceduralContextStatsRepository,
        }),
        clock: harness.clock,
      });
      const coordinated = await coordinator.coordinate({
        sessionId: DEFAULT_SESSION_ID,
        turnId: "turn_chinese_smoke",
        userMessage: chineseTurn,
        recentMessages: [],
        cognitionInput: chineseTurn,
        isSelfAudience: true,
        audienceEntityId: null,
        audienceEntity: null,
        audienceProfile: null,
        perception,
        workingMemory: createWorkingMemoryFixture(),
        selfSnapshot: {
          values: [valueRecord],
          goals: [goal],
          traits: [],
        },
        executiveFocus,
        activeValues: [valueRecord],
        scoringFeatures: retrievalScoringFeatures,
        suppressionSet: new SuppressionSet(),
        findEntityByName: () => null,
        llmClient,
        proceduralContextModel: "claude-opus-4-7",
      });
      const openQuestionRetrieval = await harness.retrievalPipeline.searchWithContext(chineseTurn, {
        limit: 3,
        goalDescriptions: [goal.description],
        primaryGoalDescription: goal.description,
        activeValues: [valueRecord],
        scoringFeatures: retrievalScoringFeatures,
        includeOpenQuestions: true,
      });

      expect(classifierFailure).not.toHaveBeenCalled();
      expect(executiveFocus.selected_goal?.id).toBe(goal.id);
      expect(coordinated.proceduralContext).toMatchObject({
        problem_kind: "code_debugging",
        domain_tags: ["typescript", "deployment"],
      });
      expect(coordinated.retrievedEpisodes[0]?.episode.id).toBe(episode.id);
      expect(coordinated.retrievedEpisodes[0]?.scoreBreakdown.goalRelevance).toBeGreaterThan(0);
      expect(coordinated.retrievedEpisodes[0]?.scoreBreakdown.valueAlignment).toBeGreaterThan(0);
      expect(coordinated.retrievedSemantic.matched_nodes[0]?.label).toBe(
        "Atlas deployment TypeScript fix",
      );
      expect(openQuestionRetrieval.open_questions.map((question) => question.id)).toContain(
        openQuestion.id,
      );
    } finally {
      await harness.cleanup();
    }
  });
});
