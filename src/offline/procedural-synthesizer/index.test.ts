import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { DEFAULT_CONFIG } from "../../config/index.js";
import { SkillSelector } from "../../memory/procedural/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { SuppressionSet } from "../../cognition/attention/index.js";
import { Reflector } from "../../cognition/reflection/index.js";
import type { RetrievalConfidence } from "../../retrieval/index.js";
import { DEFAULT_SESSION_ID, createStreamEntryId, type EntityId } from "../../util/ids.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  type OfflineTestHarness,
} from "../test-support.js";

import { ProceduralSynthesizerProcess } from "./index.js";

function proceduralConfig(
  overrides: Partial<typeof DEFAULT_CONFIG.offline.proceduralSynthesizer>,
) {
  return {
    offline: {
      ...DEFAULT_CONFIG.offline,
      proceduralSynthesizer: {
        ...DEFAULT_CONFIG.offline.proceduralSynthesizer,
        ...overrides,
      },
    },
  };
}

function createSkillCandidateResponse(input: {
  applies_when: string;
  approach?: string;
  abstraction_fit?: "too_narrow" | "usable" | "too_broad";
}) {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 5,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_skill_candidate",
        name: "EmitProceduralSkillCandidate",
        input: {
          applies_when: input.applies_when,
          approach: input.approach ?? "Compare the failing state against the last known-good state.",
          abstraction_fit: input.abstraction_fit ?? "usable",
        },
      },
    ],
  };
}

function createReflectionResponse(evidence: string) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [
            {
              classification: "success",
              evidence,
            },
          ],
        },
      },
    ],
  };
}

function createProcess(harness: OfflineTestHarness) {
  return new ProceduralSynthesizerProcess({
    skillRepository: harness.skillRepository,
    proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
    registry: harness.registry,
    clock: harness.clock,
  });
}

async function addSuccessEvidence(
  harness: OfflineTestHarness,
  input: {
    problemText?: string;
    approachSummary?: string;
    evidenceText?: string;
    audienceEntityId?: EntityId | null;
  } = {},
) {
  const sourceStreamIds = [createStreamEntryId(), createStreamEntryId()];
  const episode = await harness.episodicRepository.insert(
    createEpisodeFixture(
      {
        title: input.problemText ?? "Atlas deploy failure",
        narrative: "The deploy failed until the rollback state was compared to the clean release.",
        tags: ["deploy"],
        source_stream_ids: sourceStreamIds,
        audience_entity_id: input.audienceEntityId,
        shared: input.audienceEntityId === undefined || input.audienceEntityId === null,
      },
      [1, 0, 0, 0],
    ),
  );

  return harness.proceduralEvidenceRepository.insert({
    pendingAttemptSnapshot: {
      problem_text: input.problemText ?? "Atlas deploy failed after rollback.",
      approach_summary:
        input.approachSummary ?? "Compare the failing deploy state to the last clean release.",
      selected_skill_id: null,
      source_stream_ids: sourceStreamIds,
      turn_counter: 1,
      audience_entity_id: input.audienceEntityId ?? null,
    },
    classification: "success",
    evidenceText: input.evidenceText ?? "User confirmed the deploy worked.",
    resolvedEpisodeIds: [episode.id],
    audienceEntityId: input.audienceEntityId ?? null,
  });
}

function createRetrievalConfidence(): RetrievalConfidence {
  return {
    overall: 0.8,
    evidenceStrength: 0.8,
    coverage: 1,
    sourceDiversity: 1,
    contradictionPresent: false,
    sampleSize: 1,
  };
}

class BoundaryEmbeddingClient implements EmbeddingClient {
  constructor(private readonly candidateSimilarity: number) {}

  async embed(text: string): Promise<Float32Array> {
    return this.vector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vector(text));
  }

  private vector(text: string): Float32Array {
    if (/existing boundary skill/i.test(text)) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    if (/candidate boundary skill/i.test(text)) {
      const x = this.candidateSimilarity;
      return Float32Array.from([x, Math.sqrt(1 - x * x), 0, 0]);
    }

    return Float32Array.from([0, 1, 0, 0]);
  }
}

describe("ProceduralSynthesizerProcess", () => {
  let harness: OfflineTestHarness | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("does not synthesize clusters below min support", async () => {
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
    });
    await addSuccessEvidence(harness);

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toEqual([]);
  });

  it.each([
    { similarity: 0.87, expectedSkillCount: 2 },
    { similarity: 0.89, expectedSkillCount: 1 },
  ])("uses the dedup threshold boundary at $similarity", async ({ similarity, expectedSkillCount }) => {
    harness = await createOfflineTestHarness({
      embeddingClient: new BoundaryEmbeddingClient(similarity),
      configOverrides: proceduralConfig({ minSupport: 2, dedupThreshold: 0.88 }),
      llmClient: new FakeLLMClient({
        responses: [
          createSkillCandidateResponse({
            applies_when: "candidate boundary skill",
          }),
        ],
      }),
    });
    const sourceEpisode = await harness.episodicRepository.insert(createEpisodeFixture());
    const existing = await harness.skillRepository.add({
      applies_when: "existing boundary skill",
      approach: "Existing approach.",
      sourceEpisodes: [sourceEpisode.id],
    });
    await addSuccessEvidence(harness, {
      problemText: "boundary problem one",
      approachSummary: "boundary approach",
    });
    await addSuccessEvidence(harness, {
      problemText: "boundary problem two",
      approachSummary: "boundary approach",
    });

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});

    expect(result.errors).toEqual([]);
    expect(harness.skillRepository.list(10)).toHaveLength(expectedSkillCount);
    if (similarity >= 0.88) {
      expect(harness.skillRepository.get(existing.id)).toMatchObject({
        attempts: 1,
        successes: 1,
      });
    }
  });

  it("excludes private audience evidence from synthesis", async () => {
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
    });
    const sam = harness.entityRepository.resolve("Sam");
    await addSuccessEvidence(harness, {
      audienceEntityId: sam,
      problemText: "Sam private planning issue one",
    });
    await addSuccessEvidence(harness, {
      audienceEntityId: sam,
      problemText: "Sam private planning issue two",
    });

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toEqual([]);
  });

  it("rejects non-usable abstraction fits", async () => {
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
      llmClient: new FakeLLMClient({
        responses: [
          createSkillCandidateResponse({
            applies_when: "deployment rollback comparison",
            abstraction_fit: "too_narrow",
          }),
        ],
      }),
    });
    await addSuccessEvidence(harness);
    await addSuccessEvidence(harness, {
      problemText: "Atlas deploy failed after a second rollback.",
    });

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});

    expect(result.changes).toEqual([]);
    expect(harness.skillRepository.list()).toEqual([]);
  });

  it("does not synthesize self-validating assistant-only success evidence", async () => {
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
    });
    await addSuccessEvidence(harness, {
      evidenceText: "The assistant response said this works.",
    });
    await addSuccessEvidence(harness, {
      problemText: "Atlas deploy failed after a second rollback.",
      evidenceText: "The assistant response said this works.",
    });

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toEqual([]);
  });

  it("synthesizes evidence emitted by reflector and surfaces the skill on selection", async () => {
    harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse("User confirmed the rollback comparison worked."),
          createReflectionResponse("User confirmed the same deploy comparison worked."),
          createSkillCandidateResponse({
            applies_when: "deployment rollback comparison",
            approach: "Compare the failing deploy state against the last clean release.",
          }),
        ],
      }),
    });
    const firstSourceIds = [createStreamEntryId(), createStreamEntryId()];
    const secondSourceIds = [createStreamEntryId(), createStreamEntryId()];
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Atlas rollback fix",
          source_stream_ids: firstSourceIds,
        },
        [1, 0, 0, 0],
      ),
    );
    await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Atlas rollback fix again",
          source_stream_ids: secondSourceIds,
        },
        [1, 0, 0, 0],
      ),
    );
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "haiku",
    });

    for (const sourceStreamIds of [firstSourceIds, secondSourceIds]) {
      await reflector.reflect(
        {
          userMessage: "That worked.",
          perception: {
            entities: ["Atlas"],
            mode: "problem_solving",
            affectiveSignal: {
              valence: 0.4,
              arousal: 0,
              dominant_emotion: null,
            },
            temporalCue: null,
          },
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 2,
            current_focus: "Atlas",
            hot_entities: ["Atlas"],
            pending_intents: [],
            suppressed: [],
            mood: null,
            last_selected_skill_id: null,
            last_selected_skill_turn: null,
            pending_procedural_attempt: {
              problem_text: "Atlas deploy failed after rollback.",
              approach_summary: "Compare the failing deploy state to the last clean release.",
              selected_skill_id: null,
              source_stream_ids: sourceStreamIds,
              turn_counter: 1,
              audience_entity_id: null,
            },
            mode: "problem_solving",
            updated_at: 0,
          },
          selfSnapshot: {
            values: [],
            goals: [],
            traits: [],
          },
          deliberationResult: {
            path: "system_1",
            response: "Next.",
            thoughts: [],
            tool_calls: [],
            usage: {
              input_tokens: 1,
              output_tokens: 1,
              stop_reason: "end_turn",
            },
            decision_reason: "confidence",
            retrievedEpisodes: [],
            referencedEpisodeIds: null,
            thoughtsPersisted: false,
          },
          actionResult: {
            response: "Next.",
            tool_calls: [],
            intents: [],
            workingMemory: {
              session_id: DEFAULT_SESSION_ID,
              turn_counter: 2,
              current_focus: "Atlas",
              hot_entities: ["Atlas"],
              pending_intents: [],
              suppressed: [],
              mood: null,
              last_selected_skill_id: null,
              last_selected_skill_turn: null,
              pending_procedural_attempt: {
                problem_text: "Atlas deploy failed after rollback.",
                approach_summary: "Compare the failing deploy state to the last clean release.",
                selected_skill_id: null,
                source_stream_ids: sourceStreamIds,
                turn_counter: 1,
                audience_entity_id: null,
              },
              mode: "problem_solving",
              updated_at: 0,
            },
          },
          retrievedEpisodes: [],
          retrievalConfidence: createRetrievalConfidence(),
          episodicRepository: harness.episodicRepository,
          goalsRepository: harness.goalsRepository,
          traitsRepository: harness.traitsRepository,
          openQuestionsRepository: harness.openQuestionsRepository,
          proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
          selectedSkillId: null,
          suppressionSet: new SuppressionSet(2),
        },
        harness.streamWriter,
      );
    }

    expect(harness.proceduralEvidenceRepository.listUnconsumed()).toHaveLength(2);

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const selector = new SkillSelector({
      repository: harness.skillRepository,
      sampler: () => 0.9,
    });
    const selected = await selector.select("deployment rollback is failing", { k: 5 });

    expect(result.errors).toEqual([]);
    expect(harness.skillRepository.list()).toHaveLength(1);
    expect(selected?.skill.applies_when).toBe("deployment rollback comparison");
  });
});
