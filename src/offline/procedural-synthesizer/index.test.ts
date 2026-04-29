import { afterEach, describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { DEFAULT_CONFIG } from "../../config/index.js";
import { SkillSelector, deriveProceduralContextKey } from "../../memory/procedural/index.js";
import { createWorkingMemory, WorkingMemoryStore } from "../../memory/working/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { SuppressionSet } from "../../cognition/attention/index.js";
import { Reflector } from "../../cognition/reflection/index.js";
import type { RetrievalConfidence } from "../../retrieval/index.js";
import { StreamReader } from "../../stream/index.js";
import { createSkillsListTool } from "../../tools/index.js";
import { ManualClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createSkillId,
  createStreamEntryId,
  type EntityId,
  type SkillId,
} from "../../util/ids.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
  type OfflineTestHarness,
} from "../test-support.js";

import { ProceduralSynthesizerProcess } from "./index.js";
import { createSkillSplitReviewHandler } from "./skill-split-review.js";

function proceduralConfig(overrides: Partial<typeof DEFAULT_CONFIG.offline.proceduralSynthesizer>) {
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
  rejection_reason?: "unusable_abstraction" | "centered_proper_noun" | null;
  inputTokens?: number;
  outputTokens?: number;
}) {
  return {
    text: "",
    input_tokens: input.inputTokens ?? 10,
    output_tokens: input.outputTokens ?? 5,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_skill_candidate",
        name: "EmitProceduralSkillCandidate",
        input: {
          applies_when: input.applies_when,
          approach:
            input.approach ?? "Compare the failing state against the last known-good state.",
          abstraction_fit: input.abstraction_fit ?? "usable",
          rejection_reason: input.rejection_reason ?? null,
        },
      },
    ],
  };
}

function createSkillSplitResponse(input: {
  decision: "split" | "no_split" | "refine_in_place";
  parts?: Array<{
    applies_when: string;
    approach: string;
    target_contexts: string[];
  }>;
  rationale?: string;
  inputTokens?: number;
  outputTokens?: number;
}) {
  return {
    text: "",
    input_tokens: input.inputTokens ?? 10,
    output_tokens: input.outputTokens ?? 5,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_skill_split",
        name: "EmitSkillSplit",
        input: {
          decision: input.decision,
          ...(input.parts === undefined ? {} : { parts: input.parts }),
          rationale: input.rationale ?? "Context buckets have divergent outcomes.",
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
              attempt_turn_counter: 1,
              classification: "success",
              evidence,
              grounded: true,
              skill_actually_applied: true,
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

function evidenceEmbeddingText(problemText: string, approachSummary: string): string {
  return [problemText, approachSummary].join("\n");
}

async function addSuccessEvidence(
  harness: OfflineTestHarness,
  input: {
    problemText?: string;
    approachSummary?: string;
    evidenceText?: string;
    grounded?: boolean;
    skillActuallyApplied?: boolean;
    audienceEntityId?: EntityId | null;
    selectedSkillId?: SkillId | null;
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
      selected_skill_id: input.selectedSkillId ?? null,
      source_stream_ids: sourceStreamIds,
      turn_counter: 1,
      audience_entity_id: input.audienceEntityId ?? null,
    },
    classification: "success",
    evidenceText: input.evidenceText ?? "User confirmed the deploy worked.",
    grounded: input.grounded ?? true,
    skillActuallyApplied: input.skillActuallyApplied ?? true,
    resolvedEpisodeIds: [episode.id],
    audienceEntityId: input.audienceEntityId ?? null,
  });
}

async function addSkillWithContextStats(
  harness: OfflineTestHarness,
  input: {
    alpha?: number;
    beta?: number;
    attempts?: number;
    successes?: number;
    failures?: number;
    contexts?: Array<{
      contextKey: string;
      alpha: number;
      beta: number;
      attempts: number;
      successes: number;
      failures: number;
    }>;
  } = {},
) {
  const episode = await harness.episodicRepository.insert(createEpisodeFixture());
  const skill = await harness.skillRepository.add({
    applies_when: "reuse the comparison approach across work",
    approach: "Compare the current failed state with the last known-good state.",
    sourceEpisodes: [episode.id],
  });
  const updated = await harness.skillRepository.replace({
    ...skill,
    alpha: input.alpha ?? 10,
    beta: input.beta ?? 2,
    attempts: input.attempts ?? 10,
    successes: input.successes ?? 9,
    failures: input.failures ?? 1,
  });
  const contextRows = (
    input.contexts ?? [
      {
        contextKey: "code_debugging:typescript:self",
        alpha: 6,
        beta: 1,
        attempts: 5,
        successes: 5,
        failures: 0,
      },
      {
        contextKey: "planning:roadmap:self",
        alpha: 4,
        beta: 1,
        attempts: 3,
        successes: 3,
        failures: 0,
      },
    ]
  ).map((context) => ({
    skill_id: updated.id,
    context_key: context.contextKey,
    alpha: context.alpha,
    beta: context.beta,
    attempts: context.attempts,
    successes: context.successes,
    failures: context.failures,
    last_used: 1_000,
    last_successful: context.successes > 0 ? 1_000 : null,
    updated_at: 1_000,
  }));

  harness.skillRepository.restoreContextStats(contextRows);

  return {
    skill: updated,
    contextRows,
  };
}

function getOpenSkillSplitReview(harness: OfflineTestHarness, skillId: SkillId) {
  return harness.reviewQueueRepository
    .list({ kind: "skill_split", openOnly: true })
    .find((item) => item.refs.original_skill_id === skillId);
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

  it("does not cluster evidence for a selected skill that was not applied", async () => {
    const llm = new FakeLLMClient();
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
      llmClient: llm,
    });
    await addSuccessEvidence(harness);
    await addSuccessEvidence(harness, {
      problemText: "Atlas deploy failed after a second rollback.",
      skillActuallyApplied: false,
    });

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toEqual([]);
    expect(llm.requests).toHaveLength(0);
  });

  it("plans the LLM-generated skill text and applies without another LLM call", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillCandidateResponse({
          applies_when: "deployment rollback comparison",
          approach: "Compare the failing deploy state against the last clean release.",
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
      llmClient: llm,
    });
    await addSuccessEvidence(harness);
    await addSuccessEvidence(harness);

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());
    const preview = process.preview(plan);

    expect(plan.items[0]).toMatchObject({
      candidate: {
        applies_when: "deployment rollback comparison",
        approach: "Compare the failing deploy state against the last clean release.",
      },
      dedup_decision: {
        skill_id: null,
      },
      rejection_reason: null,
    });
    expect(preview.changes[0]?.preview).toMatchObject({
      applies_when: "deployment rollback comparison",
      approach: "Compare the failing deploy state against the last clean release.",
    });
    expect(llm.requests).toHaveLength(1);
    expect(llm.requests[0]?.max_tokens).toBe(1_500);

    const result = await process.apply(harness.createContext(), plan);

    expect(result.errors).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect(harness.skillRepository.list()).toEqual([
      expect.objectContaining({
        applies_when: "deployment rollback comparison",
      }),
    ]);
  });

  it("records one synthesized posterior outcome per supporting evidence row", async () => {
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 3 }),
      llmClient: new FakeLLMClient({
        responses: [
          createSkillCandidateResponse({
            applies_when: "deployment rollback comparison",
          }),
        ],
      }),
    });
    await addSuccessEvidence(harness);
    await addSuccessEvidence(harness);
    await addSuccessEvidence(harness);

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const [skill] = harness.skillRepository.list();

    expect(result.errors).toEqual([]);
    expect(skill).toMatchObject({
      alpha: 5,
      attempts: 3,
      successes: 3,
    });
  });

  it("does not duplicate live-recorded selected skill outcomes during synthesis", async () => {
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
      llmClient: new FakeLLMClient({
        responses: [
          createSkillCandidateResponse({
            applies_when: "deployment rollback comparison",
          }),
        ],
      }),
    });
    const sourceEpisode = await harness.episodicRepository.insert(createEpisodeFixture());
    const existing = await harness.skillRepository.add({
      applies_when: "deployment rollback comparison",
      approach: "Existing approach.",
      sourceEpisodes: [sourceEpisode.id],
    });
    const first = await addSuccessEvidence(harness, {
      selectedSkillId: existing.id,
    });
    const second = await addSuccessEvidence(harness, {
      problemText: "Atlas deploy failed after a second rollback.",
      selectedSkillId: existing.id,
    });

    for (const evidence of [first, second]) {
      harness.skillRepository.recordOutcome(existing.id, true, evidence.resolved_episode_ids);
    }

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});

    expect(result.errors).toEqual([]);
    expect(harness.skillRepository.get(existing.id)).toMatchObject({
      attempts: 2,
      successes: 2,
    });
  });

  it("uses the default budget for two cluster syntheses and aborts the third cleanly", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillCandidateResponse({
          applies_when: "deployment rollback comparison",
          inputTokens: 1_000,
          outputTokens: 500,
        }),
        createSkillCandidateResponse({
          applies_when: "roadmap planning comparison",
          inputTokens: 1_000,
          outputTokens: 500,
        }),
        createSkillCandidateResponse({
          applies_when: "reflective habit comparison",
          inputTokens: 1_000,
          outputTokens: 500,
        }),
      ],
    });
    const deployProblem = "Atlas deploy failed after rollback.";
    const deployApproach = "Compare the failing deploy state to the last clean release.";
    const roadmapProblem = "Sprint roadmap plan stalled.";
    const roadmapApproach = "Compare the plan against the goal list.";
    const reflectProblem = "Reflective habit insight was hard to apply.";
    const reflectApproach = "Compare the reflection pattern against prior insight notes.";
    harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [evidenceEmbeddingText(deployProblem, deployApproach), [1, 0, 0, 0]],
          [evidenceEmbeddingText(roadmapProblem, roadmapApproach), [0, 1, 0, 0]],
          [evidenceEmbeddingText(reflectProblem, reflectApproach), [0, 0, 1, 0]],
        ]),
      ),
    });
    await addSuccessEvidence(harness, {
      problemText: deployProblem,
      approachSummary: deployApproach,
    });
    await addSuccessEvidence(harness, {
      problemText: deployProblem,
      approachSummary: deployApproach,
    });
    await addSuccessEvidence(harness, {
      problemText: roadmapProblem,
      approachSummary: roadmapApproach,
    });
    await addSuccessEvidence(harness, {
      problemText: roadmapProblem,
      approachSummary: roadmapApproach,
    });
    await addSuccessEvidence(harness, {
      problemText: reflectProblem,
      approachSummary: reflectApproach,
    });
    await addSuccessEvidence(harness, {
      problemText: reflectProblem,
      approachSummary: reflectApproach,
    });

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});

    expect(result.budget_exhausted).toBe(true);
    expect(result.changes).toHaveLength(2);
    expect(harness.skillRepository.list()).toHaveLength(2);
    expect(llm.requests).toHaveLength(3);
    expect(llm.requests.map((request) => request.max_tokens)).toEqual([1_500, 1_500, 1_500]);
  });

  it.each([
    { similarity: 0.87, expectedSkillCount: 2 },
    { similarity: 0.89, expectedSkillCount: 1 },
  ])(
    "uses the dedup threshold boundary at $similarity",
    async ({ similarity, expectedSkillCount }) => {
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
          attempts: 2,
          successes: 2,
        });
      }
    },
  );

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

  it("queues and accepts an LLM skill split, then migrates context stats to the new skills", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: ["planning:roadmap:self"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const review = getOpenSkillSplitReview(harness, skill.id);

    expect(review).toMatchObject({
      kind: "skill_split",
      refs: expect.objectContaining({
        original_skill_id: skill.id,
        proposed_children: [
          expect.objectContaining({
            label: "TypeScript debugging comparison",
            problem: "TypeScript debugging comparison",
            approach: "Compare the compiler failure with the last passing TypeScript state.",
            context_stats: [
              expect.objectContaining({
                context_key: "code_debugging:typescript:self",
              }),
            ],
          }),
          expect.objectContaining({
            label: "Roadmap planning comparison",
            context_stats: [
              expect.objectContaining({
                context_key: "planning:roadmap:self",
              }),
            ],
          }),
        ],
        rationale: "Context buckets have divergent outcomes.",
        evidence_summary: expect.objectContaining({
          divergence: expect.any(Number),
        }),
        cooldown: expect.objectContaining({
          claimed_at: expect.any(Number),
          split_cooldown_days: 7,
        }),
      }),
    });
    await harness.reviewQueueRepository.resolve(review!.id, "accept");
    const original = harness.skillRepository.get(skill.id);
    const newSkills = (original?.superseded_by ?? []).map((skillId) =>
      harness!.skillRepository.get(skillId),
    );

    expect(result.errors).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect(result.changes).toEqual([
      expect.objectContaining({
        action: "skill_split_proposal",
        targets: expect.objectContaining({
          review_item_id: review!.id,
        }),
      }),
    ]);
    expect(original).toMatchObject({
      status: "superseded",
    });
    expect(newSkills).toEqual([
      expect.objectContaining({
        applies_when: "TypeScript debugging comparison",
        alpha: 6,
        beta: 1,
        attempts: 5,
      }),
      expect.objectContaining({
        applies_when: "Roadmap planning comparison",
        alpha: 4,
        beta: 1,
        attempts: 3,
      }),
    ]);
    expect(harness.skillRepository.listContextStatsForSkill(skill.id)).toEqual([]);
    expect(harness.skillRepository.listContextStatsForSkill(newSkills[0]!.id)).toEqual([
      expect.objectContaining({
        context_key: "code_debugging:typescript:self",
        alpha: 6,
        beta: 1,
      }),
    ]);
    expect(harness.skillRepository.listContextStatsForSkill(newSkills[1]!.id)).toEqual([
      expect.objectContaining({
        context_key: "planning:roadmap:self",
        alpha: 4,
        beta: 1,
      }),
    ]);
    expect(
      harness.auditLog.list({ process: "procedural-synthesizer" }).map((item) => item.action),
    ).toContain("skill_split");
  });

  it("logs no_split decisions without mutating skills", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "no_split",
          rationale: "The buckets reflect noisy use rather than a reusable distinction.",
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const entries = new StreamReader({ dataDir: harness.tempDir }).tail(5);

    expect(result.errors).toEqual([]);
    expect(result.changes).toEqual([]);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "active",
      superseded_by: [],
    });
    expect(entries).toContainEqual(
      expect.objectContaining({
        kind: "internal_event",
        content: expect.objectContaining({
          hook: "skill_split_decision",
          decision: "no_split",
          skill_id: skill.id,
        }),
      }),
    );

    const second = await process.run(harness.createContext(), {});

    expect(second.changes).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      last_split_attempt_at: expect.any(Number),
      splitting_at: null,
    });
  });

  it("logs malformed split tool output without mutating or retrying", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_bad_split",
              name: "EmitSkillSplit",
              input: {
                decision: "split",
              },
            },
          ],
        },
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
        skillSplitDryRun: false,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const entries = new StreamReader({ dataDir: harness.tempDir }).tail(5);

    expect(result.changes).toEqual([]);
    expect(result.errors).toHaveLength(1);
    expect(llm.requests).toHaveLength(1);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "active",
      superseded_by: [],
    });
    expect(entries).toContainEqual(
      expect.objectContaining({
        kind: "internal_event",
        content: expect.objectContaining({
          hook: "skill_split_failed",
          skill_id: skill.id,
        }),
      }),
    );

    const second = await process.run(harness.createContext(), {});

    expect(second.changes).toEqual([]);
    expect(llm.requests).toHaveLength(1);
  });

  it("suppresses a split candidate after repeated malformed split output", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_bad_split_1",
              name: "EmitSkillSplit",
              input: {
                decision: "split",
              },
            },
          ],
        },
        {
          text: "",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_bad_split_2",
              name: "EmitSkillSplit",
              input: {
                decision: "split",
              },
            },
          ],
        },
        {
          text: "",
          input_tokens: 10,
          output_tokens: 5,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_bad_split_3",
              name: "EmitSkillSplit",
              input: {
                decision: "split",
              },
            },
          ],
        },
      ],
    });
    const clock = new ManualClock(1_000_000);
    harness = await createOfflineTestHarness({
      clock,
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
        splitCooldownDays: 0.000001,
        maxSplitParseFailures: 3,
        skillSplitDryRun: false,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);
    const process = createProcess(harness);

    await process.run(harness.createContext(), {});
    clock.advance(1_000);
    await process.run(harness.createContext(), {});
    clock.advance(1_000);
    await process.run(harness.createContext(), {});
    clock.advance(1_000);
    await process.run(harness.createContext(), {});

    expect(llm.requests).toHaveLength(3);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "active",
      split_failure_count: 3,
      last_split_error: expect.any(String),
      requires_manual_review: true,
      splitting_at: null,
    });
    const tool = createSkillsListTool({
      listSkills: (limit) => harness!.skillRepository.list(limit),
    });
    const toolOutput = await tool.invoke(
      {
        limit: 5,
      },
      {
        sessionId: DEFAULT_SESSION_ID,
        origin: "deliberator",
      },
    );

    expect(toolOutput.skills.find((item) => item.id === skill.id)).toMatchObject({
      requires_manual_review: true,
    });
  });

  it("rejects split proposals that do not cover every divergent bucket", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: ["planning:roadmap:self"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
        skillSplitDryRun: false,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness, {
      contexts: [
        {
          contextKey: "code_debugging:typescript:self",
          alpha: 6,
          beta: 1,
          attempts: 5,
          successes: 5,
          failures: 0,
        },
        {
          contextKey: "planning:roadmap:self",
          alpha: 5,
          beta: 2,
          attempts: 5,
          successes: 4,
          failures: 1,
        },
        {
          contextKey: "research:sqlite:self",
          alpha: 1,
          beta: 6,
          attempts: 5,
          successes: 0,
          failures: 5,
        },
      ],
    });

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const entries = new StreamReader({ dataDir: harness.tempDir }).tail(5);

    expect(result.errors).toEqual([]);
    expect(result.changes).toEqual([]);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "active",
      superseded_by: [],
      last_split_attempt_at: expect.any(Number),
    });
    expect(harness.skillRepository.listContextStatsForSkill(skill.id)).toHaveLength(3);
    expect(entries).toContainEqual(
      expect.objectContaining({
        kind: "internal_event",
        content: expect.objectContaining({
          hook: "skill_split_decision",
          decision: "no_split",
          skill_id: skill.id,
        }),
      }),
    );
  });

  it("queues split proposals by default without writing dry-run internal events", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: ["planning:roadmap:self"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const entries = new StreamReader({ dataDir: harness.tempDir }).tail(5);
    const review = getOpenSkillSplitReview(harness, skill.id);

    expect(result.errors).toEqual([]);
    expect(result.changes).toEqual([
      expect.objectContaining({
        action: "skill_split_proposal",
        targets: expect.objectContaining({
          review_item_id: review!.id,
        }),
      }),
    ]);
    expect(review).toMatchObject({
      kind: "skill_split",
      refs: expect.objectContaining({
        original_skill_id: skill.id,
      }),
    });
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "active",
      superseded_by: [],
    });
    expect(harness.skillRepository.list()).toHaveLength(1);
    expect(
      entries.some(
        (entry) =>
          entry.kind === "internal_event" &&
          typeof entry.content === "object" &&
          entry.content !== null &&
          "hook" in entry.content &&
          "skill_id" in entry.content &&
          entry.content.hook === "skill_split_proposal" &&
          entry.content.skill_id === skill.id,
      ),
    ).toBe(false);

    await harness.reviewQueueRepository.resolve(review!.id, {
      decision: "reject",
      reason: "Operator wants to keep the general skill.",
    });
    expect(harness.reviewQueueRepository.get(review!.id)).toMatchObject({
      resolved_at: expect.any(Number),
      resolution: "reject",
      refs: expect.objectContaining({
        review_resolution: expect.objectContaining({
          reason: "Operator wants to keep the general skill.",
        }),
      }),
    });
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      last_split_attempt_at: expect.any(Number),
      splitting_at: null,
    });

    const second = await process.run(harness.createContext(), {});

    expect(second.changes).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      last_split_attempt_at: expect.any(Number),
      splitting_at: null,
    });
  });

  it("resolves a stale accepted split as rejected without applying it", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: ["planning:roadmap:self"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);

    const process = createProcess(harness);
    await process.run(harness.createContext(), {});
    const review = getOpenSkillSplitReview(harness, skill.id);

    await harness.skillRepository.supersedeWithSplits({
      skillId: skill.id,
      parts: [
        {
          applies_when: "Manual TypeScript split",
          approach: "Manual debug approach.",
          target_contexts: ["code_debugging:typescript:self"],
        },
        {
          applies_when: "Manual roadmap split",
          approach: "Manual roadmap approach.",
          target_contexts: ["planning:roadmap:self"],
        },
      ],
      supersededAt: harness.clock.now(),
    });

    const resolved = await harness.reviewQueueRepository.resolve(review!.id, "accept");

    expect(resolved).toMatchObject({
      resolution: "reject",
      refs: expect.objectContaining({
        review_resolution: expect.objectContaining({
          requested_decision: "accept",
          reason: `Skill already superseded: ${skill.id}`,
        }),
      }),
    });
    expect(harness.reviewQueueRepository.getOpen()).not.toContainEqual(
      expect.objectContaining({ id: review!.id }),
    );
  });

  it("does not call the split LLM when another run already holds the claim", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: ["planning:roadmap:self"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
        skillSplitDryRun: false,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);
    harness.skillRepository.claimSplit({
      skillId: skill.id,
      claimedAt: 10_000,
      staleBefore: 9_000,
    });

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});

    expect(result.errors).toEqual([]);
    expect(result.changes).toEqual([]);
    expect(llm.requests).toHaveLength(0);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      splitting_at: 10_000,
      last_split_attempt_at: null,
    });
  });

  it("rejects split planning when skill source episodes span private audiences", async () => {
    const audienceA = "ent_aaaaaaaaaaaaaaaa" as EntityId;
    const audienceB = "ent_bbbbbbbbbbbbbbbb" as EntityId;
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: ["planning:roadmap:self"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
        skillSplitDryRun: false,
      }),
      llmClient: llm,
    });
    const episodeA = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Audience A private skill evidence",
        audience_entity_id: audienceA,
        shared: false,
      }),
    );
    const episodeB = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Audience B private skill evidence",
        audience_entity_id: audienceB,
        shared: false,
      }),
    );
    const skill = await harness.skillRepository.add({
      applies_when: "reuse the comparison approach across private work",
      approach: "Compare the current failed state with the last known-good state.",
      sourceEpisodes: [episodeA.id, episodeB.id],
    });
    harness.skillRepository.restoreContextStats([
      {
        skill_id: skill.id,
        context_key: "code_debugging:typescript:self",
        alpha: 6,
        beta: 1,
        attempts: 5,
        successes: 5,
        failures: 0,
        last_used: 1_000,
        last_successful: 1_000,
        updated_at: 1_000,
      },
      {
        skill_id: skill.id,
        context_key: "planning:roadmap:self",
        alpha: 1,
        beta: 6,
        attempts: 5,
        successes: 0,
        failures: 5,
        last_used: 1_000,
        last_successful: null,
        updated_at: 1_000,
      },
    ]);

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const entries = new StreamReader({ dataDir: harness.tempDir }).tail(5);

    expect(result.errors).toEqual([]);
    expect(result.changes).toEqual([]);
    expect(llm.requests).toHaveLength(0);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "active",
      last_split_attempt_at: expect.any(Number),
      splitting_at: null,
    });
    expect(entries).toContainEqual(
      expect.objectContaining({
        kind: "internal_event",
        content: expect.objectContaining({
          hook: "skill_split_skipped",
          reason: "skill_source_episodes_cross_audiences",
          skill_id: skill.id,
        }),
      }),
    );
  });

  it("clears pending attempts that referenced a superseded split skill", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: ["planning:roadmap:self"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
        skillSplitDryRun: false,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);
    const workingMemoryStore = new WorkingMemoryStore({
      dataDir: harness.tempDir,
      clock: harness.clock,
    });
    workingMemoryStore.save({
      ...createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      pending_procedural_attempts: [
        {
          problem_text: "Fix the TypeScript failure.",
          approach_summary: "Use the old comparison skill.",
          selected_skill_id: skill.id,
          source_stream_ids: [createStreamEntryId()],
          turn_counter: 1,
          audience_entity_id: null,
        },
      ],
    });
    harness.reviewQueueRepository.setSkillSplitReviewHandler(
      createSkillSplitReviewHandler({
        skillRepository: harness.skillRepository,
        auditLog: harness.auditLog,
        clock: harness.clock,
        workingMemoryStore,
      }),
    );

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const review = getOpenSkillSplitReview(harness, skill.id);

    await harness.reviewQueueRepository.resolve(review!.id, "accept");

    expect(result.errors).toEqual([]);
    expect(workingMemoryStore.load(DEFAULT_SESSION_ID).pending_procedural_attempts).toEqual([
      expect.objectContaining({
        selected_skill_id: null,
      }),
    ]);
  });

  it("does not cross-write split context stats across audience scopes", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "Self TypeScript debugging comparison",
              approach: "Compare the self-scoped compiler failure with the last passing state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Known-audience TypeScript debugging comparison",
              approach: "Compare the audience-scoped compiler failure with their known baseline.",
              target_contexts: ["code_debugging:typescript:known_other"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 5,
        minDivergenceForSplit: 0.3,
        skillSplitDryRun: false,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness, {
      contexts: [
        {
          contextKey: "code_debugging:typescript:self",
          alpha: 6,
          beta: 1,
          attempts: 5,
          successes: 5,
          failures: 0,
        },
        {
          contextKey: "code_debugging:typescript:known_other",
          alpha: 1,
          beta: 6,
          attempts: 5,
          successes: 0,
          failures: 5,
        },
      ],
    });

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const review = getOpenSkillSplitReview(harness, skill.id);

    await harness.reviewQueueRepository.resolve(review!.id, "accept");
    const original = harness.skillRepository.get(skill.id);
    const newSkills = (original?.superseded_by ?? []).map((skillId) =>
      harness!.skillRepository.get(skillId),
    );

    expect(result.errors).toEqual([]);
    expect(newSkills).toHaveLength(2);
    expect(harness.skillRepository.listContextStatsForSkill(newSkills[0]!.id)).toEqual([
      expect.objectContaining({
        context_key: "code_debugging:typescript:self",
      }),
    ]);
    expect(harness.skillRepository.listContextStatsForSkill(newSkills[1]!.id)).toEqual([
      expect.objectContaining({
        context_key: "code_debugging:typescript:known_other",
      }),
    ]);
  });

  it("uses stored metadata for v2 context sketches and split audience validation", async () => {
    const selfContext = {
      problem_kind: "code_debugging" as const,
      domain_tags: ["typescript", "deployment"],
      audience_scope: "self" as const,
    };
    const knownOtherContext = {
      ...selfContext,
      audience_scope: "known_other" as const,
    };
    const selfKey = deriveProceduralContextKey(selfContext);
    const knownOtherKey = deriveProceduralContextKey(knownOtherContext);
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "Mixed-audience deployment debugging",
              approach: "This invalidly crosses audience scopes.",
              target_contexts: [selfKey, knownOtherKey],
            },
            {
              applies_when: "Duplicate self deployment debugging",
              approach: "A duplicate target keeps the proposal shaped like a split.",
              target_contexts: [selfKey],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 5,
        minDivergenceForSplit: 0.3,
      }),
      llmClient: llm,
    });
    const { contextRows } = await addSkillWithContextStats(harness, {
      contexts: [
        {
          contextKey: selfKey,
          alpha: 6,
          beta: 1,
          attempts: 5,
          successes: 5,
          failures: 0,
        },
        {
          contextKey: knownOtherKey,
          alpha: 1,
          beta: 6,
          attempts: 5,
          successes: 0,
          failures: 5,
        },
      ],
    });
    harness.skillRepository.restoreContextStats(
      contextRows.map((row) => ({
        ...row,
        procedural_context: row.context_key === selfKey ? selfContext : knownOtherContext,
      })),
    );

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(prompt).toContain("code_debugging; typescript, deployment; audience=self");
    expect(prompt).toContain("code_debugging; typescript, deployment; audience=known_other");
    expect(result.errors).toEqual([]);
    expect(result.changes).toEqual([]);
    expect(
      new StreamReader({ dataDir: harness.tempDir }).tail(5).map((entry) => entry.content),
    ).toContainEqual(
      expect.objectContaining({
        hook: "skill_split_decision",
        decision: "no_split",
        rationale: "Rejected split proposal: Skill split part crosses audience scopes",
      }),
    );
  });

  it("does not duplicate the same planned split review on repeated apply", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: ["code_debugging:typescript:self"],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: ["planning:roadmap:self"],
            },
          ],
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
        skillSplitDryRun: false,
      }),
      llmClient: llm,
    });
    const { skill } = await addSkillWithContextStats(harness);

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());
    const first = await process.apply(harness.createContext(), plan);
    const second = await process.apply(harness.createContext(), plan);
    const reviews = harness.reviewQueueRepository.list({ kind: "skill_split", openOnly: true });

    expect(first.errors).toEqual([]);
    expect(first.changes).toHaveLength(1);
    expect(second.errors).toEqual([]);
    expect(second.changes).toEqual([]);
    expect(reviews).toHaveLength(1);
    expect(llm.requests).toHaveLength(1);
    expect(
      harness.skillRepository.list(10).filter((record) => record.status === "active"),
    ).toHaveLength(1);

    await harness.reviewQueueRepository.resolve(reviews[0]!.id, "accept");

    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "superseded",
    });
    expect(
      harness.skillRepository.list(10).filter((record) => record.status === "active"),
    ).toHaveLength(2);
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

  it("uses LLM-provided centered-proper-noun rejection", async () => {
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
      llmClient: new FakeLLMClient({
        responses: [
          createSkillCandidateResponse({
            applies_when: "Atlas deployment rollback comparison",
            rejection_reason: "centered_proper_noun",
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

  it("does not synthesize ungrounded success evidence", async () => {
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
    });
    await addSuccessEvidence(harness, {
      evidenceText: "The assistant response said this works.",
      grounded: false,
    });
    await addSuccessEvidence(harness, {
      problemText: "Atlas deploy failed after a second rollback.",
      evidenceText: "The assistant response said this works.",
      grounded: false,
    });

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toEqual([]);
  });

  it("logs and retires late outcomes for superseded skills without mutating stats", async () => {
    harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [createReflectionResponse("User confirmed the old approach worked.")],
      }),
    });
    const sourceStreamIds = [createStreamEntryId(), createStreamEntryId()];
    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Late superseded skill outcome",
        source_stream_ids: sourceStreamIds,
      }),
    );
    const replacementId = createSkillId();
    const skill = await harness.skillRepository.add({
      applies_when: "old deployment comparison",
      approach: "Use the old comparison.",
      sourceEpisodes: [episode.id],
    });
    const superseded = await harness.skillRepository.replace({
      ...skill,
      status: "superseded",
      superseded_by: [replacementId],
      superseded_at: 1_000,
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      skillRepository: harness.skillRepository,
      proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
    });
    const workingMemory = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      turn_counter: 2,
      pending_procedural_attempts: [
        {
          problem_text: "Atlas deploy failed after rollback.",
          approach_summary: "Compare the failing deploy state to the last clean release.",
          selected_skill_id: superseded.id,
          source_stream_ids: sourceStreamIds,
          turn_counter: 1,
          audience_entity_id: null,
        },
      ],
    };

    const nextWorkingMemory = await reflector.reflect(
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
        workingMemory,
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
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Next.",
          tool_calls: [],
          intents: [],
          workingMemory,
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        selectedSkillId: superseded.id,
        suppressionSet: new SuppressionSet(2),
      },
      harness.streamWriter,
    );
    const entries = new StreamReader({ dataDir: harness.tempDir }).tail(5);

    expect(nextWorkingMemory.pending_procedural_attempts).toEqual([]);
    expect(harness.skillRepository.get(superseded.id)).toMatchObject({
      status: "superseded",
      alpha: superseded.alpha,
      beta: superseded.beta,
      attempts: superseded.attempts,
      successes: superseded.successes,
      failures: superseded.failures,
    });
    expect(entries).toContainEqual(
      expect.objectContaining({
        kind: "internal_event",
        content: expect.objectContaining({
          hook: "record_outcome_skipped_superseded",
          skill_id: superseded.id,
        }),
      }),
    );
  });

  it("synthesizes evidence emitted by reflector and surfaces the skill on selection", async () => {
    harness = await createOfflineTestHarness({
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [
            evidenceEmbeddingText(
              "Atlas deploy failed after rollback.",
              "Compare the failing deploy state to the last clean release.",
            ),
            [1, 0, 0, 0],
          ],
          ["deployment rollback comparison", [1, 0, 0, 0]],
          ["deployment rollback is failing", [1, 0, 0, 0]],
        ]),
      ),
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
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      skillRepository: harness.skillRepository,
      proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
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
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            suppressed: [],
            pending_procedural_attempts: [
              {
                problem_text: "Atlas deploy failed after rollback.",
                approach_summary: "Compare the failing deploy state to the last clean release.",
                selected_skill_id: null,
                source_stream_ids: sourceStreamIds,
                turn_counter: 1,
                audience_entity_id: null,
              },
            ],
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
            intents: [],
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
              pending_social_attribution: null,
              pending_trait_attribution: null,
              mood: null,
              suppressed: [],
              pending_procedural_attempts: [
                {
                  problem_text: "Atlas deploy failed after rollback.",
                  approach_summary: "Compare the failing deploy state to the last clean release.",
                  selected_skill_id: null,
                  source_stream_ids: sourceStreamIds,
                  turn_counter: 1,
                  audience_entity_id: null,
                },
              ],
              mode: "problem_solving",
              updated_at: 0,
            },
          },
          retrievedEpisodes: [],
          retrievalConfidence: createRetrievalConfidence(),
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
