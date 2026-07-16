import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { DEFAULT_CONFIG } from "../../config/index.js";
import { SkillSelector, deriveProceduralContextKey } from "../../memory/procedural/index.js";
import { createSkillSplitReviewQueueHandler } from "../../memory/review-queue/index.js";
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
  type EpisodeId,
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

const TYPESCRIPT_DEBUG_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "code_debugging",
  domain_tags: ["typescript"],
  audience_scope: "self",
});
const ROADMAP_PLANNING_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "planning",
  domain_tags: ["roadmap"],
  audience_scope: "self",
});
const SQLITE_RESEARCH_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "research",
  domain_tags: ["sqlite"],
  audience_scope: "self",
});
const TYPESCRIPT_DEBUG_KNOWN_OTHER_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "code_debugging",
  domain_tags: ["typescript"],
  audience_scope: "known_other",
});

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
    additionalResolvedEpisodeIds?: readonly EpisodeId[];
  } = {},
) {
  const sourceStreamIds = [createStreamEntryId(), createStreamEntryId()];
  const episode = await harness.episodicRepository.createEpisode(
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
    resolvedEpisodeIds: [episode.id, ...(input.additionalResolvedEpisodeIds ?? [])],
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
  const episode = await harness.episodicRepository.createEpisode(createEpisodeFixture());
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
        contextKey: TYPESCRIPT_DEBUG_CONTEXT_KEY,
        alpha: 6,
        beta: 1,
        attempts: 5,
        successes: 5,
        failures: 0,
      },
      {
        contextKey: ROADMAP_PLANNING_CONTEXT_KEY,
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

  it("batches disclosure hydration once for a procedural evidence collection", async () => {
    const evidenceCount = 24;
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: evidenceCount, maxSkillsPerRun: 1 }),
      llmClient: new FakeLLMClient({
        responses: [
          createSkillCandidateResponse({
            applies_when: "deployment rollback comparison",
          }),
        ],
      }),
    });
    const evidenceRows = [];
    const sharedEpisode = await harness.episodicRepository.createEpisode(createEpisodeFixture());

    for (let index = 0; index < evidenceCount; index += 1) {
      evidenceRows.push(
        await addSuccessEvidence(harness, {
          additionalResolvedEpisodeIds: [sharedEpisode.id],
        }),
      );
    }

    const getMany = vi.spyOn(harness.episodicRepository, "getMany");
    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());

    expect(plan.items).toHaveLength(1);
    expect(getMany).toHaveBeenCalledTimes(1);
    const lookedUpEpisodeIds = getMany.mock.calls[0]?.[0] ?? [];
    const expectedEpisodeIds = evidenceRows.flatMap((evidence) => evidence.resolved_episode_ids);
    expect(lookedUpEpisodeIds).toHaveLength(evidenceCount + 1);
    expect(new Set(lookedUpEpisodeIds)).toEqual(new Set(expectedEpisodeIds));
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
    const sourceEpisode = await harness.episodicRepository.createEpisode(createEpisodeFixture());
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

  it("exhausts a tight budget after two cluster syntheses and aborts the third cleanly", async () => {
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

    // Pin a tight budget so this exhaustion test is independent of the
    // configured default (which is sized for real runs, not two syntheses).
    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), { budget: 4_000 });

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
      const sourceEpisode = await harness.episodicRepository.createEpisode(createEpisodeFixture());
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

  it("synthesizes cross-private-audience evidence with per-row labels and an inherited label", async () => {
    const sam = "ent_aaaaaaaaaaaaaaaa" as EntityId;
    const alex = "ent_bbbbbbbbbbbbbbbb" as EntityId;
    const approach = "Compare the private plan against the active milestone list.";
    const samProblem = "Sam private planning issue one";
    const alexProblem = "Alex private planning issue two";
    const llm = new FakeLLMClient({
      responses: [
        createSkillCandidateResponse({
          applies_when: "private planning comparison",
          approach,
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [evidenceEmbeddingText(samProblem, approach), [1, 0, 0, 0]],
          [evidenceEmbeddingText(alexProblem, approach), [1, 0, 0, 0]],
        ]),
      ),
      llmClient: llm,
    });
    await addSuccessEvidence(harness, {
      audienceEntityId: sam,
      problemText: samProblem,
      approachSummary: approach,
      evidenceText: "Sam confirmed the private planning comparison worked.",
    });
    await addSuccessEvidence(harness, {
      audienceEntityId: alex,
      problemText: alexProblem,
      approachSummary: approach,
      evidenceText: "Alex confirmed the private planning comparison worked.",
    });

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const [skill] = harness.skillRepository.list();

    expect(result.errors).toEqual([]);
    expect(llm.requests).toHaveLength(1);
    expect(prompt).toContain(samProblem);
    expect(prompt).toContain(alexProblem);
    expect(prompt).toContain("relationship_private");
    expect(prompt).toContain(sam);
    expect(prompt).toContain(alex);
    expect(skill).toMatchObject({
      applies_when: "private planning comparison",
      disclosure_label: {
        disclosureClass: "relationship_private",
        originAudienceEntityIds: [sam, alex],
        privateToEntityIds: [sam, alex],
        publicToEntityIds: [],
      },
    });
  });

  it("labels unresolved procedural evidence sources as unknown instead of dropping them", async () => {
    const approach = "Compare the orphaned evidence against the intended procedure.";
    const firstProblem = "Unresolved source evidence one";
    const secondProblem = "Unresolved source evidence two";
    const llm = new FakeLLMClient({
      responses: [
        createSkillCandidateResponse({
          applies_when: "unresolved evidence comparison",
          approach,
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
      embeddingClient: new TestEmbeddingClient(
        new Map([
          [evidenceEmbeddingText(firstProblem, approach), [1, 0, 0, 0]],
          [evidenceEmbeddingText(secondProblem, approach), [1, 0, 0, 0]],
        ]),
      ),
      llmClient: llm,
    });

    for (const problemText of [firstProblem, secondProblem]) {
      harness.proceduralEvidenceRepository.insert({
        pendingAttemptSnapshot: {
          problem_text: problemText,
          approach_summary: approach,
          selected_skill_id: null,
          source_stream_ids: [createStreamEntryId()],
          turn_counter: 1,
          audience_entity_id: null,
        },
        classification: "success",
        evidenceText: `${problemText} succeeded without a resolved episode.`,
        grounded: true,
        skillActuallyApplied: true,
        resolvedEpisodeIds: [],
        audienceEntityId: null,
      });
    }

    const process = createProcess(harness);
    const result = await process.run(harness.createContext(), {});
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const [skill] = harness.skillRepository.list();

    expect(result.errors).toEqual([]);
    expect(prompt).toContain(firstProblem);
    expect(prompt).toContain(secondProblem);
    expect(prompt).toContain("disclosure_class=unknown");
    expect(skill).toMatchObject({
      applies_when: "unresolved evidence comparison",
      source_episode_ids: [],
      disclosure_label: {
        disclosureClass: "unknown",
        originAudienceEntityIds: [],
        privateToEntityIds: [],
        publicToEntityIds: [],
      },
    });
  });

  it("does not use archived exact episode matches as procedural sources", async () => {
    const problem = "Archived exact source evidence";
    const approach = "Compare the source rows against the active evidence.";
    const archivedSourceIds = [createStreamEntryId(), createStreamEntryId()];
    const visibleSourceIds = [createStreamEntryId(), createStreamEntryId()];
    const llm = new FakeLLMClient({
      responses: [
        createSkillCandidateResponse({
          applies_when: "active evidence comparison",
          approach,
        }),
      ],
    });
    harness = await createOfflineTestHarness({
      configOverrides: proceduralConfig({ minSupport: 2 }),
      embeddingClient: new TestEmbeddingClient(
        new Map([[evidenceEmbeddingText(problem, approach), [1, 0, 0, 0]]]),
      ),
      llmClient: llm,
    });
    const archivedEpisode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture({
        title: problem,
        source_stream_ids: archivedSourceIds,
      }),
    );
    harness.episodicRepository.archiveEpisode(archivedEpisode.id, {
      caller: "procedural-synthesizer.test",
      reason: "exercise exact-match visibility gate",
      process: "procedural-synthesizer",
    });
    const visibleEpisode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture({
        title: problem,
        source_stream_ids: visibleSourceIds,
      }),
    );

    harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: {
        problem_text: problem,
        approach_summary: approach,
        selected_skill_id: null,
        source_stream_ids: archivedSourceIds,
        turn_counter: 1,
        audience_entity_id: null,
      },
      classification: "success",
      evidenceText: "The archived exact source should not be reused.",
      grounded: true,
      skillActuallyApplied: true,
      resolvedEpisodeIds: [],
      audienceEntityId: null,
    });
    harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: {
        problem_text: problem,
        approach_summary: approach,
        selected_skill_id: null,
        source_stream_ids: visibleSourceIds,
        turn_counter: 2,
        audience_entity_id: null,
      },
      classification: "success",
      evidenceText: "The visible exact source should remain available.",
      grounded: true,
      skillActuallyApplied: true,
      resolvedEpisodeIds: [],
      audienceEntityId: null,
    });

    const process = createProcess(harness);
    const plan = await process.plan(harness.createContext());
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

    expect(plan.items[0]?.source_episode_ids).toEqual([visibleEpisode.id]);
    expect(prompt).not.toContain(archivedEpisode.id);
    expect(prompt).toContain(visibleEpisode.id);
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
                context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
              }),
            ],
          }),
          expect.objectContaining({
            label: "Roadmap planning comparison",
            context_stats: [
              expect.objectContaining({
                context_key: ROADMAP_PLANNING_CONTEXT_KEY,
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
        context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
        alpha: 6,
        beta: 1,
      }),
    ]);
    expect(harness.skillRepository.listContextStatsForSkill(newSkills[1]!.id)).toEqual([
      expect.objectContaining({
        context_key: ROADMAP_PLANNING_CONTEXT_KEY,
        alpha: 4,
        beta: 1,
      }),
    ]);
    expect(
      harness.auditLog.list({ process: "procedural-synthesizer" }).map((item) => item.action),
    ).toContain("skill_split");
  });

  it("keeps persisted skill and stream state identical during a heavy dry-run split plan", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
    await harness.streamWriter.append({
      kind: "internal_event",
      content: { hook: "procedural_dry_run_baseline" },
    });
    const beforeSkill = harness.skillRepository.get(skill.id);
    const beforeStats = harness.skillRepository.listContextStatsForSkill(skill.id);
    const beforeStream = new StreamReader({ dataDir: harness.tempDir }).tail(100);

    const result = await createProcess(harness).run(harness.createContext(), { dryRun: true });

    expect(result).toMatchObject({
      dryRun: true,
      errors: [],
      changes: [expect.objectContaining({ action: "skill_split_proposal" })],
    });
    expect(harness.skillRepository.get(skill.id)).toEqual(beforeSkill);
    expect(harness.skillRepository.listContextStatsForSkill(skill.id)).toEqual(beforeStats);
    expect(new StreamReader({ dataDir: harness.tempDir }).tail(100)).toEqual(beforeStream);
    expect(harness.reviewQueueRepository.list({ kind: "skill_split" })).toEqual([]);
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

  it("logs malformed split tool output without mutating after one repair attempt", async () => {
    const malformedResponse = {
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
    };
    const llm = new FakeLLMClient({
      responses: [malformedResponse, malformedResponse],
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

    expect(result.changes).toEqual([]);
    expect(result.errors).toHaveLength(1);
    expect(llm.requests).toHaveLength(2);
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
    expect(llm.requests).toHaveLength(2);
  });

  it("suppresses a split candidate after repeated malformed split output", async () => {
    const malformedResponse = {
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
    };
    const llm = new FakeLLMClient({
      responses: Array.from({ length: 6 }, () => malformedResponse),
    });
    const clock = new ManualClock(1_000_000);
    harness = await createOfflineTestHarness({
      clock,
      configOverrides: proceduralConfig({
        minContextAttemptsForSplit: 3,
        minDivergenceForSplit: 0.01,
        splitCooldownDays: 0.000001,
        maxSplitParseFailures: 3,
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

    expect(llm.requests).toHaveLength(6);
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
    const { skill } = await addSkillWithContextStats(harness, {
      contexts: [
        {
          contextKey: TYPESCRIPT_DEBUG_CONTEXT_KEY,
          alpha: 6,
          beta: 1,
          attempts: 5,
          successes: 5,
          failures: 0,
        },
        {
          contextKey: ROADMAP_PLANNING_CONTEXT_KEY,
          alpha: 5,
          beta: 2,
          attempts: 5,
          successes: 4,
          failures: 1,
        },
        {
          contextKey: SQLITE_RESEARCH_CONTEXT_KEY,
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
          target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
        },
        {
          applies_when: "Manual roadmap split",
          approach: "Manual roadmap approach.",
          target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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

  it("rejects accepted split reviews after a newer split claim takes over", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSkillSplitResponse({
          decision: "split",
          parts: [
            {
              applies_when: "TypeScript debugging comparison",
              approach: "Compare the compiler failure with the last passing TypeScript state.",
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
    const originalClaim = (review!.refs.cooldown as { claimed_at: number }).claimed_at;
    const newerClaim = originalClaim + 1_000;

    expect(
      harness.skillRepository.claimSplit({
        skillId: skill.id,
        claimedAt: newerClaim,
        staleBefore: newerClaim,
      }),
    ).toBe(true);

    const resolved = await harness.reviewQueueRepository.resolve(review!.id, "accept");

    expect(resolved).toMatchObject({
      resolution: "reject",
      refs: expect.objectContaining({
        review_resolution: expect.objectContaining({
          requested_decision: "accept",
          reason: `Skill split no longer applies: ${skill.id}`,
        }),
      }),
    });
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "active",
      splitting_at: newerClaim,
      last_split_attempt_at: expect.any(Number),
    });
    expect(harness.skillRepository.list()).toHaveLength(1);
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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

  it("plans splits from skill source episodes spanning private audiences with disclosure labels", async () => {
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
    const episodeA = await harness.episodicRepository.createEpisode(
      createEpisodeFixture({
        title: "Audience A private skill evidence",
        audience_entity_id: audienceA,
        shared: false,
      }),
    );
    const episodeB = await harness.episodicRepository.createEpisode(
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
        context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
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
        context_key: ROADMAP_PLANNING_CONTEXT_KEY,
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
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
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
    expect(llm.requests).toHaveLength(1);
    expect(prompt).toContain("Audience A private skill evidence");
    expect(prompt).toContain("Audience B private skill evidence");
    expect(prompt).toContain("relationship_private");
    expect(prompt).toContain(audienceA);
    expect(prompt).toContain(audienceB);
    expect(review?.refs).toEqual(
      expect.objectContaining({
        evidence_summary: expect.objectContaining({
          source_episode_ids: [episodeA.id, episodeB.id],
          source_disclosure_label: {
            disclosure_class: "relationship_private",
            origin_audience_entity_ids: [audienceA, audienceB],
            private_to_entity_ids: [audienceA, audienceB],
            public_to_entity_ids: [],
          },
        }),
      }),
    );
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      status: "active",
      last_split_attempt_at: null,
      splitting_at: expect.any(Number),
    });
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
    harness.reviewQueueRepository.registerHandler(
      createSkillSplitReviewQueueHandler(
        createSkillSplitReviewHandler({
          skillRepository: harness.skillRepository,
          auditLog: harness.auditLog,
          clock: harness.clock,
          workingMemoryStore,
        }),
      ),
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Known-audience TypeScript debugging comparison",
              approach: "Compare the audience-scoped compiler failure with their known baseline.",
              target_contexts: [TYPESCRIPT_DEBUG_KNOWN_OTHER_CONTEXT_KEY],
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
    const { skill } = await addSkillWithContextStats(harness, {
      contexts: [
        {
          contextKey: TYPESCRIPT_DEBUG_CONTEXT_KEY,
          alpha: 6,
          beta: 1,
          attempts: 5,
          successes: 5,
          failures: 0,
        },
        {
          contextKey: TYPESCRIPT_DEBUG_KNOWN_OTHER_CONTEXT_KEY,
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
        context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
      }),
    ]);
    expect(harness.skillRepository.listContextStatsForSkill(newSkills[1]!.id)).toEqual([
      expect.objectContaining({
        context_key: TYPESCRIPT_DEBUG_KNOWN_OTHER_CONTEXT_KEY,
      }),
    ]);
  });

  it("uses stored metadata for v2 context sketches without audience split validation", async () => {
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
              approach: "This combines different audience scopes as ordinary context metadata.",
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
        rationale: `Rejected split proposal: Split target context assigned more than once: ${selfKey}`,
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
              target_contexts: [TYPESCRIPT_DEBUG_CONTEXT_KEY],
            },
            {
              applies_when: "Roadmap planning comparison",
              approach: "Compare the roadmap against the current goal list.",
              target_contexts: [ROADMAP_PLANNING_CONTEXT_KEY],
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
    const episode = await harness.episodicRepository.createEpisode(
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

    const { workingMemory: nextWorkingMemory } = await reflector.reflect(
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
    await harness.episodicRepository.createEpisode(
      createEpisodeFixture(
        {
          title: "Atlas rollback fix",
          source_stream_ids: firstSourceIds,
        },
        [1, 0, 0, 0],
      ),
    );
    await harness.episodicRepository.createEpisode(
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
            hot_entities: ["Atlas"],
            pending_actions: [],
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
            discourse_state: {
              stop_until_substantive_content: null,
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
              hot_entities: ["Atlas"],
              pending_actions: [],
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
              discourse_state: {
                stop_until_substantive_content: null,
              },
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
