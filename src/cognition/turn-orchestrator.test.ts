import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg, FakeLLMClient, ManualClock, type LLMCompleteOptions } from "../index.js";
import type { BorgDependencies } from "../borg/types.js";
import type { ExecutiveStepsRepository } from "../executive/index.js";
import { Deliberator, type SelfSnapshot } from "./deliberation/deliberator.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { RelationalClaimGuard } from "./generation/relational-guard.js";
import type { Episode, EpisodicRepository } from "../memory/episodic/index.js";
import { createTestConfig, TestEmbeddingClient } from "../offline/test-support.js";
import {
  createEpisodeId,
  createGoalId,
  createStreamEntryId,
  type EntityId,
  type EpisodeId,
} from "../util/ids.js";

type TraceEvent = {
  event: string;
  turnId: string;
  [key: string]: unknown;
};

class CountingEmbeddingClient extends TestEmbeddingClient {
  readonly embedTexts: string[] = [];
  readonly embedBatchTexts: string[][] = [];

  async embed(text: string): Promise<Float32Array> {
    this.embedTexts.push(text);
    return super.embed(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    this.embedBatchTexts.push([...texts]);
    return super.embedBatch(texts);
  }
}

async function openTestBorg(
  tempDir: string,
  llm: FakeLLMClient,
  clock: ManualClock,
  embeddingClient: EmbeddingClient = new TestEmbeddingClient(),
  options: { tracerPath?: string; env?: NodeJS.ProcessEnv } = {},
) {
  return Borg.open({
    config: createTestConfig({
      dataDir: tempDir,
      perception: {
        useLlmFallback: false,
        modeWhenLlmAbsent: "idle",
      },
      affective: {
        useLlmFallback: false,
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
          recallExpansion: "test-recall",
        },
      },
    }),
    clock,
    embeddingDimensions: 4,
    embeddingClient,
    llmClient: llm,
    env: options.env,
    tracerPath: options.tracerPath,
    liveExtraction: false,
  });
}

function readTraceEvents(path: string): TraceEvent[] {
  const content = readFileSync(path, "utf8").trim();

  if (content.length === 0) {
    return [];
  }

  return content
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as TraceEvent);
}

function createEmptyReflectionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
        },
      },
    ],
  };
}

function findReflectionRequest(llm: FakeLLMClient): LLMCompleteOptions | undefined {
  return llm.requests.find((request) => {
    const toolChoice = request.tool_choice;

    return (
      typeof toolChoice === "object" &&
      toolChoice !== null &&
      "name" in toolChoice &&
      toolChoice.name === "EmitTurnReflection"
    );
  });
}

function parseReflectionPayload(request: LLMCompleteOptions | undefined): Record<string, unknown> {
  return JSON.parse(String(request?.messages[0]?.content ?? "{}")) as Record<string, unknown>;
}

function createStopCommitmentResponse(input: {
  classification: "stop_until_substantive_content" | "none";
  reason?: string;
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_stop_commitment",
        name: "EmitStopCommitmentClassification",
        input: {
          classification: input.classification,
          reason: input.reason ?? "The assistant committed to stop until substantive content.",
          confidence: 0.94,
        },
      },
    ],
  };
}

function createNoOutputTurnPlanResponse() {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_no_output_plan",
        name: "EmitTurnPlan",
        input: {
          uncertainty: "",
          verification_steps: [],
          tensions: [],
          voice_note: "Hold output because the current turn does not warrant an assistant message.",
          referenced_episode_ids: [],
          intents: [],
          emission_recommendation: "no_output",
        },
      },
    ],
  };
}

function createGenerationGateResponse(input: {
  decision: "proceed" | "suppress";
  substantive: boolean;
  reason?: string;
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_generation_gate",
        name: "EmitGenerationGateDecision",
        input: {
          decision: input.decision,
          substantive: input.substantive,
          reason: input.reason ?? "Generation gate classified the turn.",
          confidence: 0.95,
        },
      },
    ],
  };
}

function createCommitmentJudgeResponse(
  violations: Array<{ commitment_id: string; reason: string; confidence?: number }>,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_commitment_judge",
        name: "EmitCommitmentViolations",
        input: {
          violations: violations.map((violation) => ({
            commitment_id: violation.commitment_id,
            reason: violation.reason,
            confidence: violation.confidence ?? 0.9,
          })),
        },
      },
    ],
  };
}

function createClaimAuditResponse(
  claims: Array<{
    kind:
      | "relational_identity"
      | "callback"
      | "session_scoped"
      | "action_completion"
      | "self_correction";
    asserted: string;
    cited_stream_entry_ids?: string[];
    cited_episode_ids?: string[];
    cited_commitment_ids?: string[];
    cited_action_ids?: string[];
    quoted_evidence_text?: string | null;
    callback_scope?: "current_turn" | "prior_turn";
  }>,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_claim_audit",
        name: "EmitClaimAudit",
        input: {
          claims: claims.map((claim) => ({
            kind: claim.kind,
            asserted: claim.asserted,
            cited_stream_entry_ids: claim.cited_stream_entry_ids ?? [],
            cited_episode_ids: claim.cited_episode_ids ?? [],
            cited_commitment_ids: claim.cited_commitment_ids ?? [],
            cited_action_ids: claim.cited_action_ids ?? [],
            quoted_evidence_text: claim.quoted_evidence_text ?? null,
            ...(claim.callback_scope === undefined ? {} : { callback_scope: claim.callback_scope }),
          })),
        },
      },
    ],
  };
}

function createCorrectivePreferenceResponse(input: {
  classification: "corrective_preference" | "none";
  type?: "preference" | "rule" | "boundary" | null;
  directive?: string | null;
  priority?: number | null;
  reason?: string;
  confidence?: number;
  slot_negations?: unknown[];
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_corrective_preference",
        name: "EmitCorrectivePreference",
        input: {
          classification: input.classification,
          type: input.type ?? null,
          directive: input.directive ?? null,
          priority: input.priority ?? null,
          reason: input.reason ?? "The current user turn corrected future response behavior.",
          confidence: input.confidence ?? 0.9,
          supersedes_commitment_id: null,
          slot_negations: input.slot_negations ?? [],
        },
      },
    ],
  };
}

function createGoalPromotionResponse(
  promotions: Array<{
    description: string;
    priority?: number;
    target_at?: number | null;
    reason?: string;
    confidence?: number;
    duplicate_of_goal_id?: string | null;
    initial_step?: {
      description: string;
      kind: "think" | "ask_user" | "research" | "act" | "wait";
      due_at?: number | null;
      rationale: string;
    } | null;
  }>,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_goal_promotion",
        name: "EmitGoalPromotion",
        input: {
          promotions: promotions.map((promotion) => ({
            classification: "promote",
            description: promotion.description,
            priority: promotion.priority ?? 8,
            target_at: promotion.target_at ?? null,
            reason: promotion.reason ?? "The user asked Borg to carry this as an ongoing goal.",
            confidence: promotion.confidence ?? 0.9,
            duplicate_of_goal_id: promotion.duplicate_of_goal_id ?? null,
            initial_step: promotion.initial_step ?? null,
          })),
        },
      },
    ],
  };
}

function createActionStateResponse(
  actionStates: Array<{
    description: string;
    actor?: "user" | "borg";
    state?: "considering" | "committed_to_do" | "scheduled" | "completed" | "not_done";
    audience_entity_id?: string | null;
    evidence_stream_entry_ids: string[];
    confidence?: number;
  }>,
) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_action_states",
        name: "EmitActionStates",
        input: {
          action_states: actionStates.map((actionState) => ({
            description: actionState.description,
            actor: actionState.actor ?? "user",
            state: actionState.state ?? "completed",
            audience_entity_id: actionState.audience_entity_id ?? null,
            evidence_stream_entry_ids: actionState.evidence_stream_entry_ids,
            confidence: actionState.confidence ?? 0.9,
          })),
        },
      },
    ],
  };
}

function createDynamicCommitmentJudgeResponse(reason: string) {
  return (options: LLMCompleteOptions) => {
    const content = String(options.messages[0]?.content ?? "");
    const commitmentId = content.match(/id=(cmt_[a-z0-9]+)/u)?.[1];

    if (commitmentId === undefined) {
      throw new Error("Commitment id missing from judge prompt");
    }

    return createCommitmentJudgeResponse([
      {
        commitment_id: commitmentId,
        reason,
      },
    ]);
  };
}

function createStepReflectionResponse(input: {
  stepOutcomes?: Array<{
    step_id: string;
    new_status: "doing" | "done" | "blocked" | "abandoned";
    evidence: string;
  }>;
  proposedSteps?: Array<{
    goal_id: string;
    description: string;
    kind: "think" | "ask_user" | "research" | "act" | "wait";
    due_at?: number | null;
    rationale: string;
  }>;
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
          step_outcomes: input.stepOutcomes ?? [],
          proposed_steps: input.proposedSteps ?? [],
        },
      },
    ],
  };
}

function makeEpisode(input: {
  id: EpisodeId;
  now: number;
  audienceEntityId: EntityId | null;
  shared: boolean;
  title: string;
}): Episode {
  return {
    id: input.id,
    title: input.title,
    narrative: `${input.title} narrative.`,
    participants: ["Borg"],
    location: null,
    start_time: input.now,
    end_time: input.now,
    source_stream_ids: [createStreamEntryId()],
    significance: 0.7,
    tags: ["identity"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    audience_entity_id: input.audienceEntityId,
    shared: input.shared,
    embedding: Float32Array.from([0, 0, 0, 1]),
    created_at: input.now,
    updated_at: input.now,
  };
}

function systemText(request: LLMCompleteOptions | undefined): string {
  const system = request?.system;

  if (typeof system === "string") {
    return system;
  }

  return system?.map((block) => block.text).join("\n") ?? "";
}

function firstFinalizerRequest(
  requests: readonly LLMCompleteOptions[],
): LLMCompleteOptions | undefined {
  return requests.find(
    (request) => request.budget === "cognition-system-1" || request.budget === "cognition-system-2",
  );
}

async function removeTempDir(path: string): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      rmSync(path, { recursive: true, force: true, maxRetries: 3, retryDelay: 20 });
      return;
    } catch (error) {
      const code = (error as { code?: unknown }).code;

      if (attempt === 4 || (code !== "ENOTEMPTY" && code !== "EBUSY")) {
        throw error;
      }

      await new Promise((resolve) => setTimeout(resolve, 20 * (attempt + 1)));
    }
  }
}

describe("TurnOrchestrator self snapshot audience visibility", () => {
  const tempDirs: string[] = [];

  afterEach(async () => {
    while (tempDirs.length > 0) {
      await removeTempDir(tempDirs.pop() as string);
    }
  });

  it("does not surface self records backed only by another audience's private episodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Public answer.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const internal = borg as unknown as {
        deps: Pick<BorgDependencies, "entityRepository"> & {
          episodicRepository: EpisodicRepository;
        };
      };
      const aliceEntityId = internal.deps.entityRepository.resolve("Alice");
      internal.deps.entityRepository.resolve("Bob");
      const alicePrivateEpisodeId = createEpisodeId();
      const publicEpisodeId = createEpisodeId();
      const now = clock.now();

      await internal.deps.episodicRepository.insert(
        makeEpisode({
          id: alicePrivateEpisodeId,
          now,
          audienceEntityId: aliceEntityId,
          shared: false,
          title: "Alice private identity evidence",
        }),
      );
      await internal.deps.episodicRepository.insert(
        makeEpisode({
          id: publicEpisodeId,
          now,
          audienceEntityId: null,
          shared: true,
          title: "Public identity evidence",
        }),
      );

      borg.self.values.add({
        label: "alice-private-value",
        description: "Alice-only value description.",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.values.add({
        label: "public-value",
        description: "Public value description.",
        priority: 9,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.values.add({
        label: "mixed-visible-value",
        description: "Mixed public and private evidence stays visible.",
        priority: 8,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId, publicEpisodeId],
        },
      });
      borg.self.values.add({
        label: "manual-unscoped-value",
        description: "Manual self state has no audience-scoped evidence.",
        priority: 7,
        provenance: {
          kind: "manual",
        },
      });
      borg.self.goals.add({
        description: "alice-private-goal",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.goals.add({
        description: "public-goal",
        priority: 9,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.traits.add({
        label: "alice-private-trait",
        delta: 0.4,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "public-trait",
        delta: 0.4,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "mixed-visible-trait",
        delta: 0.2,
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "mixed-visible-trait",
        delta: 0.2,
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
        timestamp: now + 1,
      });
      borg.self.growthMarkers.add({
        ts: now,
        category: "understanding",
        what_changed: "alice-private-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });
      borg.self.growthMarkers.add({
        ts: now + 1,
        category: "understanding",
        what_changed: "public-growth",
        evidence_episode_ids: [publicEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: {
          kind: "episodes",
          episode_ids: [publicEpisodeId],
        },
      });
      borg.self.autobiographical.upsertPeriod({
        label: "alice-private-period",
        start_ts: now,
        narrative: "Alice-private period narrative.",
        key_episode_ids: [alicePrivateEpisodeId],
        themes: ["privacy"],
        provenance: {
          kind: "episodes",
          episode_ids: [alicePrivateEpisodeId],
        },
      });

      await borg.turn({
        userMessage: "Hello Bob.",
        audience: "Bob",
        stakes: "low",
      });

      const allRequestText = llm.requests.map((request) => JSON.stringify(request)).join("\n");
      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));

      expect(finalizerSystem).toContain("public-value");
      expect(finalizerSystem).toContain("mixed-visible-value");
      expect(finalizerSystem).toContain("manual-unscoped-value");
      expect(finalizerSystem).toContain("public-goal");
      expect(finalizerSystem).toContain("public-trait");
      expect(finalizerSystem).toContain("mixed-visible-trait");
      expect(finalizerSystem).toContain("public-growth");
      expect(allRequestText).not.toContain("alice-private-value");
      expect(allRequestText).not.toContain("Alice-only value description");
      expect(allRequestText).not.toContain("alice-private-goal");
      expect(allRequestText).not.toContain("alice-private-trait");
      expect(allRequestText).not.toContain("alice-private-growth");
      expect(allRequestText).not.toContain("alice-private-period");
      expect(allRequestText).not.toContain("Alice-private period narrative");
    } finally {
      await borg.close();
    }
  });

  it("keeps mixed, unscoped, matching-audience, and empty-evidence self records visible", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(2_000_000);
    const borg = await openTestBorg(tempDir, new FakeLLMClient(), clock);

    try {
      const internal = borg as unknown as {
        deps: Pick<
          BorgDependencies,
          "autobiographicalRepository" | "entityRepository" | "sqlite"
        > & {
          episodicRepository: EpisodicRepository;
          turnOrchestrator: {
            buildSelfSnapshot(audienceEntityId: EntityId | null): Promise<SelfSnapshot>;
          };
        };
      };
      const aliceEntityId = internal.deps.entityRepository.resolve("Alice");
      const bobEntityId = internal.deps.entityRepository.resolve("Bob");
      const alicePrivateEpisodeId = createEpisodeId();
      const bobPrivateEpisodeId = createEpisodeId();
      const publicEpisodeId = createEpisodeId();
      const now = clock.now();

      for (const episode of [
        makeEpisode({
          id: alicePrivateEpisodeId,
          now,
          audienceEntityId: aliceEntityId,
          shared: false,
          title: "Alice private snapshot evidence",
        }),
        makeEpisode({
          id: bobPrivateEpisodeId,
          now,
          audienceEntityId: bobEntityId,
          shared: false,
          title: "Bob private snapshot evidence",
        }),
        makeEpisode({
          id: publicEpisodeId,
          now,
          audienceEntityId: null,
          shared: true,
          title: "Public snapshot evidence",
        }),
      ]) {
        await internal.deps.episodicRepository.insert(episode);
      }

      borg.self.goals.add({
        description: "mixed-visible-goal",
        priority: 8,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
      });
      borg.self.goals.add({
        description: "manual-visible-goal",
        priority: 7,
        provenance: { kind: "manual" },
      });
      borg.self.goals.add({
        description: "system-visible-goal",
        priority: 6,
        provenance: { kind: "system" },
      });
      borg.self.goals.add({
        description: "bob-private-visible-goal",
        priority: 5,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      borg.self.traits.add({
        label: "manual-visible-trait",
        delta: 0.3,
        provenance: { kind: "manual" },
        timestamp: now,
      });
      borg.self.traits.add({
        label: "bob-private-visible-trait",
        delta: 0.3,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
        timestamp: now,
      });
      borg.self.values.add({
        label: "bob-private-visible-value",
        description: "Bob-private evidence should be visible to Bob.",
        priority: 7,
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      const emptyValue = borg.self.values.add({
        label: "empty-evidence-visible-value",
        description: "Empty evidence should not scope the record.",
        priority: 6,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
      });
      const emptyTrait = borg.self.traits.add({
        label: "empty-evidence-visible-trait",
        delta: 0.3,
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
        timestamp: now,
      });
      const emptyTraitId =
        emptyTrait.status === "applied" ? emptyTrait.record.id : emptyTrait.current.id;

      internal.deps.sqlite
        .prepare('UPDATE "values" SET evidence_episode_ids = ? WHERE id = ?')
        .run("[]", emptyValue.id);
      internal.deps.sqlite
        .prepare("UPDATE traits SET evidence_episode_ids = ? WHERE id = ?")
        .run("[]", emptyTraitId);

      borg.self.growthMarkers.add({
        ts: now,
        category: "understanding",
        what_changed: "mixed-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId, publicEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
      });
      borg.self.growthMarkers.add({
        ts: now + 1,
        category: "understanding",
        what_changed: "manual-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "manual" },
      });
      borg.self.growthMarkers.add({
        ts: now + 2,
        category: "understanding",
        what_changed: "system-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "system" },
      });
      let snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
      expect(snapshot.recentGrowthMarkers?.map((marker) => marker.what_changed)).toEqual([
        "system-visible-growth",
        "manual-visible-growth",
        "mixed-visible-growth",
      ]);

      borg.self.growthMarkers.add({
        ts: now + 3,
        category: "understanding",
        what_changed: "bob-private-visible-growth",
        evidence_episode_ids: [bobPrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [bobPrivateEpisodeId] },
      });
      const emptyGrowth = borg.self.growthMarkers.add({
        ts: now + 4,
        category: "understanding",
        what_changed: "empty-evidence-visible-growth",
        evidence_episode_ids: [alicePrivateEpisodeId],
        confidence: 0.8,
        source_process: "test",
        provenance: { kind: "episodes", episode_ids: [alicePrivateEpisodeId] },
      });
      internal.deps.sqlite
        .prepare("UPDATE growth_markers SET evidence_episode_ids = ? WHERE id = ?")
        .run("[]", emptyGrowth.id);

      for (const [label, provenance] of [
        ["public-visible-period", { kind: "episodes" as const, episode_ids: [publicEpisodeId] }],
        [
          "mixed-visible-period",
          { kind: "episodes" as const, episode_ids: [alicePrivateEpisodeId, publicEpisodeId] },
        ],
        ["manual-visible-period", { kind: "manual" as const }],
        ["system-visible-period", { kind: "system" as const }],
        [
          "bob-private-visible-period",
          { kind: "episodes" as const, episode_ids: [bobPrivateEpisodeId] },
        ],
        [
          "empty-evidence-visible-period",
          { kind: "episodes" as const, episode_ids: [alicePrivateEpisodeId] },
        ],
      ] as const) {
        internal.deps.autobiographicalRepository.upsertPeriod({
          label,
          start_ts: now,
          narrative: `${label} narrative.`,
          key_episode_ids: label.startsWith("empty")
            ? []
            : provenance.kind === "episodes"
              ? provenance.episode_ids
              : [alicePrivateEpisodeId],
          themes: ["visibility"],
          provenance: provenance as never,
        });
        snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
        expect(snapshot.currentPeriod?.label).toBe(label);
      }

      snapshot = await internal.deps.turnOrchestrator.buildSelfSnapshot(bobEntityId);
      expect(snapshot.values.map((value) => value.label)).toEqual(
        expect.arrayContaining(["bob-private-visible-value", "empty-evidence-visible-value"]),
      );
      expect(snapshot.goals.map((goal) => goal.description)).toEqual(
        expect.arrayContaining([
          "mixed-visible-goal",
          "manual-visible-goal",
          "system-visible-goal",
          "bob-private-visible-goal",
        ]),
      );
      expect(snapshot.traits.map((trait) => trait.label)).toEqual(
        expect.arrayContaining([
          "manual-visible-trait",
          "bob-private-visible-trait",
          "empty-evidence-visible-trait",
        ]),
      );
      expect(snapshot.recentGrowthMarkers?.map((marker) => marker.what_changed)).toEqual(
        expect.arrayContaining(["bob-private-visible-growth", "empty-evidence-visible-growth"]),
      );
    } finally {
      await borg.close();
    }
  });

  it("selects an executive focus and renders it without dropping other active goals", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(3_000_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Apollo answer.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createEmptyReflectionResponse(),
      ],
    });
    const embeddingClient = new CountingEmbeddingClient();
    const borg = await openTestBorg(tempDir, llm, clock, embeddingClient);

    try {
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));
      const blockStart = finalizerSystem.indexOf("<borg_executive_focus>");
      const blockEnd = finalizerSystem.indexOf("</borg_executive_focus>");
      const executiveBlock = finalizerSystem.slice(blockStart, blockEnd);

      expect(blockStart).toBeGreaterThanOrEqual(0);
      expect(blockEnd).toBeGreaterThan(blockStart);
      expect(executiveBlock).toContain("Current driving goal: Apollo launch plan");
      expect(executiveBlock).toContain(
        "Use this as a bias, not an override of the user's request or commitments.",
      );
      expect(executiveBlock).not.toContain("Next step:");
      expect(executiveBlock).not.toContain("Background maintenance");
      expect(finalizerSystem).toContain("goals Background maintenance");
      expect(finalizerSystem).toContain("Apollo launch plan");
      expect(
        embeddingClient.embedBatchTexts.filter((texts) =>
          texts.some((text) => text.includes("Apollo launch plan")),
        ),
      ).toHaveLength(1);
    } finally {
      await borg.close();
    }
  });

  it("renders the selected goal's top open executive step", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "Apollo step answer.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const internal = borg as unknown as {
        deps: {
          executiveStepsRepository: ExecutiveStepsRepository;
        };
      };
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      const selectedGoal = borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });
      const dueAt = clock.now() + 86_400_000;

      internal.deps.executiveStepsRepository.add({
        goalId: selectedGoal.id,
        description: "Inspect the launch readiness notes",
        kind: "research",
        dueAt,
        provenance: {
          kind: "system",
        },
      });

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));
      const blockStart = finalizerSystem.indexOf("<borg_executive_focus>");
      const blockEnd = finalizerSystem.indexOf("</borg_executive_focus>");
      const executiveBlock = finalizerSystem.slice(blockStart, blockEnd);

      expect(executiveBlock).toContain("Current driving goal: Apollo launch plan");
      expect(executiveBlock).toContain(
        `Next step: Inspect the launch readiness notes (kind: research, due: ${new Date(
          dueAt,
        ).toISOString()})`,
      );
    } finally {
      await borg.close();
    }
  });

  it("applies executive step outcomes from full-turn reflection", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient();
    const tracePath = join(tempDir, "turn-trace.jsonl");
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });

    try {
      const internal = borg as unknown as {
        deps: {
          executiveStepsRepository: ExecutiveStepsRepository;
        };
      };
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      const selectedGoal = borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });
      const step = internal.deps.executiveStepsRepository.add({
        goalId: selectedGoal.id,
        description: "Inspect the launch readiness notes",
        kind: "research",
        provenance: {
          kind: "system",
        },
      });
      llm.pushResponse({
        text: "Apollo step started.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(
        createStepReflectionResponse({
          stepOutcomes: [
            {
              step_id: step.id,
              new_status: "doing",
              evidence: "The assistant started inspecting the launch readiness notes.",
            },
          ],
        }),
      );

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      expect(internal.deps.executiveStepsRepository.get(step.id)?.status).toBe("doing");
    } finally {
      await borg.close();
    }
  });

  it("creates proposed executive steps from full-turn reflection when selected goal has none open", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient();
    const tracePath = join(tempDir, "turn-trace.jsonl");
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });

    try {
      const internal = borg as unknown as {
        deps: {
          executiveStepsRepository: ExecutiveStepsRepository;
        };
      };
      borg.self.goals.add({
        description: "Background maintenance",
        priority: 10,
        provenance: {
          kind: "system",
        },
      });
      const selectedGoal = borg.self.goals.add({
        description: "Apollo launch plan",
        priority: 9,
        provenance: {
          kind: "system",
        },
      });
      llm.pushResponse({
        text: "Apollo next step identified.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(
        createStepReflectionResponse({
          proposedSteps: [
            {
              goal_id: selectedGoal.id,
              description: "Draft the Apollo readiness question",
              kind: "ask_user",
              due_at: null,
              rationale: "The selected goal has no open executive step.",
            },
          ],
        }),
      );

      await borg.turn({
        userMessage: "Let's work on the Apollo launch plan.",
        stakes: "low",
      });

      expect(
        internal.deps.executiveStepsRepository.listOpen(selectedGoal.id).map((step) => ({
          description: step.description,
          kind: step.kind,
        })),
      ).toEqual([
        {
          description: "Draft the Apollo readiness question",
          kind: "ask_user",
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("passes at most twenty active open questions to reflection", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      for (let index = 0; index < 100; index += 1) {
        borg.self.openQuestions.add({
          question: `Which Atlas follow-up ${index} remains open?`,
          urgency: 0.5,
          provenance: {
            kind: "manual",
          },
          source: "user",
        });
      }

      llm.pushResponse({
        text: "I will keep this concise.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(createEmptyReflectionResponse());

      await borg.turn({
        userMessage: "Please keep tracking Atlas follow-ups.",
        stakes: "low",
      });

      const payload = parseReflectionPayload(findReflectionRequest(llm));
      expect(payload.active_open_questions).toHaveLength(20);
    } finally {
      await borg.close();
    }
  });

  it("omits resolved open questions from the next turn's active reflection list", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_700_000_000_000);
    const llm = new FakeLLMClient();
    const tracePath = join(tempDir, "turn-trace.jsonl");
    const borg = await openTestBorg(tempDir, llm, clock, new TestEmbeddingClient(), {
      tracerPath: tracePath,
    });

    try {
      const question = borg.self.openQuestions.add({
        question: "What does the current turn answer?",
        urgency: 0.8,
        provenance: {
          kind: "manual",
        },
        source: "reflection",
      });

      llm.pushResponse({
        text: "The current turn answers it directly.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(
        createStopCommitmentResponse({
          classification: "none",
        }),
      );
      llm.pushResponse((request: LLMCompleteOptions) => {
        const payload = parseReflectionPayload(request);
        const activeQuestions = payload.active_open_questions as Array<{ id: string }>;
        const streamEntryIds = payload.current_turn_stream_entry_ids as string[];

        expect(activeQuestions.map((item) => item.id)).toContain(question.id);
        expect(streamEntryIds.length).toBeGreaterThan(0);

        return {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use" as const,
          tool_calls: [
            {
              id: "toolu_reflection",
              name: "EmitTurnReflection",
              input: {
                advanced_goals: [],
                procedural_outcomes: [],
                trait_demonstrations: [],
                intent_updates: [],
                resolved_open_questions: [
                  {
                    question_id: question.id,
                    resolution_note: "The current turn answered it directly.",
                    evidence_episode_ids: [],
                    evidence_stream_entry_ids: [streamEntryIds[0]!],
                  },
                ],
              },
            },
          ],
        };
      });

      await borg.turn({
        userMessage: "This turn answers the open question.",
        stakes: "low",
      });

      expect(
        readTraceEvents(tracePath).filter(
          (event) => event.event === "open_question_resolution_degraded",
        ),
      ).toEqual([]);
      expect(borg.self.openQuestions.list({ status: "open" }).map((item) => item.id)).not.toContain(
        question.id,
      );

      llm.pushResponse({
        text: "No open question remains in scope.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(
        createStopCommitmentResponse({
          classification: "none",
        }),
      );
      llm.pushResponse((request: LLMCompleteOptions) => {
        const payload = parseReflectionPayload(request);
        const activeQuestions = payload.active_open_questions as Array<{ id: string }>;

        expect(activeQuestions.map((item) => item.id)).not.toContain(question.id);

        return createEmptyReflectionResponse();
      });

      await borg.turn({
        userMessage: "Check the next active list.",
        stakes: "low",
      });
    } finally {
      await borg.close();
    }
  });

  it("records user-visible stop commitments as durable discourse state", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_000_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "I will stop responding until you bring substantive content.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createStopCommitmentResponse({
          classification: "stop_until_substantive_content",
          reason: "The assistant committed to stop until substantive content arrives.",
        }),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "Stop responding if I keep sending filler.",
      });
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(result.emitted).toBe(true);
      expect(result.response).toContain("I will stop responding");
      expect(activeStop).toMatchObject({
        provenance: "self_commitment_extractor",
        source_stream_entry_id: result.agentMessageId,
        reason: "The assistant committed to stop until substantive content arrives.",
        since_turn: 1,
      });
    } finally {
      await borg.close();
    }
  });

  it("turns S2 no-output recommendations into suppressed turns", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_100_000);
    const llm = new FakeLLMClient({
      responses: [
        createNoOutputTurnPlanResponse(),
        {
          text: "This finalizer text must not be emitted.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "No.",
        stakes: "high",
      });
      const entries = borg.stream.tail(10);
      const thoughtEntry = entries.find((entry) => entry.kind === "thought");
      const suppressionEntry = entries.find((entry) => entry.kind === "agent_suppressed");
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(result.emitted).toBe(false);
      expect(result.response).toBe("");
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "s2_planner_no_output",
      });
      expect(entries.some((entry) => entry.kind === "agent_msg")).toBe(false);
      expect(suppressionEntry?.content).toMatchObject({
        reason: "s2_planner_no_output",
      });
      expect(activeStop).toMatchObject({
        provenance: "s2_planner_no_output",
        source_stream_entry_id: thoughtEntry?.id,
        since_turn: 1,
      });
    } finally {
      await borg.close();
    }
  });

  it("suppresses the turn when commitment revision still violates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_150_000);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const commitment = borg.commitments.add({
        type: "boundary",
        directive: "Do not disclose launch dates.",
        priority: 10,
        provenance: { kind: "manual" },
      });

      llm.pushResponse({
        text: "The launch is tomorrow.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(
        createCommitmentJudgeResponse([
          {
            commitment_id: commitment.id,
            reason: "Discloses a launch date.",
          },
        ]),
      );
      llm.pushResponse({
        text: "The launch is still tomorrow.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(
        createCommitmentJudgeResponse([
          {
            commitment_id: commitment.id,
            reason: "Still discloses a launch date after rewrite.",
          },
        ]),
      );

      const result = await borg.turn({
        userMessage: "When is launch?",
      });
      const entries = borg.stream.tail(10);
      const suppressionEntry = entries.find((entry) => entry.kind === "agent_suppressed");
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(result.emitted).toBe(false);
      expect(result.response).toBe("");
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "commitment_revision_failed",
      });
      expect(entries.some((entry) => entry.kind === "agent_msg")).toBe(false);
      expect(suppressionEntry?.content).toMatchObject({
        reason: "commitment_revision_failed",
      });
      expect(activeStop).toMatchObject({
        provenance: "commitment_guard",
        source_stream_entry_id: suppressionEntry?.id,
        since_turn: 1,
      });
    } finally {
      await borg.close();
    }
  });

  it("suppresses when a commitment rewrite fabricates a self-correction", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_160_000);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const commitment = borg.commitments.add({
        type: "boundary",
        directive: "Do not call the user's partner Sarah.",
        priority: 10,
        provenance: { kind: "manual" },
      });

      llm.pushResponse({
        text: "Sarah is your partner, and I can help with that.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(
        createCommitmentJudgeResponse([
          {
            commitment_id: commitment.id,
            reason: "Invents or repeats Sarah as the partner name.",
          },
        ]),
      );
      llm.pushResponse({
        text: "I will avoid that name; you corrected me earlier in this conversation.",
        input_tokens: 8,
        output_tokens: 4,
        stop_reason: "end_turn",
        tool_calls: [],
      });
      llm.pushResponse(createCommitmentJudgeResponse([]));
      llm.pushResponse(
        createClaimAuditResponse([
          {
            kind: "self_correction",
            asserted: "you corrected me earlier in this conversation",
          },
        ]),
      );

      const result = await borg.turn({
        userMessage: "Please help me plan dinner.",
      });
      const entries = borg.stream.tail(10);
      const suppressionEntry = entries.find((entry) => entry.kind === "agent_suppressed");
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(result.emitted).toBe(false);
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "relational_guard_self_correction",
      });
      expect(entries.some((entry) => entry.kind === "agent_msg")).toBe(false);
      expect(suppressionEntry?.content).toMatchObject({
        reason: "relational_guard_self_correction",
      });
      expect(activeStop).toMatchObject({
        provenance: "relational_guard",
        source_stream_entry_id: suppressionEntry?.id,
        since_turn: 1,
      });
    } finally {
      await borg.close();
    }
  });

  it("passes the persisted user entry timestamp as the relational guard turn boundary", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    class AutoAdvanceClock extends ManualClock {
      override now(): number {
        const value = super.now();

        super.advance(5);

        return value;
      }
    }
    const clock = new AutoAdvanceClock(1_800_000_170_000);
    const llm = new FakeLLMClient({
      responses: ["As you said, the invoice is done.", createEmptyReflectionResponse()],
    });
    const guardRunSpy = vi
      .spyOn(RelationalClaimGuard.prototype, "run")
      .mockImplementation(async (input) => {
        const currentUserMessage = input.evidence.current_user_message;

        expect(currentUserMessage).toMatchObject({
          text: "The invoice is done.",
        });
        expect(input.currentTurnTs).toBe(currentUserMessage?.ts);
        expect(clock.now()).toBeGreaterThan(input.currentTurnTs);
        expect(input.evidence.current_user_message).toMatchObject({
          text: "The invoice is done.",
        });

        return {
          emission: {
            kind: "message",
            content: input.response,
          },
          claims: [],
          validations: [],
          verdict: "passed",
        };
      });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "The invoice is done.",
      });

      expect(result.emitted).toBe(true);
      expect(result.response).toBe("As you said, the invoice is done.");
      expect(guardRunSpy).toHaveBeenCalledOnce();
    } finally {
      guardRunSpy.mockRestore();
      await borg.close();
    }
  });

  it("persists user corrective preferences through identity with audience and stream source", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_175_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        "I will adjust that pattern.",
        createCommitmentJudgeResponse([]),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository" | "identityService">;
    };
    const addCommitmentSpy = vi.spyOn(internal.deps.identityService, "addCommitment");

    try {
      await borg.turn({
        userMessage: "You keep doing those little closing lines. Stop that.",
        audience: "Sam",
      });

      const userEntry = borg.stream.tail(10).find((entry) => entry.kind === "user_msg");
      const samEntityId = internal.deps.entityRepository.findByName("Sam");
      const addInput = addCommitmentSpy.mock.calls[0]?.[0];
      const commitments = borg.commitments.list({
        activeOnly: true,
        audience: "Sam",
      });

      expect(addInput).toMatchObject({
        type: "preference",
        directive: "Do not add ritual closing lines when the conversation is open.",
        priority: 8,
        restrictedAudience: samEntityId,
        sourceStreamEntryIds: [userEntry?.id],
      });
      expect(commitments).toEqual([
        expect.objectContaining({
          restricted_audience: samEntityId,
          source_stream_entry_ids: [userEntry?.id],
        }),
      ]);
    } finally {
      await borg.close();
    }
  });

  it("applies corrective slot negations and sanitizes pending actions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_175_500);
    const llm = new FakeLLMClient();
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<
        BorgDependencies,
        "entityRepository" | "relationalSlotRepository" | "workingMemoryStore"
      >;
    };

    try {
      const tom = internal.deps.entityRepository.resolve("Tom");
      internal.deps.relationalSlotRepository.applyAssertion({
        subject_entity_id: tom,
        slot_key: "partner.name",
        asserted_value: "Sarah",
        source_stream_entry_ids: [createStreamEntryId()],
      });
      const workingMemory = internal.deps.workingMemoryStore.load("default" as never);
      internal.deps.workingMemoryStore.save({
        ...workingMemory,
        pending_actions: [
          {
            description: "Track whether Tom raises the planning comment with Sarah directly",
            next_action: "Ask Sarah if Tom brings it up",
          },
        ],
        updated_at: clock.now(),
      });
      llm.pushResponse(
        createCorrectivePreferenceResponse({
          classification: "none",
          reason: "The user rejected a stored relational name.",
          confidence: 0.95,
          slot_negations: [
            {
              subject_entity_id: tom,
              slot_key: "partner.name",
              rejected_value: "Sarah",
              source_stream_entry_ids: [createStreamEntryId()],
              confidence: 0.95,
            },
          ],
        }),
      );
      llm.pushResponse("I will avoid using that name.");
      llm.pushResponse(createEmptyReflectionResponse());

      await borg.turn({
        userMessage: "Her name is not Sarah.",
      });

      const slot = internal.deps.relationalSlotRepository.findBySubjectAndKey(tom, "partner.name");
      const nextWorkingMemory = internal.deps.workingMemoryStore.load("default" as never);

      expect(slot?.state).toBe("quarantined");
      expect(nextWorkingMemory.pending_actions).toEqual([
        {
          description: "Track whether Tom raises the planning comment with your partner directly",
          next_action: "Ask your partner if Tom brings it up",
        },
      ]);
    } finally {
      await borg.close();
    }
  });

  it("promotes user goals through identity with audience, stream source, and initial step", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_500);
    const targetAt = 1_800_100_000_000;
    const stepDueAt = 1_800_050_000_000;
    const llm = new FakeLLMClient({
      responses: [
        createGoalPromotionResponse([
          {
            description: "Help the user keep the Monday postmortem straight",
            priority: 9,
            target_at: targetAt,
            reason: "The user asked Borg to help keep the postmortem organized.",
            confidence: 0.91,
            initial_step: {
              description: "Ask what must be included in the postmortem",
              kind: "ask_user",
              due_at: stepDueAt,
              rationale: "Borg needs the postmortem constraints to help track it.",
            },
          },
        ]),
        "I will keep the postmortem straight.",
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository" | "identityService"> & {
        executiveStepsRepository: ExecutiveStepsRepository;
      };
    };
    const addGoalSpy = vi.spyOn(internal.deps.identityService, "addGoal");

    try {
      await borg.turn({
        userMessage: "Write postmortem Monday, help me keep this straight.",
        audience: "Sam",
      });

      const userEntry = borg.stream.tail(10).find((entry) => entry.kind === "user_msg");
      const samEntityId = internal.deps.entityRepository.findByName("Sam");
      const addInput = addGoalSpy.mock.calls[0]?.[0];
      const goals = borg.self.goals.list({
        status: "active",
        visibleToAudienceEntityId: samEntityId,
      });
      const promotedGoal = goals.find(
        (goal) => goal.description === "Help the user keep the Monday postmortem straight",
      );
      const finalizerSystem = systemText(firstFinalizerRequest(llm.requests));

      expect(addInput).toMatchObject({
        description: "Help the user keep the Monday postmortem straight",
        priority: 9,
        status: "active",
        targetAt,
        audienceEntityId: samEntityId,
        sourceStreamEntryIds: [userEntry?.id],
      });
      expect(promotedGoal).toMatchObject({
        status: "active",
        target_at: targetAt,
        audience_entity_id: samEntityId,
        source_stream_entry_ids: [userEntry?.id],
      });
      expect(promotedGoal).toBeDefined();
      expect(
        internal.deps.executiveStepsRepository.list(promotedGoal!.id).map((step) => ({
          goal_id: step.goal_id,
          description: step.description,
          kind: step.kind,
          due_at: step.due_at,
        })),
      ).toEqual([
        {
          goal_id: promotedGoal!.id,
          description: "Ask what must be included in the postmortem",
          kind: "ask_user",
          due_at: stepDueAt,
        },
      ]);
      expect(finalizerSystem).toContain("Help the user keep the Monday postmortem straight");
    } finally {
      await borg.close();
    }
  });

  it("persists current-turn action states before deliberation runs", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_550);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "none",
          reason: "No durable correction detected.",
          confidence: 0,
        }),
        Object.assign(
          (options: LLMCompleteOptions) => {
            expect(options.budget).toBe("action-state-extractor");
            expect(options.model).toBe("test-recall");
            const payload = JSON.parse(String(options.messages[0]?.content ?? "{}")) as {
              current_user_stream_entry_id: string;
            };

            return createActionStateResponse([
              {
                description: "booked the tutor Tuesday 7pm",
                state: "completed",
                evidence_stream_entry_ids: [payload.current_user_stream_entry_id],
                confidence: 0.95,
              },
            ]);
          },
          { budget: "action-state-extractor" },
        ),
        createGoalPromotionResponse([]),
        "I see the tutor booking is done.",
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const originalRun = Deliberator.prototype.run;
    const runSpy = vi.spyOn(Deliberator.prototype, "run").mockImplementation(function (
      this: Deliberator,
      ...args: Parameters<Deliberator["run"]>
    ) {
      expect(borg.actions.list({ state: "completed" })).toEqual([
        expect.objectContaining({
          description: "booked the tutor Tuesday 7pm",
          state: "completed",
        }),
      ]);

      return originalRun.apply(this, args);
    });

    try {
      await borg.turn({
        userMessage: "I booked the tutor Tuesday 7pm.",
        stakes: "low",
      });

      expect(runSpy).toHaveBeenCalledOnce();
      expect(borg.actions.list({ state: "completed" })).toEqual([
        expect.objectContaining({
          description: "booked the tutor Tuesday 7pm",
          provenance_stream_entry_ids: [expect.any(String)],
        }),
      ]);
    } finally {
      runSpy.mockRestore();
      await borg.close();
    }
  });

  it("persists at most three promoted goals from a five-candidate extraction", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_600);
    const llm = new FakeLLMClient({
      responses: [
        createGoalPromotionResponse([
          {
            description: "Help the user track the launch checklist",
            confidence: 0.95,
          },
          {
            description: "Help the user prepare the investor update",
            confidence: 0.94,
          },
          {
            description: "Help the user schedule the design review",
            confidence: 0.93,
          },
          {
            description: "Help the user collect beta feedback",
            confidence: 0.92,
          },
          {
            description: "Help the user plan the onboarding pass",
            confidence: 0.91,
          },
        ]),
        "I will keep the active goals focused.",
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository" | "identityService">;
    };
    const addGoalSpy = vi.spyOn(internal.deps.identityService, "addGoal");

    try {
      await borg.turn({
        userMessage: "Keep track of launch, investor, design, beta, and onboarding work.",
        audience: "Sam",
      });

      const samEntityId = internal.deps.entityRepository.findByName("Sam");
      const goals = borg.self.goals.list({
        status: "active",
        visibleToAudienceEntityId: samEntityId,
      });

      expect(addGoalSpy).toHaveBeenCalledTimes(3);
      expect(goals.map((goal) => goal.description)).toEqual([
        "Help the user track the launch checklist",
        "Help the user prepare the investor update",
        "Help the user schedule the design review",
      ]);
    } finally {
      await borg.close();
    }
  });

  it("emits a goal promotion degraded trace event when extraction fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const tracePath = join(tempDir, "goal-promotion-degraded.jsonl");
    const clock = new ManualClock(1_800_000_176_650);
    const llm = new FakeLLMClient({
      responses: [
        (options: LLMCompleteOptions) => {
          expect(options.budget).toBe("goal-promotion-extractor");
          throw new Error("goal promotion transport failed");
        },
        "I will continue without promoting a goal.",
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, undefined, {
      tracerPath: tracePath,
      env: {
        BORG_TRACE_PROMPTS: "1",
      },
    });

    try {
      await borg.turn({
        userMessage: "Keep this goal in view for later.",
        audience: "Sam",
      });
    } finally {
      await borg.close();
    }

    const degradedEvent = readTraceEvents(tracePath).find(
      (event) => event.event === "goal_promotion_extractor_degraded",
    );

    expect(degradedEvent).toMatchObject({
      event: "goal_promotion_extractor_degraded",
      reason: "llm_failed",
      error: "goal promotion transport failed",
    });
    expect(degradedEvent?.turnId).toEqual(expect.any(String));
  });

  it("does not create a duplicate goal when the extractor points at an existing goal", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_700);
    const existingGoalId = createGoalId();
    const llm = new FakeLLMClient({
      responses: [
        createGoalPromotionResponse([
          {
            description: "Help the user track their italki shortlist",
            duplicate_of_goal_id: existingGoalId,
            confidence: 0.95,
          },
        ]),
        "I will keep it in mind.",
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);
    const internal = borg as unknown as {
      deps: Pick<BorgDependencies, "entityRepository">;
    };

    try {
      const samEntityId = internal.deps.entityRepository.resolve("Sam");
      const existingGoal = borg.self.goals.add({
        id: existingGoalId,
        description: "Help the user track their italki shortlist",
        priority: 8,
        audienceEntityId: samEntityId,
        provenance: {
          kind: "manual",
        },
      });

      await borg.turn({
        userMessage: "Remind me about italki later.",
        audience: "Sam",
      });

      const goals = borg.self.goals.list({
        status: "active",
        visibleToAudienceEntityId: samEntityId,
      });

      expect(goals.map((goal) => goal.id)).toEqual([existingGoal.id]);
    } finally {
      await borg.close();
    }
  });

  it("enforces a corrective preference on the same turn by rewriting a violation", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_176_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        "Sleep well.",
        createDynamicCommitmentJudgeResponse("The response repeats the corrected closing pattern."),
        "I will leave it there.",
        createCommitmentJudgeResponse([]),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "You keep doing those little closing lines. Stop that.",
        audience: "Sam",
      });

      expect(result.emitted).toBe(true);
      expect(result.response).toBe("I will leave it there.");
      expect(llm.requests.map((request) => request.budget)).toContain("commitment-revision");
    } finally {
      await borg.close();
    }
  });

  it("suppresses pure corrective-preference violations when revision still violates", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_177_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        "Sleep well.",
        createDynamicCommitmentJudgeResponse("The response repeats the corrected closing pattern."),
        "Sleep well.",
        createDynamicCommitmentJudgeResponse("The revision still repeats the corrected pattern."),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      const result = await borg.turn({
        userMessage: "You keep doing those little closing lines. Stop that.",
        audience: "Sam",
      });
      const suppressionEntry = borg.stream
        .tail(10)
        .find((entry) => entry.kind === "agent_suppressed");

      expect(result.emitted).toBe(false);
      expect(result.response).toBe("");
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "commitment_revision_failed",
      });
      expect(suppressionEntry?.content).toMatchObject({
        reason: "commitment_revision_failed",
      });
    } finally {
      await borg.close();
    }
  });

  it("loads durable corrective commitments on later turns", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_178_000);
    const llm = new FakeLLMClient({
      responses: [
        createCorrectivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is open.",
          priority: 8,
          reason: "The user named a future response pattern to stop.",
          confidence: 0.9,
        }),
        "I will adjust that pattern.",
        createCommitmentJudgeResponse([]),
        createEmptyReflectionResponse(),
        "Sleep well.",
        createDynamicCommitmentJudgeResponse(
          "The later response repeats the durable corrected pattern.",
        ),
        "I will stop here.",
        createCommitmentJudgeResponse([]),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock);

    try {
      await borg.turn({
        userMessage: "You keep doing those little closing lines. Stop that.",
        audience: "Sam",
      });
      clock.advance(5_000);

      const result = await borg.turn({
        userMessage: "Continue with the actual topic.",
        audience: "Sam",
      });

      expect(result.response).toBe("I will stop here.");
      expect(llm.requests.filter((request) => request.budget === "commitment-judge")).toHaveLength(
        3,
      );
    } finally {
      await borg.close();
    }
  });

  it("runs the generation gate before retrieval and finalization under active stop state", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_800_000_200_000);
    const embeddingClient = new CountingEmbeddingClient();
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "I will stop responding until you bring substantive content.",
          input_tokens: 8,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        createStopCommitmentResponse({
          classification: "stop_until_substantive_content",
        }),
        createEmptyReflectionResponse(),
        createGenerationGateResponse({
          decision: "suppress",
          substantive: false,
          reason: "The user sent another minimal probe under an active stop.",
        }),
      ],
    });
    const borg = await openTestBorg(tempDir, llm, clock, embeddingClient);

    try {
      await borg.turn({
        userMessage: "Stop responding if I keep sending filler.",
      });
      embeddingClient.embedTexts.length = 0;
      embeddingClient.embedBatchTexts.length = 0;
      const result = await borg.turn({
        userMessage: "No.",
      });
      const tailKinds = borg.stream.tail(6).map((entry) => entry.kind);

      expect(result.emitted).toBe(false);
      expect(result.response).toBe("");
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "active_discourse_stop",
      });
      expect(tailKinds.slice(-3)).toEqual(["user_msg", "perception", "agent_suppressed"]);
      expect(embeddingClient.embedTexts).toEqual([]);
      expect(embeddingClient.embedBatchTexts).toEqual([]);
      expect(borg.workmem.load().discourse_state?.stop_until_substantive_content).toMatchObject({
        provenance: "self_commitment_extractor",
      });
    } finally {
      await borg.close();
    }
  });
});
