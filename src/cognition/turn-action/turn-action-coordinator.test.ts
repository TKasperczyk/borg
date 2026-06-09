import { describe, expect, it, vi } from "vitest";

import { FakeEmbeddingClient } from "../../embeddings/index.js";
import type { LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { CommitmentRecord, EntityRepository } from "../../memory/commitments/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import {
  makeCommitmentRecord,
  makeTestTurnTraceRecorder,
  makeToolUseCompleteResult,
} from "../../test-support/factories/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { CommitmentGuardRunner } from "../commitments/guard-runner.js";
import type { DeliberationResult } from "../deliberation/types.js";
import type { PendingTurnEmission } from "../generation/types.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { TurnActionCoordinator } from "./turn-action-coordinator.js";

type CommitmentViolationFixture = {
  commitment_id: string;
  reason: string;
  confidence: number;
  violating_span_or_topic?: string;
};
type MessageEmissionMetadata = Partial<
  Omit<Extract<PendingTurnEmission, { kind: "message" }>, "kind" | "content">
>;

function commitmentVerdictResponse(
  violations: readonly CommitmentViolationFixture[],
): LLMCompleteResult {
  return makeToolUseCompleteResult({
    toolName: "EmitCommitmentViolations",
    toolInput: { violations },
  });
}

function textResponse(text: string): LLMCompleteResult {
  return {
    text,
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "end_turn",
    tool_calls: [],
  };
}

function makeDeliberation(
  response: string,
  regenerateFinalResponse?: DeliberationResult["regenerateFinalResponse"],
  emissionMetadata: MessageEmissionMetadata = {},
): DeliberationResult {
  return {
    path: "system_1",
    response,
    emitted: true,
    emission: {
      kind: "message",
      content: response,
      ...emissionMetadata,
    },
    emissionRecommendation: "emit",
    thoughtStreamEntryIds: [],
    thoughts: [],
    tool_calls: [],
    usage: {
      input_tokens: 1,
      output_tokens: 1,
      stop_reason: "tool_use",
    },
    decision_reason: "test fixture",
    retrievedEpisodes: [],
    referencedEpisodeIds: null,
    intents: [],
    thoughtsPersisted: false,
    ...(regenerateFinalResponse === undefined ? {} : { regenerateFinalResponse }),
  };
}

function makeCommitmentGuardRunner(input: {
  tracer: TurnTracer;
  regenerateBeforeSuppress?: boolean;
  rewriteOnViolation?: boolean;
}): CommitmentGuardRunner {
  return new CommitmentGuardRunner({
    detectionModel: "test-commitment-judge",
    rewriteModel: "test-commitment-rewriter",
    mode: "enforce",
    entityRepository: {
      get: () => null,
      findByName: () => null,
      resolve: () => null,
    } as unknown as EntityRepository,
    tracer: input.tracer,
    ...(input.regenerateBeforeSuppress === undefined
      ? {}
      : { regenerateBeforeSuppress: input.regenerateBeforeSuppress }),
    ...(input.rewriteOnViolation === undefined
      ? {}
      : { rewriteOnViolation: input.rewriteOnViolation }),
  });
}

async function runCoordinator(input: {
  llmClient: FakeLLMClient;
  tracer: TurnTracer;
  deliberation: DeliberationResult;
  commitments: readonly CommitmentRecord[];
  turnId?: string;
  commitmentGuardOptions?: {
    regenerateBeforeSuppress?: boolean;
    rewriteOnViolation?: boolean;
  };
}) {
  const postGenerationGuardRunner = {
    run: vi.fn(async ({ response }: { response: string }) => ({
      kind: "message" as const,
      content: response,
    })),
  };
  const coordinator = new TurnActionCoordinator({
    commitmentGuardRunner: makeCommitmentGuardRunner({
      tracer: input.tracer,
      ...input.commitmentGuardOptions,
    }),
    postGenerationGuardRunner,
    embeddingClient: new FakeEmbeddingClient(8),
    pendingActionJudgeModel: "test-pending-action-judge",
    clock: new FixedClock(20_000),
    tracer: input.tracer,
  });

  const result = await coordinator.run({
    llmClient: input.llmClient,
    turnId: input.turnId ?? "turn-commitment-regeneration",
    sessionId: DEFAULT_SESSION_ID,
    deliberation: input.deliberation,
    workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 20_000),
    userMessage: "Prepare the vendor-channel deployment status.",
    cognitionInput: "Prepare the vendor-channel deployment status.",
    origin: "user",
    autonomyTrigger: null,
    applicableCommitments: input.commitments,
    perceptionEntities: [],
    retrievedEpisodes: [],
    currentUserClosureKind: null,
    audienceEntityId: null,
  });

  return {
    result,
    postGenerationGuardRunner,
  };
}

function eventsNamed(tracer: ReturnType<typeof makeTestTurnTraceRecorder>, name: string) {
  return tracer.events.filter((entry) => entry.event === name);
}

describe("TurnActionCoordinator commitment regeneration", () => {
  it("delivers regenerated clean output after a critical boundary violation", async () => {
    const tracer = makeTestTurnTraceRecorder();
    const commitment = makeCommitmentRecord({
      kind: "boundary",
      type: "boundary",
      enforcement_class: "critical",
      critical_domain: "audience_scope",
      directive_family: "vendor_channel_codename",
      directive: "Do not disclose the private deployment codename in the vendor channel.",
      priority: 10,
    });
    const initialDraft =
      "The deploy checklist is green for the vendor handoff. The private deployment codename is ORCHID-17.";
    const cleanDraft = "The deploy checklist is green for the vendor handoff.";
    const violation = {
      commitment_id: commitment.id,
      reason: "The draft discloses the private deployment codename in the vendor channel.",
      confidence: 0.99,
      violating_span_or_topic: "ORCHID-17",
    };
    const llmClient = new FakeLLMClient({
      responses: [commitmentVerdictResponse([violation]), commitmentVerdictResponse([])],
    });
    const regenerateFinalResponse = vi.fn<
      NonNullable<DeliberationResult["regenerateFinalResponse"]>
    >(async () => makeDeliberation(cleanDraft));

    const { result } = await runCoordinator({
      llmClient,
      tracer,
      deliberation: makeDeliberation(initialDraft, regenerateFinalResponse),
      commitments: [commitment],
    });

    expect(result.actionEmission).toEqual({
      kind: "message",
      content: cleanDraft,
    });
    expect(result.actionResult.response).toBe(cleanDraft);
    expect(result.deliberation.response).toBe(cleanDraft);
    expect(regenerateFinalResponse).toHaveBeenCalledTimes(1);
    const regenerationPromptSection =
      regenerateFinalResponse.mock.calls[0]?.[0].additionalPromptSections[0] ?? "";
    expect(regenerationPromptSection).toContain("ORCHID-17");
    expect(regenerationPromptSection).toContain("Do not disclose the private deployment codename");
    expect(
      llmClient.requests.filter((request) => request.budget === "commitment-judge"),
    ).toHaveLength(2);
    expect(llmClient.requests.some((request) => request.budget === "commitment-revision")).toBe(
      false,
    );
    expect(eventsNamed(tracer, "commitment_guard.regeneration_requested")).toHaveLength(1);
    expect(eventsNamed(tracer, "commitment_guard.regeneration_succeeded")).toHaveLength(1);
    expect(eventsNamed(tracer, "commitment_guard.regeneration_failed")).toHaveLength(0);
  });

  it("uses regenerated finalizer discourse metadata as the final source of truth", async () => {
    const tracer = makeTestTurnTraceRecorder();
    const commitment = makeCommitmentRecord({
      kind: "boundary",
      type: "boundary",
      enforcement_class: "critical",
      critical_domain: "audience_scope",
      directive_family: "vendor_channel_codename",
      directive: "Do not disclose the private deployment codename in the vendor channel.",
      priority: 10,
    });
    const initialDraft =
      "I will stop responding until substantive content appears. The private deployment codename is ORCHID-17.";
    const cleanDraft = "The deploy checklist is green for the vendor handoff.";
    const violation = {
      commitment_id: commitment.id,
      reason: "The draft discloses the private deployment codename in the vendor channel.",
      confidence: 0.99,
      violating_span_or_topic: "ORCHID-17",
    };
    const llmClient = new FakeLLMClient({
      responses: [commitmentVerdictResponse([violation]), commitmentVerdictResponse([])],
    });
    const regenerateFinalResponse = vi.fn<
      NonNullable<DeliberationResult["regenerateFinalResponse"]>
    >(async () => makeDeliberation(cleanDraft));

    const { result } = await runCoordinator({
      llmClient,
      tracer,
      deliberation: makeDeliberation(initialDraft, regenerateFinalResponse, {
        discourse_control: {
          kind: "stop_until_substantive_content",
          reason: "Initial draft committed to stop.",
        },
      }),
      commitments: [commitment],
      turnId: "turn-regenerated-discourse-control-drop",
    });

    expect(result.actionEmission).toEqual({
      kind: "message",
      content: cleanDraft,
    });
    expect(regenerateFinalResponse).toHaveBeenCalledTimes(1);
  });

  it("preserves discourse metadata through the commitment rewrite path", async () => {
    const tracer = makeTestTurnTraceRecorder();
    const commitment = makeCommitmentRecord({
      kind: "boundary",
      type: "boundary",
      enforcement_class: "critical",
      critical_domain: "audience_scope",
      directive_family: "vendor_channel_codename",
      directive: "Do not disclose the private deployment codename in the vendor channel.",
      priority: 10,
    });
    const discourseControl = {
      kind: "stop_until_substantive_content" as const,
      reason: "The response commits Borg to wait for substantive content.",
    };
    const initialDraft =
      "I will stop responding until substantive content appears. The private deployment codename is ORCHID-17.";
    const rewrittenDraft = "I will stop responding until substantive content appears.";
    const violation = {
      commitment_id: commitment.id,
      reason: "The draft discloses the private deployment codename in the vendor channel.",
      confidence: 0.99,
      violating_span_or_topic: "ORCHID-17",
    };
    const llmClient = new FakeLLMClient({
      responses: [
        commitmentVerdictResponse([violation]),
        textResponse(rewrittenDraft),
        commitmentVerdictResponse([]),
      ],
    });

    const { result } = await runCoordinator({
      llmClient,
      tracer,
      deliberation: makeDeliberation(initialDraft, undefined, {
        discourse_control: discourseControl,
      }),
      commitments: [commitment],
      commitmentGuardOptions: {
        regenerateBeforeSuppress: false,
        rewriteOnViolation: true,
      },
      turnId: "turn-rewritten-discourse-control-preserved",
    });

    expect(result.actionEmission).toEqual({
      kind: "message",
      content: rewrittenDraft,
      discourse_control: discourseControl,
    });
    expect(result.actionResult.response).toBe(rewrittenDraft);
    expect(eventsNamed(tracer, "commitment_guard.enforce_rewrite")).toHaveLength(1);
  });

  it("suppresses when regeneration still violates a critical boundary", async () => {
    const tracer = makeTestTurnTraceRecorder();
    const commitment = makeCommitmentRecord({
      kind: "boundary",
      type: "boundary",
      enforcement_class: "critical",
      critical_domain: "audience_scope",
      directive_family: "vendor_channel_codename",
      directive: "Do not disclose the private deployment codename in the vendor channel.",
      priority: 10,
    });
    const initialDraft =
      "The deploy checklist is green, and the private deployment codename is ORCHID-17.";
    const regeneratedDraft = "Vendor status is ready; codename ORCHID-17 remains in scope.";
    const violation = {
      commitment_id: commitment.id,
      reason: "The draft discloses the private deployment codename in the vendor channel.",
      confidence: 0.99,
      violating_span_or_topic: "ORCHID-17",
    };
    const llmClient = new FakeLLMClient({
      responses: [commitmentVerdictResponse([violation]), commitmentVerdictResponse([violation])],
    });
    const regenerateFinalResponse = vi.fn(async () => makeDeliberation(regeneratedDraft));

    const { result } = await runCoordinator({
      llmClient,
      tracer,
      deliberation: makeDeliberation(initialDraft, regenerateFinalResponse),
      commitments: [commitment],
    });

    expect(result.actionEmission).toEqual({
      kind: "suppressed",
      reason: "commitment_violation_after_regenerate",
    });
    expect(result.actionResult.response).toBe("");
    expect(result.actionResult.emitted).toBe(false);
    expect(regenerateFinalResponse).toHaveBeenCalledTimes(1);
    expect(
      llmClient.requests.filter((request) => request.budget === "commitment-judge"),
    ).toHaveLength(2);
    expect(llmClient.requests.some((request) => request.budget === "commitment-revision")).toBe(
      false,
    );
    expect(eventsNamed(tracer, "commitment_guard.regeneration_requested")).toHaveLength(1);
    expect(eventsNamed(tracer, "commitment_guard.regeneration_succeeded")).toHaveLength(0);
    expect(eventsNamed(tracer, "commitment_guard.regeneration_failed")).toHaveLength(1);
    expect(eventsNamed(tracer, "commitment_guard.regeneration_failed")[0]?.data).toMatchObject({
      reason: "still_violates",
      suppressionReason: "commitment_violation_after_regenerate",
      violationCount: 1,
      commitmentIds: [commitment.id],
    });
    expect(eventsNamed(tracer, "commitment_guard.enforce_suppression")).toHaveLength(1);
    expect(eventsNamed(tracer, "commitment_guard.enforce_suppression")[0]?.data).toMatchObject({
      reason: "commitment_violation_after_regenerate",
    });
  });

  it("passes advisory commitment violations through without regeneration", async () => {
    const tracer = makeTestTurnTraceRecorder();
    const commitment = makeCommitmentRecord({
      kind: "process_norm",
      type: "preference",
      enforcement_class: "advisory",
      critical_domain: null,
      directive_family: "status_update_format",
      directive: "Use bullet points for status updates.",
      priority: 5,
    });
    const initialDraft =
      "The status update is that build passed and review is waiting on vendor signoff.";
    const violation = {
      commitment_id: commitment.id,
      reason: "The status update is prose instead of bullet points.",
      confidence: 0.9,
    };
    const llmClient = new FakeLLMClient({
      responses: [commitmentVerdictResponse([violation])],
    });
    const regenerateFinalResponse = vi.fn(async () =>
      makeDeliberation("This regeneration path should not be used."),
    );

    const { result } = await runCoordinator({
      llmClient,
      tracer,
      deliberation: makeDeliberation(initialDraft, regenerateFinalResponse),
      commitments: [commitment],
      turnId: "turn-advisory-commitment",
    });

    expect(result.actionEmission).toEqual({
      kind: "message",
      content: initialDraft,
    });
    expect(result.actionResult.response).toBe(initialDraft);
    expect(regenerateFinalResponse).not.toHaveBeenCalled();
    expect(
      llmClient.requests.filter((request) => request.budget === "commitment-judge"),
    ).toHaveLength(1);
    expect(eventsNamed(tracer, "commitment_guard.advisory_violation_observed")).toHaveLength(1);
    expect(eventsNamed(tracer, "commitment_guard.regeneration_requested")).toHaveLength(0);
    expect(eventsNamed(tracer, "commitment_guard.enforce_suppression")).toHaveLength(0);
  });

  it("gates legacy boundary commitments with null enforcement class as critical", async () => {
    const tracer = makeTestTurnTraceRecorder();
    const commitment = {
      ...makeCommitmentRecord({
        kind: "boundary",
        type: "boundary",
        directive_family: "vendor_customer_boundary",
        directive: "Do not disclose the confidential customer name in the vendor channel.",
        priority: 10,
      }),
      enforcement_class: null,
      critical_domain: null,
    } as unknown as CommitmentRecord;
    const initialDraft =
      "The vendor notes are ready. The confidential customer name is Northstar Bank.";
    const cleanDraft = "The vendor notes are ready without customer-identifying details.";
    const violation = {
      commitment_id: commitment.id,
      reason: "The draft discloses the confidential customer name in the vendor channel.",
      confidence: 0.97,
      violating_span_or_topic: "Northstar Bank",
    };
    const llmClient = new FakeLLMClient({
      responses: [commitmentVerdictResponse([violation]), commitmentVerdictResponse([])],
    });
    const regenerateFinalResponse = vi.fn(async () => makeDeliberation(cleanDraft));

    const { result } = await runCoordinator({
      llmClient,
      tracer,
      deliberation: makeDeliberation(initialDraft, regenerateFinalResponse),
      commitments: [commitment],
      turnId: "turn-legacy-critical-boundary",
    });

    expect(result.actionEmission).toEqual({
      kind: "message",
      content: cleanDraft,
    });
    expect(regenerateFinalResponse).toHaveBeenCalledTimes(1);
    expect(eventsNamed(tracer, "commitment_guard.regeneration_requested")).toHaveLength(1);
    expect(eventsNamed(tracer, "commitment_guard.regeneration_requested")[0]?.data).toMatchObject({
      commitmentEnforcementClasses: ["critical"],
      criticalDomains: ["audience_scope"],
    });
    expect(eventsNamed(tracer, "commitment_guard.regeneration_succeeded")).toHaveLength(1);
    expect(llmClient.requests.some((request) => request.budget === "commitment-revision")).toBe(
      false,
    );
  });
});
