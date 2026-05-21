import { describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { CommitmentRecord, EntityRepository } from "../../memory/commitments/index.js";
import type { CommitmentId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { CommitmentGuardRunner } from "./guard-runner.js";

const commitmentId = "cmt_abcdefghijklmnop" as CommitmentId;

function makeCommitment(overrides: Partial<CommitmentRecord> = {}): CommitmentRecord {
  return {
    id: commitmentId,
    type: "boundary",
    kind: "boundary",
    directive_family: "launch_date_boundary",
    closure_pressure_relevance: "neutral",
    directive: "Do not discuss launch dates.",
    priority: 10,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    provenance: {
      kind: "system",
    },
    created_at: 1_000,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    last_reinforced_at: 1_000,
    ...overrides,
  };
}

function verdictResponse(violations: unknown[]): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_commitment",
        name: "EmitCommitmentViolations",
        input: {
          violations,
        },
      },
    ],
  };
}

function makeRunner(
  tracer: TurnTracer,
  mode?: "enforce" | "shadow",
  options: { regenerateBeforeSuppress?: boolean; rewriteOnViolation?: boolean } = {},
) {
  return new CommitmentGuardRunner({
    detectionModel: "judge-model",
    rewriteModel: "rewrite-model",
    ...(mode === undefined ? {} : { mode }),
    ...(options.regenerateBeforeSuppress === undefined
      ? {}
      : { regenerateBeforeSuppress: options.regenerateBeforeSuppress }),
    ...(options.rewriteOnViolation === undefined
      ? {}
      : { rewriteOnViolation: options.rewriteOnViolation }),
    entityRepository: {
      get: vi.fn(() => null),
    } as unknown as EntityRepository,
    tracer,
  });
}

describe("CommitmentGuardRunner", () => {
  it("defaults an omitted mode to enforce in trace output", async () => {
    const llm = new FakeLLMClient();
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    const result = await makeRunner(tracer, undefined).run({
      turnId: "turn-default-mode",
      llmClient: llm,
      response: "No commitments active.",
      userMessage: "Hello",
      cognitionInput: "Hello",
      origin: "user",
      autonomyTrigger: null,
      commitments: [],
      relevantEntities: [],
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "No commitments active.",
    });
    expect(tracer.emit).toHaveBeenCalledWith("commitment_check.completed", {
      turnId: "turn-default-mode",
      mode: "enforce",
      verdict: "passed",
      wouldHaveVerdict: "passed",
      rewriteTriggered: false,
      violationCount: 0,
    });
  });

  it("uses the raw user message and untrusted cognition context for autonomous turns", async () => {
    const llm = new FakeLLMClient({
      responses: [verdictResponse([])],
    });
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    const result = await makeRunner(tracer).run({
      turnId: "turn-1",
      llmClient: llm,
      response: "No launch dates.",
      userMessage: "Raw autonomous wake",
      cognitionInput: "Formatted autonomy context",
      origin: "autonomous",
      autonomyTrigger: {
        source_name: "daily",
        source_type: "trigger",
        event_id: "evt-1",
        sort_ts: 1_000,
        payload: {
          topic: "launch",
        },
      },
      commitments: [makeCommitment()],
      relevantEntities: ["Atlas"],
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "No launch dates.",
    });
    expect(llm.requests[0]?.messages[0]?.content).toContain("User message: Raw autonomous wake");
    expect(llm.requests[0]?.messages[0]?.content).toContain(
      "<borg_untrusted_autonomy_context>\nFormatted autonomy context\n</borg_untrusted_autonomy_context>",
    );
    expect(tracer.emit).toHaveBeenCalledWith("commitment_check.completed", {
      turnId: "turn-1",
      mode: "enforce",
      verdict: "passed",
      wouldHaveVerdict: "passed",
      rewriteTriggered: false,
      violationCount: 0,
    });
  });

  it("requests regeneration for enforce-eligible boundary violations by default", async () => {
    const violation = {
      commitment_id: commitmentId,
      reason: "Discloses launch date.",
      confidence: 0.9,
    };
    const llm = new FakeLLMClient({
      responses: [verdictResponse([violation])],
    });
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    const result = await makeRunner(tracer).run({
      turnId: "turn-2",
      llmClient: llm,
      response: "Launch is tomorrow.",
      userMessage: "When is launch?",
      cognitionInput: "When is launch?",
      origin: "user",
      autonomyTrigger: null,
      commitments: [makeCommitment()],
      relevantEntities: [],
    });

    expect(result.emission).toEqual({
      kind: "requires_regeneration",
      reason: "commitment_violation",
      regeneration: expect.objectContaining({
        violationCount: 1,
        commitmentIds: [commitmentId],
        promptSection: expect.stringContaining("Do not discuss launch dates."),
      }),
    });
    expect(llm.requests.map((request) => request.budget)).toEqual(["commitment-judge"]);
    expect(tracer.emit).toHaveBeenCalledWith("commitment_guard.regeneration_requested", {
      turnId: "turn-2",
      mode: "enforce",
      verdict: "requires_regeneration",
      violationCount: 1,
      commitmentIds: [commitmentId],
      commitmentKinds: ["boundary"],
    });
    expect(tracer.emit).toHaveBeenCalledWith("commitment_check.completed", {
      turnId: "turn-2",
      mode: "enforce",
      verdict: "requires_regeneration",
      wouldHaveVerdict: "suppressed",
      wouldHaveSuppressionReason: "commitment_violation",
      rewriteTriggered: false,
      violationCount: 1,
    });
  });

  it("can suppress enforce-eligible boundary violations without regeneration behind the flag", async () => {
    const violation = {
      commitment_id: commitmentId,
      reason: "Discloses launch date.",
      confidence: 0.9,
    };
    const llm = new FakeLLMClient({
      responses: [verdictResponse([violation])],
    });
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    const result = await makeRunner(tracer, "enforce", {
      regenerateBeforeSuppress: false,
    }).run({
      turnId: "turn-no-regenerate",
      llmClient: llm,
      response: "Launch is tomorrow.",
      userMessage: "When is launch?",
      cognitionInput: "When is launch?",
      origin: "user",
      autonomyTrigger: null,
      commitments: [makeCommitment()],
      relevantEntities: [],
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "commitment_violation",
    });
    expect(tracer.emit).toHaveBeenCalledWith("commitment_guard.enforce_suppression", {
      turnId: "turn-no-regenerate",
      mode: "enforce",
      verdict: "suppressed",
      reason: "commitment_violation",
      rewriteTriggered: false,
      violationCount: 1,
      commitmentIds: [commitmentId],
      commitmentKinds: ["boundary"],
    });
  });

  it("observes non-critical participant preferences in shadow without suppressing", async () => {
    const violation = {
      commitment_id: commitmentId,
      reason: "Discloses launch date.",
      confidence: 0.9,
    };
    const original = "Launch is tomorrow.";
    const llm = new FakeLLMClient({
      responses: [verdictResponse([violation])],
    });
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    const result = await makeRunner(tracer).run({
      turnId: "turn-shadow",
      llmClient: llm,
      response: original,
      userMessage: "When is launch?",
      cognitionInput: "When is launch?",
      origin: "user",
      autonomyTrigger: null,
      commitments: [
        makeCommitment({
          type: "preference",
          kind: "participant_preference",
        }),
      ],
      relevantEntities: [],
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: original,
    });
    expect(result.revised).toBe(false);
    expect(llm.requests.map((request) => request.budget)).toEqual(["commitment-judge"]);
    expect(tracer.emit).toHaveBeenCalledWith("commitment_guard.shadow_observation", {
      turnId: "turn-shadow",
      mode: "shadow",
      verdict: "passed",
      wouldHaveVerdict: "suppressed",
      wouldHaveSuppressionReason: "commitment_violation",
      rewriteTriggered: false,
      violationCount: 1,
      commitmentIds: [commitmentId],
      commitmentKinds: ["participant_preference"],
    });
    expect(tracer.emit).toHaveBeenCalledWith("commitment_check.completed", {
      turnId: "turn-shadow",
      mode: "enforce",
      verdict: "passed",
      wouldHaveVerdict: "passed",
      rewriteTriggered: false,
      violationCount: 0,
      shadowViolationCount: 1,
    });
  });

  it("keeps rewrite disabled by default but allows enabling it for critical kinds", async () => {
    const violation = {
      commitment_id: commitmentId,
      reason: "Discloses launch date.",
      confidence: 0.9,
    };
    const llm = new FakeLLMClient({
      responses: [
        verdictResponse([violation]),
        {
          text: "I can't discuss launch timing.",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        verdictResponse([]),
      ],
    });
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    const result = await makeRunner(tracer, "enforce", {
      regenerateBeforeSuppress: false,
      rewriteOnViolation: true,
    }).run({
      turnId: "turn-rewrite-enabled",
      llmClient: llm,
      response: "Launch is tomorrow.",
      userMessage: "When is launch?",
      cognitionInput: "When is launch?",
      origin: "user",
      autonomyTrigger: null,
      commitments: [makeCommitment()],
      relevantEntities: [],
    });

    expect(result.revised).toBe(true);
    expect(result.emission).toEqual({
      kind: "message",
      content: "I can't discuss launch timing.",
    });
    expect(llm.requests.map((request) => request.budget)).toEqual([
      "commitment-judge",
      "commitment-revision",
      "commitment-judge",
    ]);
    expect(tracer.emit).toHaveBeenCalledWith("commitment_guard.enforce_rewrite", {
      turnId: "turn-rewrite-enabled",
      mode: "enforce",
      verdict: "rewritten",
      rewriteTriggered: true,
      violationCount: 1,
      commitmentIds: [commitmentId],
      commitmentKinds: ["boundary"],
    });
  });
});
