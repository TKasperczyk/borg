import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import type { CommitmentRecord, EntityRepository } from "../../memory/commitments/index.js";
import type { CommitmentId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { CommitmentGuardRunner } from "./guard-runner.js";

const commitmentId = "cmt_abcdefghijklmnop" as CommitmentId;

function makeCommitment(): CommitmentRecord {
  return {
    id: commitmentId,
    type: "boundary",
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

function makeRunner(tracer: TurnTracer) {
  return new CommitmentGuardRunner({
    detectionModel: "judge-model",
    rewriteModel: "rewrite-model",
    entityRepository: {
      get: vi.fn(() => null),
    } as unknown as EntityRepository,
    tracer,
  });
}

describe("CommitmentGuardRunner", () => {
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
    expect(tracer.emit).toHaveBeenCalledWith("commitment_check", {
      turnId: "turn-1",
      verdict: "passed",
      rewriteTriggered: false,
      violationCount: 0,
    });
  });

  it("emits suppression trace when revision still violates a commitment", async () => {
    const violation = {
      commitment_id: commitmentId,
      reason: "Discloses launch date.",
      confidence: 0.9,
    };
    const llm = new FakeLLMClient({
      responses: [
        verdictResponse([violation]),
        {
          text: "Still a launch date.",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        verdictResponse([violation]),
      ],
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
      kind: "suppressed",
      reason: "commitment_revision_failed",
    });
    expect(tracer.emit).toHaveBeenCalledWith("commitment_check", {
      turnId: "turn-2",
      verdict: "suppressed",
      rewriteTriggered: true,
      violationCount: 1,
    });
  });
});
