import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import type { CommitmentRecord } from "../../memory/commitments/index.js";
import type { ClosureLoopState } from "../../memory/working/index.js";
import { createCommitmentId, createStreamEntryId } from "../../util/ids.js";
import {
  CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
  ClosurePressureGuard,
  type ClosureResponseAudit,
} from "./closure-pressure-guard.js";

function closureAuditResponse(audit: ClosureResponseAudit): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_closure_response_audit",
        name: CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
        input: audit,
      },
    ],
  };
}

function makeCommitment(directiveFamily = "honor_pause_not_closure"): CommitmentRecord {
  return {
    id: createCommitmentId(),
    type: "preference",
    directive_family: directiveFamily,
    directive: "Do not convert open pauses into closure.",
    priority: 80,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    provenance: { kind: "manual" },
    source_stream_entry_ids: [createStreamEntryId()],
    created_at: 1_000,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    last_reinforced_at: 1_000,
  };
}

function namedClosureLoop(): ClosureLoopState {
  return {
    status: "named",
    source_stream_entry_ids: [createStreamEntryId()],
    reason: "User named the closure loop.",
    since_turn: 3,
    named_at_turn: 4,
  };
}

describe("ClosurePressureGuard", () => {
  it("removes closure tails when a no-closure commitment is active", async () => {
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [
            {
              text: "Go read.",
              kind: "imperative_closer",
              rationale: "Imperative closer after substantive content.",
            },
          ],
          response_shape: "mixed",
          reason: "Substantive content plus closure tail.",
        }),
        "The shelf test is the right move.",
      ],
    });
    const tracer = {
      enabled: true,
      includePayloads: true,
      emit: vi.fn(),
    };
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      tracer,
    });

    const result = await guard.run({
      turnId: "turn-closure-tail",
      response: "The shelf test is the right move. Go read.",
      activeCommitments: [makeCommitment()],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move.",
    });
    expect(result.verdict).toBe("rewritten");
    expect(result.removed_spans).toEqual(["Go read."]);
    expect(tracer.emit).toHaveBeenCalledWith(
      "closure_response_guard",
      expect.objectContaining({
        verdict: "rewritten",
        removed_spans: ["Go read."],
        reason: "closure_spans_removed",
      }),
    );
  });

  it("suppresses closure-only responses under an active no-closure commitment", async () => {
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [
            {
              text: "Held. Book.",
              kind: "quotable_closing_tail",
              rationale: "The entire response is a closing tag.",
            },
          ],
          response_shape: "closure_only",
          reason: "Only closure pressure remains.",
        }),
      ],
    });
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
    });

    const result = await guard.run({
      turnId: "turn-closure-only",
      response: "Held. Book.",
      activeCommitments: [makeCommitment("no_sleep_closure")],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "closure_pressure_only",
    });
    expect(result.verdict).toBe("suppressed");
    expect(llm.requests.map((request) => request.budget)).toEqual(["closure-response-auditor"]);
  });

  it("passes responses with no closure-shaped content", async () => {
    const response = "The reason the soup example works is that error becomes signal.";
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [],
          response_shape: "no_closure",
          reason: "No closure-function span.",
        }),
      ],
    });
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
    });

    const result = await guard.run({
      turnId: "turn-no-closure",
      response,
      activeCommitments: [makeCommitment()],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: response,
    });
    expect(result.verdict).toBe("passed");
  });

  it("passes closure tails when no no-closure preference is active", async () => {
    const response = "The shelf test is the right move. Go read.";
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [
            {
              text: "Go read.",
              kind: "imperative_closer",
              rationale: "Imperative closer after substantive content.",
            },
          ],
          response_shape: "mixed",
          reason: "Substantive content plus closure tail.",
        }),
      ],
    });
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
    });

    const result = await guard.run({
      turnId: "turn-no-active-preference",
      response,
      activeCommitments: [],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: response,
    });
    expect(result.verdict).toBe("passed");
    expect(llm.requests.map((request) => request.budget)).toEqual(["closure-response-auditor"]);
  });

  it("enforces named closure-loop discourse state even without a commitment", async () => {
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [
            {
              text: "Standing by.",
              kind: "aphoristic_valediction",
              rationale: "Valediction after an already named closure loop.",
            },
          ],
          response_shape: "mixed",
          reason: "Substantive content plus valediction.",
        }),
        "The result is still the same: use the current shelf.",
      ],
    });
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
    });

    const result = await guard.run({
      turnId: "turn-named-loop",
      response: "The result is still the same: use the current shelf. Standing by.",
      activeCommitments: [],
      closureLoop: namedClosureLoop(),
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The result is still the same: use the current shelf.",
    });
    expect(result.verdict).toBe("rewritten");
  });
});
