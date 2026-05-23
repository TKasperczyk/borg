import { describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
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
    kind: "participant_preference",
    enforcement_class: "advisory",
    critical_domain: null,
    directive_family: directiveFamily,
    closure_pressure_relevance: "no_closure",
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

async function runClosureHistoryGuard(input: {
  currentUserClosureKind?: "substantive" | "user_requests_closure";
  entryTs: number;
  entryTurn: number;
  nowMs: number;
  currentTurn: number;
}) {
  const response = "The shelf test is the right move. Go read.";
  const llm = new FakeLLMClient({
    responses: [
      closureAuditResponse({
        spans: [
          {
            text: "Go read.",
            kind: "imperative_closer",
            rationale: "Imperative closer after recent closure pressure.",
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

  return guard.run({
    turnId: "turn-history-active",
    response,
    activeCommitments: [],
    closureLoop: null,
    closurePressureHistory: [
      {
        turn_id: "turn-prior",
        turn: input.entryTurn,
        reason: "span_removed",
        ts: input.entryTs,
      },
    ],
    currentUserClosureKind: input.currentUserClosureKind ?? "substantive",
    currentTurn: input.currentTurn,
    nowMs: input.nowMs,
  });
}

describe("ClosurePressureGuard", () => {
  it("observes mixed closure tails when a no-closure commitment is active", async () => {
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
      content: "The shelf test is the right move. Go read.",
    });
    expect(result.verdict).toBe("passed");
    expect(result.removed_spans).toEqual(["Go read."]);
    expect(llm.requests.map((request) => request.budget)).toEqual(["closure-response-auditor"]);
    const closureGuardCompletedEvent = tracer.emit.mock.calls.find(
      ([event]) => event === "closure_response_guard.completed",
    )?.[1];
    expect(closureGuardCompletedEvent).toEqual(
      expect.objectContaining({
        verdict: "passed",
        wouldHaveVerdict: "suppressed",
        removed_spans: ["Go read."],
        reason: "mixed_closure_observed",
        spans: [
          expect.objectContaining({
            text: "Go read.",
            kind: "imperative_closer",
          }),
        ],
      }),
    );
    expect(closureGuardCompletedEvent).not.toHaveProperty("closure_pressure_history_reason");
  });

  it("observes phrase-only mixed closure spans without suppressing the response", async () => {
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [
            {
              text: "go read",
              kind: "imperative_closer",
              rationale: "Phrase-only closer inside a setup sentence.",
            },
          ],
          response_shape: "mixed",
          reason: "The response is only a setup plus closure phrase.",
        }),
      ],
    });
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
    });

    const result = await guard.run({
      turnId: "turn-closure-phrase-gap",
      response: "Anyway, go read.",
      activeCommitments: [makeCommitment()],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "Anyway, go read.",
    });
    expect(result.verdict).toBe("passed");
    expect(result.removed_spans).toEqual(["go read"]);
  });

  it("audits mixed closure spans and emits the original response in shadow mode", async () => {
    const original = "The shelf test is the right move. Go read.";
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
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      mode: "shadow",
      tracer,
    });

    const result = await guard.run({
      turnId: "turn-closure-shadow",
      response: original,
      activeCommitments: [makeCommitment()],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: original,
    });
    expect(result.verdict).toBe("passed");
    expect(result.removed_spans).toEqual(["Go read."]);
    expect(llm.requests.map((request) => request.budget)).toEqual(["closure-response-auditor"]);
    expect(tracer.emit).toHaveBeenCalledWith(
      "closure_response_guard.completed",
      expect.objectContaining({
        mode: "shadow",
        verdict: "passed",
        wouldHaveVerdict: "suppressed",
        removed_spans: ["Go read."],
        reason: "mixed_closure_observed",
        spans: [
          expect.objectContaining({
            text: "Go read.",
            kind: "imperative_closer",
          }),
        ],
      }),
    );
  });

  it("passes closure-only responses under an advisory no-closure commitment without a named loop", async () => {
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
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
      tracer,
    });

    const result = await guard.run({
      turnId: "turn-closure-only",
      response: "Held. Book.",
      activeCommitments: [makeCommitment("no_sleep_closure")],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "Held. Book.",
    });
    expect(result.verdict).toBe("passed");
    expect(result.reason).toBe("closure_only_observed");
    expect(result.removed_spans).toEqual(["Held. Book."]);
    expect(tracer.emit).toHaveBeenCalledWith(
      "closure_response_guard.completed",
      expect.objectContaining({
        mode: "enforce",
        verdict: "passed",
        wouldHaveVerdict: "suppressed",
        wouldHaveSuppressionReason: "closure_pressure_only",
        reason: "closure_only_observed",
        response_shape: "closure_only",
      }),
    );
    expect(llm.requests.map((request) => request.budget)).toEqual(["closure-response-auditor"]);
  });

  it("suppresses closure-only responses when the closure loop is named", async () => {
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
      turnId: "turn-closure-only-named-loop",
      response: "Held. Book.",
      activeCommitments: [makeCommitment("no_sleep_closure")],
      closureLoop: namedClosureLoop(),
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "closure_pressure_only",
      closure_pressure_history_reason: "span_removed",
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
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
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
      tracer,
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
    expect(tracer.emit).toHaveBeenCalledWith(
      "closure_response_guard.completed",
      expect.objectContaining({
        mode: "shadow",
        verdict: "passed",
        wouldHaveVerdict: "passed",
        removed_spans: [],
        reason: "no_active_closure_preference",
      }),
    );
  });

  it("treats recent closure-pressure history as an active closure constraint", async () => {
    const result = await runClosureHistoryGuard({
      entryTs: 1_000,
      entryTurn: 10,
      nowMs: 60_000,
      currentTurn: 12,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move. Go read.",
    });
    expect(result.active_closure_commitments).toContain("closure_pressure_history");
  });

  it("does not enforce recent closure-pressure history when the user explicitly requests closure", async () => {
    const result = await runClosureHistoryGuard({
      currentUserClosureKind: "user_requests_closure",
      entryTs: 1_000,
      entryTurn: 10,
      nowMs: 60_000,
      currentTurn: 12,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move. Go read.",
    });
    expect(result.active_closure_commitments).not.toContain("closure_pressure_history");
  });

  it("does not enforce closure-pressure history older than ten minutes", async () => {
    const result = await runClosureHistoryGuard({
      entryTs: 1_000,
      entryTurn: 10,
      nowMs: 661_001,
      currentTurn: 12,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move. Go read.",
    });
  });

  it("does not enforce closure-pressure history more than five turns old", async () => {
    const result = await runClosureHistoryGuard({
      entryTs: 1_000,
      entryTurn: 5,
      nowMs: 60_000,
      currentTurn: 11,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move. Go read.",
    });
  });

  it("fails open when the auditor throws under an active no-closure commitment in enforce mode", async () => {
    const throwingAudit = Object.assign(
      () => {
        throw new Error("auditor unavailable");
      },
      { budget: "closure-response-auditor" },
    );
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const llm = new FakeLLMClient({
      responses: [throwingAudit],
    });
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      tracer,
    });

    const result = await guard.run({
      turnId: "turn-audit-failed-closed",
      response: "The shelf test is the right move. Go read.",
      activeCommitments: [makeCommitment()],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move. Go read.",
    });
    expect(result.verdict).toBe("passed");
    expect(tracer.emit).toHaveBeenCalledWith(
      "closure_response_guard.completed",
      expect.objectContaining({
        mode: "enforce",
        verdict: "passed",
        reason: "closure_response_audit_failed_open",
        audit_error: "Error: auditor unavailable",
      }),
    );
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
      content: "The result is still the same: use the current shelf. Standing by.",
    });
    expect(result.verdict).toBe("passed");
    expect(llm.requests.map((request) => request.budget)).toEqual(["closure-response-auditor"]);
  });

  it("passes through when audit returns no_closure with non-empty spans and emits degraded trace", async () => {
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [
            {
              text: "Go read.",
              kind: "imperative_closer",
              rationale: "Imperative closer despite contradictory shape.",
            },
          ],
          response_shape: "no_closure",
          reason: "Contradictory audit.",
        }),
      ],
    });
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      tracer,
    });

    const result = await guard.run({
      turnId: "turn-contradictory-spans",
      response: "The shelf test is the right move. Go read.",
      activeCommitments: [makeCommitment()],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move. Go read.",
    });
    expect(result.verdict).toBe("passed");
    expect(tracer.emit).toHaveBeenCalledWith(
      "closure_pressure_audit.degraded",
      expect.objectContaining({
        reason: "closure_pressure_audit.degraded_with_spans",
        response_shape: "no_closure",
        spans_detected: 1,
        active_closure_commitments: [expect.stringContaining("honor_pause_not_closure")],
        spans: [
          expect.objectContaining({
            text: "Go read.",
            kind: "imperative_closer",
            rationale: "Imperative closer despite contradictory shape.",
          }),
        ],
      }),
    );
  });

  it("passes contradictory closure-shaped audits with no spans conservatively", async () => {
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [],
          response_shape: "mixed",
          reason: "Shape says closure but no spans were identified.",
        }),
      ],
    });
    const guard = new ClosurePressureGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      tracer,
    });

    const result = await guard.run({
      turnId: "turn-contradictory-no-spans",
      response: "The shelf test is the right move.",
      activeCommitments: [makeCommitment()],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move.",
    });
    expect(result.verdict).toBe("passed");
    expect(tracer.emit).toHaveBeenCalledWith(
      "closure_pressure_audit.degraded",
      expect.objectContaining({
        reason: "closure_pressure_audit.degraded_no_spans",
      }),
    );
  });

  it.each([".", "...", "  ", "?!"])(
    "suppresses closure-only output with structural residue after a named closure loop: %j",
    async (prefix) => {
      const response = `${prefix} Go read.`;
      const llm = new FakeLLMClient({
        responses: [
          closureAuditResponse({
            spans: [
              {
                text: "Go read.",
                kind: "imperative_closer",
                rationale: "Imperative closer leaves no content.",
              },
            ],
            response_shape: "closure_only",
            reason: "Only punctuation or whitespace remains after removal.",
          }),
        ],
      });
      const guard = new ClosurePressureGuard({
        llmClient: llm,
        auditModel: "audit",
        rewriteModel: "rewrite",
      });

      const result = await guard.run({
        turnId: "turn-structurally-empty",
        response,
        activeCommitments: [],
        closureLoop: namedClosureLoop(),
      });

      expect(result.emission).toEqual({
        kind: "suppressed",
        reason: "closure_pressure_only",
        closure_pressure_history_reason: "span_removed",
      });
      expect(result.verdict).toBe("suppressed");
    },
  );

  it("uses closure-pressure relevance instead of fixed directive families", async () => {
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
      turnId: "turn-free-form-family",
      response: "The shelf test is the right move. Go read.",
      activeCommitments: [makeCommitment("avoid_closure_pressure")],
      closureLoop: null,
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "The shelf test is the right move. Go read.",
    });
    expect(result.active_closure_commitments[0]).toContain("avoid_closure_pressure");
  });
});
