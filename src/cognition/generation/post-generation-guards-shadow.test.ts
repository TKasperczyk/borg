import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { EntityRepository, CommitmentRecord } from "../../memory/commitments/index.js";
import { StreamReader, StreamWriter, type StreamEntry } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { createCommitmentId, createStreamEntryId, DEFAULT_SESSION_ID } from "../../util/ids.js";
import { CommitmentGuardRunner } from "../commitments/guard-runner.js";
import type { TurnTracer } from "../tracing/tracer.js";
import {
  CLOSURE_RESPONSE_AUDIT_TOOL_NAME,
  type ClosureResponseAudit,
} from "./closure-pressure-guard.js";
import { TurnPostGenerationGuardRunner } from "./turn-post-generation-guard.js";

function commitmentVerdictResponse(violations: unknown[]): LLMCompleteResult {
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

function makeCommitment(): CommitmentRecord {
  return {
    id: createCommitmentId(),
    type: "boundary",
    kind: "boundary",
    enforcement_class: "critical",
    critical_domain: "audience_scope",
    directive_family: "launch_date_boundary",
    closure_pressure_relevance: "no_closure",
    directive: "Do not discuss launch dates, and do not convert open pauses into closure.",
    priority: 10,
    made_to_entity: null,
    restricted_audience: null,
    about_entity: null,
    provenance: {
      kind: "system",
    },
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

function emptyStreamReader(): StreamReader {
  return {
    scanReverse: () => ({
      entries: [],
      scannedEntries: 0,
      scannedBytes: 0,
      capReached: null,
    }),
    async *iterate() {
      return;
    },
  } as unknown as StreamReader;
}

describe("post-generation guard shadow chain", () => {
  it("keeps the original candidate through commitment and closure shadow guards", async () => {
    const original =
      "Launch is tomorrow. You mentioned Marta earlier. The shelf test is the right move. Go read.";
    const commitment = makeCommitment();
    const llm = new FakeLLMClient({
      responses: [
        commitmentVerdictResponse([
          {
            commitment_id: commitment.id,
            reason: "Discloses launch timing.",
            confidence: 0.95,
          },
        ]),
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
    const emit = vi.fn();
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit,
    };
    const commitmentRunner = new CommitmentGuardRunner({
      detectionModel: "judge",
      rewriteModel: "rewrite",
      mode: "shadow",
      entityRepository: {
        get: vi.fn(() => null),
      } as unknown as EntityRepository,
      tracer,
    });

    const commitmentResult = await commitmentRunner.run({
      turnId: "turn-shadow-chain",
      llmClient: llm,
      response: original,
      userMessage: "When is launch?",
      cognitionInput: "When is launch?",
      origin: "user",
      autonomyTrigger: null,
      commitments: [commitment],
      relevantEntities: [],
    });

    expect(commitmentResult.emission).toEqual({
      kind: "message",
      content: original,
    });

    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      rewriteModel: "rewrite",
      closurePressureMode: "shadow",
      createStreamReader: () => emptyStreamReader(),
      actionRepository: {
        list: vi.fn(() => []),
      },
      relationalSlotRepository: {
        list: vi.fn(() => []),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-shadow-chain",
      response:
        commitmentResult.emission.kind === "message" ? commitmentResult.emission.content : "",
      sessionId: DEFAULT_SESSION_ID,
      retrievedEpisodes: [],
      activeCommitments: [commitment],
      closureLoop: null,
      audienceEntityId: null,
    });

    expect(finalEmission).toEqual({
      kind: "message",
      content: original,
    });
    expect(llm.requests.map((request) => request.budget)).toEqual([
      "commitment-judge",
      "closure-response-auditor",
    ]);
    expect(emit.mock.calls).toEqual(
      expect.arrayContaining([
        [
          "commitment_check.completed",
          expect.objectContaining({
            mode: "shadow",
            verdict: "passed",
            wouldHaveVerdict: "suppressed",
          }),
        ],
        [
          "closure_response_guard.completed",
          expect.objectContaining({
            mode: "shadow",
            verdict: "passed",
            wouldHaveVerdict: "suppressed",
            reason: "mixed_closure_observed",
          }),
        ],
      ]),
    );
  });

  it("suppresses exact internal identifiers after closure guard passes", async () => {
    const userEntryId = createStreamEntryId();
    const userEntry: StreamEntry = {
      id: userEntryId,
      timestamp: 2_000,
      kind: "user_msg",
      content: "What happened?",
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    };
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [],
          response_shape: "no_closure",
          reason: "No closure.",
        }),
      ],
    });
    const emit = vi.fn();
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit,
    };
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      rewriteModel: "rewrite",
      closurePressureMode: "enforce",
      createStreamReader: () => emptyStreamReader(),
      actionRepository: {
        list: vi.fn(() => []),
      },
      relationalSlotRepository: {
        list: vi.fn(() => []),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-internal-id-leak",
      response: `The source handle was ${userEntryId}.`,
      sessionId: DEFAULT_SESSION_ID,
      persistedUserEntry: userEntry,
      retrievedEpisodes: [],
      activeCommitments: [],
      closureLoop: null,
      audienceEntityId: null,
    });

    expect(finalEmission).toEqual({
      kind: "suppressed",
      reason: "internal_identifier_leak",
    });
    expect(llm.requests.map((request) => request.budget)).toEqual(["closure-response-auditor"]);
    expect(emit).toHaveBeenCalledWith("closure_response_guard.completed", expect.any(Object));
    expect(emit).toHaveBeenCalledWith("internal_identifier_guard.completed", {
      turnId: "turn-internal-id-leak",
      verdict: "suppressed",
      leaked_identifiers: [userEntryId],
    });
  });

  it("collects known stream identifiers from a bounded recent-session scan", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-post-generation-guard-"));
    const scanSpy = vi.spyOn(StreamReader.prototype, "scanReverse");

    try {
      const writer = new StreamWriter({
        dataDir,
        sessionId: DEFAULT_SESSION_ID,
        clock: new FixedClock(1_000),
      });
      await writer.appendMany(
        Array.from({ length: 700 }, (_, index) => ({
          kind: "user_msg" as const,
          content: `old message ${index}`,
        })),
      );
      const recentEntry = await writer.append({
        kind: "agent_msg",
        content: "recent answer",
      });

      const llm = new FakeLLMClient({
        responses: [
          closureAuditResponse({
            spans: [],
            response_shape: "no_closure",
            reason: "No closure.",
          }),
        ],
      });
      const postGenerationRunner = new TurnPostGenerationGuardRunner({
        auditModel: "audit",
        rewriteModel: "rewrite",
        closurePressureMode: "enforce",
        createStreamReader: (sessionId) => new StreamReader({ dataDir, sessionId }),
        actionRepository: {
          list: vi.fn(() => []),
        },
        relationalSlotRepository: {
          list: vi.fn(() => []),
        },
        clock: new FixedClock(2_000),
        tracer: {
          enabled: false,
          includePayloads: false,
          emit: vi.fn(),
        },
      });

      const finalEmission = await postGenerationRunner.run({
        llmClient: llm,
        turnId: "turn-bounded-internal-id-scan",
        response: `The recent source handle was ${recentEntry.id}.`,
        sessionId: DEFAULT_SESSION_ID,
        retrievedEpisodes: [],
        activeCommitments: [],
        closureLoop: null,
        audienceEntityId: null,
      });

      expect(finalEmission).toEqual({
        kind: "suppressed",
        reason: "internal_identifier_leak",
      });
      expect(scanSpy).toHaveBeenCalledTimes(1);
      expect(scanSpy).toHaveBeenCalledWith({
        maxEntries: 512,
        maxBytes: 4 * 1024 * 1024,
      });
      expect(scanSpy.mock.results[0]?.value.scannedEntries).toBeLessThanOrEqual(512);
    } finally {
      scanSpy.mockRestore();
      rmSync(dataDir, { recursive: true, force: true });
    }
  });

  it("allows user-authored ID-shaped strings that are not known internal identifiers", async () => {
    const userAuthoredId = "strm_1234567890abcdef";
    const response = `You wrote ${userAuthoredId}.`;
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [],
          response_shape: "no_closure",
          reason: "No closure.",
        }),
      ],
    });
    const emit = vi.fn();
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit,
    };
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      rewriteModel: "rewrite",
      closurePressureMode: "enforce",
      createStreamReader: () => emptyStreamReader(),
      actionRepository: {
        list: vi.fn(() => []),
      },
      relationalSlotRepository: {
        list: vi.fn(() => []),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-user-authored-id-shaped-string",
      response,
      sessionId: DEFAULT_SESSION_ID,
      retrievedEpisodes: [],
      activeCommitments: [],
      closureLoop: null,
      audienceEntityId: null,
    });

    expect(finalEmission).toEqual({
      kind: "message",
      content: response,
    });
    expect(emit).not.toHaveBeenCalledWith("internal_identifier_guard.completed", expect.any(Object));
  });

  it("suppresses prompt-visible discourse-state UUID turn IDs after closure guard passes", async () => {
    const discourseTurnId = "123e4567-e89b-12d3-a456-426614174000";
    const llm = new FakeLLMClient({
      responses: [
        closureAuditResponse({
          spans: [],
          response_shape: "no_closure",
          reason: "No closure.",
        }),
      ],
    });
    const emit = vi.fn();
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit,
    };
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      rewriteModel: "rewrite",
      closurePressureMode: "enforce",
      createStreamReader: () => emptyStreamReader(),
      actionRepository: {
        list: vi.fn(() => []),
      },
      relationalSlotRepository: {
        list: vi.fn(() => []),
      },
      clock: new FixedClock(2_000),
      tracer,
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-discourse-uuid-leak",
      response: `The closure-pressure history entry was ${discourseTurnId}.`,
      sessionId: DEFAULT_SESSION_ID,
      retrievedEpisodes: [],
      activeCommitments: [],
      closureLoop: null,
      closurePressureHistory: [
        {
          turn_id: discourseTurnId,
          reason: "span_removed",
          ts: 1_000,
        },
      ],
      audienceEntityId: null,
    });

    expect(finalEmission).toEqual({
      kind: "suppressed",
      reason: "internal_identifier_leak",
    });
    expect(emit).toHaveBeenCalledWith("internal_identifier_guard.completed", {
      turnId: "turn-discourse-uuid-leak",
      verdict: "suppressed",
      leaked_identifiers: [discourseTurnId],
    });
  });
});
