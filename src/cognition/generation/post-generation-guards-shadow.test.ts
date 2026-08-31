import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { ActionRecord, ActionRecordListFilter } from "../../memory/actions/index.js";
import type { EntityRepository, CommitmentRecord } from "../../memory/commitments/index.js";
import type { RelationalSlot } from "../../memory/relational-slots/index.js";
import type { RetrievedEpisode } from "../../retrieval/index.js";
import { StreamReader, StreamWriter, type StreamEntry } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createActionId,
  createCommitmentId,
  createEntityId,
  createEpisodeId,
  createGoalId,
  createRelationalSlotId,
  createSessionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
} from "../../util/ids.js";
import { CommitmentGuardRunner } from "../commitments/guard-runner.js";
import type { TurnTracer } from "../../tracing/tracer.js";
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

async function runInternalIdentifierGuardFixture(input: {
  response: string;
  knownInternalIdentifiers: readonly string[];
  audienceContent?: string;
  sessionSourceType?: Parameters<TurnPostGenerationGuardRunner["run"]>[0]["sessionSourceType"];
  substratePrivilegedSourceTypes?: readonly string[];
  closureAudit?: ClosureResponseAudit;
  closureLoop?: Parameters<TurnPostGenerationGuardRunner["run"]>[0]["closureLoop"];
}) {
  const llm = new FakeLLMClient({
    responses: [
      closureAuditResponse(
        input.closureAudit ?? {
          spans: [],
          response_shape: "no_closure",
          reason: "No closure.",
        },
      ),
    ],
  });
  const emit = vi.fn();
  const createStreamReader = vi.fn(() => emptyStreamReader());
  const runner = new TurnPostGenerationGuardRunner({
    auditModel: "audit",
    closurePressureMode: "enforce",
    substratePrivilegedSourceTypes: input.substratePrivilegedSourceTypes ?? ["claude_code"],
    createStreamReader,
    actionRepository: {
      list: vi.fn(() => []),
    },
    relationalSlotRepository: {
      list: vi.fn(() => []),
    },
    clock: new FixedClock(2_000),
    tracer: {
      enabled: true,
      includePayloads: false,
      emit,
    },
  });
  const persistedUserEntry =
    input.audienceContent === undefined
      ? undefined
      : ({
          id: createStreamEntryId(),
          timestamp: 2_000,
          kind: "user_msg",
          content: input.audienceContent,
          session_id: DEFAULT_SESSION_ID,
          compressed: false,
          sender_entity_id: null,
          reply_target_entity_id: null,
        } satisfies StreamEntry);
  const emission = await runner.run({
    llmClient: llm,
    turnId: "turn-internal-identifier-fixture",
    response: input.response,
    sessionId: DEFAULT_SESSION_ID,
    sessionSourceType: input.sessionSourceType,
    ...(persistedUserEntry === undefined ? {} : { persistedUserEntry }),
    retrievedEpisodes: [],
    activeCommitments: [],
    closureLoop: input.closureLoop ?? null,
    audienceEntityId: null,
    knownInternalIdentifiers: input.knownInternalIdentifiers,
  });

  return { emission, emit, createStreamReader, llm };
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

  it("suppresses substrate-only identifiers on non-exempt source types", async () => {
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
      closurePressureMode: "enforce",
      substratePrivilegedSourceTypes: ["claude_code"],
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
      sessionSourceType: "demo",
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
      session_id: DEFAULT_SESSION_ID,
      session_source_type: "demo",
      verdict: "suppressed",
      leaked_identifiers: [userEntryId],
    });
  });

  it("allows exact identifiers echoed from current-turn audience-authored content", async () => {
    const audienceSuppliedId = createActionId();
    const response = `I can inspect ${audienceSuppliedId}.`;
    const userEntry: StreamEntry = {
      id: createStreamEntryId(),
      timestamp: 2_000,
      kind: "user_msg",
      content: `Please inspect ${audienceSuppliedId}.`,
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
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      closurePressureMode: "enforce",
      substratePrivilegedSourceTypes: ["claude_code"],
      createStreamReader: () => emptyStreamReader(),
      actionRepository: {
        list: vi.fn(() => []),
      },
      relationalSlotRepository: {
        list: vi.fn(() => []),
      },
      clock: new FixedClock(2_000),
      tracer: {
        enabled: true,
        includePayloads: false,
        emit,
      },
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-current-audience-id-echo",
      response,
      sessionId: DEFAULT_SESSION_ID,
      sessionSourceType: "demo",
      persistedUserEntry: userEntry,
      retrievedEpisodes: [],
      activeCommitments: [],
      closureLoop: null,
      audienceEntityId: null,
      knownInternalIdentifiers: [audienceSuppliedId],
    });

    expect(finalEmission).toEqual({
      kind: "message",
      content: response,
    });
    expect(emit).toHaveBeenCalledWith("internal_identifier_guard.completed", {
      turnId: "turn-current-audience-id-echo",
      session_id: DEFAULT_SESSION_ID,
      session_source_type: "demo",
      verdict: "passed",
      exemption_reason: "current_turn_audience_echo",
      exempted_identifiers: [audienceSuppliedId],
    });
  });

  it("does not exempt a known identifier that is only a prefix of an inbound token", async () => {
    const knownIdentifier = "act_abcdefghijklmnop";
    const { emission, emit } = await runInternalIdentifierGuardFixture({
      response: `The action handle is ${knownIdentifier}.`,
      knownInternalIdentifiers: [knownIdentifier],
      audienceContent: `Please inspect ${knownIdentifier}x.`,
      sessionSourceType: "demo",
    });

    expect(emission).toEqual({
      kind: "suppressed",
      reason: "internal_identifier_leak",
    });
    expect(emit).toHaveBeenCalledWith(
      "internal_identifier_guard.completed",
      expect.objectContaining({
        verdict: "suppressed",
        leaked_identifiers: [knownIdentifier],
      }),
    );
  });

  it("does not exempt a known identifier embedded in a longer alphanumeric run", async () => {
    const knownIdentifier = "goal_abcdefghijklmnop";
    const { emission, emit } = await runInternalIdentifierGuardFixture({
      response: `The goal handle is ${knownIdentifier}.`,
      knownInternalIdentifiers: [knownIdentifier],
      audienceContent: `Please inspect prefix${knownIdentifier}suffix.`,
      sessionSourceType: "demo",
    });

    expect(emission).toEqual({
      kind: "suppressed",
      reason: "internal_identifier_leak",
    });
    expect(emit).toHaveBeenCalledWith(
      "internal_identifier_guard.completed",
      expect.objectContaining({
        verdict: "suppressed",
        leaked_identifiers: [knownIdentifier],
      }),
    );
  });

  it("exempts only exact audience echoes when the response also contains a substrate-only id", async () => {
    const echoedIdentifier = createActionId();
    const substrateOnlyIdentifier = createGoalId();
    const { emission, emit } = await runInternalIdentifierGuardFixture({
      response: `The handles are ${echoedIdentifier} and ${substrateOnlyIdentifier}.`,
      knownInternalIdentifiers: [echoedIdentifier, substrateOnlyIdentifier],
      audienceContent: `Please inspect ${echoedIdentifier}.`,
      sessionSourceType: "demo",
    });

    expect(emission).toEqual({
      kind: "suppressed",
      reason: "internal_identifier_leak",
    });
    expect(emit).toHaveBeenCalledWith("internal_identifier_guard.completed", {
      turnId: "turn-internal-identifier-fixture",
      session_id: DEFAULT_SESSION_ID,
      session_source_type: "demo",
      verdict: "suppressed",
      leaked_identifiers: [substrateOnlyIdentifier],
      exemption_reason: "current_turn_audience_echo",
      exempted_identifiers: [echoedIdentifier],
    });
  });

  it("skips internal-identifier enforcement on configured substrate-privileged sources", async () => {
    const substrateIdentifier = createSessionId();
    const response = `The substrate handle is ${substrateIdentifier}.`;
    const createStreamReader = vi.fn(() => emptyStreamReader());
    const listActions = vi.fn(() => []);
    const listRelationalSlots = vi.fn(() => []);
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
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      closurePressureMode: "enforce",
      substratePrivilegedSourceTypes: ["claude_code"],
      createStreamReader,
      actionRepository: {
        list: listActions,
      },
      relationalSlotRepository: {
        list: listRelationalSlots,
      },
      clock: new FixedClock(2_000),
      tracer: {
        enabled: true,
        includePayloads: false,
        emit,
      },
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-substrate-privileged-id",
      response,
      sessionId: DEFAULT_SESSION_ID,
      sessionSourceType: "claude_code",
      retrievedEpisodes: [],
      activeCommitments: [],
      closureLoop: null,
      audienceEntityId: null,
      knownInternalIdentifiers: [substrateIdentifier],
    });

    expect(finalEmission).toEqual({
      kind: "message",
      content: response,
    });
    expect(emit).toHaveBeenCalledWith("internal_identifier_guard.completed", {
      turnId: "turn-substrate-privileged-id",
      session_id: DEFAULT_SESSION_ID,
      session_source_type: "claude_code",
      verdict: "skipped",
      reason: "substrate_privileged_source_type",
    });
    expect(llm.requests.map((request) => request.budget)).toEqual(["closure-response-auditor"]);
    expect(createStreamReader).not.toHaveBeenCalled();
    expect(listActions).not.toHaveBeenCalled();
    expect(listRelationalSlots).not.toHaveBeenCalled();
  });

  it.each([
    { label: "null", sessionSourceType: null },
    { label: "undefined", sessionSourceType: undefined },
  ])(
    "keeps internal-identifier enforcement active for a $label source type with a populated allowlist",
    async ({ sessionSourceType }) => {
      const substrateIdentifier = createSessionId();
      const { emission, createStreamReader } = await runInternalIdentifierGuardFixture({
        response: `The substrate handle is ${substrateIdentifier}.`,
        knownInternalIdentifiers: [substrateIdentifier],
        sessionSourceType,
        substratePrivilegedSourceTypes: ["claude_code"],
      });

      expect(emission).toEqual({
        kind: "suppressed",
        reason: "internal_identifier_leak",
      });
      expect(createStreamReader).toHaveBeenCalledOnce();
    },
  );

  it("still enforces closure-pressure suppression on a substrate-privileged source", async () => {
    const substrateIdentifier = createSessionId();
    const { emission, emit, createStreamReader } = await runInternalIdentifierGuardFixture({
      response: `Held. Book. ${substrateIdentifier}`,
      knownInternalIdentifiers: [substrateIdentifier],
      sessionSourceType: "claude_code",
      substratePrivilegedSourceTypes: ["claude_code"],
      closureAudit: {
        spans: [
          {
            text: "Held. Book.",
            kind: "quotable_closing_tail",
            rationale: "The response applies closure pressure.",
          },
        ],
        response_shape: "closure_only",
        reason: "Only closure pressure remains.",
      },
      closureLoop: {
        status: "named",
        source_stream_entry_ids: [createStreamEntryId()],
        reason: "The closure loop is active.",
        since_turn: 3,
        named_at_turn: 4,
      },
    });

    expect(emission).toEqual({
      kind: "suppressed",
      reason: "closure_pressure_only",
      closure_pressure_history_reason: "span_removed",
    });
    expect(emit).toHaveBeenCalledWith(
      "closure_response_guard.completed",
      expect.objectContaining({
        verdict: "suppressed",
        reason: "closure_pressure_only",
      }),
    );
    expect(emit).not.toHaveBeenCalledWith("internal_identifier_guard.completed", expect.anything());
    expect(createStreamReader).not.toHaveBeenCalled();
  });

  it("suppresses known cross-session identifiers rendered in operator snapshots", async () => {
    const targetSessionId = createSessionId();
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
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      closurePressureMode: "enforce",
      createStreamReader: () => emptyStreamReader(),
      actionRepository: {
        list: vi.fn(() => []),
      },
      relationalSlotRepository: {
        list: vi.fn(() => []),
      },
      clock: new FixedClock(2_000),
      tracer: {
        enabled: true,
        includePayloads: false,
        emit,
      },
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-cross-session-id-leak",
      response: `The target session handle is ${targetSessionId}.`,
      sessionId: DEFAULT_SESSION_ID,
      retrievedEpisodes: [],
      activeCommitments: [],
      closureLoop: null,
      audienceEntityId: null,
      knownInternalIdentifiers: [targetSessionId],
    });

    expect(finalEmission).toEqual({
      kind: "suppressed",
      reason: "internal_identifier_leak",
    });
    expect(emit).toHaveBeenCalledWith("internal_identifier_guard.completed", {
      turnId: "turn-cross-session-id-leak",
      session_id: DEFAULT_SESSION_ID,
      verdict: "suppressed",
      leaked_identifiers: [targetSessionId],
    });
  });

  it("suppresses entity ids carried only by retrieved episode origins and disclosure labels", async () => {
    const originEntityId = createEntityId();
    const privateToEntityId = createEntityId();
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
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      closurePressureMode: "enforce",
      createStreamReader: () => emptyStreamReader(),
      actionRepository: {
        list: vi.fn(() => []),
      },
      relationalSlotRepository: {
        list: vi.fn(() => []),
      },
      clock: new FixedClock(2_000),
      tracer: {
        enabled: true,
        includePayloads: false,
        emit,
      },
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-disclosure-label-id-leak",
      response: `The private origin handles are ${originEntityId} and ${privateToEntityId}.`,
      sessionId: DEFAULT_SESSION_ID,
      retrievedEpisodes: [
        {
          episode: {
            id: createEpisodeId(),
            audience_entity_id: null,
            origin_audience_entity_ids: [originEntityId],
            source_stream_ids: [],
            lineage: {
              derived_from: [],
              supersedes: [],
            },
          },
          disclosureLabel: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [originEntityId],
            privateToEntityIds: [privateToEntityId],
            publicToEntityIds: [],
          },
          citationChain: [],
        } as unknown as RetrievedEpisode,
      ],
      activeCommitments: [],
      closureLoop: null,
      audienceEntityId: null,
    });

    expect(finalEmission).toEqual({
      kind: "suppressed",
      reason: "internal_identifier_leak",
    });
    expect(emit).toHaveBeenCalledWith("internal_identifier_guard.completed", {
      turnId: "turn-disclosure-label-id-leak",
      session_id: DEFAULT_SESSION_ID,
      verdict: "suppressed",
      leaked_identifiers: [originEntityId, privateToEntityId].sort(),
    });
  });

  it("does not suppress final answers merely for containing labeled private memory content", async () => {
    const aliceId = createEntityId();
    const privateMemory = "Alice said the fallback route is not ready.";
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
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      closurePressureMode: "enforce",
      createStreamReader: () => emptyStreamReader(),
      actionRepository: {
        list: vi.fn(() => []),
      },
      relationalSlotRepository: {
        list: vi.fn(() => []),
      },
      clock: new FixedClock(2_000),
      tracer: {
        enabled: true,
        includePayloads: false,
        emit,
      },
    });

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-private-content-not-policed",
      response: `I remember the relevant detail: ${privateMemory}`,
      sessionId: DEFAULT_SESSION_ID,
      retrievedEpisodes: [
        {
          episode: {
            id: createEpisodeId(),
            title: "Alice private routing note",
            narrative: privateMemory,
            audience_entity_id: null,
            origin_audience_entity_ids: [aliceId],
            source_stream_ids: [],
            lineage: {
              derived_from: [],
              supersedes: [],
            },
          },
          disclosureLabel: {
            disclosureClass: "relationship_private",
            originAudienceEntityIds: [aliceId],
            privateToEntityIds: [aliceId],
            publicToEntityIds: [],
          },
          citationChain: [],
        } as unknown as RetrievedEpisode,
      ],
      activeCommitments: [],
      closureLoop: null,
      audienceEntityId: null,
    });

    expect(finalEmission).toEqual({
      kind: "message",
      content: `I remember the relevant detail: ${privateMemory}`,
    });
    expect(emit).not.toHaveBeenCalledWith(
      "internal_identifier_guard.completed",
      expect.objectContaining({
        verdict: "suppressed",
      }),
    );
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

  it("suppresses old transcript IDs outside the scan window when surfaced by retrieved context", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-post-generation-guard-old-id-"));
    const scanSpy = vi.spyOn(StreamReader.prototype, "scanReverse");

    try {
      const writer = new StreamWriter({
        dataDir,
        sessionId: DEFAULT_SESSION_ID,
        clock: new FixedClock(1_000),
      });
      const entries = await writer.appendMany(
        Array.from({ length: 700 }, (_, index) => ({
          kind: "user_msg" as const,
          content: `old message ${index}`,
        })),
      );
      const oldTranscriptEntry = entries[0];

      if (oldTranscriptEntry === undefined) {
        throw new Error("expected old transcript entry");
      }

      const llm = new FakeLLMClient({
        responses: [
          closureAuditResponse({
            spans: [],
            response_shape: "no_closure",
            reason: "No closure.",
          }),
        ],
      });
      const commitment = {
        ...makeCommitment(),
        source_stream_entry_ids: [oldTranscriptEntry.id],
      };
      const relationalSlot = {
        id: createRelationalSlotId(),
        subject_entity_id: null,
        evidence_stream_entry_ids: [oldTranscriptEntry.id],
        contradicted_by_stream_entry_ids: [],
        alternate_values: [],
      } as unknown as RelationalSlot;
      const completedAction = {
        id: createActionId(),
        actor: "user",
        audience_entity_id: null,
        provenance_episode_ids: [],
        provenance_stream_entry_ids: [oldTranscriptEntry.id],
        updated_at: 2_000,
      } as unknown as ActionRecord;
      const retrievedEpisode = {
        episode: {
          id: createEpisodeId(),
          audience_entity_id: null,
          source_stream_ids: [oldTranscriptEntry.id],
          lineage: {
            derived_from: [],
            supersedes: [],
          },
        },
        citationChain: [],
      } as unknown as RetrievedEpisode;
      const postGenerationRunner = new TurnPostGenerationGuardRunner({
        auditModel: "audit",
        closurePressureMode: "enforce",
        createStreamReader: (sessionId) => new StreamReader({ dataDir, sessionId }),
        actionRepository: {
          list: vi.fn(() => [completedAction]),
        },
        relationalSlotRepository: {
          list: vi.fn(() => [relationalSlot]),
        },
        clock: new FixedClock(2_000),
        tracer: {
          enabled: true,
          includePayloads: false,
          emit: vi.fn(),
        },
      });

      const finalEmission = await postGenerationRunner.run({
        llmClient: llm,
        turnId: "turn-old-context-id-leak",
        response: `The old source handle was ${oldTranscriptEntry.id}.`,
        sessionId: DEFAULT_SESSION_ID,
        retrievedEpisodes: [retrievedEpisode],
        activeCommitments: [commitment],
        closureLoop: null,
        audienceEntityId: null,
      });

      expect(scanSpy.mock.results[0]?.value.entries).not.toContainEqual(
        expect.objectContaining({ id: oldTranscriptEntry.id }),
      );
      expect(finalEmission).toEqual({
        kind: "suppressed",
        reason: "internal_identifier_leak",
      });
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
    expect(emit).not.toHaveBeenCalledWith(
      "internal_identifier_guard.completed",
      expect.any(Object),
    );
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
      session_id: DEFAULT_SESSION_ID,
      verdict: "suppressed",
      leaked_identifiers: [discourseTurnId],
    });
  });

  it("uses global recent completed actions for cognition and internal-id hygiene", async () => {
    const alice = createEntityId();
    const privateAction: ActionRecord = {
      id: createActionId(),
      description: "Completed Alice private launch review",
      actor: "borg",
      audience_entity_id: alice,
      goal_id: null,
      open_question_id: null,
      state: "completed",
      confidence: 0.9,
      provenance_episode_ids: [],
      provenance_stream_entry_ids: [createStreamEntryId()],
      created_at: 1_000,
      updated_at: 2_000,
      considering_at: null,
      committed_at: null,
      scheduled_at: null,
      completed_at: 2_000,
      not_done_at: null,
      expired_at: null,
      archived_at: null,
      unknown_at: null,
      canonicalized_by_artifact_entry_id: null,
      session_scope: null,
      session_anchor_id: null,
      last_referenced_at_ms: 2_000,
      last_referenced_turn_counter: null,
    };
    const actionRepository = {
      list: vi.fn((filter: ActionRecordListFilter = {}) =>
        [privateAction].filter(
          (action) =>
            (filter.state === undefined || action.state === filter.state) &&
            (filter.recallAllAudiences === true ||
              !("audienceEntityId" in filter) ||
              (filter.audienceEntityId === null
                ? action.audience_entity_id === null
                : action.audience_entity_id === filter.audienceEntityId)),
        ),
      ),
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
    const postGenerationRunner = new TurnPostGenerationGuardRunner({
      auditModel: "audit",
      closurePressureMode: "enforce",
      createStreamReader: () => emptyStreamReader(),
      actionRepository,
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

    expect(postGenerationRunner.listRecentCompletedActionsForCognition(null)).toEqual([
      privateAction,
    ]);

    const finalEmission = await postGenerationRunner.run({
      llmClient: llm,
      turnId: "turn-private-completed-action-id-leak",
      response: `The completed action id was ${privateAction.id}.`,
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
    expect(actionRepository.list).toHaveBeenCalledWith(
      expect.objectContaining({
        state: "completed",
        recallAllAudiences: true,
      }),
    );
  });
});
