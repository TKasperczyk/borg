import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import {
  createActionId,
  createCommitmentId,
  createEntityId,
  createEpisodeId,
  createRelationalSlotId,
  createSessionId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type CommitmentId,
  type EpisodeId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  RelationalClaimGuard,
  type RelationalClaimAuditClaim,
  type RelationalGuardEvidenceManifest,
  type RelationalGuardStreamEvidence,
  validateRelationalClaims,
} from "./relational-guard.js";

function rawClaimAuditResponse(claims: readonly unknown[]): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_claim_audit",
        name: "EmitClaimAudit",
        input: {
          claims,
        },
      },
    ],
  };
}

function claimAuditResponse(claims: readonly RelationalClaimAuditClaim[]): LLMCompleteResult {
  return rawClaimAuditResponse(claims);
}

function makeClaim(
  overrides: Partial<RelationalClaimAuditClaim> & Pick<RelationalClaimAuditClaim, "kind">,
): RelationalClaimAuditClaim {
  return {
    kind: overrides.kind,
    asserted: overrides.asserted ?? "asserted claim",
    cited_stream_entry_ids: overrides.cited_stream_entry_ids ?? [],
    cited_episode_ids: overrides.cited_episode_ids ?? [],
    cited_commitment_ids: overrides.cited_commitment_ids ?? [],
    cited_action_ids: overrides.cited_action_ids ?? [],
    quoted_evidence_text: overrides.quoted_evidence_text ?? null,
    callback_scope:
      overrides.callback_scope ?? (overrides.kind === "callback" ? "prior_turn" : null),
    subject_entity_id: overrides.subject_entity_id ?? null,
    slot_key: overrides.slot_key ?? null,
    relational_slot_value: overrides.relational_slot_value ?? null,
  };
}

function streamEvidence(input: {
  id?: StreamEntryId;
  sessionId?: SessionId;
  kind?: RelationalGuardStreamEvidence["kind"];
  ts?: number;
  content?: string;
}): RelationalGuardStreamEvidence {
  const content = input.content ?? "The user provided evidence.";

  return {
    entry_id: input.id ?? createStreamEntryId(),
    role: input.kind === "agent_msg" || input.kind === "agent_suppressed" ? "assistant" : "user",
    kind: input.kind ?? "user_msg",
    session_id: input.sessionId ?? DEFAULT_SESSION_ID,
    ts: input.ts ?? 1_000,
    snippet: content,
    content,
    compressed: false,
  };
}

function baseEvidence(
  overrides: Partial<RelationalGuardEvidenceManifest> = {},
): RelationalGuardEvidenceManifest {
  return {
    current_user_message: null,
    current_session_stream_entries: [],
    retrieved_episodes: [],
    active_commitments: [],
    corrective_preferences: [],
    relational_slots: [],
    recent_completed_actions: [],
    ...overrides,
  };
}

function validate(
  claims: readonly RelationalClaimAuditClaim[],
  evidence: RelationalGuardEvidenceManifest,
  currentSessionId: SessionId = DEFAULT_SESSION_ID,
  currentTurnTs = 2_000,
) {
  return validateRelationalClaims({
    claims,
    evidence,
    currentSessionId,
    currentTurnTs,
    hasCorrectivePreferenceEvidence: () => false,
  });
}

describe("validateRelationalClaims", () => {
  it("accepts relational identity claims only with current-session user evidence", () => {
    const userEntry = streamEvidence({
      content: "My partner is Sarah.",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "relational_identity",
          asserted: "Sarah is your partner",
          cited_stream_entry_ids: [userEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [userEntry],
      }),
    );

    expect(summary.unsupported).toEqual([]);
  });

  it("rejects partner-name claims without current-session user evidence", () => {
    const priorSessionEntryId = createStreamEntryId();
    const summary = validate(
      [
        makeClaim({
          kind: "relational_identity",
          asserted: "Sarah is your partner",
          cited_stream_entry_ids: [priorSessionEntryId],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [
          streamEvidence({
            content: "My partner is Maya.",
          }),
        ],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("current session evidence");
  });

  it("rejects relational identity claims for quarantined slots despite valid user evidence", () => {
    const userEntry = streamEvidence({
      content: "My partner is Sarah.",
    });
    const subject = createEntityId();
    const summary = validate(
      [
        makeClaim({
          kind: "relational_identity",
          asserted: "Sarah is your partner",
          cited_stream_entry_ids: [userEntry.entry_id],
          subject_entity_id: subject,
          slot_key: "partner.name",
          relational_slot_value: "Sarah",
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [userEntry],
        relational_slots: [
          {
            slot_id: createRelationalSlotId(),
            subject_entity_id: subject,
            slot_key: "partner.name",
            value: "Sarah",
            state: "quarantined",
            alternate_values: [{ value: "Maya" }, { value: "Clara" }],
            neutral_phrase: "your partner",
          },
        ],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("quarantined");
    expect(summary.unsupported[0]?.reason).toContain("your partner");
  });

  it("rejects relational identity claims without slot metadata when constrained slots exist", () => {
    const userEntry = streamEvidence({
      content: "My partner is Sarah.",
    });
    const subject = createEntityId();
    const summary = validate(
      [
        makeClaim({
          kind: "relational_identity",
          asserted: "Sarah is your partner",
          cited_stream_entry_ids: [userEntry.entry_id],
          subject_entity_id: null,
          slot_key: null,
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [userEntry],
        relational_slots: [
          {
            slot_id: createRelationalSlotId(),
            subject_entity_id: subject,
            slot_key: "partner.name",
            value: "Sarah",
            state: "quarantined",
            alternate_values: [{ value: "Maya" }, { value: "Clara" }],
            neutral_phrase: "your partner",
          },
        ],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toBe(
      "relational identity claim must specify slot when constrained slots exist for the cited entity",
    );
  });

  it("rejects concrete relational identity claims for contested slots even when value matches", () => {
    const userEntry = streamEvidence({
      content: "My partner is Sarah.",
    });
    const subject = createEntityId();
    const summary = validate(
      [
        makeClaim({
          kind: "relational_identity",
          asserted: "Sarah is your partner",
          cited_stream_entry_ids: [userEntry.entry_id],
          subject_entity_id: subject,
          slot_key: "partner.name",
          relational_slot_value: "Sarah",
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [userEntry],
        relational_slots: [
          {
            slot_id: createRelationalSlotId(),
            subject_entity_id: subject,
            slot_key: "partner.name",
            value: "Sarah",
            state: "contested",
            alternate_values: [{ value: "Maya" }],
            neutral_phrase: "your partner",
          },
        ],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("contested");
    expect(summary.unsupported[0]?.reason).toContain("your partner");
  });

  it("accepts callbacks with prior current-session stream evidence and exact quotes", () => {
    const userEntry = streamEvidence({
      ts: 1_000,
      content: "I mentioned the blue notebook earlier.",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "callback",
          asserted: "You mentioned the blue notebook earlier.",
          cited_stream_entry_ids: [userEntry.entry_id],
          quoted_evidence_text: "blue notebook",
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [userEntry],
      }),
    );

    expect(summary.unsupported).toEqual([]);
  });

  it("accepts current-turn callbacks that cite the current user message", () => {
    const currentEntryId = createStreamEntryId();
    const currentTurnTs = 1_995;
    const summary = validate(
      [
        makeClaim({
          kind: "callback",
          asserted: "As you said, the invoice is done.",
          callback_scope: "current_turn",
          cited_stream_entry_ids: [currentEntryId],
        }),
      ],
      baseEvidence({
        current_user_message: {
          text: "The invoice is done.",
          stream_entry_id: currentEntryId,
          ts: currentTurnTs,
        },
      }),
      DEFAULT_SESSION_ID,
      currentTurnTs,
    );

    expect(summary.unsupported).toEqual([]);
  });

  it("rejects prior-turn callbacks that cite the current user message", () => {
    const currentEntryId = createStreamEntryId();
    const currentTurnTs = 1_995;
    const summary = validate(
      [
        makeClaim({
          kind: "callback",
          asserted: "You said earlier that the invoice is done.",
          callback_scope: "prior_turn",
          cited_stream_entry_ids: [currentEntryId],
        }),
      ],
      baseEvidence({
        current_user_message: {
          text: "The invoice is done.",
          stream_entry_id: currentEntryId,
          ts: currentTurnTs,
        },
      }),
      DEFAULT_SESSION_ID,
      currentTurnTs,
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("prior-only evidence");
  });

  it("rejects callbacks that cite suppressed or non-prior entries", () => {
    const suppressedEntry = streamEvidence({
      kind: "agent_suppressed",
      ts: 1_000,
      content: '{"reason":"generation_gate"}',
    });
    const currentTurnEntry = streamEvidence({
      ts: 2_000,
      content: "This is the current turn.",
    });
    const suppressed = validate(
      [
        makeClaim({
          kind: "callback",
          asserted: "You mentioned it earlier.",
          cited_stream_entry_ids: [suppressedEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [suppressedEntry],
      }),
    );
    const nonPrior = validate(
      [
        makeClaim({
          kind: "callback",
          asserted: "You mentioned it earlier.",
          cited_stream_entry_ids: [currentTurnEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [currentTurnEntry],
      }),
    );

    expect(suppressed.unsupported[0]?.reason).toContain("agent_suppressed");
    expect(nonPrior.unsupported[0]?.reason).toContain("not before");
  });

  it("rejects current-conversation claims that cite a prior session", () => {
    const currentSession = createSessionId();
    const priorSession = createSessionId();
    const priorEntry = streamEvidence({
      sessionId: priorSession,
      ts: 1_000,
      content: "Prior-session content.",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "session_scoped",
          asserted: "Earlier in this conversation, you said that.",
          cited_stream_entry_ids: [priorEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [priorEntry],
      }),
      currentSession,
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("outside the current session");
  });

  it("allows session-scoped claims to cite the current user message but keeps prior-turn callbacks prior-only", () => {
    const currentEntryId = createStreamEntryId();
    const evidence = baseEvidence({
      current_user_message: {
        text: "I just said the invoice is done.",
        stream_entry_id: currentEntryId,
        ts: 2_000,
      },
    });
    const sessionScoped = validate(
      [
        makeClaim({
          kind: "session_scoped",
          asserted: "You just said the invoice is done.",
          cited_stream_entry_ids: [currentEntryId],
        }),
      ],
      evidence,
    );
    const callback = validate(
      [
        makeClaim({
          kind: "callback",
          asserted: "You said earlier that the invoice is done.",
          cited_stream_entry_ids: [currentEntryId],
        }),
      ],
      evidence,
    );

    expect(sessionScoped.unsupported).toEqual([]);
    expect(callback.unsupported).toHaveLength(1);
    expect(callback.unsupported[0]?.reason).toContain("prior-only evidence");
  });

  it("rejects action completion claims without matching completed action evidence", () => {
    const commitmentId = createCommitmentId();
    const actionId = createActionId();
    const summary = validate(
      [
        makeClaim({
          kind: "action_completion",
          asserted: "You booked the iTalki trial.",
          cited_action_ids: [actionId],
          cited_commitment_ids: [commitmentId],
        }),
      ],
      baseEvidence({
        active_commitments: [
          {
            commitment_id: commitmentId,
            source_entry_id: createStreamEntryId(),
            summary: "User said they will book an iTalki trial this week.",
          },
        ],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("no cited completed action record");
  });

  it("accepts action completion claims with matching completed action evidence", () => {
    const actionId = createActionId();
    const summary = validate(
      [
        makeClaim({
          kind: "action_completion",
          asserted: "You booked the iTalki trial.",
          cited_action_ids: [actionId],
        }),
      ],
      baseEvidence({
        recent_completed_actions: [
          {
            action_id: actionId,
            description: "Booked the iTalki trial.",
            audience_entity_id: null,
            completed_at: 2_000,
          },
        ],
      }),
    );

    expect(summary.unsupported).toEqual([]);
  });

  it("validates self-correction claims only when cited correction evidence exists", () => {
    const correctionEntry = streamEvidence({
      ts: 1_000,
      content: "Stop calling my partner Sarah.",
    });
    const correctiveCommitmentId = createCommitmentId();
    const supported = validateRelationalClaims({
      claims: [
        makeClaim({
          kind: "self_correction",
          asserted: "You corrected me earlier.",
          cited_stream_entry_ids: [correctionEntry.entry_id],
        }),
      ],
      evidence: baseEvidence({
        current_session_stream_entries: [correctionEntry],
        corrective_preferences: [
          {
            commitment_id: correctiveCommitmentId,
            source_entry_id: correctionEntry.entry_id,
          },
        ],
      }),
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      hasCorrectivePreferenceEvidence: () => false,
    });
    const unsupported = validate(
      [
        makeClaim({
          kind: "self_correction",
          asserted: "You corrected me earlier.",
          cited_stream_entry_ids: [correctionEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [correctionEntry],
      }),
    );

    expect(supported.unsupported).toEqual([]);
    expect(unsupported.unsupported[0]?.reason).toContain("no corrective preference evidence");
  });

  it("rejects self-correction claims that cite the current user message", () => {
    const currentEntryId = createStreamEntryId();
    const currentTurnTs = 1_995;
    const correctiveCommitmentId = createCommitmentId();
    const summary = validateRelationalClaims({
      claims: [
        makeClaim({
          kind: "self_correction",
          asserted: "You corrected me earlier.",
          cited_stream_entry_ids: [currentEntryId],
        }),
      ],
      evidence: baseEvidence({
        current_user_message: {
          text: "Stop calling it the draft invoice.",
          stream_entry_id: currentEntryId,
          ts: currentTurnTs,
        },
        corrective_preferences: [
          {
            commitment_id: correctiveCommitmentId,
            source_entry_id: currentEntryId,
          },
        ],
      }),
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs,
      hasCorrectivePreferenceEvidence: () => true,
    });

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("prior-only evidence");
  });

  it("rejects stream and episode evidence as action completion fallback", () => {
    const episodeId = createEpisodeId();
    const streamEntry = streamEvidence({
      ts: 1_000,
      content: "I completed the invoice.",
    });
    const episodeBacked = validate(
      [
        makeClaim({
          kind: "action_completion",
          asserted: "You completed the invoice.",
          cited_episode_ids: [episodeId],
        }),
      ],
      baseEvidence({
        retrieved_episodes: [
          {
            episode_id: episodeId,
            asOf: 1_000,
            snippet: "The invoice was completed.",
          },
        ],
      }),
    );
    const quoteMismatch = validate(
      [
        makeClaim({
          kind: "action_completion",
          asserted: "You completed the invoice.",
          cited_stream_entry_ids: [streamEntry.entry_id],
          quoted_evidence_text: "paid the invoice",
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [streamEntry],
      }),
    );

    expect(episodeBacked.unsupported[0]?.reason).toContain("no cited completed action record");
    expect(quoteMismatch.unsupported[0]?.reason).toContain("no cited completed action record");
  });
});

describe("RelationalClaimGuard", () => {
  it("parses omitted citation arrays and quoted evidence with schema defaults", async () => {
    const streamEntry = streamEvidence({
      ts: 1_000,
      content: "I mentioned the invoice earlier.",
    });
    const llm = new FakeLLMClient({
      responses: [
        rawClaimAuditResponse([
          {
            kind: "callback",
            asserted: "You mentioned the invoice earlier.",
            callback_scope: "prior_turn",
            cited_stream_entry_ids: [streamEntry.entry_id],
          },
        ]),
      ],
    });
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      hasCorrectivePreferenceEvidence: () => false,
    });

    const result = await guard.run({
      turnId: "turn-schema-defaults",
      response: "You mentioned the invoice earlier.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence({
        current_session_stream_entries: [streamEntry],
      }),
    });

    expect(result.verdict).toBe("passed");
    expect(result.claims[0]).toMatchObject({
      cited_episode_ids: [],
      cited_commitment_ids: [],
      cited_action_ids: [],
      quoted_evidence_text: null,
    });
  });

  it("uses a distinct suppression reason when the initial audit fails", async () => {
    const llm = new FakeLLMClient({
      responses: [
        rawClaimAuditResponse([
          {
            kind: "callback",
            asserted: "You mentioned the invoice earlier.",
          },
        ]),
      ],
    });
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      hasCorrectivePreferenceEvidence: () => false,
    });

    const result = await guard.run({
      turnId: "turn-audit-failed",
      response: "You mentioned the invoice earlier.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "relational_guard_audit_failed",
    });
  });

  it("suppresses fabricated self-correction claims without attempting a rewrite", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "self_correction",
            asserted: "You corrected me earlier in this conversation.",
          }),
        ]),
      ],
    });
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      tracer,
      hasCorrectivePreferenceEvidence: () => false,
    });

    const result = await guard.run({
      turnId: "turn-self-correction",
      response: "You corrected me earlier in this conversation.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "relational_guard_self_correction",
    });
    expect(llm.requests.map((request) => request.budget)).toEqual(["relational-claim-auditor"]);
    expect(tracer.emit).toHaveBeenCalledWith(
      "relational_claim_guard",
      expect.objectContaining({
        verdict: "suppressed",
        suppressionReason: "relational_guard_self_correction",
        claimsUnsupported: 1,
      }),
    );
  });

  it("rewrites unsupported non-self-correction claims once and re-audits", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "callback",
            asserted: "You mentioned that earlier.",
          }),
        ]),
        {
          text: "Neutralized response.",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        claimAuditResponse([]),
      ],
    });
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      hasCorrectivePreferenceEvidence: () => false,
    });

    const result = await guard.run({
      turnId: "turn-callback",
      response: "You mentioned that earlier.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: "Neutralized response.",
    });
    expect(result.verdict).toBe("rewritten");
    expect(llm.requests.map((request) => request.budget)).toEqual([
      "relational-claim-auditor",
      "relational-guard-rewrite",
      "relational-claim-auditor",
    ]);
  });

  it("uses a distinct suppression reason when rewrite returns empty text", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "callback",
            asserted: "You mentioned that earlier.",
          }),
        ]),
        {
          text: "   ",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      hasCorrectivePreferenceEvidence: () => false,
    });

    const result = await guard.run({
      turnId: "turn-rewrite-empty",
      response: "You mentioned that earlier.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "relational_guard_rewrite_empty",
    });
  });

  it("uses a distinct suppression reason when the rewrite call fails", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "callback",
            asserted: "You mentioned that earlier.",
          }),
        ]),
        () => {
          throw new Error("rewrite transport failed");
        },
      ],
    });
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      hasCorrectivePreferenceEvidence: () => false,
    });

    const result = await guard.run({
      turnId: "turn-rewrite-call-failed",
      response: "You mentioned that earlier.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "relational_guard_rewrite_call_failed",
    });
  });

  it("uses a distinct suppression reason when re-audit fails", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "callback",
            asserted: "You mentioned that earlier.",
          }),
        ]),
        {
          text: "Neutralized response.",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        rawClaimAuditResponse([
          {
            kind: "callback",
            asserted: "You mentioned that earlier.",
          },
        ]),
      ],
    });
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      hasCorrectivePreferenceEvidence: () => false,
    });

    const result = await guard.run({
      turnId: "turn-reaudit-failed",
      response: "You mentioned that earlier.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "relational_guard_reaudit_failed",
    });
  });

  it("suppresses when unsupported claims remain after the single rewrite", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "session_scoped",
            asserted: "Earlier in this conversation, you said that.",
          }),
        ]),
        {
          text: "Earlier in this conversation, you said that.",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        claimAuditResponse([
          makeClaim({
            kind: "session_scoped",
            asserted: "Earlier in this conversation, you said that.",
          }),
        ]),
      ],
    });
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      hasCorrectivePreferenceEvidence: () => false,
    });

    const result = await guard.run({
      turnId: "turn-rewrite-failed",
      response: "Earlier in this conversation, you said that.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.emission).toEqual({
      kind: "suppressed",
      reason: "relational_guard_rewrite_unsupported",
    });
  });
});
