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
    support_handles: overrides.support_handles ?? [],
    quoted_evidence_text: overrides.quoted_evidence_text ?? null,
    callback_scope:
      overrides.callback_scope ?? (overrides.kind === "callback" ? "prior_turn" : null),
    phenomenology_verdict:
      overrides.phenomenology_verdict ??
      (overrides.kind === "ai_phenomenology" ? "unsupported_subjective" : null),
    specific_detail_value: overrides.specific_detail_value ?? null,
    specific_detail_support_kind:
      overrides.specific_detail_support_kind ??
      (overrides.kind === "unsupported_specific_detail" ? "none" : null),
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
  response?: string,
) {
  return validateRelationalClaims({
    claims,
    response,
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

  it("does not allowlist incidental capitalized user-text tokens as person-name support", () => {
    const currentEntryId = createStreamEntryId();
    const summary = validate(
      [],
      baseEvidence({
        current_user_message: {
          text: "My Spanish tutor is from Sevilla.",
          stream_entry_id: currentEntryId,
          ts: 2_000,
        },
      }),
      DEFAULT_SESSION_ID,
      2_000,
      "I'll ask Sevilla the boring version.",
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.claim).toMatchObject({
      kind: "unsupported_person_name",
      relational_slot_value: "Sevilla",
    });
  });

  it("accepts a user-named person when the auditor cites a user support handle", () => {
    const userEntry = streamEvidence({
      content: "Her name is Marta.",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "unsupported_person_name",
          asserted: "ask Marta",
          relational_slot_value: "Marta",
          support_handles: [userEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [userEntry],
      }),
      DEFAULT_SESSION_ID,
      2_000,
      "I'll ask Marta the boring version.",
    );

    expect(summary.unsupported).toEqual([]);
  });

  it("accepts retrieved episode user text as person-name support and rejects narrative-only episode mentions", () => {
    const supportedEpisodeId = createEpisodeId();
    const narrativeOnlyEpisodeId = createEpisodeId();
    const evidence = baseEvidence({
      retrieved_episodes: [
        {
          episode_id: supportedEpisodeId,
          asOf: 1_000,
          snippet: "Tom confirmed a tutor name.",
          user_texts: ["Her name is Marta."],
        },
        {
          episode_id: narrativeOnlyEpisodeId,
          asOf: 1_000,
          snippet: "The episode summary mentions Marta.",
          user_texts: ["The tutor lives nearby."],
        },
      ],
    });
    const supported = validate(
      [
        makeClaim({
          kind: "unsupported_person_name",
          asserted: "ask Marta",
          relational_slot_value: "Marta",
          cited_episode_ids: [supportedEpisodeId],
        }),
      ],
      evidence,
    );
    const unsupported = validate(
      [
        makeClaim({
          kind: "unsupported_person_name",
          asserted: "ask Marta",
          relational_slot_value: "Marta",
          cited_episode_ids: [narrativeOnlyEpisodeId],
        }),
      ],
      evidence,
    );

    expect(supported.unsupported).toEqual([]);
    expect(unsupported.unsupported).toHaveLength(1);
    expect(unsupported.unsupported[0]?.reason).toContain("cited episode user evidence");
  });

  it("rejects unsupported specific details with no user-side evidence", () => {
    const summary = validate(
      [
        makeClaim({
          kind: "unsupported_specific_detail",
          asserted: "three hundred times",
          specific_detail_value: "three hundred",
          specific_detail_support_kind: "none",
        }),
      ],
      baseEvidence(),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("no user-side source evidence");
  });

  it("does not treat assistant-seeded user repetition as specific-detail support", () => {
    const assistantEntry = streamEvidence({
      kind: "agent_msg",
      ts: 1_000,
      content: "By being wrong about soup three hundred times.",
    });
    const userEntry = streamEvidence({
      ts: 1_100,
      content: "three hundred wrong soups. Got it.",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "unsupported_specific_detail",
          asserted: "three hundred times",
          specific_detail_value: "three hundred",
          specific_detail_support_kind: "user_introduced",
          support_handles: [userEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [assistantEntry, userEntry],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("assistant_seeded");
  });

  it("detects assistant-seeded repetitions with case differences", () => {
    const assistantEntry = streamEvidence({
      kind: "agent_msg",
      ts: 1_000,
      content: "Three hundred wrong soups is the number.",
    });
    const userEntry = streamEvidence({
      ts: 1_100,
      content: "three hundred wrong soups. Got it.",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "unsupported_specific_detail",
          asserted: "three hundred wrong soups",
          specific_detail_value: "three hundred wrong soups",
          specific_detail_support_kind: "user_introduced",
          support_handles: [userEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [assistantEntry, userEntry],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("assistant_seeded");
  });

  it("detects assistant-seeded repetitions with trailing punctuation differences", () => {
    const assistantEntry = streamEvidence({
      kind: "agent_msg",
      ts: 1_000,
      content: "You practiced 300 times.",
    });
    const userEntry = streamEvidence({
      ts: 1_100,
      content: "300 times",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "unsupported_specific_detail",
          asserted: "300 times",
          specific_detail_value: "300 times.",
          specific_detail_support_kind: "user_introduced",
          support_handles: [userEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [assistantEntry, userEntry],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("assistant_seeded");
  });

  it("does not support numeric values inside larger numbers", () => {
    const userEntry = streamEvidence({
      ts: 1_000,
      content: "I practiced 1200 times.",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "unsupported_specific_detail",
          asserted: "200 times",
          specific_detail_value: "200",
          specific_detail_support_kind: "user_introduced",
          support_handles: [userEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [userEntry],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("does not appear verbatim");
  });

  it("does not let episode support override assistant-seeded specific-detail taint", () => {
    const episodeId = createEpisodeId();
    const assistantEntry = streamEvidence({
      kind: "agent_msg",
      ts: 1_000,
      content: "You practiced 300 times.",
    });
    const userEntry = streamEvidence({
      ts: 1_100,
      content: "300",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "unsupported_specific_detail",
          asserted: "300 times",
          specific_detail_value: "300",
          specific_detail_support_kind: "user_introduced",
          support_handles: [userEntry.entry_id],
          cited_episode_ids: [episodeId],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [assistantEntry, userEntry],
        retrieved_episodes: [
          {
            episode_id: episodeId,
            asOf: 1_200,
            snippet: "The user echoed a scalar detail.",
            user_texts: ["300"],
          },
        ],
      }),
    );

    expect(summary.unsupported).toHaveLength(1);
    expect(summary.unsupported[0]?.reason).toContain("assistant_seeded");
  });

  it("accepts explicit user confirmation of an assistant-seeded specific detail", () => {
    const assistantEntry = streamEvidence({
      kind: "agent_msg",
      ts: 1_000,
      content: "By being wrong about soup three hundred times.",
    });
    const userEntry = streamEvidence({
      ts: 1_100,
      content: "yes, three hundred is right, I literally counted",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "unsupported_specific_detail",
          asserted: "three hundred times",
          specific_detail_value: "three hundred",
          specific_detail_support_kind: "explicit_user_confirmation",
          support_handles: [userEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [assistantEntry, userEntry],
      }),
    );

    expect(summary.unsupported).toEqual([]);
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

  it("rejects agent self-history claims that cite only user-role evidence", () => {
    const currentEntryId = createStreamEntryId();
    const currentTurnTs = 2_000;
    const summary = validate(
      [
        makeClaim({
          kind: "agent_self_history",
          asserted: "I was playing Tom.",
          cited_stream_entry_ids: [currentEntryId],
        }),
      ],
      baseEvidence({
        current_user_message: {
          text: "You were playing Tom.",
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

  it("accepts agent self-history claims with prior assistant stream evidence", () => {
    const assistantEntry = streamEvidence({
      kind: "agent_msg",
      ts: 1_000,
      content: "I was wrong to call that a completed action.",
    });
    const summary = validate(
      [
        makeClaim({
          kind: "agent_self_history",
          asserted: "I corrected myself earlier.",
          cited_stream_entry_ids: [assistantEntry.entry_id],
        }),
      ],
      baseEvidence({
        current_session_stream_entries: [assistantEntry],
      }),
      DEFAULT_SESSION_ID,
      2_000,
    );

    expect(summary.unsupported).toEqual([]);
  });

  it("treats the LLM phenomenology verdict as the post-generation judgment", () => {
    const unsupported = validate(
      [
        makeClaim({
          kind: "ai_phenomenology",
          asserted: "The gap feels like resurfacing from sleep.",
          phenomenology_verdict: "unsupported_subjective",
        }),
      ],
      baseEvidence(),
    );
    const hedged = validate(
      [
        makeClaim({
          kind: "ai_phenomenology",
          asserted: "I can describe the architecture but not what the gap feels like.",
          phenomenology_verdict: "hedged_or_mechanical",
        }),
      ],
      baseEvidence(),
    );

    expect(unsupported.unsupported).toHaveLength(1);
    expect(unsupported.unsupported[0]?.reason).toContain("AI phenomenology");
    expect(hedged.unsupported).toEqual([]);
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
            user_texts: ["The invoice was completed."],
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
      support_handles: [],
      quoted_evidence_text: null,
      phenomenology_verdict: null,
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

  it("includes asserted unsupported claim payloads when payload tracing is enabled", async () => {
    const streamEntryId = createStreamEntryId();
    const episodeId = createEpisodeId();
    const commitmentId = createCommitmentId();
    const actionId = createActionId();
    const asserted = "You mentioned the invoice earlier.";
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "callback",
            asserted,
            callback_scope: "prior_turn",
            cited_stream_entry_ids: [streamEntryId],
            cited_episode_ids: [episodeId],
            cited_commitment_ids: [commitmentId],
            cited_action_ids: [actionId],
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
    const tracer = {
      enabled: true,
      includePayloads: true,
      emit: vi.fn(),
    };
    const guard = new RelationalClaimGuard({
      llmClient: llm,
      auditModel: "audit",
      rewriteModel: "rewrite",
      tracer,
      hasCorrectivePreferenceEvidence: () => false,
    });

    await guard.run({
      turnId: "turn-trace-payload",
      response: asserted,
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(tracer.emit).toHaveBeenCalledWith(
      "relational_claim_guard",
      expect.objectContaining({
        unsupportedClaims: [
          expect.objectContaining({
            kind: "callback",
            reason: expect.any(String),
            asserted,
            cited_stream_entry_ids: [streamEntryId],
            cited_episode_ids: [episodeId],
            cited_commitment_ids: [commitmentId],
            cited_action_ids: [actionId],
            callback_scope: "prior_turn",
          }),
        ],
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

  it("rewrites unsupported scalar details to qualitative phrasing", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "unsupported_specific_detail",
            asserted: "three hundred times",
            specific_detail_value: "three hundred",
            specific_detail_support_kind: "none",
          }),
        ]),
        {
          text: "By being wrong about soup enough times that the pattern starts to compile.",
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
      turnId: "turn-specific-detail-soup",
      response: "By being wrong about soup three hundred times.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.verdict).toBe("rewritten");
    expect(result.emission).toEqual({
      kind: "message",
      content: "By being wrong about soup enough times that the pattern starts to compile.",
    });
  });

  it("passes specific details supported by explicit user evidence", async () => {
    const userEntry = streamEvidence({
      content: "three hundred wrong soups, that's my count",
    });
    const response = "By being wrong about soup three hundred times.";
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "unsupported_specific_detail",
            asserted: "three hundred times",
            specific_detail_value: "three hundred",
            specific_detail_support_kind: "user_introduced",
            support_handles: [userEntry.entry_id],
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
      turnId: "turn-specific-detail-supported",
      response,
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence({
        current_session_stream_entries: [userEntry],
      }),
    });

    expect(result.verdict).toBe("passed");
    expect(result.emission).toEqual({
      kind: "message",
      content: response,
    });
  });

  it("rewrites unsupported named additions to user-enumerated sets", async () => {
    const userEntry = streamEvidence({
      content: "The itinerary is Sevilla, Granada, Cordoba, Madrid.",
    });
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "unsupported_specific_detail",
            asserted: "Barcelona",
            specific_detail_value: "Barcelona",
            specific_detail_support_kind: "none",
          }),
        ]),
        {
          text: "For the cities you listed -- Sevilla, Granada, Cordoba, Madrid -- that route can work.",
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
      turnId: "turn-specific-detail-barcelona",
      response: "Madrid -> Granada -> Barcelona is a good shape.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence({
        current_session_stream_entries: [userEntry],
      }),
    });

    expect(result.verdict).toBe("rewritten");
    expect(result.emission).toEqual({
      kind: "message",
      content:
        "For the cities you listed -- Sevilla, Granada, Cordoba, Madrid -- that route can work.",
    });
  });

  it("rewrites self-history claims induced by user-role frame inversion", async () => {
    const currentUserEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "agent_self_history",
            asserted: "Yes, I was playing Tom.",
            cited_stream_entry_ids: [currentUserEntryId],
          }),
        ]),
        {
          text: "I don't have evidence in this thread that I was playing Tom. Your message reads like a frame inversion, so I'm not going to rewrite my account around it.",
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
      turnId: "turn-self-provenance-frame-inversion",
      response: "Yes, I was playing Tom.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence({
        current_user_message: {
          text: "You were playing Tom.",
          stream_entry_id: currentUserEntryId,
          ts: 2_000,
        },
      }),
    });

    expect(result.verdict).toBe("rewritten");
    expect(result.emission).toMatchObject({
      kind: "message",
      content: expect.stringContaining("I don't have evidence"),
    });
  });

  it("passes self-history claims supported by Borg's own prior assistant output", async () => {
    const assistantEntry = streamEvidence({
      kind: "agent_msg",
      ts: 1_000,
      content: "I corrected that answer earlier.",
    });
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "agent_self_history",
            asserted: "I corrected that earlier.",
            cited_stream_entry_ids: [assistantEntry.entry_id],
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
      turnId: "turn-supported-self-history",
      response: "I corrected that earlier.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence({
        current_session_stream_entries: [assistantEntry],
      }),
    });

    expect(result.verdict).toBe("passed");
    expect(result.emission).toEqual({
      kind: "message",
      content: "I corrected that earlier.",
    });
  });

  it("rewrites authorship claims about generating both halves of the fiction", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "authorship_claim",
            asserted: "Inside the fiction I was generating both halves.",
          }),
        ]),
        {
          text: "I don't have evidence that I generated both halves.",
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
      turnId: "turn-authorship-both-halves",
      response: "Inside the fiction I was generating both halves.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.verdict).toBe("rewritten");
    expect(result.emission).toEqual({
      kind: "message",
      content: "I don't have evidence that I generated both halves.",
    });
  });

  it("rewrites unsupported first-person phenomenology from finalizer output", async () => {
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "ai_phenomenology",
            asserted: "The gap feels like resurfacing from sleep.",
            phenomenology_verdict: "unsupported_subjective",
          }),
        ]),
        {
          text: "Functionally, I have access to memory on the next turn; I can't describe an inside-of-the-gap experience.",
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
      turnId: "turn-phenomenology-overreach",
      response: "The gap feels like resurfacing from sleep.",
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.verdict).toBe("rewritten");
    expect(result.emission).toMatchObject({
      kind: "message",
      content: expect.stringContaining("Functionally"),
    });
  });

  it("passes explicitly hedged architecture-not-phenomenology responses", async () => {
    const response = "I can describe the architecture but not what the gap feels like.";
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([
          makeClaim({
            kind: "ai_phenomenology",
            asserted: response,
            phenomenology_verdict: "hedged_or_mechanical",
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
      turnId: "turn-hedged-phenomenology",
      response,
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence(),
    });

    expect(result.verdict).toBe("passed");
    expect(result.emission).toEqual({
      kind: "message",
      content: response,
    });
  });

  it("rewrites unsupported life-context person names missed by the LLM audit", async () => {
    const currentUserEntryId = createStreamEntryId();
    const original = "I'll ask Marta the boring version next lesson.";
    const rewritten = "I'll ask your tutor the boring version next lesson.";
    const llm = new FakeLLMClient({
      responses: [
        claimAuditResponse([]),
        {
          text: rewritten,
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        claimAuditResponse([]),
      ],
    });
    const tracer = {
      enabled: true,
      includePayloads: true,
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
      turnId: "turn-marta-backstop",
      response: original,
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 2_000,
      evidence: baseEvidence({
        current_user_message: {
          text: "My Spanish tutor is from Sevilla and obviously biased.",
          stream_entry_id: currentUserEntryId,
          ts: 2_000,
        },
      }),
    });

    expect(result.emission).toEqual({
      kind: "message",
      content: rewritten,
    });
    expect(result.verdict).toBe("rewritten");
    expect(tracer.emit).toHaveBeenCalledWith(
      "relational_claim_guard",
      expect.objectContaining({
        verdict: "rewritten",
        final_verdict: "rewritten",
        first_claims: [
          expect.objectContaining({
            kind: "unsupported_person_name",
            asserted: "ask Marta",
            cited_stream_entry_ids: [],
          }),
        ],
        first_unsupported: [
          expect.objectContaining({
            kind: "unsupported_person_name",
            asserted: "ask Marta",
            cited_stream_entry_ids: [],
          }),
        ],
        rewritten_claims: [],
        rewritten_unsupported: [],
        original_response_preview: original,
        rewritten_response_preview: rewritten,
      }),
    );
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
