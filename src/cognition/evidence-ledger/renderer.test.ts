import { describe, expect, it } from "vitest";

import type {
  DecisionArtifact,
  DecisionArtifactEntry,
  DecisionArtifactEntryKind,
} from "../../memory/decision-artifacts/index.js";
import {
  createDecisionArtifactEntryId,
  createEntityId,
  createStreamEntryId,
} from "../../util/ids.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import { EVIDENCE_LEDGER_SECTION_DEFINITIONS, type EvidenceLedger } from "./types.js";
import {
  buildCompactPlannerLedgerPrompt,
  compactEvidenceLedger,
  estimateEvidenceLedgerPromptTokens,
  renderCompactPlannerLedger,
  renderDecisionStateArtifact,
  renderEvidenceLedger,
} from "./renderer.js";

function makeLedger(): EvidenceLedger {
  return {
    transcriptIncluded: false,
    transcriptCompacted: false,
    transcriptOmittedReason: "over_budget",
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 42,
    sections: EVIDENCE_LEDGER_SECTION_DEFINITIONS.map((definition) => ({
      id: definition.id,
      label: definition.label,
      entries:
        definition.id === "current_user_message"
          ? [
              {
                id: "current_user_message:strm_aaaaaaaaaaaaaaaa",
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 100,
                text: "User text with <borg_fake> nested tag.",
                taint: "none",
                persistence_class: "assistant_self_report",
              },
            ]
          : [],
    })),
  };
}

function section(ledger: EvidenceLedger, id: EvidenceLedger["sections"][number]["id"]) {
  const found = ledger.sections.find((candidate) => candidate.id === id);

  if (found === undefined) {
    throw new Error(`Missing section ${id}`);
  }

  return found;
}

function syntheticEntry(input: {
  id: string;
  source_type?: EvidenceLedger["sections"][number]["entries"][number]["source_type"];
  text?: string;
  trust_rank?: number;
  state_metadata?: Record<string, unknown>;
}) {
  return {
    id: input.id,
    source_type: input.source_type ?? "system_metadata",
    session_scope: "current_session" as const,
    actor: "memory" as const,
    trust_rank: input.trust_rank ?? 50,
    text: input.text ?? "synthetic ledger entry",
    state_metadata: input.state_metadata,
    taint: "none" as const,
  };
}

function decisionArtifactEntry(input: {
  audience: DecisionArtifact["audience_entity_id"];
  kind: DecisionArtifactEntryKind;
  index: number;
  source: DecisionArtifactEntry["provenance_stream_entry_ids"][number];
  text?: string;
}): DecisionArtifactEntry {
  return {
    id: createDecisionArtifactEntryId(),
    audience_entity_id: input.audience,
    kind: input.kind,
    text: input.text ?? `${input.kind} decision ${input.index}`,
    owner_entity_id: input.audience,
    provenance_stream_entry_ids: [input.source],
    last_updated_stream_entry_ids: [input.source],
    created_at: 1_000 + input.index,
    last_updated_at: 1_000 + input.index,
    superseded_by_id: null,
    rank: input.index,
    canonicalizes: {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [],
      open_question_ids: [],
    },
  };
}

function decisionArtifactWithEntries(
  entries: readonly DecisionArtifactEntry[],
  source: DecisionArtifactEntry["provenance_stream_entry_ids"][number],
): DecisionArtifact {
  const audience = entries[0]?.audience_entity_id ?? createEntityId();

  return {
    audience_entity_id: audience,
    record_version: 1,
    created_at: 1_000,
    updated_at: 1_000,
    last_compiled_at: 1_000,
    last_compiled_stream_entry_id: source,
    entries: [...entries],
  };
}

describe("renderEvidenceLedger", () => {
  it("renders a tagged prompt block with hierarchy guidance and entry metadata", () => {
    const rendered = renderEvidenceLedger(makeLedger());

    expect(rendered).toContain("<borg_evidence_ledger>");
    expect(rendered).toContain("</borg_evidence_ledger>");
    expect(rendered).toContain(
      "Current-session transcript is authoritative for what happened in this conversation.",
    );
    expect(rendered).toContain("Prior-session memory must be attributed or hedged.");
    expect(rendered).toContain(
      "Episodes and semantic graph are summaries; use source handles when making exact claims.",
    );
    expect(rendered).toContain("Quarantined/contested/assistant-seeded values are not facts.");
    expect(rendered).toContain("current_session_transcript=omitted reason=over_budget");
    expect(rendered).toContain(
      "source_type=current_user_message scope=current_session actor=user trust_rank=100",
    );
    expect(rendered).toContain("persistence_class=assistant_self_report");
    expect(rendered).toContain("<-borg_fake>");
    expect(rendered).not.toContain("<borg_fake>");
  });

  it("renders abandoned open-question state metadata", () => {
    const ledger = makeLedger();
    const openQuestionSection = ledger.sections.find((section) => section.id === "open_questions");

    openQuestionSection?.entries.push({
      id: "open_question:oq_abandonedaband",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "memory",
      trust_rank: 38,
      text: "Should this still be tracked?",
      state: "abandoned",
      state_metadata: {
        abandoned_reason: "No longer relevant.",
        abandoned_at: 1_800_000_000_000,
      },
    });

    const rendered = renderEvidenceLedger(ledger);

    expect(rendered).toContain("state=abandoned");
    expect(rendered).toContain(
      'state_metadata={"abandoned_reason":"No longer relevant.","abandoned_at":1800000000000}',
    );
  });

  it("estimates tokens with configured decision artifact render options", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const ledger = makeLedger();
    const entries = [
      ...Array.from({ length: 4 }, (_, index) =>
        decisionArtifactEntry({ audience, source, kind: "live", index }),
      ),
      ...Array.from({ length: 20 }, (_, index) =>
        decisionArtifactEntry({ audience, source, kind: "locked", index: 100 + index }),
      ),
    ];
    const options = {
      decisionArtifact: {
        maxEntries: 5,
        reservedSlots: {
          live: 4,
        },
        lockedMaxEntries: 1,
      },
    };

    ledger.decisionArtifact = decisionArtifactWithEntries(entries, source);
    const rendered = renderEvidenceLedger(ledger, options) ?? "";

    expect(rendered.match(/kind=live/g)?.length ?? 0).toBe(4);
    expect(rendered.match(/kind=locked/g)?.length ?? 0).toBe(1);
    expect(estimateEvidenceLedgerPromptTokens(ledger, options)).toBe(
      estimatePromptTokens(rendered),
    );
  });
});

describe("renderDecisionStateArtifact", () => {
  it("caps a single oversized locked entry", () => {
    const now = 1_000;
    const audience = createEntityId();
    const source = createStreamEntryId();
    const artifact: DecisionArtifact = {
      audience_entity_id: audience,
      record_version: 1,
      created_at: now,
      updated_at: now,
      last_compiled_at: now,
      last_compiled_stream_entry_id: source,
      entries: [
        {
          id: createDecisionArtifactEntryId(),
          audience_entity_id: audience,
          kind: "locked",
          text: "oversized decision ".repeat(10_000),
          owner_entity_id: audience,
          provenance_stream_entry_ids: [source],
          last_updated_stream_entry_ids: [source],
          created_at: now,
          last_updated_at: now,
          superseded_by_id: null,
          rank: 0,
          canonicalizes: {
            goal_ids: [],
            commitment_ids: [],
            action_ids: [],
            open_question_ids: [],
          },
        },
      ],
    };
    const rendered = renderDecisionStateArtifact(artifact) ?? "";

    expect(estimatePromptTokens(rendered)).toBeLessThanOrEqual(3_000);
    expect(rendered).toContain(" ... [text truncated]");
  });

  it("reserves room for live entries instead of rendering locked entries first", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const entries = [
      ...Array.from({ length: 20 }, (_, index) =>
        decisionArtifactEntry({ audience, source, kind: "locked", index }),
      ),
      ...Array.from({ length: 10 }, (_, index) =>
        decisionArtifactEntry({ audience, source, kind: "live", index: 100 + index }),
      ),
    ];
    const rendered =
      renderDecisionStateArtifact(decisionArtifactWithEntries(entries, source)) ?? "";

    expect(rendered.match(/kind=live/g)?.length ?? 0).toBe(10);
    expect(rendered.match(/kind=locked/g)?.length ?? 0).toBe(14);
    expect(rendered).toContain("DecisionStateArtifact omitted:");
    expect(rendered).toContain("6 locked");
  });

  it("honors category reservations and backfills under the locked cap", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const entries = [
      ...Array.from({ length: 5 }, (_, index) =>
        decisionArtifactEntry({ audience, source, kind: "live", index }),
      ),
      ...Array.from({ length: 5 }, (_, index) =>
        decisionArtifactEntry({ audience, source, kind: "invalidated", index: 100 + index }),
      ),
      ...Array.from({ length: 5 }, (_, index) =>
        decisionArtifactEntry({ audience, source, kind: "pending", index: 200 + index }),
      ),
      ...Array.from({ length: 25 }, (_, index) =>
        decisionArtifactEntry({ audience, source, kind: "locked", index: 300 + index }),
      ),
    ];
    const rendered =
      renderDecisionStateArtifact(decisionArtifactWithEntries(entries, source)) ?? "";

    expect(rendered.match(/kind=live/g)?.length ?? 0).toBe(5);
    expect(rendered.match(/kind=invalidated/g)?.length ?? 0).toBe(5);
    expect(rendered.match(/kind=pending/g)?.length ?? 0).toBe(5);
    expect(rendered.match(/kind=locked/g)?.length ?? 0).toBe(14);
    expect(rendered).toContain("11 locked");
  });

  it("keeps one entry from each reserved category under token pressure", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const pressureText = "token pressure ".repeat(120);
    const entries = (["live", "invalidated", "pending", "locked"] as const).flatMap(
      (kind, kindIndex) =>
        Array.from({ length: 3 }, (_, index) =>
          decisionArtifactEntry({
            audience,
            source,
            kind,
            index: kindIndex * 100 + index,
            text: `${kind} ${index} ${pressureText}`,
          }),
        ),
    );
    const rendered =
      renderDecisionStateArtifact(decisionArtifactWithEntries(entries, source), {
        maxEntries: 12,
        maxTokens: 1_800,
      }) ?? "";

    expect(estimatePromptTokens(rendered)).toBeLessThanOrEqual(1_800);
    expect(rendered.match(/kind=live/g)?.length ?? 0).toBeGreaterThanOrEqual(1);
    expect(rendered.match(/kind=invalidated/g)?.length ?? 0).toBeGreaterThanOrEqual(1);
    expect(rendered.match(/kind=pending/g)?.length ?? 0).toBeGreaterThanOrEqual(1);
    expect(rendered).toContain("DecisionStateArtifact omitted:");
  });

  it("drops locked entries before collapsing the configured live reservation under token pressure", () => {
    const audience = createEntityId();
    const source = createStreamEntryId();
    const pressureText = "locked token pressure ".repeat(40);
    const entries = [
      ...Array.from({ length: 20 }, (_, index) =>
        decisionArtifactEntry({
          audience,
          source,
          kind: "locked",
          index,
          text: `locked ${index} ${pressureText}`,
        }),
      ),
      ...Array.from({ length: 8 }, (_, index) =>
        decisionArtifactEntry({
          audience,
          source,
          kind: "live",
          index: 100 + index,
          text: `live ${index}`,
        }),
      ),
    ];
    const rendered =
      renderDecisionStateArtifact(decisionArtifactWithEntries(entries, source), {
        maxEntries: 22,
        maxTokens: 1_800,
        reservedSlots: {
          live: 8,
          invalidated: 0,
          pending: 0,
        },
        lockedMaxEntries: 14,
      }) ?? "";

    expect(estimatePromptTokens(rendered)).toBeLessThanOrEqual(1_800);
    expect(rendered.match(/kind=live/g)?.length ?? 0).toBe(8);
    expect(rendered.match(/kind=locked/g)?.length ?? 0).toBeLessThan(14);
  });
});

describe("compactEvidenceLedger", () => {
  it("dedupes overlapping provenance into the highest-trust section with citations", () => {
    const ledger = makeLedger();

    section(ledger, "current_session_transcript").entries.push(
      syntheticEntry({
        id: "current_session_stream:strm_route",
        source_type: "current_session_stream",
        text: "Current transcript route fact.",
        trust_rank: 95,
      }),
    );
    section(ledger, "episodes").entries.push(
      syntheticEntry({
        id: "episode:ep_route",
        source_type: "episode",
        text: "Episode route fact.",
        trust_rank: 52,
        state_metadata: {
          episode_id: "ep_route",
          source_stream_ids: ["strm_route"],
        },
      }),
    );
    section(ledger, "semantic_graph").entries.push(
      syntheticEntry({
        id: "semantic_node:semn_route",
        source_type: "semantic_node",
        text: "Semantic route fact.",
        trust_rank: 42,
        state_metadata: {
          node_id: "semn_route",
          source_episode_ids: ["ep_route"],
        },
      }),
    );
    section(ledger, "prior_session_memory").entries.push(
      syntheticEntry({
        id: "retrieved_evidence:prior_route",
        source_type: "episode",
        text: "Prior route fact.",
        trust_rank: 30,
        state_metadata: {
          episode_id: "ep_route",
        },
      }),
    );

    const compacted = compactEvidenceLedger(ledger, {
      targetTokens: 20_000,
      hardCapTokens: 40_000,
    });
    const transcriptEntries = section(compacted.ledger, "current_session_transcript").entries;
    const canonical = transcriptEntries.find(
      (entry) => entry.id === "current_session_stream:strm_route",
    );
    const rendered = renderEvidenceLedger(compacted.ledger) ?? "";

    expect(canonical?.citations).toEqual(["ep_route", "semn_route", "strm_route"]);
    expect(section(compacted.ledger, "episodes").entries).toEqual([]);
    expect(section(compacted.ledger, "semantic_graph").entries).toEqual([]);
    expect(section(compacted.ledger, "prior_session_memory").entries).toEqual([]);
    expect(compacted.traceSummary.dedupedEntryCount).toBe(3);
    expect(rendered).toContain("[citation: ep_route, semn_route, strm_route]");
  });

  it("does not treat action-linked open question ids as action provenance", () => {
    const ledger = makeLedger();

    section(ledger, "action_states").entries.push(
      syntheticEntry({
        id: "action_thread:act_followup",
        source_type: "action_record",
        text: "Action linked to the open question.",
        trust_rank: 72,
        state_metadata: {
          current_action_id: "act_followup",
          open_question_id: "oq_followup",
          record_ids: ["act_followup"],
        },
      }),
    );
    section(ledger, "open_questions").entries.push(
      syntheticEntry({
        id: "open_question:oq_followup",
        source_type: "system_metadata",
        text: "True open question entry.",
        trust_rank: 38,
      }),
    );

    const compacted = compactEvidenceLedger(ledger, {
      targetTokens: 20_000,
      hardCapTokens: 40_000,
    });

    expect(compacted.traceSummary.dedupedEntryCount).toBe(0);
    expect(section(compacted.ledger, "action_states").entries).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: "action_thread:act_followup" })]),
    );
    expect(section(compacted.ledger, "open_questions").entries).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: "open_question:oq_followup" })]),
    );
  });

  it("preserves newest transcript entries when section caps omit chronological content", () => {
    const ledger = makeLedger();

    section(ledger, "current_session_transcript").entries = Array.from({ length: 5 }, (_, index) =>
      syntheticEntry({
        id: `current_session_stream:strm_transcript_${index}`,
        source_type: "current_session_stream",
        text: `Transcript entry ${index}`,
        trust_rank: 95,
        state_metadata: {
          stream_ids: [`strm_transcript_${index}`],
        },
      }),
    );

    const compacted = compactEvidenceLedger(ledger, {
      targetTokens: 20_000,
      hardCapTokens: 40_000,
      sectionOptions: {
        current_session_transcript: {
          maxEntries: 2,
          maxTokens: 20_000,
        },
      },
    });
    const transcriptEntryIds = section(compacted.ledger, "current_session_transcript")
      .entries.filter((entry) => entry.source_type === "current_session_stream")
      .map((entry) => entry.id);
    const rendered = renderEvidenceLedger(compacted.ledger) ?? "";

    expect(transcriptEntryIds).toEqual([
      "current_session_stream:strm_transcript_3",
      "current_session_stream:strm_transcript_4",
    ]);
    expect(rendered).toContain(
      "Evidence ledger omitted 3 older entries from current_session_transcript",
    );
  });

  it("caps oversized sections and trims to the global target with omission trailers", () => {
    const ledger = makeLedger();
    const pressureText = "budget pressure ".repeat(80);

    section(ledger, "retrieved_memory_evidence").entries = Array.from({ length: 24 }, (_, index) =>
      syntheticEntry({
        id: `retrieved_evidence:memory_${index}`,
        source_type: "episode",
        text: `${index} ${pressureText}`,
        trust_rank: 52,
        state_metadata: {
          episode_id: `ep_memory_${index}`,
        },
      }),
    );
    section(ledger, "prior_session_memory").entries = Array.from({ length: 24 }, (_, index) =>
      syntheticEntry({
        id: `retrieved_evidence:prior_${index}`,
        source_type: "episode",
        text: `${index} ${pressureText}`,
        trust_rank: 30,
        state_metadata: {
          episode_id: `ep_prior_${index}`,
        },
      }),
    );

    const compacted = compactEvidenceLedger(ledger, {
      targetTokens: 1_700,
      hardCapTokens: 5_000,
      maxEntryTextTokens: 40,
      sectionOptions: {
        retrieved_memory_evidence: {
          maxEntries: 12,
          maxTokens: 900,
        },
        prior_session_memory: {
          maxEntries: 12,
          maxTokens: 900,
        },
      },
    });
    const rendered = renderEvidenceLedger(compacted.ledger) ?? "";

    expect(compacted.traceSummary.preDedupeTokens).toBeGreaterThan(
      compacted.traceSummary.postCapTokens,
    );
    expect(compacted.traceSummary.postCapTokens).toBeLessThanOrEqual(1_700);
    expect(compacted.traceSummary.omittedEntryCountsBySection.prior_session_memory).toBeGreaterThan(
      0,
    );
    expect(rendered).toContain("Evidence ledger omitted");
    expect(rendered).toContain("lower-priority entries from prior_session_memory");
  });

  it("drops lowest-trust sections when the hard cap is exceeded", () => {
    const ledger = makeLedger();
    const pressureText = "hard cap pressure ".repeat(160);

    for (const sectionId of ["current_session_transcript", "prior_session_memory"] as const) {
      section(ledger, sectionId).entries = Array.from({ length: 12 }, (_, index) =>
        syntheticEntry({
          id:
            sectionId === "current_session_transcript"
              ? `current_session_stream:strm_hard_${index}`
              : `retrieved_evidence:prior_hard_${index}`,
          source_type:
            sectionId === "current_session_transcript" ? "current_session_stream" : "episode",
          text: `${index} ${pressureText}`,
          trust_rank: sectionId === "current_session_transcript" ? 95 : 30,
          state_metadata:
            sectionId === "prior_session_memory"
              ? {
                  episode_id: `ep_hard_${index}`,
                }
              : undefined,
        }),
      );
    }

    const compacted = compactEvidenceLedger(ledger, {
      targetTokens: 2_000,
      hardCapTokens: 5_000,
      maxEntryTextTokens: 300,
      sectionOptions: {
        current_session_transcript: {
          maxEntries: 12,
          maxTokens: 20_000,
        },
        prior_session_memory: {
          maxEntries: 12,
          maxTokens: 20_000,
        },
      },
    });
    const rendered = renderEvidenceLedger(compacted.ledger) ?? "";

    expect(compacted.traceSummary.droppedSections[0]).toBe("prior_session_memory");
    expect(compacted.traceSummary.postCapTokens).toBeLessThanOrEqual(5_000);
    expect(rendered).toContain("Evidence ledger dropped all entries from prior_session_memory");
  });
});

describe("renderCompactPlannerLedger", () => {
  it("renders only the planner-relevant ledger sections", () => {
    const ledger = makeLedger();
    const currentMessage = ledger.sections.find((section) => section.id === "current_user_message");
    const commitments = ledger.sections.find(
      (section) => section.id === "commitments_and_constraints",
    );
    const closure = ledger.sections.find((section) => section.id === "closure_discourse_state");
    const contradictions = ledger.sections.find(
      (section) => section.id === "contradictions_quarantines",
    );
    const actions = ledger.sections.find((section) => section.id === "action_states");
    const group = ledger.sections.find((section) => section.id === "group_channel_memory");
    const participants = ledger.sections.find((section) => section.id === "relational_slots");
    const transcript = ledger.sections.find(
      (section) => section.id === "current_session_transcript",
    );
    const episodes = ledger.sections.find((section) => section.id === "episodes");

    currentMessage?.entries.splice(0, currentMessage.entries.length, {
      id: "current_user_message:strm_ben",
      source_type: "current_user_message",
      session_scope: "current_session",
      actor: "user",
      trust_rank: 100,
      text: "Ben asks about a Granada to SS recovery chain.",
      state_metadata: {
        sender_entity_id: "ent_ben",
        sender_display_name: "Ben",
      },
    });
    commitments?.entries.push({
      id: "commitment:route-order",
      source_type: "commitment",
      session_scope: "current_session",
      actor: "memory",
      trust_rank: 82,
      value: "locked_spain_route_order",
      state: "active",
      text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3 / home.",
    });
    closure?.entries.push({
      id: "discourse_state:working_memory",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: 80,
      text: "mode=problem_solving; turn_counter=70",
    });
    contradictions?.entries.push({
      id: "review_queue:route-flip",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: 78,
      state: "open",
      taint: "contested",
      text: "Prior route flip claims are contested.",
    });
    actions?.entries.push({
      id: "action_thread:ss-flight",
      source_type: "action_record",
      session_scope: "current_session",
      actor: "user",
      trust_rank: 72,
      state: "completed",
      value: "Ben",
      text: "Flight booking confirms SS precedes Seville.",
    });
    group?.entries.push({
      id: "group_relational_slot:route",
      source_type: "relational_slot",
      session_scope: "current_session",
      actor: "memory",
      trust_rank: 70,
      state: "established",
      value: "trip.route_order=Madrid 3 / SS 3 / Seville 4 / Granada 3 / home",
    });
    participants?.entries.push({
      id: "relational_slot:ben-role",
      source_type: "relational_slot",
      session_scope: "current_session",
      actor: "memory",
      trust_rank: 70,
      state: "established",
      value: "participant.name=Ben",
      state_metadata: {
        subject_display_name: "Ben",
        subject_role: "speaker",
      },
    });
    transcript?.entries.push({
      id: "current_session_stream:strm_old",
      source_type: "current_session_stream",
      session_scope: "current_session",
      actor: "user",
      trust_rank: 95,
      text: "Transcript detail should not be in compact planner ledger.",
    });
    episodes?.entries.push({
      id: "episode:ep_old",
      source_type: "episode",
      session_scope: "current_session",
      actor: "memory",
      trust_rank: 52,
      text: "Episode detail should not be in compact planner ledger.",
    });

    const rendered = renderCompactPlannerLedger(ledger) ?? "";

    expect(rendered).toContain("<borg_compact_planner_ledger>");
    expect(rendered).toContain("## 1. Current User Message");
    expect(rendered).toContain("sender_display_name");
    expect(rendered).toContain("## 3. Active Commitments And Discourse Constraints");
    expect(rendered).toContain("Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3");
    expect(rendered).toContain("## 4. Current Closure And Discourse State");
    expect(rendered).toContain("## 5. Current-Session Contradictions And Quarantines");
    expect(rendered).toContain("## 6. Action States");
    expect(rendered).toContain("## 7. Group/Channel Memory");
    expect(rendered).toContain("## 8. Active Participant Memory");
    expect(rendered).not.toContain("## 2. Current-Session Transcript");
    expect(rendered).not.toContain("Transcript detail should not be in compact planner ledger.");
    expect(rendered).not.toContain("## 11. Episodes");
    expect(rendered).not.toContain("Episode detail should not be in compact planner ledger.");
  });

  it("caps oversized compact planner ledgers and reports omissions", () => {
    const ledger = makeLedger();

    for (const section of ledger.sections) {
      if (
        section.id === "current_user_message" ||
        section.id === "commitments_and_constraints" ||
        section.id === "closure_discourse_state" ||
        section.id === "contradictions_quarantines" ||
        section.id === "action_states" ||
        section.id === "group_channel_memory" ||
        section.id === "relational_slots"
      ) {
        section.entries = Array.from({ length: 80 }, (_, index) => ({
          id: `${section.id}:entry-${index}`,
          source_type:
            section.id === "action_states"
              ? "action_record"
              : section.id === "relational_slots" || section.id === "group_channel_memory"
                ? "relational_slot"
                : "system_metadata",
          session_scope: "current_session",
          actor: "memory",
          trust_rank: 70,
          state: "active",
          value: `value-${index}`,
          text: `Entry ${index} ${"budget pressure ".repeat(200)}`,
        }));
      }
    }

    const result = buildCompactPlannerLedgerPrompt(ledger);
    const rendered = result.promptSection ?? "";

    expect(estimatePromptTokens(rendered)).toBeLessThanOrEqual(8_000);
    expect(result.traceSummary.totalEstimatedTokens).toBeLessThanOrEqual(8_000);
    expect(result.traceSummary.omittedEntryCountsBySection.action_states).toBeGreaterThan(0);
    expect(rendered).toContain("Compact planner ledger omitted");
    expect(rendered).toContain("older entries");
    expect(rendered).not.toContain("## 12. Semantic Graph");
  });
});
