import { describe, expect, it } from "vitest";

import { estimatePromptTokens } from "../../util/token-estimate.js";
import { EVIDENCE_LEDGER_SECTION_DEFINITIONS, type EvidenceLedger } from "./types.js";
import {
  buildCompactPlannerLedgerPrompt,
  renderCompactPlannerLedger,
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
