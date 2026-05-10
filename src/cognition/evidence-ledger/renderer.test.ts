import { describe, expect, it } from "vitest";

import { EVIDENCE_LEDGER_SECTION_DEFINITIONS, type EvidenceLedger } from "./types.js";
import { renderEvidenceLedger } from "./renderer.js";

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
