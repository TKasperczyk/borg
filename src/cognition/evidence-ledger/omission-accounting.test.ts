import { describe, expect, it } from "vitest";
import { createSessionId, createStreamEntryId } from "../../util/ids.js";
import type { TranscriptStreamEntry } from "../../stream/index.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import { compactTranscriptEntries } from "./transcript-compaction.js";
import { estimateEvidenceLedgerPromptTokens, renderEvidenceLedger } from "./finalizer-ledger.js";
import { omissionCounts, SECTION_ACCOUNTING_MAX_CHARS } from "./omission-accounting.js";
import { compactEvidenceLedger, REQUIRED_ENTRY_TEXT_STUB } from "./compaction.js";
import { buildCompactPlannerLedgerPrompt } from "./compact-planner.js";
import { renderEntry, renderSection } from "./section-rendering.js";
import {
  EVIDENCE_LEDGER_SECTION_DEFINITIONS,
  type EvidenceLedger,
  type EvidenceLedgerEntry,
} from "./types.js";

function entry(id: string): EvidenceLedgerEntry {
  return {
    id,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 50,
    text: `Detail ${id}. `.repeat(40),
  };
}

function ledger(): EvidenceLedger {
  const actions = Array.from({ length: 12 }, (_, index) => ({
    ...entry(`action_thread:${index}`),
    source_type: "action_record" as const,
  }));
  const summary: EvidenceLedgerEntry = {
    ...entry("action_threads:older_summary"),
    state: "omitted",
    omission_accounting: {
      action_threads: {
        counts: {
          omittedThreadCount: 5,
          omittedRecordCount: 5,
          renderedThreadCount: 12,
          renderLimit: 12,
          threadsBuiltCount: 17,
          consideredRecordCount: 17,
          sourceRecordLimit: 256,
          sourceRecordTotal: 50,
          salienceDroppedThreadCount: 0,
          staleOpenOmitted: 20,
          staleOpenCountScope: "store",
        },
        assembled_entries: actions.map((action, index) => ({
          id: action.id,
          stale_open: index >= 10,
        })),
      },
    },
  };
  return {
    sections: EVIDENCE_LEDGER_SECTION_DEFINITIONS.map((section) => ({
      ...section,
      entries: section.id === "action_states" ? [...actions, summary] : [],
    })),
    transcriptIncluded: false,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
  };
}

const ENTRY_TRUNCATION_MARKER = /\n\[evidence ledger entry truncated (\d+) chars\]$/;
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u;

describe("entry truncation marker", () => {
  const MAX_ENTRY_TEXT_TOKENS = 100;
  const MAX_ENTRY_TEXT_CHARS = MAX_ENTRY_TEXT_TOKENS * 4;

  function truncatedEntryText(text: string): string {
    const long: EvidenceLedgerEntry = { ...entry("semantic_node:long"), text };
    const result = compactEvidenceLedger(
      {
        ...ledger(),
        sections: EVIDENCE_LEDGER_SECTION_DEFINITIONS.map((section) => ({
          ...section,
          entries: section.id === "semantic_graph" ? [long] : [],
        })),
      },
      {
        targetTokens: 100_000,
        hardCapTokens: 100_000,
        maxEntryTextTokens: MAX_ENTRY_TEXT_TOKENS,
      },
    );

    return result.ledger.sections
      .find((section) => section.id === "semantic_graph")!
      .entries.find((row) => row.id === long.id)!.text!;
  }

  function expectExactCount(text: string): string {
    const shown = truncatedEntryText(text);
    const marker = ENTRY_TRUNCATION_MARKER.exec(shown);

    expect(marker).not.toBeNull();

    const kept = shown.slice(0, marker!.index);

    expect(text.startsWith(kept)).toBe(true);
    expect(Number(marker![1])).toBe(text.length - kept.length);
    expect(shown.length).toBeLessThanOrEqual(MAX_ENTRY_TEXT_CHARS);

    return kept;
  }

  // Overflows on both sides of each point where the count gains a digit, with the marker's own
  // length included: the count is settled against the kept text, not against the cap.
  it.each([1, 9, 10, 53, 54, 55, 99, 100, 953, 954, 955, 999, 1_000, 9_952, 9_953, 9_954, 9_999])(
    "counts every character it replaces when the text is %i over the cap",
    (overflow) => {
      expectExactCount("x".repeat(MAX_ENTRY_TEXT_CHARS + overflow));
    },
  );

  it("counts whitespace trimmed at the cut as replaced", () => {
    const text = `${"x".repeat(300)}${" ".repeat(200)}${"y".repeat(400)}`;
    const kept = expectExactCount(text);

    expect(kept).toBe("x".repeat(300));
  });

  it("leaves a text at the cap whole", () => {
    const text = "x".repeat(MAX_ENTRY_TEXT_CHARS);

    expect(truncatedEntryText(text)).toBe(text);
  });

  it.each(["", "a"])("never ends the kept text inside an astral character (lead %j)", (lead) => {
    const kept = expectExactCount(`${lead}${"\u{1F600}".repeat(600)}`);

    expect(LONE_SURROGATE.test(kept)).toBe(false);
  });
});

describe("bounded omission accounting under ledger compaction", () => {
  it.each([
    { sectionOptions: { action_states: { maxEntries: 4, maxTokens: 100_000 } } },
    { sectionOptions: { action_states: { maxEntries: 32, maxTokens: 700 } } },
    { targetTokens: 100 },
  ])("retains one counts-only entry with reconciled action counts: %j", (options) => {
    const input = ledger();
    const before = structuredClone(input);
    const result = compactEvidenceLedger(input, {
      targetTokens: 100_000,
      hardCapTokens: 100_000,
      ...options,
    });
    const section = result.ledger.sections.find((section) => section.id === "action_states")!;
    const shown = section.entries.filter((row) => row.source_type === "action_record").length;
    const accounting = section.entries.filter((row) => row.omission_accounting !== undefined);
    expect(shown).toBeLessThan(12);
    expect(accounting).toHaveLength(1);
    expect(omissionCounts(accounting[0]!)).toMatchObject({
      threads_built: 17,
      rendered_threads: shown,
      draw_omitted_threads: 5,
      later_cut_threads: 12 - shown,
      budget_entries: 12 - shown,
      records_below_draw_floor: 33,
      older_open_store_not_shown: 22,
    });
    expect(renderEntry(accounting[0]!)).not.toContain("Detail");
    expect(renderEntry(accounting[0]!).length).toBeLessThanOrEqual(SECTION_ACCOUNTING_MAX_CHARS);
    expect(estimatePromptTokens(renderSection(section))).toBeLessThanOrEqual(
      options.sectionOptions?.action_states.maxTokens ?? 5_000,
    );
    expect(input).toEqual(before);
  });

  it("compacts description samples separately from the protected action counts", () => {
    const input = ledger();
    input.sections
      .find((section) => section.id === "action_states")!
      .entries.push({
        ...entry("action_threads:older_samples"),
        text: "Private description samples",
        state: "omitted_samples",
      });
    const result = compactEvidenceLedger(input, {
      sectionOptions: { action_states: { maxEntries: 5 } },
    });
    const section = result.ledger.sections.find((section) => section.id === "action_states")!;
    const summary = section.entries.find((row) => row.omission_accounting !== undefined)!;
    const shown = section.entries.filter((row) => row.source_type === "action_record").length;
    expect(omissionCounts(summary)).toMatchObject({
      budget_entries: 13 - shown,
      later_cut_threads: 12 - shown,
    });
    expect(renderSection(section)).not.toContain("Private description samples");
    expect(section.entries.some((row) => row.id === "action_threads:older_samples")).toBe(false);
  });

  function alternatingTranscript() {
    const input = ledger();
    input.sections.forEach((section) => {
      section.entries = [];
    });
    const sessionId = createSessionId();
    const entries: TranscriptStreamEntry[] = Array.from({ length: 400 }, (_, index) => ({
      id: createStreamEntryId(),
      timestamp: index,
      session_id: sessionId,
      kind: index % 2 === 0 ? "agent_observed" : "user_msg",
      content: `Transcript content ${index}. `.repeat(100),
      turn_status: "active",
      compressed: false,
      sender_entity_id: null,
      reply_target_entity_id: null,
    }));
    const transcript = compactTranscriptEntries({
      entries,
      budget: 1,
      resolver: {
        currentSessionId: sessionId,
        streamEntriesById: new Map(entries.map((row) => [row.id, row])),
        streamOrderById: new Map(entries.map((row, index) => [row.id, index])),
        episodeScopesById: new Map(),
        episodeSourceStreamIdsById: new Map(),
      },
    });
    input.sections.find((section) => section.id === "current_session_transcript")!.entries =
      transcript.entries;
    input.sections.find((section) => section.id === "current_user_message")!.entries = [
      {
        ...entry("current_question"),
        source_type: "current_user_message",
        text: "What should happen next?",
      },
    ];
    return input;
  }

  it("fits 400 alternating transcript entries within 8,000 tokens, with ordinary rows and one accounting entry", () => {
    const input = alternatingTranscript();
    const original = input.sections.find(
      (section) => section.id === "current_session_transcript",
    )!.entries;
    expect(original.filter((row) => row.omission_counts !== undefined).length).toBeGreaterThan(190);
    expect(original.every((row) => row.omission_accounting === undefined)).toBe(true);
    const result = compactEvidenceLedger(input);
    const section = result.ledger.sections.find(
      (section) => section.id === "current_session_transcript",
    )!;
    const notes = section.entries.filter((row) => row.omission_counts !== undefined);
    const ordinary = section.entries.filter((row) => row.source_type === "current_session_stream");
    const accounting = section.entries.filter((row) => row.omission_accounting !== undefined);
    expect(ordinary.length).toBeGreaterThan(0);
    expect(accounting).toHaveLength(1);
    const counts = omissionCounts(accounting[0]!);
    expect(
      ordinary.length +
        notes.reduce((sum, row) => sum + row.omission_counts!.transcript_entries!, 0) +
        counts.transcript_entries! +
        counts.budget_entries! -
        counts.budget_note_entries!,
    ).toBe(400);
    expect(renderEntry(accounting[0]!).length).toBeLessThanOrEqual(SECTION_ACCOUNTING_MAX_CHARS);
    expect(estimatePromptTokens(renderSection(section))).toBeLessThanOrEqual(8_000);
    expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(100_000);
  });

  it("keeps the current question within a 1,500-token global hard cap", () => {
    const result = compactEvidenceLedger(alternatingTranscript(), { hardCapTokens: 1_500 });
    expect(renderEvidenceLedger(result.ledger)).toContain("What should happen next?");
    expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(1_500);
  });

  function questionLedger(text: string): EvidenceLedger {
    const input = ledger();
    input.sections.forEach((section) => {
      section.entries = [];
    });
    input.sections.find((section) => section.id === "current_user_message")!.entries = [
      {
        id: "current_user_message:strm_aaaaaaaaaaaaaaaa",
        source_type: "current_user_message",
        session_scope: "current_session",
        actor: "user",
        trust_rank: 100,
        text,
        taint: "none",
      },
    ];
    return input;
  }

  it.each([1_500, 1_000])(
    "fits a 5,400-character question including its envelope within the section and %i-token page caps",
    (hardCapTokens) => {
      const input = questionLedger("x".repeat(5_400));
      const before = structuredClone(input);
      const result = compactEvidenceLedger(input, { hardCapTokens });
      const section = result.ledger.sections.find(
        (section) => section.id === "current_user_message",
      )!;
      expect(section.entries).toHaveLength(1);
      expect(section.entries[0]!.id).toBe("current_user_message:strm_aaaaaaaaaaaaaaaa");
      expect(section.entries[0]!.text).toContain("x".repeat(100));
      expect(section.entries[0]!.text).toContain(REQUIRED_ENTRY_TEXT_STUB);
      expect(estimatePromptTokens(renderEntry(section.entries[0]!))).toBeLessThanOrEqual(1_200);
      expect(estimatePromptTokens(renderSection(section))).toBeLessThanOrEqual(1_200);
      expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(hardCapTokens);
      expect(result.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
      expect(section.entries[0]!.omission_accounting).toBeUndefined();
      expect(input).toEqual(before);
    },
  );

  it("budgets escaped tags and Unicode as rendered without cutting a surrogate pair", () => {
    const input = questionLedger("🌱<borg_tag>".repeat(600));
    const result = compactEvidenceLedger(input, { hardCapTokens: 1_000 });
    const section = result.ledger.sections.find(
      (section) => section.id === "current_user_message",
    )!;
    const retained = section.entries[0]!.text!;
    expect(retained).toContain(REQUIRED_ENTRY_TEXT_STUB);
    expect(new TextDecoder().decode(new TextEncoder().encode(retained))).toBe(retained);
    expect(estimatePromptTokens(renderSection(section))).toBeLessThanOrEqual(1_200);
    expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(1_000);
    expect(result.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
  });

  it("returns an explicit over-budget outcome when the required envelope and stub cannot fit", () => {
    const input = questionLedger("x".repeat(5_400));
    const result = compactEvidenceLedger(input, {
      sectionOptions: { current_user_message: { maxTokens: 1 } },
      hardCapTokens: 1,
    });
    const section = result.ledger.sections.find(
      (section) => section.id === "current_user_message",
    )!;
    expect(section.entries).toHaveLength(1);
    expect(section.entries[0]!.text).toBe(REQUIRED_ENTRY_TEXT_STUB);
    expect(result.traceSummary.budgetOutcome).toEqual({
      status: "over_budget",
      reason: "irreducible_framing",
      violations: [
        {
          scope: "section",
          section_id: "current_user_message",
          cap_tokens: 1,
          rendered_tokens: estimatePromptTokens(renderSection(section)),
          required_entry_ids: [section.entries[0]!.id],
        },
        {
          scope: "page",
          cap_tokens: 1,
          rendered_tokens: estimateEvidenceLedgerPromptTokens(result.ledger),
          required_entry_ids: [section.entries[0]!.id],
        },
      ],
    });
    expect(result.traceSummary.omittedEntryCountsBySection.current_user_message).toBe(0);
  });

  it("reports a section minimum violation even when the page cap has room", () => {
    const result = compactEvidenceLedger(questionLedger("x".repeat(5_400)), {
      sectionOptions: { current_user_message: { maxTokens: 1 } },
    });
    expect(result.traceSummary.budgetOutcome).toMatchObject({
      status: "over_budget",
      violations: [
        expect.objectContaining({ scope: "section", section_id: "current_user_message" }),
      ],
    });
  });

  it("enforces the complete required-entry budget in the compact planner too", () => {
    const input = questionLedger("x".repeat(5_400));
    const result = buildCompactPlannerLedgerPrompt(input, {
      hardCapTokens: 1_000,
      maxEntryTextTokens: 1_200,
    });
    expect(estimatePromptTokens(result.promptSection!)).toBeLessThanOrEqual(1_000);
    expect(result.traceSummary.estimatedTokensBySection.current_user_message).toBeLessThanOrEqual(
      1_200,
    );
    expect(result.promptSection).toContain(REQUIRED_ENTRY_TEXT_STUB);
    expect(result.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
    const impossible = buildCompactPlannerLedgerPrompt(input, { hardCapTokens: 1 });
    expect(impossible.promptSection).toContain("current_user_message:strm_aaaaaaaaaaaaaaaa");
    expect(impossible.promptSection).toContain(REQUIRED_ENTRY_TEXT_STUB);
    expect(impossible.traceSummary.budgetOutcome).toMatchObject({
      status: "over_budget",
      reason: "irreducible_framing",
      violations: [expect.objectContaining({ scope: "page", cap_tokens: 1 })],
    });
  });

  it("folds section counts into the one global line when section accounting cannot fit", () => {
    const result = compactEvidenceLedger(ledger(), {
      sectionOptions: { action_states: { maxTokens: 1 } },
      hardCapTokens: 1_500,
    });
    const section = result.ledger.sections.find((section) => section.id === "action_states")!;
    expect(estimatePromptTokens(renderSection(section))).toBeLessThanOrEqual(1);
    expect(section.entries).toHaveLength(0);
    expect(result.ledger.compaction_accounting?.folded_sections.action_states).toMatchObject({
      budget_entries: 12,
      later_cut_threads: 12,
      older_open_store_not_shown: 22,
    });
    expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(1_500);
  });

  it("carries already folded counts through finalizer-to-planner compaction without counting them twice", () => {
    const first = compactEvidenceLedger(ledger(), {
      sectionOptions: { action_states: { maxTokens: 1 } },
    });
    const second = compactEvidenceLedger(first.ledger);
    expect(second.ledger.compaction_accounting).toEqual(first.ledger.compaction_accounting);
    const planner = buildCompactPlannerLedgerPrompt(first.ledger);
    const globalLine = planner
      .promptSection!.split("\n")
      .find((line) => line.startsWith("Ledger compaction counts:"))!;
    expect(JSON.parse(globalLine.slice("Ledger compaction counts: ".length))).toEqual(
      first.ledger.compaction_accounting,
    );
    expect(estimatePromptTokens(planner.promptSection!)).toBeLessThanOrEqual(8_000);
  });

  it("keeps required content and puts impossible accounting in the trace instead of the page", () => {
    const input = alternatingTranscript();
    const result = compactEvidenceLedger(input, { hardCapTokens: 1 });
    expect(renderEvidenceLedger(result.ledger)).toContain("current_question");
    expect(renderEvidenceLedger(result.ledger)).toContain(REQUIRED_ENTRY_TEXT_STUB);
    expect(result.traceSummary.budgetOutcome.status).toBe("over_budget");
    expect(renderEvidenceLedger(result.ledger)).not.toContain("Omission counts:");
    expect(result.ledger.compaction_accounting).toBeUndefined();
    expect(result.traceSummary.accountingOmittedFromPage?.current_session_transcript).toMatchObject(
      {
        budget_entries: 400,
        transcript_entries: 196,
        budget_note_entries: 196,
      },
    );
    expect(result.ledger.sections.flatMap((section) => section.entries)).toHaveLength(1);
  });

  it("enforces the compact planner hard cap and preserves its current question too", () => {
    const input = alternatingTranscript();
    input.sections.find((section) => section.id === "action_states")!.entries =
      ledger().sections.find((section) => section.id === "action_states")!.entries;
    const result = buildCompactPlannerLedgerPrompt(input, {
      targetTokens: 1_500,
      hardCapTokens: 1_500,
    });
    expect(result.promptSection).toContain("What should happen next?");
    expect(estimatePromptTokens(result.promptSection ?? "")).toBeLessThanOrEqual(1_500);
    const impossible = buildCompactPlannerLedgerPrompt(input, {
      targetTokens: 1,
      hardCapTokens: 1,
    });
    expect(impossible.promptSection).toContain("current_question");
    expect(impossible.promptSection).toContain(REQUIRED_ENTRY_TEXT_STUB);
    expect(impossible.traceSummary.budgetOutcome.status).toBe("over_budget");
    expect(impossible.promptSection).not.toContain("Omission counts:");
    expect(impossible.traceSummary.accountingOmittedFromPage?.action_states?.budget_entries).toBe(
      12,
    );
  });

  it.each(["episodes", "current_session_transcript", "recent_lived_experience"] as const)(
    "combines multiple accounting entries in %s and never loses counts on a section drop",
    (id) => {
      const input = ledger();
      input.sections.forEach((section) => {
        section.entries = [];
      });
      input.sections.find((section) => section.id === id)!.entries = [
        entry("first"),
        { ...entry("accounting_a"), omission_accounting: { counts: { earlier_entries: 37 } } },
        { ...entry("accounting_b"), omission_accounting: { counts: { earlier_entries: 5 } } },
        entry("last"),
      ];
      const result = compactEvidenceLedger(input, { sectionOptions: { [id]: { maxEntries: 1 } } });
      const kept = result.ledger.sections.find((section) => section.id === id)!;
      expect(kept.entries).toHaveLength(1);
      expect(omissionCounts(kept.entries[0]!)).toMatchObject({
        earlier_entries: 42,
        budget_entries: 2,
      });
      const dropped = compactEvidenceLedger(input, { hardCapTokens: 1 });
      expect(dropped.traceSummary.accountingOmittedFromPage?.[id]).toMatchObject({
        earlier_entries: 42,
        budget_entries: 2,
      });
    },
  );

  it("does not dedupe accounting through a row sharing its provenance", () => {
    const input = ledger();
    input.sections.forEach((section) => {
      section.entries = [];
    });
    input.sections.find((section) => section.id === "current_user_message")!.entries = [
      {
        ...entry("current_user_message:str_shared"),
        state_metadata: { source_stream_ids: ["str_shared"] },
      },
    ];
    input.sections.find((section) => section.id === "episodes")!.entries = [
      {
        ...entry("accounting"),
        omission_accounting: { counts: { earlier_entries: 37 } },
        state_metadata: { source_stream_ids: ["str_shared"] },
      },
    ];
    const result = compactEvidenceLedger(input);
    const kept = result.ledger.sections.find((section) => section.id === "episodes")!.entries;
    expect(kept).toHaveLength(1);
    expect(omissionCounts(kept[0]!)).toMatchObject({ earlier_entries: 37 });
  });
});
