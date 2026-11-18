import { describe, expect, it } from "vitest";
import {
  publicMemoryDisclosureLabel,
  renderMemoryDisclosureLabelForModel,
} from "../../memory/common/disclosure-label.js";
import { openQuestionRecordSchema } from "../../memory/self/types.js";
import { resolutionCorrectionSchema } from "../../memory/self/resolution-corrections.js";
import { createEpisodeId, createOpenQuestionId } from "../../util/ids.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import { compactEvidenceLedger } from "./compaction.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "./entry-metadata.js";
import { estimateEvidenceLedgerPromptTokens, renderEvidenceLedger } from "./finalizer-ledger.js";
import { omissionCounts, SECTION_ACCOUNTING_MAX_CHARS } from "./omission-accounting.js";
import { openQuestionStateMetadata } from "./open-question-handles.js";
import { ledgerCorrectionRows } from "./resolution-correction-budget.js";
import { renderEntry, renderSection } from "./section-rendering.js";
import type { EvidenceLedger, EvidenceLedgerEntry, EvidenceLedgerSection } from "./types.js";

const NOW_MS = Date.parse("2026-11-18T12:00:00.000Z");
const SECTION_CAP = 2_500;
const PAGE_CAP = 10_000;
const DISCLOSURE = publicMemoryDisclosureLabel();

function correctedNote(
  noteChars: number,
  correctionCount: number,
  correctionChars = 2_000,
): EvidenceLedgerEntry {
  const evidenceId = createEpisodeId();
  const question = openQuestionRecordSchema.parse({
    id: createOpenQuestionId(),
    question: "Which reading did the evidence support?",
    urgency: 0.5,
    status: "resolved",
    related_episode_ids: [],
    related_semantic_node_ids: [],
    provenance: { kind: "manual" },
    source: "reflection",
    created_at: NOW_MS - 10_000,
    last_touched: NOW_MS,
    resolution_evidence_episode_ids: [evidenceId],
    resolution_evidence_stream_entry_ids: [],
    resolution_note: "n".repeat(noteChars),
    resolution_corrections: Array.from({ length: correctionCount }, (_, index) =>
      resolutionCorrectionSchema.parse({
        seq: index + 1,
        created_at: NOW_MS + index,
        text: "c".repeat(correctionChars),
        targets_correction_seq: null,
        evidence: [{ id: evidenceId, basis: "own_read" }],
        writer: {
          author: "borg",
          turn_id: `correction-turn-${index}`,
          turn_kind: "autonomous",
          path: "tool",
          provenance: { kind: "manual" },
        },
        disclosure_label: DISCLOSURE,
      }),
    ),
    resolved_at: NOW_MS,
    abandoned_reason: null,
    abandoned_at: null,
  });
  return {
    id: `open_question:${question.id}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "memory",
    trust_rank: 50,
    text: question.question,
    value: question.source,
    state: appendMemoryDisclosureState({ state: question.status, disclosureLabel: DISCLOSURE }),
    state_metadata: appendMemoryDisclosureStateMetadata({
      stateMetadata: openQuestionStateMetadata(question, NOW_MS),
      disclosureLabel: DISCLOSURE,
    }),
    taint: "none",
  };
}

function ledger(...entries: EvidenceLedgerEntry[]): EvidenceLedger {
  return {
    sections: [{ id: "open_questions", label: "11. Open Questions", entries }],
    transcriptIncluded: false,
    transcriptCompacted: false,
    originalTranscriptTokenEstimate: 0,
    compactedTranscriptEntryCount: 0,
    rawPreservedUserTranscriptEntryCount: 0,
    estimatedTokens: 0,
  };
}

function checkAccounting(section: EvidenceLedgerSection, total: number, omitted: number): void {
  const summaries = section.entries.filter((entry) => entry.omission_accounting !== undefined);
  expect(summaries).toHaveLength(omitted > 0 ? 1 : 0);
  if (omitted === 0) return;
  expect(omissionCounts(summaries[0]!)).toMatchObject({
    resolution_correction_count: total,
    resolution_corrections_omitted_count: omitted,
  });
  expect(renderEntry(summaries[0]!).length).toBeLessThanOrEqual(SECTION_ACCOUNTING_MAX_CHARS);
  expect(summaries[0]!.text).toBeUndefined();
  expect(estimatePromptTokens(renderSection(section))).toBeLessThanOrEqual(SECTION_CAP);
}

describe("resolution corrections inside the ledger budget", () => {
  it("keeps the 8,000-character closing note and counts its omitted 2,000-character correction within the 2,500-token cap", () => {
    const entry = correctedNote(8_000, 1);
    const input = ledger(entry);
    const before = structuredClone(input);
    expect(estimatePromptTokens(renderSection(input.sections[0]!))).toBeGreaterThan(SECTION_CAP);

    const result = compactEvidenceLedger(input, { hardCapTokens: PAGE_CAP });
    const section = result.ledger.sections[0]!;
    const shown = section.entries.find((row) => row.id === entry.id)!;
    expect(shown).toBeDefined();
    expect(shown.state_metadata).toMatchObject({
      resolution_note: "n".repeat(8_000),
      resolution_correction_count: 1,
      resolution_corrections: [],
      resolution_corrections_omitted: { count: 1, total: 1, reason: "budget" },
    });
    checkAccounting(section, 1, 1);
    expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(PAGE_CAP);
    expect(result.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
    expect(input).toEqual(before);
  });

  it("uses the remaining section space for the newest of 20 whole labelled corrections", () => {
    const entry = correctedNote(800, 20);
    const input = ledger(entry);
    expect(ledgerCorrectionRows(entry)).toHaveLength(20);
    const result = compactEvidenceLedger(input, { hardCapTokens: PAGE_CAP });
    const section = result.ledger.sections[0]!;
    const shown = section.entries.find((row) => row.id === entry.id)!;
    const rows = ledgerCorrectionRows(shown);
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.length).toBeLessThan(20);
    expect(rows).toEqual(ledgerCorrectionRows(entry).slice(0, rows.length));
    expect(rows.map((row) => row.seq)).toEqual(
      Array.from({ length: rows.length }, (_, i) => 20 - i),
    );
    for (const row of rows) {
      expect(row.text).toHaveLength(2_000);
      expect(row.disclosure_label).toEqual(DISCLOSURE);
      expect(row.disclosure).toBe(renderMemoryDisclosureLabelForModel(DISCLOSURE));
    }
    expect(shown.state_metadata).toMatchObject({
      resolution_correction_count: 20,
      resolution_corrections_omitted: { count: 20 - rows.length, total: 20, reason: "budget" },
    });
    checkAccounting(section, 20, 20 - rows.length);
    expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(PAGE_CAP);
    expect(result.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
  });

  it("shares one section budget across notes and does not double count on repeated compaction", () => {
    const first = correctedNote(1_500, 4);
    const second = correctedNote(1_500, 3);
    const result = compactEvidenceLedger(ledger(first, second));
    const section = result.ledger.sections[0]!;
    expect(section.entries.filter((row) => row.omission_accounting === undefined)).toHaveLength(2);
    const shownCount = section.entries.reduce(
      (sum, row) => sum + ledgerCorrectionRows(row).length,
      0,
    );
    checkAccounting(section, 7, 7 - shownCount);
    const repeated = compactEvidenceLedger(result.ledger);
    checkAccounting(repeated.ledger.sections[0]!, 7, 7 - shownCount);
    expect(repeated.ledger.sections).toEqual(result.ledger.sections);
  });

  it("counts corrections of an entire note removed by the entry cap", () => {
    const first = correctedNote(20, 1, 20);
    const second = correctedNote(20, 3, 20);
    const result = compactEvidenceLedger(ledger(first, second), {
      sectionOptions: { open_questions: { maxEntries: 1 } },
    });
    const section = result.ledger.sections[0]!;
    expect(section.entries.find((row) => row.id === first.id)).toBeDefined();
    expect(section.entries.find((row) => row.id === second.id)).toBeUndefined();
    checkAccounting(section, 4, 3);
    const summary = section.entries.find((row) => row.omission_accounting !== undefined)!;
    expect(omissionCounts(summary)).toMatchObject({ budget_entries: 1 });
  });

  it("omits corrections before the parent note under a global cut and retains both counts", () => {
    const entry = correctedNote(100, 2);
    const result = compactEvidenceLedger(ledger(entry), { hardCapTokens: 1_000 });
    const section = result.ledger.sections[0]!;
    const shown = section.entries.find((row) => row.id === entry.id)!;
    expect(shown).toBeDefined();
    const count = ledgerCorrectionRows(shown).length;
    expect(count).toBeLessThan(2);
    expect(shown.state_metadata?.resolution_correction_count).toBe(2);
    checkAccounting(section, 2, 2 - count);
    expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(1_000);
    expect(result.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
  });

  it("folds correction counts into global accounting when the section cannot fit and into trace only when framing cannot fit", () => {
    const input = ledger(correctedNote(800, 20));
    const result = compactEvidenceLedger(input, {
      hardCapTokens: 1_000,
      sectionOptions: { open_questions: { maxTokens: 1 } },
    });
    expect(result.ledger.compaction_accounting?.folded_sections.open_questions).toMatchObject({
      resolution_correction_count: 20,
      resolution_corrections_omitted_count: 20,
    });
    expect(renderEvidenceLedger(result.ledger)).toContain('"resolution_correction_count":20');
    expect(estimateEvidenceLedgerPromptTokens(result.ledger)).toBeLessThanOrEqual(1_000);
    expect(result.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
    const impossible = compactEvidenceLedger(input, { hardCapTokens: 1 });
    expect(impossible.traceSummary.accountingOmittedFromPage?.open_questions).toMatchObject({
      resolution_correction_count: 20,
      resolution_corrections_omitted_count: 20,
    });
    expect(impossible.traceSummary.budgetOutcome).toMatchObject({
      status: "over_budget",
      reason: "irreducible_framing",
    });
  });

  it("adds no omission entry when every correction fits", () => {
    const entry = correctedNote(100, 2, 20);
    const result = compactEvidenceLedger(ledger(entry));
    expect(result.ledger.sections[0]!.entries).toEqual([entry]);
    checkAccounting(result.ledger.sections[0]!, 2, 0);
    expect(result.traceSummary.budgetOutcome).toEqual({ status: "within_budget" });
  });
});
