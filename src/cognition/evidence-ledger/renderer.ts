import { estimatePromptTokens } from "../../util/token-estimate.js";
import { UNTRUSTED_DATA_PREAMBLE } from "../deliberation/constants.js";
import { renderTaggedPromptBlock } from "../deliberation/prompt/sections.js";
import type {
  EvidenceLedger,
  EvidenceLedgerEntry,
  EvidenceLedgerSection,
  EvidenceLedgerSectionId,
} from "./types.js";

const COMPACT_PLANNER_LEDGER_SECTION_IDS = [
  "current_user_message",
  "commitments_and_constraints",
  "closure_discourse_state",
  "contradictions_quarantines",
  "action_states",
  "group_channel_memory",
  "relational_slots",
] as const satisfies readonly EvidenceLedgerSectionId[];

const DEFAULT_COMPACT_PLANNER_TARGET_TOKENS = 8_000;
const DEFAULT_COMPACT_PLANNER_HARD_CAP_TOKENS = 15_000;
const DEFAULT_COMPACT_ENTRY_TEXT_TOKEN_CAP = 600;

const DEFAULT_COMPACT_SECTION_OPTIONS = {
  current_user_message: {
    maxEntries: 1,
    maxTokens: 1_200,
  },
  commitments_and_constraints: {
    maxEntries: 32,
    maxTokens: 2_400,
  },
  closure_discourse_state: {
    maxEntries: 8,
    maxTokens: 700,
  },
  contradictions_quarantines: {
    maxEntries: 16,
    maxTokens: 1_000,
  },
  action_states: {
    maxEntries: 12,
    maxTokens: 1_800,
  },
  group_channel_memory: {
    maxEntries: 24,
    maxTokens: 1_600,
  },
  relational_slots: {
    maxEntries: 24,
    maxTokens: 1_600,
  },
} as const satisfies Record<
  (typeof COMPACT_PLANNER_LEDGER_SECTION_IDS)[number],
  {
    maxEntries: number;
    maxTokens: number;
  }
>;

export type CompactPlannerLedgerOptions = {
  targetTokens?: number;
  hardCapTokens?: number;
  maxEntryTextTokens?: number;
};

export type CompactPlannerLedgerTraceSummary = {
  entryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  omittedEntryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  estimatedTokensBySection: Record<EvidenceLedgerSectionId, number>;
  totalEstimatedTokens: number;
  targetTokens: number;
  hardCapTokens: number;
};

export type CompactPlannerLedgerPrompt = {
  promptSection: string | null;
  traceSummary: CompactPlannerLedgerTraceSummary;
};

const HIERARCHY_GUIDANCE = [
  "Current-session transcript is authoritative for what happened in this conversation.",
  "Prior-session memory must be attributed or hedged.",
  "Episodes and semantic graph are summaries; use source handles when making exact claims.",
  "Quarantined/contested/assistant-seeded values are not facts.",
].join("\n");

const COMPACT_PLANNER_LEDGER_GUIDANCE = [
  "CompactPlannerLedger: decision-relevant evidence slice for the S2 planner.",
  "Use these entries to check current-turn constraints before planning verification steps.",
  "Dialogue messages carry the conversational transcript; this compact ledger carries locked state, constraints, participant context, quarantines, and action threads.",
  "Quarantined/contested/assistant-seeded values are not facts.",
].join("\n");

function renderEntry(entry: EvidenceLedgerEntry): string {
  const stateMetadata =
    entry.state_metadata === undefined ? undefined : JSON.stringify(entry.state_metadata);
  const metadata = [
    `id=${entry.id}`,
    `source_type=${entry.source_type}`,
    `scope=${entry.session_scope}`,
    `actor=${entry.actor}`,
    `trust_rank=${entry.trust_rank}`,
    entry.stream_index === undefined ? null : `stream_index=${entry.stream_index}`,
    entry.state === undefined ? null : `state=${entry.state}`,
    stateMetadata === undefined ? null : `state_metadata=${stateMetadata}`,
    entry.taint === undefined ? null : `taint=${entry.taint}`,
    entry.persistence_class === undefined ? null : `persistence_class=${entry.persistence_class}`,
    entry.via_retrieval === true ? "via_retrieval=true" : null,
  ].filter((part): part is string => part !== null);
  const body = [
    entry.value === undefined ? null : `  value: ${entry.value}`,
    entry.text === undefined ? null : `  text:\n${entry.text}`,
  ].filter((part): part is string => part !== null);

  return [`- ${metadata.join(" ")}`, ...body].join("\n");
}

function normalizePositiveInteger(value: number | undefined, fallback: number): number {
  return value === undefined || !Number.isFinite(value) || value <= 0
    ? fallback
    : Math.floor(value);
}

function allSectionIds(): EvidenceLedgerSectionId[] {
  return [
    "current_user_message",
    "current_session_transcript",
    "commitments_and_constraints",
    "closure_discourse_state",
    "contradictions_quarantines",
    "action_states",
    "group_channel_memory",
    "relational_slots",
    "retrieved_raw_stream_evidence",
    "retrieved_memory_evidence",
    "episodes",
    "semantic_graph",
    "open_questions",
    "prior_session_memory",
  ];
}

function emptySectionCountRecord(): Record<EvidenceLedgerSectionId, number> {
  return Object.fromEntries(allSectionIds().map((sectionId) => [sectionId, 0])) as Record<
    EvidenceLedgerSectionId,
    number
  >;
}

function truncateTextForCompactPlannerLedger(
  value: string | undefined,
  maxTokens: number,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }

  const maxChars = Math.max(80, maxTokens * 4);

  if (value.length <= maxChars) {
    return value;
  }

  const omission = `\n[compact planner ledger truncated ${value.length - maxChars} chars]`;
  const bodyLimit = Math.max(0, maxChars - omission.length);

  return `${value.slice(0, bodyLimit).trimEnd()}${omission}`;
}

function compactEntry(entry: EvidenceLedgerEntry, maxEntryTextTokens: number): EvidenceLedgerEntry {
  return {
    ...entry,
    text: truncateTextForCompactPlannerLedger(entry.text, maxEntryTextTokens),
    value: truncateTextForCompactPlannerLedger(entry.value, Math.max(80, maxEntryTextTokens / 4)),
  };
}

function omittedEntry(section: EvidenceLedgerSection, omittedCount: number): EvidenceLedgerEntry {
  return {
    id: `compact_planner_ledger_omitted:${section.id}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: 0,
    state: "omitted",
    text: `Compact planner ledger omitted ${omittedCount} older entries from ${section.id} to stay within the planner budget.`,
    taint: "none",
  };
}

function renderCompactSection(section: EvidenceLedgerSection, omittedCount: number): string {
  const entries =
    omittedCount <= 0 ? section.entries : [...section.entries, omittedEntry(section, omittedCount)];

  return renderSection({
    ...section,
    entries,
  });
}

function compactSection(input: { section: EvidenceLedgerSection; maxEntryTextTokens: number }): {
  section: EvidenceLedgerSection;
  omittedCount: number;
  estimatedTokens: number;
} {
  const options =
    DEFAULT_COMPACT_SECTION_OPTIONS[
      input.section.id as keyof typeof DEFAULT_COMPACT_SECTION_OPTIONS
    ];
  const maxEntries = options.maxEntries;
  const maxTokens = options.maxTokens;
  const entries = input.section.entries
    .slice(0, maxEntries)
    .map((entry) => compactEntry(entry, input.maxEntryTextTokens));
  let includedEntries: EvidenceLedgerEntry[] = [];
  let omittedCount = Math.max(0, input.section.entries.length - entries.length);

  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index]!;
    const candidateEntries = [...includedEntries, entry];
    const candidateSection = {
      ...input.section,
      entries: candidateEntries,
    };
    const rendered = renderCompactSection(candidateSection, omittedCount);

    if (estimatePromptTokens(rendered) <= maxTokens || includedEntries.length === 0) {
      includedEntries = candidateEntries;
      continue;
    }

    omittedCount += entries.length - index;
    break;
  }

  const section = {
    ...input.section,
    entries: includedEntries,
  };

  return {
    section,
    omittedCount,
    estimatedTokens: estimatePromptTokens(renderCompactSection(section, omittedCount)),
  };
}

function totalCompactPromptTokens(sections: readonly CompactSectionResult[]): number {
  return estimatePromptTokens(
    renderCompactPlannerLedgerPromptSection(renderCompactPlannerLedgerContent(sections)) ?? "",
  );
}

type CompactSectionResult = {
  section: EvidenceLedgerSection;
  omittedCount: number;
};

function trimToTokenTarget(
  sections: CompactSectionResult[],
  targetTokens: number,
): CompactSectionResult[] {
  while (totalCompactPromptTokens(sections) > targetTokens) {
    const trimIndex = [...sections]
      .reverse()
      .findIndex((section) => section.section.entries.length > 0);

    if (trimIndex < 0) {
      break;
    }

    const sectionIndex = sections.length - 1 - trimIndex;
    const section = sections[sectionIndex]!;
    section.section = {
      ...section.section,
      entries: section.section.entries.slice(0, -1),
    };
    section.omittedCount += 1;
  }

  return sections;
}

function renderCompactPlannerLedgerContent(sections: readonly CompactSectionResult[]): string {
  return [
    COMPACT_PLANNER_LEDGER_GUIDANCE,
    ...sections.map((section) => renderCompactSection(section.section, section.omittedCount)),
  ].join("\n\n");
}

function renderCompactPlannerLedgerPromptSection(content: string): string | null {
  return renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
    {
      tag: "borg_compact_planner_ledger",
      content,
    },
  ]);
}

function renderSection(section: EvidenceLedgerSection): string {
  if (section.entries.length === 0) {
    return [`## ${section.label}`, "No entries."].join("\n");
  }

  const sourceTypes = [...new Set(section.entries.map((entry) => entry.source_type))].join(", ");
  const scopes = [...new Set(section.entries.map((entry) => entry.session_scope))].join(", ");

  return [
    `## ${section.label}`,
    `source_types: ${sourceTypes}`,
    `scopes: ${scopes}`,
    ...section.entries.map((entry) => renderEntry(entry)),
  ].join("\n");
}

export function buildCompactPlannerLedgerPrompt(
  ledger: EvidenceLedger,
  options: CompactPlannerLedgerOptions = {},
): CompactPlannerLedgerPrompt {
  const targetTokens = normalizePositiveInteger(
    options.targetTokens,
    DEFAULT_COMPACT_PLANNER_TARGET_TOKENS,
  );
  const hardCapTokens = Math.max(
    targetTokens,
    normalizePositiveInteger(options.hardCapTokens, DEFAULT_COMPACT_PLANNER_HARD_CAP_TOKENS),
  );
  const maxEntryTextTokens = normalizePositiveInteger(
    options.maxEntryTextTokens,
    DEFAULT_COMPACT_ENTRY_TEXT_TOKEN_CAP,
  );
  const sectionsById = new Map(ledger.sections.map((section) => [section.id, section]));
  const compactSections = COMPACT_PLANNER_LEDGER_SECTION_IDS.map((sectionId) => {
    const section = sectionsById.get(sectionId);

    if (section === undefined) {
      return {
        section: {
          id: sectionId,
          label: sectionId,
          entries: [],
        },
        omittedCount: 0,
      };
    }

    const compacted = compactSection({
      section,
      maxEntryTextTokens,
    });

    return {
      section: compacted.section,
      omittedCount: compacted.omittedCount,
    };
  });
  const trimmedSections = trimToTokenTarget(compactSections, targetTokens);
  const hardCappedSections = trimToTokenTarget(trimmedSections, hardCapTokens);
  const content = renderCompactPlannerLedgerContent(hardCappedSections);
  const promptSection = renderCompactPlannerLedgerPromptSection(content);
  const entryCountsBySection = emptySectionCountRecord();
  const omittedEntryCountsBySection = emptySectionCountRecord();
  const estimatedTokensBySection = emptySectionCountRecord();

  for (const section of hardCappedSections) {
    entryCountsBySection[section.section.id] = section.section.entries.length;
    omittedEntryCountsBySection[section.section.id] = section.omittedCount;
    estimatedTokensBySection[section.section.id] = estimatePromptTokens(
      renderCompactSection(section.section, section.omittedCount),
    );
  }

  return {
    promptSection,
    traceSummary: {
      entryCountsBySection,
      omittedEntryCountsBySection,
      estimatedTokensBySection,
      totalEstimatedTokens: estimatePromptTokens(promptSection ?? ""),
      targetTokens,
      hardCapTokens,
    },
  };
}

export function renderCompactPlannerLedger(
  ledger: EvidenceLedger,
  options: CompactPlannerLedgerOptions = {},
): string | null {
  return buildCompactPlannerLedgerPrompt(ledger, options).promptSection;
}

export function renderEvidenceLedger(ledger: EvidenceLedger): string | null {
  const transcriptStatus = ledger.transcriptIncluded
    ? ledger.transcriptCompacted
      ? "current_session_transcript=included compacted=true"
      : "current_session_transcript=included"
    : `current_session_transcript=omitted reason=${ledger.transcriptOmittedReason ?? "unknown"}`;
  const content = [
    "EvidenceLedger: prioritized evidence substrate for the final response.",
    HIERARCHY_GUIDANCE,
    transcriptStatus,
    `estimated_tokens=${ledger.estimatedTokens}`,
    ...ledger.sections.map((section) => renderSection(section)),
  ].join("\n\n");

  return renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
    {
      tag: "borg_evidence_ledger",
      content,
    },
  ]);
}

export function estimateEvidenceLedgerPromptTokens(ledger: EvidenceLedger): number {
  return estimatePromptTokens(renderEvidenceLedger(ledger) ?? "");
}
