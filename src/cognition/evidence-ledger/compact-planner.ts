import { estimatePromptTokens } from "../../util/token-estimate.js";
import { coercePositiveIntegerOrFallback } from "../../util/math.js";
import type { SharedStateArtifact } from "../../memory/shared-state/index.js";
import {
  renderSharedStateArtifact,
  summarizeSharedStateArtifactRender,
  type SharedStateRenderOptions,
} from "../shared-state/render.js";
import type { SharedStateKindCounts } from "../shared-state/selection.js";
import { UNTRUSTED_DATA_PREAMBLE } from "../prompts/base-identity.js";
import {
  PROMPT_SURFACES,
  renderPromptSurface,
  type PromptSurfaceRenderContext,
} from "../prompts/prompt-surface-registry.js";
import { renderTaggedPromptBlock } from "../deliberation/prompt/sections.js";
import { emptySectionCountRecord } from "./budget.js";
import { renderSection } from "./section-rendering.js";
import { ledgerAnnotationGuidance } from "./record-annotations.js";
import {
  capFullLedgerSection,
  materializeFullLedgerSectionState,
  fitLedgerAccounting,
  ledgerBudgetOutcome,
  type FullLedgerSectionState,
} from "./compaction.js";
import { renderGlobalAccounting } from "./omission-accounting.js";
import type {
  EvidenceLedger,
  EvidenceLedgerSectionId,
  EvidenceLedgerGlobalAccounting,
  EvidenceLedgerOmissionCountsBySection,
  EvidenceLedgerBudgetOutcome,
} from "./types.js";

const COMPACT_PLANNER_LEDGER_SECTION_IDS = [
  "current_user_message",
  "closure_discourse_state",
  "contradictions_quarantines",
  "action_states",
  "group_channel_memory",
] as const satisfies readonly EvidenceLedgerSectionId[];

const DEFAULT_COMPACT_PLANNER_TARGET_TOKENS = 8_000;
const DEFAULT_COMPACT_PLANNER_HARD_CAP_TOKENS = 15_000;
const DEFAULT_COMPACT_ENTRY_TEXT_TOKEN_CAP = 600;

const DEFAULT_COMPACT_SECTION_OPTIONS = {
  current_user_message: {
    maxEntries: 1,
    maxTokens: 1_200,
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
  sharedState?: SharedStateRenderOptions;
};

export type CompactPlannerLedgerTraceSummary = {
  entryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  omittedEntryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  estimatedTokensBySection: Record<EvidenceLedgerSectionId, number>;
  sharedStateEntryCount: number;
  sharedStateRenderedTokens: number;
  sharedStateRenderedByKind: SharedStateKindCounts;
  totalEstimatedTokens: number;
  targetTokens: number;
  hardCapTokens: number;
  accountingOmittedFromPage?: EvidenceLedgerOmissionCountsBySection;
  budgetOutcome: EvidenceLedgerBudgetOutcome;
};

export type CompactPlannerLedgerPrompt = {
  promptSection: string | null;
  traceSummary: CompactPlannerLedgerTraceSummary;
};

const COMPACT_PLANNER_LEDGER_GUIDANCE = [
  "CompactPlannerLedger: decision-relevant evidence slice for the S2 planner.",
  "I use these entries to check current-turn constraints before planning verification steps.",
  "Dialogue messages carry the conversational transcript; this compact ledger carries locked state, constraints, participant context, quarantines, and action threads.",
  "Quarantined/contested/assistant-seeded values are not facts.",
  "In goal state_metadata, counterparty_entity_id is the participant the responsibility runs toward, not an owner or an audience.",
].join("\n");

export function truncateTextForCompactPlannerLedger(
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

function renderCompactPlannerLedgerContent(
  sections: readonly FullLedgerSectionState[],
  sharedState: SharedStateArtifact | null | undefined,
  sharedStateOptions: SharedStateRenderOptions | undefined,
  accounting?: EvidenceLedgerGlobalAccounting,
): string {
  return [
    renderSharedStateArtifact(sharedState, sharedStateOptions),
    COMPACT_PLANNER_LEDGER_GUIDANCE,
    ledgerAnnotationGuidance(sections.map(materializeFullLedgerSectionState)),
    renderGlobalAccounting(accounting),
    ...sections.map((section) => renderSection(materializeFullLedgerSectionState(section))),
  ]
    .filter((part): part is string => part !== null && part.length > 0)
    .join("\n\n");
}

function renderCompactPlannerLedgerPromptSection(content: string): string | null {
  const renderContext: PromptSurfaceRenderContext = {
    renderBlock: (id) =>
      id === "borg_compact_planner_ledger"
        ? renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
            {
              tag: "borg_compact_planner_ledger",
              content,
            },
          ])
        : null,
  };

  return renderPromptSurface(PROMPT_SURFACES.compactPlannerFraming, renderContext);
}

export function buildCompactPlannerLedgerPrompt(
  ledger: EvidenceLedger,
  options: CompactPlannerLedgerOptions = {},
): CompactPlannerLedgerPrompt {
  const targetTokens = coercePositiveIntegerOrFallback(
    options.targetTokens,
    DEFAULT_COMPACT_PLANNER_TARGET_TOKENS,
  );
  const hardCapTokens = coercePositiveIntegerOrFallback(
    options.hardCapTokens,
    DEFAULT_COMPACT_PLANNER_HARD_CAP_TOKENS,
  );
  const maxEntryTextTokens = coercePositiveIntegerOrFallback(
    options.maxEntryTextTokens,
    DEFAULT_COMPACT_ENTRY_TEXT_TOKEN_CAP,
  );
  const sectionsById = new Map(ledger.sections.map((section) => [section.id, section]));
  const compactSections = COMPACT_PLANNER_LEDGER_SECTION_IDS.map((sectionId) => {
    const section = sectionsById.get(sectionId) ?? { id: sectionId, label: sectionId, entries: [] };
    return capFullLedgerSection({
      section: {
        ...section,
        entries: section.entries.map((entry) => ({
          ...entry,
          text: truncateTextForCompactPlannerLedger(entry.text, maxEntryTextTokens),
          value: truncateTextForCompactPlannerLedger(
            entry.value,
            Math.max(80, maxEntryTextTokens / 4),
          ),
        })),
      },
      maxEntryTextTokens,
      options: DEFAULT_COMPACT_SECTION_OPTIONS[sectionId],
      carriedAccounting: ledger.compaction_accounting?.folded_sections[sectionId],
    });
  });
  const fitted = fitLedgerAccounting({
    states: compactSections,
    targetTokens,
    hardCapTokens,
    sectionOrder: [...COMPACT_PLANNER_LEDGER_SECTION_IDS].reverse(),
    measure: (accounting) =>
      estimatePromptTokens(
        renderCompactPlannerLedgerPromptSection(
          renderCompactPlannerLedgerContent(
            compactSections,
            ledger.sharedState,
            options.sharedState,
            accounting,
          ),
        ) ?? "",
      ),
  });
  const content = renderCompactPlannerLedgerContent(
    compactSections,
    ledger.sharedState,
    options.sharedState,
    fitted.accounting,
  );
  const promptSection = renderCompactPlannerLedgerPromptSection(content);
  const entryCountsBySection = emptySectionCountRecord();
  const omittedEntryCountsBySection = emptySectionCountRecord();
  const estimatedTokensBySection = emptySectionCountRecord();
  const sharedStateSummary = summarizeSharedStateArtifactRender(
    ledger.sharedState,
    options.sharedState,
  );

  for (const section of compactSections) {
    entryCountsBySection[section.section.id] =
      materializeFullLedgerSectionState(section).entries.length;
    omittedEntryCountsBySection[section.section.id] = section.omittedCount;
    estimatedTokensBySection[section.section.id] = estimatePromptTokens(
      renderSection(materializeFullLedgerSectionState(section)),
    );
  }

  return {
    promptSection,
    traceSummary: {
      entryCountsBySection,
      omittedEntryCountsBySection,
      accountingOmittedFromPage: fitted.accountingOmittedFromPage,
      budgetOutcome: ledgerBudgetOutcome(
        compactSections,
        estimatePromptTokens(promptSection ?? ""),
        hardCapTokens,
      ),
      estimatedTokensBySection,
      sharedStateEntryCount: sharedStateSummary.renderedEntryCount,
      sharedStateRenderedTokens: sharedStateSummary.estimatedTokens,
      sharedStateRenderedByKind: sharedStateSummary.renderedByKind,
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
