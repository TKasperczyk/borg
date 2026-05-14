import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  renderDecisionStateArtifact,
  type DecisionArtifactRenderOptions,
} from "../decision-artifact/render.js";
import { UNTRUSTED_DATA_PREAMBLE } from "../deliberation/constants.js";
import { renderTaggedPromptBlock } from "../deliberation/prompt/sections.js";
import { renderSection } from "./section-rendering.js";
import type { EvidenceLedger } from "./types.js";

const HIERARCHY_GUIDANCE = [
  "Current-session transcript is authoritative for what happened in this conversation.",
  "Prior-session memory must be attributed or hedged.",
  "Episodes and semantic graph are summaries; use source handles when making exact claims.",
  "Quarantined/contested/assistant-seeded values are not facts.",
].join("\n");

export function renderEvidenceLedger(
  ledger: EvidenceLedger,
  options: { decisionArtifact?: DecisionArtifactRenderOptions } = {},
): string | null {
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
    renderDecisionStateArtifact(ledger.decisionArtifact, options.decisionArtifact),
    ...ledger.sections.map((section) => renderSection(section)),
  ]
    .filter((part): part is string => part !== null)
    .join("\n\n");

  return renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
    {
      tag: "borg_evidence_ledger",
      content,
    },
  ]);
}

export function estimateEvidenceLedgerPromptTokens(
  ledger: EvidenceLedger,
  options: { decisionArtifact?: DecisionArtifactRenderOptions } = {},
): number {
  return estimatePromptTokens(renderEvidenceLedger(ledger, options) ?? "");
}
