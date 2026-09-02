import { z } from "zod";

import type { SharedStateArtifact } from "../../memory/shared-state/types.js";
import type { StreamEntryPersistenceClass } from "../../stream/index.js";

export const evidenceLedgerSourceTypeSchema = z.enum([
  "current_user_message",
  "current_session_stream",
  "prior_session_stream",
  "episode",
  "semantic_node",
  "semantic_edge",
  "action_record",
  "relational_slot",
  "commitment",
  "shared_state",
  "image_attachment",
  "assistant_stream",
  "system_metadata",
]);

export type EvidenceLedgerSourceType = z.infer<typeof evidenceLedgerSourceTypeSchema>;

export type EvidenceLedgerSessionScope = "current_session" | "prior_session" | "global";
export type EvidenceLedgerActor = "user" | "assistant" | "system" | "memory";
export type EvidenceLedgerTaint = "none" | "assistant_seeded" | "quarantined" | "contested";
export type EvidenceLedgerActionSalienceClass =
  | "borg_current_turn_action"
  | "borg_memory_tracking_action"
  | "participant_pending_recent"
  | "participant_pending_stale"
  | "group_pending"
  | "completed_recent";

export type EvidenceLedgerEntry = {
  id: string;
  source_type: EvidenceLedgerSourceType;
  session_scope: EvidenceLedgerSessionScope;
  actor: EvidenceLedgerActor;
  trust_rank: number;
  text?: string;
  value?: string;
  state?: string;
  salience_class?: EvidenceLedgerActionSalienceClass;
  state_metadata?: Record<string, unknown>;
  /** Planner-only presentation metadata for the compact planner digest and its captures. */
  planner_metadata?: {
    decision_outcome_ref?: string;
    decision_summary?: string;
    decision_rationale?: string | null;
  };
  taint?: EvidenceLedgerTaint;
  persistence_class?: StreamEntryPersistenceClass;
  via_retrieval?: boolean;
  stream_index?: number;
  citations?: string[];
  citation_type?: "original_image" | "generated_perception_text" | "parent_user_message";
};

type EvidenceLedgerSectionDefinition = {
  id: string;
  label: string;
  optional?: boolean;
};

export const EVIDENCE_LEDGER_SECTION_DEFINITIONS = [
  {
    id: "current_user_message",
    label: "1. Current User Message",
  },
  {
    id: "current_session_transcript",
    label: "2. Current-Session Transcript",
  },
  {
    id: "current_session_attribution_sidebar",
    label: "Current Session Attribution Sidebar",
    optional: true,
  },
  {
    id: "attribution_matrix",
    label: "Attribution Matrix",
    optional: true,
  },
  {
    id: "closure_discourse_state",
    label: "3. Current Closure And Discourse State",
  },
  {
    id: "contradictions_quarantines",
    label: "4. Current-Session Contradictions And Quarantines",
  },
  {
    id: "action_states",
    label: "5. Action States",
  },
  {
    id: "group_channel_memory",
    label: "6. Group/Channel Memory",
  },
  {
    id: "shared_state_recall",
    label: "Cross-Audience Shared State Recall",
    optional: true,
  },
  {
    id: "retrieved_raw_stream_evidence",
    label: "7. Retrieved Raw Stream Evidence",
  },
  {
    id: "retrieved_memory_evidence",
    label: "8. Retrieved Memory Evidence",
  },
  {
    id: "episodes",
    label: "9. Episodes",
  },
  {
    id: "semantic_graph",
    label: "10. Semantic Graph",
  },
  {
    id: "open_questions",
    label: "11. Open Questions",
  },
  {
    id: "prior_session_memory",
    label: "12. Prior-Session Memory",
  },
  {
    id: "recent_lived_experience",
    label: "13. Recent Lived Experience",
    optional: true,
  },
  {
    id: "autobiographical_recall",
    label: "14. Autobiographical Recall",
    optional: true,
  },
] as const satisfies readonly EvidenceLedgerSectionDefinition[];

export type EvidenceLedgerSectionId = (typeof EVIDENCE_LEDGER_SECTION_DEFINITIONS)[number]["id"];

// Every framing figure is a count of some subset of the rows a section was assembled from, and a
// subset figure printed alone has no denominator a reader can recover: a section that assembles ten
// rows of three kinds and counts one of those kinds prints a number that is correct about the kind
// and reads as a count of the section. So the population is part of the type rather than a
// convention -- `rows_assembled` is required, every other key is a named subset of it, and no key
// but that one ever counts the section.
export type EvidenceLedgerSectionFramingCounts = Record<string, number> & {
  rows_assembled: number;
};

export type EvidenceLedgerSectionFraming = {
  text: string;
  counts?: EvidenceLedgerSectionFramingCounts;
};

// Shared by the two sections that render a self-decision label (13 and 14), so the wording
// cannot drift apart and leave one of them reading as if it carried no caveat.
//
// The label is written by summarizeAutonomousDecision() in src/autonomy/scheduler.ts, which
// reads turnResult.response and turnResult.emission and nothing else -- notably not
// turnResult.toolCalls. So the vocabulary is closed over emission kinds, and a turn that made
// an outbound tool call is described by the same label as one that made none. Observed live on
// 2026-08-28: autonomous turn 32a3cb12 (wake fired 00:08:22Z) called tool.outbound.post at
// 00:13:57Z, the delivery came back transport_failed at 00:15:28Z and filed action record
// act_mj8aryyayv8v95r3, and the turn's self-decision row at 00:15:58Z reads "Continued private
// train of thought." Both are true of the same turn; only one of them mentions the reach.
export const SELF_DECISION_LABEL_SCOPE_FRAMING =
  "A self-decision label reports that turn's emission alone, whether it spoke, stayed silent, or continued privately, and reads nothing the turn did through a tool. An autonomous turn that reached outward and had the reach fail in transport therefore still labels as private thought, and the reach and its outcome are recorded only as an action record.";

export type EvidenceLedgerSection = {
  id: EvidenceLedgerSectionId;
  label: string;
  framing?: EvidenceLedgerSectionFraming;
  entries: EvidenceLedgerEntry[];
};

export type EvidenceLedgerAudienceStanding = {
  recentLivedExperienceEntries: EvidenceLedgerEntry[];
  renderRecentLivedExperience: boolean;
  observedEventIntrospectionEntries: EvidenceLedgerEntry[];
  commitmentEntries: EvidenceLedgerEntry[];
  relationalEntries: EvidenceLedgerEntry[];
};

export type EvidenceLedgerTranscriptOmittedReason = "over_budget";

export type EvidenceLedger = {
  sections: EvidenceLedgerSection[];
  audienceStanding?: EvidenceLedgerAudienceStanding;
  sharedState?: SharedStateArtifact | null;
  transcriptIncluded: boolean;
  transcriptCompacted: boolean;
  transcriptOmittedReason?: EvidenceLedgerTranscriptOmittedReason;
  originalTranscriptTokenEstimate: number;
  compactedTranscriptEntryCount: number;
  rawPreservedUserTranscriptEntryCount: number;
  estimatedTokens: number;
  imageAttachments?: EvidenceLedgerImageAttachment[];
};

export type EvidenceLedgerImageAttachment = {
  label: string;
  originFrame?: string;
  attachment_id: string;
  byte_size?: number;
  citation_type: "original_image";
};

export type EvidenceLedgerTraceSummary = {
  entryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  estimatedTokensBySection: Record<EvidenceLedgerSectionId, number>;
  transcriptIncluded: boolean;
  transcriptCompacted: boolean;
  transcriptOmittedReason?: EvidenceLedgerTranscriptOmittedReason;
  originalTranscriptTokenEstimate: number;
  compactedTranscriptTokenEstimate: number;
  compactedEntryCount: number;
  rawPreservedUserEntryCount: number;
  totalEstimatedTokens: number;
};
