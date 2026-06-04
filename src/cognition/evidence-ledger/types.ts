import { z } from "zod";

import type { SharedStateArtifact } from "../../memory/decision-artifacts/types.js";
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
    id: "autobiographical_recall",
    label: "13. Autobiographical Recall",
    optional: true,
  },
] as const satisfies readonly EvidenceLedgerSectionDefinition[];

export type EvidenceLedgerSectionId = (typeof EVIDENCE_LEDGER_SECTION_DEFINITIONS)[number]["id"];

export type EvidenceLedgerSection = {
  id: EvidenceLedgerSectionId;
  label: string;
  entries: EvidenceLedgerEntry[];
};

export type EvidenceLedgerAudienceStanding = {
  crossSessionActivityEntries: EvidenceLedgerEntry[];
  selfDecisionIntrospectionEntries: EvidenceLedgerEntry[];
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
