import { z } from "zod";

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
  "assistant_stream",
  "system_metadata",
]);

export type EvidenceLedgerSourceType = z.infer<typeof evidenceLedgerSourceTypeSchema>;

export type EvidenceLedgerSessionScope = "current_session" | "prior_session" | "global";
export type EvidenceLedgerActor = "user" | "assistant" | "system" | "memory";
export type EvidenceLedgerTaint = "none" | "assistant_seeded" | "quarantined" | "contested";

export type EvidenceLedgerEntry = {
  id: string;
  source_type: EvidenceLedgerSourceType;
  session_scope: EvidenceLedgerSessionScope;
  actor: EvidenceLedgerActor;
  trust_rank: number;
  text?: string;
  value?: string;
  state?: string;
  state_metadata?: Record<string, unknown>;
  taint?: EvidenceLedgerTaint;
  persistence_class?: StreamEntryPersistenceClass;
  via_retrieval?: boolean;
  stream_index?: number;
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
    id: "commitments_and_constraints",
    label: "3. Active Commitments And Discourse Constraints",
  },
  {
    id: "closure_discourse_state",
    label: "4. Current Closure And Discourse State",
  },
  {
    id: "contradictions_quarantines",
    label: "5. Current-Session Contradictions And Quarantines",
  },
  {
    id: "action_states",
    label: "6. Action States",
  },
  {
    id: "relational_slots",
    label: "7. Relational And Profile Slots",
  },
  {
    id: "retrieved_raw_stream_evidence",
    label: "8. Retrieved Raw Stream Evidence",
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
] as const;

export type EvidenceLedgerSectionId = (typeof EVIDENCE_LEDGER_SECTION_DEFINITIONS)[number]["id"];

export type EvidenceLedgerSection = {
  id: EvidenceLedgerSectionId;
  label: string;
  entries: EvidenceLedgerEntry[];
};

export type EvidenceLedgerTranscriptOmittedReason = "over_budget";

export type EvidenceLedger = {
  sections: EvidenceLedgerSection[];
  transcriptIncluded: boolean;
  transcriptCompacted: boolean;
  transcriptOmittedReason?: EvidenceLedgerTranscriptOmittedReason;
  estimatedTokens: number;
};

export type EvidenceLedgerTraceSummary = {
  entryCountsBySection: Record<EvidenceLedgerSectionId, number>;
  estimatedTokensBySection: Record<EvidenceLedgerSectionId, number>;
  transcriptIncluded: boolean;
  transcriptCompacted: boolean;
  transcriptOmittedReason?: EvidenceLedgerTranscriptOmittedReason;
  totalEstimatedTokens: number;
};
