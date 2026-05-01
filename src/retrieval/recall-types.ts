import type {
  CommitmentId,
  EpisodeId,
  OpenQuestionId,
  SemanticEdgeId,
  SemanticNodeId,
  StreamEntryId,
} from "../util/ids.js";

export const RECALL_INTENT_KINDS = [
  "raw_text",
  "known_term",
  "topic",
  "relationship",
  "time",
  "recent",
  "commitment",
  "open_question",
] as const;

export type RecallIntentKind = (typeof RECALL_INTENT_KINDS)[number];

export type RecallIntentSource =
  | "raw-user-message"
  | "llm-expansion"
  | "perception-entities"
  | "audience-aliases"
  | "temporal-cue"
  | "recency";

export type RecallTimeRange = {
  start: number;
  end: number;
};

export type RecallIntent = {
  id: string;
  kind: RecallIntentKind;
  query: string;
  terms: string[];
  timeRange?: RecallTimeRange;
  strictTime?: boolean;
  priority: number;
  source: RecallIntentSource;
};

export type EvidenceSource =
  | "raw_stream"
  | "recent_raw_stream"
  | "episode"
  | "semantic_node"
  | "semantic_edge"
  | "commitment"
  | "open_question"
  | "working_state"
  | "warm_recall";

export type RecallEvidenceHandle =
  | { source: "episode"; episodeId: EpisodeId }
  | { source: "raw_stream"; streamIds: StreamEntryId[]; parentEpisodeId?: EpisodeId }
  | { source: "semantic_node"; nodeId: SemanticNodeId }
  | { source: "semantic_edge"; edgeId: SemanticEdgeId; nodeId?: SemanticNodeId }
  | { source: "commitment"; commitmentId: CommitmentId }
  | { source: "open_question"; openQuestionId: OpenQuestionId };

export type EvidenceProvenance = {
  streamIds?: StreamEntryId[];
  parentEpisodeId?: EpisodeId;
  episodeId?: EpisodeId;
  nodeId?: SemanticNodeId;
  edgeId?: SemanticEdgeId;
  commitmentId?: CommitmentId;
  openQuestionId?: OpenQuestionId;
};

export type EvidenceScoreBreakdown = {
  lexical?: number;
  vector?: number;
  recency?: number;
  salience?: number;
  provenance?: number;
  exactTerm?: number;
};

export type EvidenceItem = {
  id: string;
  source: EvidenceSource;
  text: string;
  provenance?: EvidenceProvenance;
  recallIntentId: string;
  matchedTerms: string[];
  score: number;
  scoreBreakdown: EvidenceScoreBreakdown;
};

export type EvidencePool = {
  intents: RecallIntent[];
  items: EvidenceItem[];
};
