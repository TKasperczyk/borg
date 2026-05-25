import type {
  CommitmentId,
  EpisodeId,
  OpenQuestionId,
  SemanticEdgeId,
  SemanticNodeId,
  StreamEntryId,
  AttachmentId,
  ImagePerceptionId,
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
  | "image_perception"
  | "warm_recall";

export type RecallEvidenceHandle =
  | { source: "episode"; episodeId: EpisodeId }
  | { source: "raw_stream"; streamIds: StreamEntryId[]; parentEpisodeId?: EpisodeId }
  | { source: "semantic_node"; nodeId: SemanticNodeId }
  | { source: "semantic_edge"; edgeId: SemanticEdgeId; nodeId?: SemanticNodeId }
  | { source: "commitment"; commitmentId: CommitmentId }
  | { source: "open_question"; openQuestionId: OpenQuestionId }
  | { source: "image_perception"; perceptionId: ImagePerceptionId; attachmentId: AttachmentId };

export type EvidenceProvenance = {
  streamIds?: StreamEntryId[];
  parentEpisodeId?: EpisodeId;
  episodeId?: EpisodeId;
  nodeId?: SemanticNodeId;
  edgeId?: SemanticEdgeId;
  commitmentId?: CommitmentId;
  openQuestionId?: OpenQuestionId;
  imagePerceptionId?: ImagePerceptionId;
  attachmentId?: AttachmentId;
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
  source_episode_ids?: EpisodeId[];
  partial_source_visibility?: boolean;
  source_visibility_fraction?: number;
  imageAttachmentId?: AttachmentId;
  imageLabel?: string;
  citationType?: "original_image" | "generated_perception_text" | "parent_user_message";
  imageUnavailableReason?: "budget" | "inactive";
};

export type EvidencePool = {
  intents: RecallIntent[];
  items: EvidenceItem[];
};
