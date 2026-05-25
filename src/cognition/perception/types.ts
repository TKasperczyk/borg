import type { EntityKind } from "../../memory/commitments/types.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import type { CognitiveMode } from "../types.js";

export type EntityExtractionResult = {
  entities: string[];
  entityMentions: ExtractedEntity[];
  userIdentityNames: string[];
};

export type ExtractedEntity = {
  name: string;
  kind: EntityKind;
};

export type ModeDetectionResult = {
  mode: CognitiveMode;
  isOperational: boolean;
};

export type PerceptionClassifierName =
  | "entity_extractor"
  | "mode_detector"
  | "affective_signal"
  | "temporal_cue";

export type PerceptionClassifierFailure = {
  classifier: PerceptionClassifierName;
  error: unknown;
};

export type PerceptionClassifierFailureObserver = (
  failure: PerceptionClassifierFailure,
) => Promise<void> | void;

export type ParticipantRosterAudienceRole = "speaker" | "active_participant" | "audience";

export type ParticipantRosterMember = {
  entity_id: EntityId;
  display_name: string;
  known_relationships: string[];
  audience_role: ParticipantRosterAudienceRole;
  relationship_source: string | null;
  relationship_sources?: string[];
};

export type ParticipantRosterSubject = {
  entity_id: EntityId;
  display_name: string;
  known_relationships: string[];
  relationship_source: string | null;
  relationship_sources?: string[];
};

export type ParticipantRosterUncertain = {
  entity_id: EntityId | null;
  display_name: string | null;
  known_relationships: string[];
  reason: string;
  relationship_source: string | null;
  relationship_sources?: string[];
};

export type ParticipantRoster = {
  participants: ParticipantRosterMember[];
  non_chat_subjects: ParticipantRosterSubject[];
  unknown_or_uncertain: ParticipantRosterUncertain[];
};

export type ParticipantRosterStreamEvidence = {
  entity_id: EntityId;
  display_name?: string | null;
  known_relationship: string;
  source_stream_entry_id: StreamEntryId;
};
