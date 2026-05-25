import type { LLMMessage, LLMToolDefinition } from "../../llm/index.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { SHARED_STATE_SYSTEM_PROMPT } from "../prompts/shared-state.js";
import { renderParticipantRoster, type ParticipantRoster } from "../perception/index.js";
import type { ExistingStateKeyRegistryEntry, SharedStatePromptSummary } from "./summary.js";
import type {
  SharedStateArtifactParticipantContext,
  SharedStateActionCanonicalizationCandidate,
  SharedStateCanonicalizationCandidate,
  SharedStateCanonicalizationCandidates,
  SharedStateCommitmentCanonicalizationCandidate,
  SharedStateRelationalSlotContext,
} from "./types.js";

export function buildCanonicalizationCandidatePromptPayload(
  candidates: SharedStateCanonicalizationCandidates,
): {
  active_goals: readonly SharedStateCanonicalizationCandidate[];
  active_commitments: readonly SharedStateCommitmentCanonicalizationCandidate[];
  active_actions: readonly SharedStateActionCanonicalizationCandidate[];
  open_questions: readonly SharedStateCanonicalizationCandidate[];
} {
  return {
    active_goals: candidates.goals ?? [],
    active_commitments: candidates.commitments ?? [],
    active_actions: candidates.actions ?? [],
    open_questions: candidates.openQuestions ?? [],
  };
}

export function buildSharedStateArtifactMessages(input: {
  audienceEntityId: EntityId;
  selfEntityId: EntityId;
  speakerEntityId: EntityId | null;
  participants: readonly SharedStateArtifactParticipantContext[];
  participantRoster?: ParticipantRoster | null;
  currentUserMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  promptVisibleLedger: string;
  existingStateKeyRegistry: readonly ExistingStateKeyRegistryEntry[];
  previousArtifactSummary: SharedStatePromptSummary | null;
  canonicalizationCandidates: SharedStateCanonicalizationCandidates;
  relationalSlotsContext?: readonly SharedStateRelationalSlotContext[];
  allowedSourceStreamEntryIds?: readonly StreamEntryId[];
  offLimitsSourceStreamEntryIds?: readonly StreamEntryId[];
  additionalPromptSections?: readonly string[];
}): LLMMessage[] {
  const canonicalizationCandidates = buildCanonicalizationCandidatePromptPayload(
    input.canonicalizationCandidates,
  );

  return [
    {
      role: "user",
      content: JSON.stringify({
        audience_entity_id: input.audienceEntityId,
        self_entity_id: input.selfEntityId,
        speaker_entity_id: input.speakerEntityId,
        participant_entities: input.participants.map((participant) => ({
          entity_id: participant.entityId,
          display_name: participant.displayName ?? null,
        })),
        participant_roster: renderParticipantRoster(input.participantRoster),
        current_user_turn: {
          stream_entry_id: input.currentUserStreamEntryId,
          text: input.currentUserMessage,
        },
        source_trust: {
          citation_eligible_source_stream_entry_id_count:
            input.allowedSourceStreamEntryIds?.length ?? null,
          off_limits_source_stream_entry_ids: input.offLimitsSourceStreamEntryIds ?? [],
        },
        ...(input.additionalPromptSections === undefined ||
        input.additionalPromptSections.length === 0
          ? {}
          : { additional_prompt_sections: input.additionalPromptSections }),
        existing_state_key_registry: input.existingStateKeyRegistry,
        previous_artifact_summary: input.previousArtifactSummary,
        canonicalization_candidates: canonicalizationCandidates,
        relational_slots_context: (input.relationalSlotsContext ?? []).map((slot) => ({
          id: slot.id,
          subject_entity_id: slot.subject_entity_id,
          slot_key: slot.slot_key,
          value: slot.value,
          state: slot.state,
          evidence_stream_entry_ids: slot.evidence_stream_entry_ids,
          contradicted_by_stream_entry_ids: slot.contradicted_by_stream_entry_ids,
          alternate_values: slot.alternate_values.map((alternate) => ({
            value: alternate.value,
            evidence_stream_entry_ids: alternate.evidence_stream_entry_ids,
          })),
        })),
        prompt_visible_ledger: input.promptVisibleLedger,
      }),
    },
  ];
}

export type SharedStateArtifactPromptBudget = {
  inputTokenEstimate: number;
  breakdown: Record<string, number>;
};

export function estimateSharedStateArtifactPromptBudget(input: {
  messages: readonly LLMMessage[];
  tools: readonly LLMToolDefinition[];
  previousArtifactSummary: SharedStatePromptSummary | null;
  existingStateKeyRegistry: readonly ExistingStateKeyRegistryEntry[];
  promptVisibleLedger: string;
  currentUserMessage: string;
  canonicalizationCandidates: SharedStateCanonicalizationCandidates;
}): SharedStateArtifactPromptBudget {
  const system = estimatePromptTokens(SHARED_STATE_SYSTEM_PROMPT);
  const toolSchema = estimatePromptTokens(JSON.stringify(input.tools));
  const previousArtifactSummary = estimatePromptTokens(
    JSON.stringify(input.previousArtifactSummary),
  );
  const existingStateKeyRegistry = estimatePromptTokens(
    JSON.stringify(input.existingStateKeyRegistry),
  );
  const promptVisibleLedger = estimatePromptTokens(input.promptVisibleLedger);
  const currentUserTurn = estimatePromptTokens(input.currentUserMessage);
  const canonicalizationCandidates = estimatePromptTokens(
    JSON.stringify(buildCanonicalizationCandidatePromptPayload(input.canonicalizationCandidates)),
  );
  const inputTokenEstimate =
    system +
    toolSchema +
    input.messages.reduce((sum, message) => sum + estimatePromptTokens(message.content), 0);
  const accounted =
    system +
    toolSchema +
    previousArtifactSummary +
    existingStateKeyRegistry +
    promptVisibleLedger +
    currentUserTurn +
    canonicalizationCandidates;

  return {
    inputTokenEstimate,
    breakdown: {
      system,
      tool_schema: toolSchema,
      previous_artifact_summary: previousArtifactSummary,
      existing_state_key_registry: existingStateKeyRegistry,
      prompt_visible_ledger: promptVisibleLedger,
      current_user_turn: currentUserTurn,
      canonicalization_candidates: canonicalizationCandidates,
      prompt_envelope: Math.max(0, inputTokenEstimate - accounted),
    },
  };
}
