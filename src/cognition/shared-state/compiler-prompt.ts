import type { LLMMessage, LLMToolDefinition } from "../../llm/index.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import {
  SHARED_STATE_SYSTEM_PROMPT,
  type SharedStateCompilePass,
} from "../prompts/shared-state.js";
import { renderParticipantRoster, type ParticipantRoster } from "../perception/index.js";
import {
  memoryDisclosurePayloadFields,
  relationalSlotMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import type { ExistingStateKeyRegistryEntry, SharedStatePromptSummary } from "./summary.js";
import type {
  SharedStateArtifactAudienceContext,
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
  currentAudience?: SharedStateArtifactAudienceContext | null;
  selfEntityId: EntityId;
  speakerEntityId: EntityId | null;
  participants: readonly SharedStateArtifactParticipantContext[];
  participantRoster?: ParticipantRoster | null;
  currentUserMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  currentUserTurn?: { streamEntryId: StreamEntryId; text: string } | null;
  compilePass?: SharedStateCompilePass;
  assistantResponse?: { streamEntryId: StreamEntryId; text: string } | null;
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
  const currentUserTurn =
    input.currentUserTurn === undefined
      ? {
          streamEntryId: input.currentUserStreamEntryId,
          text: input.currentUserMessage,
        }
      : input.currentUserTurn;
  const currentAudienceParticipant = input.participants.find(
    (participant) => participant.entityId === input.audienceEntityId,
  );

  return [
    {
      role: "user",
      content: JSON.stringify({
        compile_pass: input.compilePass ?? "pre_answer",
        audience_entity_id: input.audienceEntityId,
        current_audience: {
          entity_id: input.audienceEntityId,
          display_name:
            input.currentAudience?.displayName ?? currentAudienceParticipant?.displayName ?? null,
          kind: input.currentAudience?.kind ?? null,
        },
        self_entity_id: input.selfEntityId,
        speaker_entity_id: input.speakerEntityId,
        participant_entities: input.participants.map((participant) => ({
          entity_id: participant.entityId,
          display_name: participant.displayName ?? null,
        })),
        participant_roster: renderParticipantRoster(input.participantRoster),
        current_user_turn:
          currentUserTurn === null
            ? null
            : {
                stream_entry_id: currentUserTurn.streamEntryId,
                text: currentUserTurn.text,
              },
        assistant_response:
          input.assistantResponse === undefined || input.assistantResponse === null
            ? null
            : {
                stream_entry_id: input.assistantResponse.streamEntryId,
                text: input.assistantResponse.text,
              },
        source_trust: {
          citation_eligible_source_stream_entry_id_count:
            input.allowedSourceStreamEntryIds?.length ?? null,
          // The eligible set is enforced as an allowlist, so name it as one. Rendering only the
          // count next to an explicit off-limits list reads as a denylist and leaves the ids that
          // are actually citable to be inferred from the surrounding ledger.
          citation_eligible_source_stream_entry_ids: input.allowedSourceStreamEntryIds ?? null,
          off_limits_source_stream_entry_ids: input.offLimitsSourceStreamEntryIds ?? [],
        },
        ...(input.additionalPromptSections === undefined ||
        input.additionalPromptSections.length === 0
          ? {}
          : { additional_prompt_sections: input.additionalPromptSections }),
        existing_state_key_registry: input.existingStateKeyRegistry,
        previous_artifact_summary: input.previousArtifactSummary,
        canonicalization_candidates: canonicalizationCandidates,
        relational_slots_context: (input.relationalSlotsContext ?? []).map((slot) => {
          const disclosureFields = memoryDisclosurePayloadFields(
            relationalSlotMemoryDisclosureLabel(slot),
          );

          return {
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
              disclosure: disclosureFields.disclosure,
              disclosure_label: disclosureFields.disclosure_label,
            })),
            disclosure: disclosureFields.disclosure,
            disclosure_label: disclosureFields.disclosure_label,
          };
        }),
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
  systemPrompt?: string;
  messages: readonly LLMMessage[];
  tools: readonly LLMToolDefinition[];
  previousArtifactSummary: SharedStatePromptSummary | null;
  existingStateKeyRegistry: readonly ExistingStateKeyRegistryEntry[];
  promptVisibleLedger: string;
  currentUserMessage: string;
  assistantResponse?: { streamEntryId: StreamEntryId; text: string } | null;
  canonicalizationCandidates: SharedStateCanonicalizationCandidates;
}): SharedStateArtifactPromptBudget {
  const system = estimatePromptTokens(input.systemPrompt ?? SHARED_STATE_SYSTEM_PROMPT);
  const toolSchema = estimatePromptTokens(JSON.stringify(input.tools));
  const previousArtifactSummary = estimatePromptTokens(
    JSON.stringify(input.previousArtifactSummary),
  );
  const existingStateKeyRegistry = estimatePromptTokens(
    JSON.stringify(input.existingStateKeyRegistry),
  );
  const promptVisibleLedger = estimatePromptTokens(input.promptVisibleLedger);
  const currentUserTurn = estimatePromptTokens(input.currentUserMessage);
  const assistantResponse = estimatePromptTokens(JSON.stringify(input.assistantResponse ?? null));
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
    assistantResponse +
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
      assistant_response: assistantResponse,
      canonicalization_candidates: canonicalizationCandidates,
      prompt_envelope: Math.max(0, inputTokenEstimate - accounted),
    },
  };
}
