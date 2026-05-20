import type { LLMMessage, LLMToolDefinition } from "../../llm/index.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { SHARED_STATE_SYSTEM_PROMPT } from "../prompts/shared-state.js";
import type { SharedStatePromptSummary } from "./summary.js";
import type {
  SharedStateArtifactParticipantContext,
  SharedStateActionCanonicalizationCandidate,
  SharedStateCanonicalizationCandidate,
  SharedStateCanonicalizationCandidates,
  SharedStateCommitmentCanonicalizationCandidate,
} from "./schema.js";

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
  currentUserMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  promptVisibleLedger: string;
  previousArtifactSummary: SharedStatePromptSummary | null;
  canonicalizationCandidates: SharedStateCanonicalizationCandidates;
  allowedSourceStreamEntryIds?: readonly StreamEntryId[];
  offLimitsSourceStreamEntryIds?: readonly StreamEntryId[];
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
        current_user_turn: {
          stream_entry_id: input.currentUserStreamEntryId,
          text: input.currentUserMessage,
        },
        source_trust: {
          citation_eligible_source_stream_entry_id_count:
            input.allowedSourceStreamEntryIds?.length ?? null,
          off_limits_source_stream_entry_ids: input.offLimitsSourceStreamEntryIds ?? [],
        },
        previous_artifact_summary: input.previousArtifactSummary,
        canonicalization_candidates: canonicalizationCandidates,
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
  promptVisibleLedger: string;
  currentUserMessage: string;
  canonicalizationCandidates: SharedStateCanonicalizationCandidates;
}): SharedStateArtifactPromptBudget {
  const system = estimatePromptTokens(SHARED_STATE_SYSTEM_PROMPT);
  const toolSchema = estimatePromptTokens(JSON.stringify(input.tools));
  const previousArtifactSummary = estimatePromptTokens(
    JSON.stringify(input.previousArtifactSummary),
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
    promptVisibleLedger +
    currentUserTurn +
    canonicalizationCandidates;

  return {
    inputTokenEstimate,
    breakdown: {
      system,
      tool_schema: toolSchema,
      previous_artifact_summary: previousArtifactSummary,
      prompt_visible_ledger: promptVisibleLedger,
      current_user_turn: currentUserTurn,
      canonicalization_candidates: canonicalizationCandidates,
      prompt_envelope: Math.max(0, inputTokenEstimate - accounted),
    },
  };
}
