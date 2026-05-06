import type { LLMClient } from "../../llm/index.js";
import {
  commitmentSchema,
  type CommitmentRecord,
  type CommitmentRepository,
} from "../../memory/commitments/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type { WorkingMemory, WorkingMemoryStore } from "../../memory/working/index.js";
import type { Clock } from "../../util/clock.js";
import {
  createCommitmentId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import {
  CorrectivePreferenceExtractor,
  type CorrectivePreferenceCandidate,
  type ExtractCorrectivePreferenceInput,
} from "./corrective-preference-extractor.js";

const CORRECTIVE_RELATIONAL_SLOT_LIMIT = 32;

export type CorrectivePreferenceTurnServiceOptions = {
  model: string;
  commitmentRepository: Pick<CommitmentRepository, "getApplicable">;
  identityService: Pick<IdentityService, "addCommitment">;
  relationalSlotRepository: Pick<RelationalSlotRepository, "list" | "applyNegation">;
  workingMemoryStore: Pick<WorkingMemoryStore, "load" | "sanitizePendingActionsForRelationalSlot">;
  clock: Clock;
  tracer: TurnTracer;
};

export type ExtractCorrectivePreferenceForTurnInput = {
  llmClient: LLMClient;
  turnId: string;
  userMessage: string;
  persistedUserEntryId?: StreamEntryId;
  recentHistory: ExtractCorrectivePreferenceInput["recentHistory"];
  audienceEntityId: EntityId | null;
  sessionId: SessionId;
  onHookFailure: (hook: string, error: unknown, details?: Record<string, unknown>) => Promise<void>;
  trackAppliedSlotNegation: (slot: RelationalSlot) => void;
};

export type CorrectivePreferenceTurnResult = {
  commitment: CommitmentRecord | null;
  workingMemory: WorkingMemory;
};

function buildCorrectivePreferenceCommitment(input: {
  candidate: CorrectivePreferenceCandidate;
  audienceEntityId: EntityId | null;
  sourceStreamEntryIds?: CommitmentRecord["source_stream_entry_ids"];
  nowMs: number;
}): CommitmentRecord {
  return commitmentSchema.parse({
    id: createCommitmentId(),
    type: input.candidate.type,
    directive_family: input.candidate.directive_family,
    closure_pressure_relevance: input.candidate.closure_pressure_relevance,
    directive: input.candidate.directive,
    priority: input.candidate.priority,
    made_to_entity: null,
    restricted_audience: input.audienceEntityId,
    about_entity: null,
    provenance: {
      kind: "online",
      process: "corrective-preference-extractor",
    },
    ...(input.sourceStreamEntryIds === undefined || input.sourceStreamEntryIds.length === 0
      ? {}
      : { source_stream_entry_ids: input.sourceStreamEntryIds }),
    created_at: input.nowMs,
    expires_at: null,
    expired_at: null,
    revoked_at: null,
    revoked_reason: null,
    revoke_provenance: null,
    superseded_by: null,
    last_reinforced_at: input.nowMs,
  });
}

export function appendCommitmentIfMissing(
  commitments: readonly CommitmentRecord[],
  commitment: CommitmentRecord | null,
): CommitmentRecord[] {
  if (commitment === null) {
    return [...commitments];
  }

  if (commitments.some((existing) => existing.id === commitment.id)) {
    return [...commitments];
  }

  return [...commitments, commitment].sort(
    (left, right) => right.priority - left.priority || left.created_at - right.created_at,
  );
}

export class CorrectivePreferenceTurnService {
  constructor(private readonly options: CorrectivePreferenceTurnServiceOptions) {}

  async extractAndApply(
    input: ExtractCorrectivePreferenceForTurnInput,
  ): Promise<CorrectivePreferenceTurnResult> {
    let correctiveCommitment: CommitmentRecord | null = null;
    const activeCommitmentsForExtractor = this.options.commitmentRepository.getApplicable({
      audience: input.audienceEntityId,
      nowMs: this.options.clock.now(),
    });
    const correctivePreferenceExtractor = new CorrectivePreferenceExtractor({
      llmClient: input.llmClient,
      model: this.options.model,
      tracer: this.options.tracer,
      turnId: input.turnId,
      onDegraded: (reason, error) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("commitment_extractor_degraded", {
          turnId: input.turnId,
          reason,
          ...(this.options.tracer.includePayloads && error !== undefined
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      },
    });
    const correctiveExtraction = await correctivePreferenceExtractor.extractWithSlotNegations({
      userMessage: input.userMessage,
      currentUserStreamEntryId: input.persistedUserEntryId ?? null,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
      activeCommitments: activeCommitmentsForExtractor.map((commitment) => ({
        id: commitment.id,
        type: commitment.type,
        directive_family: commitment.directive_family,
        closure_pressure_relevance: commitment.closure_pressure_relevance,
        directive: commitment.directive,
        priority: commitment.priority,
      })),
      relationalSlots: this.relationalSlotsForCorrectionExtractor(),
    });
    const correctiveCandidate = correctiveExtraction.preference;

    if (correctiveCandidate !== null) {
      correctiveCommitment = buildCorrectivePreferenceCommitment({
        candidate: correctiveCandidate,
        audienceEntityId: input.audienceEntityId,
        sourceStreamEntryIds:
          input.persistedUserEntryId === undefined ? undefined : [input.persistedUserEntryId],
        nowMs: this.options.clock.now(),
      });
    }

    if (input.persistedUserEntryId !== undefined) {
      for (const negation of correctiveExtraction.slot_negations) {
        try {
          const result = this.options.relationalSlotRepository.applyNegation({
            subject_entity_id: negation.subject_entity_id,
            slot_key: negation.slot_key,
            rejected_value: negation.rejected_value,
            source_stream_entry_ids: [input.persistedUserEntryId],
          });

          if (result?.previous !== null && result?.previous !== undefined) {
            input.trackAppliedSlotNegation(result.previous);
          }

          if (result?.constrained === true) {
            this.options.workingMemoryStore.sanitizePendingActionsForRelationalSlot({
              sessionId: input.sessionId,
              values: result.values_to_neutralize,
              neutralPhrase: result.neutral_phrase,
            });
          }
        } catch (error) {
          await input.onHookFailure("relational_slot_negation", error, {
            slotKey: negation.slot_key,
          });
        }
      }
    }

    return {
      commitment: correctiveCommitment,
      workingMemory: this.options.workingMemoryStore.load(input.sessionId),
    };
  }

  async persistCommitment(input: {
    commitment: CommitmentRecord | null;
    onHookFailure: (
      hook: string,
      error: unknown,
      details?: Record<string, unknown>,
    ) => Promise<void>;
  }): Promise<void> {
    const commitment = input.commitment;

    if (commitment === null) {
      return;
    }

    try {
      this.options.identityService.addCommitment({
        id: commitment.id,
        type: commitment.type,
        directiveFamily: commitment.directive_family,
        closurePressureRelevance: commitment.closure_pressure_relevance,
        directive: commitment.directive,
        priority: commitment.priority,
        madeToEntity: commitment.made_to_entity,
        restrictedAudience: commitment.restricted_audience,
        aboutEntity: commitment.about_entity,
        provenance: commitment.provenance,
        sourceStreamEntryIds: commitment.source_stream_entry_ids,
        createdAt: commitment.created_at,
        expiresAt: commitment.expires_at,
      });
    } catch (error) {
      await input.onHookFailure("corrective_preference_commitment_persist", error, {
        commitmentId: commitment.id,
      });
    }
  }

  private relationalSlotsForCorrectionExtractor() {
    return this.options.relationalSlotRepository
      .list({ limit: CORRECTIVE_RELATIONAL_SLOT_LIMIT })
      .map((slot) => ({
        subject_entity_id: slot.subject_entity_id,
        slot_key: slot.slot_key,
        value: slot.value,
        state: slot.state,
        alternate_values: slot.alternate_values.map((alternate) => ({
          value: alternate.value,
        })),
      }));
  }
}
