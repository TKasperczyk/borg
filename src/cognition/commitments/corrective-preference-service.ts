import type { LLMClient } from "../../llm/index.js";
import {
  commitmentSchema,
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
  type CommitmentRecord,
  type CommitmentRepository,
} from "../../memory/commitments/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import { supersedeCommitment } from "../../memory/lifecycle-ops/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type { WorkingMemory, WorkingMemoryStore } from "../../memory/working/index.js";
import type { StreamEntry } from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import {
  createCommitmentId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { ParticipantRoster } from "../perception/index.js";
import { checkRelationshipLabelGrounding } from "../memory-write-relationship-gate.js";
import {
  CorrectivePreferenceExtractor,
  type CorrectivePreferenceCandidate,
  type ExtractCorrectivePreferenceInput,
} from "./corrective-preference-extractor.js";

const CORRECTIVE_RELATIONAL_SLOT_LIMIT = 32;

export type CorrectivePreferenceTurnServiceOptions = {
  model: string;
  commitmentRepository: Pick<CommitmentRepository, "get" | "getApplicable" | "supersede">;
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
  committedByEntityId?: EntityId | null;
  speakerDisplayName?: string | null;
  participantRoster?: ParticipantRoster | null;
  relationshipEvidenceStreamEntries?: readonly Pick<StreamEntry, "id" | "kind">[];
  sessionId: SessionId;
  onHookFailure: (hook: string, error: unknown, details?: Record<string, unknown>) => Promise<void>;
  trackAppliedSlotNegation: (slot: RelationalSlot) => void;
};

export type CorrectivePreferenceTurnResult = {
  commitment: CommitmentRecord | null;
  commitmentSupersession: CorrectivePreferenceSupersessionClaim | null;
  workingMemory: WorkingMemory;
};

export type CorrectivePreferenceSupersessionClaim = {
  supersededId: CommitmentRecord["id"];
  allowedActiveCommitmentIds: readonly CommitmentRecord["id"][];
};

function buildCorrectivePreferenceCommitment(input: {
  candidate: CorrectivePreferenceCandidate;
  audienceEntityId: EntityId | null;
  committedByEntityId: EntityId | null;
  sourceStreamEntryIds?: CommitmentRecord["source_stream_entry_ids"];
  nowMs: number;
}): CommitmentRecord {
  return commitmentSchema.parse({
    id: createCommitmentId(),
    type: input.candidate.type,
    kind: input.candidate.kind,
    enforcement_class: input.candidate.enforcement_class,
    critical_domain: input.candidate.critical_domain,
    directive_family: input.candidate.directive_family,
    closure_pressure_relevance: input.candidate.closure_pressure_relevance,
    directive: input.candidate.directive,
    priority: input.candidate.priority,
    made_to_entity: null,
    restricted_audience: input.audienceEntityId,
    about_entity: null,
    committed_by_entity_id: input.committedByEntityId,
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

  private traceSupersessionRejected(input: {
    turnId?: string;
    sessionId?: SessionId;
    supersededId: CommitmentRecord["id"];
    newId?: CommitmentRecord["id"];
    reason: string;
    error?: unknown;
  }): void {
    if (!this.options.tracer.enabled || input.turnId === undefined) {
      return;
    }

    this.options.tracer.emit("extraction.commitments.rejected", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      supersededId: input.supersededId,
      ...(input.newId === undefined ? {} : { newId: input.newId }),
      validationStatus: "rejected",
      reason: input.reason,
      ...(this.options.tracer.includePayloads && input.error !== undefined
        ? { error: input.error instanceof Error ? input.error.message : String(input.error) }
        : {}),
    });
  }

  private traceSupersededViaExtractor(input: {
    turnId?: string;
    sessionId?: SessionId;
    supersededId: CommitmentRecord["id"];
    newId: CommitmentRecord["id"];
  }): void {
    if (!this.options.tracer.enabled || input.turnId === undefined) {
      return;
    }

    this.options.tracer.emit("extraction.commitments.transitioned", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      supersededId: input.supersededId,
      newId: input.newId,
      validationStatus: "accepted",
    });
  }

  private traceCandidateRejectedUngrounded(input: {
    turnId?: string;
    sessionId?: SessionId;
    candidate: CorrectivePreferenceCandidate;
    protectedLabels: readonly string[];
    rejectedRelationalSlotIds: readonly string[];
    rejectedStreamEntryIds: readonly { id: string; reason: string }[];
  }): void {
    if (!this.options.tracer.enabled || input.turnId === undefined) {
      return;
    }

    this.options.tracer.emit("corrective_preference.candidate_rejected_ungrounded", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      validationStatus: "rejected",
      reason: "relationship_label_ungrounded",
      directive_family: input.candidate.directive_family,
      kind: input.candidate.kind,
      protected_relationship_labels: [...input.protectedLabels],
      relationship_evidence_relational_slot_ids: [
        ...input.candidate.relationship_evidence_relational_slot_ids,
      ],
      relationship_evidence_stream_entry_ids: [
        ...input.candidate.relationship_evidence_stream_entry_ids,
      ],
      rejected_relationship_evidence_relational_slot_ids: [...input.rejectedRelationalSlotIds],
      rejected_relationship_evidence_stream_entry_ids: [...input.rejectedStreamEntryIds],
    });
  }

  private candidateHasGroundedRelationshipLabels(input: {
    candidate: CorrectivePreferenceCandidate;
    turnId?: string;
    sessionId?: SessionId;
    persistedUserEntryId?: StreamEntryId;
    participantRoster?: ParticipantRoster | null;
    relationshipEvidenceStreamEntries?: readonly Pick<StreamEntry, "id" | "kind">[];
  }): boolean {
    const allowedStreamEntryIds = new Set<StreamEntryId>();
    const streamEntryKindById = new Map<StreamEntryId, StreamEntry["kind"]>();

    if (input.persistedUserEntryId !== undefined) {
      allowedStreamEntryIds.add(input.persistedUserEntryId);
    }

    for (const entry of input.relationshipEvidenceStreamEntries ?? []) {
      allowedStreamEntryIds.add(entry.id);
      streamEntryKindById.set(entry.id, entry.kind);
    }

    const check = checkRelationshipLabelGrounding({
      text: input.candidate.directive,
      participantRoster: input.participantRoster ?? null,
      relationshipEvidenceRelationalSlotIds:
        input.candidate.relationship_evidence_relational_slot_ids,
      relationshipEvidenceStreamEntryIds: input.candidate.relationship_evidence_stream_entry_ids,
      allowedRelationshipEvidenceStreamEntryIds: allowedStreamEntryIds,
      relationshipEvidenceStreamEntryTrust: (streamEntryId) => {
        if (streamEntryId === input.persistedUserEntryId) {
          return { allowed: true };
        }

        const kind = streamEntryKindById.get(streamEntryId);

        if (kind === undefined) {
          return { allowed: false, reason: "missing" };
        }

        return kind === "user_msg" || kind === "user_image_attachment"
          ? { allowed: true }
          : { allowed: false, reason: "not_user_msg" };
      },
    });

    if (check.grounded) {
      return true;
    }

    this.traceCandidateRejectedUngrounded({
      turnId: input.turnId,
      sessionId: input.sessionId,
      candidate: input.candidate,
      protectedLabels: check.protectedLabels,
      rejectedRelationalSlotIds: check.rejectedRelationalSlotIds,
      rejectedStreamEntryIds: check.rejectedStreamEntryIds,
    });

    return false;
  }

  private validateSupersessionClaim(input: {
    claim: CorrectivePreferenceSupersessionClaim;
    newId: CommitmentRecord["id"];
  }): { accepted: true } | { accepted: false; reason: string } {
    const allowedIds = new Set(input.claim.allowedActiveCommitmentIds);

    if (!allowedIds.has(input.claim.supersededId)) {
      return {
        accepted: false,
        reason: "not_in_allowed_active_commitments",
      };
    }

    if (input.claim.supersededId === input.newId) {
      return {
        accepted: false,
        reason: "self_supersession",
      };
    }

    const current = this.options.commitmentRepository.get(input.claim.supersededId);

    if (current === null) {
      return {
        accepted: false,
        reason: "unknown_commitment_id",
      };
    }

    const nowMs = this.options.clock.now();
    const active =
      current.revoked_at === null &&
      current.superseded_by === null &&
      current.expired_at === null &&
      (current.expires_at === null || current.expires_at > nowMs);

    if (!active) {
      return {
        accepted: false,
        reason: "commitment_not_active",
      };
    }

    return { accepted: true };
  }

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
      sessionId: input.sessionId,
      onDegraded: (reason, error, metadata) => {
        if (!this.options.tracer.enabled) {
          return;
        }

        this.options.tracer.emit("extraction.commitments.degraded", {
          turnId: input.turnId,
          ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
          label: "corrective_preference_extractor",
          reason,
          stopReason: metadata?.stopReason ?? null,
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
      speakerEntityId: input.committedByEntityId ?? null,
      speakerDisplayName: input.speakerDisplayName ?? null,
      participantRoster: input.participantRoster ?? null,
      activeCommitments: activeCommitmentsForExtractor.map((commitment) => ({
        id: commitment.id,
        type: commitment.type,
        kind: commitment.kind,
        enforcement_class: effectiveCommitmentEnforcementClass(commitment),
        critical_domain: effectiveCommitmentCriticalDomain(commitment),
        directive_family: commitment.directive_family,
        closure_pressure_relevance: commitment.closure_pressure_relevance,
        directive: commitment.directive,
        priority: commitment.priority,
      })),
      relationalSlots: this.relationalSlotsForCorrectionExtractor(),
    });
    const correctiveCandidate = correctiveExtraction.preference;
    let acceptedCorrectiveCandidate: CorrectivePreferenceCandidate | null = null;

    if (
      correctiveCandidate !== null &&
      this.candidateHasGroundedRelationshipLabels({
        candidate: correctiveCandidate,
        turnId: input.turnId,
        sessionId: input.sessionId,
        persistedUserEntryId: input.persistedUserEntryId,
        participantRoster: input.participantRoster ?? null,
        relationshipEvidenceStreamEntries: input.relationshipEvidenceStreamEntries,
      })
    ) {
      acceptedCorrectiveCandidate = correctiveCandidate;
      correctiveCommitment = buildCorrectivePreferenceCommitment({
        candidate: correctiveCandidate,
        audienceEntityId: input.audienceEntityId,
        committedByEntityId: input.committedByEntityId ?? null,
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
      commitmentSupersession:
        acceptedCorrectiveCandidate?.supersedes_commitment_id === undefined ||
        acceptedCorrectiveCandidate.supersedes_commitment_id === null
          ? null
          : {
              supersededId: acceptedCorrectiveCandidate.supersedes_commitment_id,
              allowedActiveCommitmentIds: activeCommitmentsForExtractor.map(
                (commitment) => commitment.id,
              ),
            },
      workingMemory: this.options.workingMemoryStore.load(input.sessionId),
    };
  }

  async persistCommitment(input: {
    commitment: CommitmentRecord | null;
    supersession?: CorrectivePreferenceSupersessionClaim | null;
    turnId?: string;
    sessionId?: SessionId;
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

    const supersession = input.supersession ?? null;
    const validation =
      supersession === null
        ? null
        : this.validateSupersessionClaim({
            claim: supersession,
            newId: commitment.id,
          });

    if (supersession !== null && validation !== null && !validation.accepted) {
      this.traceSupersessionRejected({
        turnId: input.turnId,
        sessionId: input.sessionId,
        supersededId: supersession.supersededId,
        reason: validation.reason,
      });
    }

    let persisted: CommitmentRecord;

    try {
      persisted = this.options.identityService.addCommitment({
        id: commitment.id,
        type: commitment.type,
        kind: commitment.kind,
        enforcementClass: effectiveCommitmentEnforcementClass(commitment),
        criticalDomain: effectiveCommitmentCriticalDomain(commitment),
        directiveFamily: commitment.directive_family,
        closurePressureRelevance: commitment.closure_pressure_relevance,
        directive: commitment.directive,
        priority: commitment.priority,
        madeToEntity: commitment.made_to_entity,
        restrictedAudience: commitment.restricted_audience,
        aboutEntity: commitment.about_entity,
        committedByEntityId: commitment.committed_by_entity_id,
        provenance: commitment.provenance,
        sourceStreamEntryIds: commitment.source_stream_entry_ids,
        createdAt: commitment.created_at,
        expiresAt: commitment.expires_at,
        ...(validation?.accepted === true ? { skipDirectiveFamilyMerge: true } : {}),
      });
    } catch (error) {
      await input.onHookFailure("corrective_preference_commitment_persist", error, {
        commitmentId: commitment.id,
      });
      return;
    }

    if (supersession === null || validation?.accepted !== true) {
      return;
    }

    try {
      const superseded = supersedeCommitment({
        commitmentId: supersession.supersededId,
        replacementCommitmentId: persisted.id,
        repository: this.options.commitmentRepository,
      });

      if (superseded.status === "no_op") {
        this.traceSupersessionRejected({
          turnId: input.turnId,
          sessionId: input.sessionId,
          supersededId: supersession.supersededId,
          newId: persisted.id,
          reason: "unknown_commitment_id",
        });
        return;
      }

      if (superseded.status === "conflict") {
        this.traceSupersessionRejected({
          turnId: input.turnId,
          sessionId: input.sessionId,
          supersededId: supersession.supersededId,
          newId: persisted.id,
          reason: "supersede_failed",
          error: superseded.error,
        });
        return;
      }

      this.traceSupersededViaExtractor({
        turnId: input.turnId,
        sessionId: input.sessionId,
        supersededId: supersession.supersededId,
        newId: persisted.id,
      });
    } catch (error) {
      this.traceSupersessionRejected({
        turnId: input.turnId,
        sessionId: input.sessionId,
        supersededId: supersession.supersededId,
        newId: persisted.id,
        reason: "supersede_failed",
        error,
      });
    }
  }

  private relationalSlotsForCorrectionExtractor() {
    return this.options.relationalSlotRepository
      .list({ limit: CORRECTIVE_RELATIONAL_SLOT_LIMIT })
      .map((slot) => ({
        id: slot.id,
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
