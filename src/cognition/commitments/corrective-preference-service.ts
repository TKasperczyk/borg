import type { LLMClient } from "../../llm/index.js";
import {
  commitmentIdSchema,
  commitmentSchema,
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
  type BorgRole,
  type CommitmentRecord,
  type CommitmentRepository,
  type EntityRepository,
} from "../../memory/commitments/index.js";
import type { IdentityService } from "../../memory/identity/index.js";
import { supersedeCommitment } from "../../memory/lifecycle-ops/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type { WorkingMemory, WorkingMemoryStore } from "../../memory/working/index.js";
import type { StreamEntry } from "../../stream/index.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import type { Clock } from "../../util/clock.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import {
  createCommitmentId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { isCreatorInOperatorContext } from "../authority.js";
import type { ParticipantRoster } from "../perception/index.js";
import type { CurrentTurnUserInputSenderAttribution } from "../turn-input.js";
import { checkRelationshipClaimGrounding } from "../../memory/common/relationship-claim-grounding.js";
import type { RelationshipClaim } from "../../memory/common/relationship-claims.js";
import {
  CorrectivePreferenceExtractor,
  type CorrectivePreferenceCandidate,
  type CorrectivePreferenceRetirementCandidate,
  type ExtractCorrectivePreferenceInput,
} from "./corrective-preference-extractor.js";

const CORRECTIVE_RELATIONAL_SLOT_LIMIT = 32;

export type CorrectivePreferenceTurnServiceOptions = {
  model: string;
  commitmentRepository: Pick<
    CommitmentRepository,
    "get" | "getApplicable" | "revoke" | "supersede"
  >;
  identityService: Pick<IdentityService, "addCommitment">;
  relationalSlotRepository: Pick<RelationalSlotRepository, "list" | "applyNegation">;
  entityRepository?: Pick<EntityRepository, "get" | "getSelf">;
  workingMemoryStore: Pick<WorkingMemoryStore, "load" | "sanitizePendingActionsForRelationalSlot">;
  clock: Clock;
  tracer: TurnTracer;
  // When enabled, propagate degraded extraction and failed mutations upstream.
  strictFailurePropagation?: boolean;
  // Optional transaction boundary for add and supersede atomicity.
  runPersistenceTransaction?: <T>(callback: () => T) => T;
};

export type ExtractCorrectivePreferenceForTurnInput = {
  llmClient: LLMClient;
  turnId: string;
  isUserTurn: boolean;
  userMessage: string;
  persistedUserEntryId?: StreamEntryId;
  sourceUserEntryIds?: readonly StreamEntryId[];
  sourceUserEntries?: readonly StreamEntry[];
  senderAttribution?: readonly CurrentTurnUserInputSenderAttribution[];
  recentHistory: ExtractCorrectivePreferenceInput["recentHistory"];
  audienceEntityId: EntityId | null;
  committedByEntityId?: EntityId | null;
  currentSenderEntityId: EntityId | null;
  currentSenderBorgRole: BorgRole | null;
  sessionAudienceRole: SessionAudienceRole;
  speakerDisplayName?: string | null;
  participantRoster?: ParticipantRoster | null;
  relationshipEvidenceStreamEntries?: readonly Pick<StreamEntry, "id" | "kind">[];
  // Cross-audience scope is gated upstream: `allowed` must already encode the
  // creator-in-operator authority check. Even so the service re-validates the
  // model's chosen target against `candidateAudiences` before honoring it, so a
  // hallucinated or out-of-set id can never redirect a commitment.
  crossAudienceTargeting?: {
    allowed: boolean;
    candidateAudiences: readonly { entity_id: EntityId; label: string }[];
  };
  sessionId: SessionId;
  onHookFailure: (hook: string, error: unknown, details?: Record<string, unknown>) => Promise<void>;
  trackAppliedSlotNegation: (slot: RelationalSlot) => void;
};

export type CorrectivePreferenceTurnResult = {
  commitment: CommitmentRecord | null;
  commitmentSupersession: CorrectivePreferenceSupersessionClaim | null;
  commitmentRetirement: CorrectivePreferenceRetirementClaim | null;
  workingMemory: WorkingMemory;
};

export type CorrectivePreferenceSupersessionClaim = {
  supersededId: CommitmentRecord["id"];
  allowedActiveCommitmentIds: readonly CommitmentRecord["id"][];
};

export type CorrectivePreferenceRetirementClaim = {
  retiredId: CommitmentRecord["id"];
  allowedActiveCommitmentIds: readonly CommitmentRecord["id"][];
  reason: string;
  confidence: number;
};

export function isCognitionRetireEligible(commitment: CommitmentRecord): boolean {
  const enforcementClass = effectiveCommitmentEnforcementClass(commitment);

  // Cardinal-rule structural host boundary: cognition may stand down behavioral
  // boundaries, but host-safety-critical domains remain operator/admin-owned.
  // Use the raw domain for the critical branch: defaulting null critical
  // boundaries to audience_scope would make malformed/legacy host boundaries
  // retirable, so critical + null fails closed.
  return enforcementClass !== "critical" || commitment.critical_domain === "audience_scope";
}

function buildCorrectivePreferenceCommitment(input: {
  candidate: CorrectivePreferenceCandidate;
  restrictedAudience: EntityId | null;
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
    restricted_audience: input.restrictedAudience,
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

function buildCorrectivePreferenceRetirementClaim(input: {
  retirement: CorrectivePreferenceRetirementCandidate | null;
  activeCommitments: readonly CommitmentRecord[];
}): CorrectivePreferenceRetirementClaim | null {
  if (input.retirement === null) {
    return null;
  }

  return {
    retiredId: input.retirement.commitmentId,
    allowedActiveCommitmentIds: input.activeCommitments.map((commitment) => commitment.id),
    reason: input.retirement.reason,
    confidence: input.retirement.confidence,
  };
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

  private traceRetirementRejected(input: {
    turnId?: string;
    sessionId?: SessionId;
    retiredId: CommitmentRecord["id"];
    reason: string;
    error?: unknown;
  }): void {
    if (!this.options.tracer.enabled || input.turnId === undefined) {
      return;
    }

    this.options.tracer.emit("extraction.commitments.rejected", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      retiredId: input.retiredId,
      validationStatus: "rejected",
      reason: input.reason,
      ...(this.options.tracer.includePayloads && input.error !== undefined
        ? { error: input.error instanceof Error ? input.error.message : String(input.error) }
        : {}),
    });
  }

  private traceRetiredViaExtractor(input: {
    turnId?: string;
    sessionId?: SessionId;
    retiredId: CommitmentRecord["id"];
  }): void {
    if (!this.options.tracer.enabled || input.turnId === undefined) {
      return;
    }

    this.options.tracer.emit("extraction.commitments.transitioned", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      retiredId: input.retiredId,
      validationStatus: "accepted",
      reason: "retired_by_corrective_preference",
    });
  }

  private traceCandidateRejectedUngrounded(input: {
    turnId?: string;
    sessionId?: SessionId;
    candidate: CorrectivePreferenceCandidate;
    ungroundedClaims: readonly RelationshipClaim[];
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
      reason: "relationship_claim_ungrounded",
      directive_family: input.candidate.directive_family,
      kind: input.candidate.kind,
      relationship_claim_label_families: [
        ...new Set(input.ungroundedClaims.map((claim) => claim.label_family)),
      ],
      relationship_claims: [...input.candidate.relationship_claims],
      ungrounded_relationship_claims: [...input.ungroundedClaims],
      rejected_relationship_claim_evidence_relational_slot_ids: [
        ...input.rejectedRelationalSlotIds,
      ],
      rejected_relationship_claim_evidence_stream_entry_ids: [...input.rejectedStreamEntryIds],
    });
  }

  private candidateHasGroundedRelationshipClaims(input: {
    candidate: CorrectivePreferenceCandidate;
    turnId?: string;
    sessionId?: SessionId;
    persistedUserEntryId?: StreamEntryId;
    sourceUserEntryIds?: readonly StreamEntryId[];
    participantRoster?: ParticipantRoster | null;
    relationshipEvidenceStreamEntries?: readonly Pick<StreamEntry, "id" | "kind">[];
  }): boolean {
    const allowedStreamEntryIds = new Set<StreamEntryId>();
    const streamEntryKindById = new Map<StreamEntryId, StreamEntry["kind"]>();

    const sourceUserEntryIds =
      input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0
        ? input.persistedUserEntryId === undefined
          ? []
          : [input.persistedUserEntryId]
        : [...input.sourceUserEntryIds];

    for (const streamEntryId of sourceUserEntryIds) {
      allowedStreamEntryIds.add(streamEntryId);
    }

    for (const entry of input.relationshipEvidenceStreamEntries ?? []) {
      allowedStreamEntryIds.add(entry.id);
      streamEntryKindById.set(entry.id, entry.kind);
    }

    const check = checkRelationshipClaimGrounding({
      claims: input.candidate.relationship_claims,
      participantRoster: input.participantRoster ?? null,
      allowedRelationshipEvidenceStreamEntryIds: allowedStreamEntryIds,
      relationshipEvidenceStreamEntryTrust: (streamEntryId) => {
        if (sourceUserEntryIds.some((sourceUserEntryId) => sourceUserEntryId === streamEntryId)) {
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
      ungroundedClaims: check.ungroundedClaims,
      rejectedRelationalSlotIds: check.rejectedRelationalSlotIds,
      rejectedStreamEntryIds: check.rejectedStreamEntryIds,
    });

    return false;
  }

  private isActiveCommitment(record: CommitmentRecord, nowMs: number): boolean {
    return (
      record.revoked_at === null &&
      record.superseded_by === null &&
      record.expired_at === null &&
      (record.expires_at === null || record.expires_at > nowMs)
    );
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

    if (!this.isActiveCommitment(current, nowMs)) {
      return {
        accepted: false,
        reason: "commitment_not_active",
      };
    }

    return { accepted: true };
  }

  private validateRetirementClaim(
    claim: CorrectivePreferenceRetirementClaim,
  ): { accepted: true; expectedVersion: number } | { accepted: false; reason: string } {
    const parsedId = commitmentIdSchema.safeParse(claim.retiredId);

    if (!parsedId.success) {
      return {
        accepted: false,
        reason: "invalid_retirement_claim",
      };
    }

    const allowedIds = new Set(claim.allowedActiveCommitmentIds);

    if (!allowedIds.has(parsedId.data)) {
      return {
        accepted: false,
        reason: "not_in_allowed_active_commitments",
      };
    }

    const current = this.options.commitmentRepository.get(parsedId.data);

    if (current === null) {
      return {
        accepted: false,
        reason: "unknown_commitment_id",
      };
    }

    if (!this.isActiveCommitment(current, this.options.clock.now())) {
      return {
        accepted: false,
        reason: "commitment_not_active",
      };
    }

    if (!isCognitionRetireEligible(current)) {
      return {
        accepted: false,
        reason: "retirement_not_eligible",
      };
    }

    if (current.record_version === undefined) {
      return {
        accepted: false,
        reason: "commitment_version_unavailable",
      };
    }

    return { accepted: true, expectedVersion: current.record_version };
  }

  private traceCrossAudienceScope(input: {
    turnId?: string;
    sessionId?: SessionId;
    validationStatus: "accepted" | "rejected";
    requestedAudienceEntityId: EntityId;
    currentAudienceEntityId: EntityId | null;
    reason: string;
  }): void {
    if (!this.options.tracer.enabled || input.turnId === undefined) {
      return;
    }

    this.options.tracer.emit("corrective_preference.cross_audience_scope", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      validationStatus: input.validationStatus,
      reason: input.reason,
      requested_audience_entity_id: input.requestedAudienceEntityId,
      current_audience_entity_id: input.currentAudienceEntityId,
    });
  }

  private traceCrossAudienceCreatorRuleDeferred(input: {
    turnId?: string;
    sessionId?: SessionId;
    candidate: CorrectivePreferenceCandidate;
    currentAudienceEntityId: EntityId | null;
    currentSenderEntityId: EntityId | null;
  }): void {
    if (!this.options.tracer.enabled || input.turnId === undefined) {
      return;
    }

    this.options.tracer.emit("corrective_preference.cross_audience_creator_deferred", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      validationStatus: "deferred",
      reason: "creator_operator_cross_audience_deferred_to_creator_directive",
      requested_audience_entity_id: input.candidate.applies_to_audience_entity_id,
      current_audience_entity_id: input.currentAudienceEntityId,
      current_sender_entity_id: input.currentSenderEntityId,
      directive_family: input.candidate.directive_family,
      kind: input.candidate.kind,
    });
  }

  // A creator/operator cross-audience rule belongs to the creator-directive
  // band, not commitments, so we suppress it here and let the creator-directive
  // extractor (active under the same isUserTurn + operator-audience + creator
  // condition) own it. This is a SAFE, loss-TOLERANT partition, not a strictly
  // lossless handoff: the deferral fires on any non-null cross-audience target,
  // BEFORE resolveCorrectiveRestrictedAudience would validate/fall back, so a
  // malformed or unlisted target is dropped rather than misfiled as a
  // current-audience commitment. That is deliberate -- routing such a candidate
  // into a commitment fallback would leak malformed operator authority into the
  // commitments band. The deferral trace records the requested target so a
  // dropped candidate stays explainable.
  private shouldDeferCrossAudienceCreatorRule(input: {
    candidate: CorrectivePreferenceCandidate;
    isUserTurn: boolean;
    currentSenderEntityId: EntityId | null;
    currentSenderBorgRole: BorgRole | null;
    sessionAudienceRole: SessionAudienceRole;
  }): boolean {
    return (
      input.isUserTurn &&
      isCreatorInOperatorContext({
        currentSenderBorgRole: input.currentSenderBorgRole,
        sessionAudienceRole: input.sessionAudienceRole,
      }) &&
      input.currentSenderEntityId !== null &&
      input.candidate.applies_to_audience_entity_id !== null
    );
  }

  // Resolve the audience a corrective commitment is scoped to. Default is the
  // current session audience. A different audience is honored ONLY when the
  // turn was authorized to cross-target (input.allowed, set upstream to the
  // creator-in-operator check) AND the model's chosen id is one of the
  // structurally-supplied candidate audiences. This is the security gate: a
  // non-authorized turn never receives candidates, and even an authorized turn
  // cannot redirect a commitment to an id outside its candidate set.
  private resolveCorrectiveRestrictedAudience(input: {
    candidate: CorrectivePreferenceCandidate;
    currentAudienceEntityId: EntityId | null;
    allowed: boolean;
    candidateAudiences: readonly { entity_id: EntityId; label: string }[];
    turnId?: string;
    sessionId?: SessionId;
  }): EntityId | null {
    const requested = input.candidate.applies_to_audience_entity_id;

    if (requested === null) {
      return input.currentAudienceEntityId;
    }

    const inCandidateSet = input.candidateAudiences.some(
      (audience) => audience.entity_id === requested,
    );

    if (input.allowed && inCandidateSet) {
      this.traceCrossAudienceScope({
        turnId: input.turnId,
        sessionId: input.sessionId,
        validationStatus: "accepted",
        requestedAudienceEntityId: requested,
        currentAudienceEntityId: input.currentAudienceEntityId,
        reason: "cross_audience_scope_applied",
      });
      return requested;
    }

    this.traceCrossAudienceScope({
      turnId: input.turnId,
      sessionId: input.sessionId,
      validationStatus: "rejected",
      requestedAudienceEntityId: requested,
      currentAudienceEntityId: input.currentAudienceEntityId,
      reason: input.allowed ? "target_not_in_candidate_set" : "cross_audience_not_authorized",
    });
    return input.currentAudienceEntityId;
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
      throwOnDegraded: this.options.strictFailurePropagation === true,
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
    const crossAudienceAllowed = input.crossAudienceTargeting?.allowed === true;
    const crossAudienceCandidates = crossAudienceAllowed
      ? (input.crossAudienceTargeting?.candidateAudiences ?? [])
      : [];
    const correctiveExtraction = await correctivePreferenceExtractor.extractWithSlotNegations({
      userMessage: input.userMessage,
      selfIdentity: this.options.entityRepository?.getSelf() ?? null,
      currentUserStreamEntryId: input.persistedUserEntryId ?? null,
      currentUserStreamEntryIds: input.sourceUserEntryIds,
      currentMessageEntries: input.sourceUserEntries,
      currentMessageSenderAttribution: input.senderAttribution,
      recentHistory: input.recentHistory,
      audienceEntityId: input.audienceEntityId,
      speakerEntityId: input.committedByEntityId ?? null,
      speakerDisplayName: input.speakerDisplayName ?? null,
      senderDisplayNameById: (entityId) =>
        this.options.entityRepository?.get(entityId)?.canonical_name ?? null,
      participantRoster: input.participantRoster ?? null,
      crossAudienceTargets: crossAudienceCandidates,
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
        restricted_audience: commitment.restricted_audience,
        made_to_entity: commitment.made_to_entity,
      })),
      relationalSlots: this.relationalSlotsForCorrectionExtractor(),
    });
    const correctiveCandidate = correctiveExtraction.preference;
    let acceptedCorrectiveCandidate: CorrectivePreferenceCandidate | null = null;

    if (
      correctiveCandidate !== null &&
      this.candidateHasGroundedRelationshipClaims({
        candidate: correctiveCandidate,
        turnId: input.turnId,
        sessionId: input.sessionId,
        persistedUserEntryId: input.persistedUserEntryId,
        participantRoster: input.participantRoster ?? null,
        sourceUserEntryIds: input.sourceUserEntryIds,
        relationshipEvidenceStreamEntries: input.relationshipEvidenceStreamEntries,
      })
    ) {
      const shouldDeferToCreatorDirective = this.shouldDeferCrossAudienceCreatorRule({
        candidate: correctiveCandidate,
        isUserTurn: input.isUserTurn,
        currentSenderEntityId: input.currentSenderEntityId,
        currentSenderBorgRole: input.currentSenderBorgRole,
        sessionAudienceRole: input.sessionAudienceRole,
      });

      if (shouldDeferToCreatorDirective) {
        this.traceCrossAudienceCreatorRuleDeferred({
          turnId: input.turnId,
          sessionId: input.sessionId,
          candidate: correctiveCandidate,
          currentAudienceEntityId: input.audienceEntityId,
          currentSenderEntityId: input.currentSenderEntityId,
        });
      } else {
        acceptedCorrectiveCandidate = correctiveCandidate;
        correctiveCommitment = buildCorrectivePreferenceCommitment({
          candidate: correctiveCandidate,
          restrictedAudience: this.resolveCorrectiveRestrictedAudience({
            candidate: correctiveCandidate,
            currentAudienceEntityId: input.audienceEntityId,
            allowed: crossAudienceAllowed,
            candidateAudiences: crossAudienceCandidates,
            turnId: input.turnId,
            sessionId: input.sessionId,
          }),
          committedByEntityId: input.committedByEntityId ?? null,
          sourceStreamEntryIds:
            input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0
              ? input.persistedUserEntryId === undefined
                ? undefined
                : [input.persistedUserEntryId]
              : [...input.sourceUserEntryIds],
          nowMs: this.options.clock.now(),
        });
      }
    }

    const sourceUserEntryIds =
      input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0
        ? input.persistedUserEntryId === undefined
          ? []
          : [input.persistedUserEntryId]
        : [...input.sourceUserEntryIds];

    if (sourceUserEntryIds.length > 0) {
      for (const negation of correctiveExtraction.slot_negations) {
        try {
          const result = this.options.relationalSlotRepository.applyNegation({
            subject_entity_id: negation.subject_entity_id,
            slot_key: negation.slot_key,
            rejected_value: negation.rejected_value,
            source_stream_entry_ids: sourceUserEntryIds,
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

          if (this.options.strictFailurePropagation === true) {
            throw error;
          }
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
      commitmentRetirement: buildCorrectivePreferenceRetirementClaim({
        retirement: correctiveExtraction.retirement,
        activeCommitments: activeCommitmentsForExtractor,
      }),
      workingMemory: this.options.workingMemoryStore.load(input.sessionId),
    };
  }

  async persistCommitment(input: {
    commitment: CommitmentRecord | null;
    supersession?: CorrectivePreferenceSupersessionClaim | null;
    retirement?: CorrectivePreferenceRetirementClaim | null;
    turnId?: string;
    sessionId?: SessionId;
    onHookFailure: (
      hook: string,
      error: unknown,
      details?: Record<string, unknown>,
    ) => Promise<void>;
  }): Promise<void> {
    const commitment = input.commitment;
    const retirement = input.retirement ?? null;

    if (commitment === null && retirement === null) {
      return;
    }

    if (commitment !== null) {
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

      const addCommitment = () =>
        this.options.identityService.addCommitment({
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

      if (this.options.strictFailurePropagation === true) {
        let failureHook = "corrective_preference_commitment_persist";

        try {
          const runTransaction = this.options.runPersistenceTransaction;

          if (runTransaction === undefined) {
            throw new Error(
              "Strict corrective-preference persistence requires a transaction boundary",
            );
          }

          const persisted = runTransaction(() => {
            const added = addCommitment();

            if (supersession === null || validation?.accepted !== true) {
              return { added, superseded: false };
            }

            failureHook = "corrective_preference_commitment_supersede";
            const superseded = supersedeCommitment({
              commitmentId: supersession.supersededId,
              replacementCommitmentId: added.id,
              repository: this.options.commitmentRepository,
            });

            if (superseded.status !== "success") {
              const failure =
                superseded.status === "conflict"
                  ? superseded.error
                  : new Error("Corrective-preference supersession target was not found");
              this.traceSupersessionRejected({
                turnId: input.turnId,
                sessionId: input.sessionId,
                supersededId: supersession.supersededId,
                newId: added.id,
                reason:
                  superseded.status === "conflict" ? "supersede_failed" : "unknown_commitment_id",
                error: failure,
              });
              throw failure;
            }

            return { added, superseded: true };
          });

          if (persisted.superseded && supersession !== null) {
            this.traceSupersededViaExtractor({
              turnId: input.turnId,
              sessionId: input.sessionId,
              supersededId: supersession.supersededId,
              newId: persisted.added.id,
            });
          }
        } catch (error) {
          await input.onHookFailure(failureHook, error, {
            commitmentId: commitment.id,
          });
          throw error;
        }
      } else {
        let persisted: CommitmentRecord | null = null;

        try {
          persisted = addCommitment();
        } catch (error) {
          await input.onHookFailure("corrective_preference_commitment_persist", error, {
            commitmentId: commitment.id,
          });
        }

        if (persisted !== null && supersession !== null && validation?.accepted === true) {
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
            } else if (superseded.status === "conflict") {
              this.traceSupersessionRejected({
                turnId: input.turnId,
                sessionId: input.sessionId,
                supersededId: supersession.supersededId,
                newId: persisted.id,
                reason: "supersede_failed",
                error: superseded.error,
              });
            } else {
              this.traceSupersededViaExtractor({
                turnId: input.turnId,
                sessionId: input.sessionId,
                supersededId: supersession.supersededId,
                newId: persisted.id,
              });
            }
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
      }
    }

    if (retirement === null) {
      return;
    }

    const retirementValidation = this.validateRetirementClaim(retirement);

    if (!retirementValidation.accepted) {
      this.traceRetirementRejected({
        turnId: input.turnId,
        sessionId: input.sessionId,
        retiredId: retirement.retiredId,
        reason: retirementValidation.reason,
      });
      return;
    }

    try {
      const retired = this.options.commitmentRepository.revoke(
        retirement.retiredId,
        retirement.reason,
        {
          kind: "online",
          process: "corrective-preference-extractor",
        },
        undefined,
        { expectedVersion: retirementValidation.expectedVersion },
      );

      if (retired === null) {
        this.traceRetirementRejected({
          turnId: input.turnId,
          sessionId: input.sessionId,
          retiredId: retirement.retiredId,
          reason: "commitment_version_conflict",
        });

        if (this.options.strictFailurePropagation === true) {
          throw new Error("Corrective-preference retirement version conflict");
        }

        return;
      }

      this.traceRetiredViaExtractor({
        turnId: input.turnId,
        sessionId: input.sessionId,
        retiredId: retirement.retiredId,
      });
    } catch (error) {
      if (error instanceof IdentityCasMismatchError) {
        this.traceRetirementRejected({
          turnId: input.turnId,
          sessionId: input.sessionId,
          retiredId: retirement.retiredId,
          reason: "commitment_version_conflict",
          error,
        });

        if (this.options.strictFailurePropagation === true) {
          await input.onHookFailure("corrective_preference_commitment_retire", error, {
            commitmentId: retirement.retiredId,
          });
          throw error;
        }

        return;
      }

      this.traceRetirementRejected({
        turnId: input.turnId,
        sessionId: input.sessionId,
        retiredId: retirement.retiredId,
        reason: "revoke_failed",
        error,
      });
      await input.onHookFailure("corrective_preference_commitment_retire", error, {
        commitmentId: retirement.retiredId,
      });

      if (this.options.strictFailurePropagation === true) {
        throw error;
      }
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
