import type { LLMClient } from "../../llm/index.js";
import type { BorgRole, EntityRepository } from "../../memory/commitments/index.js";
import type {
  CreatorDirective,
  CreatorDirectiveQueueInput,
  CreatorDirectiveRepository,
  DisclosurePolicy,
} from "../../memory/creator-directives/index.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import type { ParticipantRoster } from "../perception/index.js";
import type { RecencyMessage } from "../recency/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import {
  CreatorDirectiveExtractor,
  type CreatorDirectiveCandidate,
  type KnownCreatorDirectiveEntity,
} from "./extractor.js";

export type CreatorDirectiveTurnServiceOptions = {
  model: string;
  creatorDirectiveRepository: Pick<CreatorDirectiveRepository, "queue">;
  entityRepository: Pick<EntityRepository, "findByName" | "get">;
  tracer: TurnTracer;
};

export type ExtractCreatorDirectivesForTurnInput = {
  llmClient: LLMClient;
  turnId: string;
  isUserTurn: boolean;
  userMessage: string;
  audienceEntityId: EntityId | null;
  currentSenderEntityId: EntityId | null;
  currentSenderBorgRole: BorgRole | null;
  currentSenderDisplayName?: string | null;
  sourceSessionId: SessionId;
  persistedUserEntryId?: StreamEntryId;
  recentHistory: readonly RecencyMessage[];
  sessionId?: SessionId;
  sessionAudienceRole: SessionAudienceRole;
  participantRoster?: ParticipantRoster | null;
  knownEntities?: readonly KnownCreatorDirectiveEntity[];
};

type CandidateResolution =
  | {
      accepted: true;
      subjectEntityId: EntityId | null;
      allowedEntityIds: EntityId[];
      excludedEntityIds: EntityId[];
    }
  | { accepted: false; reason: string };

function uniqueEntityIds(values: readonly EntityId[]): EntityId[] {
  return dedupePreservingOrder(values);
}

function entityExists(
  entityRepository: Pick<EntityRepository, "get">,
  entityId: EntityId,
): boolean {
  return entityRepository.get(entityId) !== null;
}

function entityIdsFromRoster(
  roster: ParticipantRoster | null | undefined,
): KnownCreatorDirectiveEntity[] {
  if (roster === null || roster === undefined) {
    return [];
  }

  return [
    ...roster.participants.map((entity) => ({
      entity_id: entity.entity_id,
      display_name: entity.display_name,
      role: entity.audience_role,
    })),
    ...roster.non_chat_subjects.map((entity) => ({
      entity_id: entity.entity_id,
      display_name: entity.display_name,
      role: "subject",
    })),
  ];
}

export class CreatorDirectiveTurnService {
  constructor(private readonly options: CreatorDirectiveTurnServiceOptions) {}

  private trace(
    event:
      | "creator_directive_candidate_extracted"
      | "creator_directive_persisted"
      | "creator_directive_candidate_rejected",
    input: {
      turnId: string;
      sessionId?: SessionId;
      [key: string]: JsonValue | undefined;
    },
  ): void {
    if (!this.options.tracer.enabled) {
      return;
    }

    const { turnId, sessionId, ...rest } = input;

    this.options.tracer.emit(event, {
      turnId,
      ...(sessionId !== undefined ? { session_id: sessionId } : {}),
      ...rest,
    });
  }

  private reject(input: {
    turnId: string;
    sessionId?: SessionId;
    candidateIndex?: number;
    candidate?: CreatorDirectiveCandidate;
    reason: string;
    error?: unknown;
  }): void {
    this.trace("creator_directive_candidate_rejected", {
      turnId: input.turnId,
      sessionId: input.sessionId,
      ...(input.candidateIndex === undefined ? {} : { candidate_index: input.candidateIndex }),
      ...(input.candidate === undefined
        ? {}
        : {
            kind: input.candidate.kind,
            subject_kind: input.candidate.subject_kind,
            content_scope: input.candidate.disclosure_policy.content_scope,
            mention_policy: input.candidate.disclosure_policy.mention_policy,
          }),
      validationStatus: "rejected",
      reason: input.reason,
      ...(this.options.tracer.includePayloads && input.error !== undefined
        ? { error: input.error instanceof Error ? input.error.message : String(input.error) }
        : {}),
    });
  }

  private resolveExistingEntityLabel(label: string): EntityId | null {
    return this.options.entityRepository.findByName(label);
  }

  private resolveEntityIdList(input: {
    ids: readonly EntityId[];
    labels: readonly string[];
    reason: string;
  }): { accepted: true; ids: EntityId[] } | { accepted: false; reason: string } {
    const resolved: EntityId[] = [];

    for (const entityId of input.ids) {
      if (!entityExists(this.options.entityRepository, entityId)) {
        return { accepted: false, reason: input.reason };
      }

      resolved.push(entityId);
    }

    for (const label of input.labels) {
      const entityId = this.resolveExistingEntityLabel(label);

      if (entityId === null) {
        return { accepted: false, reason: input.reason };
      }

      resolved.push(entityId);
    }

    return { accepted: true, ids: uniqueEntityIds(resolved) };
  }

  private resolveCandidate(candidate: CreatorDirectiveCandidate): CandidateResolution {
    let subjectEntityId = candidate.subject_entity_id;

    if (subjectEntityId !== null && !entityExists(this.options.entityRepository, subjectEntityId)) {
      return { accepted: false, reason: "unknown_subject_entity" };
    }

    if (candidate.subject_kind === "entity" && subjectEntityId === null) {
      if (candidate.subject_label === null) {
        return { accepted: false, reason: "unknown_subject_entity" };
      }

      subjectEntityId = this.resolveExistingEntityLabel(candidate.subject_label);

      if (subjectEntityId === null) {
        return { accepted: false, reason: "unknown_subject_entity" };
      }
    }

    const allowed = this.resolveEntityIdList({
      ids: candidate.disclosure_policy.allowed_entity_ids,
      labels: candidate.disclosure_policy.allowed_entity_labels,
      reason: "unknown_allowed_entity",
    });

    if (!allowed.accepted) {
      return allowed;
    }

    const excluded = this.resolveEntityIdList({
      ids: candidate.disclosure_policy.excluded_entity_ids,
      labels: candidate.disclosure_policy.excluded_entity_labels,
      reason: "unknown_excluded_entity",
    });

    if (!excluded.accepted) {
      return excluded;
    }

    return {
      accepted: true,
      subjectEntityId,
      allowedEntityIds: allowed.ids,
      excludedEntityIds: excluded.ids,
    };
  }

  private buildDisclosurePolicy(input: {
    candidate: CreatorDirectiveCandidate;
    allowedEntityIds: readonly EntityId[];
    excludedEntityIds: readonly EntityId[];
  }): DisclosurePolicy {
    return {
      content_scope: input.candidate.disclosure_policy.content_scope,
      allowed_entity_ids: uniqueEntityIds(input.allowedEntityIds),
      excluded_entity_ids: uniqueEntityIds(input.excludedEntityIds),
      subject_may_know: input.candidate.disclosure_policy.subject_may_know,
      mention_policy: input.candidate.disclosure_policy.mention_policy,
      denied_audience_behavior: input.candidate.disclosure_policy.denied_audience_behavior,
      boundary_prompt: input.candidate.disclosure_policy.boundary_prompt,
      topic_tags: [...input.candidate.disclosure_policy.topic_tags],
    };
  }

  private buildQueueInput(input: {
    candidate: CreatorDirectiveCandidate;
    createdByEntityId: EntityId;
    sourceSessionId: SessionId;
    sourceStreamEntryId: StreamEntryId;
    subjectEntityId: EntityId | null;
    allowedEntityIds: readonly EntityId[];
    excludedEntityIds: readonly EntityId[];
  }): CreatorDirectiveQueueInput {
    return {
      kind: input.candidate.kind,
      createdByEntityId: input.createdByEntityId,
      sourceSessionId: input.sourceSessionId,
      authorizationStreamEntryIds: [input.sourceStreamEntryId],
      contentSourceStreamEntryIds: [input.sourceStreamEntryId],
      subjectKind: input.candidate.subject_kind,
      subjectEntityId: input.subjectEntityId,
      canonicalFact: input.candidate.canonical_fact,
      operationalDirective: input.candidate.operational_directive,
      disclosurePolicy: this.buildDisclosurePolicy({
        candidate: input.candidate,
        allowedEntityIds: input.allowedEntityIds,
        excludedEntityIds: input.excludedEntityIds,
      }),
      priority: input.candidate.priority,
    };
  }

  async extractAndPersist(
    input: ExtractCreatorDirectivesForTurnInput,
  ): Promise<CreatorDirective[]> {
    if (
      !input.isUserTurn ||
      input.sessionAudienceRole !== "operator" ||
      input.currentSenderBorgRole !== "creator" ||
      input.currentSenderEntityId === null
    ) {
      return [];
    }

    if (input.persistedUserEntryId === undefined) {
      this.reject({
        turnId: input.turnId,
        sessionId: input.sessionId,
        reason: "missing_persisted_user_entry",
      });
      return [];
    }

    const knownEntities = [
      ...entityIdsFromRoster(input.participantRoster),
      ...(input.knownEntities ?? []),
    ];
    const extractor = new CreatorDirectiveExtractor({
      llmClient: input.llmClient,
      model: this.options.model,
      tracer: this.options.tracer,
      turnId: input.turnId,
      sessionId: input.sessionId,
      onDegraded: (reason, error, metadata) => {
        this.reject({
          turnId: input.turnId,
          sessionId: input.sessionId,
          reason,
          error:
            metadata?.stopReason === undefined || metadata.stopReason === null
              ? error
              : (error ?? `stop_reason:${metadata.stopReason}`),
        });
      },
    });
    let candidates: CreatorDirectiveCandidate[];

    try {
      candidates = await extractor.extract({
        userMessage: input.userMessage,
        currentUserStreamEntryId: input.persistedUserEntryId,
        recentHistory: input.recentHistory,
        audienceEntityId: input.audienceEntityId,
        currentSenderEntityId: input.currentSenderEntityId,
        currentSenderDisplayName: input.currentSenderDisplayName ?? null,
        currentSenderBorgRole: input.currentSenderBorgRole,
        sessionAudienceRole: input.sessionAudienceRole,
        participantRoster: input.participantRoster ?? null,
        knownEntities,
      });
    } catch (error) {
      this.reject({
        turnId: input.turnId,
        sessionId: input.sessionId,
        reason: "extractor_failed",
        error,
      });
      return [];
    }

    const persisted: CreatorDirective[] = [];

    for (const [index, candidate] of candidates.entries()) {
      this.trace("creator_directive_candidate_extracted", {
        turnId: input.turnId,
        sessionId: input.sessionId,
        candidate_index: index,
        validationStatus: "candidate",
        kind: candidate.kind,
        subject_kind: candidate.subject_kind,
        content_scope: candidate.disclosure_policy.content_scope,
        mention_policy: candidate.disclosure_policy.mention_policy,
        priority: candidate.priority,
        confidence: candidate.confidence,
      });

      const resolution = this.resolveCandidate(candidate);

      if (!resolution.accepted) {
        this.reject({
          turnId: input.turnId,
          sessionId: input.sessionId,
          candidateIndex: index,
          candidate,
          reason: resolution.reason,
        });
        continue;
      }

      try {
        const directive = this.options.creatorDirectiveRepository.queue(
          this.buildQueueInput({
            candidate,
            createdByEntityId: input.currentSenderEntityId,
            sourceSessionId: input.sourceSessionId,
            sourceStreamEntryId: input.persistedUserEntryId,
            subjectEntityId: resolution.subjectEntityId,
            allowedEntityIds: resolution.allowedEntityIds,
            excludedEntityIds: resolution.excludedEntityIds,
          }),
        );

        persisted.push(directive);
        this.trace("creator_directive_persisted", {
          turnId: input.turnId,
          sessionId: input.sessionId,
          candidate_index: index,
          validationStatus: "accepted",
          directive_id: directive.id,
          kind: directive.kind,
          subject_kind: directive.subject_kind,
          content_scope: directive.disclosure_policy.content_scope,
          mention_policy: directive.disclosure_policy.mention_policy,
          priority: directive.priority,
        });
      } catch (error) {
        this.reject({
          turnId: input.turnId,
          sessionId: input.sessionId,
          candidateIndex: index,
          candidate,
          reason: "persist_failed",
          error,
        });
      }
    }

    return persisted;
  }
}
