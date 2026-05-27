import type { DeliberationRoutingOverride } from "../../deliberation/types.js";
import type { EvidenceLedger } from "../../evidence-ledger/index.js";
import type {
  ActualFrameAnomalyClassification,
  FrameAnomalyConversationContext,
} from "../../frame-anomaly/index.js";
import type { ActiveParticipant, ParticipantProfileContext } from "../../participants.js";
import type { PerceptionResult } from "../../types.js";
import type { BorgRole, EntityRepository } from "../../../memory/commitments/index.js";
import type { OpenQuestion, OpenQuestionsRepository } from "../../../memory/self/index.js";
import type { RelationalSlotRepository } from "../../../memory/relational-slots/index.js";
import type { SessionAudienceRole } from "../../../sessions/index.js";
import type { StreamEntry } from "../../../stream/index.js";
import type { EntityId, StreamEntryId } from "../../../util/ids.js";

const DELIBERATION_RELATIONAL_SLOT_LIMIT = 24;
const EVIDENCE_LEDGER_OPEN_QUESTION_ID_PREFIX = "open_question:";

export type BuildContradictionRoutingOverrideInput = {
  isUserTurn: boolean;
  perception: Pick<PerceptionResult, "isOperational">;
  audienceEntityId: EntityId | null;
  openQuestionsRepository: Pick<OpenQuestionsRepository, "get">;
  evidenceLedger: EvidenceLedger | null;
  enabled?: boolean;
};

export function buildContradictionRoutingOverride(
  input: BuildContradictionRoutingOverrideInput,
): DeliberationRoutingOverride | null {
  if (input.enabled === false) {
    return null;
  }

  if (!input.isUserTurn) {
    return null;
  }

  if (input.perception.isOperational !== true) {
    return null;
  }

  const evidenceVisibleOpenQuestionIds = collectEvidenceLedgerOpenQuestionIds(input.evidenceLedger);
  const openQuestions = [...evidenceVisibleOpenQuestionIds]
    .map((id) => input.openQuestionsRepository.get(id as OpenQuestion["id"]))
    .filter(
      (question): question is OpenQuestion =>
        question !== null &&
        question.status === "open" &&
        question.source === "contradiction" &&
        openQuestionScopesToAudience(question, input.audienceEntityId),
    );

  if (openQuestions.length === 0) {
    return null;
  }

  return {
    forceSystem2: true,
    reason: "open_question_contradiction",
    forcedBy: "open_question_contradiction",
    oqIds: openQuestions.map((question) => question.id),
    contradictionFingerprints: contradictionFingerprintsForOpenQuestions(openQuestions),
    openQuestions: openQuestions.map((question, index) => ({
      id: question.id,
      question: question.question,
      source: question.source,
      localHandle: `contradiction_${index + 1}`,
    })),
    audienceEntityId: input.audienceEntityId,
    isOperational: true,
  };
}

function contradictionFingerprintsForOpenQuestions(
  openQuestions: readonly OpenQuestion[],
): string[] {
  // Open-question evidence handles can grow as reflection links more records.
  // The OQ id is the stable contradiction identity used for cooldown.
  return [...new Set(openQuestions.map((question) => `open_question:${question.id}`))].sort(
    (left, right) => left.localeCompare(right),
  );
}

function collectEvidenceLedgerOpenQuestionIds(ledger: EvidenceLedger | null): Set<string> {
  const ids = new Set<string>();
  const openQuestionsSection = ledger?.sections.find((section) => section.id === "open_questions");

  if (openQuestionsSection === undefined) {
    return ids;
  }

  for (const entry of openQuestionsSection.entries) {
    if (entry.id.startsWith(EVIDENCE_LEDGER_OPEN_QUESTION_ID_PREFIX)) {
      ids.add(entry.id.slice(EVIDENCE_LEDGER_OPEN_QUESTION_ID_PREFIX.length));
    }
  }

  return ids;
}

function openQuestionScopesToAudience(
  question: Pick<OpenQuestion, "audience_entity_id">,
  audienceEntityId: EntityId | null,
): boolean {
  return question.audience_entity_id === null || question.audience_entity_id === audienceEntityId;
}

export function listConstrainedRelationalSlotsForParticipants(
  repository: RelationalSlotRepository,
  participants: readonly ActiveParticipant[],
) {
  if (participants.length === 0) {
    return repository.listConstrained({
      limit: DELIBERATION_RELATIONAL_SLOT_LIMIT,
    });
  }

  return participants
    .flatMap((participant) =>
      repository.listConstrained({
        subjectEntityId: participant.entityId,
        limit: DELIBERATION_RELATIONAL_SLOT_LIMIT,
      }),
    )
    .slice(0, DELIBERATION_RELATIONAL_SLOT_LIMIT);
}

export function listSharedStateRelationalSlotsForParticipants(
  repository: Pick<RelationalSlotRepository, "list"> | undefined,
  participants: readonly ActiveParticipant[],
) {
  const states = ["established", "contested", "quarantined"] as const;

  if (repository === undefined) {
    return [];
  }

  if (participants.length === 0) {
    return repository.list({
      states,
      limit: DELIBERATION_RELATIONAL_SLOT_LIMIT,
    });
  }

  return participants
    .flatMap((participant) =>
      repository.list({
        subjectEntityId: participant.entityId,
        states,
        limit: DELIBERATION_RELATIONAL_SLOT_LIMIT,
      }),
    )
    .slice(0, DELIBERATION_RELATIONAL_SLOT_LIMIT);
}

export function audienceProfileForParticipants(
  participantProfiles: readonly ParticipantProfileContext[],
  audienceEntityId: EntityId | null,
) {
  if (participantProfiles.length === 1) {
    return participantProfiles[0]?.profile ?? null;
  }

  if (audienceEntityId === null) {
    return null;
  }

  return (
    participantProfiles.find((participant) => participant.entityId === audienceEntityId)?.profile ??
    null
  );
}

function entityDisplayName(
  entityRepository: Pick<EntityRepository, "get">,
  entityId: EntityId | null | undefined,
): string | null {
  if (entityId === null || entityId === undefined) {
    return null;
  }

  return entityRepository.get(entityId)?.canonical_name ?? null;
}

function previousUserSenderContext(input: {
  currentUserEntryId: StreamEntryId;
  streamEntries: readonly StreamEntry[];
  entityRepository: Pick<EntityRepository, "get">;
}): FrameAnomalyConversationContext["previous_user_sender"] {
  for (let index = input.streamEntries.length - 1; index >= 0; index -= 1) {
    const entry = input.streamEntries[index];

    if (entry === undefined || entry.kind !== "user_msg" || entry.id === input.currentUserEntryId) {
      continue;
    }

    const senderEntityId = entry.sender_entity_id ?? null;

    if (senderEntityId === null) {
      return null;
    }

    return {
      id: senderEntityId,
      display_name: entityDisplayName(input.entityRepository, senderEntityId),
    };
  }

  return null;
}

export function buildFrameAnomalyConversationContext(input: {
  audienceEntityId: EntityId | null;
  audienceEntity: ReturnType<EntityRepository["get"]>;
  currentUserEntry: StreamEntry | null | undefined;
  activeParticipants: readonly ActiveParticipant[];
  participantStreamEntries: readonly StreamEntry[];
  entityRepository: Pick<EntityRepository, "get" | "resolve">;
  currentSenderEntityId: EntityId | null;
  currentSenderBorgRole: BorgRole | null;
  sessionAudienceRole: SessionAudienceRole;
}): FrameAnomalyConversationContext | undefined {
  if (input.currentUserEntry === null || input.currentUserEntry === undefined) {
    return undefined;
  }

  const currentSenderEntityId = input.currentSenderEntityId;
  const previousUserSender = previousUserSenderContext({
    currentUserEntryId: input.currentUserEntry.id,
    streamEntries: input.participantStreamEntries,
    entityRepository: input.entityRepository,
  });
  const assistantEntityId = input.entityRepository.resolve("self", {
    kind: "self",
    provenance: "assistant_seeded",
  });

  return {
    audience: {
      id: input.audienceEntityId,
      display_name: input.audienceEntity?.canonical_name ?? null,
      kind: input.audienceEntity?.kind ?? null,
    },
    current_sender: {
      id: currentSenderEntityId,
      display_name: entityDisplayName(input.entityRepository, currentSenderEntityId),
    },
    current_sender_borg_role: input.currentSenderBorgRole,
    session_audience_role: input.sessionAudienceRole,
    participants: input.activeParticipants,
    assistant_identity: {
      id: assistantEntityId,
      display_name: "Borg / Assistant",
    },
    previous_user_sender: previousUserSender,
    sender_changed_since_previous_user_turn:
      previousUserSender !== null &&
      currentSenderEntityId !== null &&
      previousUserSender.id !== currentSenderEntityId,
  };
}
