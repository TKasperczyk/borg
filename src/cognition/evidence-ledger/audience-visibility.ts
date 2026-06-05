import type { ActionRecord } from "../../memory/actions/index.js";
import type { CommitmentRecord, EntityRepository } from "../../memory/commitments/index.js";
import type { GoalRecord } from "../../memory/self/index.js";
import type { EntityId } from "../../util/ids.js";
import type { ActiveParticipant } from "../participants.js";

export function scopedCommitmentsForEntity(
  commitments: readonly CommitmentRecord[],
  entityId: EntityId,
): CommitmentRecord[] {
  return commitments.filter(
    (commitment) =>
      commitment.made_to_entity === entityId ||
      commitment.restricted_audience === entityId ||
      commitment.about_entity === entityId ||
      commitment.committed_by_entity_id === entityId,
  );
}

export function scopedGoalsForEntity(
  goals: readonly GoalRecord[],
  entityId: EntityId,
): GoalRecord[] {
  return goals.filter(
    (goal) => goal.audience_entity_id === entityId || goal.owner_entity_id === entityId,
  );
}

export function dedupeCommitments(records: readonly CommitmentRecord[]): CommitmentRecord[] {
  return [...new Map(records.map((record) => [record.id, record])).values()].sort(
    (left, right) => right.priority - left.priority || left.created_at - right.created_at,
  );
}

export function dedupeGoals(records: readonly GoalRecord[]): GoalRecord[] {
  return [...new Map(records.map((record) => [record.id, record])).values()].sort(
    (left, right) => right.priority - left.priority || left.created_at - right.created_at,
  );
}

export function dedupeActions(records: readonly ActionRecord[]): ActionRecord[] {
  return [...new Map(records.map((record) => [record.id, record])).values()].sort(
    (left, right) => right.updated_at - left.updated_at || left.id.localeCompare(right.id),
  );
}

export function visibleAudienceEntityIds(
  audienceEntityId: EntityId | null,
  activeParticipants: readonly ActiveParticipant[] | undefined,
): ReadonlySet<EntityId> {
  const ids = new Set((activeParticipants ?? []).map((participant) => participant.entityId));

  if (audienceEntityId !== null) {
    ids.add(audienceEntityId);
  }

  return ids;
}

export function audienceIsVisibleToSession(
  scopedAudienceEntityId: EntityId | null | undefined,
  currentAudienceEntityId: EntityId | null,
  activeParticipantIds: ReadonlySet<EntityId>,
): boolean {
  if (scopedAudienceEntityId === null || scopedAudienceEntityId === undefined) {
    return true;
  }

  return (
    scopedAudienceEntityId === currentAudienceEntityId ||
    activeParticipantIds.has(scopedAudienceEntityId)
  );
}

export function isActionVisibleForCurrentAudienceStanding(
  action: ActionRecord,
  audienceEntityId: EntityId | null,
  activeParticipantIds: ReadonlySet<EntityId>,
): boolean {
  return audienceIsVisibleToSession(
    action.audience_entity_id,
    audienceEntityId,
    activeParticipantIds,
  );
}

export function isCommitmentVisibleToSession(
  commitment: CommitmentRecord,
  audienceEntityId: EntityId | null,
  activeParticipantIds: ReadonlySet<EntityId>,
): boolean {
  if (commitment.restricted_audience !== null) {
    return audienceIsVisibleToSession(
      commitment.restricted_audience,
      audienceEntityId,
      activeParticipantIds,
    );
  }

  return audienceIsVisibleToSession(
    commitment.made_to_entity,
    audienceEntityId,
    activeParticipantIds,
  );
}

export function isGoalVisibleToSession(
  goal: GoalRecord,
  audienceEntityId: EntityId | null,
  activeParticipantIds: ReadonlySet<EntityId>,
): boolean {
  return audienceIsVisibleToSession(
    goal.audience_entity_id,
    audienceEntityId,
    activeParticipantIds,
  );
}

function entityIdPointsAtPerson(
  entityId: EntityId | null | undefined,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): boolean {
  return (
    entityId !== null &&
    entityId !== undefined &&
    entityRepository?.get(entityId)?.kind === "person"
  );
}

export function actionBelongsToGroupChannel(
  action: ActionRecord,
  audienceEntityId: EntityId,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): boolean {
  if (action.actor === "user" || action.audience_entity_id !== audienceEntityId) {
    return false;
  }

  return action.actor === "borg" || !entityIdPointsAtPerson(action.actor, entityRepository);
}

export function commitmentBelongsToGroupChannel(
  commitment: CommitmentRecord,
  audienceEntityId: EntityId,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): boolean {
  if (entityIdPointsAtPerson(commitment.committed_by_entity_id ?? null, entityRepository)) {
    return false;
  }

  return scopedCommitmentsForEntity([commitment], audienceEntityId).length > 0;
}

export function goalBelongsToGroupChannel(
  goal: GoalRecord,
  audienceEntityId: EntityId,
  entityRepository: Pick<EntityRepository, "get"> | undefined,
): boolean {
  if (entityIdPointsAtPerson(goal.owner_entity_id ?? null, entityRepository)) {
    return false;
  }

  return scopedGoalsForEntity([goal], audienceEntityId).length > 0;
}
