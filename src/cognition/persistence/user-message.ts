import type { ActivityEventStatus, ActivityRepository } from "../../memory/activity/index.js";
import type {
  StreamEntry,
  StreamSourceMessageKey,
  StreamTurnStatus,
  StreamWriter,
} from "../../stream/index.js";
import type { EntityId } from "../../util/ids.js";

export const ACTIVE_USER_MESSAGE_TURN_STATUS = "active" satisfies StreamTurnStatus;
export const ENQUEUED_USER_CONTACT_ACTIVITY_STATUS = "active" satisfies ActivityEventStatus;

export type PersistUserMessageOptions = {
  activityRepository?: Pick<ActivityRepository, "record">;
};

export type PersistUserMessageInput = {
  streamWriter: Pick<StreamWriter, "append">;
  userMessage: string;
  turnId?: string;
  turnStatus?: StreamTurnStatus;
  sourceMessageKey?: StreamSourceMessageKey;
  activityOccurredAt?: number;
  activityStatus: ActivityEventStatus;
  audience?: string;
  senderEntityId?: EntityId;
  speakerEntityId?: EntityId | null;
  audienceEntityId?: EntityId | null;
};

export async function persistUserMessage(
  options: PersistUserMessageOptions,
  input: PersistUserMessageInput,
): Promise<StreamEntry> {
  const persistedUserEntry = await input.streamWriter.append({
    kind: "user_msg",
    content: input.userMessage,
    ...(input.turnId === undefined ? {} : { turn_id: input.turnId }),
    turn_status: input.turnStatus ?? ACTIVE_USER_MESSAGE_TURN_STATUS,
    ...(input.audience === undefined ? {} : { audience: input.audience }),
    ...(input.senderEntityId === undefined ? {} : { sender_entity_id: input.senderEntityId }),
    ...(input.sourceMessageKey === undefined ? {} : { source_message_key: input.sourceMessageKey }),
  });
  const speakerEntityId = input.speakerEntityId ?? input.senderEntityId ?? null;

  options.activityRepository?.record({
    kind: "user_contact",
    occurredAt: input.activityOccurredAt ?? persistedUserEntry.timestamp,
    sessionId: persistedUserEntry.session_id,
    turnId: input.turnId ?? null,
    speakerEntityId,
    actorEntityId: speakerEntityId,
    audienceEntityId: input.audienceEntityId ?? null,
    participantEntityIds: [speakerEntityId, input.audienceEntityId ?? null].filter(
      (entityId): entityId is EntityId => entityId !== null,
    ),
    sourceStreamEntryIds: [persistedUserEntry.id],
    status: input.activityStatus,
  });

  return persistedUserEntry;
}
