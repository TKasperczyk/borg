import type { StreamEntry, StreamWriter } from "../../stream/index.js";
import type { PersistedTurnAttachment, TurnInputAttachment } from "../../attachments/index.js";
import type { EntityId } from "../../util/ids.js";
import type { ActivityRepository } from "../../memory/activity/index.js";
import type {
  PendingSocialAttribution,
  PendingTraitAttribution,
  WorkingMemory,
  WorkingMemoryStore,
} from "../../memory/working/index.js";
import type { SuppressionSet } from "../attention/index.js";
import type { PerceptionResult } from "../types.js";
import { ACTIVE_USER_MESSAGE_TURN_STATUS, persistUserMessage } from "./user-message.js";

export type TurnOpeningPersistenceOptions = {
  workingMemoryStore: Pick<WorkingMemoryStore, "save">;
  activityRepository?: Pick<ActivityRepository, "record">;
};

export type TurnOpeningPersistenceInput = {
  streamWriter: Pick<StreamWriter, "append" | "appendMany">;
  turnId: string;
  userMessage: string;
  attachments?: readonly TurnInputAttachment[];
  persistAttachments?: (
    input: Omit<
      import("../../attachments/index.js").PersistTurnAttachmentsInput,
      "attachments" | "streamWriter" | "parentEntry" | "turnId"
    > & {
      attachments: readonly TurnInputAttachment[];
      streamWriter: Pick<StreamWriter, "appendMany">;
      parentEntry: StreamEntry;
      turnId: string;
    },
  ) => Promise<PersistedTurnAttachment[]>;
  persistUserMessage?: boolean;
  persistPerception?: boolean;
  audience?: string;
  senderEntityId?: EntityId;
  speakerEntityId?: EntityId | null;
  audienceEntityId?: EntityId | null;
  workingMemory: WorkingMemory;
  pendingSocialAttribution: PendingSocialAttribution | null;
  pendingTraitAttribution: PendingTraitAttribution | null;
  suppressionSet: SuppressionSet;
  perception: PerceptionResult;
  now: () => number;
};

export type TurnOpeningPersistenceResult = {
  persistedUserEntry: StreamEntry | null;
  persistedAttachments: readonly PersistedTurnAttachment[];
  persistedAttachmentEntries: readonly StreamEntry[];
  currentUserContent: readonly import("../../attachments/index.js").BorgUserContentBlock[];
  persistedPerceptionEntry?: StreamEntry;
  workingMemory: WorkingMemory;
};

export class TurnOpeningPersistence {
  constructor(private readonly options: TurnOpeningPersistenceOptions) {}

  async persist(input: TurnOpeningPersistenceInput): Promise<TurnOpeningPersistenceResult> {
    const persistedUserEntry =
      input.persistUserMessage === false
        ? null
        : await persistUserMessage(this.options, {
            streamWriter: input.streamWriter,
            turnId: input.turnId,
            userMessage: input.userMessage,
            turnStatus: ACTIVE_USER_MESSAGE_TURN_STATUS,
            activityStatus: ACTIVE_USER_MESSAGE_TURN_STATUS,
            audience: input.audience,
            senderEntityId: input.senderEntityId,
            speakerEntityId: input.speakerEntityId,
            audienceEntityId: input.audienceEntityId,
          });
    const persistedAttachments =
      persistedUserEntry === null ||
      input.attachments === undefined ||
      input.attachments.length === 0 ||
      input.persistAttachments === undefined
        ? []
        : await input.persistAttachments({
            attachments: input.attachments,
            streamWriter: input.streamWriter,
            parentEntry: persistedUserEntry,
            turnId: input.turnId,
          });
    const currentUserContent = [
      {
        type: "text" as const,
        text: input.userMessage,
      },
      ...persistedAttachments.map((attachment) => attachment.contentBlock),
    ];

    const workingMemory = this.options.workingMemoryStore.save({
      ...input.workingMemory,
      pending_social_attribution: input.pendingSocialAttribution,
      pending_trait_attribution: input.pendingTraitAttribution,
      suppressed: input.suppressionSet.snapshot(),
      updated_at: input.now(),
    });

    const persistedPerceptionEntry =
      input.persistPerception === false
        ? undefined
        : await input.streamWriter.append({
            kind: "perception",
            turn_id: input.turnId,
            turn_status: ACTIVE_USER_MESSAGE_TURN_STATUS,
            content: {
              mode: input.perception.mode,
              isOperational: input.perception.isOperational === true,
              entities: input.perception.entities,
              userIdentityNames: input.perception.userIdentityNames ?? [],
              temporalCue: input.perception.temporalCue,
              affectiveSignal: input.perception.affectiveSignal,
              affectiveSignalDegraded: input.perception.affectiveSignalDegraded === true,
            },
            ...(input.audience === undefined ? {} : { audience: input.audience }),
          });

    return {
      persistedUserEntry,
      persistedAttachments,
      persistedAttachmentEntries: persistedAttachments.map((attachment) => attachment.streamEntry),
      currentUserContent,
      ...(persistedPerceptionEntry === undefined ? {} : { persistedPerceptionEntry }),
      workingMemory,
    };
  }
}
