/*
 * v1 delivery contract: Inbound is durable (committed before ack). Reply generation is replay-safe (cursor-stamped response_to + reconcile-before-generate; at-least-once). External delivery is NOT auto-retried by borg (no durable outbox in v1). Accepted D1 loss window: a crash after the stamped terminal append but before the transport delivers leaves the reply recorded-but-possibly-undelivered, not retried. Dedup is per source_message_key via the single-writer daemon's in-process serialization; cross-process concurrent writers are out of v1 scope.
 */
import type { ActivityRepository } from "../../memory/activity/index.js";
import type {
  AttachmentService,
  ImagePerceptionService,
  TurnInputAttachment,
} from "../../attachments/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type {
  SessionEnsureInput,
  SessionRecord,
  SessionsRepository,
} from "../../sessions/index.js";
import type {
  StreamConversation,
  StreamEntry,
  StreamEntryMetadata,
  StreamEntryIndexRecord,
  StreamEntryIndexRepository,
  StreamSourceMessageKey,
  StreamWriter,
} from "../../stream/index.js";
import { streamSourceMessageKeySchema } from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import { CognitionError, StreamError } from "../../util/errors.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import {
  ENQUEUED_USER_CONTACT_ACTIVITY_STATUS,
  persistUserMessage,
} from "../persistence/user-message.js";

export type BorgEnqueueMessageInput = {
  session: SessionEnsureInput & { source_external_id: string };
  userMessage: string;
  senderEntityId: EntityId;
  sourceMessageKey: StreamSourceMessageKey;
  arrivedAt?: number;
  observedAt?: number;
  conversation?: StreamConversation;
  metadata?: StreamEntryMetadata;
  audience?: string;
  audienceEntityId?: EntityId | null;
  attachments?: readonly TurnInputAttachment[];
};

export type BorgEnqueueMessageResult = {
  status: "enqueued" | "duplicate";
  sessionId: SessionId;
  streamEntryId: StreamEntryId;
};

export type MessageEnqueuerOptions = {
  sessionsRepository: Pick<SessionsRepository, "ensure" | "touch">;
  entityRepository: Pick<EntityRepository, "get">;
  activityRepository: Pick<ActivityRepository, "record" | "getByKindAndSource">;
  attachmentService: Pick<
    AttachmentService,
    "validateAttachments" | "persistParentEntryAttachments"
  >;
  imagePerceptionService: Pick<ImagePerceptionService, "perceiveAttachment">;
  entryIndex: Pick<
    StreamEntryIndexRepository,
    "lookupBySourceMessageKey" | "isPoisoned" | "setReceiptPending"
  >;
  createReceiptStreamWriter: (
    sessionId: SessionId,
  ) => Pick<StreamWriter, "append" | "appendMany" | "close">;
  repairSessionStreamEntryIndex: (sessionId: SessionId) => Promise<unknown>;
  isDuplicatePendingResponse?: (record: StreamEntryIndexRecord) => boolean;
  onReceiptReady?: (event: {
    sessionId: SessionId;
    pendingAt: number;
    entries: readonly StreamEntry[];
  }) => void;
  clock: Clock;
};

export class MessageEnqueuer {
  private readonly sessionTails = new Map<SessionId, Promise<void>>();

  constructor(private readonly options: MessageEnqueuerOptions) {}

  async enqueueMessage(input: BorgEnqueueMessageInput): Promise<BorgEnqueueMessageResult> {
    const attachments = input.attachments ?? [];
    this.options.attachmentService.validateAttachments(attachments);
    const sourceMessageKey = this.parseSourceMessageKey(input.sourceMessageKey);
    this.validateSourceMessageKeyForInputSession(sourceMessageKey, input.session);
    const arrivedAt = this.resolveArrivedAt(input.arrivedAt);
    this.requireKnownSender(input.senderEntityId);
    const session = this.options.sessionsRepository.ensure(input.session);

    return this.runForSession(session.session_id, async () => {
      if (this.options.entryIndex.isPoisoned(session.session_id)) {
        await this.repairPoisonedSessionBeforeDedup(session.session_id);
      }

      const duplicate = this.options.entryIndex.lookupBySourceMessageKey(sourceMessageKey);

      if (duplicate !== null) {
        await this.recoverPendingDuplicateReceipt({
          record: duplicate,
          input,
          session,
          arrivedAt,
          attachments,
        });
        return this.duplicateResult(duplicate);
      }

      const writer = this.options.createReceiptStreamWriter(session.session_id);

      try {
        const entry = await persistUserMessage(
          {},
          {
            streamWriter: writer,
            userMessage: input.userMessage,
            sourceMessageKey,
            observedAt: input.observedAt,
            conversation: input.conversation,
            metadata: input.metadata,
            activityOccurredAt: arrivedAt,
            activityStatus: ENQUEUED_USER_CONTACT_ACTIVITY_STATUS,
            audience: input.audience,
            senderEntityId: input.senderEntityId,
            speakerEntityId: input.senderEntityId,
            audienceEntityId: input.audienceEntityId ?? session.audience_entity_id ?? null,
            receiptPending: attachments.length > 0,
          },
        );
        const attachmentEntries =
          attachments.length === 0
            ? []
            : await this.persistAndPerceiveReceiptAttachments({
                attachments,
                streamWriter: writer,
                parentEntry: entry,
                audienceEntityId: input.audienceEntityId ?? session.audience_entity_id ?? null,
              });

        this.recordReceiptContact({
          record: {
            entryId: entry.id,
            sessionId: session.session_id,
            senderEntityId: input.senderEntityId,
          },
          input,
          session,
          arrivedAt,
        });

        this.options.sessionsRepository.touch(session.session_id, {
          at: arrivedAt,
          messageCountDelta: 1,
        });

        this.options.onReceiptReady?.({
          sessionId: session.session_id,
          pendingAt: entry.timestamp,
          entries: [entry, ...attachmentEntries],
        });

        return {
          status: "enqueued",
          sessionId: session.session_id,
          streamEntryId: entry.id,
        };
      } finally {
        writer.close();
      }
    });
  }

  private async runForSession<T>(sessionId: SessionId, task: () => Promise<T>): Promise<T> {
    const previous = this.sessionTails.get(sessionId) ?? Promise.resolve();
    const run = previous.catch(() => undefined).then(task);
    const stored = run.then(
      () => undefined,
      () => undefined,
    );

    this.sessionTails.set(sessionId, stored);

    try {
      return await run;
    } finally {
      if (this.sessionTails.get(sessionId) === stored) {
        this.sessionTails.delete(sessionId);
      }
    }
  }

  private async repairPoisonedSessionBeforeDedup(sessionId: SessionId): Promise<void> {
    try {
      await this.options.repairSessionStreamEntryIndex(sessionId);
    } catch (error) {
      throw new StreamError(`Stream entry index is poisoned for committed session ${sessionId}`, {
        cause: error,
        code: "STREAM_INDEX_POISONED",
      });
    }
  }

  private async recoverPendingDuplicateReceipt(context: {
    record: StreamEntryIndexRecord;
    input: BorgEnqueueMessageInput;
    session: SessionRecord;
    arrivedAt: number;
    attachments: readonly TurnInputAttachment[];
  }): Promise<void> {
    const receiptPending = context.record.receipt_pending === true;
    const streamEntryId = context.record.entry_id as StreamEntryId;
    const existingContact = this.options.activityRepository.getByKindAndSource("user_contact", [
      streamEntryId,
    ]);
    const needsReceiptRecovery = receiptPending || existingContact === null;

    if (
      !needsReceiptRecovery &&
      this.options.isDuplicatePendingResponse?.(context.record) !== true
    ) {
      return;
    }

    let receiptEntries: StreamEntry[] = [];

    if (needsReceiptRecovery) {
      if (context.attachments.length > 0) {
        const writer = this.options.createReceiptStreamWriter(context.record.session_id);

        try {
          receiptEntries = await this.persistAndPerceiveReceiptAttachments({
            attachments: context.attachments,
            streamWriter: writer,
            parentEntry: this.parentEntryFromDuplicateRecord(context),
            audienceEntityId:
              context.input.audienceEntityId ?? context.session.audience_entity_id ?? null,
          });
        } finally {
          writer.close();
        }
      }

      if (existingContact === null) {
        this.recordReceiptContact({
          record: {
            entryId: streamEntryId,
            sessionId: context.record.session_id,
            senderEntityId: context.record.sender_entity_id ?? context.input.senderEntityId,
          },
          input: context.input,
          session: context.session,
          arrivedAt: context.arrivedAt,
        });
        this.options.sessionsRepository.touch(context.record.session_id, {
          at: context.arrivedAt,
          messageCountDelta: 1,
        });
      }
    }

    this.options.onReceiptReady?.({
      sessionId: context.record.session_id,
      pendingAt: context.record.timestamp,
      entries: receiptEntries,
    });
  }

  private async persistAndPerceiveReceiptAttachments(input: {
    attachments: readonly TurnInputAttachment[];
    streamWriter: Pick<StreamWriter, "appendMany">;
    parentEntry: StreamEntry;
    audienceEntityId: EntityId | null;
  }): Promise<StreamEntry[]> {
    const persisted = await this.options.attachmentService.persistParentEntryAttachments({
      attachments: input.attachments,
      streamWriter: input.streamWriter,
      parentEntry: input.parentEntry,
      audienceEntityId: input.audienceEntityId,
    });
    this.options.entryIndex.setReceiptPending(input.parentEntry.id, false);

    for (const item of persisted) {
      if (item.record.perception_id !== null) {
        continue;
      }

      await this.options.imagePerceptionService.perceiveAttachment({
        attachmentId: item.attachmentId,
        turnId: input.parentEntry.id,
      });
    }

    return persisted.flatMap((item) => (item.streamEntry === null ? [] : [item.streamEntry]));
  }

  private parentEntryFromDuplicateRecord(context: {
    record: StreamEntryIndexRecord;
    input: BorgEnqueueMessageInput;
  }): StreamEntry {
    return {
      id: context.record.entry_id as StreamEntryId,
      timestamp: context.record.timestamp,
      ...(context.record.entry_index === null ? {} : { entry_index: context.record.entry_index }),
      kind: "user_msg",
      content: context.input.userMessage,
      ...(context.record.turn_id === null ? {} : { turn_id: context.record.turn_id }),
      turn_status: context.record.turn_status ?? "active",
      ...(context.input.audience === undefined ? {} : { audience: context.input.audience }),
      sender_entity_id: context.record.sender_entity_id ?? context.input.senderEntityId,
      reply_target_entity_id: null,
      source_message_key: context.input.sourceMessageKey,
      ...(context.input.observedAt === undefined ? {} : { observed_at: context.input.observedAt }),
      ...(context.input.conversation === undefined
        ? {}
        : { conversation: context.input.conversation }),
      ...(context.input.metadata === undefined ? {} : { metadata: context.input.metadata }),
      session_id: context.record.session_id,
      compressed: false,
    };
  }

  private recordReceiptContact(context: {
    record: {
      entryId: StreamEntryId;
      sessionId: SessionId;
      senderEntityId: EntityId;
    };
    input: BorgEnqueueMessageInput;
    session: SessionRecord;
    arrivedAt: number;
  }): void {
    const audienceEntityId =
      context.input.audienceEntityId ?? context.session.audience_entity_id ?? null;

    this.options.activityRepository.record({
      kind: "user_contact",
      occurredAt: context.arrivedAt,
      sessionId: context.record.sessionId,
      turnId: null,
      speakerEntityId: context.record.senderEntityId,
      actorEntityId: context.record.senderEntityId,
      audienceEntityId,
      participantEntityIds: [context.record.senderEntityId, audienceEntityId].filter(
        (entityId): entityId is EntityId => entityId !== null,
      ),
      sourceStreamEntryIds: [context.record.entryId],
      status: ENQUEUED_USER_CONTACT_ACTIVITY_STATUS,
    });
  }

  private parseSourceMessageKey(key: StreamSourceMessageKey): StreamSourceMessageKey {
    const parsed = streamSourceMessageKeySchema.safeParse(key);

    if (!parsed.success) {
      throw new CognitionError("enqueueMessage sourceMessageKey is invalid", {
        cause: parsed.error,
        code: "ENQUEUE_SOURCE_MESSAGE_KEY_INVALID",
      });
    }

    return parsed.data;
  }

  private requireKnownSender(senderEntityId: EntityId): void {
    if (this.options.entityRepository.get(senderEntityId) === null) {
      throw new CognitionError("enqueueMessage senderEntityId is unknown", {
        code: "ENQUEUE_SENDER_UNKNOWN",
      });
    }
  }

  private validateSourceMessageKeyForInputSession(
    key: StreamSourceMessageKey,
    session: SessionEnsureInput,
  ): void {
    if (key.source_type !== session.source_type) {
      throw new CognitionError("enqueueMessage sourceMessageKey source_type mismatches session", {
        code: "ENQUEUE_SOURCE_MESSAGE_KEY_MISMATCH",
      });
    }

    if (
      session.source_external_id === null ||
      session.source_external_id === undefined ||
      key.source_external_id !== session.source_external_id
    ) {
      throw new CognitionError(
        "enqueueMessage sourceMessageKey source_external_id mismatches session",
        {
          code: "ENQUEUE_SOURCE_MESSAGE_KEY_MISMATCH",
        },
      );
    }
  }

  private resolveArrivedAt(arrivedAt: number | undefined): number {
    const value = arrivedAt ?? this.options.clock.now();

    if (!Number.isFinite(value) || !Number.isInteger(value)) {
      throw new CognitionError("enqueueMessage arrivedAt must be a finite integer", {
        code: "ENQUEUE_ARRIVED_AT_INVALID",
      });
    }

    return value;
  }

  private duplicateResult(record: StreamEntryIndexRecord): BorgEnqueueMessageResult {
    return {
      status: "duplicate",
      sessionId: record.session_id,
      streamEntryId: record.entry_id as StreamEntryId,
    };
  }
}
