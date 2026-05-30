/*
 * v1 delivery contract: Inbound is durable (committed before ack). Reply generation is replay-safe (cursor-stamped response_to + reconcile-before-generate; at-least-once). External delivery is NOT auto-retried by borg (no durable outbox in v1). Accepted D1 loss window: a crash after the stamped terminal append but before the transport delivers leaves the reply recorded-but-possibly-undelivered, not retried. Dedup is per source_message_key via the single-writer daemon's in-process serialization; cross-process concurrent writers are out of v1 scope.
 */
import type { ActivityRepository } from "../../memory/activity/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type {
  SessionEnsureInput,
  SessionRecord,
  SessionsRepository,
} from "../../sessions/index.js";
import type {
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
  audience?: string;
  audienceEntityId?: EntityId | null;
  attachments?: readonly unknown[];
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
  entryIndex: Pick<
    StreamEntryIndexRepository,
    "lookupBySourceMessageKey" | "isPoisoned" | "backfillSession"
  >;
  createStreamWriter: (sessionId: SessionId) => Pick<StreamWriter, "append" | "close">;
  isDuplicatePendingResponse?: (record: StreamEntryIndexRecord) => boolean;
  onPendingDuplicate?: (record: StreamEntryIndexRecord) => void;
  clock: Clock;
};

export class MessageEnqueuer {
  private readonly sessionTails = new Map<SessionId, Promise<void>>();

  constructor(private readonly options: MessageEnqueuerOptions) {}

  async enqueueMessage(input: BorgEnqueueMessageInput): Promise<BorgEnqueueMessageResult> {
    this.validateTextOnly(input.attachments);
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
        this.recoverPendingDuplicateReceipt({
          record: duplicate,
          input,
          session,
          arrivedAt,
        });
        return this.duplicateResult(duplicate);
      }

      const writer = this.options.createStreamWriter(session.session_id);

      try {
        const entry = await persistUserMessage(
          {
            activityRepository: this.options.activityRepository,
          },
          {
            streamWriter: writer,
            userMessage: input.userMessage,
            sourceMessageKey,
            activityOccurredAt: arrivedAt,
            activityStatus: ENQUEUED_USER_CONTACT_ACTIVITY_STATUS,
            audience: input.audience,
            senderEntityId: input.senderEntityId,
            speakerEntityId: input.senderEntityId,
            audienceEntityId: input.audienceEntityId ?? session.audience_entity_id ?? null,
          },
        );

        this.options.sessionsRepository.touch(session.session_id, {
          at: arrivedAt,
          messageCountDelta: 1,
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
      await this.options.entryIndex.backfillSession(sessionId);
    } catch (error) {
      throw new StreamError(`Stream entry index is poisoned for committed session ${sessionId}`, {
        cause: error,
        code: "STREAM_INDEX_POISONED",
      });
    }
  }

  private recoverPendingDuplicateReceipt(context: {
    record: StreamEntryIndexRecord;
    input: BorgEnqueueMessageInput;
    session: SessionRecord;
    arrivedAt: number;
  }): void {
    if (this.options.isDuplicatePendingResponse?.(context.record) !== true) {
      return;
    }

    const streamEntryId = context.record.entry_id as StreamEntryId;
    const existingContact = this.options.activityRepository.getByKindAndSource("user_contact", [
      streamEntryId,
    ]);

    if (existingContact === null) {
      const speakerEntityId = context.record.sender_entity_id ?? context.input.senderEntityId;
      const audienceEntityId =
        context.input.audienceEntityId ?? context.session.audience_entity_id ?? null;

      this.options.activityRepository.record({
        kind: "user_contact",
        occurredAt: context.arrivedAt,
        sessionId: context.record.session_id,
        turnId: null,
        speakerEntityId,
        actorEntityId: speakerEntityId,
        audienceEntityId,
        participantEntityIds: [speakerEntityId, audienceEntityId].filter(
          (entityId): entityId is EntityId => entityId !== null,
        ),
        sourceStreamEntryIds: [streamEntryId],
        status: ENQUEUED_USER_CONTACT_ACTIVITY_STATUS,
      });

      this.options.sessionsRepository.touch(context.record.session_id, {
        at: context.arrivedAt,
        messageCountDelta: 1,
      });
    }

    this.options.onPendingDuplicate?.(context.record);
  }

  private validateTextOnly(attachments: readonly unknown[] | undefined): void {
    if (attachments !== undefined && attachments.length > 0) {
      throw new CognitionError("enqueueMessage v1 accepts text-only messages", {
        code: "ENQUEUE_ATTACHMENTS_UNSUPPORTED",
      });
    }
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
