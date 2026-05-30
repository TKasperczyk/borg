import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { ActivityEvent, ActivityEventRecordInput } from "../../memory/activity/index.js";
import {
  AttachmentBlobStore,
  AttachmentRepository,
  AttachmentService,
  attachmentMigrations,
  type ImagePerceptionService,
  type TurnInputAttachment,
} from "../../attachments/index.js";
import type { PersistedParentEntryAttachment } from "../../attachments/index.js";
import type { EntityRecord } from "../../memory/commitments/index.js";
import type { SessionRecord } from "../../sessions/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  streamEntryIndexMigrations,
  type StreamEntry,
  type StreamEntryIndexRecord,
  type StreamEntryInput,
  type StreamSourceMessageKey,
} from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { AttachmentError } from "../../util/errors.js";
import {
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { MessageEnqueuer } from "./enqueuer.js";

const GIF_1X1 = Uint8Array.from([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00]);

function makeSession(input: {
  sessionId: SessionId;
  audienceEntityId?: EntityId | null;
}): SessionRecord {
  return {
    session_id: input.sessionId,
    source_type: "demo",
    source_external_id: "conversation-1",
    source_url: null,
    label: "Demo",
    audience_label: "Demo room",
    audience_entity_id: input.audienceEntityId ?? null,
    conversation_kind: "thread",
    created_at: 1_000,
    last_activity_at: 1_000,
    last_turn_id: null,
    message_count: 0,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
  };
}

function makeEntity(id: EntityId): EntityRecord {
  return {
    id,
    canonical_name: "Sender",
    aliases: [],
    kind: "person",
    borg_role: null,
    name_provenance: "unknown",
    created_at: 1_000,
  };
}

function makeIndexRecord(input: {
  entryId: StreamEntryId;
  sessionId: SessionId;
}): StreamEntryIndexRecord {
  return {
    entry_id: input.entryId,
    session_id: input.sessionId,
    byte_offset: 0,
    entry_index: 0,
    timestamp: 1_000,
    kind: "user_msg",
    sender_entity_id: null,
    turn_id: null,
    turn_status: "active",
    active: true,
    receipt_pending: false,
    source_message_key_source_type: "demo",
    source_message_key_source_external_id: "conversation-1",
    source_message_key_external_message_id: "message-1",
    response_to_kind: null,
    response_to_from_cursor_ts: null,
    response_to_from_cursor_entry_id: null,
    response_to_through_cursor_ts: null,
    response_to_through_cursor_entry_id: null,
    response_to_source_entry_ids: null,
    response_to_count: null,
  };
}

function buildEntry(
  input: StreamEntryInput,
  sessionId: SessionId,
  streamEntryId: StreamEntryId,
): StreamEntry {
  return {
    ...input,
    id: streamEntryId,
    timestamp: 2_000,
    entry_index: 0,
    session_id: sessionId,
    compressed: input.compressed ?? false,
    turn_status: input.turn_status ?? "active",
    sender_entity_id:
      input.sender_entity_id === undefined ? null : (input.sender_entity_id as EntityId),
    reply_target_entity_id: null,
    response_to: input.response_to as StreamEntry["response_to"],
  };
}

function makeHarness(
  options: {
    duplicate?: StreamEntryIndexRecord | null;
    append?: (input: StreamEntryInput) => Promise<StreamEntry>;
    appendMany?: (input: readonly StreamEntryInput[]) => Promise<StreamEntry[]>;
  } = {},
) {
  const sessionId = createSessionId();
  const senderEntityId = createEntityId();
  const audienceEntityId = createEntityId();
  const streamEntryId = createStreamEntryId();
  const session = makeSession({ sessionId, audienceEntityId });
  const sourceMessageKey: StreamSourceMessageKey = {
    source_type: "demo",
    source_external_id: "conversation-1",
    external_message_id: "message-1",
  };
  const appended: StreamEntryInput[] = [];
  const activityEvents: ActivityEventRecordInput[] = [];
  const ensure = vi.fn(() => session);
  const touch = vi.fn();
  const close = vi.fn();
  const append =
    options.append ??
    vi.fn(async (input: StreamEntryInput) => {
      appended.push(input);
      return buildEntry(input, sessionId, streamEntryId);
    });
  const appendMany =
    options.appendMany ??
    vi.fn(async (inputs: readonly StreamEntryInput[]) =>
      inputs.map((input) => buildEntry(input, sessionId, createStreamEntryId())),
    );
  const receiptReadyEvents: Array<{
    sessionId: SessionId;
    pendingAt: number;
    entries: readonly StreamEntry[];
  }> = [];
  const attachmentService = {
    validateAttachments: vi.fn(() => undefined),
    persistParentEntryAttachments: vi.fn(async (): Promise<PersistedParentEntryAttachment[]> => []),
  } satisfies Pick<AttachmentService, "validateAttachments" | "persistParentEntryAttachments">;
  const imagePerceptionService = {
    perceiveAttachment: vi.fn(async () => null),
  } satisfies Pick<ImagePerceptionService, "perceiveAttachment">;
  const entryIndex = {
    lookupBySourceMessageKey: vi.fn(() => options.duplicate ?? null),
    isPoisoned: vi.fn(() => false),
    setReceiptPending: vi.fn(),
  };
  const repairSessionStreamEntryIndex = vi.fn(async () => ({ inserted: 0 }));
  const enqueuer = new MessageEnqueuer({
    sessionsRepository: {
      ensure,
      touch,
    },
    entityRepository: {
      get: vi.fn((id: EntityId) => (id === senderEntityId ? makeEntity(id) : null)),
    },
    activityRepository: {
      record: vi.fn((event: ActivityEventRecordInput) => {
        activityEvents.push(event);
        return {} as ActivityEvent;
      }),
      getByKindAndSource: vi.fn((kind, sourceStreamEntryIds) => {
        const event = activityEvents.find(
          (candidate) =>
            candidate.kind === kind &&
            candidate.sourceStreamEntryIds.length === sourceStreamEntryIds.length &&
            candidate.sourceStreamEntryIds.every(
              (entryId, index) => entryId === sourceStreamEntryIds[index],
            ),
        );

        return event === undefined ? null : ({} as ActivityEvent);
      }),
    },
    entryIndex,
    repairSessionStreamEntryIndex,
    attachmentService,
    imagePerceptionService,
    createReceiptStreamWriter: vi.fn(() => ({
      append,
      appendMany,
      close,
    })),
    onReceiptReady: vi.fn((event) => {
      receiptReadyEvents.push(event);
    }),
    clock: new ManualClock(1_500),
  });

  return {
    enqueuer,
    session,
    sessionId,
    senderEntityId,
    audienceEntityId,
    streamEntryId,
    sourceMessageKey,
    appended,
    appendMany,
    activityEvents,
    attachmentService,
    imagePerceptionService,
    entryIndex,
    repairSessionStreamEntryIndex,
    receiptReadyEvents,
    ensure,
    touch,
    close,
  };
}

const tempDirs: string[] = [];

describe("MessageEnqueuer", () => {
  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("appends a source-keyed user message and records one active pre-turn contact", async () => {
    const harness = makeHarness();

    const result = await harness.enqueuer.enqueueMessage({
      session: {
        session_id: harness.sessionId,
        source_type: "demo",
        source_external_id: "conversation-1",
        label: "Demo",
        audience_label: "Demo room",
        conversation_kind: "thread",
      },
      userMessage: "hello",
      senderEntityId: harness.senderEntityId,
      sourceMessageKey: harness.sourceMessageKey,
      arrivedAt: 4_000,
    });

    expect(result).toEqual({
      status: "enqueued",
      sessionId: harness.sessionId,
      streamEntryId: harness.streamEntryId,
    });
    expect(harness.appended).toEqual([
      {
        kind: "user_msg",
        content: "hello",
        turn_status: "active",
        sender_entity_id: harness.senderEntityId,
        source_message_key: harness.sourceMessageKey,
      },
    ]);
    expect(harness.appended[0]).not.toHaveProperty("turn_id");
    expect(harness.appended[0]).not.toHaveProperty("receipt_pending");
    expect(harness.activityEvents).toEqual([
      {
        kind: "user_contact",
        occurredAt: 4_000,
        sessionId: harness.sessionId,
        turnId: null,
        speakerEntityId: harness.senderEntityId,
        actorEntityId: harness.senderEntityId,
        audienceEntityId: harness.audienceEntityId,
        participantEntityIds: [harness.senderEntityId, harness.audienceEntityId],
        sourceStreamEntryIds: [harness.streamEntryId],
        status: "active",
      },
    ]);
    expect(harness.touch).toHaveBeenCalledWith(harness.sessionId, {
      at: 4_000,
      messageCountDelta: 1,
    });
    expect(harness.close).toHaveBeenCalledTimes(1);
  });

  it("keeps enqueue side effects when a committed append self-repairs the index", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const tempDir = mkdtempSync(join(tmpdir(), "borg-enqueue-index-repair-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const sessionId = createSessionId();
    const senderEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const session = makeSession({ sessionId, audienceEntityId });
    const sourceMessageKey: StreamSourceMessageKey = {
      source_type: "demo",
      source_external_id: "conversation-1",
      external_message_id: "message-1",
    };
    const activityEvents: ActivityEventRecordInput[] = [];
    const clock = new ManualClock(5_000);
    const touch = vi.fn();
    const closeCallbacks: Array<() => void> = [];
    const receiptReadyEvents: Array<{
      sessionId: SessionId;
      pendingAt: number;
      entries: readonly StreamEntry[];
    }> = [];
    let failNextRecord = true;
    const backfillSession = vi.fn((backfillSessionId: SessionId) =>
      entryIndex.backfillSession(backfillSessionId),
    );
    const recordEntry = vi.fn((entry: StreamEntry, byteOffset: number) => {
      if (failNextRecord) {
        failNextRecord = false;
        throw new Error("index update unavailable after fsync");
      }

      entryIndex.recordEntry(entry, byteOffset);
    });
    const createReceiptStreamWriter = vi.fn((writerSessionId: SessionId) => {
      const writer = new StreamWriter({
        dataDir: tempDir,
        sessionId: writerSessionId,
        clock,
        entryIndex: {
          isPoisoned: (poisonedSessionId: SessionId) => entryIndex.isPoisoned(poisonedSessionId),
          markPoisoned: (poisonedSessionId: SessionId) =>
            entryIndex.markPoisoned(poisonedSessionId),
          nextEntryIndex: (nextSessionId: SessionId) => entryIndex.nextEntryIndex(nextSessionId),
          recordEntry,
          backfillSession,
        } as never,
      });
      closeCallbacks.push(() => writer.close());

      return writer;
    });
    const enqueuer = new MessageEnqueuer({
      sessionsRepository: {
        ensure: vi.fn(() => session),
        touch,
      },
      entityRepository: {
        get: vi.fn((id: EntityId) => (id === senderEntityId ? makeEntity(id) : null)),
      },
      activityRepository: {
        record: vi.fn((event: ActivityEventRecordInput) => {
          activityEvents.push(event);
          return {} as ActivityEvent;
        }),
        getByKindAndSource: vi.fn((kind, sourceStreamEntryIds) => {
          const event = activityEvents.find(
            (candidate) =>
              candidate.kind === kind &&
              candidate.sourceStreamEntryIds.length === sourceStreamEntryIds.length &&
              candidate.sourceStreamEntryIds.every(
                (entryId, index) => entryId === sourceStreamEntryIds[index],
              ),
          );

          return event === undefined ? null : ({} as ActivityEvent);
        }),
      },
      entryIndex,
      repairSessionStreamEntryIndex: backfillSession,
      attachmentService: {
        validateAttachments: vi.fn(() => undefined),
        persistParentEntryAttachments: vi.fn(async () => []),
      },
      imagePerceptionService: {
        perceiveAttachment: vi.fn(async () => null),
      },
      createReceiptStreamWriter,
      onReceiptReady: vi.fn((event) => {
        receiptReadyEvents.push(event);
      }),
      clock,
    });

    try {
      const first = await enqueuer.enqueueMessage({
        session: {
          session_id: sessionId,
          source_type: "demo",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "hello after repair",
        senderEntityId,
        sourceMessageKey,
        arrivedAt: 5_000,
      });
      const duplicate = await enqueuer.enqueueMessage({
        session: {
          session_id: sessionId,
          source_type: "demo",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "redelivery after repair",
        senderEntityId,
        sourceMessageKey,
        arrivedAt: 5_001,
      });
      const nextSourceMessageKey: StreamSourceMessageKey = {
        ...sourceMessageKey,
        external_message_id: "message-2",
      };
      const durableUsers = new StreamReader({
        dataDir: tempDir,
        sessionId,
        entryIndex,
      })
        .tail(10)
        .filter((entry) => entry.kind === "user_msg");
      const next = await enqueuer.enqueueMessage({
        session: {
          session_id: sessionId,
          source_type: "demo",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "next message after repair",
        senderEntityId,
        sourceMessageKey: nextSourceMessageKey,
        arrivedAt: 5_002,
      });
      const durableUsersAfterNext = new StreamReader({
        dataDir: tempDir,
        sessionId,
        entryIndex,
      })
        .tail(10)
        .filter((entry) => entry.kind === "user_msg");

      expect(first).toEqual({
        status: "enqueued",
        sessionId,
        streamEntryId: durableUsers[0]?.id,
      });
      expect(duplicate).toEqual({
        status: "duplicate",
        sessionId,
        streamEntryId: first.streamEntryId,
      });
      expect(backfillSession).toHaveBeenCalledTimes(1);
      expect(entryIndex.lookupBySourceMessageKey(sourceMessageKey)?.entry_id).toBe(
        first.streamEntryId,
      );
      expect(durableUsers).toHaveLength(1);
      expect(durableUsers.map((entry) => entry.entry_index)).toEqual([0]);
      expect(next).toMatchObject({
        status: "enqueued",
        sessionId,
      });
      expect(durableUsersAfterNext).toHaveLength(2);
      expect(durableUsersAfterNext.map((entry) => entry.entry_index)).toEqual([0, 1]);
      expect(receiptReadyEvents.flatMap((event) => event.entries.map((entry) => entry.id))).toEqual(
        [first.streamEntryId, next.streamEntryId],
      );
      expect(activityEvents).toHaveLength(2);
      expect(activityEvents[0]).toMatchObject({
        kind: "user_contact",
        sourceStreamEntryIds: [first.streamEntryId],
        status: "active",
      });
      expect(touch).toHaveBeenCalledTimes(2);
      expect(touch).toHaveBeenCalledWith(sessionId, {
        at: 5_000,
        messageCountDelta: 1,
      });
      expect(touch).toHaveBeenCalledWith(sessionId, {
        at: 5_002,
        messageCountDelta: 1,
      });
      expect(createReceiptStreamWriter).toHaveBeenCalledTimes(2);
    } finally {
      consoleError.mockRestore();
      closeCallbacks.forEach((close) => close());
      db.close();
    }
  });

  it("fails closed while poisoned, then reschedules and restores receipt side effects on repaired duplicate", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const tempDir = mkdtempSync(join(tmpdir(), "borg-enqueue-index-poison-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...streamEntryIndexMigrations],
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const sessionId = createSessionId();
    const senderEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const session = makeSession({ sessionId, audienceEntityId });
    const sourceMessageKey: StreamSourceMessageKey = {
      source_type: "demo",
      source_external_id: "conversation-1",
      external_message_id: "message-1",
    };
    const activityEvents: ActivityEventRecordInput[] = [];
    const clock = new ManualClock(5_000);
    const touch = vi.fn();
    const receiptReadyEvents: Array<{
      sessionId: SessionId;
      pendingAt: number;
      entries: readonly StreamEntry[];
    }> = [];
    const closeCallbacks: Array<() => void> = [];
    let failNextRecord = true;
    let repairAvailable = false;
    const guardedBackfillSession = vi.fn((backfillSessionId: SessionId) => {
      if (!repairAvailable) {
        throw new Error("index repair unavailable");
      }

      return entryIndex.backfillSession(backfillSessionId);
    });
    const recordEntry = vi.fn((entry: StreamEntry, byteOffset: number) => {
      if (failNextRecord) {
        failNextRecord = false;
        throw new Error("index update unavailable after fsync");
      }

      entryIndex.recordEntry(entry, byteOffset);
    });
    const sharedEntryIndex = {
      lookupBySourceMessageKey: (key: StreamSourceMessageKey) =>
        entryIndex.lookupBySourceMessageKey(key),
      isPoisoned: (poisonedSessionId: SessionId) => entryIndex.isPoisoned(poisonedSessionId),
      setReceiptPending: (entryId: StreamEntryId, pending: boolean) =>
        entryIndex.setReceiptPending(entryId, pending),
      backfillSession: guardedBackfillSession,
    };
    const createReceiptStreamWriter = vi.fn((writerSessionId: SessionId) => {
      const writer = new StreamWriter({
        dataDir: tempDir,
        sessionId: writerSessionId,
        clock,
        logger: { error: vi.fn() },
        entryIndex: {
          ...sharedEntryIndex,
          markPoisoned: (poisonedSessionId: SessionId) =>
            entryIndex.markPoisoned(poisonedSessionId),
          nextEntryIndex: (nextSessionId: SessionId) => entryIndex.nextEntryIndex(nextSessionId),
          recordEntry,
        } as never,
      });
      closeCallbacks.push(() => writer.close());

      return writer;
    });
    const enqueuer = new MessageEnqueuer({
      sessionsRepository: {
        ensure: vi.fn(() => session),
        touch,
      },
      entityRepository: {
        get: vi.fn((id: EntityId) => (id === senderEntityId ? makeEntity(id) : null)),
      },
      activityRepository: {
        record: vi.fn((event: ActivityEventRecordInput) => {
          activityEvents.push(event);
          return {} as ActivityEvent;
        }),
        getByKindAndSource: vi.fn((kind, sourceStreamEntryIds) => {
          const event = activityEvents.find(
            (candidate) =>
              candidate.kind === kind &&
              candidate.sourceStreamEntryIds.length === sourceStreamEntryIds.length &&
              candidate.sourceStreamEntryIds.every(
                (entryId, index) => entryId === sourceStreamEntryIds[index],
              ),
          );

          return event === undefined ? null : ({} as ActivityEvent);
        }),
      },
      entryIndex: sharedEntryIndex,
      repairSessionStreamEntryIndex: guardedBackfillSession,
      attachmentService: {
        validateAttachments: vi.fn(() => undefined),
        persistParentEntryAttachments: vi.fn(async () => []),
      },
      imagePerceptionService: {
        perceiveAttachment: vi.fn(async () => null),
      },
      createReceiptStreamWriter,
      isDuplicatePendingResponse: vi.fn(
        (record) => record.kind === "user_msg" && record.turn_id === null,
      ),
      onReceiptReady: vi.fn((event) => {
        receiptReadyEvents.push(event);
      }),
      clock,
    });
    const enqueueInput = {
      session: {
        session_id: sessionId,
        source_type: "demo" as const,
        source_external_id: "conversation-1",
        label: "Demo",
        audience_label: "Demo room",
        conversation_kind: "thread" as const,
      },
      userMessage: "hello before poison",
      senderEntityId,
      sourceMessageKey,
      arrivedAt: 5_000,
    };

    try {
      await expect(enqueuer.enqueueMessage(enqueueInput)).rejects.toMatchObject({
        code: "STREAM_INDEX_POISONED",
      });
      expect(entryIndex.isPoisoned(sessionId)).toBe(true);
      expect(new StreamReader({ dataDir: tempDir, sessionId }).tail(10)).toHaveLength(1);

      await expect(enqueuer.enqueueMessage(enqueueInput)).rejects.toMatchObject({
        code: "STREAM_INDEX_POISONED",
      });
      expect(new StreamReader({ dataDir: tempDir, sessionId }).tail(10)).toHaveLength(1);
      expect(createReceiptStreamWriter).toHaveBeenCalledTimes(1);

      repairAvailable = true;
      await expect(enqueuer.enqueueMessage(enqueueInput)).resolves.toMatchObject({
        status: "duplicate",
        sessionId,
      });

      const durableUsers = new StreamReader({
        dataDir: tempDir,
        sessionId,
        entryIndex,
      })
        .tail(10)
        .filter((entry) => entry.kind === "user_msg");

      expect(entryIndex.isPoisoned(sessionId)).toBe(false);
      expect(durableUsers).toHaveLength(1);
      expect(entryIndex.lookupBySourceMessageKey(sourceMessageKey)?.entry_id).toBe(
        durableUsers[0]?.id,
      );
      expect(createReceiptStreamWriter).toHaveBeenCalledTimes(1);
      expect(receiptReadyEvents.map((event) => event.sessionId)).toEqual([sessionId]);
      expect(receiptReadyEvents.map((event) => event.pendingAt)).toEqual([
        durableUsers[0]?.timestamp,
      ]);
      expect(activityEvents).toHaveLength(1);
      expect(activityEvents[0]).toMatchObject({
        kind: "user_contact",
        sourceStreamEntryIds: [durableUsers[0]?.id],
        status: "active",
      });
      expect(touch).toHaveBeenCalledTimes(1);
      expect(touch).toHaveBeenCalledWith(sessionId, {
        at: 5_000,
        messageCountDelta: 1,
      });
    } finally {
      consoleError.mockRestore();
      closeCallbacks.forEach((close) => close());
      db.close();
    }
  });

  it("returns completed duplicates without appending, recording contact, or touching message count", async () => {
    const existingSessionId = createSessionId();
    const existingEntryId = createStreamEntryId();
    const harness = makeHarness({
      duplicate: makeIndexRecord({ entryId: existingEntryId, sessionId: existingSessionId }),
    });
    harness.activityEvents.push({
      kind: "user_contact",
      occurredAt: 3_000,
      sessionId: existingSessionId,
      turnId: null,
      speakerEntityId: harness.senderEntityId,
      actorEntityId: harness.senderEntityId,
      audienceEntityId: harness.audienceEntityId,
      participantEntityIds: [harness.senderEntityId, harness.audienceEntityId],
      sourceStreamEntryIds: [existingEntryId],
      status: "active",
    });

    const result = await harness.enqueuer.enqueueMessage({
      session: {
        session_id: harness.sessionId,
        source_type: "demo",
        source_external_id: "conversation-1",
        label: "Demo",
        audience_label: "Demo room",
        conversation_kind: "thread",
      },
      userMessage: "hello again",
      senderEntityId: harness.senderEntityId,
      sourceMessageKey: harness.sourceMessageKey,
      arrivedAt: 4_000,
    });

    expect(result).toEqual({
      status: "duplicate",
      sessionId: existingSessionId,
      streamEntryId: existingEntryId,
    });
    expect(harness.appended).toEqual([]);
    expect(harness.activityEvents).toHaveLength(1);
    expect(harness.attachmentService.persistParentEntryAttachments).not.toHaveBeenCalled();
    expect(harness.imagePerceptionService.perceiveAttachment).not.toHaveBeenCalled();
    expect(harness.touch).not.toHaveBeenCalled();
    expect(harness.close).not.toHaveBeenCalled();
  });

  it("does not notify receipt readiness until attachment persistence has resolved", async () => {
    const harness = makeHarness();
    let releasePersistence: ((value: PersistedParentEntryAttachment[]) => void) | undefined;
    harness.attachmentService.persistParentEntryAttachments.mockImplementation(
      () =>
        new Promise<PersistedParentEntryAttachment[]>((resolve) => {
          releasePersistence = resolve;
        }),
    );
    const enqueue = harness.enqueuer.enqueueMessage({
      session: {
        session_id: harness.sessionId,
        source_type: "demo",
        source_external_id: "conversation-1",
        label: "Demo",
        audience_label: "Demo room",
        conversation_kind: "thread",
      },
      userMessage: "hello with image",
      senderEntityId: harness.senderEntityId,
      sourceMessageKey: harness.sourceMessageKey,
      attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
    });

    for (let attempt = 0; attempt < 10 && releasePersistence === undefined; attempt += 1) {
      await Promise.resolve();
    }

    expect(harness.attachmentService.persistParentEntryAttachments).toHaveBeenCalledTimes(1);
    expect(harness.receiptReadyEvents).toEqual([]);
    expect(harness.touch).not.toHaveBeenCalled();
    expect(harness.appended[0]).toMatchObject({ receipt_pending: true });
    expect(harness.entryIndex.setReceiptPending).not.toHaveBeenCalled();

    releasePersistence?.([]);

    await expect(enqueue).resolves.toMatchObject({
      status: "enqueued",
      streamEntryId: harness.streamEntryId,
    });
    expect(harness.receiptReadyEvents).toHaveLength(1);
    expect(harness.receiptReadyEvents[0]?.entries.map((entry) => entry.id)).toEqual([
      harness.streamEntryId,
    ]);
    expect(harness.entryIndex.setReceiptPending).toHaveBeenCalledWith(harness.streamEntryId, false);
  });

  it("completes missing attachment and perception side effects on a pending duplicate", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-enqueue-attachment-duplicate-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(streamEntryIndexMigrations, attachmentMigrations),
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const repository = new AttachmentRepository(db);
    const service = new AttachmentService({
      repository,
      blobStore: new AttachmentBlobStore(tempDir),
      config: {
        maxBytesPerImage: 1024,
        maxWidth: 64,
        maxHeight: 64,
        maxImagesPerTurn: 4,
      },
      entryIndex,
    });
    const sessionId = createSessionId();
    const senderEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const session = makeSession({ sessionId, audienceEntityId });
    const sourceMessageKey: StreamSourceMessageKey = {
      source_type: "demo",
      source_external_id: "conversation-1",
      external_message_id: "message-1",
    };
    const clock = new ManualClock(7_000);
    const seedWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId,
      clock,
      entryIndex,
    });
    const userEntry = await seedWriter.append({
      kind: "user_msg",
      content: "committed before crash",
      sender_entity_id: senderEntityId,
      source_message_key: sourceMessageKey,
      receipt_pending: true,
    });
    seedWriter.close();
    const duplicate = entryIndex.lookupBySourceMessageKey(sourceMessageKey);
    expect(duplicate).not.toBeNull();
    const activityEvents: ActivityEventRecordInput[] = [];
    const touch = vi.fn();
    const receiptReadyEvents: Array<{
      sessionId: SessionId;
      pendingAt: number;
      entries: readonly StreamEntry[];
    }> = [];
    const perceivedAttachmentIds: string[] = [];
    const closeCallbacks: Array<() => void> = [];
    const createReceiptStreamWriter = vi.fn((writerSessionId: SessionId) => {
      const writer = new StreamWriter({
        dataDir: tempDir,
        sessionId: writerSessionId,
        clock,
        entryIndex,
      });
      closeCallbacks.push(() => writer.close());

      return writer;
    });
    const enqueuer = new MessageEnqueuer({
      sessionsRepository: {
        ensure: vi.fn(() => session),
        touch,
      },
      entityRepository: {
        get: vi.fn((id: EntityId) => (id === senderEntityId ? makeEntity(id) : null)),
      },
      activityRepository: {
        record: vi.fn((event: ActivityEventRecordInput) => {
          activityEvents.push(event);
          return {} as ActivityEvent;
        }),
        getByKindAndSource: vi.fn((kind, sourceStreamEntryIds) => {
          const event = activityEvents.find(
            (candidate) =>
              candidate.kind === kind &&
              candidate.sourceStreamEntryIds.length === sourceStreamEntryIds.length &&
              candidate.sourceStreamEntryIds.every(
                (entryId, index) => entryId === sourceStreamEntryIds[index],
              ),
          );

          return event === undefined ? null : ({} as ActivityEvent);
        }),
      },
      attachmentService: service,
      imagePerceptionService: {
        perceiveAttachment: vi.fn(async ({ attachmentId }) => {
          perceivedAttachmentIds.push(attachmentId);
          return null;
        }),
      },
      entryIndex: {
        lookupBySourceMessageKey: vi.fn(() => duplicate),
        isPoisoned: vi.fn(() => false),
        setReceiptPending: vi.fn((entryId: StreamEntryId, pending: boolean) =>
          entryIndex.setReceiptPending(entryId, pending),
        ),
      },
      repairSessionStreamEntryIndex: vi.fn(async () => ({ inserted: 0 })),
      createReceiptStreamWriter,
      isDuplicatePendingResponse: vi.fn(() => false),
      onReceiptReady: vi.fn((event) => {
        receiptReadyEvents.push(event);
      }),
      clock,
    });

    try {
      await expect(
        enqueuer.enqueueMessage({
          session: {
            session_id: sessionId,
            source_type: "demo",
            source_external_id: "conversation-1",
            label: "Demo",
            audience_label: "Demo room",
            conversation_kind: "thread",
          },
          userMessage: "redelivery",
          senderEntityId,
          sourceMessageKey,
          arrivedAt: 7_000,
          attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
        }),
      ).resolves.toEqual({
        status: "duplicate",
        sessionId,
        streamEntryId: userEntry.id,
      });

      const records = repository.listByParentEntry(userEntry.id);
      const imageEntries = new StreamReader({
        dataDir: tempDir,
        sessionId,
        entryIndex,
      })
        .tail(10)
        .filter((entry) => entry.kind === "user_image_attachment");

      expect(records).toHaveLength(1);
      expect(records[0]).toMatchObject({
        media_type: "image/gif",
        parent_entry_id: userEntry.id,
        parent_turn_id: null,
        stream_entry_id: imageEntries[0]?.id,
        active: true,
      });
      expect(imageEntries).toHaveLength(1);
      expect(imageEntries[0]).not.toHaveProperty("turn_id");
      expect(entryIndex.lookup(userEntry.id)?.receipt_pending).toBe(false);
      expect(perceivedAttachmentIds).toEqual([records[0]?.attachment_id]);
      expect(activityEvents).toHaveLength(1);
      expect(activityEvents[0]).toMatchObject({
        kind: "user_contact",
        sourceStreamEntryIds: [userEntry.id],
        status: "active",
      });
      expect(touch).toHaveBeenCalledWith(sessionId, {
        at: 7_000,
        messageCountDelta: 1,
      });
      expect(receiptReadyEvents).toHaveLength(1);
      expect(receiptReadyEvents[0]?.entries.map((entry) => entry.id)).toEqual([
        imageEntries[0]?.id,
      ]);
    } finally {
      closeCallbacks.forEach((close) => close());
      db.close();
    }
  });

  it("rejects mismatched or missing session source fields before ensure", async () => {
    const mismatchHarness = makeHarness();

    await expect(
      mismatchHarness.enqueuer.enqueueMessage({
        session: {
          session_id: mismatchHarness.sessionId,
          source_type: "slack",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "hello",
        senderEntityId: mismatchHarness.senderEntityId,
        sourceMessageKey: mismatchHarness.sourceMessageKey,
      }),
    ).rejects.toMatchObject({
      code: "ENQUEUE_SOURCE_MESSAGE_KEY_MISMATCH",
    });
    expect(mismatchHarness.ensure).not.toHaveBeenCalled();
    expect(mismatchHarness.appended).toEqual([]);

    const missingExternalIdHarness = makeHarness();

    await expect(
      missingExternalIdHarness.enqueuer.enqueueMessage({
        session: {
          session_id: missingExternalIdHarness.sessionId,
          source_type: "demo",
          source_external_id: null,
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        } as never,
        userMessage: "hello",
        senderEntityId: missingExternalIdHarness.senderEntityId,
        sourceMessageKey: missingExternalIdHarness.sourceMessageKey,
      }),
    ).rejects.toMatchObject({
      code: "ENQUEUE_SOURCE_MESSAGE_KEY_MISMATCH",
    });
    expect(missingExternalIdHarness.ensure).not.toHaveBeenCalled();
    expect(missingExternalIdHarness.appended).toEqual([]);
  });

  it("rejects non-finite or non-integer arrivedAt before append", async () => {
    const harness = makeHarness();

    await expect(
      harness.enqueuer.enqueueMessage({
        session: {
          session_id: harness.sessionId,
          source_type: "demo",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "hello",
        senderEntityId: harness.senderEntityId,
        sourceMessageKey: harness.sourceMessageKey,
        arrivedAt: Number.POSITIVE_INFINITY,
      }),
    ).rejects.toMatchObject({
      code: "ENQUEUE_ARRIVED_AT_INVALID",
    });
    expect(harness.ensure).not.toHaveBeenCalled();
    expect(harness.appended).toEqual([]);

    await expect(
      harness.enqueuer.enqueueMessage({
        session: {
          session_id: harness.sessionId,
          source_type: "demo",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "hello",
        senderEntityId: harness.senderEntityId,
        sourceMessageKey: harness.sourceMessageKey,
        arrivedAt: 4_000.5,
      }),
    ).rejects.toMatchObject({
      code: "ENQUEUE_ARRIVED_AT_INVALID",
    });
    expect(harness.ensure).not.toHaveBeenCalled();
    expect(harness.appended).toEqual([]);
  });

  it("rejects unknown senders and invalid attachments before append", async () => {
    const unknownSenderHarness = makeHarness();

    await expect(
      unknownSenderHarness.enqueuer.enqueueMessage({
        session: {
          session_id: unknownSenderHarness.sessionId,
          source_type: "demo",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "hello",
        senderEntityId: createEntityId(),
        sourceMessageKey: unknownSenderHarness.sourceMessageKey,
      }),
    ).rejects.toMatchObject({
      code: "ENQUEUE_SENDER_UNKNOWN",
    });
    expect(unknownSenderHarness.appended).toEqual([]);

    const attachmentHarness = makeHarness();
    attachmentHarness.attachmentService.validateAttachments.mockImplementation(() => {
      throw new AttachmentError("Unsupported image media type: image/bmp", {
        code: "ATTACHMENT_UNSUPPORTED_MEDIA_TYPE",
      });
    });

    await expect(
      attachmentHarness.enqueuer.enqueueMessage({
        session: {
          session_id: attachmentHarness.sessionId,
          source_type: "demo",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "hello",
        senderEntityId: attachmentHarness.senderEntityId,
        sourceMessageKey: attachmentHarness.sourceMessageKey,
        attachments: [
          {
            mediaType: "image/bmp",
            bytes: Uint8Array.of(1, 2, 3),
          } as unknown as TurnInputAttachment,
        ],
      }),
    ).rejects.toMatchObject({
      code: "ATTACHMENT_UNSUPPORTED_MEDIA_TYPE",
    });
    expect(attachmentHarness.appended).toEqual([]);
  });

  it("resolves only after the append promise resolves", async () => {
    let releaseAppend: ((entry: StreamEntry) => void) | undefined;
    const appendStarted: StreamEntryInput[] = [];
    const harness = makeHarness({
      append: vi.fn(
        (input: StreamEntryInput) =>
          new Promise<StreamEntry>((resolve) => {
            appendStarted.push(input);
            releaseAppend = resolve;
          }),
      ),
    });
    let resolved = false;
    const enqueue = harness.enqueuer
      .enqueueMessage({
        session: {
          session_id: harness.sessionId,
          source_type: "demo",
          source_external_id: "conversation-1",
          label: "Demo",
          audience_label: "Demo room",
          conversation_kind: "thread",
        },
        userMessage: "hello",
        senderEntityId: harness.senderEntityId,
        sourceMessageKey: harness.sourceMessageKey,
      })
      .then((result) => {
        resolved = true;
        return result;
      });

    await Promise.resolve();
    await Promise.resolve();

    expect(appendStarted).toHaveLength(1);
    expect(resolved).toBe(false);
    expect(harness.touch).not.toHaveBeenCalled();

    releaseAppend?.(
      buildEntry(appendStarted[0] as StreamEntryInput, harness.sessionId, harness.streamEntryId),
    );
    await expect(enqueue).resolves.toMatchObject({
      status: "enqueued",
      streamEntryId: harness.streamEntryId,
    });
    expect(resolved).toBe(true);
    expect(harness.touch).toHaveBeenCalledTimes(1);
  });
});
