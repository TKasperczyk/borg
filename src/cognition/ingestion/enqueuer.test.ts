import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { ActivityEvent, ActivityEventRecordInput } from "../../memory/activity/index.js";
import type { EntityRecord } from "../../memory/commitments/index.js";
import type { SessionRecord } from "../../sessions/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
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
import {
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { MessageEnqueuer } from "./enqueuer.js";

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
    entryIndex: {
      lookupBySourceMessageKey: vi.fn(() => options.duplicate ?? null),
      isPoisoned: vi.fn(() => false),
      backfillSession: vi.fn(async () => ({ inserted: 0 })),
    },
    createStreamWriter: vi.fn(() => ({
      append,
      close,
    })),
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
    activityEvents,
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
    const observedAppends: StreamEntry[] = [];
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
    const createStreamWriter = vi.fn((writerSessionId: SessionId) => {
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
        onAppend: (entries) => {
          observedAppends.push(...entries);
        },
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
      createStreamWriter,
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
      expect(observedAppends.map((entry) => entry.id)).toEqual([
        first.streamEntryId,
        next.streamEntryId,
      ]);
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
      expect(createStreamWriter).toHaveBeenCalledTimes(2);
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
    const pendingDuplicateSchedules: StreamEntryIndexRecord[] = [];
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
      backfillSession: guardedBackfillSession,
    };
    const createStreamWriter = vi.fn((writerSessionId: SessionId) => {
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
      createStreamWriter,
      isDuplicatePendingResponse: vi.fn(
        (record) => record.kind === "user_msg" && record.turn_id === null,
      ),
      onPendingDuplicate: vi.fn((record) => {
        pendingDuplicateSchedules.push(record);
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
      expect(createStreamWriter).toHaveBeenCalledTimes(1);

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
      expect(createStreamWriter).toHaveBeenCalledTimes(1);
      expect(pendingDuplicateSchedules.map((record) => record.entry_id)).toEqual([
        durableUsers[0]?.id,
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

  it("returns duplicates without appending, recording contact, or touching message count", async () => {
    const existingSessionId = createSessionId();
    const existingEntryId = createStreamEntryId();
    const harness = makeHarness({
      duplicate: makeIndexRecord({ entryId: existingEntryId, sessionId: existingSessionId }),
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
    expect(harness.activityEvents).toEqual([]);
    expect(harness.touch).not.toHaveBeenCalled();
    expect(harness.close).not.toHaveBeenCalled();
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

  it("rejects unknown senders and non-empty attachments before append", async () => {
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
        attachments: [{}],
      }),
    ).rejects.toMatchObject({
      code: "ENQUEUE_ATTACHMENTS_UNSUPPORTED",
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
