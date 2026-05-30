import { afterEach, describe, expect, it, vi } from "vitest";

import { AttachmentBlobStore, type AttachmentService } from "../../attachments/index.js";
import type { MessageConnector } from "../../outbound/index.js";
import {
  ABORTED_TURN_EVENT,
  QUARANTINED_USER_ENTRY_EVENT,
  StreamWriter,
  type StreamEntryIndexRepository,
} from "../../stream/index.js";
import type { StreamCursor, StreamEntry, StreamResponseTo } from "../../stream/index.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import {
  Borg,
  FakeLLMClient,
  ManualClock,
  ScriptedEmbeddingClient,
  borgInternals,
  createEmptyReflectionResponse,
  createEmitAnswerResponse,
  createSessionId,
  createTestConfig,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

const TERMINAL_KINDS = new Set<StreamEntry["kind"]>([
  "agent_msg",
  "agent_suppressed",
  "agent_observed",
]);
const GIF_1X1 = Uint8Array.from([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00]);
const GIF_1X1_ALT = Uint8Array.from([
  0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x21,
]);

type HarnessBorgInternals = {
  deps: {
    chatResponseWatermarkCoordinator: {
      getWatermark(sessionId: SessionId): StreamCursor | null;
      advanceThrough(sessionId: SessionId, cursor: StreamCursor): unknown;
    };
    createStreamWriter(sessionId: SessionId): StreamWriter;
    entryIndex: StreamEntryIndexRepository;
    messageEnqueuer: {
      options: {
        attachmentService: {
          persistParentEntryAttachments: (
            input: Parameters<AttachmentService["persistParentEntryAttachments"]>[0],
          ) => ReturnType<AttachmentService["persistParentEntryAttachments"]>;
        };
      };
    };
    outboundDelivery: {
      deliver(input: {
        session: NonNullable<ReturnType<Borg["sessions"]["get"]>>;
        message: {
          content: string;
          streamInput?: Omit<Parameters<StreamWriter["append"]>[0], "kind" | "content">;
        };
      }): Promise<unknown>;
    };
    turnOrchestrator: {
      options: {
        createStreamWriter(sessionId: SessionId): StreamWriter;
      };
    };
  };
};

function finalizerRequests(llm: FakeLLMClient) {
  return llm.requests.filter(
    (request) => request.budget === "cognition-system-1" || request.budget === "cognition-system-2",
  );
}

function imagePerceptionRequests(llm: FakeLLMClient) {
  return llm.requests.filter((request) => request.budget === "image-perception");
}

function requestTextMessages(request: ReturnType<typeof finalizerRequests>[number] | undefined) {
  return request?.messages.map((message) => message.content) ?? [];
}

function createImagePerceptionResponse(
  input: {
    caption?: string;
    visibleText?: readonly string[];
    searchTerms?: readonly string[];
  } = {},
) {
  return {
    messageBlocks: [
      {
        type: "tool_use" as const,
        id: "toolu_image",
        name: "EmitImagePerception",
        input: {
          caption: input.caption ?? "small test image",
          image_kind: "other",
          visible_text: [...(input.visibleText ?? [])],
          objects: [],
          people_or_roles: [],
          scene: "test fixture",
          colors_and_visual_attributes: [],
          spatial_relationships: [],
          possible_user_relevant_details: [],
          search_terms: [...(input.searchTerms ?? ["test image"])],
          uncertainties: [],
        },
      },
    ],
    input_tokens: 4,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
  };
}

function deferred<T = void>() {
  let resolve: (value: T | PromiseLike<T>) => void = () => {};
  let reject: (reason?: unknown) => void = () => {};
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return {
    promise,
    resolve,
    reject,
  };
}

function terminalEntries(borg: Borg, sessionId: SessionId): StreamEntry[] {
  return borg.stream
    .tail(200, { session: sessionId })
    .filter((entry) => TERMINAL_KINDS.has(entry.kind));
}

function abortedTurnEntries(borg: Borg, sessionId: SessionId): StreamEntry[] {
  return borg.stream
    .tail(200, { session: sessionId })
    .filter(
      (entry) =>
        entry.kind === "internal_event" &&
        entry.content !== null &&
        typeof entry.content === "object" &&
        !Array.isArray(entry.content) &&
        (entry.content as { event?: unknown }).event === ABORTED_TURN_EVENT,
    );
}

function userEntries(borg: Borg, sessionId: SessionId): StreamEntry[] {
  return borg.stream.tail(200, { session: sessionId }).filter((entry) => entry.kind === "user_msg");
}

function imageAttachmentEntries(borg: Borg, sessionId: SessionId): StreamEntry[] {
  return borg.stream
    .tail(200, { session: sessionId })
    .filter((entry) => entry.kind === "user_image_attachment");
}

async function waitForTerminalCount(input: {
  borg: Borg;
  sessionId: SessionId;
  count: number;
}): Promise<void> {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (terminalEntries(input.borg, input.sessionId).length === input.count) {
      return;
    }

    await new Promise((resolve) => setTimeout(resolve, 5));
  }

  expect(terminalEntries(input.borg, input.sessionId)).toHaveLength(input.count);
}

function cursorFor(entry: Pick<StreamEntry, "id" | "timestamp">): StreamCursor {
  return {
    ts: entry.timestamp,
    entryId: entry.id,
  };
}

function responseToFor(
  entries: readonly Pick<StreamEntry, "id" | "timestamp">[],
  fromCursorExclusive: StreamCursor | null = null,
): StreamResponseTo {
  const last = entries[entries.length - 1];

  if (last === undefined) {
    throw new Error("response_to requires at least one source entry");
  }

  return {
    kind: "stream_backlog",
    from_cursor_exclusive: fromCursorExclusive,
    through_cursor_inclusive: cursorFor(last),
    source_entry_ids: entries.map((entry) => entry.id),
    count: entries.length,
  };
}

async function appendQuarantineMarker(input: {
  borg: Borg;
  sessionId: SessionId;
  entries: readonly Pick<StreamEntry, "id">[];
}): Promise<void> {
  const source = input.entries[0];

  await input.borg.stream.append(
    {
      kind: "internal_event",
      content: {
        event: QUARANTINED_USER_ENTRY_EVENT,
        source_stream_entry_id: source?.id ?? null,
        cited_stream_entry_ids: input.entries.map((entry) => entry.id),
      },
    },
    { session: input.sessionId },
  );
}

async function openHarness(input: {
  tempDir: string;
  clock: ManualClock;
  llm: FakeLLMClient;
  outboundConnectors?: readonly MessageConnector[];
}): Promise<Borg> {
  return Borg.open({
    config: createTestConfig({
      dataDir: input.tempDir,
      perception: {
        llmEnabled: false,
      },
      affective: {
        llmEnabled: false,
      },
      embedding: {
        baseUrl: "http://localhost:1234/v1",
        apiKey: "test",
        model: "fake-embed",
        dims: 4,
      },
      anthropic: {
        auth: "api-key",
        apiKey: "test",
        models: {
          cognition: "sonnet",
          background: "haiku",
          extraction: "haiku",
          recallExpansion: "haiku",
        },
      },
    }),
    clock: input.clock,
    embeddingDimensions: 4,
    embeddingClient: new ScriptedEmbeddingClient(),
    llmClient: input.llm,
    liveExtraction: false,
    outboundConnectors: input.outboundConnectors,
  });
}

function createConversationHarness(input: {
  externalId: string;
  label?: string;
  sessionId?: SessionId;
}) {
  return {
    session_id: input.sessionId ?? createSessionId(),
    source_type: "demo" as const,
    source_external_id: input.externalId,
    label: input.label ?? input.externalId,
    audience_label: input.label ?? input.externalId,
    conversation_kind: "thread" as const,
  };
}

async function enqueueOne(input: {
  borg: Borg;
  clock: ManualClock;
  session: ReturnType<typeof createConversationHarness>;
  senderEntityId: ReturnType<Borg["entities"]["resolve"]>;
  externalMessageId?: string;
  text?: string;
}) {
  await input.borg.enqueueMessage({
    session: input.session,
    userMessage: input.text ?? "queued message",
    senderEntityId: input.senderEntityId,
    sourceMessageKey: {
      source_type: "demo",
      source_external_id: input.session.source_external_id,
      external_message_id: input.externalMessageId ?? "message-1",
    },
    arrivedAt: input.clock.now(),
  });
}

function installTerminalIndexFailure(input: {
  internal: HarnessBorgInternals;
  tempDir: string;
  clock: ManualClock;
  repair: "succeed" | "fail";
}): { backfillCalls: () => number } {
  const entryIndex = input.internal.deps.entryIndex;
  let failedTerminalRecord = false;
  let backfillCallCount = 0;

  input.internal.deps.turnOrchestrator.options.createStreamWriter = (sessionId) =>
    new StreamWriter({
      dataDir: input.tempDir,
      sessionId,
      clock: input.clock,
      entryIndex: {
        isPoisoned: (poisonedSessionId: SessionId) => entryIndex.isPoisoned(poisonedSessionId),
        markPoisoned: (poisonedSessionId: SessionId) => entryIndex.markPoisoned(poisonedSessionId),
        nextEntryIndex: (nextSessionId: SessionId) => entryIndex.nextEntryIndex(nextSessionId),
        recordEntry: (entry: StreamEntry, byteOffset: number) => {
          if (!failedTerminalRecord && TERMINAL_KINDS.has(entry.kind)) {
            failedTerminalRecord = true;
            throw new Error("test terminal index update failure");
          }

          entryIndex.recordEntry(entry, byteOffset);
        },
        backfillSession: (backfillSessionId: SessionId) => {
          backfillCallCount += 1;

          if (input.repair === "fail") {
            throw new Error("test terminal index repair failure");
          }

          return entryIndex.backfillSession(backfillSessionId);
        },
      } as never,
    });

  return {
    backfillCalls: () => backfillCallCount,
  };
}

describe("async ingest Borg E2E crash and duplicate matrix", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("drains an enqueued message after closing and reopening before the first tick", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-reopen-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_000_000);
    const session = createConversationHarness({ externalId: "restart-queue" });
    const enqueueLlm = new FakeLLMClient();
    let borg = await openHarness({ tempDir, clock, llm: enqueueLlm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await enqueueOne({ borg, clock, session, senderEntityId });
      expect(userEntries(borg, session.session_id)).toHaveLength(1);
      expect(terminalEntries(borg, session.session_id)).toHaveLength(0);
      await borg.close();

      const llm = new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("answer after restart"),
          createEmptyReflectionResponse(),
        ],
      });
      borg = await openHarness({ tempDir, clock, llm });

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });

      const users = userEntries(borg, session.session_id);
      const terminals = terminalEntries(borg, session.session_id);
      const stamp = responseToFor(users);
      const internal = borgInternals<HarnessBorgInternals>(borg);

      expect(users).toHaveLength(1);
      expect(terminals).toHaveLength(1);
      expect(terminals[0]).toMatchObject({
        kind: "agent_msg",
        content: "answer after restart",
        response_to: stamp,
      });
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toEqual(stamp.through_cursor_inclusive);
      expect(finalizerRequests(llm)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("startup scan crash-replays a fully received image message without re-running perception", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-image-reopen-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_000_500);
    const session = createConversationHarness({ externalId: "image-restart-queue" });
    const enqueueLlm = new FakeLLMClient({
      responses: [
        createImagePerceptionResponse({
          caption: "receipt-time image caption",
          visibleText: ["receipt visible text"],
          searchTerms: ["receipt image"],
        }),
      ],
    });
    let borg = await openHarness({ tempDir, clock, llm: enqueueLlm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await borg.enqueueMessage({
        session,
        userMessage: "queued image before crash",
        senderEntityId,
        sourceMessageKey: {
          source_type: "demo",
          source_external_id: session.source_external_id,
          external_message_id: "image-before-drain",
        },
        arrivedAt: clock.now(),
        attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
      });

      const [source] = userEntries(borg, session.session_id);
      const internal = borgInternals<HarnessBorgInternals>(borg);

      if (source === undefined) {
        throw new Error("expected queued image source entry");
      }

      expect(imageAttachmentEntries(borg, session.session_id)).toHaveLength(1);
      expect(internal.deps.entryIndex.lookup(source.id)?.receipt_pending).toBe(false);
      expect(terminalEntries(borg, session.session_id)).toHaveLength(0);
      expect(imagePerceptionRequests(enqueueLlm)).toHaveLength(1);

      await borg.close();

      const replayLlm = new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("answer after image restart"),
          createEmptyReflectionResponse(),
        ],
      });
      borg = await openHarness({ tempDir, clock, llm: replayLlm });
      borg.inbox.catchUp.start();
      await waitForTerminalCount({ borg, sessionId: session.session_id, count: 1 });

      const terminals = terminalEntries(borg, session.session_id);
      const stamp = responseToFor([source]);
      const reopenedInternal = borgInternals<HarnessBorgInternals>(borg);
      const finalizerPrompt = requestTextMessages(finalizerRequests(replayLlm).at(0)).join("\n");

      expect(terminals).toHaveLength(1);
      expect(terminals[0]).toMatchObject({
        kind: "agent_msg",
        content: "answer after image restart",
        response_to: stamp,
      });
      expect(finalizerPrompt).toContain('<attachments count="1">');
      expect(finalizerPrompt).toContain("receipt-time image caption");
      expect(finalizerPrompt).toContain("receipt visible text");
      expect(
        reopenedInternal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toEqual(stamp.through_cursor_inclusive);
      expect(finalizerRequests(replayLlm)).toHaveLength(1);
      expect(imagePerceptionRequests(replayLlm)).toHaveLength(0);
      expect(imagePerceptionRequests(enqueueLlm)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("does not drain an image enqueue until receipt persistence clears readiness", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-receipt-window-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_001_000);
    const session = createConversationHarness({ externalId: "receipt-window" });
    const llm = new FakeLLMClient({
      responses: [
        createImagePerceptionResponse(),
        createEmitAnswerResponse("answer after image receipt"),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openHarness({ tempDir, clock, llm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");
      const internal = borgInternals<HarnessBorgInternals>(borg);
      const originalPersist =
        internal.deps.messageEnqueuer.options.attachmentService.persistParentEntryAttachments.bind(
          internal.deps.messageEnqueuer.options.attachmentService,
        );
      const persistStarted = deferred<void>();
      const releasePersistence = deferred<void>();

      internal.deps.messageEnqueuer.options.attachmentService.persistParentEntryAttachments = vi.fn(
        async (input) => {
          persistStarted.resolve();
          await releasePersistence.promise;
          return originalPersist(input);
        },
      );

      borg.inbox.catchUp.start();
      const enqueue = borg.enqueueMessage({
        session,
        userMessage: "queued image message",
        senderEntityId,
        sourceMessageKey: {
          source_type: "demo",
          source_external_id: session.source_external_id,
          external_message_id: "image-message",
        },
        arrivedAt: clock.now(),
        attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
      });

      await persistStarted.promise;
      const [source] = userEntries(borg, session.session_id);

      if (source === undefined) {
        throw new Error("expected queued source entry");
      }

      expect(internal.deps.entryIndex.lookup(source.id)?.receipt_pending).toBe(true);
      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "empty",
        drained: 0,
        hasMore: false,
      });
      expect(terminalEntries(borg, session.session_id)).toHaveLength(0);
      expect(finalizerRequests(llm)).toHaveLength(0);

      releasePersistence.resolve();
      await expect(enqueue).resolves.toMatchObject({
        status: "enqueued",
        streamEntryId: source.id,
      });
      expect(internal.deps.entryIndex.lookup(source.id)?.receipt_pending).toBe(false);

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });

      const terminals = terminalEntries(borg, session.session_id);
      const finalizerPrompt = requestTextMessages(finalizerRequests(llm).at(0)).join("\n");

      expect(terminals).toHaveLength(1);
      expect(terminals[0]).toMatchObject({
        kind: "agent_msg",
        content: "answer after image receipt",
        response_to: responseToFor([source]),
      });
      expect(finalizerPrompt).toContain('<attachments count="1">');
      expect(finalizerPrompt).toContain("small test image");
      expect(finalizerRequests(llm)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("skips a receipt-pending image message on reopen until redelivery completes the receipt", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-receipt-pending-reopen-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_002_000);
    const session = createConversationHarness({ externalId: "receipt-pending-reopen" });
    const sourceMessageKey = {
      source_type: "demo" as const,
      source_external_id: session.source_external_id,
      external_message_id: "image-redelivery",
    };
    let borg = await openHarness({ tempDir, clock, llm: new FakeLLMClient() });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      borg.sessions.ensure(session);
      await borg.stream.append(
        {
          kind: "user_msg",
          content: "crashed before image receipt",
          sender_entity_id: senderEntityId,
          source_message_key: sourceMessageKey,
          receipt_pending: true,
        },
        { session: session.session_id },
      );
      const [source] = userEntries(borg, session.session_id);

      if (source === undefined) {
        throw new Error("expected queued source entry");
      }

      await borg.close();

      const llm = new FakeLLMClient({
        responses: [
          createImagePerceptionResponse(),
          createEmitAnswerResponse("answer after image redelivery"),
          createEmptyReflectionResponse(),
        ],
      });
      borg = await openHarness({ tempDir, clock, llm });
      const internal = borgInternals<HarnessBorgInternals>(borg);

      borg.inbox.catchUp.start();
      await Promise.resolve();
      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "empty",
        drained: 0,
        hasMore: false,
      });
      expect(terminalEntries(borg, session.session_id)).toHaveLength(0);
      expect(finalizerRequests(llm)).toHaveLength(0);
      expect(imagePerceptionRequests(llm)).toHaveLength(0);
      expect(internal.deps.entryIndex.lookup(source.id)?.receipt_pending).toBe(true);

      await expect(
        borg.enqueueMessage({
          session,
          userMessage: "redelivered with image",
          senderEntityId,
          sourceMessageKey,
          arrivedAt: clock.now(),
          attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
        }),
      ).resolves.toEqual({
        status: "duplicate",
        sessionId: session.session_id,
        streamEntryId: source.id,
      });
      expect(internal.deps.entryIndex.lookup(source.id)?.receipt_pending).toBe(false);
      expect(imagePerceptionRequests(llm)).toHaveLength(1);

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });
      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "empty",
        drained: 0,
        hasMore: false,
      });

      const terminals = terminalEntries(borg, session.session_id);
      const finalizerPrompt = requestTextMessages(finalizerRequests(llm).at(0)).join("\n");

      expect(terminals).toHaveLength(1);
      expect(terminals[0]).toMatchObject({
        kind: "agent_msg",
        content: "answer after image redelivery",
        response_to: responseToFor([source]),
      });
      expect(finalizerPrompt).toContain('<attachments count="1">');
      expect(finalizerPrompt).toContain("small test image");
      expect(finalizerRequests(llm)).toHaveLength(1);
      expect(imagePerceptionRequests(llm)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("dedupes a completed image enqueue without rewriting blobs or re-running perception", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-image-duplicate-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_003_000);
    const session = createConversationHarness({ externalId: "image-duplicate" });
    const llm = new FakeLLMClient({
      responses: [createImagePerceptionResponse()],
    });
    const blobWrite = vi.spyOn(AttachmentBlobStore.prototype, "write");
    const borg = await openHarness({ tempDir, clock, llm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");
      const sourceMessageKey = {
        source_type: "demo" as const,
        source_external_id: session.source_external_id,
        external_message_id: "image-duplicate-key",
      };
      const first = await borg.enqueueMessage({
        session,
        userMessage: "first image delivery",
        senderEntityId,
        sourceMessageKey,
        arrivedAt: clock.now(),
        attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
      });
      const writesAfterFirst = blobWrite.mock.calls.length;
      const perceptionCallsAfterFirst = imagePerceptionRequests(llm).length;
      const imageEntriesAfterFirst = imageAttachmentEntries(borg, session.session_id).length;

      expect(first.status).toBe("enqueued");
      expect(writesAfterFirst).toBe(1);
      expect(perceptionCallsAfterFirst).toBe(1);
      expect(imageEntriesAfterFirst).toBe(1);

      await expect(
        borg.enqueueMessage({
          session,
          userMessage: "duplicate image delivery",
          senderEntityId,
          sourceMessageKey,
          arrivedAt: clock.now(),
          attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
        }),
      ).resolves.toEqual({
        status: "duplicate",
        sessionId: session.session_id,
        streamEntryId: first.streamEntryId,
      });

      expect(blobWrite).toHaveBeenCalledTimes(writesAfterFirst);
      expect(imagePerceptionRequests(llm)).toHaveLength(perceptionCallsAfterFirst);
      expect(imageAttachmentEntries(borg, session.session_id)).toHaveLength(imageEntriesAfterFirst);
      expect(userEntries(borg, session.session_id)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("coalesces two image messages into one ordered batch with per-message perceptions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-image-coalesce-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_004_000);
    const session = createConversationHarness({ externalId: "image-coalesce" });
    const llm = new FakeLLMClient({
      responses: [
        createImagePerceptionResponse({
          caption: "first message image caption",
          visibleText: ["first message visible text"],
          searchTerms: ["first image"],
        }),
        createImagePerceptionResponse({
          caption: "second message image caption",
          visibleText: ["second message visible text"],
          searchTerms: ["second image"],
        }),
        createEmitAnswerResponse("answer after two image messages"),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openHarness({ tempDir, clock, llm });

    try {
      const senderA = borg.entities.resolve("Sender A");
      const senderB = borg.entities.resolve("Sender B");

      await borg.enqueueMessage({
        session,
        userMessage: "first image message",
        senderEntityId: senderA,
        sourceMessageKey: {
          source_type: "demo",
          source_external_id: session.source_external_id,
          external_message_id: "image-coalesce-1",
        },
        arrivedAt: clock.now(),
        attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
      });
      clock.advance(10);
      await borg.enqueueMessage({
        session,
        userMessage: "second image message",
        senderEntityId: senderB,
        sourceMessageKey: {
          source_type: "demo",
          source_external_id: session.source_external_id,
          external_message_id: "image-coalesce-2",
        },
        arrivedAt: clock.now(),
        attachments: [{ mediaType: "image/gif", bytes: GIF_1X1_ALT }],
      });

      const users = userEntries(borg, session.session_id);

      expect(users).toHaveLength(2);
      expect(imageAttachmentEntries(borg, session.session_id)).toHaveLength(2);
      expect(imagePerceptionRequests(llm)).toHaveLength(2);

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 2,
        hasMore: false,
      });

      const terminals = terminalEntries(borg, session.session_id);
      const stamp = responseToFor(users);
      const finalizerPrompt = requestTextMessages(finalizerRequests(llm).at(0)).join("\n");
      const firstMessageOffset = finalizerPrompt.indexOf("first image message");
      const secondMessageOffset = finalizerPrompt.indexOf("second image message");
      const firstCaptionOffset = finalizerPrompt.indexOf("first message image caption");
      const secondCaptionOffset = finalizerPrompt.indexOf("second message image caption");

      expect(terminals).toHaveLength(1);
      expect(terminals[0]).toMatchObject({
        kind: "agent_msg",
        content: "answer after two image messages",
        response_to: stamp,
      });
      expect(terminals[0]?.response_to).toMatchObject({
        kind: "stream_backlog",
        source_entry_ids: users.map((entry) => entry.id),
        count: 2,
      });
      expect(firstMessageOffset).toBeGreaterThanOrEqual(0);
      expect(secondMessageOffset).toBeGreaterThan(firstMessageOffset);
      expect(firstCaptionOffset).toBeGreaterThan(firstMessageOffset);
      expect(firstCaptionOffset).toBeLessThan(secondMessageOffset);
      expect(secondCaptionOffset).toBeGreaterThan(secondMessageOffset);
      expect(finalizerPrompt).toContain("first message visible text");
      expect(finalizerPrompt).toContain("second message visible text");
      expect(finalizerRequests(llm)).toHaveLength(1);
      expect(imagePerceptionRequests(llm)).toHaveLength(2);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("renders multiple images from one async message in provided order after reload", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-image-order-reload-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_004_500);
    const session = createConversationHarness({ externalId: "image-order-reload" });
    const enqueueLlm = new FakeLLMClient({
      responses: [
        createImagePerceptionResponse({
          caption: "first provided image caption",
          searchTerms: ["first provided image"],
        }),
        createImagePerceptionResponse({
          caption: "second provided image caption",
          searchTerms: ["second provided image"],
        }),
      ],
    });
    let borg = await openHarness({ tempDir, clock, llm: enqueueLlm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await borg.enqueueMessage({
        session,
        userMessage: "one message with two images",
        senderEntityId,
        sourceMessageKey: {
          source_type: "demo",
          source_external_id: session.source_external_id,
          external_message_id: "two-images-one-message",
        },
        arrivedAt: clock.now(),
        attachments: [
          { mediaType: "image/gif", bytes: GIF_1X1 },
          { mediaType: "image/gif", bytes: GIF_1X1_ALT },
        ],
      });

      const [source] = userEntries(borg, session.session_id);

      if (source === undefined) {
        throw new Error("expected queued source entry");
      }

      expect(imageAttachmentEntries(borg, session.session_id)).toHaveLength(2);
      expect(imagePerceptionRequests(enqueueLlm)).toHaveLength(2);

      await borg.close();

      const replayLlm = new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("answer after ordered image reload"),
          createEmptyReflectionResponse(),
        ],
      });
      borg = await openHarness({ tempDir, clock, llm: replayLlm });

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });

      const finalizerPrompt = requestTextMessages(finalizerRequests(replayLlm).at(0)).join("\n");
      const firstAttachmentOffset = finalizerPrompt.indexOf('<attachment index="1"');
      const secondAttachmentOffset = finalizerPrompt.indexOf('<attachment index="2"');
      const firstCaptionOffset = finalizerPrompt.indexOf("first provided image caption");
      const secondCaptionOffset = finalizerPrompt.indexOf("second provided image caption");

      expect(terminalEntries(borg, session.session_id)).toHaveLength(1);
      expect(firstAttachmentOffset).toBeGreaterThanOrEqual(0);
      expect(secondAttachmentOffset).toBeGreaterThan(firstAttachmentOffset);
      expect(firstCaptionOffset).toBeGreaterThan(firstAttachmentOffset);
      expect(firstCaptionOffset).toBeLessThan(secondAttachmentOffset);
      expect(secondCaptionOffset).toBeGreaterThan(secondAttachmentOffset);
      expect(finalizerRequests(replayLlm)).toHaveLength(1);
      expect(imagePerceptionRequests(replayLlm)).toHaveLength(0);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("drains only queued messages and never replays an already-answered normal turn", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-normal-plus-queued-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_005_000);
    const session = createConversationHarness({ externalId: "normal-plus-queued" });
    const llm = new FakeLLMClient({
      responses: [
        createEmitAnswerResponse("normal answer"),
        createEmptyReflectionResponse(),
        createEmitAnswerResponse("queued answer"),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openHarness({ tempDir, clock, llm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      borg.sessions.ensure(session);

      await borg.turn({
        sessionId: session.session_id,
        audience: session.audience_label,
        userMessage: "normal answered turn",
      });
      clock.advance(10);
      await enqueueOne({
        borg,
        clock,
        session,
        senderEntityId,
        externalMessageId: "queued-after-normal",
        text: "queued turn only",
      });

      const users = userEntries(borg, session.session_id);
      const normalUser = users.find((entry) => entry.turn_id !== undefined);
      const queuedUser = users.find((entry) => entry.turn_id === undefined);

      if (normalUser === undefined || queuedUser === undefined) {
        throw new Error("expected one normal answered user and one queued user");
      }

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });

      const stampedTerminal = terminalEntries(borg, session.session_id).find(
        (entry) => entry.response_to?.kind === "stream_backlog",
      );

      expect(stampedTerminal?.content).toBe("queued answer");
      expect(stampedTerminal?.response_to).toMatchObject({
        kind: "stream_backlog",
        source_entry_ids: [queuedUser.id],
        count: 1,
      });
      expect(stampedTerminal?.response_to?.source_entry_ids).not.toContain(normalUser.id);
      expect(finalizerRequests(llm)).toHaveLength(2);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("reruns after a pre-append terminal crash without persisting a terminal stamp", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-pre-append-crash-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_010_000);
    const session = createConversationHarness({ externalId: "pre-append-crash" });
    const failedLlm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("lost before append")],
    });
    let borg = await openHarness({ tempDir, clock, llm: failedLlm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await enqueueOne({ borg, clock, session, senderEntityId });

      const internal = borgInternals<HarnessBorgInternals>(borg);
      const originalCreateStreamWriter =
        internal.deps.turnOrchestrator.options.createStreamWriter.bind(
          internal.deps.turnOrchestrator.options,
        );
      let failedTerminalAppend = false;

      internal.deps.turnOrchestrator.options.createStreamWriter = (sessionId) => {
        const writer = originalCreateStreamWriter(sessionId);
        const originalAppend = writer.append.bind(writer);

        writer.append = async (entryInput) => {
          if (entryInput.kind === "agent_msg" && !failedTerminalAppend) {
            failedTerminalAppend = true;
            throw new Error("test pre-append terminal crash");
          }

          return originalAppend(entryInput);
        };

        return writer;
      };

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "error",
        drained: 0,
        hasMore: true,
      });
      expect(failedTerminalAppend).toBe(true);
      expect(terminalEntries(borg, session.session_id)).toHaveLength(0);
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toBeNull();

      await borg.close();

      const retryLlm = new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("answer after retry"),
          createEmptyReflectionResponse(),
        ],
      });
      borg = await openHarness({ tempDir, clock, llm: retryLlm });

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });

      const terminals = terminalEntries(borg, session.session_id);

      expect(terminals).toHaveLength(1);
      expect(terminals[0]).toMatchObject({
        kind: "agent_msg",
        content: "answer after retry",
      });
      expect(terminals[0]?.response_to).toMatchObject({
        kind: "stream_backlog",
        count: 1,
      });
      expect(finalizerRequests(retryLlm)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("reconciles a stamped terminal reply after watermark advance fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-watermark-crash-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_020_000);
    const session = createConversationHarness({ externalId: "watermark-crash" });
    const failedLlm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("persisted before watermark")],
    });
    let borg = await openHarness({ tempDir, clock, llm: failedLlm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await enqueueOne({ borg, clock, session, senderEntityId });

      const internal = borgInternals<HarnessBorgInternals>(borg);
      const originalAdvance = internal.deps.chatResponseWatermarkCoordinator.advanceThrough.bind(
        internal.deps.chatResponseWatermarkCoordinator,
      );
      let failedAdvance = false;

      internal.deps.chatResponseWatermarkCoordinator.advanceThrough = (sessionId, cursor) => {
        if (!failedAdvance) {
          failedAdvance = true;
          throw new Error("test watermark crash");
        }

        return originalAdvance(sessionId, cursor);
      };

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "error",
        drained: 0,
        hasMore: true,
      });

      const stamped = terminalEntries(borg, session.session_id);

      expect(failedAdvance).toBe(true);
      expect(stamped).toHaveLength(1);
      expect(stamped[0]).toMatchObject({
        kind: "agent_msg",
        content: "persisted before watermark",
      });
      expect(stamped[0]?.response_to).toMatchObject({
        kind: "stream_backlog",
        count: 1,
      });
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toBeNull();

      await borg.close();

      const retryLlm = new FakeLLMClient({
        responses: [createEmitAnswerResponse("must not be used")],
      });
      borg = await openHarness({ tempDir, clock, llm: retryLlm });

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "empty",
        drained: 0,
        hasMore: false,
      });

      const terminalsAfterReconcile = terminalEntries(borg, session.session_id);
      const reopenedInternal = borgInternals<HarnessBorgInternals>(borg);

      expect(terminalsAfterReconcile).toHaveLength(1);
      expect(terminalsAfterReconcile[0]?.id).toBe(stamped[0]?.id);
      expect(
        reopenedInternal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toEqual((stamped[0]?.response_to as StreamResponseTo).through_cursor_inclusive);
      expect(finalizerRequests(retryLlm)).toHaveLength(0);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("answers a quarantined inactive queued message after replay before terminal response", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-quarantine-before-terminal-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_025_000);
    const session = createConversationHarness({ externalId: "quarantine-before-terminal" });
    let borg = await openHarness({
      tempDir,
      clock,
      llm: new FakeLLMClient(),
    });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await enqueueOne({ borg, clock, session, senderEntityId });
      const [source] = userEntries(borg, session.session_id);

      if (source === undefined) {
        throw new Error("expected queued source entry");
      }

      await appendQuarantineMarker({
        borg,
        sessionId: session.session_id,
        entries: [source],
      });

      const internal = borgInternals<HarnessBorgInternals>(borg);

      expect(internal.deps.entryIndex.lookup(source.id)?.active).toBe(false);
      expect(terminalEntries(borg, session.session_id)).toHaveLength(0);

      await borg.close();

      const retryLlm = new FakeLLMClient({
        responses: [
          createEmitAnswerResponse("answer after quarantine replay"),
          createEmptyReflectionResponse(),
        ],
      });
      borg = await openHarness({ tempDir, clock, llm: retryLlm });

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });

      const terminals = terminalEntries(borg, session.session_id);

      expect(terminals).toHaveLength(1);
      expect(terminals[0]).toMatchObject({
        kind: "agent_msg",
        content: "answer after quarantine replay",
        response_to: responseToFor([source]),
      });
      expect(finalizerRequests(retryLlm)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it.each<StreamEntry["kind"]>(["agent_msg", "agent_observed", "agent_suppressed"])(
    "reconciles a quarantined %s terminal after replay without regenerating",
    async (kind) => {
      const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-quarantine-terminal-"));
      tempDirs.push(tempDir);
      const clock = new ManualClock(1_900_000_026_000);
      const session = createConversationHarness({ externalId: `quarantine-terminal-${kind}` });
      let borg = await openHarness({
        tempDir,
        clock,
        llm: new FakeLLMClient(),
      });

      try {
        const senderEntityId = borg.entities.resolve("Sender");

        await enqueueOne({ borg, clock, session, senderEntityId });
        const [source] = userEntries(borg, session.session_id);

        if (source === undefined) {
          throw new Error("expected queued source entry");
        }

        const responseTo = responseToFor([source]);

        await appendQuarantineMarker({
          borg,
          sessionId: session.session_id,
          entries: [source],
        });
        await borg.stream.append(
          {
            kind,
            content: "terminal before watermark",
            response_to: responseTo,
          },
          { session: session.session_id },
        );

        const internal = borgInternals<HarnessBorgInternals>(borg);

        expect(internal.deps.entryIndex.lookup(source.id)?.active).toBe(false);
        expect(
          internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
        ).toBeNull();

        await borg.close();

        const retryLlm = new FakeLLMClient({
          responses: [createEmitAnswerResponse("must not be used")],
        });
        borg = await openHarness({ tempDir, clock, llm: retryLlm });

        await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
          status: "empty",
          drained: 0,
          hasMore: false,
        });

        const terminals = terminalEntries(borg, session.session_id);
        const reopenedInternal = borgInternals<HarnessBorgInternals>(borg);

        expect(terminals).toHaveLength(1);
        expect(terminals[0]).toMatchObject({
          kind,
          response_to: responseTo,
        });
        expect(
          reopenedInternal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
        ).toEqual(responseTo.through_cursor_inclusive);
        expect(finalizerRequests(retryLlm)).toHaveLength(0);
      } finally {
        await borg.close().catch(() => undefined);
      }
    },
  );

  it("self-repairs a terminal append index failure without a second generation", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-terminal-index-repair-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_027_000);
    const session = createConversationHarness({ externalId: "terminal-index-repair" });
    const llm = new FakeLLMClient({
      responses: [
        createEmitAnswerResponse("terminal survived repair"),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openHarness({ tempDir, clock, llm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await enqueueOne({ borg, clock, session, senderEntityId });

      const internal = borgInternals<HarnessBorgInternals>(borg);
      const fault = installTerminalIndexFailure({
        internal,
        tempDir,
        clock,
        repair: "succeed",
      });

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });
      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "empty",
        drained: 0,
        hasMore: false,
      });

      const terminals = terminalEntries(borg, session.session_id);
      const users = userEntries(borg, session.session_id);

      expect(fault.backfillCalls()).toBe(1);
      expect(terminals).toHaveLength(1);
      expect(terminals[0]).toMatchObject({
        kind: "agent_msg",
        content: "terminal survived repair",
        response_to: responseToFor(users),
      });
      expect(internal.deps.entryIndex.lookup(terminals[0]!.id)).not.toBeNull();
      expect(finalizerRequests(llm)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("uses repair-only retry after a poisoned terminal append and reconciles without a second reply", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-terminal-index-poison-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_028_000);
    const session = createConversationHarness({ externalId: "terminal-index-poison" });
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("terminal before poison")],
    });
    const borg = await openHarness({ tempDir, clock, llm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await enqueueOne({ borg, clock, session, senderEntityId });

      const internal = borgInternals<HarnessBorgInternals>(borg);
      const fault = installTerminalIndexFailure({
        internal,
        tempDir,
        clock,
        repair: "fail",
      });

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "error",
        drained: 0,
        hasMore: true,
      });

      const [durableTerminal] = terminalEntries(borg, session.session_id);

      if (durableTerminal === undefined) {
        throw new Error("expected durable terminal entry after poisoned append");
      }

      expect(fault.backfillCalls()).toBe(2);
      expect(internal.deps.entryIndex.lookup(durableTerminal.id)).toBeNull();
      expect(abortedTurnEntries(borg, session.session_id)).toHaveLength(0);
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toBeNull();

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 0,
        hasMore: true,
      });
      expect(internal.deps.entryIndex.lookup(durableTerminal.id)).not.toBeNull();
      expect(finalizerRequests(llm)).toHaveLength(1);

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "empty",
        drained: 0,
        hasMore: false,
      });

      const terminals = terminalEntries(borg, session.session_id);
      const streamEntries = borg.stream.tail(200, { session: session.session_id });
      const entryIndexes = streamEntries.map((entry) => entry.entry_index);

      expect(terminals).toHaveLength(1);
      expect(terminals[0]?.id).toBe(durableTerminal.id);
      expect(entryIndexes.every((entryIndex) => entryIndex !== undefined)).toBe(true);
      expect(new Set(entryIndexes).size).toBe(entryIndexes.length);
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toEqual((durableTerminal.response_to as StreamResponseTo).through_cursor_inclusive);
      expect(finalizerRequests(llm)).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("treats a directly delivered stamped outbound message as answered on catch-up", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-direct-outbound-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_030_000);
    const session = createConversationHarness({ externalId: "direct-outbound" });
    const connectorDeliver = vi.fn(async () => ({
      status: "transported" as const,
      externalMessageId: "outbound-1",
    }));
    const connector: MessageConnector = {
      sourceType: "demo",
      deliver: connectorDeliver,
    };
    const llm = new FakeLLMClient({
      responses: [createEmitAnswerResponse("must not be used")],
    });
    const borg = await openHarness({
      tempDir,
      clock,
      llm,
      outboundConnectors: [connector],
    });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await enqueueOne({ borg, clock, session, senderEntityId });

      const [source] = userEntries(borg, session.session_id);
      const responseTo = responseToFor([source!]);
      const sessionRecord = borg.sessions.get(session.session_id);
      const internal = borgInternals<HarnessBorgInternals>(borg);

      if (sessionRecord === null) {
        throw new Error("expected session record");
      }

      await internal.deps.outboundDelivery.deliver({
        session: sessionRecord,
        message: {
          content: "direct outbound answer",
          streamInput: {
            response_to: responseTo,
          },
        },
      });

      expect(connectorDeliver).toHaveBeenCalledTimes(1);
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toBeNull();

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "empty",
        drained: 0,
        hasMore: false,
      });

      expect(connectorDeliver).toHaveBeenCalledTimes(1);
      expect(terminalEntries(borg, session.session_id)).toHaveLength(1);
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toEqual(responseTo.through_cursor_inclusive);
      expect(finalizerRequests(llm)).toHaveLength(0);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("does not auto-retry connector delivery for a stamped reply appended before send", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-d1-loss-contract-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_040_000);
    const session = createConversationHarness({ externalId: "d1-loss-contract" });
    let borg = await openHarness({
      tempDir,
      clock,
      llm: new FakeLLMClient(),
    });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      await enqueueOne({ borg, clock, session, senderEntityId });

      const [source] = userEntries(borg, session.session_id);
      const responseTo = responseToFor([source!]);

      await borg.stream.append(
        {
          kind: "agent_msg",
          content: "appended before connector send",
          response_to: responseTo,
        },
        { session: session.session_id },
      );

      await borg.close();

      const connectorDeliver = vi.fn(async () => ({
        status: "transported" as const,
        externalMessageId: "should-not-exist",
      }));
      const connector: MessageConnector = {
        sourceType: "demo",
        deliver: connectorDeliver,
      };
      const retryLlm = new FakeLLMClient({
        responses: [createEmitAnswerResponse("must not be used")],
      });
      borg = await openHarness({
        tempDir,
        clock,
        llm: retryLlm,
        outboundConnectors: [connector],
      });

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "empty",
        drained: 0,
        hasMore: false,
      });

      const internal = borgInternals<HarnessBorgInternals>(borg);

      expect(connectorDeliver).not.toHaveBeenCalled();
      expect(terminalEntries(borg, session.session_id)).toHaveLength(1);
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toEqual(responseTo.through_cursor_inclusive);
      expect(finalizerRequests(retryLlm)).toHaveLength(0);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("answers a bounded prefix, advances to entry 16, then drains the remainder", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-async-e2e-prefix-cap-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_900_000_050_000);
    const session = createConversationHarness({ externalId: "prefix-cap" });
    const llm = new FakeLLMClient({
      responses: [
        createEmitAnswerResponse("answered first prefix"),
        createEmptyReflectionResponse(),
        createEmitAnswerResponse("answered remainder"),
        createEmptyReflectionResponse(),
      ],
    });
    const borg = await openHarness({ tempDir, clock, llm });

    try {
      const senderEntityId = borg.entities.resolve("Sender");

      for (let index = 1; index <= 17; index += 1) {
        await enqueueOne({
          borg,
          clock,
          session,
          senderEntityId,
          externalMessageId: `message-${index}`,
          text: `m${index}`,
        });
        clock.advance(10);
      }

      const users = userEntries(borg, session.session_id);

      expect(users).toHaveLength(17);

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 16,
        hasMore: true,
      });

      const firstTerminals = terminalEntries(borg, session.session_id);
      const firstBatch = users.slice(0, 16);
      const firstResponseTo = responseToFor(firstBatch);
      const internal = borgInternals<HarnessBorgInternals>(borg);

      expect(firstTerminals).toHaveLength(1);
      expect(firstTerminals[0]?.response_to).toEqual(firstResponseTo);
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toEqual(firstResponseTo.through_cursor_inclusive);

      await expect(borg.inbox.catchUp.tick(session.session_id)).resolves.toMatchObject({
        status: "drained",
        drained: 1,
        hasMore: false,
      });

      const allTerminals = terminalEntries(borg, session.session_id);
      const secondResponseTo = responseToFor(
        [users[16]!],
        firstResponseTo.through_cursor_inclusive,
      );

      expect(allTerminals).toHaveLength(2);
      expect(allTerminals[1]?.response_to).toEqual(secondResponseTo);
      expect(
        internal.deps.chatResponseWatermarkCoordinator.getWatermark(session.session_id),
      ).toEqual(secondResponseTo.through_cursor_inclusive);
      expect(finalizerRequests(llm)).toHaveLength(2);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });
});
