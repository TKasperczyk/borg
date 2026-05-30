import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";

import {
  DemoMessageConnector,
  QUARANTINED_USER_ENTRY_EVENT,
  type Config,
  type EntityId,
  type FrameAnomalyKind,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type MessageConnector,
  type SessionId,
  type StreamEntryId,
  type TurnInputAttachment,
} from "../src/index.js";
import type {
  StreamCursor,
  StreamEntry,
  StreamResponseTo,
  StreamWriter,
} from "../src/stream/index.js";
import {
  Borg,
  FakeLLMClient,
  ManualClock,
  ScriptedEmbeddingClient,
  borgInternals,
  createEmptyReflectionResponse,
  createEmitAnswerResponse,
  createSessionId,
  createStreamEntryId,
  createTestConfig,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "../src/borg/__tests__/test-helpers.js";

const TERMINAL_KINDS = new Set<StreamEntry["kind"]>([
  "agent_msg",
  "agent_suppressed",
  "agent_observed",
]);
const GIF_1X1 = Uint8Array.from([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00]);
const GIF_1X1_ALT = Uint8Array.from([
  0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x21,
]);

type ScenarioSession = ReturnType<typeof createConversationHarness>;
type ConfigOverrides = Parameters<typeof createTestConfig>[0];

type HarnessBorgInternals = {
  deps: {
    sqlite: {
      prepare(sql: string): {
        get(...values: unknown[]): unknown;
      };
    };
    chatResponseCatchUpWorker: {
      stop(options?: { graceful?: boolean }): Promise<void>;
    };
    chatResponseWatermarkCoordinator: {
      getWatermark(sessionId: SessionId): StreamCursor | null;
      advanceThrough(sessionId: SessionId, cursor: StreamCursor): unknown;
    };
    activityRepository: {
      record(input: {
        kind: "user_contact" | "borg_replied" | "turn_completed";
        occurredAt: number;
        sessionId: SessionId;
        turnId?: string | null;
        speakerEntityId?: EntityId | null;
        actorEntityId?: EntityId | null;
        audienceEntityId?: EntityId | null;
        sourceStreamEntryIds: readonly StreamEntryId[];
        status?: "active" | "inactive";
      }): unknown;
    };
    turnOrchestrator: {
      options: {
        createStreamWriter(sessionId: SessionId): StreamWriter;
      };
    };
  };
};

type TraceEvent = {
  event?: unknown;
  disposition?: unknown;
  session_audience_role?: unknown;
  current_sender_borg_role?: unknown;
  [key: string]: unknown;
};

type StepStats = {
  passed: number;
  failed: number;
};

function createFrameAnomalyResponse(input: {
  kind: FrameAnomalyKind;
  confidence?: number;
  rationale?: string;
}): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_frame_anomaly",
        name: "ClassifyFrameAnomaly",
        input: {
          kind: input.kind,
          confidence: input.confidence ?? (input.kind === "normal" ? 0.9 : 0.98),
          rationale: input.rationale ?? "The frame anomaly classifier categorized the turn.",
        },
      },
    ],
  };
}

function createImagePerceptionResponse(input: {
  caption: string;
  visibleText: readonly string[];
  searchTerms: readonly string[];
}) {
  return {
    messageBlocks: [
      {
        type: "tool_use" as const,
        id: "toolu_image",
        name: "EmitImagePerception",
        input: {
          caption: input.caption,
          image_kind: "other",
          visible_text: [...input.visibleText],
          objects: [],
          people_or_roles: [],
          scene: "scripted image fixture",
          colors_and_visual_attributes: [],
          spatial_relationships: [],
          possible_user_relevant_details: [],
          search_terms: [...input.searchTerms],
          uncertainties: [],
        },
      },
    ],
    input_tokens: 4,
    output_tokens: 4,
    stop_reason: "tool_use" as const,
  };
}

function createConversationHarness(input: {
  externalId: string;
  label?: string;
  sessionId?: SessionId;
  audienceEntityId?: EntityId | null;
  audienceRole?: "participant" | "operator";
  conversationKind?: "thread" | "channel" | "dm" | "demo";
}) {
  return {
    session_id: input.sessionId ?? createSessionId(),
    source_type: "demo" as const,
    source_external_id: input.externalId,
    label: input.label ?? input.externalId,
    audience_label: input.label ?? input.externalId,
    audience_entity_id: input.audienceEntityId ?? null,
    conversation_kind: input.conversationKind ?? ("thread" as const),
    audience_role: input.audienceRole ?? ("participant" as const),
  };
}

async function openHarness(input: {
  tempDir: string;
  clock: ManualClock;
  llm: FakeLLMClient;
  tracerPath?: string;
  outboundConnectors?: readonly MessageConnector[];
  configOverrides?: ConfigOverrides;
}): Promise<Borg> {
  const configOverrides = input.configOverrides ?? {};
  const config: Config = createTestConfig({
    ...configOverrides,
    dataDir: input.tempDir,
    perception: {
      llmEnabled: false,
      ...configOverrides.perception,
    },
    affective: {
      llmEnabled: false,
      ...configOverrides.affective,
    },
    embedding: {
      baseUrl: "http://localhost:1234/v1",
      apiKey: "test",
      model: "fake-embed",
      dims: 4,
      ...configOverrides.embedding,
    },
    anthropic: {
      auth: "api-key",
      apiKey: "test",
      ...configOverrides.anthropic,
      models: {
        cognition: "sonnet",
        background: "haiku",
        extraction: "haiku",
        recallExpansion: "haiku",
        ...configOverrides.anthropic?.models,
      },
    },
  });

  const borg = await Borg.open({
    config,
    clock: input.clock,
    embeddingDimensions: 4,
    embeddingClient: new ScriptedEmbeddingClient(),
    llmClient: input.llm,
    liveExtraction: false,
    tracerPath: input.tracerPath,
    outboundConnectors: input.outboundConnectors,
  });

  await borgInternals<HarnessBorgInternals>(borg).deps.chatResponseCatchUpWorker.stop({
    graceful: false,
  });

  return borg;
}

function finalizerRequests(llm: FakeLLMClient): LLMCompleteOptions[] {
  return llm.requests.filter(
    (request) => request.budget === "cognition-system-1" || request.budget === "cognition-system-2",
  );
}

function imagePerceptionRequests(llm: FakeLLMClient): LLMCompleteOptions[] {
  return llm.requests.filter((request) => request.budget === "image-perception");
}

function terminalEntries(borg: Borg, sessionId: SessionId): StreamEntry[] {
  return borg.stream
    .tail(300, { session: sessionId })
    .filter((entry) => TERMINAL_KINDS.has(entry.kind));
}

function agentMessages(
  borg: Borg,
  sessionId: SessionId,
): Array<StreamEntry & { kind: "agent_msg" }> {
  return borg.stream
    .tail(300, { session: sessionId })
    .filter((entry): entry is StreamEntry & { kind: "agent_msg" } => entry.kind === "agent_msg");
}

function userEntries(borg: Borg, sessionId: SessionId): Array<StreamEntry & { kind: "user_msg" }> {
  return borg.stream
    .tail(300, { session: sessionId })
    .filter((entry): entry is StreamEntry & { kind: "user_msg" } => entry.kind === "user_msg");
}

function imageAttachmentEntries(
  borg: Borg,
  sessionId: SessionId,
): Array<StreamEntry & { kind: "user_image_attachment" }> {
  return borg.stream
    .tail(300, { session: sessionId })
    .filter(
      (entry): entry is StreamEntry & { kind: "user_image_attachment" } =>
        entry.kind === "user_image_attachment",
    );
}

function streamEntryById(borg: Borg, sessionId: SessionId, id: StreamEntryId): StreamEntry {
  const entry = borg.stream
    .tail(300, { session: sessionId })
    .find((candidate) => candidate.id === id);

  assert.ok(entry, `expected stream entry ${id}`);
  return entry;
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

  assert.ok(last, "response_to requires at least one source entry");

  return {
    kind: "stream_backlog",
    from_cursor_exclusive: fromCursorExclusive,
    through_cursor_inclusive: cursorFor(last),
    source_entry_ids: entries.map((entry) => entry.id),
    count: entries.length,
  };
}

function sourceMessageKey(session: ScenarioSession, externalMessageId: string) {
  return {
    source_type: "demo" as const,
    source_external_id: session.source_external_id,
    external_message_id: externalMessageId,
  };
}

async function enqueueOne(input: {
  borg: Borg;
  clock: ManualClock;
  session: ScenarioSession;
  senderEntityId: EntityId;
  externalMessageId: string;
  text: string;
  advanceMs?: number;
  attachments?: readonly TurnInputAttachment[];
}) {
  const result = await input.borg.enqueueMessage({
    session: input.session,
    userMessage: input.text,
    senderEntityId: input.senderEntityId,
    sourceMessageKey: sourceMessageKey(input.session, input.externalMessageId),
    arrivedAt: input.clock.now(),
    audience: input.session.audience_label,
    audienceEntityId: input.session.audience_entity_id,
    attachments: input.attachments,
  });

  input.clock.advance(input.advanceMs ?? 10);
  return result;
}

function systemText(request: LLMCompleteOptions | undefined): string {
  const system = request?.system;

  if (typeof system === "string") {
    return system;
  }

  if (Array.isArray(system)) {
    return system.map((block) => block.text).join("\n");
  }

  return "";
}

function requestText(request: LLMCompleteOptions | undefined): string {
  return [systemText(request), ...(request?.messages.map((message) => message.content) ?? [])].join(
    "\n",
  );
}

function extractInboundBatch(request: LLMCompleteOptions | undefined): string {
  const text = requestText(request);
  const start = text.indexOf("<inbound_batch");
  const endTag = "</inbound_batch>";
  const end = text.indexOf(endTag, start);

  if (start < 0 || end < 0) {
    return "(inbound batch block not found in finalizer prompt)";
  }

  return text.slice(start, end + endTag.length);
}

function countSql(borg: Borg, sql: string): number {
  const row = borgInternals<HarnessBorgInternals>(borg).deps.sqlite.prepare(sql).get() as
    | { count?: unknown }
    | undefined;

  return Number(row?.count ?? 0);
}

function countUserContacts(borg: Borg): number {
  return countSql(
    borg,
    "SELECT COUNT(*) AS count FROM activity_events WHERE kind = 'user_contact'",
  );
}

function sessionMessageCount(borg: Borg, sessionId: SessionId): number {
  const session = borg.sessions.get(sessionId);

  assert.ok(session, `expected session ${sessionId}`);
  return session.message_count;
}

function readTraceEvents(path: string): TraceEvent[] {
  if (!existsSync(path)) {
    return [];
  }

  const content = readFileSync(path, "utf8").trim();

  if (content.length === 0) {
    return [];
  }

  return content
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as TraceEvent);
}

function internalEventMatches(entry: StreamEntry, eventName: string): boolean {
  if (
    entry.kind !== "internal_event" ||
    typeof entry.content !== "object" ||
    entry.content === null
  ) {
    return false;
  }

  return (entry.content as { event?: unknown }).event === eventName;
}

function formatJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function indentBlock(text: string): string {
  return text
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n");
}

function describeError(error: unknown): string {
  if (error instanceof Error) {
    return error.stack ?? error.message;
  }

  return String(error);
}

async function runStep(
  stats: StepStats,
  title: string,
  action: () => Promise<void>,
): Promise<void> {
  console.log("");
  console.log(`STEP ${stats.passed + stats.failed + 1}: ${title}`);

  try {
    await action();
    stats.passed += 1;
    console.log(`PASS: ${title}`);
  } catch (error) {
    stats.failed += 1;
    console.log(`FAIL: ${title}`);
    console.log(indentBlock(describeError(error)));
  }
}

async function main(): Promise<void> {
  const stats: StepStats = { passed: 0, failed: 0 };
  const tempDir = mkdtempSync(join(tmpdir(), "borg-async-ingest-scenario-"));
  const tracePath = join(tempDir, "trace.jsonl");
  const clock = new ManualClock(1_900_100_000_000);
  const llm = new FakeLLMClient({
    responses: [
      createEmitAnswerResponse("Burst received: I saw the three launch-room updates together."),
      createEmptyReflectionResponse(),
      createImagePerceptionResponse({
        caption: "first crash-replay image caption",
        visibleText: ["first crash visible text"],
        searchTerms: ["first crash image"],
      }),
      createImagePerceptionResponse({
        caption: "second crash-replay image caption",
        visibleText: ["second crash visible text"],
        searchTerms: ["second crash image"],
      }),
      createEmitAnswerResponse("Image burst recovered with both stored perceptions."),
      createEmptyReflectionResponse(),
      createEmitAnswerResponse(
        "Remainder received: I picked up the follow-up after the watermark.",
      ),
      createEmptyReflectionResponse(),
      createFrameAnomalyResponse({
        kind: "roleplay_inversion",
        confidence: 0.98,
        rationale: "The multi-sender batch is not a single trusted operator control surface.",
      }),
      createEmitAnswerResponse("Multi-sender batch handled without privileged authority."),
      createEmptyReflectionResponse(),
      createEmitAnswerResponse("Crash batch persisted before the watermark advanced."),
    ],
  });
  const outboundConnectors = [new DemoMessageConnector()];
  const configOverrides: ConfigOverrides = {
    generation: {
      evidenceLedger: {
        enabled: true,
        currentSessionTranscriptTokenBudget: 50_000,
      },
    },
  };

  let borg: Borg | null = await openHarness({
    tempDir,
    clock,
    llm,
    tracerPath: tracePath,
    outboundConnectors,
    configOverrides,
  });
  let session: ScenarioSession | null = null;
  let creatorId: EntityId | null = null;
  let senderAId: EntityId | null = null;
  let senderBId: EntityId | null = null;
  let groupId: EntityId | null = null;
  let lastWatermark: StreamCursor | null = null;
  let crashStampedId: StreamEntryId | null = null;

  function requireBorg(): Borg {
    assert.ok(borg, "borg is not open");
    return borg;
  }

  function requireSession(): ScenarioSession {
    assert.ok(session, "session is not set up");
    return session;
  }

  try {
    console.log("Async ingest scenario");
    console.log(`temp_data_dir=${tempDir}`);
    console.log("model=FakeLLMClient");
    console.log("embeddings=ScriptedEmbeddingClient");
    console.log("liveExtraction=false perception.llmEnabled=false");

    await runStep(stats, "SETUP operator group with creator and two sender entities", async () => {
      const current = requireBorg();

      creatorId = current.entities.resolve("Tom", { kind: "person" });
      senderAId = current.entities.resolve("Riley", { kind: "person" });
      senderBId = current.entities.resolve("Morgan", { kind: "person" });
      groupId = current.entities.resolve("Operator Room", { kind: "group" });
      current.entities.setBorgRole(creatorId, "creator");

      session = createConversationHarness({
        externalId: "async-scenario-operator-room",
        label: "Operator Room",
        audienceEntityId: groupId,
        audienceRole: "operator",
        conversationKind: "channel",
      });
      current.sessions.ensure(session);

      console.log(`session_id=${session.session_id}`);
      console.log(`audience=${session.audience_label} role=${session.audience_role}`);
      console.log(`creator Tom=${creatorId}`);
      console.log(`sender A Riley=${senderAId}`);
      console.log(`sender B Morgan=${senderBId}`);
      assert.equal(current.sessions.get(session.session_id)?.message_count, 0);
    });

    await runStep(
      stats,
      "BURST + COALESCE answers three queued messages as one catch-up turn",
      async () => {
        const current = requireBorg();
        const currentSession = requireSession();
        assert.ok(senderAId, "sender A missing");

        const beforeFinalizers = finalizerRequests(llm).length;
        const enqueued = [];

        for (const [externalMessageId, text] of [
          ["burst-1", "Riley: deploy window is now 14:00 UTC."],
          ["burst-2", "Riley: the staging smoke test passed."],
          ["burst-3", "Riley: hold the public note until Morgan confirms."],
        ] as const) {
          enqueued.push(
            await enqueueOne({
              borg: current,
              clock,
              session: currentSession,
              senderEntityId: senderAId,
              externalMessageId,
              text,
            }),
          );
        }

        const users = userEntries(current, currentSession.session_id);
        const batchUsers = users.slice(-3);
        const tickResult = await current.inbox.catchUp.tick(currentSession.session_id);
        const terminals = terminalEntries(current, currentSession.session_id);
        const agent = agentMessages(current, currentSession.session_id).at(-1);
        const expectedStamp = responseToFor(batchUsers);
        const watermark = borgInternals<HarnessBorgInternals>(
          current,
        ).deps.chatResponseWatermarkCoordinator.getWatermark(currentSession.session_id);
        const finalizer = finalizerRequests(llm).slice(beforeFinalizers)[0];

        assert.deepEqual(
          enqueued.map((result) => result.status),
          ["enqueued", "enqueued", "enqueued"],
        );
        assert.deepEqual(tickResult, {
          sessionId: currentSession.session_id,
          status: "drained",
          drained: 3,
          hasMore: false,
        });
        assert.equal(terminals.length, 1);
        assert.ok(agent, "expected terminal agent_msg");
        assert.equal(
          agent.content,
          "Burst received: I saw the three launch-room updates together.",
        );
        assert.deepEqual(agent.response_to, expectedStamp);
        assert.equal(agent.response_to?.count, 3);
        assert.deepEqual(watermark, expectedStamp.through_cursor_inclusive);

        lastWatermark = expectedStamp.through_cursor_inclusive;

        console.log(
          `enqueued_source_keys=${enqueued.map((result) => result.streamEntryId).join(", ")}`,
        );
        console.log(`tick=${formatJson(tickResult)}`);
        console.log("rendered_batch:");
        console.log(indentBlock(extractInboundBatch(finalizer)));
        console.log("stamp:");
        console.log(indentBlock(formatJson(agent.response_to)));
        console.log(`watermark=${formatJson(watermark)}`);
      },
    );

    await runStep(
      stats,
      "IMAGE BURST CRASH-REPLAY drains stored perceptions once after reopen",
      async () => {
        const current = requireBorg();
        const currentSession = requireSession();
        assert.ok(senderAId, "sender A missing");
        assert.ok(senderBId, "sender B missing");
        assert.ok(lastWatermark, "previous watermark missing");

        const beforeFinalizers = finalizerRequests(llm).length;
        const beforeVisionCalls = imagePerceptionRequests(llm).length;
        const first = await enqueueOne({
          borg: current,
          clock,
          session: currentSession,
          senderEntityId: senderAId,
          externalMessageId: "image-crash-1",
          text: "Riley: image one arrived before the restart.",
          attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
        });
        const second = await enqueueOne({
          borg: current,
          clock,
          session: currentSession,
          senderEntityId: senderBId,
          externalMessageId: "image-crash-2",
          text: "Morgan: image two is part of the same recovered burst.",
          attachments: [{ mediaType: "image/gif", bytes: GIF_1X1_ALT }],
        });
        const sourceEntries = [
          streamEntryById(current, currentSession.session_id, first.streamEntryId),
          streamEntryById(current, currentSession.session_id, second.streamEntryId),
        ];
        const expectedStamp = responseToFor(sourceEntries, lastWatermark);
        const visionCallsAfterReceipt = imagePerceptionRequests(llm).length;

        assert.equal(visionCallsAfterReceipt, beforeVisionCalls + 2);
        assert.equal(imageAttachmentEntries(current, currentSession.session_id).length, 2);
        assert.equal(terminalEntries(current, currentSession.session_id).length, 1);

        await current.close();
        borg = null;
        borg = await openHarness({
          tempDir,
          clock,
          llm,
          tracerPath: tracePath,
          outboundConnectors,
          configOverrides,
        });

        const reopened = requireBorg();
        const tickResult = await reopened.inbox.catchUp.tick(currentSession.session_id);
        const agent = agentMessages(reopened, currentSession.session_id).at(-1);
        const finalizer = finalizerRequests(llm).slice(beforeFinalizers)[0];
        const renderedBatch = extractInboundBatch(finalizer);
        const firstMessageOffset = renderedBatch.indexOf("Riley: image one");
        const secondMessageOffset = renderedBatch.indexOf("Morgan: image two");
        const firstCaptionOffset = renderedBatch.indexOf("first crash-replay image caption");
        const secondCaptionOffset = renderedBatch.indexOf("second crash-replay image caption");
        const watermark = borgInternals<HarnessBorgInternals>(
          reopened,
        ).deps.chatResponseWatermarkCoordinator.getWatermark(currentSession.session_id);
        const matchingAgents = agentMessages(reopened, currentSession.session_id).filter(
          (entry) => entry.response_to?.source_entry_ids.includes(first.streamEntryId) === true,
        );

        assert.deepEqual(tickResult, {
          sessionId: currentSession.session_id,
          status: "drained",
          drained: 2,
          hasMore: false,
        });
        assert.ok(agent, "expected image replay terminal agent_msg");
        assert.equal(agent.content, "Image burst recovered with both stored perceptions.");
        assert.deepEqual(agent.response_to, expectedStamp);
        assert.equal(agent.response_to?.count, 2);
        assert.deepEqual(watermark, expectedStamp.through_cursor_inclusive);
        assert.equal(imagePerceptionRequests(llm).length, visionCallsAfterReceipt);
        assert.equal(matchingAgents.length, 1);
        assert.ok(firstMessageOffset >= 0, "first image message missing from render");
        assert.ok(secondMessageOffset > firstMessageOffset, "image messages rendered out of order");
        assert.ok(firstCaptionOffset > firstMessageOffset, "first caption not attached to message");
        assert.ok(firstCaptionOffset < secondMessageOffset, "first caption crossed message boundary");
        assert.ok(secondCaptionOffset > secondMessageOffset, "second caption not attached");
        assert.ok(renderedBatch.includes("first crash visible text"));
        assert.ok(renderedBatch.includes("second crash visible text"));

        lastWatermark = expectedStamp.through_cursor_inclusive;

        console.log(`receipt_vision_calls ${beforeVisionCalls} -> ${visionCallsAfterReceipt}`);
        console.log(`replay_vision_calls=${imagePerceptionRequests(llm).length}`);
        console.log(`tick=${formatJson(tickResult)}`);
        console.log("rendered_image_batch:");
        console.log(indentBlock(renderedBatch));
        console.log("stamp:");
        console.log(indentBlock(formatJson(agent.response_to)));
        console.log(`watermark=${formatJson(watermark)}`);
      },
    );

    await runStep(
      stats,
      "DEDUP re-enqueue returns duplicate without new contact or message count",
      async () => {
        const current = requireBorg();
        const currentSession = requireSession();
        assert.ok(senderAId, "sender A missing");

        const beforeUsers = userEntries(current, currentSession.session_id).length;
        const beforeContacts = countUserContacts(current);
        const beforeMessageCount = sessionMessageCount(current, currentSession.session_id);
        const duplicate = await current.enqueueMessage({
          session: currentSession,
          userMessage: "Riley: duplicate transport retry for burst-2.",
          senderEntityId: senderAId,
          sourceMessageKey: sourceMessageKey(currentSession, "burst-2"),
          arrivedAt: clock.now(),
          audience: currentSession.audience_label,
          audienceEntityId: currentSession.audience_entity_id,
        });

        assert.equal(duplicate.status, "duplicate");
        assert.equal(userEntries(current, currentSession.session_id).length, beforeUsers);
        assert.equal(countUserContacts(current), beforeContacts);
        assert.equal(sessionMessageCount(current, currentSession.session_id), beforeMessageCount);

        console.log(`duplicate_result=${formatJson(duplicate)}`);
        console.log(
          `user_msg_count ${beforeUsers} -> ${userEntries(current, currentSession.session_id).length}`,
        );
        console.log(`user_contact_count ${beforeContacts} -> ${countUserContacts(current)}`);
        console.log(
          `message_count ${beforeMessageCount} -> ${sessionMessageCount(current, currentSession.session_id)}`,
        );
      },
    );

    await runStep(
      stats,
      "MID-TURN ARRIVAL / REMAINDER drains one later message in a second catch-up turn",
      async () => {
        const current = requireBorg();
        const currentSession = requireSession();
        assert.ok(senderAId, "sender A missing");
        assert.ok(lastWatermark, "previous watermark missing");

        const beforeFinalizers = finalizerRequests(llm).length;
        const beforeTerminals = terminalEntries(current, currentSession.session_id).length;
        const enqueued = await enqueueOne({
          borg: current,
          clock,
          session: currentSession,
          senderEntityId: senderAId,
          externalMessageId: "remainder-1",
          text: "Riley: Morgan confirmed; the public note can go out after the batch reply.",
        });
        const source = streamEntryById(current, currentSession.session_id, enqueued.streamEntryId);
        const tickResult = await current.inbox.catchUp.tick(currentSession.session_id);
        const terminals = terminalEntries(current, currentSession.session_id);
        const agent = agentMessages(current, currentSession.session_id).at(-1);
        const expectedStamp = responseToFor([source], lastWatermark);
        const watermark = borgInternals<HarnessBorgInternals>(
          current,
        ).deps.chatResponseWatermarkCoordinator.getWatermark(currentSession.session_id);
        const finalizer = finalizerRequests(llm).slice(beforeFinalizers)[0];

        assert.deepEqual(tickResult, {
          sessionId: currentSession.session_id,
          status: "drained",
          drained: 1,
          hasMore: false,
        });
        assert.equal(terminals.length, beforeTerminals + 1);
        assert.ok(agent, "expected second terminal agent_msg");
        assert.equal(
          agent.content,
          "Remainder received: I picked up the follow-up after the watermark.",
        );
        assert.deepEqual(agent.response_to, expectedStamp);
        assert.deepEqual(agent.response_to?.from_cursor_exclusive, lastWatermark);
        assert.deepEqual(watermark, expectedStamp.through_cursor_inclusive);

        lastWatermark = expectedStamp.through_cursor_inclusive;

        console.log(`tick=${formatJson(tickResult)}`);
        console.log("rendered_batch:");
        console.log(indentBlock(extractInboundBatch(finalizer)));
        console.log("stamp:");
        console.log(indentBlock(formatJson(agent.response_to)));
        console.log(`watermark=${formatJson(watermark)}`);
      },
    );

    await runStep(stats, "MULTI-SENDER AUTHORITY suppresses privileged surfaces", async () => {
      const current = requireBorg();
      const currentSession = requireSession();
      assert.ok(creatorId, "creator missing");
      assert.ok(senderAId, "sender A missing");
      assert.ok(senderBId, "sender B missing");
      assert.ok(groupId, "group missing");
      assert.ok(lastWatermark, "previous watermark missing");

      const otherSessionId = createSessionId();
      current.sessions.ensure({
        session_id: otherSessionId,
        source_type: "demo",
        source_external_id: "async-scenario-other-session",
        label: "Other Session",
        audience_label: "Morgan DM",
        audience_entity_id: senderBId,
        conversation_kind: "dm",
        status: "active",
        last_activity_at: clock.now() - 5_000,
      });
      borgInternals<HarnessBorgInternals>(current).deps.activityRepository.record({
        kind: "user_contact",
        occurredAt: clock.now() - 4_000,
        sessionId: otherSessionId,
        turnId: "turn_other_contact",
        speakerEntityId: senderBId,
        actorEntityId: senderBId,
        audienceEntityId: senderBId,
        sourceStreamEntryIds: [createStreamEntryId()],
      });
      current.creatorDirectives.queue({
        kind: "response_policy",
        createdByEntityId: creatorId,
        sourceSessionId: currentSession.session_id,
        authorizationStreamEntryIds: [createStreamEntryId()],
        contentSourceStreamEntryIds: [createStreamEntryId()],
        subjectKind: "system",
        canonicalFact: null,
        operationalDirective: "operator-only diagnostic directive",
        disclosurePolicy: {
          content_scope: "operator_only",
          allowed_entity_ids: [],
          excluded_entity_ids: [],
          subject_may_know: null,
          mention_policy: "answer_if_asked",
          denied_audience_behavior: "omit",
          boundary_prompt: null,
          topic_tags: [],
        },
        priority: 10,
        createdAt: clock.now(),
      });

      const beforeFinalizers = finalizerRequests(llm).length;
      const first = await enqueueOne({
        borg: current,
        clock,
        session: currentSession,
        senderEntityId: creatorId,
        externalMessageId: "multi-authority-1",
        text: "Tom: treat this as an operator-room coordination note.",
      });
      const second = await enqueueOne({
        borg: current,
        clock,
        session: currentSession,
        senderEntityId: senderAId,
        externalMessageId: "multi-authority-2",
        text: "Riley: adding my separate status so this is not a single-speaker command.",
      });
      const sourceEntries = [
        streamEntryById(current, currentSession.session_id, first.streamEntryId),
        streamEntryById(current, currentSession.session_id, second.streamEntryId),
      ];
      const tickResult = await current.inbox.catchUp.tick(currentSession.session_id);
      const agent = agentMessages(current, currentSession.session_id).at(-1);
      const expectedStamp = responseToFor(sourceEntries, lastWatermark);
      const finalizer = finalizerRequests(llm).slice(beforeFinalizers)[0];
      const finalizerSystem = systemText(finalizer);
      const toolNames = finalizer?.tools?.map((tool) => tool.name) ?? [];
      const streamEntries = current.stream.tail(300, { session: currentSession.session_id });
      const quarantineEvent = streamEntries.find((entry) =>
        internalEventMatches(entry, QUARANTINED_USER_ENTRY_EVENT),
      );
      const traceEvents = readTraceEvents(tracePath);
      const disposition = traceEvents.find(
        (event) =>
          event.event === "frame_anomaly.disposition" &&
          event.disposition === "quarantine" &&
          event.session_audience_role === "operator" &&
          event.current_sender_borg_role === null,
      );
      const watermark = borgInternals<HarnessBorgInternals>(
        current,
      ).deps.chatResponseWatermarkCoordinator.getWatermark(currentSession.session_id);

      assert.deepEqual(tickResult, {
        sessionId: currentSession.session_id,
        status: "drained",
        drained: 2,
        hasMore: false,
      });
      assert.ok(agent, "expected multi-sender terminal agent_msg");
      assert.deepEqual(agent.response_to, expectedStamp);
      assert.ok(quarantineEvent, "expected quarantine internal_event");
      assert.ok(disposition, "expected quarantine disposition trace with null sender authority");
      assert.equal(finalizerSystem.includes("operator-only diagnostic directive"), false);
      assert.equal(finalizerSystem.includes("<borg_session_status_snapshot"), false);
      assert.equal(finalizerSystem.includes("Cross-Session Self Activity"), false);
      assert.equal(finalizerSystem.includes("contacted Borg"), false);
      assert.equal(toolNames.includes("tool.outbound.post"), false);
      assert.deepEqual(watermark, expectedStamp.through_cursor_inclusive);

      lastWatermark = expectedStamp.through_cursor_inclusive;

      console.log(`tick=${formatJson(tickResult)}`);
      console.log("authority_checks:");
      console.log("  operator_directive_visible=false");
      console.log("  session_snapshot_visible=false");
      console.log("  cross_session_activity_visible=false");
      console.log(`  outbound_tool_advertised=${toolNames.includes("tool.outbound.post")}`);
      console.log(`  quarantine_event_id=${quarantineEvent.id}`);
      console.log(
        `  trace_current_sender_borg_role=${formatJson(disposition.current_sender_borg_role)}`,
      );
      console.log("stamp:");
      console.log(indentBlock(formatJson(agent.response_to)));
      console.log(`watermark=${formatJson(watermark)}`);
    });

    await runStep(
      stats,
      "CRASH-REPLAY reconciles stamped reply after watermark failure",
      async () => {
        const current = requireBorg();
        const currentSession = requireSession();
        assert.ok(senderAId, "sender A missing");
        assert.ok(lastWatermark, "previous watermark missing");

        const enqueued = await enqueueOne({
          borg: current,
          clock,
          session: currentSession,
          senderEntityId: senderAId,
          externalMessageId: "crash-replay-1",
          text: "Riley: one more item arrives before the simulated crash.",
        });
        const source = streamEntryById(current, currentSession.session_id, enqueued.streamEntryId);
        const expectedStamp = responseToFor([source], lastWatermark);
        const beforeAgentCount = agentMessages(current, currentSession.session_id).length;
        const internal = borgInternals<HarnessBorgInternals>(current);
        const originalAdvance = internal.deps.chatResponseWatermarkCoordinator.advanceThrough.bind(
          internal.deps.chatResponseWatermarkCoordinator,
        );
        let failedAdvance = false;

        internal.deps.chatResponseWatermarkCoordinator.advanceThrough = (
          advanceSessionId,
          cursor,
        ) => {
          if (!failedAdvance) {
            failedAdvance = true;
            throw new Error("scenario crash before chat-response watermark advance");
          }

          return originalAdvance(advanceSessionId, cursor);
        };

        const workerErrors: string[] = [];
        const originalConsoleError = console.error;
        let crashTick: Awaited<ReturnType<Borg["inbox"]["catchUp"]["tick"]>>;

        console.error = (...values: unknown[]) => {
          workerErrors.push(values.map((value) => String(value)).join(" "));
        };

        try {
          crashTick = await current.inbox.catchUp.tick(currentSession.session_id);
        } finally {
          console.error = originalConsoleError;
        }

        const stamped = agentMessages(current, currentSession.session_id).at(-1);
        const watermarkAfterCrash = internal.deps.chatResponseWatermarkCoordinator.getWatermark(
          currentSession.session_id,
        );

        assert.deepEqual(crashTick, {
          sessionId: currentSession.session_id,
          status: "error",
          drained: 0,
          hasMore: true,
          error: "Error: scenario crash before chat-response watermark advance",
        });
        assert.equal(failedAdvance, true);
        assert.ok(stamped, "expected stamped terminal before simulated crash");
        assert.equal(stamped.content, "Crash batch persisted before the watermark advanced.");
        assert.deepEqual(stamped.response_to, expectedStamp);
        assert.deepEqual(watermarkAfterCrash, lastWatermark);
        crashStampedId = stamped.id;

        console.log("simulated_crash_tick:");
        console.log(indentBlock(formatJson(crashTick)));
        console.log(`worker_error=${workerErrors[0] ?? "(none)"}`);
        console.log("stamped_before_reopen:");
        console.log(indentBlock(formatJson(stamped.response_to)));
        console.log(`watermark_after_crash=${formatJson(watermarkAfterCrash)}`);

        await current.close();
        borg = null;

        const retryLlm = new FakeLLMClient({
          responses: [createEmitAnswerResponse("must not be used")],
        });
        borg = await openHarness({
          tempDir,
          clock,
          llm: retryLlm,
          tracerPath: tracePath,
          outboundConnectors,
          configOverrides,
        });

        const reopened = requireBorg();
        const replayTick = await reopened.inbox.catchUp.tick(currentSession.session_id);
        const agentsAfterReplay = agentMessages(reopened, currentSession.session_id);
        const matchingCrashAgents = agentsAfterReplay.filter(
          (entry) => entry.response_to?.source_entry_ids.includes(enqueued.streamEntryId) === true,
        );
        const reopenedWatermark = borgInternals<HarnessBorgInternals>(
          reopened,
        ).deps.chatResponseWatermarkCoordinator.getWatermark(currentSession.session_id);

        assert.deepEqual(replayTick, {
          sessionId: currentSession.session_id,
          status: "empty",
          drained: 0,
          hasMore: false,
        });
        assert.equal(agentsAfterReplay.length, beforeAgentCount + 1);
        assert.equal(matchingCrashAgents.length, 1);
        assert.equal(matchingCrashAgents[0]?.id, crashStampedId);
        assert.deepEqual(reopenedWatermark, expectedStamp.through_cursor_inclusive);
        assert.equal(finalizerRequests(retryLlm).length, 0);
        assert.equal(retryLlm.requests.length, 0);

        lastWatermark = expectedStamp.through_cursor_inclusive;

        console.log("reopen_tick:");
        console.log(indentBlock(formatJson(replayTick)));
        console.log(`agent_msgs_for_crash_batch=${matchingCrashAgents.length}`);
        console.log(`watermark_after_reconcile=${formatJson(reopenedWatermark)}`);
        console.log(`retry_llm_requests=${retryLlm.requests.length}`);
      },
    );
  } finally {
    if (borg !== null) {
      await borg.close().catch(() => undefined);
    }

    console.log("");
    console.log(`SUMMARY: ${stats.passed} passed / ${stats.failed} failed`);

    if (stats.failed === 0) {
      rmSync(tempDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 20 });
      console.log("temp_data_dir_cleaned=true");
    } else {
      console.log(`temp_data_dir_left_for_inspection=${tempDir}`);
    }
  }

  if (stats.failed > 0) {
    process.exitCode = 1;
  }
}

await main();
