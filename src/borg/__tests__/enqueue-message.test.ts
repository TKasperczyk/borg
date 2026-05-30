import { existsSync, readFileSync } from "node:fs";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  Borg,
  FakeLLMClient,
  ManualClock,
  ScriptedEmbeddingClient,
  borgInternals,
  createEntityId,
  createSessionId,
  createTestConfig,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

const GIF_1X1 = Uint8Array.from([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00]);

function createImagePerceptionResponse() {
  return {
    messageBlocks: [
      {
        type: "tool_use" as const,
        id: "toolu_image",
        name: "EmitImagePerception",
        input: {
          caption: "small test image",
          image_kind: "other",
          visible_text: [],
          objects: [],
          people_or_roles: [],
          scene: "test fixture",
          colors_and_visual_attributes: [],
          spatial_relationships: [],
          possible_user_relevant_details: [],
          search_terms: ["test image"],
          uncertainties: [],
        },
      },
    ],
    input_tokens: 4,
    output_tokens: 4,
    stop_reason: "tool_use",
  };
}

describe("Borg.enqueueMessage", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  async function openHarness(options: { llmClient?: FakeLLMClient } = {}) {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-enqueue-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
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
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: options.llmClient ?? new FakeLLMClient(),
      liveExtraction: false,
    });

    return { borg, clock };
  }

  it("durably enqueues one source-keyed user message and dedupes repeats", async () => {
    const { borg } = await openHarness();

    try {
      const sessionId = createSessionId();
      const senderEntityId = borg.entities.resolve("Sender");
      const audienceEntityId = borg.entities.resolve("Demo room", { kind: "group" });
      const session = {
        session_id: sessionId,
        source_type: "demo" as const,
        source_external_id: "conversation-1",
        label: "Demo",
        audience_label: "Demo room",
        audience_entity_id: audienceEntityId,
        conversation_kind: "thread" as const,
      };
      const sourceMessageKey = {
        source_type: "demo",
        source_external_id: "conversation-1",
        external_message_id: "message-1",
      };
      const notify = vi.spyOn(
        borgInternals<{
          deps: {
            chatResponseCatchUpWorker: {
              onPendingSession(sessionId: unknown, pendingAt: number): void;
            };
          };
        }>(borg).deps.chatResponseCatchUpWorker,
        "onPendingSession",
      );

      const first = await borg.enqueueMessage({
        session,
        userMessage: "hello from the daemon",
        senderEntityId,
        sourceMessageKey,
        arrivedAt: 5_000,
        audience: "Demo room",
        audienceEntityId,
      });
      expect(notify).toHaveBeenCalledTimes(1);
      const duplicate = await borg.enqueueMessage({
        session,
        userMessage: "hello from the daemon",
        senderEntityId,
        sourceMessageKey,
        arrivedAt: 6_000,
        audience: "Demo room",
        audienceEntityId,
      });
      const entries = borg.stream.tail(10, { session: sessionId });
      const internal = borgInternals<{
        deps: {
          entryIndex: {
            lookupBySourceMessageKey(key: typeof sourceMessageKey): { entry_id: string } | null;
          };
          sqlite: {
            prepare(sql: string): {
              all(): unknown[];
            };
          };
        };
      }>(borg);
      const activityRows = internal.deps.sqlite
        .prepare(
          `SELECT kind, occurred_at, session_id, turn_id, speaker_entity_id, actor_entity_id,
                  audience_entity_id, participant_entity_ids, source_stream_entry_ids, status
           FROM activity_events`,
        )
        .all() as Array<{
        kind: string;
        occurred_at: number;
        session_id: string;
        turn_id: string | null;
        speaker_entity_id: string | null;
        actor_entity_id: string | null;
        audience_entity_id: string | null;
        participant_entity_ids: string;
        source_stream_entry_ids: string;
        status: string;
      }>;
      const sessionRecord = borg.sessions.get(sessionId);

      expect(first.status).toBe("enqueued");
      expect(duplicate).toEqual({
        status: "duplicate",
        sessionId,
        streamEntryId: first.streamEntryId,
      });
      expect(entries).toHaveLength(1);
      expect(entries[0]).toMatchObject({
        id: first.streamEntryId,
        kind: "user_msg",
        content: "hello from the daemon",
        sender_entity_id: senderEntityId,
        source_message_key: sourceMessageKey,
      });
      expect(entries[0]).not.toHaveProperty("turn_id");
      expect(entries[0]).not.toHaveProperty("receipt_pending");
      expect(internal.deps.entryIndex.lookupBySourceMessageKey(sourceMessageKey)?.entry_id).toBe(
        first.streamEntryId,
      );
      expect(notify).toHaveBeenCalledWith(sessionId, entries[0]?.timestamp);
      expect(activityRows).toEqual([
        {
          kind: "user_contact",
          occurred_at: 5_000,
          session_id: sessionId,
          turn_id: null,
          speaker_entity_id: senderEntityId,
          actor_entity_id: senderEntityId,
          audience_entity_id: audienceEntityId,
          participant_entity_ids: JSON.stringify([senderEntityId, audienceEntityId]),
          source_stream_entry_ids: JSON.stringify([first.streamEntryId]),
          status: "active",
        },
      ]);
      expect(sessionRecord?.message_count).toBe(1);
      expect(sessionRecord?.last_activity_at).toBe(5_000);
      expect(sessionRecord?.last_turn_id).toBeNull();
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("accepts image attachments and stores blob, attachment ref, and durable perception", async () => {
    const llmClient = new FakeLLMClient({
      responses: [createImagePerceptionResponse()],
    });
    const { borg } = await openHarness({ llmClient });

    try {
      const sessionId = createSessionId();
      const senderEntityId = borg.entities.resolve("Sender");
      const audienceEntityId = borg.entities.resolve("Demo room", { kind: "group" });
      const session = {
        session_id: sessionId,
        source_type: "demo" as const,
        source_external_id: "conversation-1",
        label: "Demo",
        audience_label: "Demo room",
        audience_entity_id: audienceEntityId,
        conversation_kind: "thread" as const,
      };
      const sourceMessageKey = {
        source_type: "demo",
        source_external_id: "conversation-1",
        external_message_id: "message-with-image",
      };

      const first = await borg.enqueueMessage({
        session,
        userMessage: "image attached",
        senderEntityId,
        sourceMessageKey,
        arrivedAt: 5_000,
        audience: "Demo room",
        audienceEntityId,
        attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
      });
      const duplicate = await borg.enqueueMessage({
        session,
        userMessage: "image attached",
        senderEntityId,
        sourceMessageKey,
        arrivedAt: 6_000,
        audience: "Demo room",
        audienceEntityId,
        attachments: [{ mediaType: "image/gif", bytes: GIF_1X1 }],
      });
      const entries = borg.stream.tail(10, { session: sessionId });
      const internal = borgInternals<{
        deps: {
          config: { dataDir: string };
          sqlite: {
            prepare(sql: string): {
              all(...params: unknown[]): unknown[];
            };
          };
        };
      }>(borg);
      const attachmentRows = internal.deps.sqlite
        .prepare(
          `SELECT attachment_id, media_type, byte_size, width, height, storage_ref,
                  parent_entry_id, stream_entry_id, parent_turn_id, perception_id,
                  text_embedding_ref, active
           FROM stream_attachments`,
        )
        .all() as Array<{
        attachment_id: string;
        media_type: string;
        byte_size: number;
        width: number;
        height: number;
        storage_ref: string;
        parent_entry_id: string;
        stream_entry_id: string | null;
        parent_turn_id: string | null;
        perception_id: string | null;
        text_embedding_ref: string | null;
        active: number;
      }>;
      const perceptionRows = internal.deps.sqlite
        .prepare(
          `SELECT artifact_id, attachment_id, parent_entry_id, parent_turn_id,
                  stream_entry_id, active
           FROM image_perception_artifacts`,
        )
        .all() as Array<{
        artifact_id: string;
        attachment_id: string;
        parent_entry_id: string;
        parent_turn_id: string | null;
        stream_entry_id: string | null;
        active: number;
      }>;
      const payloadRows = internal.deps.sqlite
        .prepare("SELECT caption, image_kind FROM image_perception_payloads")
        .all() as Array<{ caption: string; image_kind: string }>;
      const imageEntry = entries.find((entry) => entry.kind === "user_image_attachment");

      expect(first.status).toBe("enqueued");
      expect(duplicate).toEqual({
        status: "duplicate",
        sessionId,
        streamEntryId: first.streamEntryId,
      });
      expect(entries.map((entry) => entry.kind)).toEqual(["user_msg", "user_image_attachment"]);
      expect(imageEntry).toBeDefined();
      expect(imageEntry).not.toHaveProperty("turn_id");
      expect(imageEntry?.content).toMatchObject({
        type: "image_ref",
        media_type: "image/gif",
        parent_entry_id: first.streamEntryId,
      });
      expect(attachmentRows).toHaveLength(1);
      expect(attachmentRows[0]).toMatchObject({
        media_type: "image/gif",
        byte_size: GIF_1X1.byteLength,
        width: 1,
        height: 1,
        parent_entry_id: first.streamEntryId,
        stream_entry_id: imageEntry?.id,
        parent_turn_id: null,
        active: 1,
      });
      expect(attachmentRows[0]?.perception_id).not.toBeNull();
      expect(attachmentRows[0]?.text_embedding_ref).not.toBeNull();
      expect(existsSync(join(internal.deps.config.dataDir, attachmentRows[0]!.storage_ref))).toBe(
        true,
      );
      expect(
        readFileSync(join(internal.deps.config.dataDir, attachmentRows[0]!.storage_ref)),
      ).toEqual(Buffer.from(GIF_1X1));
      expect(perceptionRows).toEqual([
        {
          artifact_id: attachmentRows[0]?.perception_id,
          attachment_id: attachmentRows[0]?.attachment_id,
          parent_entry_id: first.streamEntryId,
          parent_turn_id: null,
          stream_entry_id: imageEntry?.id,
          active: 1,
        },
      ]);
      expect(payloadRows).toEqual([
        {
          caption: "small test image",
          image_kind: "other",
        },
      ]);
      expect(llmClient.converseRequests).toHaveLength(1);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });

  it("rejects unknown senders and unsupported attachments before appending", async () => {
    const { borg } = await openHarness();

    try {
      const sessionId = createSessionId();
      const senderEntityId = borg.entities.resolve("Sender");
      const session = {
        session_id: sessionId,
        source_type: "demo" as const,
        source_external_id: "conversation-1",
        label: "Demo",
        audience_label: "Demo room",
        conversation_kind: "thread" as const,
      };
      const sourceMessageKey = {
        source_type: "demo",
        source_external_id: "conversation-1",
        external_message_id: "message-1",
      };

      await expect(
        borg.enqueueMessage({
          session,
          userMessage: "hello",
          senderEntityId: createEntityId(),
          sourceMessageKey,
        }),
      ).rejects.toMatchObject({
        code: "ENQUEUE_SENDER_UNKNOWN",
      });
      await expect(
        borg.enqueueMessage({
          session,
          userMessage: "hello",
          senderEntityId,
          sourceMessageKey,
          attachments: [
            {
              mediaType: "image/bmp",
              bytes: Uint8Array.of(1, 2, 3),
            } as never,
          ],
        }),
      ).rejects.toMatchObject({
        code: "ATTACHMENT_UNSUPPORTED_MEDIA_TYPE",
      });
      expect(borg.stream.tail(10, { session: sessionId })).toEqual([]);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });
});
