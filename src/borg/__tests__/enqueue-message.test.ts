import { afterEach, describe, expect, it } from "vitest";

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

describe("Borg.enqueueMessage", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  async function openHarness() {
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
      llmClient: new FakeLLMClient(),
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

      const first = await borg.enqueueMessage({
        session,
        userMessage: "hello from the daemon",
        senderEntityId,
        sourceMessageKey,
        arrivedAt: 5_000,
        audience: "Demo room",
        audienceEntityId,
      });
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
      expect(internal.deps.entryIndex.lookupBySourceMessageKey(sourceMessageKey)?.entry_id).toBe(
        first.streamEntryId,
      );
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

  it("rejects unknown senders and non-empty attachments before appending", async () => {
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
          attachments: [{}],
        }),
      ).rejects.toMatchObject({
        code: "ENQUEUE_ATTACHMENTS_UNSUPPORTED",
      });
      expect(borg.stream.tail(10, { session: sessionId })).toEqual([]);
    } finally {
      await borg.close().catch(() => undefined);
    }
  });
});
