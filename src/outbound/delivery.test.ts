import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { SessionRecord } from "../sessions/index.js";
import { StreamReader, StreamWriter, type StreamEntry } from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { createSessionId } from "../util/ids.js";

import { MessageConnectorRegistry } from "./connector-registry.js";
import { OutboundDelivery } from "./delivery.js";
import type { MessageConnector } from "./types.js";

const NOW_MS = 1_700_000_000_000;

function session(overrides: Partial<SessionRecord> = {}): SessionRecord {
  return {
    session_id: createSessionId(),
    source_type: "demo",
    source_external_id: null,
    source_url: null,
    label: "demo",
    audience_label: "Alice",
    audience_entity_id: null,
    conversation_kind: "demo",
    created_at: NOW_MS,
    last_activity_at: NOW_MS,
    last_turn_id: null,
    message_count: 0,
    status: "active",
    privacy_level: "payload_off",
    participation_policy: "active",
    audience_role: "participant",
    ...overrides,
  };
}

describe("OutboundDelivery", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("appends the outbound message to the target stream before connector delivery", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-outbound-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(NOW_MS);
    const targetSession = session();
    const delivered: StreamEntry[] = [];
    const connector: MessageConnector = {
      sourceType: "demo",
      async deliver(input) {
        delivered.push(input.streamEntry);
        return {
          status: "transported",
          externalMessageId: "demo-message-1",
        };
      },
    };
    const appended: StreamEntry[] = [];
    const delivery = new OutboundDelivery({
      connectorRegistry: new MessageConnectorRegistry([connector]),
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: tempDir,
          sessionId,
          clock,
          onAppend: (entries) => appended.push(...entries),
        }),
      clock,
    });

    const result = await delivery.deliver({
      session: targetSession,
      message: {
        content: "Outbound hello",
        streamInput: {
          turn_id: "turn-outbound",
          audience: "Alice",
        },
      },
    });

    expect(result).toMatchObject({
      status: "transported",
      sourceType: "demo",
      externalMessageId: "demo-message-1",
    });
    expect(delivered.map((entry) => entry.id)).toEqual([result.streamEntry.id]);
    expect(appended.map((entry) => entry.session_id)).toEqual([targetSession.session_id]);

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: targetSession.session_id,
    }).tail(5);
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      kind: "agent_msg",
      content: "Outbound hello",
      turn_id: "turn-outbound",
      audience: "Alice",
      session_id: targetSession.session_id,
    });
  });

  it("keeps composed messages as memory and logs when no connector is wired", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-outbound-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(NOW_MS);
    const targetSession = session({
      source_type: "slack",
    });
    const delivery = new OutboundDelivery({
      connectorRegistry: new MessageConnectorRegistry(),
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: tempDir,
          sessionId,
          clock,
        }),
      clock,
    });

    const result = await delivery.deliver({
      session: targetSession,
      message: {
        content: "Composed only",
      },
    });

    expect(result.status).toBe("composed_not_transported");

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: targetSession.session_id,
    }).tail(5);
    expect(entries.map((entry) => entry.kind)).toEqual(["agent_msg", "internal_event"]);
    expect(entries[0]?.content).toBe("Composed only");
    expect(entries[1]?.content).toMatchObject({
      event: "outbound_delivery.no_connector",
      status: "composed_not_transported",
      source_type: "slack",
      outbound_stream_entry_id: result.streamEntry.id,
    });
  });

  it("delivers only the appended agent_msg entry, never suppressed draft content", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-outbound-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(NOW_MS);
    const targetSession = session();
    const delivered: StreamEntry[] = [];
    const connector: MessageConnector = {
      sourceType: "demo",
      async deliver(input) {
        delivered.push(input.streamEntry);
        return {
          status: "transported",
          externalMessageId: "demo-message-2",
        };
      },
    };
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: targetSession.session_id,
      clock,
    });
    const delivery = new OutboundDelivery({
      connectorRegistry: new MessageConnectorRegistry([connector]),
      createStreamWriter: () => writer,
      clock,
    });

    try {
      const suppressedEntry = await writer.append({
        kind: "agent_suppressed",
        content: {
          reason: "invalid_tool_after_regenerate",
          undelivered_draft: { text: "Suppressed draft must stay out of transport." },
        },
      });
      clock.advance(10);

      const result = await delivery.deliver({
        session: targetSession,
        message: {
          content: "Transported reply",
          streamInput: {
            kind: "agent_suppressed",
            content: {
              reason: "invalid_tool_after_regenerate",
              undelivered_draft: { text: "Malformed metadata draft" },
            },
            turn_id: "turn-outbound",
          } as never,
        },
        streamWriter: writer,
      });

      expect(result.streamEntry).toMatchObject({
        kind: "agent_msg",
        content: "Transported reply",
        turn_id: "turn-outbound",
      });
      expect(delivered).toEqual([
        expect.objectContaining({
          id: result.streamEntry.id,
          kind: "agent_msg",
          content: "Transported reply",
        }),
      ]);
      expect(delivered.map((entry) => entry.id)).not.toContain(suppressedEntry.id);
    } finally {
      writer.close();
    }

    const entries = new StreamReader({
      dataDir: tempDir,
      sessionId: targetSession.session_id,
    }).tail(5);
    expect(entries.map((entry) => entry.kind)).toEqual(["agent_suppressed", "agent_msg"]);
    expect(entries[1]?.content).toBe("Transported reply");
  });
});
