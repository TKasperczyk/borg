import { describe, expect, it, vi } from "vitest";

import type { SessionRecord } from "../sessions/index.js";
import { SessionBusyError } from "../util/errors.js";
import { createSessionId, createStreamEntryId } from "../util/ids.js";

import { runDirectedOutboundTurn } from "./outbound-turn.js";

const NOW_MS = 1_700_000_000_000;

function session(input: Partial<SessionRecord> = {}): SessionRecord {
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
    ...input,
  };
}

describe("runDirectedOutboundTurn", () => {
  it("propagates target-scoped delivery status from the orchestrated turn", async () => {
    const targetSession = session();
    const agentMessageId = createStreamEntryId();
    const turnOrchestrator = {
      run: vi.fn(async () => ({
        turn_id: "turn-target",
        mode: "relational" as const,
        path: "system_2" as const,
        response: "Target message",
        emitted: true,
        emission: {
          kind: "message" as const,
          content: "Target message",
          agentMessageId,
        },
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: null,
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId,
        outboundDelivery: {
          status: "transported" as const,
          streamEntryId: agentMessageId,
          sourceType: "demo" as const,
          externalMessageId: "demo-message-1",
        },
      })),
    };

    await expect(
      runDirectedOutboundTurn(
        {
          turnOrchestrator,
        },
        {
          targetSession,
          instruction: "Reach out.",
          authorizationKind: "manual_creator_operator",
        },
      ),
    ).resolves.toMatchObject({
      targetSessionId: targetSession.session_id,
      status: "completed",
      turnId: "turn-target",
      emitted: true,
      response: "Target message",
      agentMessageId,
      delivery: {
        status: "transported",
        streamEntryId: agentMessageId,
        sourceType: "demo",
        externalMessageId: "demo-message-1",
      },
    });
    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: targetSession.session_id,
        audience: "Alice",
        origin: "directed_outbound",
        userMessage: expect.stringContaining(
          "A structurally authorized creator in an operator context directed Borg",
        ),
      }),
    );
  });

  it("renders autonomous policy provenance truthfully", async () => {
    const targetSession = session();
    const turnOrchestrator = {
      run: vi.fn(async () => ({
        turn_id: "turn-target",
        mode: "relational" as const,
        path: "system_2" as const,
        response: "Target message",
        emitted: true,
        emission: {
          kind: "message" as const,
          content: "Target message",
          agentMessageId: createStreamEntryId(),
        },
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: null,
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
      })),
    };

    await runDirectedOutboundTurn(
      {
        turnOrchestrator,
      },
      {
        targetSession,
        instruction: "Reach out.",
        authorizationKind: "autonomous_policy",
      },
    );

    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        userMessage: expect.stringContaining(
          "An autonomous wake, structurally authorized by proactive outbound policy",
        ),
      }),
    );
    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        userMessage: expect.not.stringContaining("creator in an operator context"),
      }),
    );
  });

  it("scrubs internal ids from the injected instruction before target composition", async () => {
    const targetSession = session();
    const turnOrchestrator = {
      run: vi.fn(async () => ({
        turn_id: "turn-target",
        mode: "relational" as const,
        path: "system_2" as const,
        response: "Target message",
        emitted: true,
        emission: {
          kind: "message" as const,
          content: "Target message",
          agentMessageId: createStreamEntryId(),
        },
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: null,
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
      })),
    };

    await runDirectedOutboundTurn(
      {
        turnOrchestrator,
      },
      {
        targetSession,
        instruction: "Use sess_aaaaaaaaaaaaaaaa and <borg_hidden> literally.",
        authorizationKind: "manual_creator_operator",
      },
    );

    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        userMessage: expect.not.stringContaining("sess_aaaaaaaaaaaaaaaa"),
      }),
    );
    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        userMessage: expect.stringContaining("[internal_id]"),
      }),
    );
    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        userMessage: expect.stringContaining("<-borg_hidden>"),
      }),
    );
  });

  it("returns target_busy when the target session lock is occupied", async () => {
    const targetSession = session();
    const turnOrchestrator = {
      run: vi.fn(async () => {
        throw new SessionBusyError("Session is busy", {
          code: "SESSION_TURN_BUSY",
        });
      }),
    };

    await expect(
      runDirectedOutboundTurn(
        {
          turnOrchestrator,
        },
        {
          targetSession,
          instruction: "Reach out.",
          authorizationKind: "manual_creator_operator",
        },
      ),
    ).resolves.toEqual({
      targetSessionId: targetSession.session_id,
      status: "target_busy",
      emitted: false,
      response: "",
    });
  });
});
