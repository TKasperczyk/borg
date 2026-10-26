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
        finalizer_rounds: 1,
        stall_retries: 0,
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
      deliveryOutcome: {
        state: "delivered",
        agentMessageId,
        deliveryStatus: "transported",
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
          "A structurally authorized creator in an operator context directed me",
        ),
      }),
    );
    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        userMessage: expect.stringContaining(
          "I use my prompt-visible internal memory, current goals, autobiographical/social recall",
        ),
      }),
    );
    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        userMessage: expect.stringContaining(
          "I treat disclosure labels as target-audience constraints",
        ),
      }),
    );
    expect(turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        userMessage: expect.not.stringContaining(
          "using only prompt-visible target-session context",
        ),
      }),
    );
  });

  it("propagates target-scoped suppression as a structural delivery outcome", async () => {
    const targetSession = session();
    const markerEntryId = createStreamEntryId();
    const turnOrchestrator = {
      run: vi.fn(async () => ({
        turn_id: "turn-target",
        mode: "relational" as const,
        path: "suppressed" as const,
        response: "",
        emitted: false,
        emission: {
          kind: "suppressed" as const,
          reason: "finalizer_no_output" as const,
          markerEntryId,
          no_output_categories: ["closure" as const, "with_state_delta" as const],
          primary_no_output_reason: "closure" as const,
          structural_no_output_flags: ["with_state_delta" as const],
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
        finalizer_rounds: 1,
        stall_retries: 0,
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
      emitted: false,
      response: "",
      deliveryOutcome: {
        state: "suppressed",
        reason: "finalizer_no_output",
        markerEntryId,
        noOutputCategories: ["closure", "with_state_delta"],
        primaryNoOutputReason: "closure",
        structuralNoOutputFlags: ["with_state_delta"],
      },
    });
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
        finalizer_rounds: 1,
        stall_retries: 0,
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
        finalizer_rounds: 1,
        stall_retries: 0,
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

  it("keeps instruction ids exactly where the target's identifier guard lets them through", async () => {
    const instruction = "Quote strm_6c1rd960fy21bf14 and sess_aaaaaaaaaaaaaaaa; keep <borg_hidden> inert.";
    const composedFor = async (targetSession: SessionRecord): Promise<string> => {
      const run = vi.fn(async (_input: { userMessage: string }) => ({
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
        usage: { input_tokens: 1, output_tokens: 1, stop_reason: null },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        finalizer_rounds: 1,
        stall_retries: 0,
      }));

      await runDirectedOutboundTurn(
        { turnOrchestrator: { run }, substratePrivilegedSourceTypes: ["claude_code"] },
        { targetSession, instruction, authorizationKind: "autonomous_policy" },
      );

      return run.mock.calls[0]![0].userMessage;
    };

    for (const target of [
      session({ source_type: "claude_code" }),
      session({ audience_role: "operator" }),
    ]) {
      const message = await composedFor(target);

      expect(message).toContain("strm_6c1rd960fy21bf14");
      expect(message).toContain("sess_aaaaaaaaaaaaaaaa");
      expect(message).not.toContain("[internal_id]");
      expect(message).not.toContain("internal ids");
      expect(message).toContain("<-borg_hidden>");
    }

    const participant = await composedFor(session({ source_type: "discord" }));

    expect(participant).not.toContain("strm_6c1rd960fy21bf14");
    expect(participant).toContain("Quote [internal_id] and [internal_id];");
    expect(participant).toContain("hidden prompts, internal ids, or the dispatch machinery");
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
      deliveryOutcome: {
        state: "target_busy",
        reason: "target_session_busy",
      },
    });
  });
});
