import { describe, expect, it, vi } from "vitest";

import type { SessionRecord } from "../../sessions/index.js";
import type { AutonomousOutboundPolicy, MessageConnectorRegistry } from "../../outbound/index.js";
import { ToolError } from "../../util/errors.js";
import { createSessionId, createStreamEntryId, type SessionId } from "../../util/ids.js";

import { createOutboundPostTool } from "./outbound-post.js";

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

function repository(sessions: readonly SessionRecord[]) {
  const rows = new Map<SessionId, SessionRecord>(
    sessions.map((record) => [record.session_id, record]),
  );

  return {
    get: (sessionId: SessionId) => rows.get(sessionId) ?? null,
  };
}

function connectorRegistry(
  sourceTypes: readonly SessionRecord["source_type"][] = ["demo"],
): Pick<MessageConnectorRegistry, "has"> {
  return {
    has: vi.fn((sourceType) => sourceTypes.includes(sourceType)),
  };
}

describe("tool.outbound.post", () => {
  it("is available to deliberator and autonomous origins", () => {
    const tool = createOutboundPostTool({
      sessionsRepository: repository([]),
      connectorRegistry: connectorRegistry(),
      postOutbound: vi.fn(),
    });

    expect(tool.allowedOrigins).toEqual(["autonomous", "deliberator"]);
    expect(tool.writeScope).toBe("write");
  });

  it("runs a target-scoped outbound turn when a creator directs from an operator session", async () => {
    const operatorSession = session({
      audience_role: "operator",
    });
    const targetSession = session({
      audience_label: "Alice",
    });
    const deliveryStreamEntryId = createStreamEntryId();
    const postOutbound = vi.fn(async () => ({
      targetSessionId: targetSession.session_id,
      status: "completed" as const,
      turnId: "turn-target",
      emitted: true,
      response: "Target-scoped message",
      agentMessageId: "strm_aaaaaaaaaaaaaaaa",
      delivery: {
        status: "transported" as const,
        streamEntryId: deliveryStreamEntryId,
        sourceType: "demo" as const,
        externalMessageId: "demo-message-1",
      },
    }));
    const tool = createOutboundPostTool({
      sessionsRepository: repository([operatorSession, targetSession]),
      connectorRegistry: connectorRegistry(),
      postOutbound,
    });

    const output = await tool.invoke(
      {
        target_session_id: targetSession.session_id,
        instruction: "Tell Alice the checklist is ready.",
      },
      {
        sessionId: operatorSession.session_id,
        origin: "deliberator",
        sessionAudienceRole: "operator",
        currentSenderBorgRole: "creator",
      },
    );

    expect(postOutbound).toHaveBeenCalledWith({
      targetSession,
      instruction: "Tell Alice the checklist is ready.",
      authorizationKind: "manual_creator_operator",
    });
    expect(output.outbound).toMatchObject({
      target_session_id: targetSession.session_id,
      status: "completed",
      turn_id: "turn-target",
      emitted: true,
      response: "Target-scoped message",
      agent_message_id: "strm_aaaaaaaaaaaaaaaa",
      delivery: {
        status: "transported",
        stream_entry_id: deliveryStreamEntryId,
        source_type: "demo",
        external_message_id: "demo-message-1",
      },
    });
  });

  it("allows autonomous outbound only through the autonomous structural policy", async () => {
    const currentSession = session();
    const targetSession = session({
      audience_label: "Alice",
    });
    const autonomousOutboundPolicy = {
      assertAuthorized: vi.fn(),
    } satisfies Pick<AutonomousOutboundPolicy, "assertAuthorized">;
    const postOutbound = vi.fn(async () => ({
      targetSessionId: targetSession.session_id,
      status: "completed" as const,
      turnId: "turn-target",
      emitted: true,
      response: "Target-scoped message",
    }));
    const tool = createOutboundPostTool({
      sessionsRepository: repository([currentSession, targetSession]),
      connectorRegistry: connectorRegistry(),
      autonomousOutboundPolicy,
      postOutbound,
    });

    await expect(
      tool.invoke(
        {
          target_session_id: targetSession.session_id,
          instruction: "Reach out.",
        },
        {
          sessionId: currentSession.session_id,
          origin: "autonomous",
        },
      ),
    ).resolves.toMatchObject({
      outbound: {
        target_session_id: targetSession.session_id,
        status: "completed",
        response: "Target-scoped message",
      },
    });
    expect(autonomousOutboundPolicy.assertAuthorized).toHaveBeenCalledWith({
      currentSessionId: currentSession.session_id,
      targetSession,
    });
    expect(postOutbound).toHaveBeenCalledWith({
      targetSession,
      instruction: "Reach out.",
      authorizationKind: "autonomous_policy",
    });
  });

  it("passes the current autonomous turn id into structural policy checks", async () => {
    const currentSession = session();
    const targetSession = session({
      audience_label: "Alice",
    });
    const autonomousOutboundPolicy = {
      assertAuthorized: vi.fn(),
    } satisfies Pick<AutonomousOutboundPolicy, "assertAuthorized">;
    const tool = createOutboundPostTool({
      sessionsRepository: repository([currentSession, targetSession]),
      connectorRegistry: connectorRegistry(),
      autonomousOutboundPolicy,
      postOutbound: vi.fn(async () => ({
        targetSessionId: targetSession.session_id,
        status: "completed" as const,
        turnId: "turn-target",
        emitted: true,
        response: "Target-scoped message",
      })),
    });

    await tool.invoke(
      {
        target_session_id: targetSession.session_id,
        instruction: "Reach out.",
      },
      {
        sessionId: currentSession.session_id,
        origin: "autonomous",
        turnId: "turn-autonomous-current",
      },
    );

    expect(autonomousOutboundPolicy.assertAuthorized).toHaveBeenCalledWith({
      currentSessionId: currentSession.session_id,
      targetSession,
      currentTurnId: "turn-autonomous-current",
    });
  });

  it("rejects autonomous outbound when the autonomous policy is missing or denies the target", async () => {
    const currentSession = session();
    const targetSession = session();
    const missingPolicyTool = createOutboundPostTool({
      sessionsRepository: repository([currentSession, targetSession]),
      connectorRegistry: connectorRegistry(),
      postOutbound: vi.fn(),
    });

    await expect(
      missingPolicyTool.invoke(
        {
          target_session_id: targetSession.session_id,
          instruction: "Reach out.",
        },
        {
          sessionId: currentSession.session_id,
          origin: "autonomous",
        },
      ),
    ).rejects.toMatchObject({
      code: "AUTONOMOUS_OUTBOUND_NOT_CONFIGURED",
    });

    const deniedPolicyTool = createOutboundPostTool({
      sessionsRepository: repository([currentSession, targetSession]),
      connectorRegistry: connectorRegistry(),
      autonomousOutboundPolicy: {
        assertAuthorized() {
          throw new ToolError("cap exceeded", {
            code: "AUTONOMOUS_OUTBOUND_CAP_EXCEEDED",
          });
        },
      },
      postOutbound: vi.fn(),
    });

    await expect(
      deniedPolicyTool.invoke(
        {
          target_session_id: targetSession.session_id,
          instruction: "Reach out.",
        },
        {
          sessionId: currentSession.session_id,
          origin: "autonomous",
        },
      ),
    ).rejects.toMatchObject({
      code: "AUTONOMOUS_OUTBOUND_CAP_EXCEEDED",
    });
  });

  it("rejects structurally unauthorized senders", async () => {
    const currentSession = session({
      audience_role: "operator",
    });
    const targetSession = session();
    const tool = createOutboundPostTool({
      sessionsRepository: repository([currentSession, targetSession]),
      connectorRegistry: connectorRegistry(),
      postOutbound: vi.fn(),
    });

    await expect(
      tool.invoke(
        {
          target_session_id: targetSession.session_id,
          instruction: "Reach out.",
        },
        {
          sessionId: currentSession.session_id,
          origin: "deliberator",
          sessionAudienceRole: "operator",
          currentSenderBorgRole: null,
        },
      ),
    ).rejects.toMatchObject({
      code: "OUTBOUND_POST_UNAUTHORIZED",
    });

    await expect(
      tool.invoke(
        {
          target_session_id: targetSession.session_id,
          instruction: "Reach out.",
        },
        {
          sessionId: currentSession.session_id,
          origin: "deliberator",
          sessionAudienceRole: "participant",
          currentSenderBorgRole: "creator",
        },
      ),
    ).rejects.toMatchObject({
      code: "OUTBOUND_POST_UNAUTHORIZED",
    });
  });

  it("returns composed_not_transported for manual targets without a wired connector", async () => {
    const operatorSession = session({
      audience_role: "operator",
    });
    const targetSession = session({
      source_type: "slack",
    });
    const postOutbound = vi.fn();
    const tool = createOutboundPostTool({
      sessionsRepository: repository([operatorSession, targetSession]),
      connectorRegistry: connectorRegistry(["demo"]),
      postOutbound,
    });

    await expect(
      tool.invoke(
        {
          target_session_id: targetSession.session_id,
          instruction: "Reach out.",
        },
        {
          sessionId: operatorSession.session_id,
          origin: "deliberator",
          sessionAudienceRole: "operator",
          currentSenderBorgRole: "creator",
        },
      ),
    ).resolves.toEqual({
      outbound: {
        target_session_id: targetSession.session_id,
        status: "not_transportable",
        emitted: false,
        response: "",
        delivery: {
          status: "composed_not_transported",
          source_type: "slack",
          error: "No wired outbound connector for target session source_type.",
        },
      },
    });
    expect(postOutbound).not.toHaveBeenCalled();
  });

  it("allows tool.outbound.post only once per directing turn", async () => {
    const operatorSession = session({
      audience_role: "operator",
    });
    const targetSession = session();
    const postOutbound = vi.fn(async () => ({
      targetSessionId: targetSession.session_id,
      status: "completed" as const,
      emitted: true,
      response: "Target-scoped message",
    }));
    const tool = createOutboundPostTool({
      sessionsRepository: repository([operatorSession, targetSession]),
      connectorRegistry: connectorRegistry(),
      postOutbound,
    });
    const context = {
      sessionId: operatorSession.session_id,
      origin: "deliberator" as const,
      sessionAudienceRole: "operator" as const,
      currentSenderBorgRole: "creator" as const,
      turnId: "turn-directing",
    };

    await tool.invoke(
      {
        target_session_id: targetSession.session_id,
        instruction: "Reach out.",
      },
      context,
    );

    await expect(
      tool.invoke(
        {
          target_session_id: targetSession.session_id,
          instruction: "Reach out again.",
        },
        context,
      ),
    ).rejects.toMatchObject({
      code: "OUTBOUND_POST_ALREADY_USED_IN_TURN",
    });
    expect(postOutbound).toHaveBeenCalledTimes(1);
  });

  it("returns a structural busy result when the target session is occupied", async () => {
    const operatorSession = session({
      audience_role: "operator",
    });
    const targetSession = session();
    const tool = createOutboundPostTool({
      sessionsRepository: repository([operatorSession, targetSession]),
      connectorRegistry: connectorRegistry(),
      postOutbound: vi.fn(async () => ({
        targetSessionId: targetSession.session_id,
        status: "target_busy" as const,
        emitted: false,
        response: "",
      })),
    });

    await expect(
      tool.invoke(
        {
          target_session_id: targetSession.session_id,
          instruction: "Reach out.",
        },
        {
          sessionId: operatorSession.session_id,
          origin: "deliberator",
          sessionAudienceRole: "operator",
          currentSenderBorgRole: "creator",
        },
      ),
    ).resolves.toEqual({
      outbound: {
        target_session_id: targetSession.session_id,
        status: "target_busy",
        emitted: false,
        response: "",
      },
    });
  });

  it("rejects invalid target shape structurally", async () => {
    const operatorSession = session({
      audience_role: "operator",
    });
    const archivedTarget = session({
      status: "archived",
    });
    const tool = createOutboundPostTool({
      sessionsRepository: repository([operatorSession, archivedTarget]),
      connectorRegistry: connectorRegistry(),
      postOutbound: vi.fn(),
    });

    await expect(
      tool.invoke(
        {
          target_session_id: operatorSession.session_id,
          instruction: "Self target.",
        },
        {
          sessionId: operatorSession.session_id,
          origin: "deliberator",
          sessionAudienceRole: "operator",
        },
      ),
    ).rejects.toMatchObject({
      code: "OUTBOUND_POST_TARGET_IS_CURRENT_SESSION",
    });

    await expect(
      tool.invoke(
        {
          target_session_id: archivedTarget.session_id,
          instruction: "Archived target.",
        },
        {
          sessionId: operatorSession.session_id,
          origin: "deliberator",
          sessionAudienceRole: "operator",
        },
      ),
    ).rejects.toMatchObject({
      code: "OUTBOUND_POST_TARGET_ARCHIVED",
    });
  });
});
