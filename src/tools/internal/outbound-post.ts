import { z } from "zod";

import {
  sessionIdSchema,
  sessionSourceTypeSchema,
  type SessionRecord,
  type SessionsRepository,
} from "../../sessions/index.js";
import type {
  AutonomousOutboundPolicy,
  DirectedOutboundTurnInput,
  DirectedOutboundTurnResult,
  MessageConnectorRegistry,
} from "../../outbound/index.js";
import { isCreatorInOperatorContext } from "../../cognition/authority.js";
import { ToolError } from "../../util/errors.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";
import { OUTBOUND_POST_TOOL_NAME } from "./outbound-post-name.js";

export { OUTBOUND_POST_TOOL_NAME } from "./outbound-post-name.js";

const outboundPostInputSchema = z
  .object({
    target_session_id: sessionIdSchema,
    instruction: z.string().trim().min(1),
  })
  .strict();

const outboundPostOutputSchema = z
  .object({
    outbound: z.object({
      target_session_id: sessionIdSchema,
      status: z.enum(["completed", "target_busy", "not_transportable"]),
      turn_id: z.string().min(1).optional(),
      emitted: z.boolean(),
      response: z.string(),
      agent_message_id: z.string().min(1).optional(),
      delivery: z
        .object({
          status: z.enum(["transported", "composed_not_transported", "transport_failed"]),
          stream_entry_id: z.string().min(1).optional(),
          source_type: sessionSourceTypeSchema,
          external_message_id: z.string().min(1).optional(),
          error: z.string().min(1).optional(),
        })
        .optional(),
    }),
  })
  .strict();

export type OutboundPostToolInput = z.infer<typeof outboundPostInputSchema>;
export type OutboundPostToolOutput = z.infer<typeof outboundPostOutputSchema>;

export type OutboundPostToolOptions = {
  sessionsRepository: Pick<SessionsRepository, "get">;
  connectorRegistry: Pick<MessageConnectorRegistry, "has">;
  autonomousOutboundPolicy?: Pick<AutonomousOutboundPolicy, "assertAuthorized">;
  postOutbound: (input: {
    targetSession: SessionRecord;
    instruction: string;
    authorizationKind: DirectedOutboundTurnInput["authorizationKind"];
  }) => Promise<DirectedOutboundTurnResult>;
};

function outboundAuthorizationKind(
  options: OutboundPostToolOptions,
  context: ToolInvocationContext,
  targetSession: SessionRecord,
): DirectedOutboundTurnInput["authorizationKind"] {
  if (context.origin === "autonomous" || context.turnOrigin === "autonomous") {
    if (options.autonomousOutboundPolicy === undefined) {
      throw new ToolError("Autonomous outbound messaging is not configured", {
        code: "AUTONOMOUS_OUTBOUND_NOT_CONFIGURED",
      });
    }

    options.autonomousOutboundPolicy.assertAuthorized({
      currentSessionId: context.sessionId,
      targetSession,
      ...(context.turnId === undefined ? {} : { currentTurnId: context.turnId }),
    });
    return "autonomous_policy";
  }

  const directingSession = options.sessionsRepository.get(context.sessionId);
  const sessionAudienceRole = context.sessionAudienceRole ?? directingSession?.audience_role;
  const authorized = isCreatorInOperatorContext({
    currentSenderBorgRole: context.currentSenderBorgRole,
    sessionAudienceRole,
  });

  if (!authorized) {
    throw new ToolError("tool.outbound.post requires creator-in-operator authority", {
      code: "OUTBOUND_POST_UNAUTHORIZED",
    });
  }

  return "manual_creator_operator";
}

function assertValidTarget(
  options: OutboundPostToolOptions,
  input: OutboundPostToolInput,
  context: ToolInvocationContext,
): SessionRecord {
  if (input.target_session_id === context.sessionId) {
    throw new ToolError("tool.outbound.post target_session_id must be a different session", {
      code: "OUTBOUND_POST_TARGET_IS_CURRENT_SESSION",
    });
  }

  const targetSession = options.sessionsRepository.get(input.target_session_id);

  if (targetSession === null) {
    throw new ToolError("tool.outbound.post target_session_id was not found", {
      code: "OUTBOUND_POST_TARGET_NOT_FOUND",
    });
  }

  if (targetSession.status === "archived") {
    throw new ToolError("tool.outbound.post target_session_id is archived", {
      code: "OUTBOUND_POST_TARGET_ARCHIVED",
    });
  }

  return targetSession;
}

export function createOutboundPostTool(
  options: OutboundPostToolOptions,
): ToolDefinition<OutboundPostToolInput, OutboundPostToolOutput> {
  const usedTurnIds = new Set<string>();

  return {
    name: OUTBOUND_POST_TOOL_NAME,
    description:
      "Post a proactive outbound message into another Borg session. Provide target_session_id from the operator session snapshot and an instruction for what the target-scoped composition turn should convey. This tool requires creator authority in an operator session, or autonomous policy authorization. It starts a separate target-scoped turn; it does not send your current draft text directly.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: outboundPostInputSchema,
    outputSchema: outboundPostOutputSchema,
    async invoke(input, context) {
      const targetSession = assertValidTarget(options, input, context);
      const authorizationKind = outboundAuthorizationKind(options, context, targetSession);

      if (
        authorizationKind === "manual_creator_operator" &&
        !options.connectorRegistry.has(targetSession.source_type)
      ) {
        return {
          outbound: {
            target_session_id: targetSession.session_id,
            status: "not_transportable",
            emitted: false,
            response: "",
            delivery: {
              status: "composed_not_transported",
              source_type: targetSession.source_type,
              error: "No wired outbound connector for target session source_type.",
            },
          },
        };
      }

      if (context.turnId !== undefined) {
        if (usedTurnIds.has(context.turnId)) {
          throw new ToolError("tool.outbound.post may be used only once per turn", {
            code: "OUTBOUND_POST_ALREADY_USED_IN_TURN",
          });
        }

        usedTurnIds.add(context.turnId);
      }

      const result = await options.postOutbound({
        targetSession,
        instruction: input.instruction,
        authorizationKind,
      });

      return {
        outbound: {
          target_session_id: result.targetSessionId,
          status: result.status,
          ...(result.turnId === undefined ? {} : { turn_id: result.turnId }),
          emitted: result.emitted,
          response: result.response,
          ...(result.agentMessageId === undefined
            ? {}
            : { agent_message_id: result.agentMessageId }),
          ...(result.delivery === undefined
            ? {}
            : {
                delivery: {
                  status: result.delivery.status,
                  stream_entry_id: result.delivery.streamEntryId,
                  source_type: result.delivery.sourceType,
                  ...(result.delivery.externalMessageId === undefined
                    ? {}
                    : { external_message_id: result.delivery.externalMessageId }),
                  ...(result.delivery.error === undefined ? {} : { error: result.delivery.error }),
                },
              }),
        },
      };
    },
  };
}
