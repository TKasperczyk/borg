import { z } from "zod";

import {
  sessionIdSchema,
  sessionSourceTypeSchema,
  type SessionRecord,
  type SessionsRepository,
} from "../../sessions/index.js";
import type { ActionRecord, ActionRepository, ActionState } from "../../memory/actions/index.js";
import type {
  AutonomousOutboundPolicy,
  DirectedOutboundDeliveryOutcome,
  DirectedOutboundTurnInput,
  DirectedOutboundTurnResult,
  MessageConnectorRegistry,
} from "../../outbound/index.js";
import { isCreatorInOperatorContext } from "../../cognition/authority.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { ToolError } from "../../util/errors.js";
import { createActionId, type StreamEntryId } from "../../util/ids.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";
import { OUTBOUND_POST_TOOL_NAME } from "./outbound-post-name.js";

export { OUTBOUND_POST_TOOL_NAME } from "./outbound-post-name.js";

const outboundPostInputSchema = z
  .object({
    target_session_id: sessionIdSchema,
    instruction: z.string().trim().min(1),
  })
  .strict();

const outboundDeliveryOutcomeSchema = z.discriminatedUnion("state", [
  z
    .object({
      state: z.literal("delivered"),
      agent_message_id: z.string().min(1),
      delivery_status: z
        .enum(["transported", "composed_not_transported", "transport_failed"])
        .optional(),
      source_type: sessionSourceTypeSchema.optional(),
      external_message_id: z.string().min(1).optional(),
    })
    .strict(),
  z
    .object({
      state: z.literal("suppressed"),
      reason: z.string().min(1),
      marker_entry_id: z.string().min(1).optional(),
      no_output_categories: z.array(z.string().min(1)).optional(),
      primary_no_output_reason: z.string().min(1).optional(),
      structural_no_output_flags: z.array(z.string().min(1)).optional(),
    })
    .strict(),
  z
    .object({
      state: z.literal("not_emitted"),
      emission_kind: z.enum(["observed", "continue_thought"]),
      marker_entry_id: z.string().min(1).optional(),
    })
    .strict(),
  z
    .object({
      state: z.literal("not_transportable"),
      reason: z.enum(["no_wired_outbound_connector", "composed_not_transported"]),
      agent_message_id: z.string().min(1).optional(),
      stream_entry_id: z.string().min(1).optional(),
      source_type: sessionSourceTypeSchema,
      error: z.string().min(1).optional(),
    })
    .strict(),
  z
    .object({
      state: z.literal("transport_failed"),
      reason: z.literal("transport_failed"),
      agent_message_id: z.string().min(1),
      stream_entry_id: z.string().min(1),
      source_type: sessionSourceTypeSchema,
      error: z.string().min(1).optional(),
    })
    .strict(),
  z
    .object({
      state: z.literal("target_busy"),
      reason: z.literal("target_session_busy"),
    })
    .strict(),
]);

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
      delivery_outcome: outboundDeliveryOutcomeSchema,
    }),
  })
  .strict();

export type OutboundPostToolInput = z.infer<typeof outboundPostInputSchema>;
export type OutboundPostToolOutput = z.infer<typeof outboundPostOutputSchema>;

export type OutboundPostToolOptions = {
  sessionsRepository: Pick<SessionsRepository, "get">;
  connectorRegistry: Pick<MessageConnectorRegistry, "has">;
  autonomousOutboundPolicy?: Pick<AutonomousOutboundPolicy, "assertAuthorized">;
  actionRepository: Pick<ActionRepository, "add">;
  clock?: Clock;
  postOutbound: (input: {
    targetSession: SessionRecord;
    instruction: string;
    authorizationKind: DirectedOutboundTurnInput["authorizationKind"];
  }) => Promise<DirectedOutboundTurnResult>;
};

type OutboundPostDeliveryOutcome = OutboundPostToolOutput["outbound"]["delivery_outcome"];

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

function deliveryOutcomeForTool(
  outcome: DirectedOutboundDeliveryOutcome,
): OutboundPostDeliveryOutcome {
  switch (outcome.state) {
    case "delivered":
      return {
        state: "delivered",
        agent_message_id: outcome.agentMessageId,
        ...(outcome.deliveryStatus === undefined
          ? {}
          : { delivery_status: outcome.deliveryStatus }),
        ...(outcome.sourceType === undefined ? {} : { source_type: outcome.sourceType }),
        ...(outcome.externalMessageId === undefined
          ? {}
          : { external_message_id: outcome.externalMessageId }),
      };
    case "suppressed":
      return {
        state: "suppressed",
        reason: outcome.reason,
        ...(outcome.markerEntryId === undefined ? {} : { marker_entry_id: outcome.markerEntryId }),
        ...(outcome.noOutputCategories === undefined
          ? {}
          : { no_output_categories: [...outcome.noOutputCategories] }),
        ...(outcome.primaryNoOutputReason === undefined
          ? {}
          : { primary_no_output_reason: outcome.primaryNoOutputReason }),
        ...(outcome.structuralNoOutputFlags === undefined
          ? {}
          : { structural_no_output_flags: [...outcome.structuralNoOutputFlags] }),
      };
    case "not_emitted":
      return {
        state: "not_emitted",
        emission_kind: outcome.emissionKind,
        ...(outcome.markerEntryId === undefined ? {} : { marker_entry_id: outcome.markerEntryId }),
      };
    case "not_transportable":
      return {
        state: "not_transportable",
        reason: outcome.reason,
        agent_message_id: outcome.agentMessageId,
        stream_entry_id: outcome.streamEntryId,
        source_type: outcome.sourceType,
        ...(outcome.error === undefined ? {} : { error: outcome.error }),
      };
    case "transport_failed":
      return {
        state: "transport_failed",
        reason: outcome.reason,
        agent_message_id: outcome.agentMessageId,
        stream_entry_id: outcome.streamEntryId,
        source_type: outcome.sourceType,
        ...(outcome.error === undefined ? {} : { error: outcome.error }),
      };
    case "target_busy":
      return {
        state: "target_busy",
        reason: outcome.reason,
      };
  }
}

function structuralOutcomeReason(outcome: OutboundPostDeliveryOutcome): string {
  switch (outcome.state) {
    case "delivered":
      return "delivered";
    case "suppressed":
      return outcome.reason;
    case "not_emitted":
      return outcome.emission_kind;
    case "not_transportable":
    case "transport_failed":
    case "target_busy":
      return outcome.reason;
  }
}

function actionStateForOutcome(outcome: OutboundPostDeliveryOutcome): ActionState {
  return outcome.state === "delivered" ? "completed" : "not_done";
}

// Whether the target-scoped turn was ever entered. Two outcomes refuse before dispatch: a held
// session lock, and the pre-flight no-connector return below, which never calls postOutbound. The
// tool's `not_transportable` widens the turn runner's variant with that second reason, so there the
// reason literal rather than the state separates "composed and stranded" from "never dispatched".
function outboundAttemptStartedTargetTurn(outcome: OutboundPostDeliveryOutcome): boolean {
  switch (outcome.state) {
    case "delivered":
    case "suppressed":
    case "not_emitted":
    case "transport_failed":
      return true;
    case "not_transportable":
      return outcome.reason === "composed_not_transported";
    case "target_busy":
      return false;
  }
}

// An ActionRecord carries exactly one free-text field, and on the two outcomes above where the
// target turn never started it is the entire record of the reach: no agent message, no suppression
// marker, no delivery event exists to read it against. `Outbound post to X: <reason>.` names a post
// in every case, so on those two it asserted a composition that was never written -- and the missing
// downstream artifacts then read as erasure rather than as the expected shape of a refused dispatch.
function outboundActionDescription(
  targetLabel: string,
  outcome: OutboundPostDeliveryOutcome,
): string {
  const reason = structuralOutcomeReason(outcome);

  return outboundAttemptStartedTargetTurn(outcome)
    ? `Outbound post to ${targetLabel}: ${reason}.`
    : `Outbound post to ${targetLabel} not started: ${reason} -- the target turn never ran, so nothing was composed and no entry was written to that session.`;
}

function outcomeStreamEntryIds(outcome: OutboundPostDeliveryOutcome): StreamEntryId[] {
  switch (outcome.state) {
    case "delivered":
      return [outcome.agent_message_id as StreamEntryId];
    case "suppressed":
      return outcome.marker_entry_id === undefined
        ? []
        : [outcome.marker_entry_id as StreamEntryId];
    case "not_emitted":
      return outcome.marker_entry_id === undefined
        ? []
        : [outcome.marker_entry_id as StreamEntryId];
    case "not_transportable":
      return [
        ...(outcome.agent_message_id === undefined
          ? []
          : [outcome.agent_message_id as StreamEntryId]),
        ...(outcome.stream_entry_id === undefined
          ? []
          : [outcome.stream_entry_id as StreamEntryId]),
      ];
    case "transport_failed":
      return [outcome.agent_message_id as StreamEntryId, outcome.stream_entry_id as StreamEntryId];
    case "target_busy":
      return [];
  }
}

function uniqueStreamEntryIds(entryIds: readonly (StreamEntryId | undefined)[]): StreamEntryId[] {
  return [
    ...new Set(entryIds.filter((entryId): entryId is StreamEntryId => entryId !== undefined)),
  ];
}

function targetLabel(targetSession: SessionRecord): string {
  const audienceLabel = targetSession.audience_label.trim();

  if (audienceLabel.length > 0) {
    return audienceLabel;
  }

  const sessionLabel = targetSession.label.trim();

  return sessionLabel.length > 0 ? sessionLabel : targetSession.session_id;
}

function persistOutboundActionRecord(input: {
  options: OutboundPostToolOptions;
  context: ToolInvocationContext;
  clock: Clock;
  targetSession: SessionRecord;
  outcome: OutboundPostDeliveryOutcome;
}): void {
  const provenanceStreamEntryIds = uniqueStreamEntryIds([
    input.context.toolCallEntryId,
    ...outcomeStreamEntryIds(input.outcome),
  ]);

  if (provenanceStreamEntryIds.length === 0) {
    return;
  }

  const nowMs = input.clock.now();
  const state = actionStateForOutcome(input.outcome);
  const description = outboundActionDescription(targetLabel(input.targetSession), input.outcome);
  const record: ActionRecord = {
    id: createActionId(),
    description,
    actor: "borg",
    audience_entity_id: input.targetSession.audience_entity_id,
    goal_id: null,
    open_question_id: null,
    state,
    confidence: 1,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: provenanceStreamEntryIds,
    created_at: nowMs,
    updated_at: nowMs,
    considering_at: null,
    committed_at: null,
    scheduled_at: null,
    completed_at: state === "completed" ? nowMs : null,
    not_done_at: state === "not_done" ? nowMs : null,
    expired_at: null,
    archived_at: null,
    unknown_at: null,
    session_scope: "current_session",
    session_anchor_id: input.targetSession.session_id,
    last_referenced_at_ms: null,
    last_referenced_turn_counter: null,
    last_referenced_turn_global: null,
  };

  input.options.actionRepository.add(record, { creationSource: "tool" });
}

export function createOutboundPostTool(
  options: OutboundPostToolOptions,
): ToolDefinition<OutboundPostToolInput, OutboundPostToolOutput> {
  const usedTurnIds = new Set<string>();
  const clock = options.clock ?? new SystemClock();

  return {
    name: OUTBOUND_POST_TOOL_NAME,
    description:
      "Post a proactive outbound message into another Borg session. Provide target_session_id from the operator session snapshot and an instruction for what the target-scoped composition turn should convey. This tool requires creator authority in an operator session, or autonomous policy authorization. It starts a separate target-scoped turn; it does not send my current draft text directly.",
    menuSummary: "Post outbound only to a structurally authorized target session.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: outboundPostInputSchema,
    outputSchema: outboundPostOutputSchema,
    async invoke(input, context) {
      const targetSession = assertValidTarget(options, input, context);
      const authorizationKind = outboundAuthorizationKind(options, context, targetSession);

      if (context.turnId !== undefined) {
        if (usedTurnIds.has(context.turnId)) {
          throw new ToolError("tool.outbound.post may be used only once per turn", {
            code: "OUTBOUND_POST_ALREADY_USED_IN_TURN",
          });
        }

        usedTurnIds.add(context.turnId);
      }

      if (
        authorizationKind === "manual_creator_operator" &&
        !options.connectorRegistry.has(targetSession.source_type)
      ) {
        const deliveryOutcome: OutboundPostDeliveryOutcome = {
          state: "not_transportable",
          reason: "no_wired_outbound_connector",
          source_type: targetSession.source_type,
          error: "No wired outbound connector for target session source_type.",
        };
        persistOutboundActionRecord({
          options,
          context,
          clock,
          targetSession,
          outcome: deliveryOutcome,
        });

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
            delivery_outcome: deliveryOutcome,
          },
        };
      }

      const result = await options.postOutbound({
        targetSession,
        instruction: input.instruction,
        authorizationKind,
      });
      const deliveryOutcome = deliveryOutcomeForTool(result.deliveryOutcome);
      persistOutboundActionRecord({
        options,
        context,
        clock,
        targetSession,
        outcome: deliveryOutcome,
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
          delivery_outcome: deliveryOutcome,
        },
      };
    },
  };
}
