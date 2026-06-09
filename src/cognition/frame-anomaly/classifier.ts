import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { BorgRole, EntityKind } from "../../memory/commitments/index.js";
import { unknownMemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import type { JsonValue } from "../../util/json-value.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import { isPlainRecord } from "../../util/guards.js";
import type { ActiveParticipant } from "../participants.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import { FRAME_ANOMALY_SYSTEM_PROMPT } from "../prompts/frame-anomaly.js";
import type { RecencyMessage } from "../recency/index.js";
import { summarizeTraceValueShape, toTraceJsonValue, type TurnTracer } from "../../tracing/tracer.js";
import {
  summarizeToolResponseShape,
} from "../../tracing/llm-call-trace.js";
import {
  type FrameAnomalyClassification,
  type FrameAnomalyKind,
  frameAnomalyKindSchema,
} from "./types.js";

export const FRAME_ANOMALY_CLASSIFIER_TOOL_NAME = "ClassifyFrameAnomaly";
const FRAME_ANOMALY_RATIONALE_MAX_CHARS = 2_000;
const FRAME_ANOMALY_TOOL_FIELDS = ["kind", "confidence", "rationale"] as const;
const FRAME_ANOMALY_KIND_ALIASES: Readonly<Record<string, FrameAnomalyKind>> = {
  no_anomaly: "normal",
  none: "normal",
  safe: "normal",
  assistant_identity_claim: "assistant_self_claim_in_user_role",
  roleplay_claim: "frame_assignment_claim",
};

const frameAnomalyClassificationSchema = z
  .object({
    kind: frameAnomalyKindSchema.describe(
      "normal unless the current user-role message makes anomalous claims about assistant identity, system prompt, frame assignment, authorship, or roleplay inversion.",
    ),
    confidence: z.number().min(0).max(1).default(0),
    rationale: z.string().max(FRAME_ANOMALY_RATIONALE_MAX_CHARS).default(""),
  })
  .passthrough();

const FRAME_ANOMALY_CLASSIFIER_TOOL = {
  name: FRAME_ANOMALY_CLASSIFIER_TOOL_NAME,
  description: "Classify whether the current user-role message carries a frame-provenance anomaly.",
  inputSchema: toToolInputSchema(frameAnomalyClassificationSchema),
} satisfies LLMToolDefinition;

export type FrameAnomalyClassifierDegradedReason =
  | "llm_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload";

export type FrameAnomalyClassifierOptions = {
  llmClient?: LLMClient;
  model?: string;
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  onDegraded?: (
    reason: FrameAnomalyClassifierDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type FrameAnomalyConversationEntityContext = {
  id: EntityId | null;
  display_name: string | null;
};

export type FrameAnomalyConversationContext = {
  audience: FrameAnomalyConversationEntityContext & {
    kind: EntityKind | null;
  };
  current_sender: FrameAnomalyConversationEntityContext;
  current_sender_borg_role: BorgRole | null;
  session_audience_role: SessionAudienceRole;
  participants: readonly ActiveParticipant[];
  assistant_identity: FrameAnomalyConversationEntityContext;
  previous_user_sender: FrameAnomalyConversationEntityContext | null;
  sender_changed_since_previous_user_turn: boolean;
};

export type ClassifyFrameAnomalyInput = {
  userMessage: string;
  recentHistory: readonly RecencyMessage[];
  conversationContext?: FrameAnomalyConversationContext;
};

class MissingFrameAnomalyToolCallError extends Error {}
class InvalidFrameAnomalyPayloadError extends Error {}

function degradedClassification(
  reason: FrameAnomalyClassifierDegradedReason,
): FrameAnomalyClassification {
  return {
    status: "degraded",
    reason,
  };
}

type FrameAnomalyPayloadNormalization = {
  field: string;
  action: string;
  from?: JsonValue;
  to?: JsonValue;
};

type NormalizedFrameAnomalyPayload = {
  payload: z.input<typeof frameAnomalyClassificationSchema>;
  normalizations: FrameAnomalyPayloadNormalization[];
};

function jsonValueOrNull(value: unknown): JsonValue {
  return toTraceJsonValue(value);
}

function normalizeMachineLabel(value: string): string {
  return value.trim().toLowerCase();
}

function normalizeFrameAnomalyKind(
  value: unknown,
  normalizations: FrameAnomalyPayloadNormalization[],
): FrameAnomalyKind {
  if (typeof value !== "string") {
    throw new InvalidFrameAnomalyPayloadError("Frame anomaly classifier omitted kind.");
  }

  const normalized = normalizeMachineLabel(value);
  const alias = FRAME_ANOMALY_KIND_ALIASES[normalized];
  const candidate = alias ?? normalized;

  if (alias !== undefined) {
    normalizations.push({
      field: "kind",
      action: "alias_mapped",
      from: value,
      to: alias,
    });
  } else if (candidate !== value) {
    normalizations.push({
      field: "kind",
      action: "machine_label_normalized",
      from: value,
      to: candidate,
    });
  }

  const parsed = frameAnomalyKindSchema.safeParse(candidate);

  if (!parsed.success) {
    throw new InvalidFrameAnomalyPayloadError(`Invalid frame anomaly kind: ${String(value)}`);
  }

  return parsed.data;
}

function normalizeConfidence(
  value: unknown,
  normalizations: FrameAnomalyPayloadNormalization[],
): number {
  let confidence = 0;

  if (value === undefined) {
    normalizations.push({
      field: "confidence",
      action: "defaulted",
      to: 0,
    });
  } else if (typeof value === "string") {
    const parsed = Number(value);
    confidence = Number.isFinite(parsed) ? parsed : 0;
    normalizations.push({
      field: "confidence",
      action: Number.isFinite(parsed) ? "string_coerced" : "invalid_string_defaulted",
      from: value,
      to: confidence,
    });
  } else if (typeof value === "number" && Number.isFinite(value)) {
    confidence = value;
  } else {
    normalizations.push({
      field: "confidence",
      action: "invalid_type_defaulted",
      from: jsonValueOrNull(value),
      to: 0,
    });
  }

  const clamped = Math.min(1, Math.max(0, confidence));

  if (clamped !== confidence) {
    normalizations.push({
      field: "confidence",
      action: "clamped",
      from: confidence,
      to: clamped,
    });
  }

  return clamped;
}

function normalizeRationale(
  value: unknown,
  normalizations: FrameAnomalyPayloadNormalization[],
): string {
  let rationale = "";

  if (value === undefined) {
    normalizations.push({
      field: "rationale",
      action: "defaulted",
      to: "",
    });
  } else if (typeof value === "string") {
    rationale = value.trim();
  } else {
    normalizations.push({
      field: "rationale",
      action: "invalid_type_defaulted",
      from: jsonValueOrNull(value),
      to: "",
    });
  }

  if (rationale.length > FRAME_ANOMALY_RATIONALE_MAX_CHARS) {
    normalizations.push({
      field: "rationale",
      action: "truncated",
      from: rationale.length,
      to: FRAME_ANOMALY_RATIONALE_MAX_CHARS,
    });
    return rationale.slice(0, FRAME_ANOMALY_RATIONALE_MAX_CHARS);
  }

  return rationale;
}

function normalizeFrameAnomalyToolInput(input: unknown): NormalizedFrameAnomalyPayload {
  if (!isPlainRecord(input)) {
    throw new InvalidFrameAnomalyPayloadError("Frame anomaly classifier input was not an object.");
  }

  const normalizations: FrameAnomalyPayloadNormalization[] = [];
  const allowedFields = new Set<string>(FRAME_ANOMALY_TOOL_FIELDS);
  const extraFields = Object.keys(input).filter((field) => !allowedFields.has(field));

  if (extraFields.length > 0) {
    normalizations.push({
      field: "*",
      action: "extra_fields_ignored",
      from: extraFields,
    });
  }

  return {
    payload: {
      kind: normalizeFrameAnomalyKind(input.kind, normalizations),
      confidence: normalizeConfidence(input.confidence, normalizations),
      rationale: normalizeRationale(input.rationale, normalizations),
    },
    normalizations,
  };
}

function buildConversationContextPayload(
  context: FrameAnomalyConversationContext,
): Record<string, JsonValue> {
  return {
    audience: {
      id: context.audience.id,
      display_name: context.audience.display_name,
      kind: context.audience.kind,
    },
    current_sender: {
      id: context.current_sender.id,
      display_name: context.current_sender.display_name,
    },
    current_sender_borg_role: context.current_sender_borg_role,
    session_audience_role: context.session_audience_role,
    participants: context.participants.map((participant) => ({
      id: participant.entityId,
      display_name: participant.displayName,
      role: participant.role,
    })),
    assistant_identity: {
      id: context.assistant_identity.id,
      display_name: context.assistant_identity.display_name,
    },
    previous_user_sender:
      context.previous_user_sender === null
        ? null
        : {
            id: context.previous_user_sender.id,
            display_name: context.previous_user_sender.display_name,
          },
    sender_changed_since_previous_user_turn: context.sender_changed_since_previous_user_turn,
  };
}

function buildFrameAnomalyMessages(input: ClassifyFrameAnomalyInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        current_user_message: input.userMessage,
        ...(input.conversationContext === undefined
          ? {}
          : {
              conversation_context: buildConversationContextPayload(input.conversationContext),
            }),
        recent_assistant_turns: input.recentHistory
          .filter((message) => message.role === "assistant")
          .slice(-6)
          .map((message) => ({
            content: message.content,
            stream_entry_id: message.stream_entry_id,
            ts: message.ts,
            ...memoryDisclosurePayloadFields(unknownMemoryDisclosureLabel()),
          })),
      }),
    },
  ];
}

function traceFrameAnomalyClassified(options: {
  tracer?: TurnTracer;
  turnId?: string;
  classification: FrameAnomalyClassification;
  rawToolInput?: unknown;
  normalizations?: readonly FrameAnomalyPayloadNormalization[];
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  const payload = {
    turnId: options.turnId,
    status: options.classification.status,
    ...(options.classification.status === "ok"
      ? {
          kind: options.classification.kind,
          confidence: options.classification.confidence,
          rationaleLength: options.classification.rationale.length,
        }
      : {
          reason: options.classification.reason,
        }),
    normalizations: (options.normalizations ?? []).map((normalization) => ({
      ...normalization,
    })),
    ...(options.rawToolInput !== undefined
      ? { rawToolInputShape: summarizeTraceValueShape(options.rawToolInput) }
      : {}),
    ...(options.rawToolInput !== undefined && options.tracer.includePayloads
      ? { rawToolInput: toTraceJsonValue(options.rawToolInput) }
      : {}),
  } satisfies Record<string, JsonValue | undefined> & { turnId: string };

  options.tracer.emit("frame_anomaly.completed", payload);
}

function parseToolInput(
  input: unknown,
  traceOptions: {
    tracer?: TurnTracer;
    turnId?: string;
  } = {},
  rawToolInput: unknown = input,
): FrameAnomalyClassification {
  const normalized = normalizeFrameAnomalyToolInput(input);
  const parsed = frameAnomalyClassificationSchema.safeParse(normalized.payload);

  if (!parsed.success) {
    throw parsed.error;
  }

  const classification: FrameAnomalyClassification = {
    status: "ok",
    kind: parsed.data.kind,
    confidence: parsed.data.confidence,
    rationale: parsed.data.rationale,
  };

  traceFrameAnomalyClassified({
    ...traceOptions,
    classification,
    rawToolInput,
    normalizations: normalized.normalizations,
  });

  return classification;
}

export class FrameAnomalyClassifier {
  constructor(private readonly options: FrameAnomalyClassifierOptions = {}) {}

  private async degraded(
    reason: FrameAnomalyClassifierDegradedReason,
    error?: unknown,
  ): Promise<FrameAnomalyClassification> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    const classification = degradedClassification(reason);

    traceFrameAnomalyClassified({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      classification,
      normalizations: [],
    });

    return classification;
  }

  async classify(input: ClassifyFrameAnomalyInput): Promise<FrameAnomalyClassification> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    const messages = buildFrameAnomalyMessages(input);
    const tools = [FRAME_ANOMALY_CLASSIFIER_TOOL];

    try {
      return (
        await callStructuredTool({
          llmClient: this.options.llmClient,
          request: {
            model: this.options.model,
            system: FRAME_ANOMALY_SYSTEM_PROMPT,
            messages,
            tools,
            tool_choice: { type: "tool", name: FRAME_ANOMALY_CLASSIFIER_TOOL_NAME },
            max_tokens: EXTRACTOR_MAX_TOKENS_DEFAULT,
            budget: "frame-anomaly-classifier",
          },
          toolName: FRAME_ANOMALY_CLASSIFIER_TOOL_NAME,
          parse: (input) =>
            parseToolInput(
              input,
              {
                tracer: this.options.tracer,
                turnId: this.options.turnId,
              },
              input,
            ),
          trace: {
            tracer: this.options.tracer,
            turnId: this.options.turnId,
            sessionId: this.options.sessionId,
            label: "frame_anomaly_classifier",
            systemPrompt: FRAME_ANOMALY_SYSTEM_PROMPT,
            messages,
            tools,
            responseShape: summarizeToolResponseShape,
          },
        })
      ).parsed;
    } catch (error) {
      if (isStructuredToolCallError(error, "missing_tool_call")) {
        return this.degraded(
          "missing_tool_call",
          new MissingFrameAnomalyToolCallError(
            `Frame anomaly classifier did not emit ${FRAME_ANOMALY_CLASSIFIER_TOOL_NAME}`,
          ),
        );
      }

      return this.degraded(
        error instanceof MissingFrameAnomalyToolCallError
          ? "missing_tool_call"
          : isStructuredToolCallError(error, "invalid_payload") ||
              error instanceof z.ZodError ||
              error instanceof InvalidFrameAnomalyPayloadError
            ? "invalid_payload"
            : "llm_failed",
        isStructuredToolCallError(error) ? (error.cause ?? error) : error,
      );
    }
  }
}
