import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { StreamEntryId } from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import type { RecencyMessage } from "../recency/index.js";
import { toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";

export const CLOSURE_LOOP_DIALOGUE_ACTS = [
  "substantive",
  "signoff",
  "reopening_after_signoff",
  "assistant_imperative_closer",
  "assistant_valediction",
  "minimal_acknowledgment",
  "meta_objection_to_closure",
] as const;

export type ClosureLoopDialogueAct = (typeof CLOSURE_LOOP_DIALOGUE_ACTS)[number];

export const CLOSURE_LOOP_CLASSIFIER_TOOL_NAME = "ClassifyClosureLoopDialogueActs";
const CLOSURE_LOOP_RATIONALE_MAX_CHARS = 2_000;
const CLOSURE_LOOP_CLASSIFICATION_FIELDS = ["messages", "confidence", "rationale"] as const;
const CLOSURE_LOOP_MESSAGE_FIELDS = ["message_ref", "role", "act"] as const;

const closureLoopClassifiedMessageSchema = z
  .object({
    message_ref: z.string().min(1),
    role: z.enum(["user", "assistant"]),
    act: z.enum(CLOSURE_LOOP_DIALOGUE_ACTS),
  })
  .passthrough();

const closureLoopClassificationSchema = z
  .object({
    messages: z.array(closureLoopClassifiedMessageSchema),
    confidence: z.number().min(0).max(1).default(0),
    rationale: z.string().max(CLOSURE_LOOP_RATIONALE_MAX_CHARS).default(""),
  })
  .passthrough();

const CLOSURE_LOOP_CLASSIFIER_TOOL = {
  name: CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
  description:
    "Classify recent user/assistant messages into dialogue acts for closure-loop detection.",
  inputSchema: toToolInputSchema(closureLoopClassificationSchema),
} satisfies LLMToolDefinition;

const CLOSURE_LOOP_SYSTEM_PROMPT = [
  "Classify each supplied dialogue message by discourse function.",
  "Use exactly one act from the schema for each message_ref.",
  "substantive: asks, answers, introduces information, changes topic, makes a real request, or otherwise advances content.",
  "signoff: user-side goodbye, leaving, phone-down, sleep, or closure intent.",
  "reopening_after_signoff: user resumes with real content after a prior goodbye.",
  "assistant_imperative_closer: assistant-side command or push that functions as a closer.",
  "assistant_valediction: assistant-side goodbye, send-off, farewell, or closure token.",
  "minimal_acknowledgment: short assistant-side acknowledgment that does not add content.",
  "meta_objection_to_closure: names, objects to, or analyzes the closure loop itself.",
  "Judge semantic function across languages. Do not rely on exact words, punctuation, capitalization, or phrase shape.",
  "Return classifications only for supplied message_ref values. Use the tool exactly once.",
].join("\n");

export type ClosureLoopClassifiedMessage = z.infer<typeof closureLoopClassifiedMessageSchema>;

export type ClosureLoopClassification = {
  messages: ClosureLoopClassifiedMessage[];
  confidence: number;
  rationale: string;
  degraded: boolean;
};

export type ClosureLoopClassifierDegradedReason =
  | "llm_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload";

export type ClosureLoopClassifierOptions = {
  llmClient?: LLMClient;
  model?: string;
  tracer?: TurnTracer;
  turnId?: string;
  onDegraded?: (
    reason: ClosureLoopClassifierDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type ClosureLoopMessageForClassification = {
  message_ref: string;
  role: "user" | "assistant";
  content: string;
  stream_entry_id?: StreamEntryId;
  ts: number;
};

export type ClassifyClosureLoopInput = {
  messages: readonly ClosureLoopMessageForClassification[];
};

export type ClosureLoopAssessment = {
  closureLoopDetected: boolean;
  currentUserAct: ClosureLoopDialogueAct | null;
  currentUserClosureShaped: boolean;
  currentUserSubstantive: boolean;
  mutualClosureCycles: number;
  sourceStreamEntryIds: StreamEntryId[];
  reason: string;
};

class MissingClosureLoopToolCallError extends Error {}
class InvalidClosureLoopPayloadError extends Error {}

type ClosureLoopPayloadNormalization = {
  field: string;
  action: string;
  messageRef?: string;
  from?: JsonValue;
  to?: JsonValue;
};

type NormalizedClosureLoopPayload = {
  payload: z.input<typeof closureLoopClassificationSchema>;
  normalizations: ClosureLoopPayloadNormalization[];
};

function degradedClassification(
  reason: ClosureLoopClassifierDegradedReason,
): ClosureLoopClassification {
  return {
    messages: [],
    confidence: 0,
    rationale: `Closure-loop classifier degraded: ${reason}.`,
    degraded: true,
  };
}

export function buildClosureLoopMessageWindow(input: {
  recentHistory: readonly RecencyMessage[];
  currentUserMessage: string;
  currentUserEntryId: StreamEntryId;
}): ClosureLoopMessageForClassification[] {
  return [
    ...input.recentHistory.slice(-6).map(
      (message): ClosureLoopMessageForClassification => ({
        message_ref: message.stream_entry_id,
        role: message.role,
        content: message.content,
        stream_entry_id: message.stream_entry_id,
        ts: message.ts,
      }),
    ),
    {
      message_ref: input.currentUserEntryId,
      role: "user",
      content: input.currentUserMessage,
      stream_entry_id: input.currentUserEntryId,
      ts: Number.MAX_SAFE_INTEGER,
    },
  ];
}

function buildClosureLoopMessages(input: ClassifyClosureLoopInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        dialogue_window: input.messages.map((message) => ({
          message_ref: message.message_ref,
          role: message.role,
          content: message.content,
          ts: message.ts,
        })),
      }),
    },
  ];
}

function validateSuppliedClosureLoopRefs(
  suppliedMessages: readonly ClosureLoopMessageForClassification[],
): void {
  const suppliedRefs = new Map<string, number>();

  for (const message of suppliedMessages) {
    suppliedRefs.set(message.message_ref, (suppliedRefs.get(message.message_ref) ?? 0) + 1);
  }

  for (const [messageRef, count] of suppliedRefs) {
    if (count !== 1) {
      throw new InvalidClosureLoopPayloadError(
        `Closure-loop classifier received duplicate supplied message_ref ${messageRef}`,
      );
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function normalizeClosureLoopConfidence(
  value: unknown,
  normalizations: ClosureLoopPayloadNormalization[],
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
      from: toTraceJsonValue(value),
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

function normalizeClosureLoopRationale(
  value: unknown,
  normalizations: ClosureLoopPayloadNormalization[],
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
      from: toTraceJsonValue(value),
      to: "",
    });
  }

  if (rationale.length > CLOSURE_LOOP_RATIONALE_MAX_CHARS) {
    normalizations.push({
      field: "rationale",
      action: "truncated",
      from: rationale.length,
      to: CLOSURE_LOOP_RATIONALE_MAX_CHARS,
    });
    return rationale.slice(0, CLOSURE_LOOP_RATIONALE_MAX_CHARS);
  }

  return rationale;
}

function normalizeClosureLoopAct(
  value: unknown,
  messageRef: string,
  normalizations: ClosureLoopPayloadNormalization[],
): ClosureLoopDialogueAct {
  const parsed = z.enum(CLOSURE_LOOP_DIALOGUE_ACTS).safeParse(value);

  if (parsed.success) {
    return parsed.data;
  }

  normalizations.push({
    field: "act",
    messageRef,
    action: "invalid_or_missing_defaulted",
    from: toTraceJsonValue(value),
    to: "substantive",
  });
  return "substantive";
}

function normalizeClosureLoopToolInput(
  input: unknown,
  suppliedMessages: readonly ClosureLoopMessageForClassification[],
): NormalizedClosureLoopPayload {
  validateSuppliedClosureLoopRefs(suppliedMessages);

  if (!isRecord(input)) {
    throw new InvalidClosureLoopPayloadError("Closure-loop classifier input was not an object.");
  }

  const normalizations: ClosureLoopPayloadNormalization[] = [];
  const suppliedByRef = new Map(suppliedMessages.map((message) => [message.message_ref, message]));
  const allowedClassificationFields = new Set<string>(CLOSURE_LOOP_CLASSIFICATION_FIELDS);
  const extraClassificationFields = Object.keys(input).filter(
    (field) => !allowedClassificationFields.has(field),
  );

  if (extraClassificationFields.length > 0) {
    normalizations.push({
      field: "*",
      action: "extra_fields_ignored",
      from: extraClassificationFields,
    });
  }

  const rawMessages = input.messages;

  if (!Array.isArray(rawMessages)) {
    throw new InvalidClosureLoopPayloadError("Closure-loop classifier omitted messages.");
  }

  const firstByRef = new Map<string, ClosureLoopClassifiedMessage>();
  const seenRefs = new Set<string>();
  const allowedMessageFields = new Set<string>(CLOSURE_LOOP_MESSAGE_FIELDS);

  for (const rawMessage of rawMessages) {
    if (!isRecord(rawMessage)) {
      normalizations.push({
        field: "messages",
        action: "invalid_message_ignored",
        from: toTraceJsonValue(rawMessage),
      });
      continue;
    }

    const messageRef = typeof rawMessage.message_ref === "string" ? rawMessage.message_ref : "";

    if (messageRef.length === 0) {
      normalizations.push({
        field: "message_ref",
        action: "invalid_message_ignored",
        from: toTraceJsonValue(rawMessage.message_ref),
      });
      continue;
    }

    const supplied = suppliedByRef.get(messageRef);

    if (supplied === undefined) {
      normalizations.push({
        field: "message_ref",
        messageRef,
        action: "unknown_ref_ignored",
        from: messageRef,
      });
      continue;
    }

    if (seenRefs.has(messageRef)) {
      normalizations.push({
        field: "message_ref",
        messageRef,
        action: "duplicate_ref_ignored",
        from: messageRef,
      });
      continue;
    }

    seenRefs.add(messageRef);

    const extraMessageFields = Object.keys(rawMessage).filter(
      (field) => !allowedMessageFields.has(field),
    );

    if (extraMessageFields.length > 0) {
      normalizations.push({
        field: "*",
        messageRef,
        action: "message_extra_fields_ignored",
        from: extraMessageFields,
      });
    }

    if (rawMessage.role !== supplied.role) {
      normalizations.push({
        field: "role",
        messageRef,
        action: rawMessage.role === undefined ? "filled_from_supplied_ref" : "corrected_from_ref",
        from: toTraceJsonValue(rawMessage.role),
        to: supplied.role,
      });
    }

    firstByRef.set(messageRef, {
      message_ref: messageRef,
      role: supplied.role,
      act: normalizeClosureLoopAct(rawMessage.act, messageRef, normalizations),
    });
  }

  const messages = suppliedMessages.map((message) => {
    const classified = firstByRef.get(message.message_ref);

    if (classified !== undefined) {
      return classified;
    }

    normalizations.push({
      field: "message_ref",
      messageRef: message.message_ref,
      action: "missing_ref_filled_substantive",
      to: "substantive",
    });

    return {
      message_ref: message.message_ref,
      role: message.role,
      act: "substantive" as const,
    };
  });

  return {
    payload: {
      messages,
      confidence: normalizeClosureLoopConfidence(input.confidence, normalizations),
      rationale: normalizeClosureLoopRationale(input.rationale, normalizations),
    },
    normalizations,
  };
}

function traceClosureLoopPayloadNormalized(options: {
  tracer?: TurnTracer;
  turnId?: string;
  rawToolInput: unknown;
  normalizations: readonly ClosureLoopPayloadNormalization[];
}): void {
  if (
    options.tracer?.enabled !== true ||
    options.turnId === undefined ||
    options.normalizations.length === 0
  ) {
    return;
  }

  options.tracer.emit("closure_loop_classifier_payload_normalized", {
    turnId: options.turnId,
    normalizations: options.normalizations.map((normalization) => ({ ...normalization })),
    ...(options.tracer.includePayloads
      ? { rawToolInput: toTraceJsonValue(options.rawToolInput) }
      : {}),
  });
}

function parseClosureLoopResponse(
  result: LLMCompleteResult,
  suppliedMessages: readonly ClosureLoopMessageForClassification[],
  traceOptions: {
    tracer?: TurnTracer;
    turnId?: string;
  } = {},
): ClosureLoopClassification {
  const call = result.tool_calls.find(
    (toolCall) => toolCall.name === CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
  );

  if (call === undefined) {
    throw new MissingClosureLoopToolCallError(
      `Closure-loop classifier did not emit ${CLOSURE_LOOP_CLASSIFIER_TOOL_NAME}`,
    );
  }

  const normalized = normalizeClosureLoopToolInput(call.input, suppliedMessages);
  const parsed = closureLoopClassificationSchema.safeParse(normalized.payload);

  if (!parsed.success) {
    throw parsed.error;
  }

  traceClosureLoopPayloadNormalized({
    ...traceOptions,
    rawToolInput: call.input,
    normalizations: normalized.normalizations,
  });

  return {
    messages: parsed.data.messages,
    confidence: parsed.data.confidence,
    rationale: parsed.data.rationale,
    degraded: false,
  };
}

function summarizeClosureLoopResponseShape(response: LLMCompleteResult): JsonValue {
  return {
    textLength: response.text.length,
    toolUseBlocks: response.tool_calls.map((call) => ({
      id: call.id,
      name: call.name,
    })),
  };
}

function countCompletePromptChars(systemPrompt: string, messages: readonly LLMMessage[]): number {
  return (
    systemPrompt.length +
    messages.reduce((sum, message) => sum + message.role.length + message.content.length, 0)
  );
}

function traceLlmCallStarted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  model: string;
  messages: readonly LLMMessage[];
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_started", {
      turnId: options.turnId,
      label: "closure_loop_classifier",
      model: options.model,
      promptCharCount: countCompletePromptChars(CLOSURE_LOOP_SYSTEM_PROMPT, options.messages),
    });
  }
}

function traceLlmCallResponse(options: {
  tracer?: TurnTracer;
  turnId?: string;
  response: LLMCompleteResult;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_response", {
      turnId: options.turnId,
      label: "closure_loop_classifier",
      responseShape: summarizeClosureLoopResponseShape(options.response),
      stopReason: options.response.stop_reason,
      usage: {
        inputTokens: options.response.input_tokens,
        outputTokens: options.response.output_tokens,
      },
    });
  }
}

function traceLlmCallError(options: {
  tracer?: TurnTracer;
  turnId?: string;
  error: unknown;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_response", {
      turnId: options.turnId,
      label: "closure_loop_classifier",
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
    });
  }
}

function isUserClosureAct(act: ClosureLoopDialogueAct): boolean {
  return act === "signoff";
}

function isAssistantClosureAct(act: ClosureLoopDialogueAct): boolean {
  return (
    act === "assistant_imperative_closer" ||
    act === "assistant_valediction" ||
    act === "minimal_acknowledgment"
  );
}

function isSubstantiveForClosureState(act: ClosureLoopDialogueAct): boolean {
  return (
    act === "substantive" ||
    act === "reopening_after_signoff" ||
    act === "meta_objection_to_closure"
  );
}

export function assessClosureLoopClassification(input: {
  classification: ClosureLoopClassification;
  suppliedMessages: readonly ClosureLoopMessageForClassification[];
  currentUserRef: string;
}): ClosureLoopAssessment {
  const supplied = new Map(input.suppliedMessages.map((message) => [message.message_ref, message]));
  const ordered = input.classification.messages
    .filter((message) => supplied.has(message.message_ref))
    .sort((left, right) => {
      const leftTs = supplied.get(left.message_ref)?.ts ?? 0;
      const rightTs = supplied.get(right.message_ref)?.ts ?? 0;

      return leftTs - rightTs;
    });
  let pendingUserClosure = false;
  let mutualClosureCycles = 0;

  for (const message of ordered) {
    if (isSubstantiveForClosureState(message.act)) {
      pendingUserClosure = false;
      mutualClosureCycles = 0;
      continue;
    }

    if (message.role === "user") {
      pendingUserClosure = isUserClosureAct(message.act);
      continue;
    }

    if (pendingUserClosure && isAssistantClosureAct(message.act)) {
      mutualClosureCycles += 1;
    }

    pendingUserClosure = false;
  }

  const currentUser = ordered.find((message) => message.message_ref === input.currentUserRef);
  const currentUserAct = currentUser?.role === "user" ? currentUser.act : null;
  const currentUserClosureShaped =
    currentUserAct === null ? false : isUserClosureAct(currentUserAct);
  const currentUserSubstantive =
    currentUserAct === null ? false : isSubstantiveForClosureState(currentUserAct);
  const closureLoopDetected =
    mutualClosureCycles >= 2 &&
    (currentUserClosureShaped || ordered[ordered.length - 1]?.role === "assistant");
  const sourceStreamEntryIds: StreamEntryId[] = [];

  for (const message of ordered) {
    const suppliedMessage = supplied.get(message.message_ref);

    if (suppliedMessage?.stream_entry_id === undefined) {
      continue;
    }

    if (sourceStreamEntryIds.some((entryId) => entryId === suppliedMessage.stream_entry_id)) {
      continue;
    }

    sourceStreamEntryIds.push(suppliedMessage.stream_entry_id);
  }

  return {
    closureLoopDetected,
    currentUserAct,
    currentUserClosureShaped,
    currentUserSubstantive,
    mutualClosureCycles,
    sourceStreamEntryIds,
    reason: input.classification.rationale,
  };
}

export function assessDegradedClosureLoopFallback(input: {
  suppliedMessages: readonly ClosureLoopMessageForClassification[];
  currentUserRef: string;
  priorClosureLoopActive: boolean;
}): ClosureLoopAssessment {
  const currentUser = input.suppliedMessages.find(
    (message) => message.message_ref === input.currentUserRef && message.role === "user",
  );
  const currentUserShort = (currentUser?.content.trim().length ?? Number.POSITIVE_INFINITY) < 80;
  const ambiguousClosureBeat = input.priorClosureLoopActive && currentUserShort;
  const sourceStreamEntryIds: StreamEntryId[] = [];

  for (const message of input.suppliedMessages) {
    if (message.stream_entry_id === undefined) {
      continue;
    }

    if (sourceStreamEntryIds.some((entryId) => entryId === message.stream_entry_id)) {
      continue;
    }

    sourceStreamEntryIds.push(message.stream_entry_id);
  }

  return {
    closureLoopDetected: false,
    currentUserAct: null,
    currentUserClosureShaped: false,
    currentUserSubstantive: false,
    mutualClosureCycles: 0,
    sourceStreamEntryIds,
    reason: ambiguousClosureBeat
      ? "Closure-loop classifier degraded; short-turn fallback was ambiguous before assistant emission, so suppression failed open."
      : "Closure-loop classifier degraded; fallback found no high-confidence closure suppression signal.",
  };
}

export class ClosureLoopClassifier {
  constructor(private readonly options: ClosureLoopClassifierOptions = {}) {}

  private async degraded(
    reason: ClosureLoopClassifierDegradedReason,
    error?: unknown,
  ): Promise<ClosureLoopClassification> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return degradedClassification(reason);
  }

  async classify(input: ClassifyClosureLoopInput): Promise<ClosureLoopClassification> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    const messages = buildClosureLoopMessages(input);

    traceLlmCallStarted({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      model: this.options.model,
      messages,
    });

    let response: LLMCompleteResult;

    try {
      response = await this.options.llmClient.complete({
        model: this.options.model,
        system: CLOSURE_LOOP_SYSTEM_PROMPT,
        messages,
        tools: [CLOSURE_LOOP_CLASSIFIER_TOOL],
        tool_choice: { type: "tool", name: CLOSURE_LOOP_CLASSIFIER_TOOL_NAME },
        max_tokens: 700,
        budget: "closure-loop-classifier",
      });
    } catch (error) {
      traceLlmCallError({
        tracer: this.options.tracer,
        turnId: this.options.turnId,
        error,
      });

      return this.degraded("llm_failed", error);
    }

    traceLlmCallResponse({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      response,
    });

    try {
      return parseClosureLoopResponse(response, input.messages, {
        tracer: this.options.tracer,
        turnId: this.options.turnId,
      });
    } catch (error) {
      return this.degraded(
        error instanceof MissingClosureLoopToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError || error instanceof InvalidClosureLoopPayloadError
            ? "invalid_payload"
            : "llm_failed",
        error,
      );
    }
  }
}
