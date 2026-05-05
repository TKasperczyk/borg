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
import type { TurnTracer } from "../tracing/tracer.js";

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

const closureLoopClassifiedMessageSchema = z
  .object({
    message_ref: z.string().min(1),
    role: z.enum(["user", "assistant"]),
    act: z.enum(CLOSURE_LOOP_DIALOGUE_ACTS),
  })
  .strict();

const closureLoopClassificationSchema = z
  .object({
    messages: z.array(closureLoopClassifiedMessageSchema),
    confidence: z.number().min(0).max(1),
    rationale: z.string().min(1).max(600),
  })
  .strict();

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

function validateClosureLoopResponseCoverage(input: {
  classification: ClosureLoopClassification;
  suppliedMessages: readonly ClosureLoopMessageForClassification[];
}): void {
  const suppliedRefs = new Map<string, number>();
  const classifiedRefs = new Map<string, number>();

  for (const message of input.suppliedMessages) {
    suppliedRefs.set(message.message_ref, (suppliedRefs.get(message.message_ref) ?? 0) + 1);
  }

  for (const message of input.classification.messages) {
    classifiedRefs.set(message.message_ref, (classifiedRefs.get(message.message_ref) ?? 0) + 1);
  }

  for (const [messageRef, count] of suppliedRefs) {
    if (count !== 1) {
      throw new InvalidClosureLoopPayloadError(
        `Closure-loop classifier received duplicate supplied message_ref ${messageRef}`,
      );
    }

    if (classifiedRefs.get(messageRef) !== 1) {
      throw new InvalidClosureLoopPayloadError(
        `Closure-loop classifier omitted supplied message_ref ${messageRef}`,
      );
    }
  }

  for (const [messageRef, count] of classifiedRefs) {
    if (!suppliedRefs.has(messageRef)) {
      throw new InvalidClosureLoopPayloadError(
        `Closure-loop classifier emitted unknown message_ref ${messageRef}`,
      );
    }

    if (count !== 1) {
      throw new InvalidClosureLoopPayloadError(
        `Closure-loop classifier duplicated message_ref ${messageRef}`,
      );
    }
  }
}

function parseClosureLoopResponse(
  result: LLMCompleteResult,
  suppliedMessages: readonly ClosureLoopMessageForClassification[],
): ClosureLoopClassification {
  const call = result.tool_calls.find(
    (toolCall) => toolCall.name === CLOSURE_LOOP_CLASSIFIER_TOOL_NAME,
  );

  if (call === undefined) {
    throw new MissingClosureLoopToolCallError(
      `Closure-loop classifier did not emit ${CLOSURE_LOOP_CLASSIFIER_TOOL_NAME}`,
    );
  }

  const parsed = closureLoopClassificationSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  const classification = {
    messages: parsed.data.messages,
    confidence: parsed.data.confidence,
    rationale: parsed.data.rationale.trim(),
    degraded: false,
  };

  validateClosureLoopResponseCoverage({
    classification,
    suppliedMessages,
  });

  return classification;
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
      return parseClosureLoopResponse(response, input.messages);
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
