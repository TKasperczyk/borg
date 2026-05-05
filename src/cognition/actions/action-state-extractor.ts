import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  actionEntityIdSchema,
  type ActionRecord,
  type ActionRepository,
  type ActionState,
} from "../../memory/actions/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { JsonValue } from "../../util/json-value.js";
import { createActionId, type EntityId, type StreamEntryId } from "../../util/ids.js";
import type { RecencyMessage } from "../recency/index.js";
import type { TurnTracer } from "../tracing/tracer.js";

const ACTION_STATE_TOOL_NAME = "EmitActionStates";

const extractedActionStateSchema = z.enum([
  "considering",
  "committed_to_do",
  "scheduled",
  "completed",
  "not_done",
]);

const actionStateCandidateSchema = z
  .object({
    description: z.string().trim().min(1),
    actor: z.enum(["user", "borg"]),
    state: extractedActionStateSchema,
    audience_entity_id: actionEntityIdSchema.nullable().optional(),
    evidence_stream_entry_ids: z.array(z.string().min(1)),
    confidence: z.number().min(0).max(1),
  })
  .strict();

const actionStateOutputSchema = z
  .object({
    action_states: z
      .array(actionStateCandidateSchema)
      .describe(
        "Action-state assertions in the current user message. Emit an empty array when none are present.",
      ),
  })
  .strict();

const ACTION_STATE_TOOL = {
  name: ACTION_STATE_TOOL_NAME,
  description:
    "Extract action states asserted by the current user message, citing the current user stream entry.",
  inputSchema: toToolInputSchema(actionStateOutputSchema),
} satisfies LLMToolDefinition;

const ACTION_STATE_SYSTEM_PROMPT = [
  "Extract action-state assertions from the current user message.",
  "Use recent_history only to understand elliptical references. The evidence must be in current_user_message, and every emitted item must cite current_user_stream_entry_id.",
  "Emit an empty action_states array when the current user message contains no action-state assertion.",
  "Do NOT emit action records for messages about the conversation frame, roleplay, system prompt, or the agent's own prior behavior. Action records are for user-world actions only.",
  "Judge semantic intent across languages. Do not rely on wording, punctuation, capitalization, or phrase shapes.",
  "Set audience_entity_id only when the current message clearly scopes the action to a supplied audience; otherwise use null so Borg can default it to the current audience.",
  "",
  "States:",
  "- considering: the user is weighing or contemplating an action, not committing.",
  "- committed_to_do: the user says they will do something or intends to do it.",
  "- scheduled: the action is arranged for a time, appointment, or calendar-like slot.",
  "- completed: the user says the action was done, booked, sent, finished, or carried out.",
  "- not_done: the user says the action has not happened, was abandoned, or was not completed.",
  "",
  "Examples:",
  '- "I booked it Tuesday 7pm" -> completed.',
  '- "Tuesday 7pm with Marisol" when the user just confirmed booking happened -> completed.',
  '- "I\'ll try to book this weekend" -> committed_to_do.',
  '- "Booked it for Tuesday 7pm" -> scheduled, or completed if already past, but if time cannot be known prefer scheduled for future-tense arrangements and completed when the user says they did it.',
  '- "Yeah, I haven\'t gotten to it" -> not_done.',
  '- "Maybe I should try iTalki" -> considering.',
  "Return only the required tool call.",
].join("\n");

type ActionStateToolInput = z.infer<typeof actionStateOutputSchema>;
type ParsedActionStateCandidate = z.infer<typeof actionStateCandidateSchema>;

class MissingActionStateToolCallError extends Error {}

export type ActionStateExtractorDegradedReason =
  | "llm_unavailable"
  | "repository_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload"
  | "repository_failed";

export type ActionStateExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  actionRepository?: Pick<ActionRepository, "add">;
  clock?: Clock;
  tracer?: TurnTracer;
  turnId?: string;
  onDegraded?: (
    reason: ActionStateExtractorDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type ExtractActionStatesInput = {
  userMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  recentHistory: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
};

function buildActionStateMessages(input: ExtractActionStatesInput): LLMMessage[] {
  return [
    {
      role: "user",
      content: JSON.stringify({
        current_user_message: input.userMessage,
        current_user_stream_entry_id: input.currentUserStreamEntryId,
        recent_history: input.recentHistory.slice(-8).map((message) => ({
          role: message.role,
          content: message.content,
        })),
        audience_entity_id: input.audienceEntityId,
      }),
    },
  ];
}

function parseResponse(result: LLMCompleteResult): ActionStateToolInput {
  const call = result.tool_calls.find((toolCall) => toolCall.name === ACTION_STATE_TOOL_NAME);

  if (call === undefined) {
    throw new MissingActionStateToolCallError(
      `Action state extractor did not emit ${ACTION_STATE_TOOL_NAME}`,
    );
  }

  const parsed = actionStateOutputSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return parsed.data;
}

function hasCurrentUserEvidence(
  candidate: ParsedActionStateCandidate,
  currentUserStreamEntryId: StreamEntryId,
): boolean {
  return candidate.evidence_stream_entry_ids.some(
    (entryId) => entryId === currentUserStreamEntryId,
  );
}

function stateTimestampPatch(
  state: ActionState,
  timestamp: number,
): Partial<
  Pick<
    ActionRecord,
    | "considering_at"
    | "committed_at"
    | "scheduled_at"
    | "completed_at"
    | "not_done_at"
    | "unknown_at"
  >
> {
  switch (state) {
    case "considering":
      return { considering_at: timestamp };
    case "committed_to_do":
      return { committed_at: timestamp };
    case "scheduled":
      return { scheduled_at: timestamp };
    case "completed":
      return { completed_at: timestamp };
    case "not_done":
      return { not_done_at: timestamp };
    case "unknown":
      return { unknown_at: timestamp };
  }
}

function toActionRecord(input: {
  candidate: ParsedActionStateCandidate;
  currentUserStreamEntryId: StreamEntryId;
  audienceEntityId: EntityId | null;
  nowMs: number;
}): ActionRecord {
  return {
    id: createActionId(),
    description: input.candidate.description,
    actor: input.candidate.actor,
    audience_entity_id: input.candidate.audience_entity_id ?? input.audienceEntityId,
    state: input.candidate.state,
    confidence: input.candidate.confidence,
    provenance_episode_ids: [],
    provenance_stream_entry_ids: [input.currentUserStreamEntryId],
    created_at: input.nowMs,
    updated_at: input.nowMs,
    considering_at: null,
    committed_at: null,
    scheduled_at: null,
    completed_at: null,
    not_done_at: null,
    unknown_at: null,
    ...stateTimestampPatch(input.candidate.state, input.nowMs),
  };
}

function summarizeActionStateResponseShape(response: LLMCompleteResult): JsonValue {
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

function summarizeToolSchemas(tools: readonly LLMToolDefinition[]): JsonValue {
  return tools.map((tool) => ({
    name: tool.name,
    propertyCount:
      tool.inputSchema.properties === undefined
        ? 0
        : Object.keys(tool.inputSchema.properties).length,
    required: Array.isArray(tool.inputSchema.required) ? tool.inputSchema.required.map(String) : [],
  }));
}

function traceLlmCallStarted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  model: string;
  messages: readonly LLMMessage[];
  tools: readonly LLMToolDefinition[];
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_started", {
      turnId: options.turnId,
      label: "action_state_extractor",
      model: options.model,
      promptCharCount: countCompletePromptChars(ACTION_STATE_SYSTEM_PROMPT, options.messages),
      toolSchemas: summarizeToolSchemas(options.tools),
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
      label: "action_state_extractor",
      responseShape: summarizeActionStateResponseShape(options.response),
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
      label: "action_state_extractor",
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
    });
  }
}

export class ActionStateExtractor {
  private readonly clock: Clock;

  constructor(private readonly options: ActionStateExtractorOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
  }

  private async degraded(
    reason: ActionStateExtractorDegradedReason,
    error?: unknown,
  ): Promise<ActionRecord[]> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return [];
  }

  async extract(input: ExtractActionStatesInput): Promise<ActionRecord[]> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    if (this.options.actionRepository === undefined) {
      return this.degraded("repository_unavailable");
    }

    const messages = buildActionStateMessages(input);
    const tools = [ACTION_STATE_TOOL];

    traceLlmCallStarted({
      tracer: this.options.tracer,
      turnId: this.options.turnId,
      model: this.options.model,
      messages,
      tools,
    });

    let response: LLMCompleteResult;

    try {
      response = await this.options.llmClient.complete({
        model: this.options.model,
        system: ACTION_STATE_SYSTEM_PROMPT,
        messages,
        tools,
        tool_choice: { type: "tool", name: ACTION_STATE_TOOL_NAME },
        max_tokens: 768,
        budget: "action-state-extractor",
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

    let parsed: ActionStateToolInput;

    try {
      parsed = parseResponse(response);
    } catch (error) {
      return this.degraded(
        error instanceof MissingActionStateToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError
            ? "invalid_payload"
            : "llm_failed",
        error,
      );
    }

    const persisted: ActionRecord[] = [];
    const nowMs = this.clock.now();

    for (const candidate of parsed.action_states) {
      if (!hasCurrentUserEvidence(candidate, input.currentUserStreamEntryId)) {
        continue;
      }

      const record = toActionRecord({
        candidate,
        currentUserStreamEntryId: input.currentUserStreamEntryId,
        audienceEntityId: input.audienceEntityId,
        nowMs,
      });

      try {
        this.options.actionRepository.add(record);
        persisted.push(record);
      } catch (error) {
        await this.degraded("repository_failed", error);
      }
    }

    return persisted;
  }
}

export { ACTION_STATE_TOOL_NAME };
