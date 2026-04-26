// Routes S1/S2 final response generation through the deliberator tool loop.
import type { LLMClient, LLMContentBlockMessage } from "../../llm/index.js";
import type { ToolDefinition, ToolDispatcher } from "../../tools/index.js";
import type { SessionId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { executeToolLoop, type ToolLoopResult } from "../action/index.js";

export type RunFinalizerOptions = {
  llmClient: LLMClient;
  dispatcher: ToolDispatcher;
  sessionId: SessionId;
  model: string;
  baseSystemPrompt: string;
  initialMessages: readonly LLMContentBlockMessage[];
  tools: readonly ToolDefinition[];
  userEntryId: string | undefined;
  maxTokens: number;
  path: "system_1" | "system_2";
  additionalPromptSections?: readonly (string | null)[];
  tracer?: TurnTracer;
  turnId?: string;
};

export async function runFinalizer(options: RunFinalizerOptions): Promise<ToolLoopResult> {
  const toolProvenance =
    options.userEntryId === undefined ? undefined : { user_entry_id: options.userEntryId };
  const systemPrompt =
    options.additionalPromptSections === undefined
      ? options.baseSystemPrompt
      : [options.baseSystemPrompt, ...options.additionalPromptSections]
          .filter((section): section is string => section !== null)
          .join("\n\n");

  return executeToolLoop({
    llmClient: options.llmClient,
    dispatcher: options.dispatcher,
    sessionId: options.sessionId,
    model: options.model,
    systemPrompt,
    initialMessages: options.initialMessages,
    tools: options.tools,
    origin: "deliberator",
    provenance: toolProvenance,
    maxTokens: options.maxTokens,
    budget: options.path === "system_1" ? "cognition-system-1" : "cognition-system-2",
    tracer: options.tracer,
    turnId: options.turnId,
    traceLabel: `${options.path}_finalizer`,
  });
}
