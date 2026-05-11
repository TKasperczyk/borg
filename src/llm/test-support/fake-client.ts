import { LLMError } from "../../util/errors.js";
import {
  LLMStructuredOutputParseError,
  type LLMClient,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type LLMContentBlock,
  type LLMConverseOptions,
  type LLMConverseResult,
  type LLMMessage,
  type LLMTextBlock,
  type LLMToolDefinition,
  type LLMToolResultBlock,
  type LLMToolUseBlock,
  type TokenUsageSink,
} from "../index.js";

type FakeLLMResponseValue =
  | string
  | readonly LLMContentBlock[]
  | LLMCompleteResult
  | LLMConverseResult;

export type FakeLLMResponse =
  | FakeLLMResponseValue
  | ((options: LLMCompleteOptions) => FakeLLMResponseValue | Promise<FakeLLMResponseValue>)
  | ((options: LLMConverseOptions) => FakeLLMResponseValue | Promise<FakeLLMResponseValue>);

export type FakeLLMClientOptions = {
  responses?: FakeLLMResponse[];
  usageSink?: TokenUsageSink;
  oauthToolNameTransport?: boolean;
};

export function createFakeEmitAnswerResponse(
  text: string,
  usage: { inputTokens: number; outputTokens: number } = { inputTokens: 8, outputTokens: 4 },
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: usage.inputTokens,
    output_tokens: usage.outputTokens,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_emit_answer",
        name: "EmitAnswer",
        input: { text },
      },
    ],
  };
}

function parseStructuredOutputText(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    throw new LLMStructuredOutputParseError(text, error);
  }
}

function transformToolNameForOAuth(name: string): string {
  if (!name) {
    return name;
  }

  if (name.startsWith("mcp__")) {
    return name;
  }

  if (name.charAt(0) === name.charAt(0).toUpperCase() && /[A-Z]/.test(name.charAt(0))) {
    return name;
  }

  const normalized = name.replace(/[^A-Za-z0-9_]/g, "_");

  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function oauthTransportToolDefinitions(
  tools: readonly LLMToolDefinition[] | undefined,
): readonly LLMToolDefinition[] | undefined {
  return tools?.map((tool) => {
    const transformedName = transformToolNameForOAuth(tool.name);

    if (transformedName === tool.name) {
      return tool;
    }

    return {
      ...tool,
      name: transformedName,
    };
  });
}

function oauthTransportToolChoice(
  toolChoice: LLMCompleteOptions["tool_choice"],
): LLMCompleteOptions["tool_choice"] {
  if (toolChoice?.type !== "tool") {
    return toolChoice;
  }

  const transformedName = transformToolNameForOAuth(toolChoice.name);

  if (transformedName === toolChoice.name) {
    return toolChoice;
  }

  return {
    ...toolChoice,
    name: transformedName,
  };
}

function oauthTransportContentBlock(block: LLMContentBlock): LLMContentBlock {
  if (block.type !== "tool_use") {
    return block;
  }

  const transformedName = transformToolNameForOAuth(block.name);

  if (transformedName === block.name) {
    return block;
  }

  return {
    ...block,
    name: transformedName,
  };
}

function oauthTransportConverseOptions(options: LLMConverseOptions): LLMConverseOptions {
  return {
    ...options,
    tools: oauthTransportToolDefinitions(options.tools),
    tool_choice: oauthTransportToolChoice(options.tool_choice),
    messages: options.messages.map((message) => ({
      role: message.role,
      content: message.content.map((block) => oauthTransportContentBlock(block)),
    })),
  };
}

function oauthTransportCompleteOptions(options: LLMCompleteOptions): LLMCompleteOptions {
  return {
    ...options,
    tools: oauthTransportToolDefinitions(options.tools),
    tool_choice: oauthTransportToolChoice(options.tool_choice),
  };
}

function inferStopReasonFromBlocks(blocks: readonly LLMContentBlock[]): string | null {
  return blocks.some((block) => block.type === "tool_use") ? "tool_use" : "end_turn";
}

function blocksFromCompleteResult(result: LLMCompleteResult): LLMContentBlock[] {
  const blocks: LLMContentBlock[] = [];

  if (result.text.length > 0) {
    blocks.push({
      type: "text",
      text: result.text,
    });
  }

  for (const call of result.tool_calls) {
    blocks.push({
      type: "tool_use",
      id: call.id,
      name: call.name,
      input: call.input,
    });
  }

  return blocks;
}

function isFakeBlockArray(response: FakeLLMResponseValue): response is readonly LLMContentBlock[] {
  return Array.isArray(response);
}

function isFakeConverseResult(response: FakeLLMResponseValue): response is LLMConverseResult {
  return typeof response === "object" && response !== null && "messageBlocks" in response;
}

function normalizeFakeConverseResponse(response: FakeLLMResponseValue): LLMConverseResult {
  if (typeof response === "string") {
    return {
      messageBlocks: [
        {
          type: "text",
          text: response,
        },
      ],
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "end_turn",
    };
  }

  if (isFakeBlockArray(response)) {
    return {
      messageBlocks: [...response],
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: inferStopReasonFromBlocks(response),
    };
  }

  if (isFakeConverseResult(response)) {
    return response;
  }

  return {
    messageBlocks: blocksFromCompleteResult(response),
    input_tokens: response.input_tokens,
    output_tokens: response.output_tokens,
    stop_reason: response.stop_reason,
  };
}

function normalizeFakeCompleteResponse(response: FakeLLMResponseValue): LLMCompleteResult {
  if (typeof response === "string") {
    return {
      text: response,
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "end_turn",
      tool_calls: [],
    };
  }

  if (isFakeBlockArray(response)) {
    return {
      text: response
        .filter((block): block is LLMTextBlock => block.type === "text")
        .map((block) => block.text)
        .join(""),
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: inferStopReasonFromBlocks(response),
      tool_calls: response
        .filter((block): block is LLMToolUseBlock => block.type === "tool_use")
        .map((block) => ({
          id: block.id,
          name: block.name,
          input: block.input,
        })),
    };
  }

  if (isFakeConverseResult(response)) {
    return normalizeFakeCompleteResponse(response.messageBlocks);
  }

  return response;
}

function withFakeStructuredOutput(
  options: Pick<LLMCompleteOptions, "output_config">,
  result: LLMCompleteResult,
): LLMCompleteResult {
  if (options.output_config === undefined || "structured_output" in result) {
    return result;
  }

  return {
    ...result,
    structured_output: parseStructuredOutputText(result.text),
  };
}

function withFakeConversationStructuredOutput(
  options: Pick<LLMConverseOptions, "output_config">,
  result: LLMConverseResult,
): LLMConverseResult {
  if (options.output_config === undefined || "structured_output" in result) {
    return result;
  }

  const text = result.messageBlocks
    .filter((block): block is LLMTextBlock => block.type === "text")
    .map((block) => block.text)
    .join("");

  return {
    ...result,
    structured_output: parseStructuredOutputText(text),
  };
}

function flattenBlockContentForCompatibility(content: LLMToolResultBlock["content"]): string {
  if (typeof content === "string") {
    return content;
  }

  return content.map((block) => block.text).join("");
}

function flattenMessageBlocksForCompatibility(blocks: readonly LLMContentBlock[]): string {
  return blocks
    .map((block) => {
      if (block.type === "text") {
        return block.text;
      }

      if (block.type === "tool_use") {
        return `[tool_use ${block.name}]`;
      }

      return flattenBlockContentForCompatibility(block.content);
    })
    .join("");
}

function toCompleteCompatibleRequest(options: LLMConverseOptions): LLMCompleteOptions {
  return {
    ...options,
    messages: options.messages.map((message) => ({
      role: message.role,
      content: flattenMessageBlocksForCompatibility(message.content),
    })),
  };
}

function isProceduralContextFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "procedural-context";
}

function isStopCommitmentFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "generation-stop-commitment";
}

function isPendingActionJudgeFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "pending-action-judge";
}

function isCorrectivePreferenceFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "corrective-preference-extractor";
}

function isGoalPromotionFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "goal-promotion-extractor";
}

function isActionStateExtractorFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "action-state-extractor";
}

function isFrameAnomalyClassifierFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "frame-anomaly-classifier";
}

function isPersonaRoleBleedClassifierFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "persona-role-bleed-classifier";
}

function isClosureLoopClassifierFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "closure-loop-classifier";
}

function isRelationalClaimAuditorFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "relational-claim-auditor";
}

function isClosureResponseAuditorFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "closure-response-auditor";
}

function isRecallExpansionFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "recall-expansion";
}

function scriptedResponseBudget(response: FakeLLMResponse | undefined): string | undefined {
  if (typeof response !== "function") {
    return undefined;
  }

  const budget = (response as { budget?: unknown }).budget;

  return typeof budget === "string" ? budget : undefined;
}

function isProceduralContextResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitProceduralContext");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitProceduralContext",
    );
  }

  return false;
}

function isStopCommitmentResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some(
      (toolCall) => toolCall.name === "EmitStopCommitmentClassification",
    );
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitStopCommitmentClassification",
    );
  }

  return false;
}

function isPendingActionJudgeResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "ClassifyPendingAction");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "ClassifyPendingAction",
    );
  }

  return false;
}

function isCorrectivePreferenceResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitCorrectivePreference");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitCorrectivePreference",
    );
  }

  return false;
}

function isGoalPromotionResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitGoalPromotion");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitGoalPromotion",
    );
  }

  return false;
}

function isActionStateResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitActionStates");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitActionStates",
    );
  }

  return false;
}

function isFrameAnomalyResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "ClassifyFrameAnomaly");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "ClassifyFrameAnomaly",
    );
  }

  return false;
}

function isPersonaRoleBleedResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "ClassifyPersonaRoleBleed");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "ClassifyPersonaRoleBleed",
    );
  }

  return false;
}

function isClosureLoopClassificationResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some(
      (toolCall) => toolCall.name === "ClassifyClosureLoopDialogueActs",
    );
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "ClassifyClosureLoopDialogueActs",
    );
  }

  return false;
}

function isRelationalClaimAuditResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitClaimAudit");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitClaimAudit",
    );
  }

  return false;
}

function isClosureResponseAuditResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitClosureResponseAudit");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitClosureResponseAudit",
    );
  }

  return false;
}

function isRecallExpansionResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitRecallExpansion");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitRecallExpansion",
    );
  }

  return false;
}

function defaultProceduralContextResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_procedural_context",
        name: "EmitProceduralContext",
        input: {
          problem_kind: "other",
          domain_tags: [],
          confidence: 0,
        },
      },
    ],
  };
}

function defaultStopCommitmentResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_stop_commitment",
        name: "EmitStopCommitmentClassification",
        input: {
          classification: "none",
          directive_family: null,
          reason: "No operational no-output commitment.",
          confidence: 0,
        },
      },
    ],
  };
}

function defaultPendingActionJudgeResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_pending_action_judge",
        name: "ClassifyPendingAction",
        input: {
          classification: "action",
          reason: "Accepted by test fallback.",
          confidence: 1,
        },
      },
    ],
  };
}

function defaultCorrectivePreferenceResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_corrective_preference",
        name: "EmitCorrectivePreference",
        input: {
          classification: "none",
          type: null,
          directive: null,
          directive_family: null,
          closure_pressure_relevance: null,
          priority: null,
          reason: "No durable correction detected.",
          confidence: 0,
          supersedes_commitment_id: null,
          slot_negations: [],
        },
      },
    ],
  };
}

function defaultGoalPromotionResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_goal_promotion",
        name: "EmitGoalPromotion",
        input: {
          promotions: [],
        },
      },
    ],
  };
}

function defaultActionStateResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_action_state",
        name: "EmitActionStates",
        input: {
          action_states: [],
        },
      },
    ],
  };
}

function defaultFrameAnomalyResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_frame_anomaly",
        name: "ClassifyFrameAnomaly",
        input: {
          kind: "normal",
          confidence: 0,
          rationale: "No frame-provenance anomaly detected.",
        },
      },
    ],
  };
}

function defaultPersonaRoleBleedResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_persona_role_bleed",
        name: "ClassifyPersonaRoleBleed",
        input: {
          category: "tom_persona",
          confidence: 0,
          rationale: "No persona role bleed detected.",
        },
      },
    ],
  };
}

function defaultClosureLoopClassifiedMessages(
  messages: readonly LLMMessage[],
): Array<{ message_ref: string; role: "user" | "assistant"; act: "substantive" }> {
  const request = messages[0];

  if (request === undefined) {
    return [];
  }

  try {
    const parsed = JSON.parse(request.content) as { dialogue_window?: unknown };
    const dialogueWindow = parsed.dialogue_window;

    if (!Array.isArray(dialogueWindow)) {
      return [];
    }

    const classified: Array<{
      message_ref: string;
      role: "user" | "assistant";
      act: "substantive";
    }> = [];

    for (const item of dialogueWindow) {
      if (typeof item !== "object" || item === null) {
        continue;
      }

      const message = item as { message_ref?: unknown; role?: unknown };

      if (
        typeof message.message_ref !== "string" ||
        (message.role !== "user" && message.role !== "assistant")
      ) {
        continue;
      }

      classified.push({
        message_ref: message.message_ref,
        role: message.role,
        act: "substantive",
      });
    }

    return classified;
  } catch {
    return [];
  }
}

function defaultClosureLoopClassificationResponse(
  messages: readonly LLMMessage[] = [],
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_closure_loop",
        name: "ClassifyClosureLoopDialogueActs",
        input: {
          messages: defaultClosureLoopClassifiedMessages(messages),
          confidence: 0,
          rationale: "No closure-loop classification scripted.",
        },
      },
    ],
  };
}

function defaultRelationalClaimAuditResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_relational_claim_audit",
        name: "EmitClaimAudit",
        input: {
          claims: [],
        },
      },
    ],
  };
}

function defaultClosureResponseAuditResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_closure_response_audit",
        name: "EmitClosureResponseAudit",
        input: {
          spans: [],
          response_shape: "no_closure",
          reason: "No closure-response audit scripted.",
        },
      },
    ],
  };
}

export class FakeLLMClient implements LLMClient {
  private readonly usageSink?: TokenUsageSink;
  private readonly oauthToolNameTransport: boolean;
  readonly requests: LLMCompleteOptions[] = [];
  readonly converseRequests: LLMConverseOptions[] = [];
  private readonly responses: FakeLLMResponse[];

  constructor(options: FakeLLMClientOptions = {}) {
    this.responses = [...(options.responses ?? [])];
    this.usageSink = options.usageSink;
    this.oauthToolNameTransport = options.oauthToolNameTransport ?? false;
  }

  pushResponse(response: FakeLLMResponse): void {
    this.responses.push(response);
  }

  async complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    const transportOptions = this.oauthToolNameTransport
      ? oauthTransportCompleteOptions(options)
      : options;
    const response = this.responses[0];

    if (
      isRecallExpansionFallbackRequest(options) &&
      typeof response !== "function" &&
      !isRecallExpansionResponse(response)
    ) {
      throw new LLMError("FakeLLMClient has no scripted recall expansion response available");
    }

    this.requests.push(transportOptions);

    if (
      isStopCommitmentFallbackRequest(options) &&
      typeof response !== "function" &&
      !isStopCommitmentResponse(response)
    ) {
      return defaultStopCommitmentResponse();
    }

    if (isProceduralContextFallbackRequest(options) && !isProceduralContextResponse(response)) {
      return defaultProceduralContextResponse();
    }

    if (
      isPendingActionJudgeFallbackRequest(options) &&
      typeof response !== "function" &&
      !isPendingActionJudgeResponse(response)
    ) {
      return defaultPendingActionJudgeResponse();
    }

    if (
      isCorrectivePreferenceFallbackRequest(options) &&
      !isCorrectivePreferenceResponse(response)
    ) {
      return defaultCorrectivePreferenceResponse();
    }

    if (
      isGoalPromotionFallbackRequest(options) &&
      typeof response !== "function" &&
      !isGoalPromotionResponse(response)
    ) {
      return defaultGoalPromotionResponse();
    }

    if (
      isActionStateExtractorFallbackRequest(options) &&
      scriptedResponseBudget(response) !== "action-state-extractor" &&
      !isActionStateResponse(response)
    ) {
      return defaultActionStateResponse();
    }

    if (
      isFrameAnomalyClassifierFallbackRequest(options) &&
      scriptedResponseBudget(response) !== "frame-anomaly-classifier" &&
      !isFrameAnomalyResponse(response)
    ) {
      return defaultFrameAnomalyResponse();
    }

    if (
      isPersonaRoleBleedClassifierFallbackRequest(options) &&
      scriptedResponseBudget(response) !== "persona-role-bleed-classifier" &&
      !isPersonaRoleBleedResponse(response)
    ) {
      return defaultPersonaRoleBleedResponse();
    }

    if (
      isClosureLoopClassifierFallbackRequest(options) &&
      scriptedResponseBudget(response) !== "closure-loop-classifier" &&
      !isClosureLoopClassificationResponse(response)
    ) {
      return defaultClosureLoopClassificationResponse(options.messages);
    }

    if (
      isRelationalClaimAuditorFallbackRequest(options) &&
      !isRelationalClaimAuditResponse(response)
    ) {
      return defaultRelationalClaimAuditResponse();
    }

    if (
      isClosureResponseAuditorFallbackRequest(options) &&
      scriptedResponseBudget(response) !== "closure-response-auditor" &&
      !isClosureResponseAuditResponse(response)
    ) {
      return defaultClosureResponseAuditResponse();
    }

    this.responses.shift();

    if (response === undefined) {
      throw new LLMError("FakeLLMClient has no scripted response available");
    }

    const resolved =
      typeof response === "function"
        ? await (
            response as (
              options: LLMCompleteOptions,
            ) => FakeLLMResponseValue | Promise<FakeLLMResponseValue>
          )(transportOptions)
        : response;
    const normalized = withFakeStructuredOutput(
      transportOptions,
      normalizeFakeCompleteResponse(resolved),
    );

    if (this.usageSink !== undefined) {
      await this.usageSink({
        budget: options.budget,
        model: options.model,
        input_tokens: normalized.input_tokens,
        output_tokens: normalized.output_tokens,
      });
    }

    return normalized;
  }

  async converse(options: LLMConverseOptions): Promise<LLMConverseResult> {
    const transportOptions = this.oauthToolNameTransport
      ? oauthTransportConverseOptions(options)
      : options;
    this.converseRequests.push(transportOptions);
    this.requests.push(toCompleteCompatibleRequest(transportOptions));
    const response = this.responses.shift();

    if (response === undefined) {
      throw new LLMError("FakeLLMClient has no scripted response available");
    }

    const resolved =
      typeof response === "function"
        ? await (
            response as (
              options: LLMConverseOptions,
            ) => FakeLLMResponseValue | Promise<FakeLLMResponseValue>
          )(transportOptions)
        : response;
    const normalized = withFakeConversationStructuredOutput(
      transportOptions,
      normalizeFakeConverseResponse(resolved),
    );

    if (this.usageSink !== undefined) {
      await this.usageSink({
        budget: options.budget,
        model: options.model,
        input_tokens: normalized.input_tokens,
        output_tokens: normalized.output_tokens,
      });
    }

    return normalized;
  }
}
