import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { reviewKindSchema, type ReviewQueueItem } from "../semantic/index.js";
import {
  episodeIdHelpers,
  parseEpisodeId,
  parseSemanticNodeId,
  semanticNodeIdHelpers,
  type EntityId,
  type EpisodeId,
  type SemanticNodeId,
} from "../../util/ids.js";

const episodeIdToolSchema = z.string().regex(episodeIdHelpers.pattern, {
  message: "Invalid episode id",
});
const semanticNodeIdToolSchema = z.string().regex(semanticNodeIdHelpers.pattern, {
  message: "Invalid semantic node id",
});

const reviewOpenQuestionProposalSchema = z
  .object({
    question: z.string().trim().min(1),
    urgency: z.number().min(0).max(1),
    related_episode_ids: z.array(episodeIdToolSchema).default([]),
    related_semantic_node_ids: z.array(semanticNodeIdToolSchema).default([]),
  })
  .strict();

const REVIEW_OPEN_QUESTION_TOOL_NAME = "EmitReviewOpenQuestion";
export const REVIEW_OPEN_QUESTION_TOOL = {
  name: REVIEW_OPEN_QUESTION_TOOL_NAME,
  description: "Emit a structured open-question proposal for a review queue item.",
  inputSchema: toToolInputSchema(reviewOpenQuestionProposalSchema),
} satisfies LLMToolDefinition;

const REVIEW_OPEN_QUESTION_SYSTEM_PROMPT = [
  "You turn a durable memory-review queue item into one concise open question for self-memory.",
  "",
  "Use the structured fields as evidence: review kind, reason, refs, target identifiers, labels, patches, and allowed related IDs. The review item itself is already persisted, so do not restate metadata or invent missing facts.",
  "",
  "Write the question in the user's language when the review reason, labels, memory text, or surrounding fields make that language clear. If the language is not clear, choose the natural language that best fits the supplied text.",
  "",
  "Emit only IDs that are present in the allowed related ID lists. If no allowed ID is relevant, emit an empty list for that ID type.",
].join("\n");

export type OpenQuestionProposal = {
  question: string;
  urgency: number;
  related_episode_ids: EpisodeId[];
  related_semantic_node_ids: SemanticNodeId[];
};

export type ReviewOpenQuestionContext = {
  audience_entity_id: EntityId | null;
  allowed_episode_ids: readonly EpisodeId[];
  allowed_semantic_node_ids: readonly SemanticNodeId[];
};

export type ReviewOpenQuestionExtractorDegradedEvent = {
  reason: "llm_unavailable" | "llm_call_failed" | "missing_tool" | "invalid_payload";
  review_item_id: number;
  review_kind: z.infer<typeof reviewKindSchema>;
  error?: string;
};

export type ReviewOpenQuestionExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  onDegraded?: (event: ReviewOpenQuestionExtractorDegradedEvent) => void | Promise<void>;
};

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

function findProposalToolCall(result: LLMCompleteResult) {
  return result.tool_calls.find((toolCall) => toolCall.name === REVIEW_OPEN_QUESTION_TOOL_NAME);
}

function toOpenQuestionProposal(
  input: z.infer<typeof reviewOpenQuestionProposalSchema>,
): OpenQuestionProposal {
  return {
    question: input.question,
    urgency: input.urgency,
    related_episode_ids: input.related_episode_ids.map((id) => parseEpisodeId(id)),
    related_semantic_node_ids: input.related_semantic_node_ids.map((id) =>
      parseSemanticNodeId(id),
    ),
  };
}

export class ReviewOpenQuestionExtractor {
  constructor(private readonly options: ReviewOpenQuestionExtractorOptions = {}) {}

  private async reportDegraded(
    item: ReviewQueueItem,
    event: Omit<ReviewOpenQuestionExtractorDegradedEvent, "review_item_id" | "review_kind">,
  ): Promise<void> {
    try {
      await this.options.onDegraded?.({
        ...event,
        review_item_id: item.id,
        review_kind: item.kind,
      });
    } catch {
      // Observability must not turn a fail-closed extractor miss into a write failure.
    }
  }

  async extract(
    item: ReviewQueueItem,
    context: ReviewOpenQuestionContext,
  ): Promise<OpenQuestionProposal | null> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      await this.reportDegraded(item, {
        reason: "llm_unavailable",
      });
      return null;
    }

    let result: LLMCompleteResult;

    try {
      result = await this.options.llmClient.complete({
        model: this.options.model,
        system: REVIEW_OPEN_QUESTION_SYSTEM_PROMPT,
        messages: [
          {
            role: "user",
            content: JSON.stringify({
              review_item: {
                id: item.id,
                kind: item.kind,
                reason: item.reason,
                refs: item.refs,
              },
              context,
            }),
          },
        ],
        tools: [REVIEW_OPEN_QUESTION_TOOL],
        tool_choice: { type: "tool", name: REVIEW_OPEN_QUESTION_TOOL_NAME },
        max_tokens: 1_000,
        budget: "offline-review-open-question",
      });
    } catch (error) {
      await this.reportDegraded(item, {
        reason: "llm_call_failed",
        error: errorMessage(error),
      });
      return null;
    }

    const call = findProposalToolCall(result);

    if (call === undefined) {
      await this.reportDegraded(item, {
        reason: "missing_tool",
      });
      return null;
    }

    const parsed = reviewOpenQuestionProposalSchema.safeParse(call.input);

    if (!parsed.success) {
      await this.reportDegraded(item, {
        reason: "invalid_payload",
        error: parsed.error.message,
      });
      return null;
    }

    return toOpenQuestionProposal(parsed.data);
  }
}
