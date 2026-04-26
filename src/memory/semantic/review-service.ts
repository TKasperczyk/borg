import { z } from "zod";

import {
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { ReviewQueueInsertInput } from "./review-queue.js";
import type { SemanticNodeRepository } from "./repository.js";
import type { SemanticNode } from "./types.js";

const CONTRADICTION_JUDGE_TOOL_NAME = "EmitContradictionJudgment";
const contradictionJudgeSchema = z.object({
  contradicts: z.boolean(),
  confidence: z.number().min(0).max(1),
  reason: z.string().min(1).optional(),
});
const CONTRADICTION_JUDGE_TOOL = {
  name: CONTRADICTION_JUDGE_TOOL_NAME,
  description:
    "Judge whether two semantic propositions genuinely contradict each other (a direct or morphological negation, an opposite quantifier, a stated fact vs. its denial). Reinforcements, variants, elaborations, and merely similar statements are NOT contradictions.",
  inputSchema: toToolInputSchema(contradictionJudgeSchema),
} satisfies LLMToolDefinition;

export type SemanticReviewServiceOptions = {
  nodeRepository: SemanticNodeRepository;
  enqueueReview?: (input: ReviewQueueInsertInput) => ReviewQueueInsertInput | unknown;
  llmClient?: LLMClient;
  contradictionJudgeModel?: string;
  onDuplicateReviewError?: (error: unknown, node: SemanticNode) => void | Promise<void>;
};

async function judgeContradiction(
  left: Pick<SemanticNode, "label" | "description">,
  right: Pick<SemanticNode, "label" | "description">,
  llmClient: LLMClient,
  model: string,
): Promise<boolean> {
  try {
    const response = await llmClient.complete({
      model,
      system:
        "You judge whether two semantic propositions genuinely contradict each other. Contradictions include direct negation (X is true vs X is not true), morphological negation (important vs unimportant), opposite quantifiers (always vs sometimes, never vs often), and stated-fact vs its denial. Reinforcements, refinements, variants, elaborations, and merely similar statements are NOT contradictions. Default to contradicts=false when unsure.",
      messages: [
        {
          role: "user",
          content: [
            `Proposition A: ${left.label} -- ${left.description}`,
            `Proposition B: ${right.label} -- ${right.description}`,
          ].join("\n"),
        },
      ],
      tools: [CONTRADICTION_JUDGE_TOOL],
      tool_choice: { type: "tool", name: CONTRADICTION_JUDGE_TOOL_NAME },
      max_tokens: 400,
      budget: "semantic-contradiction-judge",
    });

    const call = response.tool_calls.find(
      (toolCall) => toolCall.name === CONTRADICTION_JUDGE_TOOL_NAME,
    );
    if (call === undefined) {
      return false;
    }

    const parsed = contradictionJudgeSchema.safeParse(call.input);
    if (!parsed.success) {
      return false;
    }

    return parsed.data.contradicts && parsed.data.confidence >= 0.5;
  } catch {
    // If the judge call fails for any reason, default to not flagging a
    // contradiction. Worst case: a genuine contradiction slips through; it
    // can still be caught by an explicit user-driven review later.
    return false;
  }
}

export class SemanticReviewService {
  constructor(private readonly options: SemanticReviewServiceOptions) {}

  queueDuplicateReview(node: SemanticNode): void {
    void this.reviewDuplicateCandidate(node).catch((error) => {
      try {
        void Promise.resolve(this.options.onDuplicateReviewError?.(error, node)).catch(
          () => undefined,
        );
      } catch {
        // Best-effort background observability only.
      }
    });
  }

  async reviewDuplicateCandidate(node: SemanticNode): Promise<void> {
    if (
      this.options.enqueueReview === undefined ||
      this.options.llmClient === undefined ||
      this.options.contradictionJudgeModel === undefined ||
      node.kind !== "proposition"
    ) {
      return;
    }

    const matches = await this.options.nodeRepository.searchByVector(node.embedding, {
      limit: 3,
      minSimilarity: 0.9,
      kindFilter: ["proposition"],
      includeArchived: false,
    });

    for (const match of matches) {
      if (match.node.id === node.id) {
        continue;
      }

      const substantiveDifference =
        match.node.label.toLowerCase() !== node.label.toLowerCase() ||
        match.node.description.toLowerCase() !== node.description.toLowerCase();

      if (!substantiveDifference) {
        continue;
      }

      const contradicts = await judgeContradiction(
        node,
        match.node,
        this.options.llmClient,
        this.options.contradictionJudgeModel,
      );

      if (!contradicts) {
        continue;
      }

      this.options.enqueueReview({
        kind: "duplicate",
        refs: {
          node_ids: [node.id, match.node.id],
        },
        reason: `Nearby proposition appears to conflict with ${match.node.label}`,
      });
      break;
    }
  }
}
