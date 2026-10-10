import { FailureReportTracker } from "../../util/failure-reports.js";
import { similarityThresholds, type SimilarityConfigSource } from "../../config/similarity.js";
import { z } from "zod";

import {
  callStructuredTool,
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { ReviewQueueInsertInput } from "../review-queue/review-queue.js";
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
  similarityConfig?: SimilarityConfigSource;
  nodeRepository: SemanticNodeRepository;
  enqueueReview?: (input: ReviewQueueInsertInput) => ReviewQueueInsertInput | unknown;
  llmClient?: LLMClient;
  contradictionJudgeModel?: string;
  onDuplicateReviewError?: (error: unknown, node: SemanticNode) => void | Promise<void>;
};

export type SemanticReviewQueueOptions = {
  sourceProcess?: string;
  traceTurnId?: string;
};

async function judgeContradiction(
  left: Pick<SemanticNode, "label" | "description">,
  right: Pick<SemanticNode, "label" | "description">,
  llmClient: LLMClient,
  model: string,
): Promise<boolean> {
  try {
    const result = await callStructuredTool({
      llmClient,
      request: {
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
      },
      toolName: CONTRADICTION_JUDGE_TOOL_NAME,
      parse: (input) => contradictionJudgeSchema.parse(input),
    });
    const parsed = result.parsed;

    return parsed.contradicts && parsed.confidence >= 0.5;
  } catch {
    // If the judge call fails for any reason, default to not flagging a
    // contradiction. Worst case: a genuine contradiction slips through; it
    // can still be caught by an explicit user-driven review later.
    return false;
  }
}

export class SemanticReviewService {
  private readonly pendingDuplicateReviews = new Set<Promise<void>>();
  private readonly failureReports = new FailureReportTracker("semantic duplicate review");

  constructor(private readonly options: SemanticReviewServiceOptions) {}

  queueDuplicateReview(node: SemanticNode, options: SemanticReviewQueueOptions = {}): void {
    const task = this.reviewDuplicateCandidate(node, options).catch(async (error) => {
      await this.failureReports.report(() => this.options.onDuplicateReviewError?.(error, node));
    });
    this.pendingDuplicateReviews.add(task);
    void task.finally(() => {
      this.pendingDuplicateReviews.delete(task);
    });
  }

  async flushDuplicateReviews(): Promise<void> {
    // A duplicate review can enqueue an open-question hook or append its own
    // failure report. Drain it before the review queue and before closing stores.
    while (this.pendingDuplicateReviews.size > 0) {
      await Promise.all([...this.pendingDuplicateReviews]);
    }
    await this.failureReports.flush();
  }

  async reviewDuplicateCandidate(
    node: SemanticNode,
    options: SemanticReviewQueueOptions = {},
  ): Promise<void> {
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
      minSimilarity: similarityThresholds(this.options.similarityConfig).semanticDuplicateReview,
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
        sourceProcess: options.sourceProcess,
        traceTurnId: options.traceTurnId,
      });
      break;
    }
  }
}
