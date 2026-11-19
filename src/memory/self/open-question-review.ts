import { z } from "zod";
import { reviewResolutionSchema, storedReviewKindSchema } from "../review-queue/labels.js";

// One fixed-size status label per question. Review reasons, refs and applying-state
// payloads are deliberately not part of this presentation.
export const openQuestionReviewSchema = z.discriminatedUnion("status", [
  z.object({
    id: z.number().int().positive(),
    kind: storedReviewKindSchema,
    status: z.literal("open"),
  }),
  z.object({
    id: z.number().int().positive(),
    kind: storedReviewKindSchema,
    status: z.literal("resolved"),
    resolution: reviewResolutionSchema.nullable(),
    resolved_at: z.iso.datetime(),
  }),
  z.object({
    id: z.number().int().positive(),
    kind: z.null(),
    status: z.enum(["missing", "unavailable"]),
  }),
]);

export type OpenQuestionReview = z.infer<typeof openQuestionReviewSchema>;
type ReviewCarrier = { review_id?: number | null; review?: OpenQuestionReview };

export function openQuestionReviewFields(question: ReviewCarrier): { review?: OpenQuestionReview } {
  if (question.review !== undefined) return { review: question.review };
  if (question.review_id == null) return {};
  return {
    review: { id: question.review_id, kind: null, status: "unavailable" },
  };
}

export function renderOpenQuestionReview(question: ReviewCarrier): string {
  const { review } = openQuestionReviewFields(question);
  return review === undefined ? "" : ` [review=${JSON.stringify(review)}]`;
}
