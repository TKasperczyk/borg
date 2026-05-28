import { z } from "zod";

import type { ReviewQueueHandler } from "../review-queue.js";

const refsSchema = z
  .object({
    target_type: z.literal("semantic_node_candidate"),
    label: z.string().min(1),
    description: z.string().min(1),
    relationship_claim_label_families: z.array(z.string().min(1)).default([]),
    relationship_claims: z.array(z.unknown()).default([]),
    ungrounded_relationship_claims: z.array(z.unknown()).default([]),
  })
  .passthrough();

type Refs = z.infer<typeof refsSchema>;

export function createRelationshipClaimUngroundedReviewQueueHandler(): ReviewQueueHandler<
  "relationship_claim_ungrounded",
  Refs
> {
  return {
    kind: "relationship_claim_ungrounded",
    refsSchema,
    allowedResolutions: new Set(["dismiss", "reject", "accept", "keep"] as const),
    transactionScope: () => "sqlite",
    apply: () => undefined,
  };
}
