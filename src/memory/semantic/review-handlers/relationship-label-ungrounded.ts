import { z } from "zod";

import type { ReviewQueueHandler } from "../review-queue.js";

const refsSchema = z
  .object({
    target_type: z.literal("semantic_node_candidate"),
    label: z.string().min(1),
    description: z.string().min(1),
    protected_relationship_labels: z.array(z.string().min(1)),
    relationship_evidence_relational_slot_ids: z.array(z.string().min(1)),
  })
  .passthrough();

type Refs = z.infer<typeof refsSchema>;

export function createRelationshipLabelUngroundedReviewQueueHandler(): ReviewQueueHandler<
  "relationship_label_ungrounded",
  Refs
> {
  return {
    kind: "relationship_label_ungrounded",
    refsSchema,
    allowedResolutions: new Set(["dismiss", "reject", "accept", "keep"] as const),
    transactionScope: () => "sqlite",
    apply: () => undefined,
  };
}
