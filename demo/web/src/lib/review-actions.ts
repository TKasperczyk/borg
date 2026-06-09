import {
  patchCorrectionReview,
  patchReview,
  patchReviewItem,
  resolveCreatorDirectiveReconciliation,
} from "../api/client";
import type { ReviewKind, ReviewResolution, ReviewRow } from "../api/types";

export const GENERIC_REVIEW_ACTIONS: Record<ReviewKind, ReviewResolution[]> = {
  contradiction: ["keep_both", "supersede", "invalidate", "dismiss"],
  duplicate: ["keep_both", "supersede", "invalidate", "dismiss"],
  new_insight: ["accept", "invalidate", "dismiss"],
  misattribution: ["accept", "reject", "dismiss"],
  temporal_drift: ["accept", "reject", "dismiss"],
  identity_inconsistency: ["accept", "reject", "dismiss"],
  correction: ["accept", "reject"],
  belief_revision: ["dismiss"],
  skill_split: ["accept", "reject"],
  creator_directive_reconciliation: [],
  commitment_reconciliation: ["accept", "reject", "dismiss", "keep"],
};

export type ResolveReviewActionInput = {
  row: ReviewRow;
  action: ReviewResolution;
  note?: string;
  winnerNodeId?: string;
  survivorId?: string;
};

function isCorrectionReviewAction(action: ReviewResolution): action is "accept" | "reject" {
  return action === "accept" || action === "reject";
}

function optionalNote(note: string | undefined): string | undefined {
  const trimmed = note?.trim();
  return trimmed === undefined || trimmed.length === 0 ? undefined : trimmed;
}

export async function resolveReviewAction({
  row,
  action,
  note,
  winnerNodeId,
  survivorId,
}: ResolveReviewActionInput): Promise<ReviewRow> {
  const trimmedNote = optionalNote(note);

  if (row.kind === "correction") {
    if (!isCorrectionReviewAction(action)) {
      throw new Error("correction reviews only support accept or reject");
    }
    return patchCorrectionReview(row.id, {
      action,
      ...(trimmedNote === undefined ? {} : { note: trimmedNote }),
    });
  }

  if (row.kind === "belief_revision") {
    if (action !== "dismiss") {
      throw new Error("belief revision reviews only support dismiss");
    }
    return patchReviewItem(row.id, {
      action: "dismiss",
      ...(trimmedNote === undefined ? {} : { note: trimmedNote }),
    });
  }

  if (row.kind === "creator_directive_reconciliation") {
    if (action === "keep") {
      return resolveCreatorDirectiveReconciliation(row.id, {
        action: "keep",
        ...(trimmedNote === undefined ? {} : { reason: trimmedNote }),
      });
    }

    if (action === "supersede") {
      if (survivorId === undefined || survivorId.length === 0) {
        throw new Error("select a scope survivor");
      }
      return resolveCreatorDirectiveReconciliation(row.id, {
        action: "supersede",
        survivor_id: survivorId,
        ...(trimmedNote === undefined ? {} : { reason: trimmedNote }),
      });
    }

    throw new Error("creator directive reconciliation only supports keep or supersede");
  }

  return patchReview(row.id, {
    action,
    ...(trimmedNote === undefined ? {} : { note: trimmedNote }),
    ...(winnerNodeId === undefined ? {} : { winner_node_id: winnerNodeId }),
  });
}
