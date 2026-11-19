import { z } from "zod";

export const REVIEW_KINDS = [
  "contradiction",
  "duplicate",
  "new_insight",
  "misattribution",
  "temporal_drift",
  "identity_inconsistency",
  "correction",
  "belief_revision",
  "skill_split",
  "creator_directive_reconciliation",
  "commitment_reconciliation",
] as const;
const LEGACY_REVIEW_KINDS = ["relationship_claim_ungrounded"] as const;
const STORED_REVIEW_KINDS = [...REVIEW_KINDS, ...LEGACY_REVIEW_KINDS] as const;
export const REVIEW_RESOLUTIONS = [
  "keep_both",
  "supersede",
  "invalidate",
  "dismiss",
  "accept",
  "reject",
  "keep",
  "weaken",
  "archive_node",
  "invalidate_edge",
] as const;

export const reviewKindSchema = z.enum(REVIEW_KINDS);
export const storedReviewKindSchema = z.enum(STORED_REVIEW_KINDS);
export const reviewResolutionSchema = z.enum(REVIEW_RESOLUTIONS);
