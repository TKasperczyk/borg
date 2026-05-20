import type { CommitmentType } from "../commitments/types.js";

export const SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES = [
  "promise",
  "rule",
] as const satisfies readonly CommitmentType[];

export type SharedStateCommitmentCanonicalizationType =
  (typeof SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES)[number];
