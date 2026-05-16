import type { CommitmentType } from "../../memory/commitments/index.js";

export const DECISION_ARTIFACT_COMMITMENT_CANONICALIZATION_TYPES = [
  "promise",
  "rule",
] as const satisfies readonly CommitmentType[];

export type DecisionArtifactCommitmentCanonicalizationType =
  (typeof DECISION_ARTIFACT_COMMITMENT_CANONICALIZATION_TYPES)[number];
