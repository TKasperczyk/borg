import type {
  CommitmentRecord,
  CommitmentRepository,
  CommitmentType,
} from "../../memory/commitments/index.js";
import type { SharedStateEntry } from "../../memory/decision-artifacts/index.js";
import type { CommitmentId } from "../../util/ids.js";
import {
  SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES,
  type SharedStateCommitmentCanonicalizationType,
} from "./commitment-canonicalization.js";
import {
  RECONCILIATION_PROVENANCE,
  errorMessage,
  type SharedStateReconciliationResult,
} from "./reconciliation-summary.js";

const SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPE_SET = new Set<CommitmentType>(
  SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES,
);

export function isTerminalCommitment(
  commitment: NonNullable<ReturnType<CommitmentRepository["get"]>>,
  nowMs: number,
): boolean {
  return (
    commitment.revoked_at !== null ||
    commitment.expired_at !== null ||
    (commitment.expires_at !== null && commitment.expires_at <= nowMs) ||
    commitment.superseded_by !== null
  );
}

export function isSharedStateArtifactCanonicalizableCommitmentType(
  type: CommitmentRecord["type"],
): type is SharedStateCommitmentCanonicalizationType {
  return SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPE_SET.has(type);
}

export function reconcileCommitmentCanonicalizations(input: {
  entry: SharedStateEntry;
  commitmentIds: readonly CommitmentId[];
  repository: Pick<CommitmentRepository, "get" | "revoke"> | undefined;
  retiredCommitments: Set<CommitmentId>;
  result: SharedStateReconciliationResult;
  nowMs: number;
}): void {
  for (const commitmentId of input.commitmentIds) {
    input.result.commitments_revoked_attempted += 1;

    if (input.retiredCommitments.has(commitmentId)) {
      input.result.commitments_revoked_skipped += 1;
      continue;
    }

    if (input.repository === undefined) {
      input.result.commitments_revoked_skipped += 1;
      continue;
    }

    try {
      const commitment = input.repository.get(commitmentId);

      if (commitment !== null && isTerminalCommitment(commitment, input.nowMs)) {
        input.result.commitments_revoked_skipped += 1;
        continue;
      }

      if (
        commitment !== null &&
        !isSharedStateArtifactCanonicalizableCommitmentType(commitment.type)
      ) {
        input.result.commitments_revoked_skipped += 1;
        input.result.skipped_commitments.push({
          channel: "commitment",
          id: commitmentId,
          artifactEntryId: input.entry.id,
          reason: "non_canonicalizable_commitment_type",
          commitmentType: commitment.type,
        });
        continue;
      }

      const retired = input.repository.revoke(
        commitmentId,
        `canonicalized_by_artifact_entry_id=${input.entry.id}`,
        RECONCILIATION_PROVENANCE,
        undefined,
        {
          canonicalizedByArtifactEntryId: input.entry.id,
        },
      );

      if (retired === null) {
        input.result.commitments_revoked_skipped += 1;
        input.result.errors.push({
          channel: "commitment",
          id: commitmentId,
          artifactEntryId: input.entry.id,
          message: `Unknown commitment id: ${commitmentId}`,
        });
        continue;
      }

      input.retiredCommitments.add(commitmentId);
      input.result.commitments_retired += 1;
      input.result.commitments_revoked_succeeded += 1;
    } catch (error) {
      input.result.errors.push({
        channel: "commitment",
        id: commitmentId,
        artifactEntryId: input.entry.id,
        message: errorMessage(error),
      });
    }
  }
}
