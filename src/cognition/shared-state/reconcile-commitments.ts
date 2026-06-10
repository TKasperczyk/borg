import type { CommitmentRepository } from "../../memory/commitments/index.js";
import type { SharedStateEntry } from "../../memory/shared-state/index.js";
import {
  canonicalizeCommitmentWithSharedStateEntry,
  type LifecycleTracer,
} from "../../memory/lifecycle-ops/index.js";
import type { CommitmentId } from "../../util/ids.js";
import { errorMessage, type SharedStateReconciliationResult } from "./reconciliation-summary.js";

export {
  isSharedStateArtifactCanonicalizableCommitmentType,
  isTerminalCommitment,
} from "../../memory/lifecycle-ops/index.js";

export function reconcileCommitmentCanonicalizations(input: {
  entry: SharedStateEntry;
  commitmentIds: readonly CommitmentId[];
  repository: Pick<CommitmentRepository, "get" | "revoke"> | undefined;
  retiredCommitments: Set<CommitmentId>;
  result: SharedStateReconciliationResult;
  nowMs: number;
  tracer?: LifecycleTracer;
  turnId?: string;
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
      const result = canonicalizeCommitmentWithSharedStateEntry({
        commitmentId,
        entry: input.entry,
        repository: input.repository,
        nowMs: input.nowMs,
        tracer: input.tracer,
        turnId: input.turnId,
      });

      if (
        result.status === "no_op" &&
        result.reason === "non_canonicalizable_commitment_type" &&
        result.value?.commitment !== null &&
        result.value?.commitment !== undefined
      ) {
        input.result.commitments_revoked_skipped += 1;
        input.result.skipped_commitments.push({
          channel: "commitment",
          id: commitmentId,
          artifactEntryId: input.entry.id,
          reason: "non_canonicalizable_commitment_type",
          commitmentType: result.value.commitment.type,
        });
        continue;
      }

      if (result.status === "no_op" && result.reason === "missing") {
        input.result.commitments_revoked_skipped += 1;
        input.result.errors.push({
          channel: "commitment",
          id: commitmentId,
          artifactEntryId: input.entry.id,
          message: `Unknown commitment id: ${commitmentId}`,
        });
        continue;
      }

      if (result.status === "no_op") {
        input.result.commitments_revoked_skipped += 1;
        continue;
      }

      if (result.status === "conflict") {
        input.result.commitments_revoked_skipped += 1;
        input.result.errors.push({
          channel: "commitment",
          id: commitmentId,
          artifactEntryId: input.entry.id,
          message: errorMessage(result.error),
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
