import type { CommitmentRepository } from "../commitments/repository.js";
import type { CommitmentRecord } from "../commitments/types.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import type { CommitmentId } from "../../util/ids.js";
import type { LifecycleOperationResult } from "./types.js";

export type SupersedeCommitmentRepository = Pick<CommitmentRepository, "supersede">;

export function supersedeCommitment(input: {
  commitmentId: CommitmentId;
  replacementCommitmentId: CommitmentId;
  repository: SupersedeCommitmentRepository;
}): LifecycleOperationResult<{ commitment: CommitmentRecord | null }> {
  try {
    const commitment = input.repository.supersede(
      input.commitmentId,
      input.replacementCommitmentId,
    );

    if (commitment === null) {
      return {
        status: "no_op",
        reason: "missing",
        value: {
          commitment: null,
        },
      };
    }

    return {
      status: "success",
      value: {
        commitment,
      },
    };
  } catch (error) {
    if (error instanceof IdentityCasMismatchError) {
      return {
        status: "conflict",
        error,
      };
    }

    throw error;
  }
}
