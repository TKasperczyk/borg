import { StorageError } from "../../util/errors.js";
import { type CommitmentId } from "../../util/ids.js";
import { type Provenance } from "../common/provenance.js";
import { CommitmentRepository, commitmentPatchSchema, type CommitmentRecord } from "../commitments/index.js";
import { TraitsRepository, ValuesRepository } from "../self/index.js";
import {
  traitPatchSchema,
  valuePatchSchema,
  type TraitRecord,
  type ValueRecord,
} from "../self/types.js";

import { IdentityEventRepository } from "./repository.js";
import { IdentityGuard } from "./guard.js";

export type IdentityUpdateOptions = {
  throughReview?: boolean;
  reason?: string | null;
  reviewItemId?: number | null;
};

export type IdentityUpdateResult<T> =
  | {
      status: "applied";
      record: T;
      overwriteWithoutReview: boolean;
    }
  | {
      status: "requires_review";
      current: T;
    };

export type IdentityServiceOptions = {
  valuesRepository: ValuesRepository;
  traitsRepository: TraitsRepository;
  commitmentRepository: CommitmentRepository;
  identityEventRepository: IdentityEventRepository;
  guard?: IdentityGuard;
};

export class IdentityService {
  private readonly guard: IdentityGuard;

  constructor(private readonly options: IdentityServiceOptions) {
    this.guard = options.guard ?? new IdentityGuard();
  }

  listEvents(
    ...args: Parameters<IdentityEventRepository["list"]>
  ): ReturnType<IdentityEventRepository["list"]> {
    return this.options.identityEventRepository.list(...args);
  }

  updateValue(
    valueId: ValueRecord["id"],
    patch: unknown,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<ValueRecord> {
    const current = this.options.valuesRepository.get(valueId);

    if (current === null) {
      throw new StorageError(`Unknown value id: ${valueId}`, {
        code: "VALUE_NOT_FOUND",
      });
    }

    const parsedPatch = valuePatchSchema.parse(patch);

    if (Object.keys(parsedPatch).length === 0) {
      return {
        status: "applied",
        record: current,
        overwriteWithoutReview: false,
      };
    }

    const decision = this.guard.evaluateChange({
      current,
      provenance,
      throughReview: options.throughReview,
    });

    if (!decision.allowed) {
      return {
        status: "requires_review",
        current,
      };
    }

    return {
      status: "applied",
      record: this.options.valuesRepository.update(valueId, parsedPatch, provenance, {
        reason: options.reason,
        reviewItemId: options.reviewItemId,
        overwriteWithoutReview: decision.overwrite_without_review,
      }),
      overwriteWithoutReview: decision.overwrite_without_review,
    };
  }

  updateTrait(
    traitId: TraitRecord["id"],
    patch: unknown,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<TraitRecord> {
    const current = this.options.traitsRepository.get(traitId);

    if (current === null) {
      throw new StorageError(`Unknown trait id: ${traitId}`, {
        code: "TRAIT_NOT_FOUND",
      });
    }

    const parsedPatch = traitPatchSchema.parse(patch);

    if (Object.keys(parsedPatch).length === 0) {
      return {
        status: "applied",
        record: current,
        overwriteWithoutReview: false,
      };
    }

    const decision = this.guard.evaluateChange({
      current,
      provenance,
      throughReview: options.throughReview,
    });

    if (!decision.allowed) {
      return {
        status: "requires_review",
        current,
      };
    }

    return {
      status: "applied",
      record: this.options.traitsRepository.update(traitId, parsedPatch, provenance, {
        reason: options.reason,
        reviewItemId: options.reviewItemId,
        overwriteWithoutReview: decision.overwrite_without_review,
      }),
      overwriteWithoutReview: decision.overwrite_without_review,
    };
  }

  updateCommitment(
    commitmentId: CommitmentId,
    patch: unknown,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<CommitmentRecord> {
    const current = this.options.commitmentRepository.get(commitmentId);

    if (current === null) {
      throw new StorageError(`Unknown commitment id: ${commitmentId}`, {
        code: "COMMITMENT_NOT_FOUND",
      });
    }

    const parsedPatch = commitmentPatchSchema.parse(patch);

    if (Object.keys(parsedPatch).length === 0) {
      return {
        status: "applied",
        record: current,
        overwriteWithoutReview: false,
      };
    }

    const decision = this.guard.evaluateChange({
      current: {
        state:
          current.revoked_at === null &&
          current.expired_at === null &&
          current.superseded_by === null
            ? "established"
            : "candidate",
        provenance: current.provenance,
      },
      provenance,
      throughReview: options.throughReview,
    });

    if (!decision.allowed) {
      return {
        status: "requires_review",
        current,
      };
    }

    const record = this.options.commitmentRepository.update(commitmentId, parsedPatch, provenance, {
      reason: options.reason,
      reviewItemId: options.reviewItemId,
      overwriteWithoutReview: decision.overwrite_without_review,
    });

    if (record === null) {
      throw new StorageError(`Unknown commitment id: ${commitmentId}`, {
        code: "COMMITMENT_NOT_FOUND",
      });
    }

    return {
      status: "applied",
      record,
      overwriteWithoutReview: decision.overwrite_without_review,
    };
  }
}
