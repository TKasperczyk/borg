import { StorageError } from "../../util/errors.js";
import {
  type AutobiographicalPeriodId,
  type CommitmentId,
  type GrowthMarkerId,
  type OpenQuestionId,
} from "../../util/ids.js";
import { type Provenance } from "../common/provenance.js";
import { CommitmentRepository, commitmentPatchSchema, type CommitmentRecord } from "../commitments/index.js";
import {
  AutobiographicalRepository,
  GrowthMarkersRepository,
  GoalsRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
  type AutobiographicalPeriod,
  type GrowthMarker,
  type OpenQuestion,
  autobiographicalPeriodPatchSchema,
  growthMarkerPatchSchema,
  openQuestionPatchSchema,
} from "../self/index.js";
import {
  goalPatchSchema,
  type GoalRecord,
  traitPatchSchema,
  valuePatchSchema,
  type TraitRecord,
  type ValueRecord,
} from "../self/types.js";

import { IdentityEventRepository } from "./repository.js";
import { IdentityGuard, type IdentityGuardState } from "./guard.js";

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
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  autobiographicalRepository: AutobiographicalRepository;
  growthMarkersRepository: GrowthMarkersRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  commitmentRepository: CommitmentRepository;
  identityEventRepository: IdentityEventRepository;
  guard?: IdentityGuard;
};

function goalGuardState(current: GoalRecord): IdentityGuardState {
  return {
    state: current.status === "active" ? "established" : "candidate",
    provenance: current.provenance,
  };
}

function autobiographicalPeriodGuardState(current: AutobiographicalPeriod): IdentityGuardState {
  return {
    state: "established",
    provenance: current.provenance,
  };
}

function growthMarkerGuardState(current: GrowthMarker): IdentityGuardState {
  return {
    state: "established",
    provenance: current.provenance,
  };
}

function openQuestionGuardState(current: OpenQuestion): IdentityGuardState {
  const relatedEpisodeBackedProvenance =
    current.provenance?.kind === "episodes"
      ? current.provenance
      : current.provenance === null && current.related_episode_ids.length > 0
        ? {
            kind: "episodes" as const,
            episode_ids: [...new Set(current.related_episode_ids)],
          }
        : current.provenance ?? undefined;

  return {
    state: current.status === "open" ? "established" : "candidate",
    provenance: relatedEpisodeBackedProvenance,
  };
}

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
      record: this.options.valuesRepository.update(
        valueId,
        {
          ...parsedPatch,
          provenance,
        },
        provenance,
        {
          reason: options.reason,
          reviewItemId: options.reviewItemId,
          overwriteWithoutReview: decision.overwrite_without_review,
        },
      ),
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
      record: this.options.traitsRepository.update(
        traitId,
        {
          ...parsedPatch,
          provenance,
        },
        provenance,
        {
          reason: options.reason,
          reviewItemId: options.reviewItemId,
          overwriteWithoutReview: decision.overwrite_without_review,
        },
      ),
      overwriteWithoutReview: decision.overwrite_without_review,
    };
  }

  updateGoal(
    goalId: GoalRecord["id"],
    patch: unknown,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<GoalRecord> {
    const current = this.options.goalsRepository.get(goalId);

    if (current === null) {
      throw new StorageError(`Unknown goal id: ${goalId}`, {
        code: "GOAL_NOT_FOUND",
      });
    }

    const parsedPatch = goalPatchSchema.parse(patch);

    if (Object.keys(parsedPatch).length === 0) {
      return {
        status: "applied",
        record: current,
        overwriteWithoutReview: false,
      };
    }

    const decision = this.guard.evaluateChange({
      current: goalGuardState(current),
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
      record: this.options.goalsRepository.update(
        goalId,
        {
          ...parsedPatch,
          provenance,
        },
        provenance,
        {
          reason: options.reason,
          reviewItemId: options.reviewItemId,
          overwriteWithoutReview: decision.overwrite_without_review,
        },
      ),
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

    const record = this.options.commitmentRepository.update(
      commitmentId,
      {
        ...parsedPatch,
        provenance,
      },
      provenance,
      {
        reason: options.reason,
        reviewItemId: options.reviewItemId,
        overwriteWithoutReview: decision.overwrite_without_review,
      },
    );

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

  updatePeriod(
    periodId: AutobiographicalPeriodId,
    patch: unknown,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<AutobiographicalPeriod> {
    const current = this.options.autobiographicalRepository.getPeriod(periodId);

    if (current === null) {
      throw new StorageError(`Unknown autobiographical period id: ${periodId}`, {
        code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
      });
    }

    const parsedPatch = autobiographicalPeriodPatchSchema.parse(patch);

    if (Object.keys(parsedPatch).length === 0) {
      return {
        status: "applied",
        record: current,
        overwriteWithoutReview: false,
      };
    }

    const decision = this.guard.evaluateChange({
      current: autobiographicalPeriodGuardState(current),
      provenance,
      throughReview: options.throughReview,
    });

    if (!decision.allowed) {
      return {
        status: "requires_review",
        current,
      };
    }

    const record = this.options.autobiographicalRepository.upsertPeriod({
      ...current,
      ...parsedPatch,
      provenance,
    });

    this.options.identityEventRepository.record({
      record_type: "autobiographical_period",
      record_id: periodId,
      action: options.reviewItemId === null || options.reviewItemId === undefined ? "update" : "correction_apply",
      old_value: current,
      new_value: record,
      reason: options.reason ?? null,
      provenance,
      review_item_id: options.reviewItemId ?? null,
      overwrite_without_review: decision.overwrite_without_review,
    });

    return {
      status: "applied",
      record,
      overwriteWithoutReview: decision.overwrite_without_review,
    };
  }

  updateGrowthMarker(
    markerId: GrowthMarkerId,
    patch: unknown,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<GrowthMarker> {
    const current = this.options.growthMarkersRepository.get(markerId);

    if (current === null) {
      throw new StorageError(`Unknown growth marker id: ${markerId}`, {
        code: "GROWTH_MARKER_NOT_FOUND",
      });
    }

    const parsedPatch = growthMarkerPatchSchema.parse(patch);

    if (Object.keys(parsedPatch).length === 0) {
      return {
        status: "applied",
        record: current,
        overwriteWithoutReview: false,
      };
    }

    const decision = this.guard.evaluateChange({
      current: growthMarkerGuardState(current),
      provenance,
      throughReview: options.throughReview,
    });

    if (!decision.allowed) {
      return {
        status: "requires_review",
        current,
      };
    }

    const record = this.options.growthMarkersRepository.update(markerId, {
      ...parsedPatch,
      provenance,
    });

    this.options.identityEventRepository.record({
      record_type: "growth_marker",
      record_id: markerId,
      action: options.reviewItemId === null || options.reviewItemId === undefined ? "update" : "correction_apply",
      old_value: current,
      new_value: record,
      reason: options.reason ?? null,
      provenance,
      review_item_id: options.reviewItemId ?? null,
      overwrite_without_review: decision.overwrite_without_review,
    });

    return {
      status: "applied",
      record,
      overwriteWithoutReview: decision.overwrite_without_review,
    };
  }

  updateOpenQuestion(
    openQuestionId: OpenQuestionId,
    patch: unknown,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<OpenQuestion> {
    const current = this.options.openQuestionsRepository.get(openQuestionId);

    if (current === null) {
      throw new StorageError(`Unknown open question id: ${openQuestionId}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
    }

    const parsedPatch = openQuestionPatchSchema.parse(patch);

    if (Object.keys(parsedPatch).length === 0) {
      return {
        status: "applied",
        record: current,
        overwriteWithoutReview: false,
      };
    }

    const decision = this.guard.evaluateChange({
      current: openQuestionGuardState(current),
      provenance,
      throughReview: options.throughReview,
    });

    if (!decision.allowed) {
      return {
        status: "requires_review",
        current,
      };
    }

    const record = this.options.openQuestionsRepository.update(openQuestionId, {
      ...parsedPatch,
      provenance,
    });

    this.options.identityEventRepository.record({
      record_type: "open_question",
      record_id: openQuestionId,
      action: options.reviewItemId === null || options.reviewItemId === undefined ? "update" : "correction_apply",
      old_value: current,
      new_value: record,
      reason: options.reason ?? null,
      provenance,
      review_item_id: options.reviewItemId ?? null,
      overwrite_without_review: decision.overwrite_without_review,
    });

    return {
      status: "applied",
      record,
      overwriteWithoutReview: decision.overwrite_without_review,
    };
  }
}
