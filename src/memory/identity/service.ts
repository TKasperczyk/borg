import { StorageError } from "../../util/errors.js";
import {
  type AutobiographicalPeriodId,
  type CommitmentId,
  type GrowthMarkerId,
  type OpenQuestionId,
} from "../../util/ids.js";
import { type Provenance } from "../common/provenance.js";
import {
  CommitmentRepository,
  commitmentPatchSchema,
  type CommitmentRecord,
} from "../commitments/index.js";
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
  buildOpenQuestionDedupeKey,
  growthMarkerPatchSchema,
  openQuestionPatchSchema,
} from "../self/index.js";
import {
  goalPatchSchema,
  type GoalRecord,
  type GoalStatus,
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
  };
}

function autobiographicalPeriodGuardState(current: AutobiographicalPeriod): IdentityGuardState {
  return {
    state: "established",
  };
}

function growthMarkerGuardState(current: GrowthMarker): IdentityGuardState {
  return {
    state: "established",
  };
}

function openQuestionGuardState(current: OpenQuestion): IdentityGuardState {
  return {
    state: current.status === "open" ? "established" : "candidate",
  };
}

function identityEventAction(
  defaultAction: string,
  options: IdentityUpdateOptions,
): "correction_apply" | string {
  return options.reviewItemId === null || options.reviewItemId === undefined
    ? defaultAction
    : "correction_apply";
}

function openQuestionCreationProvenance(
  input: Parameters<OpenQuestionsRepository["add"]>[0],
): Provenance {
  if (input.provenance !== undefined && input.provenance !== null) {
    return input.provenance;
  }

  if (input.related_episode_ids !== undefined && input.related_episode_ids.length > 0) {
    return {
      kind: "episodes",
      episode_ids: [...input.related_episode_ids],
    };
  }

  if (input.source === "user") {
    return {
      kind: "manual",
    };
  }

  if (
    input.source === "reflection" ||
    input.source === "contradiction" ||
    input.source === "ruminator" ||
    input.source === "overseer"
  ) {
    return {
      kind: "offline",
      process: input.source,
    };
  }

  return {
    kind: "system",
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

  addValue(input: Parameters<ValuesRepository["add"]>[0]): ValueRecord {
    const decision = this.guard.evaluateChange({
      current: null,
      provenance: input.provenance,
    });

    if (!decision.allowed) {
      throw new StorageError("Value creation unexpectedly required review", {
        code: "IDENTITY_GUARD_CREATE_REJECTED",
      });
    }

    return this.options.valuesRepository.add(input);
  }

  reinforceValue(
    valueId: ValueRecord["id"],
    provenance: Provenance,
    timestamp?: number,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<ValueRecord> {
    const current = this.options.valuesRepository.get(valueId);

    if (current === null) {
      throw new StorageError(`Unknown value id: ${valueId}`, {
        code: "VALUE_NOT_FOUND",
      });
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
      record: this.options.valuesRepository.reinforce(valueId, provenance, timestamp),
    };
  }

  addGoal(input: Parameters<GoalsRepository["add"]>[0]): GoalRecord {
    const decision = this.guard.evaluateChange({
      current: null,
      provenance: input.provenance,
    });

    if (!decision.allowed) {
      throw new StorageError("Goal creation unexpectedly required review", {
        code: "IDENTITY_GUARD_CREATE_REJECTED",
      });
    }

    return this.options.goalsRepository.add(input);
  }

  addPeriod(
    input: Parameters<AutobiographicalRepository["upsertPeriod"]>[0],
  ): AutobiographicalPeriod {
    if (input.id !== undefined && this.options.autobiographicalRepository.getPeriod(input.id)) {
      throw new StorageError(`Autobiographical period already exists: ${input.id}`, {
        code: "AUTOBIOGRAPHICAL_PERIOD_ALREADY_EXISTS",
      });
    }

    const currentOpenPeriod =
      input.end_ts === undefined || input.end_ts === null
        ? this.options.autobiographicalRepository.currentPeriod()
        : null;
    const periodClosedByCreate =
      currentOpenPeriod !== null && (input.id === undefined || currentOpenPeriod.id !== input.id)
        ? currentOpenPeriod
        : null;

    if (periodClosedByCreate !== null) {
      const closeDecision = this.guard.evaluateChange({
        current: autobiographicalPeriodGuardState(periodClosedByCreate),
        provenance: input.provenance,
      });

      if (!closeDecision.allowed) {
        throw new StorageError(
          "Autobiographical period creation would close an established period and requires review",
          {
            code: "IDENTITY_GUARD_CREATE_REJECTED",
          },
        );
      }
    }

    const decision = this.guard.evaluateChange({
      current: null,
      provenance: input.provenance,
    });

    if (!decision.allowed) {
      throw new StorageError("Autobiographical period creation unexpectedly required review", {
        code: "IDENTITY_GUARD_CREATE_REJECTED",
      });
    }

    return this.options.identityEventRepository.runInTransaction(() => {
      const period = this.options.autobiographicalRepository.upsertPeriod(input);

      if (periodClosedByCreate !== null) {
        const closedPeriod = this.options.autobiographicalRepository.getPeriod(
          periodClosedByCreate.id,
        );

        if (closedPeriod === null) {
          throw new StorageError(
            `Unknown autobiographical period id: ${periodClosedByCreate.id}`,
            {
              code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
            },
          );
        }

        this.options.identityEventRepository.record({
          record_type: "autobiographical_period",
          record_id: periodClosedByCreate.id,
          action: "close",
          old_value: periodClosedByCreate,
          new_value: closedPeriod,
          provenance: input.provenance,
        });
      }

      this.options.identityEventRepository.record({
        record_type: "autobiographical_period",
        record_id: period.id,
        action: "create",
        old_value: null,
        new_value: period,
        provenance: input.provenance,
      });

      return period;
    });
  }

  updateGoalStatus(
    goalId: GoalRecord["id"],
    status: GoalStatus,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<GoalRecord> {
    return this.updateGoal(goalId, { status }, provenance, options);
  }

  updateGoalProgress(
    goalId: GoalRecord["id"],
    progressNotes: string,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<GoalRecord> {
    return this.updateGoal(goalId, { progress_notes: progressNotes }, provenance, options);
  }

  addTrait(input: Parameters<TraitsRepository["reinforce"]>[0]): IdentityUpdateResult<TraitRecord> {
    return this.reinforceTrait(input);
  }

  reinforceTrait(
    input: Parameters<TraitsRepository["reinforce"]>[0],
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<TraitRecord> {
    const current =
      this.options.traitsRepository.list().find((trait) => trait.label === input.label) ?? null;

    if (current === null) {
      return {
        status: "applied",
        record: this.options.traitsRepository.reinforce(input),
      };
    }

    const decision = this.guard.evaluateChange({
      current,
      provenance: input.provenance,
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
      record: this.options.traitsRepository.reinforce(input),
    };
  }

  addGrowthMarker(input: Parameters<GrowthMarkersRepository["add"]>[0]): GrowthMarker {
    const decision = this.guard.evaluateChange({
      current: null,
      provenance: input.provenance,
    });

    if (!decision.allowed) {
      throw new StorageError("Growth marker creation unexpectedly required review", {
        code: "IDENTITY_GUARD_CREATE_REJECTED",
      });
    }

    return this.options.identityEventRepository.runInTransaction(() => {
      const marker = this.options.growthMarkersRepository.add(input);

      this.options.identityEventRepository.record({
        record_type: "growth_marker",
        record_id: marker.id,
        action: "create",
        old_value: null,
        new_value: marker,
        provenance: input.provenance,
      });

      return marker;
    });
  }

  addOpenQuestion(input: Parameters<OpenQuestionsRepository["add"]>[0]): OpenQuestion {
    const provenance = openQuestionCreationProvenance(input);
    const existing = this.options.openQuestionsRepository.getByDedupeKey(
      buildOpenQuestionDedupeKey({
        question: input.question,
        relatedEpisodeIds: input.related_episode_ids ?? [],
        relatedSemanticNodeIds: input.related_semantic_node_ids ?? [],
      }),
    );

    if (existing !== null) {
      return existing;
    }

    if (input.id !== undefined && this.options.openQuestionsRepository.get(input.id) !== null) {
      throw new StorageError(`Open question already exists: ${input.id}`, {
        code: "OPEN_QUESTION_ALREADY_EXISTS",
      });
    }

    const decision = this.guard.evaluateChange({
      current: null,
      provenance,
    });

    if (!decision.allowed) {
      throw new StorageError("Open question creation unexpectedly required review", {
        code: "IDENTITY_GUARD_CREATE_REJECTED",
      });
    }

    return this.options.identityEventRepository.runInTransaction(() => {
      const question = this.options.openQuestionsRepository.add(input);

      this.options.identityEventRepository.record({
        record_type: "open_question",
        record_id: question.id,
        action: "create",
        old_value: null,
        new_value: question,
        provenance,
      });

      return question;
    });
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
        },
      ),
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
        },
      ),
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
        },
      ),
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
    };
  }

  closePeriod(
    periodId: AutobiographicalPeriodId,
    closedAt: number,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<AutobiographicalPeriod> {
    const current = this.options.autobiographicalRepository.getPeriod(periodId);

    if (current === null) {
      throw new StorageError(`Unknown autobiographical period id: ${periodId}`, {
        code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
      });
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

    const record = this.options.identityEventRepository.runInTransaction(() => {
      this.options.autobiographicalRepository.closePeriod(periodId, closedAt);
      const updated = this.options.autobiographicalRepository.getPeriod(periodId);

      if (updated === null) {
        throw new StorageError(`Unknown autobiographical period id: ${periodId}`, {
          code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
        });
      }

      this.options.identityEventRepository.record({
        record_type: "autobiographical_period",
        record_id: periodId,
        action: identityEventAction("close", options),
        old_value: current,
        new_value: updated,
        reason: options.reason ?? null,
        provenance,
        review_item_id: options.reviewItemId ?? null,
      });

      return updated;
    });

    return {
      status: "applied",
      record,
    };
  }

  resolveOpenQuestion(
    openQuestionId: OpenQuestionId,
    resolution: Parameters<OpenQuestionsRepository["resolve"]>[1],
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<OpenQuestion> {
    const current = this.options.openQuestionsRepository.get(openQuestionId);

    if (current === null) {
      throw new StorageError(`Unknown open question id: ${openQuestionId}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
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

    const record = this.options.identityEventRepository.runInTransaction(() => {
      const resolved = this.options.openQuestionsRepository.resolve(openQuestionId, resolution);

      this.options.identityEventRepository.record({
        record_type: "open_question",
        record_id: openQuestionId,
        action: identityEventAction("resolve", options),
        old_value: current,
        new_value: resolved,
        reason: options.reason ?? null,
        provenance,
        review_item_id: options.reviewItemId ?? null,
      });

      return resolved;
    });

    return {
      status: "applied",
      record,
    };
  }

  abandonOpenQuestion(
    openQuestionId: OpenQuestionId,
    reason: string,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<OpenQuestion> {
    const current = this.options.openQuestionsRepository.get(openQuestionId);

    if (current === null) {
      throw new StorageError(`Unknown open question id: ${openQuestionId}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
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

    const record = this.options.identityEventRepository.runInTransaction(() => {
      const abandoned = this.options.openQuestionsRepository.abandon(openQuestionId, reason);

      this.options.identityEventRepository.record({
        record_type: "open_question",
        record_id: openQuestionId,
        action: identityEventAction("abandon", options),
        old_value: current,
        new_value: abandoned,
        reason: options.reason ?? reason,
        provenance,
        review_item_id: options.reviewItemId ?? null,
      });

      return abandoned;
    });

    return {
      status: "applied",
      record,
    };
  }

  bumpOpenQuestionUrgency(
    openQuestionId: OpenQuestionId,
    delta: number,
    provenance: Provenance,
    options: IdentityUpdateOptions = {},
  ): IdentityUpdateResult<OpenQuestion> {
    const current = this.options.openQuestionsRepository.get(openQuestionId);

    if (current === null) {
      throw new StorageError(`Unknown open question id: ${openQuestionId}`, {
        code: "OPEN_QUESTION_NOT_FOUND",
      });
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

    const record = this.options.identityEventRepository.runInTransaction(() => {
      const bumped = this.options.openQuestionsRepository.bumpUrgency(openQuestionId, delta);

      this.options.identityEventRepository.record({
        record_type: "open_question",
        record_id: openQuestionId,
        action: identityEventAction("bump_urgency", options),
        old_value: current,
        new_value: bumped,
        reason: options.reason ?? null,
        provenance,
        review_item_id: options.reviewItemId ?? null,
      });

      return bumped;
    });

    return {
      status: "applied",
      record,
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

    const record = this.options.identityEventRepository.runInTransaction(() => {
      const updated = this.options.autobiographicalRepository.upsertPeriod({
        ...current,
        ...parsedPatch,
        provenance,
      });

      this.options.identityEventRepository.record({
        record_type: "autobiographical_period",
        record_id: periodId,
        action:
          options.reviewItemId === null || options.reviewItemId === undefined
            ? "update"
            : "correction_apply",
        old_value: current,
        new_value: updated,
        reason: options.reason ?? null,
        provenance,
        review_item_id: options.reviewItemId ?? null,
      });

      return updated;
    });

    return {
      status: "applied",
      record,
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

    const record = this.options.identityEventRepository.runInTransaction(() => {
      const updated = this.options.growthMarkersRepository.update(markerId, {
        ...parsedPatch,
        provenance,
      });

      this.options.identityEventRepository.record({
        record_type: "growth_marker",
        record_id: markerId,
        action:
          options.reviewItemId === null || options.reviewItemId === undefined
            ? "update"
            : "correction_apply",
        old_value: current,
        new_value: updated,
        reason: options.reason ?? null,
        provenance,
        review_item_id: options.reviewItemId ?? null,
      });

      return updated;
    });

    return {
      status: "applied",
      record,
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

    const record = this.options.identityEventRepository.runInTransaction(() => {
      const updated = this.options.openQuestionsRepository.update(openQuestionId, {
        ...parsedPatch,
        provenance,
      });

      this.options.identityEventRepository.record({
        record_type: "open_question",
        record_id: openQuestionId,
        action:
          options.reviewItemId === null || options.reviewItemId === undefined
            ? "update"
            : "correction_apply",
        old_value: current,
        new_value: updated,
        reason: options.reason ?? null,
        provenance,
        review_item_id: options.reviewItemId ?? null,
      });

      return updated;
    });

    return {
      status: "applied",
      record,
    };
  }
}
