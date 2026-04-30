import { z } from "zod";

import { commitmentIdSchema, commitmentPatchSchema } from "../../commitments/index.js";
import { provenanceSchema, type Provenance } from "../../common/provenance.js";
import { episodeIdSchema } from "../../episodic/index.js";
import {
  autobiographicalPeriodIdSchema,
  autobiographicalPeriodPatchSchema,
  autobiographicalPeriodSchema,
  goalIdSchema,
  goalPatchSchema,
  traitIdSchema,
  traitPatchSchema,
  valueIdSchema,
  valuePatchSchema,
} from "../../self/index.js";
import { SemanticError } from "../../../util/errors.js";
import { semanticEdgeIdSchema, semanticNodeIdSchema } from "../types.js";
import type {
  ReviewHandlerContext,
  ReviewQueueHandler,
  ReviewResolution,
} from "../review-queue.js";
import { closeSemanticEdgeFromReview } from "./semantic-edge-closure.js";

const IDENTITY_INCONSISTENCY_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "accept",
  "reject",
  "dismiss",
]);

const nonEmptyValuePatchSchema = valuePatchSchema.refine((patch) => Object.keys(patch).length > 0, {
  message: "Value repair patch must not be empty",
});
const nonEmptyTraitPatchSchema = traitPatchSchema.refine((patch) => Object.keys(patch).length > 0, {
  message: "Trait repair patch must not be empty",
});
const nonEmptyCommitmentPatchSchema = commitmentPatchSchema.refine(
  (patch) => Object.keys(patch).length > 0,
  {
    message: "Commitment repair patch must not be empty",
  },
);
const nonEmptyGoalPatchSchema = goalPatchSchema.refine((patch) => Object.keys(patch).length > 0, {
  message: "Goal repair patch must not be empty",
});
const nonEmptyAutobiographicalPeriodPatchSchema = autobiographicalPeriodPatchSchema.refine(
  (patch) => Object.keys(patch).length > 0,
  {
    message: "Autobiographical period repair patch must not be empty",
  },
);

const sourceTargetSchema = z
  .object({
    source_target_type: z.enum(["episode", "semantic_node", "semantic_edge"]).optional(),
    source_target_id: z
      .union([episodeIdSchema, semanticNodeIdSchema, semanticEdgeIdSchema])
      .optional(),
  })
  .strict();

const identityReviewBaseSchema = z
  .object({
    proposed_provenance: provenanceSchema.optional(),
    evidence_episode_ids: z.array(episodeIdSchema).min(1).optional(),
  })
  .merge(sourceTargetSchema);

const valueRepairRefsSchema = z.discriminatedUnion("repair_op", [
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("value"),
      target_id: valueIdSchema,
      repair_op: z.literal("reinforce"),
      evidence_episode_ids: z.array(episodeIdSchema).min(1),
    })
    .strict(),
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("value"),
      target_id: valueIdSchema,
      repair_op: z.literal("contradict"),
      evidence_episode_ids: z.array(episodeIdSchema).min(1),
    })
    .strict(),
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("value"),
      target_id: valueIdSchema,
      repair_op: z.literal("patch"),
      patch: nonEmptyValuePatchSchema,
    })
    .strict(),
]);

const traitRepairRefsSchema = z.discriminatedUnion("repair_op", [
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("trait"),
      target_id: traitIdSchema,
      repair_op: z.literal("reinforce"),
      evidence_episode_ids: z.array(episodeIdSchema).min(1),
    })
    .strict(),
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("trait"),
      target_id: traitIdSchema,
      repair_op: z.literal("contradict"),
      evidence_episode_ids: z.array(episodeIdSchema).min(1),
    })
    .strict(),
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("trait"),
      target_id: traitIdSchema,
      repair_op: z.literal("patch"),
      patch: nonEmptyTraitPatchSchema,
    })
    .strict(),
]);

export const identityInconsistencyReviewRefsSchema = z.union([
  ...valueRepairRefsSchema.options,
  ...traitRepairRefsSchema.options,
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("commitment"),
      target_id: commitmentIdSchema,
      repair_op: z.literal("patch"),
      patch: nonEmptyCommitmentPatchSchema,
    })
    .strict(),
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("goal"),
      target_id: goalIdSchema,
      repair_op: z.literal("patch"),
      patch: nonEmptyGoalPatchSchema,
    })
    .strict(),
  identityReviewBaseSchema
    .extend({
      target_type: z.literal("autobiographical_period"),
      target_id: autobiographicalPeriodIdSchema,
      repair_op: z.literal("patch"),
      patch: nonEmptyAutobiographicalPeriodPatchSchema,
      next_period_open_payload: autobiographicalPeriodSchema.optional(),
    })
    .strict(),
  z
    .object({
      target_type: z.literal("semantic_edge"),
      target_kind: z.literal("semantic_edge"),
      target_id: semanticEdgeIdSchema,
      suggested_valid_to: z.number().finite().optional(),
      by_edge_id: semanticEdgeIdSchema.optional(),
      reason: z.string().min(1),
      proposed_provenance: provenanceSchema.optional(),
      source_target_type: z.literal("semantic_edge"),
      source_target_id: semanticEdgeIdSchema,
    })
    .strict(),
]);

export type IdentityInconsistencyReviewRefs = z.infer<typeof identityInconsistencyReviewRefsSchema>;

function reviewProvenance(refs: { proposed_provenance?: Provenance }): Provenance {
  return refs.proposed_provenance ?? { kind: "manual" };
}

function evidenceProvenance(refs: { evidence_episode_ids: z.infer<typeof episodeIdSchema>[] }) {
  return {
    kind: "episodes" as const,
    episode_ids: refs.evidence_episode_ids,
  };
}

function assertApplied(input: { status: string; label: string; targetId: string }): void {
  if (input.status !== "applied") {
    throw new SemanticError(`${input.label} ${input.targetId} still requires review`, {
      code: "IDENTITY_REVIEW_REQUIRED",
    });
  }
}

function requireIdentityService(ctx: ReviewHandlerContext) {
  if (ctx.identityService === undefined) {
    throw new SemanticError("Identity service is required for identity patch repair", {
      code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
    });
  }

  return ctx.identityService;
}

export function createIdentityInconsistencyReviewQueueHandler(): ReviewQueueHandler<
  "identity_inconsistency",
  IdentityInconsistencyReviewRefs
> {
  return {
    kind: "identity_inconsistency",
    refsSchema: identityInconsistencyReviewRefsSchema,
    allowedResolutions: IDENTITY_INCONSISTENCY_REVIEW_RESOLUTIONS,
    transactionScope: () => "sqlite",
    apply({ item, refs, resolution, ctx }) {
      if (resolution.decision !== "accept") {
        return;
      }

      if (refs.target_type === "semantic_edge") {
        closeSemanticEdgeFromReview({
          item,
          repair: {
            edgeId: refs.target_id,
            validTo: refs.suggested_valid_to,
            byEdgeId: refs.by_edge_id,
            reason: refs.reason,
          },
          ctx,
        });
        return;
      }

      if (refs.target_type === "value") {
        if (refs.repair_op === "reinforce") {
          if (ctx.valuesRepository === undefined) {
            throw new SemanticError(
              "Values repository is required for value reinforcement repair",
              {
                code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
              },
            );
          }

          ctx.valuesRepository.reinforce(refs.target_id, evidenceProvenance(refs), ctx.clock.now());
          return;
        }

        if (refs.repair_op === "contradict") {
          if (ctx.valuesRepository === undefined) {
            throw new SemanticError(
              "Values repository is required for value contradiction repair",
              {
                code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
              },
            );
          }

          ctx.valuesRepository.recordContradiction({
            valueId: refs.target_id,
            provenance: evidenceProvenance(refs),
            timestamp: ctx.clock.now(),
          });
          return;
        }

        const result = requireIdentityService(ctx).updateValue(
          refs.target_id,
          refs.patch,
          reviewProvenance(refs),
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );
        assertApplied({
          status: result.status,
          label: "Identity patch for value",
          targetId: refs.target_id,
        });
        return;
      }

      if (refs.target_type === "trait") {
        if (refs.repair_op === "reinforce") {
          if (ctx.traitsRepository === undefined) {
            throw new SemanticError(
              "Traits repository is required for trait reinforcement repair",
              {
                code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
              },
            );
          }

          const current = ctx.traitsRepository.get(refs.target_id);

          if (current === null) {
            throw new SemanticError(
              `Unknown trait id for reinforcement repair: ${refs.target_id}`,
              {
                code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
              },
            );
          }

          ctx.traitsRepository.reinforce({
            label: current.label,
            delta: 0.05,
            provenance: evidenceProvenance(refs),
            timestamp: ctx.clock.now(),
          });
          return;
        }

        if (refs.repair_op === "contradict") {
          if (ctx.traitsRepository === undefined) {
            throw new SemanticError(
              "Traits repository is required for trait contradiction repair",
              {
                code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
              },
            );
          }

          const current = ctx.traitsRepository.get(refs.target_id);

          if (current === null) {
            throw new SemanticError(
              `Unknown trait id for contradiction repair: ${refs.target_id}`,
              {
                code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
              },
            );
          }

          ctx.traitsRepository.recordContradiction({
            label: current.label,
            provenance: evidenceProvenance(refs),
            timestamp: ctx.clock.now(),
          });
          return;
        }

        const result = requireIdentityService(ctx).updateTrait(
          refs.target_id,
          refs.patch,
          reviewProvenance(refs),
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );
        assertApplied({
          status: result.status,
          label: "Identity patch for trait",
          targetId: refs.target_id,
        });
        return;
      }

      if (refs.target_type === "commitment") {
        const result = requireIdentityService(ctx).updateCommitment(
          refs.target_id,
          refs.patch,
          reviewProvenance(refs),
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );
        assertApplied({
          status: result.status,
          label: "Identity patch for commitment",
          targetId: refs.target_id,
        });
        return;
      }

      if (refs.target_type === "goal") {
        const result = requireIdentityService(ctx).updateGoal(
          refs.target_id,
          refs.patch,
          reviewProvenance(refs),
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );
        assertApplied({
          status: result.status,
          label: "Identity patch for goal",
          targetId: refs.target_id,
        });
        return;
      }

      const identityService = requireIdentityService(ctx);
      const applyPeriodPatch = () => {
        const result = identityService.updatePeriod(
          refs.target_id,
          refs.patch,
          reviewProvenance(refs),
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );
        assertApplied({
          status: result.status,
          label: "Identity patch for autobiographical period",
          targetId: refs.target_id,
        });
      };

      if (refs.next_period_open_payload === undefined) {
        applyPeriodPatch();
        return;
      }

      if (ctx.autobiographicalRepository === undefined) {
        throw new SemanticError(
          "Autobiographical repository is required for period rollover repair",
          {
            code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
          },
        );
      }

      ctx.autobiographicalRepository.runInTransaction(() => {
        applyPeriodPatch();
        identityService.addPeriod(refs.next_period_open_payload!);
      });
    },
  };
}
