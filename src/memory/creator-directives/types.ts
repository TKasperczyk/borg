import { z } from "zod";

import { borgRoleSchema, type BorgRole } from "../commitments/types.js";
import { sessionAudienceRoleSchema, type SessionAudienceRole } from "../../sessions/types.js";
import {
  creatorDirectiveIdHelpers,
  entityIdHelpers,
  isSessionId,
  parseSessionId,
  streamEntryIdHelpers,
  type CreatorDirectiveId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";

export const CREATOR_DIRECTIVE_STATUSES = ["active", "superseded", "revoked"] as const;
export const CREATOR_DIRECTIVE_KINDS = [
  "self_identity",
  "subject_fact",
  "disclosure_boundary",
  "response_policy",
  "routing_instruction",
] as const;
export const CREATOR_DIRECTIVE_SUBJECT_KINDS = [
  "borg_self",
  "entity",
  "system",
  "unknown",
] as const;
export const CREATOR_DIRECTIVE_SEMANTIC_SLOTS = ["public_name"] as const;
export const CREATOR_DIRECTIVE_CONTENT_SCOPES = [
  "operator_only",
  "public",
  "allow_list",
  "subject_only",
  "all_except",
] as const;
export const CREATOR_DIRECTIVE_ACTIVATION_SCOPES = [
  "same_as_disclosure",
  "operator_only",
  "public",
  "allow_list",
  "subject_only",
  "all_except",
] as const;
export const CREATOR_DIRECTIVE_MENTION_POLICIES = [
  "proactive",
  "answer_if_asked",
  "only_if_topic_raised",
  "never_mention",
] as const;
export const CREATOR_DIRECTIVE_DENIED_AUDIENCE_BEHAVIORS = [
  "omit",
  "render_boundary_when_relevant",
] as const;
export const CREATOR_DIRECTIVE_RENDER_MODES = ["content", "boundary", "omit"] as const;
export const CREATOR_DIRECTIVE_RENDER_REASONS = [
  "public",
  "explicit_allow",
  "subject_allowed",
  "operator_only",
  "self_cognition_operator_only",
  "explicit_exclude_boundary",
  "unauthorized_omit",
  "subject_may_not_know",
  "operator_only_omitted",
  "group_contains_excluded_entity",
  "same_turn_n_plus_one",
] as const;
export const CREATOR_DIRECTIVE_ACTIVATION_REASONS = [
  "same_as_disclosure",
  "same_as_disclosure_omitted",
  "operator_only",
  "self_cognition_operator_only",
  "operator_only_omitted",
  "public",
  "explicit_allow",
  "unauthorized_omit",
  "subject_allowed",
  "subject_not_present",
  "all_except",
  "explicit_exclude",
  "group_contains_excluded_entity",
] as const;

const DEFAULT_CREATOR_DIRECTIVE_ACTIVATION_POLICY: {
  scope: "same_as_disclosure";
  allowed_entity_ids: EntityId[];
  excluded_entity_ids: EntityId[];
} = {
  scope: "same_as_disclosure",
  allowed_entity_ids: [],
  excluded_entity_ids: [],
};

export const creatorDirectiveIdSchema = z
  .string()
  .refine((value) => creatorDirectiveIdHelpers.is(value), {
    message: "Invalid creator directive id",
  })
  .transform((value) => value as CreatorDirectiveId);

export const creatorDirectiveSessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid creator directive session id",
  })
  .transform((value) => parseSessionId(value));

export const creatorDirectiveEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid creator directive entity id",
  })
  .transform((value) => value as EntityId);

export const creatorDirectiveStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid creator directive stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const creatorDirectiveStatusSchema = z.enum(CREATOR_DIRECTIVE_STATUSES);
export const creatorDirectiveKindSchema = z.enum(CREATOR_DIRECTIVE_KINDS);
export const creatorDirectiveSubjectKindSchema = z.enum(CREATOR_DIRECTIVE_SUBJECT_KINDS);
export const creatorDirectiveSemanticSlotSchema = z.enum(CREATOR_DIRECTIVE_SEMANTIC_SLOTS);
export const creatorDirectiveContentScopeSchema = z.enum(CREATOR_DIRECTIVE_CONTENT_SCOPES);
export const creatorDirectiveActivationScopeSchema = z.enum(CREATOR_DIRECTIVE_ACTIVATION_SCOPES);
export const creatorDirectiveMentionPolicySchema = z.enum(CREATOR_DIRECTIVE_MENTION_POLICIES);
export const creatorDirectiveDeniedAudienceBehaviorSchema = z.enum(
  CREATOR_DIRECTIVE_DENIED_AUDIENCE_BEHAVIORS,
);
export const creatorDirectiveRenderModeSchema = z.enum(CREATOR_DIRECTIVE_RENDER_MODES);
export const creatorDirectiveRenderReasonSchema = z.enum(CREATOR_DIRECTIVE_RENDER_REASONS);
export const creatorDirectiveActivationReasonSchema = z.enum(CREATOR_DIRECTIVE_ACTIVATION_REASONS);

export const creatorDirectiveTopicTagSchema = z
  .string()
  .transform((value) => value.normalize("NFKC").trim().toLowerCase())
  .pipe(z.string().min(1).max(64));

const operationalDirectiveSchema = z.string().trim().min(1);

function creatorDirectiveKindRequiresOperationalDirective(
  kind: z.infer<typeof creatorDirectiveKindSchema>,
): boolean {
  return kind === "response_policy" || kind === "routing_instruction";
}

type ScopeEntityFieldIssueContext = {
  addIssue(issue: { code: "custom"; path: string[]; message: string }): void;
};

type SharedScopeEntityFieldScope = Exclude<
  z.infer<typeof creatorDirectiveContentScopeSchema>,
  "operator_only"
>;

function addScopeEntityFieldIssues(
  ctx: ScopeEntityFieldIssueContext,
  scope: SharedScopeEntityFieldScope,
  allowedEntityIds: readonly EntityId[],
  excludedEntityIds: readonly EntityId[],
): void {
  if (scope === "public") {
    if (allowedEntityIds.length > 0) {
      ctx.addIssue({
        code: "custom",
        path: ["allowed_entity_ids"],
        message: "public scope requires empty allowed_entity_ids",
      });
    }

    if (excludedEntityIds.length > 0) {
      ctx.addIssue({
        code: "custom",
        path: ["excluded_entity_ids"],
        message: "public scope requires empty excluded_entity_ids",
      });
    }
  }

  if (scope === "allow_list" && allowedEntityIds.length === 0) {
    ctx.addIssue({
      code: "custom",
      path: ["allowed_entity_ids"],
      message: "allow_list requires at least one allowed entity",
    });
  }

  if (scope === "allow_list") {
    const allowedEntityIdSet = new Set(allowedEntityIds);
    if (excludedEntityIds.some((id) => allowedEntityIdSet.has(id))) {
      ctx.addIssue({
        code: "custom",
        path: ["excluded_entity_ids"],
        message: "allow_list allowed and excluded entity ids must not overlap",
      });
    }
  }

  if (scope === "all_except" && allowedEntityIds.length > 0) {
    ctx.addIssue({
      code: "custom",
      path: ["allowed_entity_ids"],
      message: "all_except requires empty allowed_entity_ids",
    });
  }

  if (scope === "all_except" && excludedEntityIds.length === 0) {
    ctx.addIssue({
      code: "custom",
      path: ["excluded_entity_ids"],
      message: "all_except requires at least one excluded entity",
    });
  }

  if (scope === "subject_only") {
    if (allowedEntityIds.length > 0) {
      ctx.addIssue({
        code: "custom",
        path: ["allowed_entity_ids"],
        message: "subject_only requires empty allowed_entity_ids",
      });
    }

    if (excludedEntityIds.length > 0) {
      ctx.addIssue({
        code: "custom",
        path: ["excluded_entity_ids"],
        message: "subject_only requires empty excluded_entity_ids",
      });
    }
  }
}

export const disclosurePolicySchema = z
  .object({
    content_scope: creatorDirectiveContentScopeSchema,
    allowed_entity_ids: z.array(creatorDirectiveEntityIdSchema),
    excluded_entity_ids: z.array(creatorDirectiveEntityIdSchema),
    subject_may_know: z.boolean().nullable(),
    mention_policy: creatorDirectiveMentionPolicySchema,
    denied_audience_behavior: creatorDirectiveDeniedAudienceBehaviorSchema,
    boundary_prompt: z.string().trim().min(1).nullable(),
    topic_tags: z.array(creatorDirectiveTopicTagSchema).max(32),
  })
  .superRefine((value, ctx) => {
    if (value.content_scope !== "operator_only") {
      addScopeEntityFieldIssues(
        ctx,
        value.content_scope,
        value.allowed_entity_ids,
        value.excluded_entity_ids,
      );
    }

    if (value.content_scope === "operator_only" && value.allowed_entity_ids.length > 0) {
      ctx.addIssue({
        code: "custom",
        path: ["allowed_entity_ids"],
        message: "operator_only requires empty allowed_entity_ids",
      });
    }

    if (value.content_scope === "subject_only") {
      if (value.subject_may_know === false) {
        ctx.addIssue({
          code: "custom",
          path: ["subject_may_know"],
          message: "subject_only requires subject_may_know to be true or null",
        });
      }
    }

    if (
      value.denied_audience_behavior === "render_boundary_when_relevant" &&
      value.boundary_prompt === null
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["boundary_prompt"],
        message: "render_boundary_when_relevant requires boundary_prompt",
      });
    }
  });

export const activationPolicySchema = z
  .object({
    scope: creatorDirectiveActivationScopeSchema,
    allowed_entity_ids: z.array(creatorDirectiveEntityIdSchema),
    excluded_entity_ids: z.array(creatorDirectiveEntityIdSchema),
  })
  .superRefine((value, ctx) => {
    if (value.scope === "same_as_disclosure") {
      if (value.allowed_entity_ids.length > 0) {
        ctx.addIssue({
          code: "custom",
          path: ["allowed_entity_ids"],
          message: "same_as_disclosure scope requires empty allowed_entity_ids",
        });
      }

      if (value.excluded_entity_ids.length > 0) {
        ctx.addIssue({
          code: "custom",
          path: ["excluded_entity_ids"],
          message: "same_as_disclosure scope requires empty excluded_entity_ids",
        });
      }
    }

    if (value.scope === "operator_only") {
      if (value.allowed_entity_ids.length > 0) {
        ctx.addIssue({
          code: "custom",
          path: ["allowed_entity_ids"],
          message: "operator_only requires empty allowed_entity_ids",
        });
      }

      if (value.excluded_entity_ids.length > 0) {
        ctx.addIssue({
          code: "custom",
          path: ["excluded_entity_ids"],
          message: "operator_only requires empty excluded_entity_ids",
        });
      }
    }

    if (value.scope !== "same_as_disclosure" && value.scope !== "operator_only") {
      addScopeEntityFieldIssues(
        ctx,
        value.scope,
        value.allowed_entity_ids,
        value.excluded_entity_ids,
      );
    }
  });

function subjectMayKnowPolicyIsValid(input: {
  subjectEntityId: EntityId | null | undefined;
  policy: DisclosurePolicy;
}): boolean {
  if (
    input.policy.subject_may_know !== false ||
    input.subjectEntityId === null ||
    input.subjectEntityId === undefined
  ) {
    return true;
  }

  if (input.policy.content_scope === "operator_only") {
    return true;
  }

  return input.policy.excluded_entity_ids.some((id) => id === input.subjectEntityId);
}

export const creatorDirectiveSchema = z
  .object({
    id: creatorDirectiveIdSchema,
    record_version: z.number().int().positive(),
    status: creatorDirectiveStatusSchema,
    kind: creatorDirectiveKindSchema,
    created_by_entity_id: creatorDirectiveEntityIdSchema,
    source_session_id: creatorDirectiveSessionIdSchema,
    authorization_stream_entry_ids: z.array(creatorDirectiveStreamEntryIdSchema).min(1),
    content_source_stream_entry_ids: z.array(creatorDirectiveStreamEntryIdSchema).min(1),
    subject_kind: creatorDirectiveSubjectKindSchema,
    subject_entity_id: creatorDirectiveEntityIdSchema.nullable(),
    semantic_slot: creatorDirectiveSemanticSlotSchema.nullable(),
    canonical_fact: z.string().trim().min(1).nullable(),
    operational_directive: operationalDirectiveSchema.nullable(),
    disclosure_policy: disclosurePolicySchema,
    activation_policy: activationPolicySchema,
    priority: z.number().int(),
    superseded_by: creatorDirectiveIdSchema.nullable(),
    revoked_reason: z.string().trim().min(1).nullable(),
    created_at: z.number().int().finite(),
    updated_at: z.number().int().finite(),
  })
  .superRefine((value, ctx) => {
    if (value.subject_kind === "entity" && value.subject_entity_id === null) {
      ctx.addIssue({
        code: "custom",
        path: ["subject_entity_id"],
        message: "entity subject requires subject_entity_id",
      });
    }

    if (
      value.disclosure_policy.content_scope === "subject_only" &&
      value.subject_entity_id === null
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["subject_entity_id"],
        message: "subject_only requires subject_entity_id",
      });
    }

    if (value.activation_policy.scope === "subject_only" && value.subject_entity_id === null) {
      ctx.addIssue({
        code: "custom",
        path: ["subject_entity_id"],
        message: "subject_only activation requires subject_entity_id",
      });
    }

    if (
      !subjectMayKnowPolicyIsValid({
        subjectEntityId: value.subject_entity_id,
        policy: value.disclosure_policy,
      })
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["disclosure_policy", "subject_may_know"],
        message: "subject_may_know=false requires subject exclusion or operator_only scope",
      });
    }

    if (value.semantic_slot !== null && value.canonical_fact === null) {
      ctx.addIssue({
        code: "custom",
        path: ["canonical_fact"],
        message: "slotted creator directive requires canonical_fact",
      });
    }

    if (
      creatorDirectiveKindRequiresOperationalDirective(value.kind) &&
      value.operational_directive === null
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["operational_directive"],
        message: "behavioral creator directive requires operational_directive",
      });
    }
  });

export const creatorDirectiveQueueInputSchema = z
  .object({
    id: creatorDirectiveIdSchema.optional(),
    kind: creatorDirectiveKindSchema,
    createdByEntityId: creatorDirectiveEntityIdSchema,
    sourceSessionId: creatorDirectiveSessionIdSchema,
    authorizationStreamEntryIds: z.array(creatorDirectiveStreamEntryIdSchema).min(1),
    contentSourceStreamEntryIds: z.array(creatorDirectiveStreamEntryIdSchema).min(1),
    subjectKind: creatorDirectiveSubjectKindSchema,
    subjectEntityId: creatorDirectiveEntityIdSchema.nullable().optional(),
    semanticSlot: creatorDirectiveSemanticSlotSchema.nullable().optional(),
    semanticValue: z.string().trim().min(1).nullable().optional(),
    canonicalFact: z.string().trim().min(1).nullable().optional(),
    operationalDirective: operationalDirectiveSchema.nullable().optional(),
    disclosurePolicy: disclosurePolicySchema,
    activationPolicy: activationPolicySchema.default(DEFAULT_CREATOR_DIRECTIVE_ACTIVATION_POLICY),
    priority: z.number().int(),
    createdAt: z.number().int().finite().optional(),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (
      value.subjectKind === "entity" &&
      (value.subjectEntityId === undefined || value.subjectEntityId === null)
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["subjectEntityId"],
        message: "entity subject requires subjectEntityId",
      });
    }

    if (
      value.disclosurePolicy.content_scope === "subject_only" &&
      (value.subjectEntityId === undefined || value.subjectEntityId === null)
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["subjectEntityId"],
        message: "subject_only requires subjectEntityId",
      });
    }

    if (
      value.activationPolicy.scope === "subject_only" &&
      (value.subjectEntityId === undefined || value.subjectEntityId === null)
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["subjectEntityId"],
        message: "subject_only activation requires subjectEntityId",
      });
    }

    if (
      !subjectMayKnowPolicyIsValid({
        subjectEntityId: value.subjectEntityId,
        policy: value.disclosurePolicy,
      })
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["disclosurePolicy", "subject_may_know"],
        message: "subject_may_know=false requires subject exclusion or operator_only scope",
      });
    }

    if (value.semanticSlot !== undefined && value.semanticSlot !== null) {
      if (value.semanticValue === undefined || value.semanticValue === null) {
        ctx.addIssue({
          code: "custom",
          path: ["semanticValue"],
          message: "slotted creator directive requires semanticValue",
        });
      }
    }

    if (
      (value.semanticSlot === undefined || value.semanticSlot === null) &&
      value.semanticValue !== undefined &&
      value.semanticValue !== null
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["semanticValue"],
        message: "semanticValue requires semanticSlot",
      });
    }

    if (
      creatorDirectiveKindRequiresOperationalDirective(value.kind) &&
      (value.operationalDirective === undefined || value.operationalDirective === null)
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["operationalDirective"],
        message: "behavioral creator directive requires operationalDirective",
      });
    }
  });

export const creatorDirectiveListFilterSchema = z
  .object({
    status: creatorDirectiveStatusSchema.optional(),
    kind: creatorDirectiveKindSchema.optional(),
    createdByEntityId: creatorDirectiveEntityIdSchema.optional(),
    sourceSessionId: creatorDirectiveSessionIdSchema.optional(),
    subjectKind: creatorDirectiveSubjectKindSchema.optional(),
    subjectEntityId: creatorDirectiveEntityIdSchema.nullable().optional(),
    semanticSlot: creatorDirectiveSemanticSlotSchema.nullable().optional(),
    topicTag: creatorDirectiveTopicTagSchema.optional(),
  })
  .strict();

export const creatorDirectiveApplicableOptionsSchema = z
  .object({
    currentAudienceEntityId: creatorDirectiveEntityIdSchema.nullable(),
    currentSenderBorgRole: borgRoleSchema.nullable().optional(),
    trustedTenantOperator: z.boolean().optional(),
    isPrivateSelfCognition: z.boolean().optional(),
    participantEntityIds: z.array(creatorDirectiveEntityIdSchema).optional(),
    allowListAudienceEntityIds: z.array(creatorDirectiveEntityIdSchema).optional(),
    sessionRole: sessionAudienceRoleSchema,
  })
  .strict();

export type DisclosurePolicy = z.infer<typeof disclosurePolicySchema>;
export type ActivationPolicy = z.infer<typeof activationPolicySchema>;
export type CreatorDirective = z.infer<typeof creatorDirectiveSchema>;
export type CreatorDirectiveStatus = z.infer<typeof creatorDirectiveStatusSchema>;
export type CreatorDirectiveKind = z.infer<typeof creatorDirectiveKindSchema>;
export type CreatorDirectiveSubjectKind = z.infer<typeof creatorDirectiveSubjectKindSchema>;
export type CreatorDirectiveSemanticSlot = z.infer<typeof creatorDirectiveSemanticSlotSchema>;
export type CreatorDirectiveContentScope = z.infer<typeof creatorDirectiveContentScopeSchema>;
export type CreatorDirectiveActivationScope = z.infer<typeof creatorDirectiveActivationScopeSchema>;
export type CreatorDirectiveMentionPolicy = z.infer<typeof creatorDirectiveMentionPolicySchema>;
export type CreatorDirectiveDeniedAudienceBehavior = z.infer<
  typeof creatorDirectiveDeniedAudienceBehaviorSchema
>;
export type CreatorDirectiveRenderMode = z.infer<typeof creatorDirectiveRenderModeSchema>;
export type CreatorDirectiveRenderReason = z.infer<typeof creatorDirectiveRenderReasonSchema>;
export type CreatorDirectiveActivationReason = z.infer<
  typeof creatorDirectiveActivationReasonSchema
>;
export type CreatorDirectiveQueueInput = z.input<typeof creatorDirectiveQueueInputSchema>;
export type CreatorDirectiveListFilter = z.infer<typeof creatorDirectiveListFilterSchema>;
export type CreatorDirectiveApplicableOptions = z.infer<
  typeof creatorDirectiveApplicableOptionsSchema
>;
export type CreatorDirectiveApplicable = {
  directive: CreatorDirective;
  recipient_entity_ids: readonly EntityId[];
  activation: {
    active: boolean;
    reason: CreatorDirectiveActivationReason;
  };
  disclosure: {
    render_mode: CreatorDirectiveRenderMode;
    reason: CreatorDirectiveRenderReason;
  };
  render_mode: CreatorDirectiveRenderMode;
  reason: CreatorDirectiveRenderReason;
};
export type {
  BorgRole,
  CreatorDirectiveId,
  EntityId,
  SessionAudienceRole,
  SessionId,
  StreamEntryId,
};
