import { z } from "zod";

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
export const CREATOR_DIRECTIVE_CONTENT_SCOPES = [
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
export const creatorDirectiveContentScopeSchema = z.enum(CREATOR_DIRECTIVE_CONTENT_SCOPES);
export const creatorDirectiveMentionPolicySchema = z.enum(CREATOR_DIRECTIVE_MENTION_POLICIES);
export const creatorDirectiveDeniedAudienceBehaviorSchema = z.enum(
  CREATOR_DIRECTIVE_DENIED_AUDIENCE_BEHAVIORS,
);
export const creatorDirectiveRenderModeSchema = z.enum(CREATOR_DIRECTIVE_RENDER_MODES);

export const creatorDirectiveTopicTagSchema = z
  .string()
  .transform((value) => value.normalize("NFKC").trim().toLowerCase())
  .pipe(z.string().min(1).max(64));

export const disclosurePolicySchema = z.object({
  content_scope: creatorDirectiveContentScopeSchema,
  allowed_entity_ids: z.array(creatorDirectiveEntityIdSchema),
  excluded_entity_ids: z.array(creatorDirectiveEntityIdSchema),
  subject_may_know: z.boolean().nullable(),
  mention_policy: creatorDirectiveMentionPolicySchema,
  denied_audience_behavior: creatorDirectiveDeniedAudienceBehaviorSchema,
  boundary_prompt: z.string().trim().min(1).nullable(),
  topic_tags: z.array(creatorDirectiveTopicTagSchema).max(32),
});

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
    canonical_fact: z.string().trim().min(1).nullable(),
    operational_directive: z.string().trim().min(1),
    disclosure_policy: disclosurePolicySchema,
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
    canonicalFact: z.string().trim().min(1).nullable().optional(),
    operationalDirective: z.string().trim().min(1),
    disclosurePolicy: disclosurePolicySchema,
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
  });

export const creatorDirectiveListFilterSchema = z
  .object({
    status: creatorDirectiveStatusSchema.optional(),
    kind: creatorDirectiveKindSchema.optional(),
    createdByEntityId: creatorDirectiveEntityIdSchema.optional(),
    sourceSessionId: creatorDirectiveSessionIdSchema.optional(),
    subjectKind: creatorDirectiveSubjectKindSchema.optional(),
    subjectEntityId: creatorDirectiveEntityIdSchema.nullable().optional(),
    topicTag: creatorDirectiveTopicTagSchema.optional(),
  })
  .strict();

export const creatorDirectiveApplicableOptionsSchema = z
  .object({
    currentAudienceEntityId: creatorDirectiveEntityIdSchema.nullable(),
    participantEntityIds: z.array(creatorDirectiveEntityIdSchema).optional(),
    perceivedEntityIds: z.array(creatorDirectiveEntityIdSchema).optional(),
    topicTags: z.array(creatorDirectiveTopicTagSchema).optional(),
    sessionRole: sessionAudienceRoleSchema,
  })
  .strict();

export type DisclosurePolicy = z.infer<typeof disclosurePolicySchema>;
export type CreatorDirective = z.infer<typeof creatorDirectiveSchema>;
export type CreatorDirectiveStatus = z.infer<typeof creatorDirectiveStatusSchema>;
export type CreatorDirectiveKind = z.infer<typeof creatorDirectiveKindSchema>;
export type CreatorDirectiveSubjectKind = z.infer<typeof creatorDirectiveSubjectKindSchema>;
export type CreatorDirectiveContentScope = z.infer<typeof creatorDirectiveContentScopeSchema>;
export type CreatorDirectiveMentionPolicy = z.infer<typeof creatorDirectiveMentionPolicySchema>;
export type CreatorDirectiveDeniedAudienceBehavior = z.infer<
  typeof creatorDirectiveDeniedAudienceBehaviorSchema
>;
export type CreatorDirectiveRenderMode = z.infer<typeof creatorDirectiveRenderModeSchema>;
export type CreatorDirectiveQueueInput = z.infer<typeof creatorDirectiveQueueInputSchema>;
export type CreatorDirectiveListFilter = z.infer<typeof creatorDirectiveListFilterSchema>;
export type CreatorDirectiveApplicableOptions = z.infer<
  typeof creatorDirectiveApplicableOptionsSchema
>;
export type CreatorDirectiveApplicable = {
  directive: CreatorDirective;
  render_mode: CreatorDirectiveRenderMode;
};
export type { CreatorDirectiveId, EntityId, SessionAudienceRole, SessionId, StreamEntryId };
