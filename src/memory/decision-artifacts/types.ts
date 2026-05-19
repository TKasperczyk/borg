import { z } from "zod";

import {
  actionIdHelpers,
  commitmentIdHelpers,
  sharedStateEntryIdHelpers,
  entityIdHelpers,
  goalIdHelpers,
  openQuestionIdHelpers,
  streamEntryIdHelpers,
  type ActionId,
  type CommitmentId,
  type SharedStateEntryId,
  type EntityId,
  type GoalId,
  type OpenQuestionId,
  type StreamEntryId,
} from "../../util/ids.js";

export const SHARED_STATE_ENTRY_KINDS = [
  "locked",
  "live",
  "tentative",
  "invalidated",
  "pending",
] as const;

export const ACTIVE_SHARED_STATE_ENTRY_KINDS = ["locked", "live"] as const;

export const sharedStateEntryIdSchema = z
  .string()
  .refine((value) => sharedStateEntryIdHelpers.is(value), {
    message: "Invalid shared state entry id",
  })
  .transform((value) => value as SharedStateEntryId);

export const sharedStateEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid shared state entity id",
  })
  .transform((value) => value as EntityId);

export const sharedStateStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid shared state stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const sharedStateEntryKindSchema = z.enum(SHARED_STATE_ENTRY_KINDS);

export const sharedStateGoalIdSchema = z
  .string()
  .refine((value) => goalIdHelpers.is(value), {
    message: "Invalid shared state canonicalized goal id",
  })
  .transform((value) => value as GoalId);

export const sharedStateCommitmentIdSchema = z
  .string()
  .refine((value) => commitmentIdHelpers.is(value), {
    message: "Invalid shared state canonicalized commitment id",
  })
  .transform((value) => value as CommitmentId);

export const sharedStateActionIdSchema = z
  .string()
  .refine((value) => actionIdHelpers.is(value), {
    message: "Invalid shared state canonicalized action id",
  })
  .transform((value) => value as ActionId);

export const sharedStateOpenQuestionIdSchema = z
  .string()
  .refine((value) => openQuestionIdHelpers.is(value), {
    message: "Invalid shared state canonicalized open question id",
  })
  .transform((value) => value as OpenQuestionId);

export const sharedStateCanonicalizesSchema = z
  .object({
    goal_ids: z.array(sharedStateGoalIdSchema).default([]),
    commitment_ids: z.array(sharedStateCommitmentIdSchema).default([]),
    action_ids: z.array(sharedStateActionIdSchema).default([]),
    open_question_ids: z.array(sharedStateOpenQuestionIdSchema).default([]),
  })
  .strict();

export const sharedStateEntrySchema = z
  .object({
    id: sharedStateEntryIdSchema,
    audience_entity_id: sharedStateEntityIdSchema,
    kind: sharedStateEntryKindSchema,
    text: z.string().trim().min(1),
    owner_entity_id: sharedStateEntityIdSchema.nullable(),
    provenance_stream_entry_ids: z.array(sharedStateStreamEntryIdSchema).min(1),
    last_updated_stream_entry_ids: z.array(sharedStateStreamEntryIdSchema).min(1),
    created_at: z.number().finite(),
    last_updated_at: z.number().finite(),
    superseded_by_id: sharedStateEntryIdSchema.nullable(),
    rank: z.number().int().nonnegative(),
    canonicalizes: sharedStateCanonicalizesSchema,
  })
  .strict();

export const sharedStateArtifactSchema = z
  .object({
    audience_entity_id: sharedStateEntityIdSchema,
    record_version: z.number().int().positive(),
    created_at: z.number().finite(),
    updated_at: z.number().finite(),
    last_compiled_at: z.number().finite().nullable(),
    last_compiled_stream_entry_id: sharedStateStreamEntryIdSchema.nullable(),
    entries: z.array(sharedStateEntrySchema),
  })
  .strict();

export type SharedStateEntryKind = z.infer<typeof sharedStateEntryKindSchema>;
export type SharedStateCanonicalizes = z.infer<typeof sharedStateCanonicalizesSchema>;
export type SharedStateEntry = z.infer<typeof sharedStateEntrySchema>;
export type SharedStateArtifact = z.infer<typeof sharedStateArtifactSchema>;
export type SharedStateSourceTrustRejectionReason = "quarantined" | "inactive";
export type SharedStateSourceTrustResult =
  | {
      allowed: true;
      reason?: never;
    }
  | {
      allowed: false;
      reason?: SharedStateSourceTrustRejectionReason;
    };
export type SharedStateSourceTrustValidator = (
  streamEntryId: StreamEntryId,
) => SharedStateSourceTrustResult;

export const allowAllSharedStateSourceTrustValidator: SharedStateSourceTrustValidator = () => ({
  allowed: true,
});
