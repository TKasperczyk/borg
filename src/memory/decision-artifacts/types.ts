import { z } from "zod";

import {
  actionIdHelpers,
  commitmentIdHelpers,
  decisionArtifactEntryIdHelpers,
  entityIdHelpers,
  goalIdHelpers,
  openQuestionIdHelpers,
  streamEntryIdHelpers,
  type ActionId,
  type CommitmentId,
  type DecisionArtifactEntryId,
  type EntityId,
  type GoalId,
  type OpenQuestionId,
  type StreamEntryId,
} from "../../util/ids.js";

export const DECISION_ARTIFACT_ENTRY_KINDS = [
  "locked",
  "live",
  "tentative",
  "invalidated",
  "pending",
] as const;

export const ACTIVE_DECISION_ARTIFACT_ENTRY_KINDS = ["locked", "live"] as const;

export const decisionArtifactEntryIdSchema = z
  .string()
  .refine((value) => decisionArtifactEntryIdHelpers.is(value), {
    message: "Invalid decision artifact entry id",
  })
  .transform((value) => value as DecisionArtifactEntryId);

export const decisionArtifactEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid decision artifact entity id",
  })
  .transform((value) => value as EntityId);

export const decisionArtifactStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid decision artifact stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const decisionArtifactEntryKindSchema = z.enum(DECISION_ARTIFACT_ENTRY_KINDS);

export const decisionArtifactGoalIdSchema = z
  .string()
  .refine((value) => goalIdHelpers.is(value), {
    message: "Invalid decision artifact canonicalized goal id",
  })
  .transform((value) => value as GoalId);

export const decisionArtifactCommitmentIdSchema = z
  .string()
  .refine((value) => commitmentIdHelpers.is(value), {
    message: "Invalid decision artifact canonicalized commitment id",
  })
  .transform((value) => value as CommitmentId);

export const decisionArtifactActionIdSchema = z
  .string()
  .refine((value) => actionIdHelpers.is(value), {
    message: "Invalid decision artifact canonicalized action id",
  })
  .transform((value) => value as ActionId);

export const decisionArtifactOpenQuestionIdSchema = z
  .string()
  .refine((value) => openQuestionIdHelpers.is(value), {
    message: "Invalid decision artifact canonicalized open question id",
  })
  .transform((value) => value as OpenQuestionId);

export const decisionArtifactCanonicalizesSchema = z
  .object({
    goal_ids: z.array(decisionArtifactGoalIdSchema).default([]),
    commitment_ids: z.array(decisionArtifactCommitmentIdSchema).default([]),
    action_ids: z.array(decisionArtifactActionIdSchema).default([]),
    open_question_ids: z.array(decisionArtifactOpenQuestionIdSchema).default([]),
  })
  .strict();

export const decisionArtifactEntrySchema = z
  .object({
    id: decisionArtifactEntryIdSchema,
    audience_entity_id: decisionArtifactEntityIdSchema,
    kind: decisionArtifactEntryKindSchema,
    text: z.string().trim().min(1),
    owner_entity_id: decisionArtifactEntityIdSchema.nullable(),
    provenance_stream_entry_ids: z.array(decisionArtifactStreamEntryIdSchema).min(1),
    last_updated_stream_entry_ids: z.array(decisionArtifactStreamEntryIdSchema).min(1),
    created_at: z.number().finite(),
    last_updated_at: z.number().finite(),
    superseded_by_id: decisionArtifactEntryIdSchema.nullable(),
    rank: z.number().int().nonnegative(),
    canonicalizes: decisionArtifactCanonicalizesSchema,
  })
  .strict();

export const decisionArtifactSchema = z
  .object({
    audience_entity_id: decisionArtifactEntityIdSchema,
    record_version: z.number().int().positive(),
    created_at: z.number().finite(),
    updated_at: z.number().finite(),
    last_compiled_at: z.number().finite().nullable(),
    last_compiled_stream_entry_id: decisionArtifactStreamEntryIdSchema.nullable(),
    entries: z.array(decisionArtifactEntrySchema),
  })
  .strict();

export type DecisionArtifactEntryKind = z.infer<typeof decisionArtifactEntryKindSchema>;
export type DecisionArtifactCanonicalizes = z.infer<typeof decisionArtifactCanonicalizesSchema>;
export type DecisionArtifactEntry = z.infer<typeof decisionArtifactEntrySchema>;
export type DecisionArtifact = z.infer<typeof decisionArtifactSchema>;
