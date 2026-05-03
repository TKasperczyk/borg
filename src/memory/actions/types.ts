import { z } from "zod";

import {
  actionIdHelpers,
  entityIdHelpers,
  episodeIdHelpers,
  streamEntryIdHelpers,
  type ActionId,
  type EntityId,
  type EpisodeId,
  type StreamEntryId,
} from "../../util/ids.js";

export const ACTION_STATES = [
  "considering",
  "committed_to_do",
  "scheduled",
  "completed",
  "not_done",
  "unknown",
] as const;

export const actionIdSchema = z
  .string()
  .refine((value) => actionIdHelpers.is(value), {
    message: "Invalid action id",
  })
  .transform((value) => value as ActionId);

export const actionEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid action entity id",
  })
  .transform((value) => value as EntityId);

export const actionActorSchema = z.union([z.enum(["user", "borg"]), actionEntityIdSchema]);

export const actionStateSchema = z.enum(ACTION_STATES);

export const actionEpisodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid action provenance episode id",
  })
  .transform((value) => value as EpisodeId);

export const actionStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid action provenance stream entry id",
  })
  .transform((value) => value as StreamEntryId);

const actionRecordShape = z.object({
  id: actionIdSchema,
  description: z.string().min(1),
  actor: actionActorSchema,
  audience_entity_id: actionEntityIdSchema.nullable(),
  state: actionStateSchema,
  confidence: z.number().min(0).max(1),
  provenance_episode_ids: z.array(actionEpisodeIdSchema),
  provenance_stream_entry_ids: z.array(actionStreamEntryIdSchema),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
  considering_at: z.number().finite().nullable(),
  committed_at: z.number().finite().nullable(),
  scheduled_at: z.number().finite().nullable(),
  completed_at: z.number().finite().nullable(),
  not_done_at: z.number().finite().nullable(),
  unknown_at: z.number().finite().nullable(),
});

export const actionRecordSchema = actionRecordShape
  .refine((value) => value.updated_at >= value.created_at, {
    message: "Action updated_at must be greater than or equal to created_at",
    path: ["updated_at"],
  })
  .refine(
    (value) =>
      value.provenance_episode_ids.length > 0 || value.provenance_stream_entry_ids.length > 0,
    {
      message: "Action record requires episode or stream provenance",
      path: ["provenance_stream_entry_ids"],
    },
  );

export const actionRecordPatchSchema = actionRecordShape
  .omit({
    id: true,
    created_at: true,
  })
  .partial()
  .strict();

export type ActionState = z.infer<typeof actionStateSchema>;
export type ActionActor = z.infer<typeof actionActorSchema>;
export type ActionRecord = z.infer<typeof actionRecordSchema>;
export type ActionRecordPatch = z.infer<typeof actionRecordPatchSchema>;
