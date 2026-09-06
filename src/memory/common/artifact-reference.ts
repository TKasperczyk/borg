import { z } from "zod";

import { executiveStepIdSchema } from "../../executive/types.js";
import { streamEntryIdSchema } from "../../util/id-schemas.js";
import {
  openQuestionIdHelpers,
  scheduledWakeIdHelpers,
  type OpenQuestionId,
  type ScheduledWakeId,
} from "../../util/ids.js";

const openQuestionIdSchema = z
  .string()
  .refine(openQuestionIdHelpers.is)
  .transform((id) => id as OpenQuestionId);

// Record handles, shared with reflection. Existence/state checks belong to the
// writer; the schema does not judge what the artifact's language demonstrates.
export const artifactReferenceSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("journal_entry"), id: z.number().int().positive() }).strict(),
  z.object({ kind: z.literal("created_open_question"), id: openQuestionIdSchema }).strict(),
  z.object({ kind: z.literal("resolved_open_question"), id: openQuestionIdSchema }).strict(),
  z
    .object({
      kind: z.literal("scheduled_wake"),
      id: z
        .string()
        .refine(scheduledWakeIdHelpers.is)
        .transform((id) => id as ScheduledWakeId),
    })
    .strict(),
  z.object({ kind: z.literal("executive_step_outcome"), id: executiveStepIdSchema }).strict(),
  z.object({ kind: z.literal("delivered_outbound_post"), id: streamEntryIdSchema }).strict(),
  z.object({ kind: z.literal("stream_entry"), id: streamEntryIdSchema }).strict(),
]);

export type ArtifactReference = z.infer<typeof artifactReferenceSchema>;

export const deliveredOutboundPostArtifactOutputSchema = z.object({
  outbound: z.object({
    delivery_outcome: z.object({
      state: z.literal("delivered"),
      agent_message_id: streamEntryIdSchema,
    }),
  }),
});
