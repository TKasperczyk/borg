import { z } from "zod";

import { entityIdHelpers, type EntityId } from "../../util/ids.js";

export const TRAIN_OF_THOUGHT_DISCLOSURE_CLASSES = ["self_private"] as const;

export const trainOfThoughtEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid train of thought self entity id",
  })
  .transform((value) => value as EntityId);

export const trainOfThoughtDisclosureClassSchema = z.enum(TRAIN_OF_THOUGHT_DISCLOSURE_CLASSES);

export const trainOfThoughtSchema = z
  .object({
    self_entity_id: trainOfThoughtEntityIdSchema,
    text: z.string(),
    disclosure_class: trainOfThoughtDisclosureClassSchema,
    created_at: z.number().int().finite(),
    updated_at: z.number().int().finite(),
  })
  .strict();

export type TrainOfThoughtDisclosureClass = z.infer<typeof trainOfThoughtDisclosureClassSchema>;
export type TrainOfThought = z.infer<typeof trainOfThoughtSchema>;
