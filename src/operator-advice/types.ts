import { z } from "zod";

import {
  entityIdHelpers,
  isSessionId,
  operatorAdviceIdHelpers,
  parseSessionId,
  type EntityId,
  type OperatorAdviceId,
  type SessionId,
  type StreamEntryId,
} from "../util/ids.js";

export const MAX_ADVICE_TEXT_LENGTH = 4_000;

export const operatorAdviceIdSchema = z
  .string()
  .refine((value) => operatorAdviceIdHelpers.is(value), {
    message: "Invalid operator advice id",
  })
  .transform((value) => value as OperatorAdviceId);

export const operatorAdviceSessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid operator advice session id",
  })
  .transform((value) => parseSessionId(value));

export const operatorAdviceEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid operator advice audience entity id",
  })
  .transform((value) => value as EntityId);

export const operatorAdviceRecordSchema = z.object({
  id: operatorAdviceIdSchema,
  session_id: operatorAdviceSessionIdSchema.nullable(),
  audience_entity_id: operatorAdviceEntityIdSchema.nullable(),
  text: z.string().min(1).max(MAX_ADVICE_TEXT_LENGTH),
  created_at: z.number().int().finite(),
  expires_at: z.number().int().finite().nullable(),
  consumed_at: z.number().int().finite().nullable(),
  consumed_by_turn_id: z.string().min(1).nullable(),
  canceled_at: z.number().int().finite().nullable(),
});

export const operatorAdviceQueueInputSchema = z
  .object({
    text: z.string().trim().min(1).max(MAX_ADVICE_TEXT_LENGTH),
    session_id: operatorAdviceSessionIdSchema.nullable().optional(),
    audience_entity_id: operatorAdviceEntityIdSchema.nullable().optional(),
    expires_at: z.number().int().finite().nullable().optional(),
  })
  .strict()
  .refine(
    (input) =>
      (input.session_id ?? null) !== null || (input.audience_entity_id ?? null) !== null,
    {
      message: "Operator advice requires session_id or audience_entity_id",
      path: ["session_id"],
    },
  );

export const operatorAdviceListFilterSchema = z
  .object({
    pendingOnly: z.boolean().optional(),
    session_id: operatorAdviceSessionIdSchema.nullable().optional(),
    audience_entity_id: operatorAdviceEntityIdSchema.nullable().optional(),
    limit: z.number().int().positive().max(1_000).optional(),
  })
  .strict();

export const operatorAdviceConsumePendingScopeSchema = z
  .object({
    session_id: operatorAdviceSessionIdSchema.nullable().optional(),
    audience_entity_id: operatorAdviceEntityIdSchema.nullable().optional(),
  })
  .strict();

export const operatorAdviceMarkConsumedInputSchema = z
  .object({
    turn_id: z.string().min(1),
    now: z.number().int().finite().optional(),
  })
  .strict();

export type OperatorAdviceRecord = z.infer<typeof operatorAdviceRecordSchema>;
export type OperatorAdviceQueueInput = z.input<typeof operatorAdviceQueueInputSchema>;
export type OperatorAdviceListFilter = z.input<typeof operatorAdviceListFilterSchema>;
export type OperatorAdviceStatus = "pending" | "consumed" | "canceled" | "expired";
export type OperatorAdviceConsumePendingScope = z.input<
  typeof operatorAdviceConsumePendingScopeSchema
>;
export type OperatorAdviceMarkConsumedInput = z.input<
  typeof operatorAdviceMarkConsumedInputSchema
>;

export type OperatorAdviceDelivery = {
  records: OperatorAdviceRecord[];
  renderedText: string | null;
  auditEntryId: StreamEntryId | null;
};

export type OperatorAdvicePromptDelivery = {
  text: string | null;
  ids: readonly OperatorAdviceId[];
};

export type OperatorAdviceConsumerFacade = {
  consumePending(
    scope: OperatorAdviceConsumePendingScope,
    options: OperatorAdviceMarkConsumedInput,
  ): Promise<OperatorAdviceDelivery>;
};

export function operatorAdviceStatus(
  record: OperatorAdviceRecord,
  now: number,
): OperatorAdviceStatus {
  if (record.consumed_at !== null) {
    return "consumed";
  }

  if (record.canceled_at !== null) {
    return "canceled";
  }

  if (record.expires_at !== null && record.expires_at <= now) {
    return "expired";
  }

  return "pending";
}

export type { OperatorAdviceId };
