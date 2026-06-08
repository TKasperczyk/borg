import { z } from "zod";

import {
  SCHEDULED_WAKE_STATUSES,
  scheduledWakeSchema,
  type ScheduledWake,
  type ScheduledWakeStatus,
} from "../../autonomy/index.js";
import { memoryDisclosurePayloadFields } from "../../cognition/disclosure-labels.js";
import {
  memoryDisclosureLabelMetadataSchema,
  selfPrivateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import type { ToolDefinition } from "../dispatcher.js";

const DEFAULT_LIST_LIMIT = 20;

const scheduledWakesListInputSchema = z.object({
  status: z.enum(SCHEDULED_WAKE_STATUSES).optional(),
  limit: z.number().int().positive().max(100).optional(),
});

const scheduledWakesListOutputSchema = z.object({
  scheduledWakes: z.array(
    scheduledWakeSchema.extend({
      disclosure: z.string().min(1),
      disclosure_label: memoryDisclosureLabelMetadataSchema,
    }),
  ),
});

export type ScheduledWakesListToolOptions = {
  listScheduledWakes: (input: { status?: ScheduledWakeStatus; limit: number }) => ScheduledWake[];
};

export function createScheduledWakesListTool(
  options: ScheduledWakesListToolOptions,
): ToolDefinition<
  z.infer<typeof scheduledWakesListInputSchema>,
  z.infer<typeof scheduledWakesListOutputSchema>
> {
  return {
    name: "tool.scheduledWakes.list",
    description:
      "I list my scheduled self-wakes. Defaults to pending (not yet fired) wakes; I pass status to inspect fired or cancelled ones. I use this to review what I have already scheduled before adding or cancelling one.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: scheduledWakesListInputSchema,
    outputSchema: scheduledWakesListOutputSchema,
    async invoke(input) {
      return {
        scheduledWakes: options
          .listScheduledWakes({
            status: input.status ?? "pending",
            limit: input.limit ?? DEFAULT_LIST_LIMIT,
          })
          .map((wake) => ({
            ...wake,
            ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
          })),
      };
    },
  };
}
