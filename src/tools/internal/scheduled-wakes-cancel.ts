import { z } from "zod";

import { scheduledWakeSchema, type ScheduledWake } from "../../autonomy/index.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import {
  memoryDisclosureLabelMetadataSchema,
  selfPrivateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import {
  parseScheduledWakeId,
  scheduledWakeIdHelpers,
  type ScheduledWakeId,
} from "../../util/ids.js";
import type { ToolDefinition } from "../dispatcher.js";

const scheduledWakesCancelInputSchema = z.object({
  scheduled_wake_id: z.string().refine((value) => scheduledWakeIdHelpers.is(value), {
    message: "Invalid scheduled wake id",
  }),
});

const scheduledWakesCancelOutputSchema = z.object({
  scheduledWake: scheduledWakeSchema
    .extend({
      disclosure: z.string().min(1),
      disclosure_label: memoryDisclosureLabelMetadataSchema,
    })
    .nullable(),
});

export type ScheduledWakesCancelToolOptions = {
  cancelScheduledWake: (id: ScheduledWakeId) => ScheduledWake | null;
};

export function createScheduledWakesCancelTool(
  options: ScheduledWakesCancelToolOptions,
): ToolDefinition<
  z.infer<typeof scheduledWakesCancelInputSchema>,
  z.infer<typeof scheduledWakesCancelOutputSchema>
> {
  return {
    name: "tool.scheduledWakes.cancel",
    description:
      "Cancel a pending scheduled self-wake by id (get the id from tool.scheduledWakes.list). Returns the cancelled wake, or null if there was no pending wake with that id.",
    menuSummary: "Cancel a pending scheduled self-wake.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: scheduledWakesCancelInputSchema,
    outputSchema: scheduledWakesCancelOutputSchema,
    async invoke(input) {
      const scheduledWake = options.cancelScheduledWake(parseScheduledWakeId(input.scheduled_wake_id));

      return {
        scheduledWake:
          scheduledWake === null
            ? null
            : {
                ...scheduledWake,
                ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
              },
      };
    },
  };
}
