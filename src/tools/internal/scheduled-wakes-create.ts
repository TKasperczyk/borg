import { z } from "zod";

import { scheduledWakeSchema, type ScheduledWake } from "../../autonomy/index.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import {
  memoryDisclosureLabelMetadataSchema,
  selfPrivateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import type { ToolDefinition } from "../dispatcher.js";

// Cap the scheduling horizon at ~10 years: generous for any real use, and it
// keeps fire_at well inside safe-integer / SQLite INTEGER range.
const MAX_DELAY_SECONDS = 10 * 365 * 24 * 60 * 60;

const scheduledWakesCreateInputSchema = z.object({
  delay_seconds: z.number().finite().positive().max(MAX_DELAY_SECONDS),
  note: z.string().min(1),
});

const scheduledWakesCreateOutputSchema = z.object({
  scheduledWake: scheduledWakeSchema.extend({
    disclosure: z.string().min(1),
    disclosure_label: memoryDisclosureLabelMetadataSchema,
  }),
});

export type ScheduledWakesCreateToolOptions = {
  scheduleWake: (input: {
    delaySeconds: number;
    note: string;
    originSessionId: string | null;
  }) => ScheduledWake;
};

export function createScheduledWakesCreateTool(
  options: ScheduledWakesCreateToolOptions,
): ToolDefinition<
  z.infer<typeof scheduledWakesCreateInputSchema>,
  z.infer<typeof scheduledWakesCreateOutputSchema>
> {
  return {
    name: "tool.scheduledWakes.create",
    description:
      "Schedule a one-time wake for my future self, firing once after delay_seconds from now. I use note to tell my future self why I am waking and what to revisit or do. The wake runs as a private self-turn, not a message to anyone.",
    menuSummary: "Schedule a one-time wake for my future self.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: scheduledWakesCreateInputSchema,
    outputSchema: scheduledWakesCreateOutputSchema,
    async invoke(input, context) {
      const scheduledWake = options.scheduleWake({
        delaySeconds: input.delay_seconds,
        note: input.note,
        originSessionId: context.sessionId,
      });

      return {
        scheduledWake: {
          ...scheduledWake,
          ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
        },
      };
    },
  };
}
