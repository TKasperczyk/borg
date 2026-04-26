import { z } from "zod";

import {
  identityEventSchema,
  identityRecordTypeSchema,
  type IdentityEvent,
  type IdentityRecordType,
} from "../../memory/identity/index.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const identityEventsListInputSchema = z.object({
  recordType: identityRecordTypeSchema.optional(),
  recordId: z.string().min(1).optional(),
  limit: z.number().int().positive().max(25).optional(),
});

const identityEventsListOutputSchema = z.object({
  events: z.array(identityEventSchema),
});

export type IdentityEventsListToolOptions = {
  listEvents: (
    options: {
      recordType?: IdentityRecordType;
      recordId?: string;
      limit?: number;
    },
    context: ToolInvocationContext,
  ) => IdentityEvent[] | Promise<IdentityEvent[]>;
};

export function createIdentityEventsListTool(
  options: IdentityEventsListToolOptions,
): ToolDefinition<
  z.infer<typeof identityEventsListInputSchema>,
  z.infer<typeof identityEventsListOutputSchema>
> {
  return {
    name: "tool.identityEvents.list",
    description:
      "List recent identity events visible to the current audience. Event types without audience metadata are treated as global.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: identityEventsListInputSchema,
    outputSchema: identityEventsListOutputSchema,
    async invoke(input, context) {
      return {
        events: await options.listEvents(
          {
            recordType: input.recordType,
            recordId: input.recordId,
            limit: input.limit ?? 10,
          },
          context,
        ),
      };
    },
  };
}
