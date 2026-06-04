import { z } from "zod";

import {
  identityEventSchema,
  identityRecordTypeSchema,
  type IdentityEvent,
  type IdentityRecordType,
} from "../../memory/identity/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import {
  identityEventMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "../../cognition/disclosure-labels.js";
import type { MemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const identityEventsListInputSchema = z.object({
  recordType: identityRecordTypeSchema.optional(),
  recordId: z.string().min(1).optional(),
  limit: z.number().int().positive().max(25).optional(),
});

const identityEventForCognitionSchema = identityEventSchema.extend({
  disclosure: z.string().min(1),
  disclosure_label: memoryDisclosureLabelMetadataSchema,
});

const identityEventsListForCognitionOutputSchema = z.object({
  events: z.array(identityEventForCognitionSchema),
});

export type IdentityEventsListForCognitionToolOptions = {
  listEvents: (
    options: {
      recordType?: IdentityRecordType;
      recordId?: string;
      limit?: number;
    },
    context: ToolInvocationContext,
  ) => IdentityEvent[] | Promise<IdentityEvent[]>;
  disclosureLabelForEvent?: (
    event: IdentityEvent,
    context: ToolInvocationContext,
  ) => MemoryDisclosureLabel | Promise<MemoryDisclosureLabel>;
};

export function createIdentityEventsListForCognitionTool(
  options: IdentityEventsListForCognitionToolOptions,
): ToolDefinition<
  z.infer<typeof identityEventsListInputSchema>,
  z.infer<typeof identityEventsListForCognitionOutputSchema>
> {
  return {
    name: "tool.identityEvents.listForCognition",
    description: "List recent identity events from Sol's global memory with disclosure labels.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: identityEventsListInputSchema,
    outputSchema: identityEventsListForCognitionOutputSchema,
    async invoke(input, context) {
      const events = await options.listEvents(
        {
          recordType: input.recordType,
          recordId: input.recordId,
          limit: input.limit ?? 10,
        },
        context,
      );

      return {
        events: await Promise.all(
          events.map(async (event) => ({
            ...event,
            ...memoryDisclosurePayloadFields(
              await (options.disclosureLabelForEvent?.(event, context) ??
                identityEventMemoryDisclosureLabel(event)),
            ),
          })),
        ),
      };
    },
  };
}
