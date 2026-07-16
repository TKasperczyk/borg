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
} from "../../memory/common/disclosure-serializers.js";
import type { MemoryDisclosureLabel } from "../../memory/common/disclosure-label.js";
import { mapWithDisclosureConcurrency } from "../../retrieval/index.js";
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
  disclosureLabelsForEvents?: (
    events: readonly IdentityEvent[],
    context: ToolInvocationContext,
  ) =>
    | ReadonlyMap<IdentityEvent["id"], MemoryDisclosureLabel>
    | Promise<ReadonlyMap<IdentityEvent["id"], MemoryDisclosureLabel>>;
};

export function createIdentityEventsListForCognitionTool(
  options: IdentityEventsListForCognitionToolOptions,
): ToolDefinition<
  z.infer<typeof identityEventsListInputSchema>,
  z.infer<typeof identityEventsListForCognitionOutputSchema>
> {
  return {
    name: "tool.identityEvents.listForCognition",
    description: "List recent identity events from the being's global memory with disclosure labels.",
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
      const disclosureLabels = await options.disclosureLabelsForEvents?.(events, context);

      return {
        events: await mapWithDisclosureConcurrency(events, async (event) => {
          const disclosureLabel =
            disclosureLabels?.get(event.id) ??
            (await (options.disclosureLabelForEvent?.(event, context) ??
              identityEventMemoryDisclosureLabel(event)));

          return {
            ...event,
            ...memoryDisclosurePayloadFields(disclosureLabel),
          };
        }),
      };
    },
  };
}
