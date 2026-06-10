import { z } from "zod";

import {
  memoryDisclosurePayloadFields,
  selfPrivateMemoryDisclosureLabel,
} from "../../memory/common/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import {
  trainOfThoughtJournalEntrySchema,
  type TrainOfThoughtJournalEntry,
} from "../../memory/train-of-thought/index.js";
import type { EntityId } from "../../util/ids.js";
import type { ToolDefinition } from "../dispatcher.js";

const journalAppendInputSchema = z
  .object({
    text: z.string().min(1),
  })
  .strict();

const journalAppendOutputSchema = z.object({
  journalEntry: trainOfThoughtJournalEntrySchema.extend({
    disclosure: z.string().min(1),
    disclosure_label: memoryDisclosureLabelMetadataSchema,
  }),
});

export type JournalAppendToolOptions = {
  resolveSelfEntityId: () => EntityId;
  appendJournalEntry: (input: {
    text: string;
    selfEntityId: EntityId;
    sourceTurnId?: string | null;
  }) => TrainOfThoughtJournalEntry;
};

export function createJournalAppendTool(
  options: JournalAppendToolOptions,
): ToolDefinition<z.infer<typeof journalAppendInputSchema>, z.infer<typeof journalAppendOutputSchema>> {
  return {
    name: "tool.journal.append",
    description:
      "Append a self-private journal entry during an autonomous reflection turn. I use this for private interior notes I want retained without ending the turn.",
    menuSummary: "Append a self-private journal entry without ending the turn.",
    allowedOrigins: ["autonomous"],
    writeScope: "write",
    inputSchema: journalAppendInputSchema,
    outputSchema: journalAppendOutputSchema,
    async invoke(input, context) {
      const journalEntry = options.appendJournalEntry({
        text: input.text,
        selfEntityId: options.resolveSelfEntityId(),
        sourceTurnId: context.turnId ?? null,
      });

      return {
        journalEntry: {
          ...journalEntry,
          ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
        },
      };
    },
  };
}
