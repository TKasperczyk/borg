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
      "Append a self-private journal entry during an autonomous reflection turn. I use this for private interior notes I want retained without ending the turn. An entry is immutable once written: no tool amends or deletes one, so I cannot go back and mark an earlier entry as mistaken the way I can resolve a question, cancel a wake, or retire a goal. A correction is a new entry naming the one it corrects, and the earlier text stays exactly as written. Nothing on either entry links them: the browse that reads them back takes an origin-time range and no text or id query, so a correction is reachable only from a range whose end is at or after the correction's own time, and reading back to the corrected entry's own window can never contain it. That browse returns newest first and pages backwards, so where a range does span both, the correction arrives before the entry it corrects.",
    menuSummary:
      "Append a self-private journal entry without ending the turn; an entry is immutable once written, so a correction is a new entry naming the one it corrects and never a change to it.",
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
