import { z } from "zod";
import {
  memoryDisclosureLabelMetadataSchema,
  type MemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import type { StreamEntry } from "../../stream/index.js";
import { streamEntryIdSchema } from "../../util/id-schemas.js";
import type { StreamEntryId } from "../../util/ids.js";
import { stringifyPromptContent } from "../../util/token-estimate.js";
import type { ToolDefinition } from "../dispatcher.js";

const inputSchema = z.object({ stream_entry_id: streamEntryIdSchema }).strict();
const outputSchema = z.object({
  entry: z
    .object({
      id: z.string(),
      session_id: z.string(),
      kind: z.string(),
      timestamp: z.number(),
      text: z.string(),
      active: z.boolean(),
      turn_status: z.string().nullable(),
      audience: z.string().nullable(),
      sender_entity_id: z.string().nullable(),
      persistence_class: z.string().nullable(),
      disclosure: z.string(),
      disclosure_label: memoryDisclosureLabelMetadataSchema,
    })
    .nullable(),
});

export function createStreamGetTool(options: {
  readEntry: (
    id: StreamEntryId,
  ) => Promise<{
    entry: StreamEntry;
    active: boolean;
    disclosureLabel: MemoryDisclosureLabel;
  } | null>;
}): ToolDefinition<z.infer<typeof inputSchema>, z.infer<typeof outputSchema>> {
  return {
    name: "tool.stream.get",
    description:
      "Read the full text of one stream record by id; use when an excerpt was cut. Includes origin, lifecycle, and disclosure labels.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema,
    outputSchema,
    async invoke(input) {
      const result = await options.readEntry(input.stream_entry_id);
      if (result === null) return { entry: null };
      const { entry } = result;
      return {
        entry: {
          id: entry.id,
          session_id: entry.session_id,
          kind: entry.kind,
          timestamp: entry.timestamp,
          text: stringifyPromptContent(entry.content),
          active: result.active,
          turn_status: entry.turn_status ?? null,
          audience: entry.audience ?? null,
          sender_entity_id: entry.sender_entity_id ?? null,
          persistence_class: entry.persistence_class ?? null,
          ...memoryDisclosurePayloadFields(result.disclosureLabel),
        },
      };
    },
  };
}
