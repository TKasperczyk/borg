import {
  frameAnomalyAnnotationSchema,
  frameAnomalyAnnotationFields,
} from "../../stream/frame-anomaly.js";
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
const recordedToolCallSchema = z.object({
  name: z.string().nullable(),
  input: z.unknown(),
  ok: z.boolean().nullable(),
});
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
      frame_anomalies: z.array(frameAnomalyAnnotationSchema).optional(),
      tool_calls: z.array(recordedToolCallSchema).optional(),
    })
    .nullable(),
});

// A message row can carry the tool-call records of the turn that wrote it. Keep one item per
// record with its name, input and ok flag, and drop the output, which can be far larger than the
// message; a read's content can be fetched again from its input. A field not in the recorded
// shape becomes null instead of being guessed.
function recordedToolCalls(calls: readonly unknown[]): z.infer<typeof recordedToolCallSchema>[] {
  return calls.map((call) => {
    const record: Record<string, unknown> =
      typeof call === "object" && call !== null ? (call as Record<string, unknown>) : {};

    return {
      name: typeof record.name === "string" ? record.name : null,
      input: record.input ?? null,
      ok: typeof record.ok === "boolean" ? record.ok : null,
    };
  });
}

export function createStreamGetTool(options: {
  readEntry: (id: StreamEntryId) => Promise<{
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
    recordIdInputs: [
      {
        path: "stream_entry_id",
        kinds: ["stream_entry"],
      },
    ],
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
          ...frameAnomalyAnnotationFields(entry.frame_anomalies),
          active: result.active,
          turn_status: entry.turn_status ?? null,
          audience: entry.audience ?? null,
          sender_entity_id: entry.sender_entity_id ?? null,
          persistence_class: entry.persistence_class ?? null,
          ...(entry.tool_calls === undefined
            ? {}
            : { tool_calls: recordedToolCalls(entry.tool_calls) }),
          ...memoryDisclosurePayloadFields(result.disclosureLabel),
        },
      };
    },
  };
}
