import { z } from "zod";

import {
  IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS,
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
import { headTailTextExcerpt } from "../../util/text-excerpt.js";
import type { ToolDefinition, ToolInvocationContext } from "../dispatcher.js";

const identityEventsListInputSchema = z.object({
  recordType: identityRecordTypeSchema.optional(),
  recordId: z.string().min(1).optional(),
  limit: z.number().int().positive().max(25).optional(),
});

const identityEventChangeExcerptSchema = z
  .object({
    format: z.enum(["top_level_fields_old_to_new", "whole_value_old_to_new"]),
    changed_fields: z.array(z.string()).nullable(),
    excerpt_head: z.string().max(IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS),
    excerpt_tail: z.string().max(IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS).nullable(),
    excerpt_exact: z.boolean(),
    excerpt_chars: z
      .number()
      .int()
      .nonnegative()
      .max(IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS),
    source_chars: z.number().int().nonnegative(),
  })
  .superRefine((value, context) => {
    if (
      value.excerpt_head.length + (value.excerpt_tail?.length ?? 0) >
      IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS
    ) {
      context.addIssue({
        code: "custom",
        message: "Identity event change excerpt exceeds its character budget",
        path: ["excerpt_tail"],
      });
    }
  });

const identityEventForCognitionSchema = identityEventSchema
  .omit({ old_value: true, new_value: true })
  .extend({
    change: identityEventChangeExcerptSchema,
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function jsonText(value: unknown): string {
  return JSON.stringify(value) ?? "null";
}

function changedTopLevelFields(
  oldValue: Record<string, unknown>,
  newValue: Record<string, unknown>,
): string[] {
  const keys = [...new Set([...Object.keys(oldValue), ...Object.keys(newValue)])].sort(
    (left, right) => left.localeCompare(right),
  );

  return keys.filter((key) => {
    const oldHasKey = Object.prototype.hasOwnProperty.call(oldValue, key);
    const newHasKey = Object.prototype.hasOwnProperty.call(newValue, key);

    return oldHasKey !== newHasKey || jsonText(oldValue[key]) !== jsonText(newValue[key]);
  });
}

type ChangedValue = { present: boolean; value?: unknown };
type FieldChange = { old: ChangedValue; new: ChangedValue };

function identityEventChangeSource(event: IdentityEvent):
  | {
      format: "top_level_fields_old_to_new";
      changedFields: string[];
      value: { changed_fields: string[]; field_changes: Record<string, FieldChange> };
    }
  | {
      format: "whole_value_old_to_new";
      changedFields: null;
      value: FieldChange;
    } {
  const oldValue = event.old_value;
  const newValue = event.new_value;

  if (!isRecord(oldValue) || !isRecord(newValue)) {
    return {
      format: "whole_value_old_to_new",
      changedFields: null,
      value: {
        old: { present: oldValue !== null, value: oldValue },
        new: { present: newValue !== null, value: newValue },
      },
    };
  }

  const changedFields = changedTopLevelFields(oldValue, newValue);
  const fieldChanges = Object.fromEntries(
    changedFields.map((field) => {
      const oldPresent = Object.prototype.hasOwnProperty.call(oldValue, field);
      const newPresent = Object.prototype.hasOwnProperty.call(newValue, field);

      return [
        field,
        {
          old: {
            present: oldPresent,
            ...(oldPresent ? { value: oldValue[field] } : {}),
          },
          new: {
            present: newPresent,
            ...(newPresent ? { value: newValue[field] } : {}),
          },
        },
      ];
    }),
  );

  return {
    format: "top_level_fields_old_to_new",
    changedFields,
    value: {
      changed_fields: changedFields,
      field_changes: fieldChanges,
    },
  };
}

function excerptChangedValue(change: ChangedValue, maxChars: number): unknown {
  if (!change.present) {
    return change;
  }
  const excerpt = headTailTextExcerpt(jsonText(change.value), maxChars);
  if (excerpt.exact) {
    return change;
  }
  const bounded = {
    present: true,
    value_excerpt: {
      head: excerpt.head,
      tail: excerpt.tail,
      source_chars: excerpt.totalChars,
    },
  };
  // Keep short scalars whole: marking a cut must actually save space after
  // accounting for the excerpt's structural metadata and JSON escaping.
  return jsonText(bounded).length < jsonText(change).length ? bounded : change;
}

function excerptFieldChange(change: FieldChange, maxChars: number): unknown {
  return {
    old: excerptChangedValue(change.old, maxChars),
    new: excerptChangedValue(change.new, maxChars),
  };
}

function identityEventChangeExcerpt(event: IdentityEvent) {
  const source = identityEventChangeSource(event);
  const sourceText = jsonText(source.value);
  let bounded = sourceText;

  if (sourceText.length > IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS) {
    const selectedFields = [...(source.changedFields ?? [])];
    const render = (maxChars: number): string =>
      jsonText(
        source.format === "whole_value_old_to_new"
          ? excerptFieldChange(source.value, maxChars)
          : {
              changed_fields: selectedFields,
              field_changes: Object.fromEntries(
                selectedFields.map((field) => [
                  field,
                  excerptFieldChange(source.value.field_changes[field]!, maxChars),
                ]),
              ),
              ...(selectedFields.length < source.changedFields.length
                ? { omitted_fields: source.changedFields.length - selectedFields.length }
                : {}),
            },
      );

    // Keep field names, presence, old/new structure, and short comparisons
    // before allocating room to large values. If even that exceeds the cap,
    // count whole trailing omissions; the outer changed_fields stays complete.
    bounded = render(0);
    while (
      bounded.length > IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS &&
      selectedFields.length > 0
    ) {
      selectedFields.pop();
      bounded = render(0);
    }

    let low = 0;
    let high = IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS;
    while (low <= high) {
      const candidateBudget = Math.floor((low + high) / 2);
      const candidate = render(candidateBudget);
      if (candidate.length <= IDENTITY_EVENT_COGNITION_CHANGE_EXCERPT_MAX_CHARS) {
        bounded = candidate;
        low = candidateBudget + 1;
      } else {
        high = candidateBudget - 1;
      }
    }
  }

  return {
    format: source.format,
    changed_fields: source.changedFields,
    excerpt_head: bounded,
    excerpt_tail: null,
    excerpt_exact: bounded === sourceText,
    excerpt_chars: bounded.length,
    source_chars: sourceText.length,
  };
}

export function createIdentityEventsListForCognitionTool(
  options: IdentityEventsListForCognitionToolOptions,
): ToolDefinition<
  z.infer<typeof identityEventsListInputSchema>,
  z.infer<typeof identityEventsListForCognitionOutputSchema>
> {
  return {
    name: "tool.identityEvents.listForCognition",
    description:
      "List recent identity events from the being's global memory with disclosure labels and bounded mechanical old-to-new change excerpts.",
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
            id: event.id,
            record_type: event.record_type,
            record_id: event.record_id,
            action: event.action,
            reason: event.reason,
            provenance: event.provenance,
            review_item_id: event.review_item_id,
            overwrite_without_review: event.overwrite_without_review,
            ts: event.ts,
            change: identityEventChangeExcerpt(event),
            ...memoryDisclosurePayloadFields(disclosureLabel),
          };
        }),
      };
    },
  };
}
