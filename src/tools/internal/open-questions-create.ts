import { z } from "zod";

import {
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
} from "../../memory/common/disclosure-serializers.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import { type OpenQuestion, openQuestionSchema } from "../../memory/self/index.js";
import { episodeIdSchema } from "../../memory/episodic/index.js";
import { semanticNodeIdSchema } from "../../memory/semantic/types.js";
import type { EntityId } from "../../util/ids.js";
import type { ToolDefinition } from "../dispatcher.js";

const openQuestionsCreateInputSchema = z.object({
  question: z.string().min(1),
  urgency: z.number().min(0).max(1).optional(),
  related_episode_ids: z.array(episodeIdSchema).optional(),
  related_semantic_node_ids: z.array(semanticNodeIdSchema).optional(),
});

const openQuestionsCreateOutputSchema = z.object({
  openQuestion: openQuestionSchema.extend({
    disclosure: z.string().min(1),
    disclosure_label: memoryDisclosureLabelMetadataSchema,
  }),
});

export type OpenQuestionsCreateToolOptions = {
  createOpenQuestion: (input: {
    question: string;
    urgency: number;
    related_episode_ids: readonly z.infer<typeof episodeIdSchema>[];
    related_semantic_node_ids: readonly z.infer<typeof semanticNodeIdSchema>[];
    audience_entity_id: EntityId | null;
    provenance: { kind: "system" };
    source: "autonomy" | "deliberator";
  }) => OpenQuestion;
};

export function createOpenQuestionsCreateTool(
  options: OpenQuestionsCreateToolOptions,
): ToolDefinition<
  z.infer<typeof openQuestionsCreateInputSchema>,
  z.infer<typeof openQuestionsCreateOutputSchema>
> {
  return {
    name: "tool.openQuestions.create",
    description: "Create a new open question in self-memory.",
    menuSummary: "Create a self-memory open question.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: openQuestionsCreateInputSchema,
    outputSchema: openQuestionsCreateOutputSchema,
    async invoke(input, context) {
      const openQuestion = options.createOpenQuestion({
        question: input.question,
        urgency: input.urgency ?? 0.5,
        related_episode_ids: input.related_episode_ids ?? [],
        related_semantic_node_ids: input.related_semantic_node_ids ?? [],
        audience_entity_id: context.audienceEntityId ?? null,
        provenance: {
          kind: "system",
        },
        source: context.origin === "deliberator" ? "deliberator" : "autonomy",
      });

      return {
        openQuestion: {
          ...openQuestion,
          ...memoryDisclosurePayloadFields(openQuestionMemoryDisclosureLabel(openQuestion)),
        },
      };
    },
  };
}
