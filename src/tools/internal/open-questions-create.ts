import { z } from "zod";

import {
  type OpenQuestion,
  openQuestionSchema,
} from "../../memory/self/index.js";
import { episodeIdSchema } from "../../memory/episodic/index.js";
import { semanticNodeIdSchema } from "../../memory/semantic/types.js";
import type { ToolDefinition } from "../dispatcher.js";

const openQuestionsCreateInputSchema = z.object({
  question: z.string().min(1),
  urgency: z.number().min(0).max(1).optional(),
  related_episode_ids: z.array(episodeIdSchema).optional(),
  related_semantic_node_ids: z.array(semanticNodeIdSchema).optional(),
});

const openQuestionsCreateOutputSchema = z.object({
  openQuestion: openQuestionSchema,
});

export type OpenQuestionsCreateToolOptions = {
  createOpenQuestion: (input: {
    question: string;
    urgency: number;
    related_episode_ids: readonly z.infer<typeof episodeIdSchema>[];
    related_semantic_node_ids: readonly z.infer<typeof semanticNodeIdSchema>[];
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
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: openQuestionsCreateInputSchema,
    outputSchema: openQuestionsCreateOutputSchema,
    async invoke(input, context) {
      return {
        openQuestion: options.createOpenQuestion({
          question: input.question,
          urgency: input.urgency ?? 0.5,
          related_episode_ids: input.related_episode_ids ?? [],
          related_semantic_node_ids: input.related_semantic_node_ids ?? [],
          provenance: {
            kind: "system",
          },
          source: context.origin === "deliberator" ? "deliberator" : "autonomy",
        }),
      };
    },
  };
}
