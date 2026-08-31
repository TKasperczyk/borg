import { z } from "zod";

import {
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
  unknownMemoryDisclosureLabel,
} from "../../memory/common/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import { goalIdSchema, goalStatusSchema, type GoalsRepository } from "../../memory/self/index.js";
import type { ToolDefinition } from "../dispatcher.js";

const MAX_GOAL_DESCRIPTION_CHARS = 160;

const goalsRetireInputSchema = z
  .object({
    goal_id: goalIdSchema,
    reason: z.string().min(1),
  })
  .strict();

const goalRetirementSummarySchema = z.object({
  id: goalIdSchema,
  description: z.string().nullable(),
  status: goalStatusSchema.or(z.literal("absent")),
  disclosure: z.string().min(1),
  disclosure_label: memoryDisclosureLabelMetadataSchema,
});

const goalsRetireOutputSchema = z.discriminatedUnion("status", [
  z.object({
    status: z.literal("applied"),
    goal: goalRetirementSummarySchema.extend({
      description: z.string(),
      status: z.literal("abandoned"),
    }),
  }),
  z.object({
    status: z.literal("no_op"),
    reason: z.enum(["missing", "not_active"]),
    goal: goalRetirementSummarySchema,
  }),
]);

export type GoalsRetireToolOptions = {
  goalsRepository: Pick<GoalsRepository, "retire">;
};

function truncateGoalDescription(description: string): string {
  if (description.length <= MAX_GOAL_DESCRIPTION_CHARS) {
    return description;
  }

  return `${description.slice(0, MAX_GOAL_DESCRIPTION_CHARS - 3)}...`;
}

export function createGoalsRetireTool(
  options: GoalsRetireToolOptions,
): ToolDefinition<z.infer<typeof goalsRetireInputSchema>, z.infer<typeof goalsRetireOutputSchema>> {
  return {
    name: "tool.goals.retire",
    description:
      "Retire one active goal when its premise is answered or superseded. This marks the goal abandoned, records my reason, and abandons its open executive steps. Missing or inactive goals return a no-op with their actual status.",
    menuSummary: "Retire one of my own goals as done/superseded, with my reason.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: goalsRetireInputSchema,
    outputSchema: goalsRetireOutputSchema,
    async invoke(input) {
      const result = options.goalsRepository.retire(input.goal_id, input.reason, {
        kind: "online",
        process: "tool.goals.retire",
      });

      if (result.status === "applied") {
        return {
          status: "applied",
          goal: {
            id: result.goal.id,
            description: truncateGoalDescription(result.goal.description),
            status: "abandoned",
            ...memoryDisclosurePayloadFields(goalMemoryDisclosureLabel(result.goal)),
          },
        };
      }

      const disclosureLabel =
        result.goal === null
          ? unknownMemoryDisclosureLabel()
          : goalMemoryDisclosureLabel(result.goal);

      return {
        status: "no_op",
        reason: result.reason,
        goal: {
          id: input.goal_id,
          description:
            result.goal === null ? null : truncateGoalDescription(result.goal.description),
          status: result.goal?.status ?? "absent",
          ...memoryDisclosurePayloadFields(disclosureLabel),
        },
      };
    },
  };
}
