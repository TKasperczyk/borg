import { z } from "zod";

import {
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
  unknownMemoryDisclosureLabel,
} from "../../memory/common/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import {
  goalIdSchema,
  goalStatusSchema,
  goalRetirementDispositionSchema,
  type GoalsRepository,
} from "../../memory/self/index.js";
import type { GoalRecord } from "../../memory/self/index.js";
import type { MemoryDisclosureLabel } from "../../memory/common/index.js";
import type { ToolDefinition } from "../dispatcher.js";

const MAX_GOAL_DESCRIPTION_CHARS = 160;

const goalsRetireInputSchema = z
  .object({
    goal_id: goalIdSchema,
    disposition: goalRetirementDispositionSchema,
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
      status: goalRetirementDispositionSchema,
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
  disclosureLabelForGoal?: (goal: GoalRecord) => MemoryDisclosureLabel;
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
      "Retire one active goal with a required disposition and my reason: done records that the goal is complete; superseded records that it has been replaced; abandoned records that I stopped pursuing it. Each disposition writes that exact terminal status and closes outstanding executive steps as abandoned without claiming those steps were completed. Missing or inactive goals return a no-op with their actual status.",
    menuSummary:
      "Retire one of my own goals as done, superseded, or abandoned, with my reason; close outstanding steps without claiming completion.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: goalsRetireInputSchema,
    outputSchema: goalsRetireOutputSchema,
    async invoke(input) {
      const result = options.goalsRepository.retire(
        input.goal_id,
        input.reason,
        {
          kind: "online",
          process: "tool.goals.retire",
        },
        input.disposition,
      );

      if (result.status === "applied") {
        const disclosureLabel =
          options.disclosureLabelForGoal?.(result.goal) ?? goalMemoryDisclosureLabel(result.goal);
        return {
          status: "applied",
          goal: {
            id: result.goal.id,
            description: truncateGoalDescription(result.goal.description),
            status: goalRetirementDispositionSchema.parse(result.goal.status),
            ...memoryDisclosurePayloadFields(disclosureLabel),
          },
        };
      }

      const disclosureLabel =
        result.goal === null
          ? unknownMemoryDisclosureLabel()
          : (options.disclosureLabelForGoal?.(result.goal) ??
            goalMemoryDisclosureLabel(result.goal));

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
