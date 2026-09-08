import { z } from "zod";

import {
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
  type MemoryDisclosureLabel,
} from "../../memory/common/index.js";
import { memoryDisclosureLabelMetadataSchema } from "../../memory/common/disclosure-label.js";
import {
  goalBlockInputSchema,
  goalBlockStateFields,
  goalIdSchema,
  goalSchema,
  type GoalRecord,
  type GoalsRepository,
} from "../../memory/self/index.js";
import type { ToolDefinition } from "../dispatcher.js";

const blockInputSchema = goalBlockInputSchema.extend({ goal_id: goalIdSchema }).strict();
const unblockInputSchema = z
  .object({ goal_id: goalIdSchema, reason: z.string().trim().min(1) })
  .strict();
const outputSchema = z.object({
  goal: goalSchema.pick({ id: true, description: true, status: true, block_history: true }).extend({
    disclosure: z.string(),
    disclosure_label: memoryDisclosureLabelMetadataSchema,
  }),
});

type Options = {
  goalsRepository: Pick<GoalsRepository, "block" | "unblock">;
  disclosureLabelForGoal?: (goal: GoalRecord) => MemoryDisclosureLabel;
};

function result(goal: GoalRecord, options: Options): z.infer<typeof outputSchema> {
  return {
    goal: {
      id: goal.id,
      description: goal.description,
      status: goal.status,
      block_history: goalBlockStateFields(goal).block_history,
      ...memoryDisclosurePayloadFields(
        options.disclosureLabelForGoal?.(goal) ?? goalMemoryDisclosureLabel(goal),
      ),
    },
  };
}

export function createGoalsBlockTool(
  options: Options,
): ToolDefinition<z.infer<typeof blockInputSchema>, z.infer<typeof outputSchema>> {
  return {
    name: "tool.goals.block",
    description:
      "Only block goals I attempted and found unavailable; unattempted goals stay active. Declare attempted_unavailable with my reason and exactly one named blocker: another existing goal, an existing entity, or an until timestamp in Unix milliseconds. Optional attempt_evidence names an existing artifact from this or an earlier turn. Blocking pauses executive competition and scheduling pressure, preserves open steps and recall, and records its history and disclosure labels. A terminal status of the blocker goal (done or abandoned, including retirement), a later inbound entry from the entity, or the timestamp passing reactivates it with an audited basis. A blocker goal becoming blocked is not a terminal status.",
    menuSummary: "Block an attempted but unavailable goal with a named blocker and reason.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: blockInputSchema,
    outputSchema,
    async invoke(input) {
      const { goal_id, ...block } = input;
      return result(
        options.goalsRepository.block(goal_id, block, {
          kind: "online",
          process: "tool.goals.block",
        }),
        options,
      );
    },
  };
}

export function createGoalsUnblockTool(
  options: Options,
): ToolDefinition<z.infer<typeof unblockInputSchema>, z.infer<typeof outputSchema>> {
  return {
    name: "tool.goals.unblock",
    description:
      "Return a blocked goal to active with my reason, keeping its block history. An already unblocked goal returns its actual state unchanged.",
    menuSummary: "Unblock a goal with my reason, preserving its history.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "write",
    inputSchema: unblockInputSchema,
    outputSchema,
    async invoke(input) {
      return result(
        options.goalsRepository.unblock(input.goal_id, input.reason, {
          kind: "online",
          process: "tool.goals.unblock",
        }),
        options,
      );
    },
  };
}
