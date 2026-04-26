import { z } from "zod";

import { skillSchema, type SkillRecord } from "../../memory/procedural/index.js";
import type { ToolDefinition } from "../dispatcher.js";

const skillsListInputSchema = z
  .object({
    limit: z.number().int().positive().max(50).optional(),
  })
  .strict();

const skillsListOutputSchema = z.object({
  skills: z.array(skillSchema),
});

export type SkillsListToolOptions = {
  listSkills: (limit: number) => SkillRecord[];
};

const DEFAULT_LIMIT = 20;

export function createSkillsListTool(
  options: SkillsListToolOptions,
): ToolDefinition<
  z.infer<typeof skillsListInputSchema>,
  z.infer<typeof skillsListOutputSchema>
> {
  return {
    name: "tool.skills.list",
    description:
      "List recorded procedural skills (most recently updated first). Use to introspect what's in the procedural memory band when no skill activated this turn or when reflecting on the substrate.",
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: skillsListInputSchema,
    outputSchema: skillsListOutputSchema,
    async invoke(input) {
      return {
        skills: options.listSkills(input.limit ?? DEFAULT_LIMIT),
      };
    },
  };
}
