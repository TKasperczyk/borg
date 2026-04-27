import { z } from "zod";

import {
  skillContextStatsSchema,
  skillSchema,
  type SkillContextStatsRecord,
  type SkillRecord,
} from "../../memory/procedural/index.js";
import type { ToolDefinition } from "../dispatcher.js";

const skillsListInputSchema = z
  .object({
    limit: z.number().int().positive().max(50).optional(),
  })
  .strict();

const skillToolSchema = skillSchema.omit({
  source_episode_ids: true,
});

const skillsListOutputSchema = z.object({
  skills: z.array(skillToolSchema),
  context_stats_by_skill_id: z.record(z.string(), z.array(skillContextStatsSchema)).optional(),
});

export type SkillsListToolOptions = {
  listSkills: (limit: number) => SkillRecord[];
  listContextStatsForSkill?: (skillId: SkillRecord["id"]) => SkillContextStatsRecord[];
};

const DEFAULT_LIMIT = 20;

function toSkillToolOutput(skill: SkillRecord): z.infer<typeof skillToolSchema> {
  const { source_episode_ids: _sourceEpisodeIds, ...safeSkill } = skill;

  return skillToolSchema.parse(safeSkill);
}

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
      const skills = options.listSkills(input.limit ?? DEFAULT_LIMIT);
      const safeSkills = skills.map((skill) => toSkillToolOutput(skill));
      const contextStatsBySkillId =
        options.listContextStatsForSkill === undefined
          ? undefined
          : Object.fromEntries(
              skills
                .map((skill) => [
                  skill.id,
                  options.listContextStatsForSkill?.(skill.id) ?? [],
                ] as const)
                .filter(([, stats]) => stats.length > 0),
            );

      return {
        skills: safeSkills,
        ...(contextStatsBySkillId === undefined
          ? {}
          : { context_stats_by_skill_id: contextStatsBySkillId }),
      };
    },
  };
}
