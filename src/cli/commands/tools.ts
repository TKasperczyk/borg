// Procedural skill CLI commands exposed as the tool-related command group.
import type { CAC } from "cac";

import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveEpisodeId, resolveSkillId } from "../helpers/id-resolvers.js";
import { parseLimit, parseRequiredText } from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerToolCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("skill <action> [arg]", "Manage procedural skills")
    .option("--applies-when <text>", "Context where the skill applies")
    .option("--approach <text>", "Suggested approach")
    .option("--episode <id>", "Source episode id")
    .option("--limit <count>", "Maximum number of results", {
      default: 10,
      type: [Number],
    })
    .action(async (action: string, arg: string | undefined, commandOptions: CommandOptions) => {
      if (action === "add") {
        const skill = await withBorg(options, async (borg) =>
          borg.skills.add({
            applies_when: parseRequiredText(
              commandOptions.appliesWhen ?? commandOptions["applies-when"],
              "--applies-when",
            ),
            approach: parseRequiredText(commandOptions.approach, "--approach"),
            sourceEpisodes: [resolveEpisodeId(commandOptions.episode)],
          }),
        );
        writeLine(stdout, JSON.stringify(skill, null, 2));
        return;
      }

      if (action === "list") {
        const skills = await withBorg(options, async (borg) =>
          borg.skills.list(parseLimit(commandOptions.limit)),
        );
        writeLine(stdout, JSON.stringify(skills, null, 2));
        return;
      }

      if (action === "show") {
        const skill = await withBorg(options, async (borg) => borg.skills.get(resolveSkillId(arg)));

        if (skill === null) {
          throw new CliError(`Skill not found: ${arg}`, {
            code: "CLI_NOT_FOUND",
          });
        }

        writeLine(stdout, JSON.stringify(skill, null, 2));
        return;
      }

      if (action === "select") {
        const result = await withBorg(options, async (borg) =>
          borg.skills.select(parseRequiredText(arg, "<context>"), {
            k: parseLimit(commandOptions.limit),
          }),
        );
        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      throw new CliError(`Unknown skill action: ${action}`);
    });
}
