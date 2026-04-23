// Social memory CLI commands for profiles and trust adjustments.
import type { CAC } from "cac";

import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { parseFiniteNumber, parseRequiredText } from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerSocialCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("social <action> <entity> [delta]", "Manage social profiles")
    .action(
      async (
        action: string,
        entity: string | undefined,
        delta: string | undefined,
        _commandOptions: CommandOptions,
      ) => {
        const entityName = parseRequiredText(entity, "<entity>");

        if (action === "profile") {
          const profile = await withBorg(options, async (borg) =>
            borg.social.getProfile(entityName),
          );
          writeLine(stdout, JSON.stringify(profile, null, 2));
          return;
        }

        if (action === "upsert") {
          const profile = await withBorg(options, async (borg) =>
            borg.social.upsertProfile(entityName),
          );
          writeLine(stdout, JSON.stringify(profile, null, 2));
          return;
        }

        if (action === "adjust-trust") {
          const profile = await withBorg(options, async (borg) =>
            borg.social.adjustTrust(entityName, parseFiniteNumber(delta, "<delta>"), {
              kind: "manual",
            }),
          );
          writeLine(stdout, JSON.stringify(profile, null, 2));
          return;
        }

        throw new CliError(`Unknown social action: ${action}`);
      },
    );
}
