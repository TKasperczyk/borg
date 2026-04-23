// Affective memory CLI commands for mood state inspection.
import type { CAC } from "cac";

import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveSessionId } from "../helpers/id-resolvers.js";
import { parseSinceToTimestamp } from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerAffectiveCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("mood <action>", "Inspect affective mood state")
    .option("--session <id>", "Session id")
    .option("--since <duration>", "Relative duration like 1h or epoch ms")
    .option("--until <duration>", "Relative duration like 1h or epoch ms")
    .action(async (action: string, commandOptions: CommandOptions) => {
      const sessionId = resolveSessionId(commandOptions.session);

      if (action === "current") {
        const mood = await withBorg(options, async (borg) => borg.mood.current(sessionId));
        writeLine(stdout, JSON.stringify(mood, null, 2));
        return;
      }

      if (action === "history") {
        const history = await withBorg(options, async (borg) =>
          borg.mood.history(sessionId, {
            fromTs: parseSinceToTimestamp(commandOptions.since, "--since"),
            toTs: parseSinceToTimestamp(commandOptions.until, "--until"),
          }),
        );
        writeLine(stdout, JSON.stringify(history, null, 2));
        return;
      }

      throw new CliError(`Unknown mood action: ${action}`);
    });
}
