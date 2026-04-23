// Stream CLI commands for reading and appending JSONL stream entries.
import type { CAC } from "cac";

import { streamEntryKindSchema } from "../../stream/index.js";
import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveSessionId } from "../helpers/id-resolvers.js";
import { parseLimit, parseRequiredText } from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerStreamCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("stream <action>", "Read and write stream entries")
    .option("--n <count>", "Number of entries to tail", {
      default: 20,
      type: [Number],
    })
    .option("--session <id>", "Session id to use")
    .option("--kind <kind>", "Kind for stream append")
    .option("--content <text>", "Content for stream append")
    .action(async (action: string, commandOptions: CommandOptions) => {
      const sessionId = resolveSessionId(commandOptions.session);

      if (action === "tail") {
        const limit = parseLimit(commandOptions.n, "--n");
        const entries = await withBorg(options, async (borg) =>
          borg.stream.tail(limit, {
            session: sessionId,
          }),
        );

        for (const entry of entries) {
          writeLine(stdout, JSON.stringify(entry));
        }

        return;
      }

      if (action === "append") {
        const parsedKind = streamEntryKindSchema.safeParse(commandOptions.kind);

        if (!parsedKind.success) {
          throw new CliError("--kind must be one of the supported stream kinds");
        }

        const content = parseRequiredText(commandOptions.content, "--content");
        const entry = await withBorg(options, async (borg) =>
          borg.stream.append(
            {
              kind: parsedKind.data,
              content,
            },
            {
              session: sessionId,
            },
          ),
        );

        writeLine(stdout, JSON.stringify(entry));
        return;
      }

      throw new CliError(`Unknown stream action: ${action}`);
    });
}
