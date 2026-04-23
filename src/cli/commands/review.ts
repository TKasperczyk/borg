// Review queue CLI commands for listing and resolving human review items.
import type { CAC } from "cac";

import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveSemanticNodeId } from "../helpers/id-resolvers.js";
import { parseRequiredText, parseReviewKind, parseReviewResolution } from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerReviewCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("review <action> [arg1] [arg2]", "Inspect and resolve review items")
    .option("--kind <kind>", "Review item kind")
    .option("--winner-node-id <id>", "Winner node id for duplicate/contradiction resolution")
    .option("--accept", "Accept a correction review item")
    .option("--reject", "Reject a correction review item")
    .action(
      async (
        action: string,
        arg1: string | undefined,
        arg2: string | undefined,
        commandOptions: CommandOptions,
      ) => {
        if (action === "list") {
          const items = await withBorg(options, async (borg) =>
            borg.review.list({
              kind:
                commandOptions.kind === undefined
                  ? undefined
                  : parseReviewKind(commandOptions.kind),
              openOnly: true,
            }),
          );
          writeLine(stdout, JSON.stringify(items, null, 2));
          return;
        }

        if (action === "resolve") {
          const itemId = Number(parseRequiredText(arg1, "<id>"));

          if (!Number.isInteger(itemId) || itemId <= 0) {
            throw new CliError("Review item id must be a positive integer");
          }

          if (commandOptions.accept === true && commandOptions.reject === true) {
            throw new CliError("--accept and --reject cannot be combined");
          }

          const decision =
            commandOptions.accept === true
              ? "accept"
              : commandOptions.reject === true
                ? "reject"
                : parseReviewResolution(arg2);

          const resolved = await withBorg(options, async (borg) =>
            borg.review.resolve(
              itemId,
              typeof commandOptions.winnerNodeId === "string"
                ? {
                    decision,
                    winner_node_id: resolveSemanticNodeId(commandOptions.winnerNodeId),
                  }
                : decision,
            ),
          );
          writeLine(stdout, JSON.stringify(resolved, null, 2));
          return;
        }

        throw new CliError(`Unknown review action: ${action}`);
      },
    );
}
