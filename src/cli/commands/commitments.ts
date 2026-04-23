// Commitment CLI commands for adding, revoking, and listing active commitments.
import type { CAC } from "cac";

import { parseEpisodeId } from "../../util/ids.js";
import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveCommitmentId } from "../helpers/id-resolvers.js";
import {
  parseCommitmentType,
  parseIdList,
  parsePriority,
  parseRequiredText,
} from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerCommitmentCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("commitment <action> [arg]", "Manage commitments")
    .option("--type <type>", "Commitment type")
    .option("--directive <text>", "Commitment directive")
    .option("--priority <priority>", "Priority", {
      type: [Number],
    })
    .option("--audience <entity>", "Restricted audience name")
    .option("--about <entity>", "About-entity name")
    .option("--made-to <entity>", "Made-to entity name")
    .option("--source-episodes <ids>", "Comma-separated episode ids")
    .option("--reason <text>", "Revocation reason")
    .action(async (action: string, arg: string | undefined, commandOptions: CommandOptions) => {
      if (action === "add") {
        const commitment = await withBorg(options, async (borg) =>
          borg.commitments.add({
            type: parseCommitmentType(commandOptions.type),
            directive: parseRequiredText(commandOptions.directive, "--directive"),
            priority: parsePriority(commandOptions.priority),
            audience:
              typeof commandOptions.audience === "string" ? commandOptions.audience : undefined,
            about: typeof commandOptions.about === "string" ? commandOptions.about : undefined,
            madeTo:
              typeof (commandOptions.madeTo ?? commandOptions["made-to"]) === "string"
                ? String(commandOptions.madeTo ?? commandOptions["made-to"])
                : undefined,
            provenance: (() => {
              const sourceEpisodeIds = parseIdList(
                commandOptions.sourceEpisodes ?? commandOptions["source-episodes"],
                (value) => parseEpisodeId(value),
                "--source-episodes",
              );
              return sourceEpisodeIds.length > 0
                ? {
                    kind: "episodes" as const,
                    episode_ids: sourceEpisodeIds,
                  }
                : {
                    kind: "manual" as const,
                  };
            })(),
          }),
        );
        writeLine(stdout, JSON.stringify(commitment, null, 2));
        return;
      }

      if (action === "revoke") {
        const revoked = await withBorg(options, async (borg) =>
          borg.commitments.revoke(
            resolveCommitmentId(arg),
            parseRequiredText(commandOptions.reason, "--reason"),
            {
              kind: "manual",
            },
          ),
        );
        writeLine(stdout, JSON.stringify(revoked, null, 2));
        return;
      }

      if (action === "list") {
        const commitments = await withBorg(options, async (borg) =>
          borg.commitments.list({
            activeOnly: true,
            audience:
              typeof commandOptions.audience === "string" ? commandOptions.audience : undefined,
          }),
        );
        writeLine(stdout, JSON.stringify(commitments, null, 2));
        return;
      }

      throw new CliError(`Unknown commitment action: ${action}`);
    });
}
