// Episodic memory CLI commands for search, show, and extraction.
import type { CAC } from "cac";

import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveEpisodeId } from "../helpers/id-resolvers.js";
import {
  parseLimit,
  parseRequiredText,
  parseSinceToTimestamp,
  parseStringList,
  resolveEpisodeVisibilityOptions,
} from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerEpisodeCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("episode <action> [arg]", "Search, show, or extract episodic memories")
    .option("--limit <count>", "Maximum number of results", {
      default: 5,
      type: [Number],
    })
    .option("--since <duration>", "Relative duration like 1h or epoch ms")
    .option("--until <duration>", 'Relative duration like 1h, "now", or epoch ms')
    .option("--entities <list>", "Comma-separated participant/tag terms for entity retrieval")
    .option("--audience <name>", "Audience label for scoped visibility")
    .option("--all", "Search across all audiences")
    .action(async (action: string, arg: string | undefined, commandOptions: CommandOptions) => {
      if (action === "search") {
        const query = parseRequiredText(arg, "<query>");
        const nowMs = Date.now();
        const sinceTs = parseSinceToTimestamp(commandOptions.since, "--since", nowMs);
        const untilTs = parseSinceToTimestamp(commandOptions.until, "--until", nowMs);
        const entityTerms = parseStringList(commandOptions.entities, "--entities");
        const visibility = resolveEpisodeVisibilityOptions(commandOptions);
        const timeRange =
          sinceTs === undefined && untilTs === undefined
            ? undefined
            : {
                start: sinceTs ?? Number.NEGATIVE_INFINITY,
                end: untilTs ?? nowMs,
              };

        if (sinceTs !== undefined && untilTs !== undefined && sinceTs > untilTs) {
          throw new CliError("--since cannot be later than --until");
        }

        const results = await withBorg(options, async (borg) =>
          borg.episodic.search(query, {
            ...visibility,
            limit: parseLimit(commandOptions.limit),
            entityTerms: entityTerms.length === 0 ? undefined : entityTerms,
            timeRange,
            strictTimeRange: timeRange !== undefined ? true : undefined,
          }),
        );

        writeLine(stdout, JSON.stringify(results, null, 2));
        return;
      }

      if (action === "show") {
        const episodeId = resolveEpisodeId(arg);
        const visibility = resolveEpisodeVisibilityOptions(commandOptions);
        const result = await withBorg(options, async (borg) =>
          borg.episodic.get(episodeId, visibility),
        );

        if (result === null) {
          throw new CliError(`Episode not found: ${episodeId}`, {
            code: "CLI_NOT_FOUND",
          });
        }

        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      if (action === "extract") {
        const result = await withBorg(options, async (borg) =>
          borg.episodic.extract({
            sinceTs: parseSinceToTimestamp(commandOptions.since, "--since"),
          }),
        );

        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      throw new CliError(`Unknown episode action: ${action}`);
    });
}
