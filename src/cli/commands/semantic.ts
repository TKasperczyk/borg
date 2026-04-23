// Semantic memory CLI commands for nodes, edges, and graph walks.
import type { CAC } from "cac";

import { parseEpisodeId } from "../../util/ids.js";
import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveSemanticNodeId } from "../helpers/id-resolvers.js";
import {
  parseIdList,
  parseLimit,
  parsePositiveInteger,
  parseRequiredText,
  parseSemanticNodeKind,
  parseSemanticRelation,
} from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerSemanticCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("semantic <subject> <action> [arg]", "Manage semantic nodes, edges, and walks")
    .option("--kind <kind>", "Node kind")
    .option("--label <label>", "Node label")
    .option("--description <text>", "Node description")
    .option("--domain <domain>", "Node domain")
    .option("--aliases <aliases>", "Comma-separated aliases")
    .option("--source-episodes <ids>", "Comma-separated episode ids")
    .option("--from <id>", "From node id")
    .option("--to <id>", "To node id")
    .option("--relation <relation>", "Semantic relation")
    .option("--confidence <confidence>", "Confidence", {
      type: [Number],
    })
    .option("--evidence-episodes <ids>", "Comma-separated episode ids")
    .option("--depth <depth>", "Walk depth", {
      default: 2,
      type: [Number],
    })
    .option("--limit <count>", "Maximum results", {
      default: 10,
      type: [Number],
    })
    .action(
      async (
        subject: string,
        action: string,
        arg: string | undefined,
        commandOptions: CommandOptions,
      ) => {
        if (subject === "node") {
          if (action === "add") {
            const node = await withBorg(options, async (borg) =>
              borg.semantic.nodes.add({
                kind: parseSemanticNodeKind(commandOptions.kind),
                label: parseRequiredText(commandOptions.label, "--label"),
                description: parseRequiredText(commandOptions.description, "--description"),
                domain:
                  typeof commandOptions.domain === "string" ? commandOptions.domain : undefined,
                aliases:
                  typeof commandOptions.aliases === "string"
                    ? commandOptions.aliases
                        .split(",")
                        .map((value) => value.trim())
                        .filter((value) => value.length > 0)
                    : [],
                confidence:
                  typeof commandOptions.confidence === "number"
                    ? commandOptions.confidence
                    : undefined,
                sourceEpisodeIds: parseIdList(
                  commandOptions.sourceEpisodes ?? commandOptions["source-episodes"],
                  (value) => parseEpisodeId(value),
                  "--source-episodes",
                ) as ReturnType<typeof parseIdList> as never,
              }),
            );
            writeLine(stdout, JSON.stringify(node, null, 2));
            return;
          }

          if (action === "show") {
            const node = await withBorg(options, async (borg) =>
              borg.semantic.nodes.get(resolveSemanticNodeId(arg)),
            );

            if (node === null) {
              throw new CliError(`Semantic node not found: ${arg}`, {
                code: "CLI_NOT_FOUND",
              });
            }

            writeLine(stdout, JSON.stringify(node, null, 2));
            return;
          }

          if (action === "search") {
            const results = await withBorg(options, async (borg) =>
              borg.semantic.nodes.search(parseRequiredText(arg, "<query>"), {
                limit: parseLimit(commandOptions.limit),
              }),
            );
            writeLine(stdout, JSON.stringify(results, null, 2));
            return;
          }

          if (action === "list") {
            const results = await withBorg(options, async (borg) => borg.semantic.nodes.list());
            writeLine(stdout, JSON.stringify(results, null, 2));
            return;
          }
        }

        if (subject === "edge") {
          if (action === "add") {
            const edge = await withBorg(options, async (borg) =>
              borg.semantic.edges.add({
                from_node_id: resolveSemanticNodeId(commandOptions.from),
                to_node_id: resolveSemanticNodeId(commandOptions.to),
                relation: parseSemanticRelation(commandOptions.relation),
                confidence:
                  typeof commandOptions.confidence === "number" ? commandOptions.confidence : 0.6,
                evidence_episode_ids: parseIdList(
                  commandOptions.evidenceEpisodes ?? commandOptions["evidence-episodes"],
                  (value) => parseEpisodeId(value),
                  "--evidence-episodes",
                ) as ReturnType<typeof parseIdList> as never,
                created_at: Date.now(),
                last_verified_at: Date.now(),
              }),
            );
            writeLine(stdout, JSON.stringify(edge, null, 2));
            return;
          }

          if (action === "list") {
            const edges = await withBorg(options, async (borg) =>
              borg.semantic.edges.list({
                fromId:
                  commandOptions.from === undefined
                    ? undefined
                    : resolveSemanticNodeId(commandOptions.from),
                toId:
                  commandOptions.to === undefined
                    ? undefined
                    : resolveSemanticNodeId(commandOptions.to),
                relation:
                  commandOptions.relation === undefined
                    ? undefined
                    : parseSemanticRelation(commandOptions.relation),
              }),
            );
            writeLine(stdout, JSON.stringify(edges, null, 2));
            return;
          }
        }

        if (subject === "walk") {
          const walked = await withBorg(options, async (borg) =>
            borg.semantic.walk(resolveSemanticNodeId(action), {
              depth: parsePositiveInteger(commandOptions.depth, "--depth"),
            }),
          );
          writeLine(stdout, JSON.stringify(walked, null, 2));
          return;
        }

        throw new CliError(`Unknown semantic command: ${subject} ${action}`);
      },
    );
}
