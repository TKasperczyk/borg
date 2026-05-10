// Review queue CLI commands for listing and resolving human review items.
import type { CAC } from "cac";

import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveSemanticNodeId } from "../helpers/id-resolvers.js";
import {
  parseOptionalPositiveInteger,
  parseRequiredText,
  parseReviewKind,
  parseReviewResolution,
} from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";
import type { ReviewQueueItem } from "../../memory/semantic/index.js";

function renderReviewItem(
  item: ReviewQueueItem,
): ReviewQueueItem | (ReviewQueueItem & { summary: Record<string, unknown> }) {
  if (item.kind !== "skill_split") {
    return item;
  }

  const proposedChildren = Array.isArray(item.refs.proposed_children)
    ? item.refs.proposed_children
    : [];

  return {
    ...item,
    summary: {
      original_skill_id: item.refs.original_skill_id,
      children: proposedChildren.map((child) =>
        child !== null && typeof child === "object" && "label" in child
          ? (child as { label: unknown }).label
          : null,
      ),
      rationale: item.refs.rationale,
      resolution: item.refs.review_resolution,
    },
  };
}

export function registerReviewCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, stderr, options } = deps;

  cli
    .command("review <action> [arg1] [arg2]", "Inspect and resolve review items")
    .option("--kind <kind>", "Review item kind")
    .option("--data-dir <path>", "Borg data directory")
    .option("--max-age-days <days>", "Only revalidate items older than this many days")
    .option("--winner-node-id <id>", "Winner node id for duplicate/contradiction resolution")
    .option("--accept", "Accept a correction review item")
    .option("--reject", "Reject a correction review item")
    .option("--reason <reason>", "Resolution reason for review rejection")
    .action(
      async (
        action: string,
        arg1: string | undefined,
        arg2: string | undefined,
        commandOptions: CommandOptions,
      ) => {
        const dataDir =
          commandOptions.dataDir === undefined
            ? undefined
            : parseRequiredText(commandOptions.dataDir, "--data-dir");
        const borgOptions = dataDir === undefined ? options : { ...options, dataDir };

        if (action === "list") {
          const items = await withBorg(borgOptions, async (borg) =>
            borg.review.list({
              kind:
                commandOptions.kind === undefined
                  ? undefined
                  : parseReviewKind(commandOptions.kind),
              openOnly: true,
            }),
          );
          writeLine(
            stdout,
            JSON.stringify(
              items.map((item) => renderReviewItem(item)),
              null,
              2,
            ),
          );
          return;
        }

        if (action === "revalidate") {
          const kind = parseReviewKind(commandOptions.kind);

          if (kind !== "misattribution") {
            throw new CliError("review revalidate currently supports --kind misattribution");
          }

          const maxAgeDays = parseOptionalPositiveInteger(
            commandOptions.maxAgeDays,
            "--max-age-days",
          );
          const result = await withBorg(borgOptions, async (borg) =>
            borg.review.revalidate({
              kind,
              ...(maxAgeDays === undefined ? {} : { maxAgeDays }),
            }),
          );

          for (const warning of result.warnings) {
            writeLine(stderr, `Warning: ${warning}`);
          }

          writeLine(stdout, JSON.stringify(result, null, 2));
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
          const reason =
            typeof commandOptions.reason === "string" ? commandOptions.reason.trim() : undefined;

          const resolved = await withBorg(borgOptions, async (borg) =>
            borg.review.resolve(
              itemId,
              typeof commandOptions.winnerNodeId === "string" || reason !== undefined
                ? {
                    decision,
                    ...(typeof commandOptions.winnerNodeId === "string"
                      ? { winner_node_id: resolveSemanticNodeId(commandOptions.winnerNodeId) }
                      : {}),
                    ...(reason === undefined ? {} : { reason }),
                  }
                : decision,
            ),
          );
          writeLine(
            stdout,
            JSON.stringify(resolved === null ? null : renderReviewItem(resolved), null, 2),
          );
          return;
        }

        throw new CliError(`Unknown review action: ${action}`);
      },
    );
}
