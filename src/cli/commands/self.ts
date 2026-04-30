// Self-memory CLI commands for goals, values, periods, growth, questions, traits, and working memory.
import type { CAC } from "cac";

import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import {
  resolveAutobiographicalPeriodId,
  resolveEpisodeId,
  resolveGoalId,
  resolveOpenQuestionId,
  resolveSessionId,
  resolveValueId,
} from "../helpers/id-resolvers.js";
import {
  clamp,
  parseFiniteNumber,
  parseGoalStatus,
  parseGrowthMarkerCategory,
  parseLimit,
  parseOpenQuestionStatus,
  parsePriority,
  parseRequiredText,
  parseSinceToTimestamp,
} from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

function requireIdentityApplied<T>(
  result:
    | {
        status: "applied";
        record: T;
      }
    | {
        status: "requires_review";
        current: T;
      },
  action: string,
): T {
  if (result.status === "applied") {
    return result.record;
  }

  throw new CliError(`${action} requires identity review`, {
    code: "IDENTITY_REVIEW_REQUIRED",
  });
}

export function registerSelfCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("goal <action> [arg]", "Manage self goals")
    .option("--description <text>", "Goal description")
    .option("--priority <priority>", "Goal priority", {
      type: [Number],
    })
    .option("--parent <id>", "Parent goal id")
    .option("--status <status>", "Goal status filter")
    .option("--note <text>", "Progress note")
    .action(async (action: string, arg: string | undefined, commandOptions: CommandOptions) => {
      if (action === "add") {
        const goal = await withBorg(options, async (borg) =>
          borg.self.goals.add({
            description: parseRequiredText(commandOptions.description, "--description"),
            priority: parsePriority(commandOptions.priority),
            parentId:
              commandOptions.parent === undefined ? null : resolveGoalId(commandOptions.parent),
            provenance: {
              kind: "manual",
            },
          }),
        );

        writeLine(stdout, JSON.stringify(goal, null, 2));
        return;
      }

      if (action === "list") {
        const status =
          commandOptions.status === undefined ? undefined : parseGoalStatus(commandOptions.status);
        const goals = await withBorg(options, async (borg) => borg.self.goals.list({ status }));
        writeLine(stdout, JSON.stringify(goals, null, 2));
        return;
      }

      if (action === "done") {
        const goalId = resolveGoalId(arg);
        await withBorg(options, async (borg) => {
          borg.self.goals.updateStatus(goalId, "done", {
            kind: "manual",
          });
        });
        writeLine(stdout, JSON.stringify({ id: goalId, status: "done" }));
        return;
      }

      if (action === "block") {
        const goalId = resolveGoalId(arg);
        await withBorg(options, async (borg) => {
          borg.self.goals.updateStatus(goalId, "blocked", {
            kind: "manual",
          });
        });
        writeLine(stdout, JSON.stringify({ id: goalId, status: "blocked" }));
        return;
      }

      if (action === "progress") {
        const goalId = resolveGoalId(arg);
        const note = parseRequiredText(commandOptions.note, "--note");
        await withBorg(options, async (borg) => {
          borg.self.goals.updateProgress(goalId, note, {
            kind: "manual",
          });
        });
        writeLine(stdout, JSON.stringify({ id: goalId, progress_notes: note }));
        return;
      }

      throw new CliError(`Unknown goal action: ${action}`);
    });

  cli
    .command("value <action> [arg]", "Manage self values")
    .option("--label <text>", "Value label")
    .option("--description <text>", "Value description")
    .option("--priority <priority>", "Value priority", {
      type: [Number],
    })
    .action(async (action: string, arg: string | undefined, commandOptions: CommandOptions) => {
      if (action === "add") {
        const value = await withBorg(options, async (borg) =>
          borg.self.values.add({
            label: parseRequiredText(commandOptions.label, "--label"),
            description: parseRequiredText(commandOptions.description, "--description"),
            priority: parsePriority(commandOptions.priority),
            provenance: {
              kind: "manual",
            },
          }),
        );

        writeLine(stdout, JSON.stringify(value, null, 2));
        return;
      }

      if (action === "list") {
        const values = await withBorg(options, async (borg) => borg.self.values.list());
        writeLine(stdout, JSON.stringify(values, null, 2));
        return;
      }

      if (action === "affirm") {
        const valueId = resolveValueId(arg);
        await withBorg(options, async (borg) => {
          borg.self.values.update(
            valueId,
            {
              last_affirmed: Date.now(),
            },
            {
              kind: "manual",
            },
          );
        });
        writeLine(stdout, JSON.stringify({ id: valueId, affirmed: true }));
        return;
      }

      throw new CliError(`Unknown value action: ${action}`);
    });

  cli
    .command("period <action> [arg]", "Manage autobiographical periods")
    .option("--limit <count>", "Maximum number of periods", {
      default: 20,
      type: [Number],
    })
    .option("--narrative <text>", "Autobiographical period narrative")
    .action(async (action: string, arg: string | undefined, commandOptions: CommandOptions) => {
      if (action === "current") {
        const period = await withBorg(options, async (borg) =>
          borg.self.autobiographical.currentPeriod(),
        );
        writeLine(stdout, JSON.stringify(period, null, 2));
        return;
      }

      if (action === "list") {
        const periods = await withBorg(options, async (borg) =>
          borg.self.autobiographical.listPeriods({
            limit: parseLimit(commandOptions.limit),
          }),
        );
        writeLine(stdout, JSON.stringify(periods, null, 2));
        return;
      }

      if (action === "open") {
        const period = await withBorg(options, async (borg) =>
          borg.self.autobiographical.upsertPeriod({
            label: parseRequiredText(arg, "<label>"),
            start_ts: Date.now(),
            narrative: parseRequiredText(commandOptions.narrative, "--narrative"),
            provenance: {
              kind: "manual",
            },
          }),
        );
        writeLine(stdout, JSON.stringify(period, null, 2));
        return;
      }

      if (action === "close") {
        const periodId = resolveAutobiographicalPeriodId(arg);
        await withBorg(options, async (borg) => {
          requireIdentityApplied(
            borg.self.autobiographical.closePeriod(periodId, Date.now(), {
              kind: "manual",
            }),
            "Closing period",
          );
        });
        writeLine(stdout, JSON.stringify({ id: periodId, closed: true }));
        return;
      }

      if (action === "show") {
        const periodId = resolveAutobiographicalPeriodId(arg);
        const period = await withBorg(options, async (borg) =>
          borg.self.autobiographical.getPeriod(periodId),
        );

        if (period === null) {
          throw new CliError(`Period not found: ${periodId}`, {
            code: "CLI_NOT_FOUND",
          });
        }

        writeLine(stdout, JSON.stringify(period, null, 2));
        return;
      }

      throw new CliError(`Unknown period action: ${action}`);
    });

  cli
    .command("growth <action> [arg1] [arg2]", "Manage growth markers")
    .option("--since <duration>", "Relative duration like 1h or epoch ms")
    .option("--until <duration>", "Relative duration like 1h or epoch ms")
    .option("--category <category>", "Growth marker category")
    .option("--episode <id>", "Evidence episode id")
    .action(
      async (
        action: string,
        arg1: string | undefined,
        arg2: string | undefined,
        commandOptions: CommandOptions,
      ) => {
        if (action === "list") {
          const markers = await withBorg(options, async (borg) =>
            borg.self.growthMarkers.list({
              sinceTs: parseSinceToTimestamp(commandOptions.since),
              untilTs: parseSinceToTimestamp(commandOptions.until, "--until"),
              category:
                commandOptions.category === undefined
                  ? undefined
                  : parseGrowthMarkerCategory(commandOptions.category),
            }),
          );
          writeLine(stdout, JSON.stringify(markers, null, 2));
          return;
        }

        if (action === "add") {
          const marker = await withBorg(options, async (borg) =>
            borg.self.growthMarkers.add({
              ts: Date.now(),
              category: parseGrowthMarkerCategory(arg1),
              what_changed: parseRequiredText(arg2, "<change>"),
              evidence_episode_ids: [resolveEpisodeId(commandOptions.episode)],
              confidence: 0.6,
              source_process: "manual",
              provenance: {
                kind: "manual",
              },
            }),
          );
          writeLine(stdout, JSON.stringify(marker, null, 2));
          return;
        }

        throw new CliError(`Unknown growth action: ${action}`);
      },
    );

  cli
    .command("question <action> [arg1] [arg2]", "Manage open questions")
    .option("--status <status>", "Open question status")
    .option("--min-urgency <value>", "Minimum urgency", {
      type: [Number],
    })
    .option("--episode <id>", "Resolution episode id")
    .option("--note <text>", "Resolution note")
    .option("--reason <text>", "Abandon reason")
    .action(
      async (
        action: string,
        arg1: string | undefined,
        arg2: string | undefined,
        commandOptions: CommandOptions,
      ) => {
        if (action === "list") {
          const minUrgency =
            commandOptions.minUrgency === undefined
              ? undefined
              : clamp(parseFiniteNumber(commandOptions.minUrgency, "--min-urgency"), 0, 1);
          const questions = await withBorg(options, async (borg) =>
            borg.self.openQuestions.list({
              status:
                commandOptions.status === undefined
                  ? undefined
                  : parseOpenQuestionStatus(commandOptions.status),
              minUrgency,
            }),
          );
          writeLine(stdout, JSON.stringify(questions, null, 2));
          return;
        }

        if (action === "add") {
          const question = await withBorg(options, async (borg) =>
            borg.self.openQuestions.add({
              question: parseRequiredText(arg1, "<question>"),
              urgency: 0.4,
              provenance: {
                kind: "manual",
              },
              source: "user",
            }),
          );
          writeLine(stdout, JSON.stringify(question, null, 2));
          return;
        }

        if (action === "resolve") {
          const questionId = resolveOpenQuestionId(arg1);
          const resolved = await withBorg(options, async (borg) =>
            requireIdentityApplied(
              borg.self.openQuestions.resolve(
                questionId,
                {
                  resolution_episode_id: resolveEpisodeId(commandOptions.episode),
                  resolution_note:
                    commandOptions.note === undefined
                      ? null
                      : parseRequiredText(commandOptions.note, "--note"),
                },
                {
                  kind: "manual",
                },
              ),
              "Resolving open question",
            ),
          );
          writeLine(stdout, JSON.stringify(resolved, null, 2));
          return;
        }

        if (action === "abandon") {
          const questionId = resolveOpenQuestionId(arg1);
          const abandoned = await withBorg(options, async (borg) =>
            requireIdentityApplied(
              borg.self.openQuestions.abandon(
                questionId,
                typeof commandOptions.reason === "string"
                  ? commandOptions.reason
                  : "Abandoned from CLI",
                {
                  kind: "manual",
                },
              ),
              "Abandoning open question",
            ),
          );
          writeLine(stdout, JSON.stringify(abandoned, null, 2));
          return;
        }

        if (action === "bump") {
          const questionId = resolveOpenQuestionId(arg1);
          const bumped = await withBorg(options, async (borg) =>
            requireIdentityApplied(
              borg.self.openQuestions.bumpUrgency(questionId, parseFiniteNumber(arg2, "<delta>"), {
                kind: "manual",
              }),
              "Bumping open question urgency",
            ),
          );
          writeLine(stdout, JSON.stringify(bumped, null, 2));
          return;
        }

        throw new CliError(`Unknown question action: ${action}`);
      },
    );

  cli.command("trait <action>", "Inspect traits").action(async (action: string) => {
    if (action !== "show") {
      throw new CliError(`Unknown trait action: ${action}`);
    }

    const traits = await withBorg(options, async (borg) => borg.self.traits.list());
    writeLine(stdout, JSON.stringify(traits, null, 2));
  });

  cli
    .command("workmem <action>", "Inspect or clear working memory")
    .option("--session <id>", "Session id to use")
    .action(async (action: string, commandOptions: CommandOptions) => {
      const sessionId = resolveSessionId(commandOptions.session);

      if (action === "show") {
        const state = await withBorg(options, async (borg) => borg.workmem.load(sessionId));
        writeLine(stdout, JSON.stringify(state, null, 2));
        return;
      }

      if (action === "clear") {
        await withBorg(options, async (borg) => {
          borg.workmem.clear(sessionId);
        });
        writeLine(stdout, JSON.stringify({ session: sessionId, cleared: true }));
        return;
      }

      throw new CliError(`Unknown workmem action: ${action}`);
    });
}
