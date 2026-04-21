import { cac } from "cac";

import { Borg, VERSION, loadConfig, redactConfig } from "../index.js";
import { goalStatusSchema } from "../memory/self/index.js";
import { streamEntryKindSchema } from "../stream/index.js";
import { BorgError } from "../util/errors.js";
import {
  DEFAULT_SESSION_ID,
  parseEpisodeId,
  parseGoalId,
  parseSessionId,
  parseValueId,
} from "../util/ids.js";

type Output = Pick<NodeJS.WriteStream, "write">;

export type RunCliOptions = {
  stdout?: Output;
  stderr?: Output;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
};

class CliError extends BorgError {
  constructor(message: string, options: { cause?: unknown; code?: string } = {}) {
    super(options.code ?? "CLI_ARGUMENT", message, options);
  }
}

function writeLine(output: Output, line: string): void {
  output.write(`${line}\n`);
}

function parseLimit(value: unknown, flag = "--limit"): number {
  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (!Number.isInteger(candidate) || candidate <= 0) {
    throw new CliError(`${flag} must be a positive integer`);
  }

  return candidate;
}

function parsePriority(value: unknown, fallback = 0): number {
  if (value === undefined) {
    return fallback;
  }

  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (typeof candidate !== "number" || !Number.isFinite(candidate)) {
    throw new CliError("--priority must be a finite number");
  }

  return candidate;
}

function parseRequiredText(value: unknown, flag: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(`${flag} is required`);
  }

  return value.trim();
}

function resolveSessionId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    return DEFAULT_SESSION_ID;
  }

  try {
    return parseSessionId(value);
  } catch (error) {
    throw new CliError(`Invalid session id: ${value}`, {
      cause: error,
    });
  }
}

function resolveEpisodeId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Episode id is required");
  }

  try {
    return parseEpisodeId(value);
  } catch (error) {
    throw new CliError(`Invalid episode id: ${value}`, {
      cause: error,
    });
  }
}

function resolveGoalId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Goal id is required");
  }

  try {
    return parseGoalId(value);
  } catch (error) {
    throw new CliError(`Invalid goal id: ${value}`, {
      cause: error,
    });
  }
}

function resolveValueId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Value id is required");
  }

  try {
    return parseValueId(value);
  } catch (error) {
    throw new CliError(`Invalid value id: ${value}`, {
      cause: error,
    });
  }
}

function parseSinceToTimestamp(value: unknown, nowMs = Date.now()): number | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("--since must be a duration like 1h or an epoch milliseconds timestamp");
  }

  const trimmed = value.trim();
  const absolute = Number(trimmed);

  if (Number.isFinite(absolute) && trimmed === String(absolute)) {
    return absolute;
  }

  const match = trimmed.match(/^(\d+)([smhd])$/);

  if (match === null) {
    throw new CliError("--since must be a duration like 1h or an epoch milliseconds timestamp");
  }

  const amount = Number(match[1]);
  const unit = match[2];
  const multiplier =
    unit === "s" ? 1_000 : unit === "m" ? 60_000 : unit === "h" ? 3_600_000 : 86_400_000;

  return nowMs - amount * multiplier;
}

function parseGoalStatus(value: unknown) {
  const parsed = goalStatusSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--status must be one of: active, done, abandoned, blocked", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

async function withBorg<T>(options: RunCliOptions, fn: (borg: Borg) => Promise<T>): Promise<T> {
  const borg = await Borg.open({
    env: options.env,
    dataDir: options.dataDir,
  });

  try {
    return await fn(borg);
  } finally {
    await borg.close();
  }
}

export async function runCli(argv: string[], options: RunCliOptions = {}): Promise<number> {
  const stdout = options.stdout ?? process.stdout;
  const stderr = options.stderr ?? process.stderr;
  const env = options.env ?? process.env;

  const cli = cac("borg");
  cli.help();
  cli.option("-v, --version", "Display version number");

  cli.command("version", "Print borg version").action(() => {
    writeLine(stdout, `borg ${VERSION}`);
  });

  cli.command("config <action>", "Inspect borg config").action((action: string) => {
    if (action !== "show") {
      throw new CliError(`Unknown config action: ${action}`);
    }

    const config = loadConfig({ env, dataDir: options.dataDir });
    writeLine(stdout, JSON.stringify(redactConfig(config), null, 2));
  });

  cli
    .command("stream <action>", "Read and write stream entries")
    .option("--n <count>", "Number of entries to tail", {
      default: 20,
      type: [Number],
    })
    .option("--session <id>", "Session id to use")
    .option("--kind <kind>", "Kind for stream append")
    .option("--content <text>", "Content for stream append")
    .action(async (action: string, commandOptions: Record<string, unknown>) => {
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

  cli
    .command("episode <action> [arg]", "Search, show, or extract episodic memories")
    .option("--limit <count>", "Maximum number of results", {
      default: 5,
      type: [Number],
    })
    .option("--since <duration>", "Relative duration like 1h or epoch ms")
    .action(
      async (action: string, arg: string | undefined, commandOptions: Record<string, unknown>) => {
        if (action === "search") {
          const query = parseRequiredText(arg, "<query>");
          const sinceTs = parseSinceToTimestamp(commandOptions.since);
          const results = await withBorg(options, async (borg) =>
            borg.episodic.search(query, {
              limit: parseLimit(commandOptions.limit),
              timeRange:
                sinceTs === undefined
                  ? undefined
                  : {
                      start: sinceTs,
                      end: Date.now(),
                    },
            }),
          );

          writeLine(stdout, JSON.stringify(results, null, 2));
          return;
        }

        if (action === "show") {
          const episodeId = resolveEpisodeId(arg);
          const result = await withBorg(options, async (borg) => borg.episodic.get(episodeId));

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
              sinceTs: parseSinceToTimestamp(commandOptions.since),
            }),
          );

          writeLine(stdout, JSON.stringify(result, null, 2));
          return;
        }

        throw new CliError(`Unknown episode action: ${action}`);
      },
    );

  cli
    .command("goal <action> [arg]", "Manage self goals")
    .option("--description <text>", "Goal description")
    .option("--priority <priority>", "Goal priority", {
      type: [Number],
    })
    .option("--parent <id>", "Parent goal id")
    .option("--status <status>", "Goal status filter")
    .option("--note <text>", "Progress note")
    .action(
      async (action: string, arg: string | undefined, commandOptions: Record<string, unknown>) => {
        if (action === "add") {
          const goal = await withBorg(options, async (borg) =>
            borg.self.goals.add({
              description: parseRequiredText(commandOptions.description, "--description"),
              priority: parsePriority(commandOptions.priority),
              parentId:
                commandOptions.parent === undefined ? null : resolveGoalId(commandOptions.parent),
            }),
          );

          writeLine(stdout, JSON.stringify(goal, null, 2));
          return;
        }

        if (action === "list") {
          const status =
            commandOptions.status === undefined
              ? undefined
              : parseGoalStatus(commandOptions.status);
          const goals = await withBorg(options, async (borg) => borg.self.goals.list({ status }));
          writeLine(stdout, JSON.stringify(goals, null, 2));
          return;
        }

        if (action === "done") {
          const goalId = resolveGoalId(arg);
          await withBorg(options, async (borg) => {
            borg.self.goals.updateStatus(goalId, "done");
          });
          writeLine(stdout, JSON.stringify({ id: goalId, status: "done" }));
          return;
        }

        if (action === "block") {
          const goalId = resolveGoalId(arg);
          await withBorg(options, async (borg) => {
            borg.self.goals.updateStatus(goalId, "blocked");
          });
          writeLine(stdout, JSON.stringify({ id: goalId, status: "blocked" }));
          return;
        }

        if (action === "progress") {
          const goalId = resolveGoalId(arg);
          const note = parseRequiredText(commandOptions.note, "--note");
          await withBorg(options, async (borg) => {
            borg.self.goals.updateProgress(goalId, note);
          });
          writeLine(stdout, JSON.stringify({ id: goalId, progress_notes: note }));
          return;
        }

        throw new CliError(`Unknown goal action: ${action}`);
      },
    );

  cli
    .command("value <action> [arg]", "Manage self values")
    .option("--label <text>", "Value label")
    .option("--description <text>", "Value description")
    .option("--priority <priority>", "Value priority", {
      type: [Number],
    })
    .action(
      async (action: string, arg: string | undefined, commandOptions: Record<string, unknown>) => {
        if (action === "add") {
          const value = await withBorg(options, async (borg) =>
            borg.self.values.add({
              label: parseRequiredText(commandOptions.label, "--label"),
              description: parseRequiredText(commandOptions.description, "--description"),
              priority: parsePriority(commandOptions.priority),
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
            borg.self.values.affirm(valueId);
          });
          writeLine(stdout, JSON.stringify({ id: valueId, affirmed: true }));
          return;
        }

        throw new CliError(`Unknown value action: ${action}`);
      },
    );

  cli.command("trait <action>", "Inspect traits").action(async (action: string) => {
    if (action !== "show") {
      throw new CliError(`Unknown trait action: ${action}`);
    }

    const traits = await withBorg(options, async (borg) => borg.self.traits.list());
    writeLine(stdout, JSON.stringify(traits, null, 2));
  });

  try {
    const parsed = cli.parse(argv, { run: false });

    if (parsed.options.version === true && cli.matchedCommand === undefined) {
      writeLine(stdout, `borg ${VERSION}`);
      return 0;
    }

    if (cli.matchedCommand === undefined) {
      if (parsed.args[0] === undefined) {
        cli.outputHelp();
        return 0;
      }

      writeLine(stderr, `Error: unknown command "${parsed.args[0]}"`);
      return 1;
    }

    await Promise.resolve(cli.runMatchedCommand());
    return 0;
  } catch (error) {
    writeLine(stderr, `Error: ${error instanceof Error ? error.message : String(error)}`);
    return 1;
  }
}
