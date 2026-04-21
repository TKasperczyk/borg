import { cac } from "cac";

import { loadConfig, redactConfig } from "../config/index.js";
import { StreamReader, StreamWriter, streamEntryKindSchema } from "../stream/index.js";
import { BorgError } from "../util/errors.js";
import { DEFAULT_SESSION_ID, parseSessionId } from "../util/ids.js";
import { VERSION } from "../index.js";

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

function parseLimit(value: unknown): number {
  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (!Number.isInteger(candidate) || candidate <= 0) {
    throw new CliError("--n must be a positive integer");
  }

  return candidate;
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
      const config = loadConfig({ env, dataDir: options.dataDir });
      const sessionId = resolveSessionId(commandOptions.session);

      if (action === "tail") {
        const limit = parseLimit(commandOptions.n);
        const reader = new StreamReader({
          dataDir: config.dataDir,
          sessionId,
        });

        for (const entry of reader.tail(limit)) {
          writeLine(stdout, JSON.stringify(entry));
        }

        return;
      }

      if (action === "append") {
        const parsedKind = streamEntryKindSchema.safeParse(commandOptions.kind);

        if (!parsedKind.success) {
          throw new CliError("--kind must be one of the supported stream kinds");
        }

        if (typeof commandOptions.content !== "string" || commandOptions.content.length === 0) {
          throw new CliError("--content is required for stream append");
        }

        const writer = new StreamWriter({
          dataDir: config.dataDir,
          sessionId,
        });

        try {
          const entry = await writer.append({
            kind: parsedKind.data,
            content: commandOptions.content,
          });

          writeLine(stdout, JSON.stringify(entry));
        } finally {
          writer.close();
        }

        return;
      }

      throw new CliError(`Unknown stream action: ${action}`);
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
