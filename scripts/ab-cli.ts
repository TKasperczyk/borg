import { pathToFileURL } from "node:url";

export type AbReplayCliArgs = {
  mode: "dry" | "live";
  dataDir?: string;
  inputPath?: string;
  outputPath?: string;
  limit?: number;
  includeNonCompleted: boolean;
};

export type AbJudgeCliArgs = {
  dataDir?: string;
  inputPath?: string;
  capturesPath?: string;
  outputPath?: string;
  summaryPath?: string;
  limit?: number;
};

type Usage = {
  command: string;
  synopsis: string;
  details?: readonly string[];
};

function usageError(usage: Usage, message?: string): never {
  if (message !== undefined) {
    console.error(message);
    console.error("");
  }
  console.error(
    [
      `Usage: pnpm ${usage.command} -- ${usage.synopsis}`,
      ...(usage.details === undefined ? [] : ["", ...usage.details]),
    ].join("\n"),
  );
  process.exit(1);
}

function requiredValue(argv: readonly string[], index: number, flag: string, usage: Usage): string {
  const value = argv[index + 1];
  if (value === undefined || value.length === 0) usageError(usage, `${flag} requires a value`);
  return value;
}

function positiveLimit(value: string, usage: Usage): number {
  const limit = Number(value);
  if (!Number.isInteger(limit) || limit <= 0) {
    usageError(usage, "--limit must be a positive integer");
  }
  return limit;
}

export function parseAbReplayCliArgs(
  argv: readonly string[],
  options: { command: string; includeNonCompleted: boolean },
): AbReplayCliArgs {
  const usage: Usage = {
    command: options.command,
    synopsis: `[--dry|--live] [--data-dir DIR] [--input FILE] [--output FILE] [--limit N]${options.includeNonCompleted ? " [--include-non-completed]" : ""}`,
    details: [
      "--dry is the default and performs no LLM calls.",
      "--live runs both presentation variants through the existing unary LLM transport.",
      ...(options.includeNonCompleted
        ? ["Degraded/threw source captures are excluded unless --include-non-completed is set."]
        : []),
    ],
  };
  const args: AbReplayCliArgs = {
    mode: "dry",
    includeNonCompleted: false,
  };
  let modeSeen: AbReplayCliArgs["mode"] | undefined;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--dry":
      case "--live": {
        const requestedMode = arg === "--live" ? "live" : "dry";
        if (modeSeen !== undefined && modeSeen !== requestedMode) {
          usageError(usage, "Choose exactly one of --dry or --live");
        }
        args.mode = requestedMode;
        modeSeen = requestedMode;
        break;
      }
      case "--data-dir":
        args.dataDir = requiredValue(argv, index, arg, usage);
        index += 1;
        break;
      case "--input":
        args.inputPath = requiredValue(argv, index, arg, usage);
        index += 1;
        break;
      case "--output":
        args.outputPath = requiredValue(argv, index, arg, usage);
        index += 1;
        break;
      case "--limit":
        args.limit = positiveLimit(requiredValue(argv, index, arg, usage), usage);
        index += 1;
        break;
      case "--include-non-completed":
        if (!options.includeNonCompleted) {
          usageError(usage, `Unknown argument: ${arg}`);
        }
        args.includeNonCompleted = true;
        break;
      case "--help":
      case "-h":
        usageError(usage);
      default:
        usageError(usage, `Unknown argument: ${arg}`);
    }
  }
  return args;
}

export function parseAbJudgeCliArgs(argv: readonly string[], command: string): AbJudgeCliArgs {
  const usage: Usage = {
    command,
    synopsis:
      "[--data-dir DIR] [--input FILE] [--captures FILE] [--output FILE] [--summary FILE] [--limit N]",
    details: [
      "The cognition model judges eligible live replay pairs through the unary LLM transport.",
      "Outputs must be direct children of <dataDir>/captures.",
    ],
  };
  const args: AbJudgeCliArgs = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--data-dir":
        args.dataDir = requiredValue(argv, index, arg, usage);
        index += 1;
        break;
      case "--input":
        args.inputPath = requiredValue(argv, index, arg, usage);
        index += 1;
        break;
      case "--captures":
        args.capturesPath = requiredValue(argv, index, arg, usage);
        index += 1;
        break;
      case "--output":
        args.outputPath = requiredValue(argv, index, arg, usage);
        index += 1;
        break;
      case "--summary":
        args.summaryPath = requiredValue(argv, index, arg, usage);
        index += 1;
        break;
      case "--limit":
        args.limit = positiveLimit(requiredValue(argv, index, arg, usage), usage);
        index += 1;
        break;
      case "--help":
      case "-h":
        usageError(usage);
      default:
        usageError(usage, `Unknown argument: ${arg}`);
    }
  }
  return args;
}

export function runAbCliEntrypoint(
  importMetaUrl: string,
  run: (argv: readonly string[]) => Promise<void>,
): void {
  if (importMetaUrl !== pathToFileURL(process.argv[1] ?? "").href) return;
  void run(process.argv.slice(2)).catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
