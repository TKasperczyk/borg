import { cac } from "cac";

import {
  formatCredentialPathForDisplay,
  getFreshCredentials,
  loadCredentials,
} from "../auth/claude-oauth.js";
import { Borg, VERSION, loadConfig, redactConfig, type BorgOpenOptions } from "../index.js";
import { commitmentTypeSchema } from "../memory/commitments/index.js";
import {
  goalStatusSchema,
  growthMarkerCategorySchema,
  openQuestionStatusSchema,
} from "../memory/self/index.js";
import {
  reviewKindSchema,
  reviewResolutionSchema,
  semanticRelationSchema,
  semanticNodeKindSchema,
} from "../memory/semantic/index.js";
import {
  OFFLINE_PROCESS_NAMES,
  maintenancePlanSchema,
  type MaintenancePlan,
  type OfflineProcessName,
} from "../offline/index.js";
import { streamEntryKindSchema } from "../stream/index.js";
import { readJsonFile, writeJsonFileAtomic } from "../util/atomic-write.js";
import { BorgError } from "../util/errors.js";
import {
  DEFAULT_SESSION_ID,
  parseAutobiographicalPeriodId,
  parseAuditId,
  parseCommitmentId,
  parseEpisodeId,
  parseGoalId,
  parseOpenQuestionId,
  parseMaintenanceRunId,
  parseSemanticNodeId,
  parseSessionId,
  parseSkillId,
  parseValueId,
} from "../util/ids.js";

type Output = Pick<NodeJS.WriteStream, "write">;

export type RunCliOptions = {
  stdout?: Output;
  stderr?: Output;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
  openBorg?: (options: BorgOpenOptions) => Promise<Borg>;
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

function resolveEpisodeVisibilityOptions(commandOptions: Record<string, unknown>): {
  audience?: string | null;
  crossAudience?: boolean;
} {
  const audience =
    typeof commandOptions.audience === "string"
      ? parseRequiredText(commandOptions.audience, "--audience")
      : undefined;
  const crossAudience = commandOptions.all === true;

  if (audience !== undefined && crossAudience) {
    throw new CliError("--audience cannot be combined with --all");
  }

  return {
    audience,
    crossAudience,
  };
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

function resolveSemanticNodeId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Semantic node id is required");
  }

  try {
    return parseSemanticNodeId(value);
  } catch (error) {
    throw new CliError(`Invalid semantic node id: ${value}`, {
      cause: error,
    });
  }
}

function resolveCommitmentId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Commitment id is required");
  }

  try {
    return parseCommitmentId(value);
  } catch (error) {
    throw new CliError(`Invalid commitment id: ${value}`, {
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

function resolveAutobiographicalPeriodId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Period id is required");
  }

  try {
    return parseAutobiographicalPeriodId(value);
  } catch (error) {
    throw new CliError(`Invalid period id: ${value}`, {
      cause: error,
    });
  }
}

function resolveOpenQuestionId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Open question id is required");
  }

  try {
    return parseOpenQuestionId(value);
  } catch (error) {
    throw new CliError(`Invalid open question id: ${value}`, {
      cause: error,
    });
  }
}

function resolveSkillId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Skill id is required");
  }

  try {
    return parseSkillId(value);
  } catch (error) {
    throw new CliError(`Invalid skill id: ${value}`, {
      cause: error,
    });
  }
}

function parseSinceToTimestamp(
  value: unknown,
  flag = "--since",
  nowMs = Date.now(),
): number | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(`${flag} must be a duration like 1h, "now", or an epoch milliseconds timestamp`);
  }

  const trimmed = value.trim();

  if (trimmed === "now") {
    return nowMs;
  }

  const absolute = Number(trimmed);

  if (Number.isFinite(absolute) && trimmed === String(absolute)) {
    return absolute;
  }

  const match = trimmed.match(/^(\d+)([smhd])$/);

  if (match === null) {
    throw new CliError(`${flag} must be a duration like 1h, "now", or an epoch milliseconds timestamp`);
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

function parseGrowthMarkerCategory(value: unknown) {
  const parsed = growthMarkerCategorySchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError(
      `--category must be one of: ${growthMarkerCategorySchema.options.join(", ")}`,
      {
        cause: parsed.error,
      },
    );
  }

  return parsed.data;
}

function parseOpenQuestionStatus(value: unknown) {
  const parsed = openQuestionStatusSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError(`--status must be one of: ${openQuestionStatusSchema.options.join(", ")}`, {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function parseSemanticNodeKind(value: unknown) {
  const parsed = semanticNodeKindSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--kind must be one of: concept, entity, proposition", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function parseSemanticRelation(value: unknown) {
  const parsed = semanticRelationSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--relation must be a supported semantic relation", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function parseCommitmentType(value: unknown) {
  const parsed = commitmentTypeSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--type must be one of: promise, boundary, rule, preference", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function parseReviewKind(value: unknown) {
  const parsed = reviewKindSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError(`--kind must be one of: ${reviewKindSchema.options.join(", ")}`, {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function parseReviewResolution(value: unknown) {
  const parsed = reviewResolutionSchema.safeParse(value);

  if (!parsed.success) {
    throw new CliError("--decision must be one of: keep_both, supersede, invalidate, dismiss", {
      cause: parsed.error,
    });
  }

  return parsed.data;
}

function parseIdList(
  value: unknown,
  itemParser: (value: string) => string,
  flag: string,
): string[] {
  if (value === undefined) {
    return [];
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(`${flag} must be a comma-separated list`);
  }

  return value
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .map((item) => itemParser(item));
}

function parseStringList(value: unknown, flag: string): string[] {
  return parseIdList(value, (item) => item, flag);
}

function parsePositiveInteger(value: unknown, flag: string): number {
  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (!Number.isInteger(candidate) || candidate <= 0) {
    throw new CliError(`${flag} must be a positive integer`);
  }

  return candidate;
}

function parseOptionalPositiveInteger(value: unknown, flag: string): number | undefined {
  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (
    candidate === undefined ||
    candidate === null ||
    candidate === false ||
    (typeof candidate === "number" && Number.isNaN(candidate)) ||
    (typeof candidate === "string" && candidate.trim() === "")
  ) {
    return undefined;
  }

  if (typeof candidate !== "number" && typeof candidate !== "string") {
    return undefined;
  }

  return parsePositiveInteger(candidate, flag);
}

function parseFiniteNumber(value: unknown, label: string): number {
  const candidate = Array.isArray(value) ? value.at(-1) : value;
  const numeric =
    typeof candidate === "number"
      ? candidate
      : typeof candidate === "string" && candidate.trim() !== ""
        ? Number(candidate)
        : Number.NaN;

  if (!Number.isFinite(numeric)) {
    throw new CliError(`${label} must be a finite number`);
  }

  return numeric;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseStakes(value: unknown): "low" | "medium" | "high" | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (value === "low" || value === "medium" || value === "high") {
    return value;
  }

  throw new CliError("--stakes must be one of: low, medium, high");
}

function parseBudget(value: unknown): number | undefined {
  if (value === undefined) {
    return undefined;
  }

  const candidate = Array.isArray(value) ? value.at(-1) : value;

  if (candidate === undefined) {
    return undefined;
  }

  if (typeof candidate === "string" && candidate.trim() !== "") {
    const parsed = Number(candidate);

    if (!Number.isInteger(parsed) || parsed <= 0) {
      throw new CliError("--budget must be a positive integer");
    }

    return parsed;
  }

  return parsePositiveInteger(candidate, "--budget");
}

function parseOptionalPath(value: unknown, flag: string): string | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError(`${flag} must be a file path`);
  }

  return value.trim();
}

function maskSecretTail(value: string): string {
  const trimmed = value.trim();
  const tail = trimmed.slice(-4);
  return `***${tail}`;
}

function formatExpiryDelta(ms: number): string {
  if (ms <= 0) {
    return "expired";
  }

  const totalMinutes = Math.floor(ms / 60_000);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;

  if (hours <= 0) {
    return `${minutes}m`;
  }

  return `${hours}h ${minutes}m`;
}

function parseOfflineProcessName(value: unknown) {
  if (typeof value !== "string" || !OFFLINE_PROCESS_NAMES.includes(value as never)) {
    throw new CliError(`--process must be one of: ${OFFLINE_PROCESS_NAMES.join(", ")}`);
  }

  return value as OfflineProcessName;
}

function parseOfflineProcessList(value: unknown) {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("--process must be a comma-separated list");
  }

  return [
    ...new Set(
      value
        .split(",")
        .map((item) => item.trim())
        .filter((item) => item.length > 0)
        .map((item) => parseOfflineProcessName(item)),
    ),
  ] satisfies OfflineProcessName[];
}

function resolveMaintenanceRunId(value: unknown) {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Maintenance run id is required");
  }

  try {
    return parseMaintenanceRunId(value);
  } catch (error) {
    throw new CliError(`Invalid maintenance run id: ${value}`, {
      cause: error,
    });
  }
}

function resolveAuditId(value: unknown) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new CliError("Audit id is required");
  }

  try {
    return parseAuditId(value);
  } catch (error) {
    throw new CliError(`Invalid audit id: ${value}`, {
      cause: error,
    });
  }
}

async function withBorg<T>(options: RunCliOptions, fn: (borg: Borg) => Promise<T>): Promise<T> {
  const openOptions: BorgOpenOptions = {
    env: options.env,
    dataDir: options.dataDir,
  };
  const borg = await (options.openBorg?.(openOptions) ?? Borg.open(openOptions));

  try {
    return await fn(borg);
  } finally {
    await borg.close();
  }
}

function readMaintenancePlan(planPath: string): MaintenancePlan {
  const rawPlan = readJsonFile<unknown>(planPath);

  if (rawPlan === undefined) {
    throw new CliError(`Plan file not found: ${planPath}`, {
      code: "CLI_NOT_FOUND",
    });
  }

  const parsed = maintenancePlanSchema.safeParse(rawPlan);

  if (!parsed.success) {
    throw new CliError(`Invalid maintenance plan file: ${planPath}`, {
      cause: parsed.error,
    });
  }

  return parsed.data;
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

  cli
    .command("auth <action>", "Inspect or refresh Anthropic authentication")
    .action(async (action: string) => {
      const config = loadConfig({ env, dataDir: options.dataDir });
      const apiKeyFromEnv = env.ANTHROPIC_API_KEY?.trim();
      const authTokenFromEnv = env.ANTHROPIC_AUTH_TOKEN?.trim();

      if (action === "status") {
        if (config.anthropic.auth !== "oauth" && apiKeyFromEnv) {
          writeLine(stdout, `api-key via env (ANTHROPIC_API_KEY=${maskSecretTail(apiKeyFromEnv)})`);
          return;
        }

        if (config.anthropic.auth !== "oauth" && config.anthropic.apiKey?.trim()) {
          writeLine(
            stdout,
            `api-key via config (anthropic.apiKey=${maskSecretTail(config.anthropic.apiKey)})`,
          );
          return;
        }

        if (config.anthropic.auth !== "api-key" && authTokenFromEnv) {
          writeLine(
            stdout,
            `oauth via env (ANTHROPIC_AUTH_TOKEN=${maskSecretTail(authTokenFromEnv)})`,
          );
          return;
        }

        const credentials = await getFreshCredentials({ env });

        if (credentials !== null) {
          writeLine(
            stdout,
            `oauth via ${formatCredentialPathForDisplay({ env })} (expires in ${formatExpiryDelta(
              credentials.expiresAt - Date.now(),
            )})`,
          );
          return;
        }

        writeLine(stdout, "no credentials detected (run 'claude /login' or set ANTHROPIC_API_KEY)");
        return;
      }

      if (action === "refresh") {
        if (config.anthropic.auth !== "oauth" && config.anthropic.apiKey?.trim()) {
          writeLine(stdout, "api-key auth active; no OAuth refresh needed");
          return;
        }

        if (config.anthropic.auth !== "api-key" && authTokenFromEnv) {
          writeLine(stdout, "oauth env token active; no shared credentials file to refresh");
          return;
        }

        if (loadCredentials({ env }) === null) {
          writeLine(
            stdout,
            "no credentials detected (run 'claude /login' or set ANTHROPIC_API_KEY)",
          );
          return;
        }

        const credentials = await getFreshCredentials({
          env,
          forceRefresh: true,
        });

        if (credentials === null) {
          throw new CliError("Failed to refresh Claude OAuth credentials", {
            code: "AUTH_REFRESH_FAILED",
          });
        }

        writeLine(
          stdout,
          `oauth refreshed via ${formatCredentialPathForDisplay({ env })} (expires in ${formatExpiryDelta(
            credentials.expiresAt - Date.now(),
          )})`,
        );
        return;
      }

      throw new CliError(`Unknown auth action: ${action}`);
    });

  cli
    .command("turn <message>", "Run one cognitive turn")
    .option("--session <id>", "Session id to use")
    .option("--audience <audience>", "Audience label for the stream")
    .option("--stakes <stakes>", "Turn stakes: low | medium | high")
    .action(async (message: string, commandOptions: Record<string, unknown>) => {
      const result = await withBorg(options, async (borg) =>
        borg.turn({
          userMessage: parseRequiredText(message, "<message>"),
          sessionId: resolveSessionId(commandOptions.session),
          audience:
            typeof commandOptions.audience === "string" ? commandOptions.audience : undefined,
          stakes: parseStakes(commandOptions.stakes),
        }),
      );

      writeLine(stdout, result.response);
      writeLine(
        stdout,
        `[mode=${result.mode}] [path=${result.path}] [tokens=${result.usage.input_tokens}/${result.usage.output_tokens}] [retrieved=${result.retrievedEpisodeIds.join(",") || "none"}]`,
      );
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
    .option("--until <duration>", "Relative duration like 1h, \"now\", or epoch ms")
    .option("--entities <list>", "Comma-separated participant/tag terms for entity retrieval")
    .option("--audience <name>", "Audience label for scoped visibility")
    .option("--all", "Search across all audiences")
    .action(
      async (action: string, arg: string | undefined, commandOptions: Record<string, unknown>) => {
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
      },
    );

  cli
    .command("dream [action]", "Run offline maintenance processes")
    .option("--dry-run", "Preview changes without applying them")
    .option("--budget <tokens>", "Override token budget")
    .option("--process <names>", "Comma-separated process names")
    .option("--max-questions <count>", "Limit questions per ruminator run", {
      type: [Number],
    })
    .option("--output <path>", "Write a generated dry-run plan to a file")
    .option("--plan <path>", "Maintenance plan file to apply")
    .action(async (action: string | undefined, commandOptions: Record<string, unknown>) => {
      const budget = parseBudget(commandOptions.budget);
      const processList = parseOfflineProcessList(commandOptions.process);
      const dryRun = commandOptions.dryRun === true;
      const maxQuestions = parseOptionalPositiveInteger(
        commandOptions.maxQuestions,
        "--max-questions",
      );
      const outputPath = parseOptionalPath(commandOptions.output, "--output");
      const planPath = parseOptionalPath(commandOptions.plan, "--plan");

      if (action === "apply") {
        if (planPath === undefined) {
          throw new CliError("--plan is required for dream apply");
        }

        if (outputPath !== undefined) {
          throw new CliError("--output cannot be used with dream apply");
        }

        if (dryRun) {
          throw new CliError("--dry-run cannot be used with dream apply");
        }

        const result = await withBorg(options, async (borg) =>
          borg.dream.apply(readMaintenancePlan(planPath)),
        );
        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      if (planPath !== undefined) {
        throw new CliError("--plan can only be used with dream apply");
      }

      if (outputPath !== undefined && !dryRun) {
        throw new CliError("--output requires --dry-run");
      }

      const selectedProcesses: OfflineProcessName[] | undefined =
        action === undefined
          ? processList
          : action === "consolidate"
            ? ["consolidator"]
            : action === "reflect"
              ? ["reflector"]
              : action === "curate"
                ? ["curator"]
                : action === "oversee"
                  ? ["overseer"]
                  : action === "ruminate"
                    ? ["ruminator"]
                    : action === "narrate"
                      ? ["self-narrator"]
                      : undefined;

      if (selectedProcesses === undefined) {
        throw new CliError(`Unknown dream action: ${action}`);
      }

      const processOverrides =
        maxQuestions === undefined || !selectedProcesses.includes("ruminator")
          ? undefined
          : {
              ruminator: {
                params: {
                  maxQuestionsPerRun: maxQuestions,
                },
              },
            };

      if (outputPath !== undefined) {
        const response = await withBorg(options, async (borg) => {
          const plan = await borg.dream.plan({
            budget,
            processes: selectedProcesses,
            processOverrides,
          });
          writeJsonFileAtomic(outputPath, plan);

          return {
            ...borg.dream.preview(plan),
            plan_path: outputPath,
          };
        });

        writeLine(stdout, JSON.stringify(response, null, 2));
        return;
      }

      const result = await withBorg(options, async (borg) => {
        return borg.dream({
          dryRun,
          budget,
          processes: selectedProcesses,
          processOverrides,
        });
      });

      writeLine(stdout, JSON.stringify(result, null, 2));
    });

  cli
    .command("audit <action> [arg]", "Inspect or revert maintenance audit entries")
    .option("--run-id <id>", "Filter by maintenance run id")
    .option("--process <name>", "Filter by maintenance process")
    .option("--reverted", "Only include reverted entries")
    .action(
      async (action: string, arg: string | undefined, commandOptions: Record<string, unknown>) => {
        if (action === "list") {
          const items = await withBorg(options, async (borg) =>
            borg.audit.list({
              runId: resolveMaintenanceRunId(commandOptions.runId ?? commandOptions["run-id"]),
              process:
                commandOptions.process === undefined
                  ? undefined
                  : parseOfflineProcessName(commandOptions.process),
              reverted: commandOptions.reverted === true ? true : undefined,
            }),
          );
          writeLine(stdout, JSON.stringify(items, null, 2));
          return;
        }

        if (action === "revert") {
          const item = await withBorg(options, async (borg) =>
            borg.audit.revert(resolveAuditId(arg)),
          );
          writeLine(stdout, JSON.stringify(item, null, 2));
          return;
        }

        throw new CliError(`Unknown audit action: ${action}`);
      },
    );

  cli
    .command("semantic <subject> <action> [arg]", "Manage semantic nodes, edges, and walks")
    .option("--kind <kind>", "Node kind")
    .option("--label <label>", "Node label")
    .option("--description <text>", "Node description")
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
        commandOptions: Record<string, unknown>,
      ) => {
        if (subject === "node") {
          if (action === "add") {
            const node = await withBorg(options, async (borg) =>
              borg.semantic.nodes.add({
                kind: parseSemanticNodeKind(commandOptions.kind),
                label: parseRequiredText(commandOptions.label, "--label"),
                description: parseRequiredText(commandOptions.description, "--description"),
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

  cli
    .command("period <action> [arg]", "Manage autobiographical periods")
    .option("--limit <count>", "Maximum number of periods", {
      default: 20,
      type: [Number],
    })
    .action(
      async (action: string, arg: string | undefined, commandOptions: Record<string, unknown>) => {
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
              narrative: "A new autobiographical period began.",
            }),
          );
          writeLine(stdout, JSON.stringify(period, null, 2));
          return;
        }

        if (action === "close") {
          const periodId = resolveAutobiographicalPeriodId(arg);
          await withBorg(options, async (borg) => {
            borg.self.autobiographical.closePeriod(periodId, Date.now());
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
      },
    );

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
        commandOptions: Record<string, unknown>,
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
        commandOptions: Record<string, unknown>,
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
              source: "user",
            }),
          );
          writeLine(stdout, JSON.stringify(question, null, 2));
          return;
        }

        if (action === "resolve") {
          const questionId = resolveOpenQuestionId(arg1);
          const resolved = await withBorg(options, async (borg) =>
            borg.self.openQuestions.resolve(questionId, {
              resolution_episode_id: resolveEpisodeId(commandOptions.episode),
              resolution_note:
                commandOptions.note === undefined
                  ? null
                  : parseRequiredText(commandOptions.note, "--note"),
            }),
          );
          writeLine(stdout, JSON.stringify(resolved, null, 2));
          return;
        }

        if (action === "abandon") {
          const questionId = resolveOpenQuestionId(arg1);
          const abandoned = await withBorg(options, async (borg) =>
            borg.self.openQuestions.abandon(
              questionId,
              typeof commandOptions.reason === "string"
                ? commandOptions.reason
                : "Abandoned from CLI",
            ),
          );
          writeLine(stdout, JSON.stringify(abandoned, null, 2));
          return;
        }

        if (action === "bump") {
          const questionId = resolveOpenQuestionId(arg1);
          const bumped = await withBorg(options, async (borg) =>
            borg.self.openQuestions.bumpUrgency(questionId, parseFiniteNumber(arg2, "<delta>")),
          );
          writeLine(stdout, JSON.stringify(bumped, null, 2));
          return;
        }

        throw new CliError(`Unknown question action: ${action}`);
      },
    );

  cli
    .command("skill <action> [arg]", "Manage procedural skills")
    .option("--applies-when <text>", "Context where the skill applies")
    .option("--approach <text>", "Suggested approach")
    .option("--episode <id>", "Source episode id")
    .option("--limit <count>", "Maximum number of results", {
      default: 10,
      type: [Number],
    })
    .action(
      async (action: string, arg: string | undefined, commandOptions: Record<string, unknown>) => {
        if (action === "add") {
          const skill = await withBorg(options, async (borg) =>
            borg.skills.add({
              applies_when: parseRequiredText(
                commandOptions.appliesWhen ?? commandOptions["applies-when"],
                "--applies-when",
              ),
              approach: parseRequiredText(commandOptions.approach, "--approach"),
              sourceEpisodes: [resolveEpisodeId(commandOptions.episode)],
            }),
          );
          writeLine(stdout, JSON.stringify(skill, null, 2));
          return;
        }

        if (action === "list") {
          const skills = await withBorg(options, async (borg) =>
            borg.skills.list(parseLimit(commandOptions.limit)),
          );
          writeLine(stdout, JSON.stringify(skills, null, 2));
          return;
        }

        if (action === "show") {
          const skill = await withBorg(options, async (borg) =>
            borg.skills.get(resolveSkillId(arg)),
          );

          if (skill === null) {
            throw new CliError(`Skill not found: ${arg}`, {
              code: "CLI_NOT_FOUND",
            });
          }

          writeLine(stdout, JSON.stringify(skill, null, 2));
          return;
        }

        if (action === "select") {
          const result = await withBorg(options, async (borg) =>
            borg.skills.select(parseRequiredText(arg, "<context>"), {
              k: parseLimit(commandOptions.limit),
            }),
          );
          writeLine(stdout, JSON.stringify(result, null, 2));
          return;
        }

        throw new CliError(`Unknown skill action: ${action}`);
      },
    );

  cli
    .command("mood <action>", "Inspect affective mood state")
    .option("--session <id>", "Session id")
    .option("--since <duration>", "Relative duration like 1h or epoch ms")
    .option("--until <duration>", "Relative duration like 1h or epoch ms")
    .action(async (action: string, commandOptions: Record<string, unknown>) => {
      const sessionId = resolveSessionId(commandOptions.session);

      if (action === "current") {
        const mood = await withBorg(options, async (borg) => borg.mood.current(sessionId));
        writeLine(stdout, JSON.stringify(mood, null, 2));
        return;
      }

      if (action === "history") {
        const history = await withBorg(options, async (borg) =>
          borg.mood.history(sessionId, {
            fromTs: parseSinceToTimestamp(commandOptions.since, "--since"),
            toTs: parseSinceToTimestamp(commandOptions.until, "--until"),
          }),
        );
        writeLine(stdout, JSON.stringify(history, null, 2));
        return;
      }

      throw new CliError(`Unknown mood action: ${action}`);
    });

  cli
    .command("social <action> <entity> [delta]", "Manage social profiles")
    .action(
      async (
        action: string,
        entity: string | undefined,
        delta: string | undefined,
        _commandOptions: Record<string, unknown>,
      ) => {
        const entityName = parseRequiredText(entity, "<entity>");

        if (action === "profile") {
          const profile = await withBorg(options, async (borg) =>
            borg.social.getProfile(entityName),
          );
          writeLine(stdout, JSON.stringify(profile, null, 2));
          return;
        }

        if (action === "upsert") {
          const profile = await withBorg(options, async (borg) =>
            borg.social.upsertProfile(entityName),
          );
          writeLine(stdout, JSON.stringify(profile, null, 2));
          return;
        }

        if (action === "adjust-trust") {
          const profile = await withBorg(options, async (borg) =>
            borg.social.adjustTrust(entityName, parseFiniteNumber(delta, "<delta>")),
          );
          writeLine(stdout, JSON.stringify(profile, null, 2));
          return;
        }

        throw new CliError(`Unknown social action: ${action}`);
      },
    );

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
    .action(
      async (action: string, arg: string | undefined, commandOptions: Record<string, unknown>) => {
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
              sourceEpisodeIds: parseIdList(
                commandOptions.sourceEpisodes ?? commandOptions["source-episodes"],
                (value) => parseEpisodeId(value),
                "--source-episodes",
              ) as ReturnType<typeof parseIdList> as never,
            }),
          );
          writeLine(stdout, JSON.stringify(commitment, null, 2));
          return;
        }

        if (action === "revoke") {
          const revoked = await withBorg(options, async (borg) =>
            borg.commitments.revoke(resolveCommitmentId(arg)),
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
      },
    );

  cli
    .command("review <action> [arg1] [arg2]", "Inspect and resolve review items")
    .option("--kind <kind>", "Review item kind")
    .action(
      async (
        action: string,
        arg1: string | undefined,
        arg2: string | undefined,
        commandOptions: Record<string, unknown>,
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

          const resolved = await withBorg(options, async (borg) =>
            borg.review.resolve(itemId, parseReviewResolution(arg2)),
          );
          writeLine(stdout, JSON.stringify(resolved, null, 2));
          return;
        }

        throw new CliError(`Unknown review action: ${action}`);
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
    .action(async (action: string, commandOptions: Record<string, unknown>) => {
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
