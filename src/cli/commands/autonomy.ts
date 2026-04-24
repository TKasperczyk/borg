// Autonomy maintenance CLI commands for dream process runs and audit inspection.
import type { CAC } from "cac";

import {
  maintenancePlanSchema,
  type MaintenancePlan,
  type OfflineProcessName,
} from "../../offline/index.js";
import { readJsonFile, writeJsonFileAtomic } from "../../util/atomic-write.js";
import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import { resolveAuditId, resolveMaintenanceRunId } from "../helpers/id-resolvers.js";
import {
  parseBudget,
  parseOfflineProcessList,
  parseOfflineProcessName,
  parseOptionalPath,
  parseOptionalPositiveInteger,
} from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

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

export function registerAutonomyCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

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
    .action(async (action: string | undefined, commandOptions: CommandOptions) => {
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
    .action(async (action: string, arg: string | undefined, commandOptions: CommandOptions) => {
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
    });

  cli
    .command("maintenance tick", "Fire a single maintenance cadence (light or heavy)")
    .option("--cadence <name>", "Cadence to run: light or heavy", { default: "light" })
    .action(async (commandOptions: CommandOptions) => {
      const cadence = commandOptions.cadence;

      if (cadence !== "light" && cadence !== "heavy") {
        throw new CliError(`Unknown maintenance cadence: ${String(cadence)} (expected light|heavy)`);
      }

      const result = await withBorg(options, async (borg) =>
        borg.maintenance.scheduler.tick(cadence),
      );

      writeLine(stdout, JSON.stringify(result, null, 2));
    });
}
