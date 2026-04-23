// Correction and identity-audit CLI commands for forgetting, explaining, patching, and event listing.
import type { CAC } from "cac";

import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { writeLine } from "../helpers/formatters.js";
import {
  parseIdentityRecordType,
  parseJsonObject,
  parseLimit,
  parseRequiredText,
} from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerCorrectionCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, options } = deps;

  cli
    .command("correction <action> [arg]", "Manage corrections and identity audit")
    .option("--patch <json>", "JSON patch object")
    .option("--entity <name>", "Entity name")
    .option("--limit <count>", "Maximum number of events", {
      default: 20,
      type: [Number],
    })
    .option("--record-type <type>", "Identity event record type")
    .option("--record-id <id>", "Identity event record id")
    .action(async (action: string, arg: string | undefined, commandOptions: CommandOptions) => {
      if (action === "forget") {
        const result = await withBorg(options, async (borg) =>
          borg.correction.forget(parseRequiredText(arg, "<id>")),
        );
        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      if (action === "why") {
        const result = await withBorg(options, async (borg) =>
          borg.correction.why(parseRequiredText(arg, "<id>")),
        );
        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      if (action === "correct") {
        const result = await withBorg(options, async (borg) =>
          borg.correction.correct(
            parseRequiredText(arg, "<id>"),
            parseJsonObject(commandOptions.patch, "--patch"),
          ),
        );
        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      if (action === "about-me") {
        const result = await withBorg(options, async (borg) =>
          borg.correction.rememberAboutMe({
            entity: typeof commandOptions.entity === "string" ? commandOptions.entity : undefined,
          }),
        );
        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      if (action === "events") {
        const result = await withBorg(options, async (borg) =>
          borg.correction.listIdentityEvents({
            limit: parseLimit(commandOptions.limit),
            recordType:
              commandOptions.recordType === undefined && commandOptions["record-type"] === undefined
                ? undefined
                : parseIdentityRecordType(
                    commandOptions.recordType ?? commandOptions["record-type"],
                  ),
            recordId:
              typeof (commandOptions.recordId ?? commandOptions["record-id"]) === "string"
                ? String(commandOptions.recordId ?? commandOptions["record-id"])
                : undefined,
          }),
        );
        writeLine(stdout, JSON.stringify(result, null, 2));
        return;
      }

      throw new CliError(`Unknown correction action: ${action}`);
    });
}
