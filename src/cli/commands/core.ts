// Core CLI commands for version, auth, config inspection, and a single cognitive turn.
import type { CAC } from "cac";

import {
  formatCredentialPathForDisplay,
  getFreshCredentials,
  loadCredentials,
} from "../../auth/claude-oauth.js";
import { VERSION, loadConfig, redactConfig } from "../../index.js";
import { withBorg } from "../helpers/borg.js";
import { CliError } from "../helpers/errors.js";
import { formatExpiryDelta, maskSecretTail, writeLine } from "../helpers/formatters.js";
import { resolveSessionId } from "../helpers/id-resolvers.js";
import { parseRequiredText, parseStakes } from "../helpers/parsers.js";
import type { CliCommandDeps, CommandOptions } from "../types.js";

export function registerCoreCommands(cli: CAC, deps: CliCommandDeps): void {
  const { stdout, env, options } = deps;

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
    .action(async (message: string, commandOptions: CommandOptions) => {
      const result = await withBorg(options, async (borg) =>
        borg.turn({
          userMessage: parseRequiredText(message, "<message>"),
          sessionId: resolveSessionId(commandOptions.session),
          audience:
            typeof commandOptions.audience === "string" ? commandOptions.audience : undefined,
          stakes: parseStakes(commandOptions.stakes),
        }),
      );

      if (result.emitted) {
        writeLine(stdout, result.response);
      }

      const suppression =
        result.emission.kind === "suppressed" ? ` [suppression=${result.emission.reason}]` : "";
      writeLine(
        stdout,
        `[mode=${result.mode}] [path=${result.path}] [emitted=${result.emitted}]${suppression} [tokens=${result.usage.input_tokens}/${result.usage.output_tokens}] [retrieved=${result.retrievedEpisodeIds.join(",") || "none"}]`,
      );
    });

  cli.command("config <action>", "Inspect borg config").action((action: string) => {
    if (action !== "show") {
      throw new CliError(`Unknown config action: ${action}`);
    }

    const config = loadConfig({ env, dataDir: options.dataDir });
    writeLine(stdout, JSON.stringify(redactConfig(config), null, 2));
  });
}
