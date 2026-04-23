// Thin CLI entrypoint that wires cac, global flags, command registrars, and top-level errors.
import { cac } from "cac";

import { registerAffectiveCommands } from "./commands/affective.js";
import { registerAutonomyCommands } from "./commands/autonomy.js";
import { registerCommitmentCommands } from "./commands/commitments.js";
import { registerCoreCommands } from "./commands/core.js";
import { registerCorrectionCommands } from "./commands/correction.js";
import { registerEpisodeCommands } from "./commands/episodes.js";
import { registerReviewCommands } from "./commands/review.js";
import { registerSelfCommands } from "./commands/self.js";
import { registerSemanticCommands } from "./commands/semantic.js";
import { registerSocialCommands } from "./commands/social.js";
import { registerStreamCommands } from "./commands/stream.js";
import { registerToolCommands } from "./commands/tools.js";
import { writeLine } from "./helpers/formatters.js";
import { VERSION } from "../index.js";
import type { CliCommandDeps, RunCliOptions } from "./types.js";

export type { RunCliOptions } from "./types.js";

export async function runCli(argv: string[], options: RunCliOptions = {}): Promise<number> {
  const stdout = options.stdout ?? process.stdout;
  const stderr = options.stderr ?? process.stderr;
  const env = options.env ?? process.env;

  const cli = cac("borg");
  cli.help();
  cli.option("-v, --version", "Display version number");

  const deps: CliCommandDeps = {
    stdout,
    stderr,
    env,
    options,
  };

  registerCoreCommands(cli, deps);
  registerStreamCommands(cli, deps);
  registerEpisodeCommands(cli, deps);
  registerAutonomyCommands(cli, deps);
  registerSemanticCommands(cli, deps);
  registerSelfCommands(cli, deps);
  registerToolCommands(cli, deps);
  registerAffectiveCommands(cli, deps);
  registerSocialCommands(cli, deps);
  registerCommitmentCommands(cli, deps);
  registerReviewCommands(cli, deps);
  registerCorrectionCommands(cli, deps);

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
