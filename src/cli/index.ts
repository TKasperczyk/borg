// Borg CLI entry. Kept intentionally thin: the CLI drives library APIs
// for operational tasks (ingest, retrieve, dream, consolidate, inspect).
// Individual subcommands are added by subsequent sprints.

import { VERSION } from "../index.js";

function main(argv: string[]): void {
  const [, , command] = argv;
  if (command === "version" || command === "--version" || command === "-v") {
    process.stdout.write(`borg ${VERSION}\n`);
    return;
  }
  process.stdout.write(
    `borg ${VERSION}\n` +
      `\n` +
      `No subcommands registered yet. See ARCHITECTURE.md for the build plan.\n`,
  );
}

main(process.argv);
