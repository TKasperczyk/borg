import { runCli } from "./app.js";

const exitCode = await runCli(process.argv);

if (exitCode !== 0) {
  process.exitCode = exitCode;
}
