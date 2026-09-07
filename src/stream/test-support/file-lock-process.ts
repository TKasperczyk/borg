// Real process/OS-lock coverage; each child simulates a different pod hostname.
import os from "node:os";
import { syncBuiltinESMExports } from "node:module";

os.hostname = () => process.argv[3]!;
syncBuiltinESMExports();
const { acquireFileLockLease } = await import("../file-lock.js");
let lease: Awaited<ReturnType<typeof acquireFileLockLease>> | undefined;

process.on("message", async (action) => {
  if (action === "acquire") {
    try {
      lease = await acquireFileLockLease(process.argv[2]!, { timeoutMs: 0 });
      process.send?.("acquired");
    } catch (error) {
      process.send?.(error instanceof Error ? error.message : String(error));
    }
  } else if (action === "release") {
    await lease?.release();
    process.disconnect?.();
  }
});
process.send?.("ready");
