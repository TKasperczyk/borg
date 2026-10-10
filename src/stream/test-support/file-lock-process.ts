// Real process/OS-lock coverage; each child simulates a different pod hostname.
import os from "node:os";
import { syncBuiltinESMExports } from "node:module";

os.hostname = () => process.argv[3]!;
syncBuiltinESMExports();
const { acquireFileLockLease, isFileLockLive } = await import("../file-lock.js");
let lease: Awaited<ReturnType<typeof acquireFileLockLease>> | undefined;
const realNow = Date.now;
let clockOffset = 0;
Date.now = () => realNow() + clockOffset;

process.on("message", async (action: string | { advanceMs: number }) => {
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
  } else if (action === "observe") {
    process.send?.(isFileLockLive(process.argv[2]!));
  } else if (typeof action === "object") {
    clockOffset += action.advanceMs;
    process.send?.("advanced");
  }
});
process.send?.("ready");
