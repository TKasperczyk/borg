import { tryAcquireFileLockGuard } from "../file-lock-guard.js";

// Root can write chmod(0444) files. Drop privileges after loading modules so
// this regression also exercises real permission denial in root-run CI.
if (process.getuid?.() === 0) {
  process.setgid!(65534);
  process.setuid!(65534);
}

let guard: ReturnType<typeof tryAcquireFileLockGuard> = null;
process.on("message", (action) => {
  if (action === "acquire") {
    try {
      guard = tryAcquireFileLockGuard(process.argv[2]!);
      process.send?.(guard === null ? "busy" : "acquired");
    } catch (error) {
      process.send?.(error instanceof Error ? error.message : String(error));
    }
  } else if (action === "release") {
    guard?.close();
    process.disconnect?.();
  }
});
process.send?.("ready");
