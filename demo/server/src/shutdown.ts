import { settleBeforeDeadline } from "borg";

// A signal sent to the process group also reaches pnpm and the tsx CLI, which
// relay it here again (tsx after ~30 ms); that copy is not a second request.
const RELAYED_SIGNAL_WINDOW_MS = 1_000;

export function createDemoShutdownHandler(options: {
  beginDrain: () => Promise<void>;
  closeStores: () => Promise<void>;
  forceCloseConnections: () => void;
  exit: (code: number) => void;
  logger?: Pick<Console, "log" | "error">;
}) {
  let first: { signal: NodeJS.Signals; at: number } | undefined;
  const logger = options.logger ?? console;
  return async (signal: NodeJS.Signals): Promise<void> => {
    if (first) {
      if (signal === first.signal && Date.now() - first.at < RELAYED_SIGNAL_WINDOW_MS) return;
      options.exit(1);
      return;
    }
    first = { signal, at: Date.now() };
    logger.log(`Received ${signal}; draining Borg`);
    let exitCode = 0;
    for (const phase of [
      { name: "drain", budgetMs: 75_000, run: options.beginDrain },
      { name: "store close", budgetMs: 10_000, run: options.closeStores },
    ]) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const deadline = new Promise<void>((resolve) => {
        timer = setTimeout(resolve, phase.budgetMs);
      });
      const result = await settleBeforeDeadline(phase.run, deadline);
      clearTimeout(timer);
      if (result.status !== "completed") {
        exitCode = 1;
        logger.error(
          `Borg shutdown ${phase.name}: ${result.status}`,
          result.status === "error" ? result.error : "deadline reached",
        );
      }
      if (phase.name === "drain") {
        try {
          options.forceCloseConnections();
        } catch (error) {
          exitCode = 1;
          logger.error("Borg shutdown connection close failed", error);
        }
      }
    }
    options.exit(exitCode);
  };
}
