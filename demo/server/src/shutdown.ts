import { settleBeforeDeadline } from "borg";
import type { BorgHandle } from "./reset.js";

// A signal sent to the process group also reaches pnpm and the tsx CLI, which
// relay it here again (tsx after ~30 ms); that copy is not a second request.
const RELAYED_SIGNAL_WINDOW_MS = 1_000;

export function createDemoShutdownHandler(options: {
  beginDrain: (signal: AbortSignal) => Promise<void>;
  closeStores: (outcome: { drained: boolean }) => Promise<void>;
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
    let drained = false;
    const drainAbort = new AbortController();
    for (const phase of [
      { name: "drain", budgetMs: 75_000, run: () => options.beginDrain(drainAbort.signal) },
      { name: "store close", budgetMs: 10_000, run: () => options.closeStores({ drained }) },
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
        drained = result.status === "completed";
        // A timed-out stage may settle later. Prevent it from starting another
        // drain stage after we have moved on to forced store closure.
        drainAbort.abort();
        try {
          options.forceCloseConnections();
        } catch (error) {
          drained = false;
          exitCode = 1;
          logger.error("Borg shutdown connection close failed", error);
        }
      }
    }
    options.exit(exitCode);
  };
}

/** The production host ordering, shared with the HTTP/Borg integration tests. */
export function createDemoRuntimeShutdownHandler(options: {
  borgHandle: BorgHandle;
  onShutdownRequested: () => void;
  beginHttpDrain: () => Promise<void>;
  stopHost: readonly (() => void | Promise<void>)[];
  stopConnector?: () => Promise<void>;
  forceCloseConnections: () => void;
  exit: (code: number) => void;
  logger?: Pick<Console, "log" | "error">;
}) {
  return createDemoShutdownHandler({
    beginDrain: async (signal) => {
      options.onShutdownRequested();
      // No await before BOTH gates close. Facade-using HTTP handlers are covered
      // by their host leases; connector-held turn/enqueue handles close here too.
      const httpDrain = options.beginHttpDrain();
      options.borgHandle.current.closeAdmission();
      const results = await Promise.allSettled([
        httpDrain,
        ...options.stopHost.map(async (stop) => await stop()),
      ]);
      const errors = results.flatMap((result) =>
        result.status === "rejected" ? [result.reason] : [],
      );
      if (signal.aborted) return;
      try {
        await options.stopConnector?.();
      } catch (error) {
        errors.push(error);
      }
      if (signal.aborted) return;
      try {
        await options.borgHandle.current.beginShutdown();
      } catch (error) {
        errors.push(error);
      }
      if (errors.length > 0) throw new AggregateError(errors, "Borg host drain failed");
    },
    closeStores: async ({ drained }) => {
      if (options.borgHandle.state !== "dead") {
        options.borgHandle.state = "closing";
        await options.borgHandle.current.close({ drain: false, retainBankLease: !drained });
        options.borgHandle.state = "dead";
      }
    },
    forceCloseConnections: options.forceCloseConnections,
    exit: options.exit,
    logger: options.logger,
  });
}
