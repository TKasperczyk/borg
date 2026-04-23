import { Borg, type TickResult } from "../src/index.ts";

const GRACEFUL_SHUTDOWN_TIMEOUT_MS = 10_000;

function writeStderr(line: string): void {
  process.stderr.write(`${line}\n`);
}

function formatTickResult(result: TickResult): string {
  return JSON.stringify(result);
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T | symbol> {
  const timeoutSymbol = Symbol("timeout");
  let timeoutHandle: NodeJS.Timeout | undefined;

  try {
    return await Promise.race([
      promise,
      new Promise<symbol>((resolve) => {
        timeoutHandle = setTimeout(() => {
          resolve(timeoutSymbol);
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timeoutHandle !== undefined) {
      clearTimeout(timeoutHandle);
    }
  }
}

async function main(): Promise<void> {
  const borg = await Borg.open();
  const scheduler = borg.autonomy.scheduler;
  let shuttingDown = false;

  const shutdown = async (reason: string) => {
    if (shuttingDown) {
      return;
    }

    shuttingDown = true;
    writeStderr(`[daemon] stopping: ${reason}`);

    const stopResult = await withTimeout(
      scheduler.stop({ graceful: true }),
      GRACEFUL_SHUTDOWN_TIMEOUT_MS,
    );

    if (typeof stopResult === "symbol") {
      writeStderr(
        `[daemon] scheduler stop timed out after ${GRACEFUL_SHUTDOWN_TIMEOUT_MS}ms; closing anyway`,
      );
    }

    await borg.close();
  };

  scheduler.setObserver({
    onTick: (result) => {
      writeStderr(`[daemon] tick ${formatTickResult(result)}`);
    },
    onError: (error) => {
      writeStderr(
        `[daemon] tick-error ${
          error instanceof Error ? `${error.name}: ${error.message}` : String(error)
        }`,
      );
    },
  });

  process.on("SIGINT", () => {
    void shutdown("SIGINT").finally(() => {
      process.exit(0);
    });
  });

  process.on("SIGTERM", () => {
    void shutdown("SIGTERM").finally(() => {
      process.exit(0);
    });
  });

  if (!scheduler.isEnabled()) {
    writeStderr("[daemon] autonomy disabled; exiting");
    await borg.close();
    return;
  }

  scheduler.start();
  writeStderr("[daemon] started");
}

main().catch((error) => {
  writeStderr(
    `[daemon] fatal ${error instanceof Error ? `${error.name}: ${error.message}` : String(error)}`,
  );
  process.exitCode = 1;
});
