import { pathToFileURL } from "node:url";

import {
  Borg,
  type MaintenanceCadence,
  type MaintenanceTickResult,
  type TickResult,
} from "../src/index.ts";

const GRACEFUL_SHUTDOWN_TIMEOUT_MS = 10_000;
const DAEMON_SIGNALS = ["SIGINT", "SIGTERM"] as const;

type DaemonSignal = (typeof DAEMON_SIGNALS)[number];

type SchedulerStopOptions = {
  graceful?: boolean;
};

type AutonomySchedulerLike = {
  isEnabled(): boolean;
  start(): void;
  stop(options?: SchedulerStopOptions): Promise<void>;
  setObserver(
    observer: {
      onTick?(result: TickResult): void | Promise<void>;
      onError?(error: unknown): void | Promise<void>;
    } | null,
  ): void;
};

type MaintenanceSchedulerLike = {
  isEnabled(): boolean;
  start(): void;
  stop(options?: SchedulerStopOptions): Promise<void>;
  setObserver(
    observer: {
      onTick?(result: MaintenanceTickResult): void | Promise<void>;
      onError?(error: unknown, cadence: MaintenanceCadence): void | Promise<void>;
    } | null,
  ): void;
};

export type DaemonBorg = {
  autonomy: {
    scheduler: AutonomySchedulerLike;
  };
  maintenance: {
    scheduler: MaintenanceSchedulerLike;
  };
  close(): Promise<void>;
};

export type DaemonRunResult = {
  status: "started" | "disabled";
  shutdown(reason: string): Promise<void>;
};

export type RunDaemonOptions = {
  openBorg?: () => Promise<DaemonBorg>;
  writeStderr?: (line: string) => void;
  signalTarget?: {
    on(signal: DaemonSignal, listener: () => void): unknown;
  };
  exit?: (code?: number) => void;
  shutdownTimeoutMs?: number;
};

function writeStderr(line: string): void {
  process.stderr.write(`${line}\n`);
}

function formatTickResult(result: TickResult): string {
  return JSON.stringify(result);
}

function formatMaintenanceTickResult(result: MaintenanceTickResult): string {
  return JSON.stringify(result);
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
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

async function stopSchedulers(
  autonomyScheduler: AutonomySchedulerLike,
  maintenanceScheduler: MaintenanceSchedulerLike,
  timeoutMs: number,
  write: (line: string) => void,
): Promise<unknown[]> {
  const stopTasks = [
    {
      name: "autonomy",
      stop: () => autonomyScheduler.stop({ graceful: true }),
    },
    {
      name: "maintenance",
      stop: () => maintenanceScheduler.stop({ graceful: true }),
    },
  ] as const;

  const stopResult = await withTimeout(
    Promise.allSettled(stopTasks.map((task) => Promise.resolve().then(() => task.stop()))),
    timeoutMs,
  );

  if (typeof stopResult === "symbol") {
    write(`[daemon] scheduler stop timed out after ${timeoutMs}ms; closing anyway`);
    return [];
  }

  const errors: unknown[] = [];

  stopResult.forEach((result, index) => {
    if (result.status === "fulfilled") {
      return;
    }

    const task = stopTasks[index];

    if (task === undefined) {
      errors.push(result.reason);
      write(`[daemon] scheduler stop failed ${formatError(result.reason)}`);
      return;
    }

    errors.push(result.reason);
    write(`[daemon] ${task.name} scheduler stop failed ${formatError(result.reason)}`);
  });

  return errors;
}

export async function runDaemon(options: RunDaemonOptions = {}): Promise<DaemonRunResult> {
  const borg = await (options.openBorg?.() ?? Borg.open());
  const autonomyScheduler = borg.autonomy.scheduler;
  const maintenanceScheduler = borg.maintenance.scheduler;
  const write = options.writeStderr ?? writeStderr;
  const signalTarget = options.signalTarget ?? process;
  const exit = options.exit ?? ((code?: number) => process.exit(code));
  const shutdownTimeoutMs = options.shutdownTimeoutMs ?? GRACEFUL_SHUTDOWN_TIMEOUT_MS;
  let shuttingDown = false;

  const shutdown = async (reason: string) => {
    if (shuttingDown) {
      return;
    }

    shuttingDown = true;
    write(`[daemon] stopping: ${reason}`);

    const errors = await stopSchedulers(
      autonomyScheduler,
      maintenanceScheduler,
      shutdownTimeoutMs,
      write,
    );

    try {
      await borg.close();
    } catch (error) {
      errors.push(error);
      write(`[daemon] close failed ${formatError(error)}`);
    }

    if (errors.length > 0) {
      throw new AggregateError(errors, "Daemon shutdown completed with errors");
    }
  };

  autonomyScheduler.setObserver({
    onTick: (result) => {
      write(`[daemon] autonomy tick ${formatTickResult(result)}`);
    },
    onError: (error) => {
      write(`[daemon] autonomy tick-error ${formatError(error)}`);
    },
  });

  maintenanceScheduler.setObserver({
    onTick: (result) => {
      write(`[daemon] maintenance tick ${formatMaintenanceTickResult(result)}`);
    },
    onError: (error, cadence) => {
      write(`[daemon] maintenance tick-error cadence=${cadence} ${formatError(error)}`);
    },
  });

  for (const signal of DAEMON_SIGNALS) {
    signalTarget.on(signal, () => {
      void shutdown(signal).then(
        () => {
          exit(0);
        },
        (error) => {
          write(`[daemon] shutdown-error ${formatError(error)}`);
          exit(1);
        },
      );
    });
  }

  const autonomyEnabled = autonomyScheduler.isEnabled();
  const maintenanceEnabled = maintenanceScheduler.isEnabled();

  if (!autonomyEnabled && !maintenanceEnabled) {
    write("[daemon] autonomy and maintenance disabled; exiting");
    await borg.close();
    return {
      status: "disabled",
      shutdown,
    };
  }

  if (autonomyEnabled) {
    autonomyScheduler.start();
    write("[daemon] autonomy scheduler started");
  } else {
    write("[daemon] autonomy scheduler disabled");
  }

  if (maintenanceEnabled) {
    maintenanceScheduler.start();
    write("[daemon] maintenance scheduler started");
  } else {
    write("[daemon] maintenance scheduler disabled");
  }

  write("[daemon] started");

  return {
    status: "started",
    shutdown,
  };
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  runDaemon().catch((error) => {
    writeStderr(`[daemon] fatal ${formatError(error)}`);
    process.exitCode = 1;
  });
}
