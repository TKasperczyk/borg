import { promises as fs } from "node:fs";
import { join } from "node:path";

import { Borg, DemoMessageConnector } from "borg";

import type { LiveBridge } from "./live.js";

export type BorgHandle = {
  current: Borg;
  state?: BorgHandleState;
};

export type BorgHandleState = "open" | "closing" | "dead";

export type ResetBorgControllerOptions = {
  dataDir: string;
  live: LiveBridge;
  borgHandle: BorgHandle;
  openBorg?: () => Promise<Borg>;
};

export class ResetInProgressError extends Error {
  constructor() {
    super("Reset already in progress");
    this.name = "ResetInProgressError";
  }
}

async function wipeDataDirContents(dir: string): Promise<void> {
  let entries: string[];
  try {
    entries = await fs.readdir(dir);
  } catch (error: unknown) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return;
    }
    throw error;
  }

  await Promise.all(
    entries.map((name) => fs.rm(join(dir, name), { recursive: true, force: true })),
  );
}

async function closeHandleForReset(handle: BorgHandle): Promise<void> {
  if (handle.state === "dead") {
    return;
  }

  handle.state = "closing";
  try {
    await handle.current.autonomy.scheduler.stop().catch(() => undefined);
    await handle.current.close();
  } finally {
    handle.state = "dead";
  }
}

export function createResetBorgController(
  options: ResetBorgControllerOptions,
): () => Promise<void> {
  const openBorg =
    options.openBorg ??
    (() =>
      Borg.open({
        dataDir: options.dataDir,
        tracer: options.live.tracer,
        onStreamAppend: options.live.onStreamAppend,
        outboundConnectors: [new DemoMessageConnector()],
      }));
  let resetting = false;

  return async (): Promise<void> => {
    if (resetting) {
      throw new ResetInProgressError();
    }

    resetting = true;
    try {
      options.live.broadcaster.clearAllSessionBuffers();
      await closeHandleForReset(options.borgHandle);
      await wipeDataDirContents(options.dataDir);
      const nextBorg = await openBorg();
      options.borgHandle.current = nextBorg;
      options.borgHandle.state = "open";
      nextBorg.inbox.catchUp.start();
      nextBorg.autonomy.scheduler.start();
      options.live.ledgerCache.clear();
      options.live.resetTraceState();
      options.live.broadcaster.broadcast({ type: "borg:reset", ts: Date.now() });
    } finally {
      resetting = false;
    }
  };
}
