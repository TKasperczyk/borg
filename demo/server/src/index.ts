import { serve } from "@hono/node-server";
import { Borg } from "borg";

import { createDemoServerApp, ensureDemoDefaultSession } from "./app.js";
import { createLiveBridge } from "./live.js";
import { createResetBorgController, type BorgHandle } from "./reset.js";

function readPort(): number {
  const raw = process.env.PORT ?? "7740";
  const port = Number.parseInt(raw, 10);

  if (!Number.isInteger(port) || port <= 0 || port > 65_535) {
    throw new Error(`Invalid PORT: ${raw}`);
  }

  return port;
}

function readCorsOrigins(): string[] {
  const configured = process.env.DEMO_CORS_ORIGINS ?? process.env.DEMO_CORS_ORIGIN;

  if (configured === undefined || configured.trim().length === 0) {
    return ["http://localhost:5173"];
  }

  return configured
    .split(",")
    .map((origin) => origin.trim())
    .filter((origin) => origin.length > 0);
}

const dataDir = process.env.BORG_DATA_DIR ?? ".borg-data/demo";
const port = readPort();
const live = createLiveBridge();

async function openDemoBorg(): Promise<Borg> {
  const borg = await Borg.open({
    dataDir,
    tracer: live.tracer,
    onStreamAppend: live.onStreamAppend,
  });
  ensureDemoDefaultSession(borg);
  return borg;
}

const borgHandle: BorgHandle = {
  current: await openDemoBorg(),
};

const resetBorg = createResetBorgController({ dataDir, live, borgHandle, openBorg: openDemoBorg });

const { app, injectWebSocket } = createDemoServerApp({
  borgHandle,
  live,
  corsOrigins: readCorsOrigins(),
  resetBorg,
});
const server = serve({
  fetch: app.fetch,
  port,
});

injectWebSocket(server);

console.log(`Borg demo server listening on http://localhost:${port}`);

let shuttingDown = false;
const shutdown = async (signal: NodeJS.Signals) => {
  if (shuttingDown) {
    return;
  }

  shuttingDown = true;
  console.log(`Received ${signal}; shutting down`);

  live.broadcaster.closeAll();
  const serverWithConnectionCloser = server as typeof server & {
    closeAllConnections?: () => void;
  };
  await new Promise<void>((resolve) => {
    const forceCloseTimer = setTimeout(() => {
      serverWithConnectionCloser.closeAllConnections?.();
      resolve();
    }, 5_000);

    server.close(() => {
      clearTimeout(forceCloseTimer);
      resolve();
    });
  });
  if (borgHandle.state !== "dead" && borgHandle.state !== "closing") {
    await borgHandle.current.close();
  }
};

function exitAfterShutdown(signal: NodeJS.Signals): void {
  void shutdown(signal)
    .then(() => process.exit(0))
    .catch((error: unknown) => {
      console.error("Borg demo server shutdown failed", error);
      process.exit(1);
    });
}

process.once("SIGINT", (signal) => {
  exitAfterShutdown(signal);
});
process.once("SIGTERM", (signal) => {
  exitAfterShutdown(signal);
});
