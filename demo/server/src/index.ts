import { pathToFileURL } from "node:url";

import { serve } from "@hono/node-server";
import { Borg, DemoMessageConnector, type MessageConnector } from "borg";

import {
  createDemoServerApp,
  ensureDemoDefaultSession,
  wireMaintenanceSchedulerLiveObserver,
} from "./app.js";
import { createLiveBridge } from "./live.js";
import { createResetBorgController, type BorgHandle } from "./reset.js";

// Generic hook: load an external connector plugin (e.g. a chat bridge living outside this
// repo) when EXTRA_CONNECTOR_MODULE is set. The plugin supplies outbound connectors to
// register at Borg.open and a start/stop lifecycle. The demo server stays agnostic to what
// the plugin does -- it never references any specific platform.
type ExternalConnectorPlugin = {
  outboundConnectors: MessageConnector[];
  start(ctx: {
    getBorg: () => Borg;
    log?: (level: string, message: string) => void;
  }): Promise<void>;
  stop(): Promise<void>;
};

async function loadExternalConnectorPlugin(): Promise<ExternalConnectorPlugin | null> {
  const modulePath = process.env.EXTRA_CONNECTOR_MODULE;
  if (modulePath === undefined || modulePath.trim() === "") {
    return null;
  }
  const mod = (await import(pathToFileURL(modulePath).href)) as {
    createDemoPlugin?: () => ExternalConnectorPlugin;
  };
  if (typeof mod.createDemoPlugin !== "function") {
    throw new Error(`EXTRA_CONNECTOR_MODULE ${modulePath} does not export createDemoPlugin()`);
  }
  return mod.createDemoPlugin();
}

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
const demoCreatorEntityName = process.env.DEMO_CREATOR_ENTITY_NAME ?? undefined;
const port = readPort();
const live = createLiveBridge();
const connectorPlugin = await loadExternalConnectorPlugin();

async function openDemoBorg(): Promise<Borg> {
  const borg = await Borg.open({
    dataDir,
    tracer: live.tracer,
    onStreamAppend: live.onStreamAppend,
    outboundConnectors: [
      new DemoMessageConnector(),
      ...(connectorPlugin?.outboundConnectors ?? []),
    ],
  });
  ensureDemoDefaultSession(borg, { demoCreatorEntityName });
  return borg;
}

const borgHandle: BorgHandle = {
  current: await openDemoBorg(),
};
borgHandle.current.inbox.catchUp.start();
// Run as a full autonomous runtime: the scheduler fires self-initiated wakes on its triggers
// (expiring commitments, dormant open questions, due goals, executive focus). Without this
// call borg never self-initiates. Proactive outbound during those wakes is separately gated.
borgHandle.current.autonomy.scheduler.start();
wireMaintenanceSchedulerLiveObserver(borgHandle.current, live);
borgHandle.current.maintenance.scheduler.start();

if (connectorPlugin) {
  try {
    await connectorPlugin.start({
      getBorg: () => borgHandle.current,
      log: (level, message) => console.log(`[connector] ${level}: ${message}`),
    });
  } catch (error) {
    console.error("external connector plugin start failed; shutting down borg", error);
    await connectorPlugin.stop().catch(() => undefined);
    await borgHandle.current.close().catch(() => undefined);
    throw error;
  }
}

const resetBorg = createResetBorgController({ dataDir, live, borgHandle, openBorg: openDemoBorg });

const { app, injectWebSocket } = createDemoServerApp({
  borgHandle,
  live,
  corsOrigins: readCorsOrigins(),
  resetBorg,
  demoCreatorEntityName,
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
  if (connectorPlugin) {
    await connectorPlugin.stop().catch((error: unknown) => {
      console.error("external connector plugin stop failed", error);
    });
  }
  if (borgHandle.state !== "dead" && borgHandle.state !== "closing") {
    await borgHandle.current.autonomy.scheduler.stop().catch(() => undefined);
    await borgHandle.current.maintenance.scheduler.stop().catch(() => undefined);
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
