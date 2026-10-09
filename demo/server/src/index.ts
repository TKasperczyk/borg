import { createDemoShutdownHandler } from "./shutdown.js";
import { pathToFileURL } from "node:url";

import { serve } from "@hono/node-server";
import {
  Borg,
  DemoMessageConnector,
  compositeTracer,
  createTurnTracer,
  loadConfig,
  type MessageConnector,
} from "borg";

import {
  createDemoServerApp,
  ensureDemoDefaultSession,
  runtimeConfigFromConfig,
  serializeStreamEntries,
  wireMaintenanceSchedulerLiveObserver,
} from "./app.js";
import { wireAutonomySchedulerObserver } from "./autonomy-observer.js";
import { createLiveBridge } from "./live.js";
import { MemorySampler, readMemorySamplerConfig } from "./memory.js";
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

const configuredDataDir = process.env.BORG_DATA_DIR ?? ".borg-data/demo";
const demoConfig = loadConfig({ dataDir: configuredDataDir });
const dataDir = demoConfig.dataDir;
const demoCreatorEntityName = process.env.DEMO_CREATOR_ENTITY_NAME ?? undefined;
const port = readPort();
const live = createLiveBridge();
const hostTracer = compositeTracer([createTurnTracer({ env: process.env }), live.tracer]);
const memorySamplerConfig = readMemorySamplerConfig();
const connectorPlugin = await loadExternalConnectorPlugin();

let shutdownRequested = false;

async function openDemoBorg(): Promise<Borg> {
  if (shutdownRequested) throw new Error("Borg is shutting down");
  const borg = await Borg.open({
    config: demoConfig,
    tracer: live.tracer,
    onStreamAppend: live.onStreamAppend,
    outboundConnectors: [
      new DemoMessageConnector(),
      ...(connectorPlugin?.outboundConnectors ?? []),
    ],
  });
  if (shutdownRequested) {
    await borg.close();
    throw new Error("Borg is shutting down");
  }
  ensureDemoDefaultSession(borg, { demoCreatorEntityName });
  // Attach before start, including when reset opens a replacement Borg.
  wireAutonomySchedulerObserver(borg.autonomy.scheduler, hostTracer);
  return borg;
}

const borgHandle: BorgHandle = {
  current: await openDemoBorg(),
};
live.setStreamEntrySerializer((entries) => serializeStreamEntries(borgHandle.current, entries));
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

const memorySampler = new MemorySampler({
  config: memorySamplerConfig,
  getStorage: () =>
    borgHandle.state === "closing" || borgHandle.state === "dead"
      ? null
      : borgHandle.current.maintenance,
  getActivity: () => ({
    ...live.broadcaster.memoryStats(),
    ledgerCacheEntries: live.ledgerCache.size,
  }),
  tracer: hostTracer,
});

const { app, injectWebSocket, beginShutdown } = createDemoServerApp({
  memorySampler,
  borgHandle,
  live,
  corsOrigins: readCorsOrigins(),
  resetBorg,
  demoCreatorEntityName,
  runtimeConfig: runtimeConfigFromConfig(demoConfig),
});
const server = serve({
  fetch: app.fetch,
  port,
});

injectWebSocket(server);
memorySampler.start();

console.log(`Borg demo server listening on http://localhost:${port}`);

const shutdown = createDemoShutdownHandler({
  beginDrain: async () => {
    shutdownRequested = true;
    // Every admission path closes before the first await, including plugin-held Borg handles.
    const httpDrain = beginShutdown();
    const borgDrain = borgHandle.current.beginShutdown();
    const results = await Promise.allSettled(
      [
        () => httpDrain,
        () => borgDrain,
        () => live.broadcaster.closeAll(),
        () => memorySampler.stop(),
        () => connectorPlugin?.stop(),
        () => new Promise<void>((resolve) => server.close(() => resolve())),
      ].map(async (stop) => await stop()),
    );
    const errors = results.flatMap((result) =>
      result.status === "rejected" ? [result.reason] : [],
    );
    if (errors.length > 0) throw new AggregateError(errors, "Borg drain failed");
  },
  closeStores: async () => {
    if (borgHandle.state !== "dead") {
      borgHandle.state = "closing";
      // All draining used the first deadline. Never enter it again after timeout.
      await borgHandle.current.close({ drain: false });
      borgHandle.state = "dead";
    }
  },
  forceCloseConnections: () => {
    (server as typeof server & { closeAllConnections?: () => void }).closeAllConnections?.();
  },
  exit: (code) => process.exit(code),
});

process.on("SIGINT", (signal) => {
  void shutdown(signal);
});
process.on("SIGTERM", (signal) => {
  void shutdown(signal);
});
