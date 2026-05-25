import { serve } from "@hono/node-server";
import { Borg } from "borg";

import { createDemoServerApp } from "./app.js";
import { createLiveBridge } from "./live.js";

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
const borg = await Borg.open({
  dataDir,
  tracer: live.tracer,
  onStreamAppend: live.onStreamAppend,
});
const { app, injectWebSocket } = createDemoServerApp({
  borg,
  live,
  corsOrigins: readCorsOrigins(),
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

  await new Promise<void>((resolve) => {
    server.close(() => resolve());
  });
  await borg.close();
};

process.once("SIGINT", (signal) => {
  void shutdown(signal).then(() => process.exit(0));
});
process.once("SIGTERM", (signal) => {
  void shutdown(signal).then(() => process.exit(0));
});
