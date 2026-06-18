// borg memory sidecar: a long-lived HTTP service exposing per-tenant long-term
// memory (one being per tenant via BorgPool) to an external consumer such as the
// Python "team-agent" service.
//
// Run: BORG_MEMORY_TOKEN=... LLM_API_KEY=aif-... NODE_EXTRA_CA_CERTS=/path/ca.pem \
//        tsx scripts/memory-sidecar.ts
//
// TLS to the kratos inference endpoint uses the process-level NODE_EXTRA_CA_CERTS
// (the endpoint is server-CA-only, no client cert), matching the embedding client.
import { createServer } from "node:http";

import OpenAI from "openai";

import {
  BorgPool,
  OpenAICompatibleEmbeddingClient,
  OpenAICompatibleLLMClient,
  type OpenAIChatCompletionsClient,
} from "../src/index.js";
import { createMemoryHandler } from "../src/sidecar/memory-handler.js";

function requireEnv(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === "") {
    console.error(`memory-sidecar: missing required env ${name}`);
    process.exit(1);
  }
  return value;
}

const token = requireEnv("BORG_MEMORY_TOKEN");
const apiKey = requireEnv("LLM_API_KEY");
const baseUrl = process.env.KRATOS_BASE_URL ?? "https://inference.kratos.omc.hdp.it.p4/v1";
const llmModel = process.env.LLM_MODEL ?? "generative-apis/qwen3-235b-a22b-instruct-2507";
const embeddingModel = process.env.EMBEDDING_MODEL ?? "generative-apis/qwen3-embedding-8b";
const embeddingDims = Number(process.env.EMBEDDING_DIMS ?? 4096);
const root = process.env.BORG_DATA_ROOT ?? "./data/borg";
const host = process.env.BORG_MEMORY_HOST ?? "127.0.0.1";
const port = Number(process.env.BORG_MEMORY_PORT ?? 8088);
const maxOpen = Number(process.env.BORG_MEMORY_MAX_OPEN ?? 32);
// Bound every provider call so a hung kratos can't pin a request + pool slot
// (and block shutdown) indefinitely.
const requestTimeoutMs = Number(process.env.BORG_MEMORY_LLM_TIMEOUT_MS ?? 120_000);
const shutdownTimeoutMs = Number(process.env.BORG_MEMORY_SHUTDOWN_MS ?? 15_000);

// borg picks the LLM model per cognition slot from config (which reads process.env
// at Borg.open). The memory paths touch extraction + recall-expansion; point them
// (and the rest, defensively) at the injected Qwen model unless already overridden.
for (const slot of [
  "BORG_MODEL_EXTRACTION",
  "BORG_MODEL_RECALL_EXPANSION",
  "BORG_MODEL_COGNITION",
  "BORG_MODEL_BACKGROUND",
  "BORG_MODEL_CREATOR_DIRECTIVE",
]) {
  process.env[slot] ??= llmModel;
}

// One timed OpenAI client shared by both adapters (stateless, tenant-independent;
// the timeout/retry bound applies to extraction AND embedding calls).
const openai = new OpenAI({ apiKey, baseURL: baseUrl, timeout: requestTimeoutMs, maxRetries: 1 });
const llmClient = new OpenAICompatibleLLMClient({
  client: openai as unknown as OpenAIChatCompletionsClient,
});
const embeddingClient = new OpenAICompatibleEmbeddingClient({
  client: openai,
  model: embeddingModel,
  dims: embeddingDims,
});

const pool = new BorgPool({
  root,
  maxOpen,
  openOptions: { embeddingDimensions: embeddingDims, embeddingClient, llmClient },
});

const server = createServer(createMemoryHandler({ pool, token }));
server.listen(port, host, () => {
  console.log(`borg memory sidecar listening on ${host}:${port} (root=${root}, maxOpen=${maxOpen})`);
});

let shuttingDown = false;
function closeServer(): Promise<void> {
  return new Promise((resolve) => server.close(() => resolve()));
}
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms).unref();
  });
}

async function shutdown(signal: string): Promise<void> {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  console.log(`memory-sidecar: ${signal} received, draining in-flight requests...`);
  server.closeIdleConnections();
  // Stop accepting and let accepted requests finish (their withTenant completes),
  // bounded by a hard timeout so a wedged connection can't block forever.
  await Promise.race([closeServer(), delay(shutdownTimeoutMs)]);
  try {
    // shutdown() (not closeAll()) is a barrier: it rejects any new withTenant so a
    // request still in readBody can't open a being that escapes this drain and gets
    // killed mid-write at process.exit. It drains in-flight ops before closing.
    await pool.shutdown();
  } catch (error) {
    console.error("memory-sidecar: error during pool shutdown", error);
  }
  process.exit(0);
}

process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));
