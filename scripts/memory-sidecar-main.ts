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

import { createCachingEmbeddingClient } from "../src/embeddings/cache.js";
import { StallGuardEmbeddingClient } from "../src/embeddings/stall-guard.js";
import {
  BorgPool,
  OpenAICompatibleEmbeddingClient,
  OpenAICompatibleLLMClient,
  loadConfig,
  type OpenAIChatCompletionsClient,
} from "../src/index.js";
import { createMemoryHandler } from "../src/sidecar/memory-handler.js";
import {
  memoryCommitmentExtractionBudgetFromEnv,
  memoryCommitmentExtractionEnabledFromEnv,
} from "../src/sidecar/memory-commitment-extraction.js";
import {
  MemoryMaintenanceCoordinator,
  memoryMaintenanceConfigFromConfig,
  memorySelfNameFromEnv,
} from "../src/sidecar/memory-maintenance.js";
import { drainMemorySidecar } from "../src/sidecar/memory-sidecar-shutdown.js";
import {
  MemoryTraceRegistry,
  memoryTraceCapacityFromEnv,
  memoryTraceEnabledFromEnv,
  memoryTraceMaxTenantsFromEnv,
} from "../src/sidecar/memory-trace.js";

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
const recallAbstainThresholdRaw = Number(process.env.BORG_RECALL_ABSTAIN_THRESHOLD ?? 0);
const recallAbstainThreshold = Number.isFinite(recallAbstainThresholdRaw)
  ? recallAbstainThresholdRaw
  : 0;
// Hard ceiling on a single recall, so the client's own timeout never fires
// first and downgrades a structured degradation into an opaque transport
// error. Keep it below the caller's recall timeout (team-agent:
// memory.recall_timeout).
const recallDeadlineMsRaw = Number(process.env.BORG_RECALL_DEADLINE_MS ?? 4000);
const recallDeadlineMs = Number.isFinite(recallDeadlineMsRaw) ? recallDeadlineMsRaw : 4000;
// Bound every provider call so a hung kratos can't pin a request + pool slot
// (and block shutdown) indefinitely.
const requestTimeoutMs = Number(process.env.BORG_MEMORY_LLM_TIMEOUT_MS ?? 120_000);
// Tight per-attempt stall guard for embedding calls (see StallGuardEmbeddingClient).
const embeddingStallTimeoutMs = Number(process.env.BORG_EMBEDDING_STALL_TIMEOUT_MS ?? 1000);
const embeddingStallBatchTimeoutMs = Number(
  process.env.BORG_EMBEDDING_STALL_BATCH_TIMEOUT_MS ?? 20_000,
);
const embeddingStallRetries = Number(process.env.BORG_EMBEDDING_STALL_RETRIES ?? 1);
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
// Wrapped in the same LRU cache Borg.open would apply to its own client
// (borg/clients.ts): injecting a bare client here bypassed it, so every recall
// re-embedded identical intent queries (and each tenant's active commitment
// directives) over the network — ~24 embedding calls per /memory/recall.
// One cache instance is shared by all tenants; keys are model+dims+text.
// Stall guard inside the cache: cache hits never race a timer, misses get a
// short per-attempt cap + retry instead of inheriting the 120s client bound.
// The gateway's embedding backend intermittently hangs single requests while
// a fresh retry completes at healthy latency (~0.2-0.35s), and the recall
// path must answer within its client's hard 5s budget.
const embeddingClient = createCachingEmbeddingClient(
  new StallGuardEmbeddingClient(
    new OpenAICompatibleEmbeddingClient({
      client: openai,
      model: embeddingModel,
      dims: embeddingDims,
    }),
    {
      timeoutMs: embeddingStallTimeoutMs,
      batchTimeoutMs: embeddingStallBatchTimeoutMs,
      maxRetries: embeddingStallRetries,
    },
  ),
  { model: embeddingModel, dims: embeddingDims },
);
const traceRegistry = memoryTraceEnabledFromEnv(process.env)
  ? new MemoryTraceRegistry({
      capacity: memoryTraceCapacityFromEnv(process.env),
      maxTenants: memoryTraceMaxTenantsFromEnv(process.env),
      includePayloads: true,
    })
  : undefined;
// This root snapshot controls sidecar-wide admission/scheduling only. Do not
// pass it to BorgPool: each being must load <root>/<tenant>/config.json itself.
const sidecarConfig = loadConfig({ env: process.env, dataDir: root });
const selfName = memorySelfNameFromEnv(process.env);

const pool = new BorgPool({
  root,
  maxOpen,
  openOptions: {
    embeddingDimensions: embeddingDims,
    embeddingClient,
    llmClient,
    liveCommitmentExtraction: memoryCommitmentExtractionEnabledFromEnv(process.env),
    liveCommitmentExtractionBudget: memoryCommitmentExtractionBudgetFromEnv(process.env),
  },
  initializeBeing: (_tenantId, borg) => {
    // This is idempotent tenant provisioning performed by the pool lifecycle,
    // not a dream mutation. It intentionally also runs before a dry-run dream.
    borg.entities.ensureSelf(selfName, { provenance: "config_default_user" });
  },
  ...(traceRegistry === undefined
    ? {}
    : {
        tracerFor: (tenantId: string) => traceRegistry.tracerFor(tenantId),
      }),
});
const maintenanceCoordinator = new MemoryMaintenanceCoordinator({
  pool,
  config: memoryMaintenanceConfigFromConfig(sidecarConfig),
});

const server = createServer(
  createMemoryHandler({
    pool,
    token,
    maintenanceCoordinator,
    recallAbstainThreshold,
    recallDeadlineMs,
    ...(traceRegistry === undefined ? {} : { traceRegistry }),
  }),
);
server.listen(port, host, () => {
  console.log(
    `borg memory sidecar listening on ${host}:${port} (root=${root}, maxOpen=${maxOpen})`,
  );
});

let shuttingDown = false;
function closeServer(): Promise<void> {
  return new Promise((resolve) => server.close(() => resolve()));
}

async function shutdown(signal: string): Promise<void> {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  console.log(`memory-sidecar: ${signal} received, draining in-flight requests...`);
  const result = await drainMemorySidecar({
    timeoutMs: shutdownTimeoutMs,
    beginShutdown: () => maintenanceCoordinator.beginShutdown(),
    forceFinalizeMaintenance: () => maintenanceCoordinator.forceFinalizeAborted(),
    onAbandoned: (runIds) => {
      if (runIds.length > 0) {
        console.error(
          `memory-sidecar: abandoned maintenance runs during shutdown: ${runIds.join(", ")}`,
        );
      }
    },
    closeIdleConnections: () => server.closeIdleConnections(),
    closeHttp: closeServer,
    // shutdown() (not closeAll()) is a barrier: it rejects any new withTenant so
    // accepted requests cannot open a being that escapes the drain.
    shutdownPool: () => pool.shutdown(),
  });
  if (result.http.status === "timed_out") {
    console.error("memory-sidecar: HTTP drain exceeded the shutdown deadline");
  } else if (result.http.status === "error") {
    console.error("memory-sidecar: error during HTTP drain", result.http.error);
  }
  if (result.pool.status === "timed_out") {
    console.error("memory-sidecar: pool shutdown exceeded the shutdown deadline");
  } else if (result.pool.status === "error") {
    console.error("memory-sidecar: error during pool shutdown", result.pool.error);
  }
  process.exit(0);
}

process.on("SIGTERM", () => void shutdown("SIGTERM"));
process.on("SIGINT", () => void shutdown("SIGINT"));
