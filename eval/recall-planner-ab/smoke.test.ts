import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { DatabaseSync } from "node:sqlite";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { episodicMigrations } from "../../src/memory/episodic/migrations.js";
import { commitmentMigrations } from "../../src/memory/commitments/index.js";
import {
  EpisodicRepository,
  createEpisodesTableSchema,
  type Episode,
} from "../../src/memory/episodic/index.js";
import { retrievalMigrations } from "../../src/retrieval/migrations.js";
import {
  StreamEntryIndexRepository,
  StreamWriter,
  streamEntryIndexMigrations,
} from "../../src/stream/index.js";
import { LanceDbStore } from "../../src/storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../../src/storage/sqlite/index.js";
import { FixedClock } from "../../src/util/clock.js";
import { createSessionId } from "../../src/util/ids.js";

import type { RecallPlannerAbResults, RecallPlannerCase } from "./types.js";
import { main } from "./cli.js";

const TARGET_ID = "ep_aaaaaaaaaaaaaaaa";
const DISTRACTOR_ID = "ep_bbbbbbbbbbbbbbbb";

type JsonObject = Record<string, unknown>;

function responseJson(response: ServerResponse, status: number, value: unknown): void {
  const body = JSON.stringify(value);
  response.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  response.end(body);
}

async function requestJson(request: IncomingMessage): Promise<JsonObject> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8")) as JsonObject;
}

function semanticVariantCount(body: JsonObject): number {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const prompt = messages
    .flatMap((message) => {
      if (message === null || typeof message !== "object" || Array.isArray(message)) {
        return [];
      }
      const content = (message as JsonObject).content;
      return typeof content === "string" ? [content] : [];
    })
    .join("\n");
  const match = /SEMANTIC_VARIANT_COUNT:\s*(\d+)/.exec(prompt);
  return match === null ? 1 : Number(match[1]);
}

async function startFakeGateway(): Promise<{
  baseUrl: string;
  requests: JsonObject[];
  close(): Promise<void>;
}> {
  const requests: JsonObject[] = [];
  const server = createServer((request, response) => {
    void (async () => {
      const path = new URL(request.url ?? "/", "http://localhost").pathname;

      if (request.method === "GET" && path === "/v1/models") {
        responseJson(response, 200, {
          object: "list",
          data: [
            {
              id: "fake-embedding",
              object: "model",
              created: 0,
              owned_by: "test",
              dimensions: 4,
            },
            {
              id: "fake-planner",
              object: "model",
              created: 0,
              owned_by: "test",
            },
          ],
        });
        return;
      }

      if (request.method === "POST" && path === "/v1/embeddings") {
        const body = await requestJson(request);
        requests.push(body);
        const inputs = Array.isArray(body.input) ? body.input : [body.input];
        responseJson(response, 200, {
          object: "list",
          model: body.model,
          data: inputs.map((raw, index) => {
            const text = typeof raw === "string" ? raw : "";
            const targetsExpectedEpisode =
              text === "Maja Chen wybrała wdrożenie Omega po porównaniu wariantów." ||
              text === "Maja Chen" ||
              text === "Omega";
            return {
              object: "embedding",
              index,
              embedding: targetsExpectedEpisode ? [1, 0, 0, 0] : [0, 1, 0, 0],
            };
          }),
          usage: { prompt_tokens: 1, total_tokens: 1 },
        });
        return;
      }

      if (request.method === "POST" && path === "/v1/chat/completions") {
        const body = await requestJson(request);
        requests.push(body);
        const count = semanticVariantCount(body);
        const variants = Array.from({ length: count }, (_, index) => ({
          strategy:
            count === 1
              ? "combined"
              : index === 0
                ? "verbatim_preserving"
                : index === 1
                  ? "memory_owner_voice"
                  : index === 2
                    ? "aspect_focused"
                    : "additional",
          query: "Maja Chen wybrała wdrożenie Omega po porównaniu wariantów.",
        }));
        responseJson(response, 200, {
          id: "chatcmpl-test",
          object: "chat.completion",
          created: 0,
          model: body.model,
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call_recall_plan",
                    type: "function",
                    function: {
                      name: "EmitRecallQueryPlan",
                      arguments: JSON.stringify({
                        resolved_query:
                          "Który wariant wdrożenia Omega wybrała Maja Chen po porównaniu?",
                        semantic_variants: variants,
                        named_terms: ["Maja Chen", "Omega"],
                        typed_queries: [],
                      }),
                    },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
          usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 },
        });
        return;
      }

      responseJson(response, 404, { error: { message: `Unhandled ${request.method} ${path}` } });
    })().catch((error: unknown) => {
      responseJson(response, 500, {
        error: { message: error instanceof Error ? error.message : String(error) },
      });
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("Fake gateway did not receive a TCP address");
  }

  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    requests,
    close: () => closeServer(server),
  };
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.close((error) => (error === undefined ? resolve() : reject(error)));
  });
}

function episode(input: {
  id: typeof TARGET_ID | typeof DISTRACTOR_ID;
  sourceId: Episode["source_stream_ids"][number];
  title: string;
  narrative: string;
  participants: string[];
  tags: string[];
  significance: number;
  embedding: number[];
  timestamp: number;
}): Episode {
  return {
    id: input.id as Episode["id"],
    title: input.title,
    narrative: input.narrative,
    participants: input.participants,
    location: null,
    start_time: input.timestamp - 1_000,
    end_time: input.timestamp,
    source_stream_ids: [input.sourceId],
    significance: input.significance,
    tags: input.tags,
    confidence: 0.95,
    lineage: { derived_from: [], supersedes: [] },
    emotional_arc: null,
    embedding: Float32Array.from(input.embedding),
    created_at: input.timestamp,
    updated_at: input.timestamp,
  };
}

async function createTinyBank(bankDir: string): Promise<void> {
  const now = Date.now();
  const store = new LanceDbStore({ uri: join(bankDir, "lancedb") });
  const db = openDatabase(join(bankDir, "borg.db"), {
    migrations: composeMigrations(
      episodicMigrations,
      commitmentMigrations,
      retrievalMigrations,
      streamEntryIndexMigrations,
    ),
  });
  const table = await store.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(4),
  });
  const repository = new EpisodicRepository({
    table,
    db,
    clock: new FixedClock(now),
  });
  const entryIndex = new StreamEntryIndexRepository({ db, dataDir: bankDir });
  const writer = new StreamWriter({
    dataDir: bankDir,
    sessionId: createSessionId(),
    clock: new FixedClock(now),
    entryIndex,
  });

  try {
    const targetSource = await writer.append({
      kind: "user_msg",
      content: "Maja Chen porównała warianty wdrożenia Omega.",
    });
    const distractorSource = await writer.append({
      kind: "user_msg",
      content: "Luźna rozmowa o spotkaniu zespołu.",
    });
    await repository.createEpisode(
      episode({
        id: TARGET_ID,
        sourceId: targetSource.id,
        title: "Decyzja Maji o Omega",
        narrative: "Maja Chen wybrała pierwszy wariant wdrożenia Omega.",
        participants: ["Maja Chen"],
        tags: ["Omega"],
        significance: 0.95,
        embedding: [1, 0, 0, 0],
        timestamp: now - 1_000,
      }),
    );
    await repository.createEpisode(
      episode({
        id: DISTRACTOR_ID,
        sourceId: distractorSource.id,
        title: "Spotkanie zespołu",
        narrative: "Zespół krótko omówił termin następnego spotkania.",
        participants: ["team"],
        tags: ["spotkanie"],
        significance: 0.05,
        embedding: [0, 1, 0, 0],
        timestamp: now,
      }),
    );
    db.prepare(
      "INSERT INTO episode_index_metadata (key, value) VALUES ('lance_backfilled_at', ?)",
    ).run(String(now));
  } finally {
    writer.close();
    db.close();
    await store.close();
  }
}

function smokeCase(): RecallPlannerCase {
  return {
    id: "smoke-referential-focus",
    focus: "A co ona wtedy wybrała?",
    context_turns: [
      { role: "user", content: "Przypomnij mi rozmowę z Mają o projekcie Omega." },
      {
        role: "assistant",
        content: "Maja Chen porównywała dwa warianty wdrożenia Omega.",
      },
    ],
    identity: {
      memory_owner_name: "team-agent",
      current_sender_name: "Tomasz",
      current_venue: { type: "personal", name: "Tomasz" },
      entity_terms: ["Maja Chen", "Omega"],
    },
    owner_recent_activity: [],
    expected_episode_ids: [TARGET_ID],
  };
}

describe("recall planner A/B smoke", () => {
  const cleanup: Array<() => Promise<void> | void> = [];

  afterEach(async () => {
    vi.restoreAllMocks();
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("runs the real pipeline against a tiny copied bank and a local fake gateway", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-recall-planner-ab-"));
    const bankDir = join(root, "bank");
    const outDir = join(root, "out");
    const casesPath = join(root, "cases.json");
    cleanup.push(() => rmSync(root, { recursive: true, force: true }));
    await createTinyBank(bankDir);
    writeFileSync(casesPath, `${JSON.stringify([smokeCase()], null, 2)}\n`, { mode: 0o600 });
    const gateway = await startFakeGateway();
    cleanup.push(() => gateway.close());
    const stdout = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const args = [
      "--data-dir",
      bankDir,
      "--cases",
      casesPath,
      "--out",
      outDir,
      "--variant-counts",
      "1,3",
      "--embedding-model",
      "fake-embedding",
      "--baseline",
    ];
    const env = {
      KRATOS_BASE_URL: gateway.baseUrl,
      LLM_API_KEY: "test-key",
      BORG_MODEL_RECALL_EXPANSION: "fake-planner",
    };

    await expect(main(args, env)).resolves.toBe(0);

    const results = JSON.parse(
      readFileSync(join(outDir, "results.json"), "utf8"),
    ) as RecallPlannerAbResults;
    const baseline = results.runs.find((run) => run.configuration_id === "baseline");
    const planner = results.runs.find((run) => run.configuration_id === "planner-n1");
    const plannerThree = results.runs.find((run) => run.configuration_id === "planner-n3");

    expect(baseline).toMatchObject({
      status: "completed",
      best_expected_rank: 2,
      planner_output: null,
    });
    expect(planner).toMatchObject({
      status: "completed",
      best_expected_rank: 1,
      planner_output: {
        resolved_query: "Który wariant wdrożenia Omega wybrała Maja Chen po porównaniu?",
        named_terms: ["Maja Chen", "Omega"],
      },
    });
    expect(planner?.lane_ranks.map((lane) => lane.intent_kind)).toEqual(
      expect.arrayContaining(["raw_text", "semantic_query", "known_term", "recent"]),
    );
    expect(planner?.lane_ranks.find((lane) => lane.intent_kind === "semantic_query")).toMatchObject(
      {
        intent_query: "Maja Chen wybrała wdrożenie Omega po porównaniu wariantów.",
        candidates: expect.arrayContaining([
          expect.objectContaining({ rank: 1, episode_id: TARGET_ID }),
        ]),
      },
    );
    expect(plannerThree?.planner_output?.semantic_variants).toHaveLength(3);
    expect(results.summaries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          configuration_id: "baseline",
          metrics: expect.objectContaining({ recall_at_1: 0 }),
        }),
        expect.objectContaining({
          configuration_id: "planner-n1",
          metrics: expect.objectContaining({ recall_at_1: 1 }),
        }),
      ]),
    );
    expect(readFileSync(join(outDir, "report.md"), "utf8")).toContain(
      "Baseline: raw FOCUS-blob lane only",
    );
    expect(stdout).toHaveBeenCalled();

    const plannerRequest = gateway.requests.find((request) => request.model === "fake-planner");
    expect(JSON.stringify(plannerRequest)).toContain("Maja Chen porównywała dwa warianty");
    expect(JSON.stringify(plannerRequest)).toContain("A co ona wtedy wybrała?");

    const postRequestCount = gateway.requests.length;
    await expect(main(args, env)).resolves.toBe(0);
    expect(gateway.requests).toHaveLength(postRequestCount);
    const warmResults = JSON.parse(
      readFileSync(join(outDir, "results.json"), "utf8"),
    ) as RecallPlannerAbResults;
    for (const run of warmResults.runs) {
      expect(run.embedding.gateway_attempt_count).toBe(0);
      expect(run.embedding.disk_cache_hit_count).toBeGreaterThan(0);
      if (run.configuration_id !== "baseline") {
        expect(run.planner_cache_hit).toBe(true);
      }
    }
    expect(statSync(outDir).mode & 0o777).toBe(0o700);
    expect(statSync(join(outDir, "cache")).mode & 0o777).toBe(0o700);
    expect(statSync(join(outDir, "results.json")).mode & 0o777).toBe(0o600);
    expect(statSync(join(outDir, "report.md")).mode & 0o777).toBe(0o600);

    const readOnlyCheck = new DatabaseSync(join(bankDir, "borg.db"), { readOnly: true });
    try {
      const row = readOnlyCheck.prepare("SELECT COUNT(*) AS count FROM retrieval_log").get() as {
        count: number;
      };
      expect(row.count).toBe(0);
    } finally {
      readOnlyCheck.close();
    }
  }, 30_000);
});
