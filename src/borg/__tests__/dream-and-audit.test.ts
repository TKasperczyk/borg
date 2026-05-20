import { afterEach, describe, expect, it, vi } from "vitest";

import {
  FakeLLMClient,
  episodicMigrations,
  EpisodicRepository,
  createEpisodesTableSchema,
  selfMigrations,
  retrievalMigrations,
  LanceDbStore,
  composeMigrations,
  openDatabase,
  ManualClock,
  createTestConfig,
  Borg,
  ScriptedEmbeddingClient,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

describe("Borg", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("runs offline maintenance through the Borg facade and exposes audit reversal", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const nowMs = 100 * 24 * 60 * 60 * 1_000;
    const clock = new ManualClock(nowMs);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(episodicMigrations, selfMigrations, retrievalMigrations),
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock,
    });

    await repo.insert({
      id: "ep_cccccccccccccccc" as never,
      title: "Old quiet note",
      narrative: "A stale note that should be archived by the curator.",
      participants: ["team"],
      location: null,
      start_time: nowMs - 50 * 24 * 60 * 60 * 1_000,
      end_time: nowMs - 50 * 24 * 60 * 60 * 1_000 + 1,
      source_stream_ids: ["strm_cccccccccccccccc" as never],
      significance: 0.2,
      tags: ["quiet"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([0, 1, 0, 0]),
      created_at: nowMs - 50 * 24 * 60 * 60 * 1_000,
      updated_at: nowMs - 50 * 24 * 60 * 60 * 1_000,
    });
    db.close();
    await store.close();

    const borg = await Borg.open({
      config: createTestConfig({
        dataDir: tempDir,
        perception: {
          llmEnabled: false,
        },
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "sonnet",
            background: "haiku",
            extraction: "haiku",
          },
        },
        offline: {
          consolidator: {
            enabled: true,
            similarityThreshold: 0.82,
            minClusterSize: 2,
            maxClustersPerRun: 2,
            budget: 15_000,
          },
          reflector: {
            enabled: true,
            minSupport: 3,
            ceilingConfidence: 0.5,
            maxInsightsPerRun: 2,
            budget: 30_000,
          },
          curator: {
            enabled: true,
            t1Heat: 5,
            t2Heat: 15,
            t3DemoteHeat: 3,
            archiveAgeDays: 45,
            archiveMinHeat: 1,
          },
          overseer: {
            enabled: true,
            lookbackHours: 24,
            maxChecksPerRun: 8,
            budget: 20_000,
          },
        },
      }),
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const result = await borg.dream.curate();
      expect(result.results[0]?.process).toBe("curator");

      const audits = borg.audit.list({
        process: "curator",
      });
      expect(audits.length).toBeGreaterThan(0);
      expect((await borg.episodic.get("ep_cccccccccccccccc" as never))?.episode.id).toBeUndefined();

      const archiveAudit = audits.find((audit) => audit.action === "archive");
      expect(archiveAudit).toBeDefined();

      const reverted = await borg.audit.revert(archiveAudit!.id);
      expect(reverted?.reverted_at).not.toBeNull();
      expect((await borg.episodic.get("ep_cccccccccccccccc" as never))?.episode.id).toBe(
        "ep_cccccccccccccccc",
      );
    } finally {
      await borg.close();
    }
  });
});
