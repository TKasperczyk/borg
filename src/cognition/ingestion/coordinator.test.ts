import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import { EntityRepository } from "../../memory/commitments/index.js";
import { EpisodicExtractor } from "../../memory/episodic/extractor.js";
import { episodicMigrations } from "../../memory/episodic/migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "../../memory/episodic/repository.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";
import { selfMigrations } from "../../memory/self/migrations.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import {
  StreamReader,
  StreamWatermarkRepository,
  StreamWriter,
  streamWatermarkMigrations,
} from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { EmbeddingError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";

import { StreamIngestionCoordinator } from "./coordinator.js";

type ExtractCall = {
  session?: unknown;
  sinceTs?: number;
  sinceCursor?: {
    ts: number;
    entryId: string;
  };
  untilTs?: number;
};

const EPISODE_TOOL_NAME = "EmitEpisodeCandidates";

function createEpisodeToolResponse(episodes: unknown[]) {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 20,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: EPISODE_TOOL_NAME,
        input: {
          episodes,
        },
      },
    ],
  };
}

function createFakeExtractor(
  calls: ExtractCall[],
  result: { inserted: number; updated: number; skipped: number } = {
    inserted: 1,
    updated: 0,
    skipped: 0,
  },
  // The coordinator only uses extractFromStream, so the rest of the
  // EpisodicExtractor surface can be stubbed out to the minimum the types
  // require. This keeps the test free of a real LLM client.
): never {
  const extractor = {
    async extractFromStream(options: ExtractCall): Promise<typeof result> {
      calls.push(options);
      return result;
    },
  };
  return extractor as unknown as never;
}

describe("StreamIngestionCoordinator", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  function createTempDir(): string {
    const dir = mkdtempSync(join(tmpdir(), "borg-ingestion-"));
    tempDirs.push(dir);
    return dir;
  }

  async function seedStream(
    dataDir: string,
    entries: readonly { kind: "user_msg" | "agent_msg"; content: string; ts: number }[],
  ): Promise<void> {
    for (const entry of entries) {
      const writer = new StreamWriter({
        dataDir,
        sessionId: DEFAULT_SESSION_ID,
        clock: new ManualClock(entry.ts),
      });
      try {
        await writer.append({ kind: entry.kind, content: entry.content });
      } finally {
        writer.close();
      }
    }
  }

  function openRepo(): {
    repo: StreamWatermarkRepository;
    close: () => void;
  } {
    const db = openDatabase(":memory:", {
      migrations: streamWatermarkMigrations,
    });
    return {
      repo: new StreamWatermarkRepository({ db, clock: new ManualClock(1_000) }),
      close: () => db.close(),
    };
  }

  it("no-ops when there are fewer new entries than the threshold", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [{ kind: "user_msg", content: "hi", ts: 100 }]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor: createFakeExtractor(calls),
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
    });

    try {
      const result = await coordinator.ingest(DEFAULT_SESSION_ID);
      expect(result.ran).toBe(false);
      expect(result.processedEntries).toBe(1);
      expect(calls).toHaveLength(0);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).toBeNull();
    } finally {
      close();
    }
  });

  it("runs extraction and advances the watermark when threshold met", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "debug pgvector", ts: 100 },
      { kind: "agent_msg", content: "check dims first", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor: createFakeExtractor(calls),
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
    });

    try {
      const result = await coordinator.ingest(DEFAULT_SESSION_ID);
      expect(result.ran).toBe(true);
      expect(result.processedEntries).toBe(2);
      expect(calls).toHaveLength(1);
      expect(calls[0]?.session).toBe(DEFAULT_SESSION_ID);
      expect(calls[0]?.sinceTs).toBeUndefined();
      expect(calls[0]?.sinceCursor).toBeUndefined();

      const watermark = repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      expect(watermark?.lastTs).toBe(110);
      expect(watermark?.lastEntryId).not.toBeNull();
      expect(
        (
          coordinator as unknown as {
            trackedSessions: Set<string>;
          }
        ).trackedSessions.has(DEFAULT_SESSION_ID),
      ).toBe(false);
    } finally {
      close();
    }
  });

  it("second call resumes past the watermark without reprocessing old entries", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "first", ts: 100 },
      { kind: "agent_msg", content: "reply", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor: createFakeExtractor(calls),
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
    });

    try {
      await coordinator.ingest(DEFAULT_SESSION_ID);
      expect(calls).toHaveLength(1);
      const firstWatermark = repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      expect(firstWatermark?.lastTs).toBe(110);
      expect(firstWatermark?.lastEntryId).not.toBeNull();

      // Add two more entries -- a new turn.
      await seedStream(dataDir, [
        { kind: "user_msg", content: "followup", ts: 200 },
        { kind: "agent_msg", content: "second reply", ts: 210 },
      ]);

      await coordinator.ingest(DEFAULT_SESSION_ID);
      expect(calls).toHaveLength(2);
      expect(calls[1]?.sinceTs).toBeUndefined();
      expect(calls[1]?.sinceCursor).toEqual({
        ts: firstWatermark?.lastTs,
        entryId: firstWatermark?.lastEntryId,
      });
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(210);
    } finally {
      close();
    }
  });

  it("stores the last processed entry pair and resumes later same-ms appends with a cursor", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "first", ts: 100 },
      { kind: "agent_msg", content: "reply", ts: 100 },
    ]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor: createFakeExtractor(calls),
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 1,
    });

    try {
      await coordinator.ingest(DEFAULT_SESSION_ID);
      const firstWatermark = repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      const initialTail = new StreamReader({
        dataDir,
        sessionId: DEFAULT_SESSION_ID,
      }).tail(2);
      const secondEntry = initialTail[1];

      expect(firstWatermark?.lastTs).toBe(100);
      expect(firstWatermark?.lastEntryId).toBe(secondEntry?.id ?? null);

      await seedStream(dataDir, [{ kind: "user_msg", content: "late same-ms", ts: 100 }]);
      await coordinator.ingest(DEFAULT_SESSION_ID);

      expect(calls[1]?.sinceCursor).toEqual({
        ts: 100,
        entryId: secondEntry?.id,
      });

      const secondWatermark = repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      const latestEntry = new StreamReader({
        dataDir,
        sessionId: DEFAULT_SESSION_ID,
      }).tail(1)[0];

      expect(secondWatermark?.lastTs).toBe(100);
      expect(secondWatermark?.lastEntryId).toBe(latestEntry?.id ?? null);
    } finally {
      close();
    }
  });

  it("falls back to sinceTs without polluting the stream or reprocessing leftovers", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "first", ts: 101 },
      { kind: "agent_msg", content: "reply", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    repo.set("episodic-extractor", DEFAULT_SESSION_ID, {
      lastTs: 100,
      lastEntryId: null,
    });
    const extractedContents: string[][] = [];
    const extractedKinds: string[][] = [];
    const fallbackNotices: string[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor: {
        async extractFromStream(options: ExtractCall): Promise<{
          inserted: number;
          updated: number;
          skipped: number;
        }> {
          const reader = new StreamReader({
            dataDir,
            sessionId: DEFAULT_SESSION_ID,
          });
          const batchContents: string[] = [];
          const batchKinds: string[] = [];

          for await (const entry of reader.iterate({
            sinceTs: options.sinceTs,
            sinceCursor: options.sinceCursor,
            untilTs: options.untilTs,
          })) {
            batchContents.push(String(entry.content));
            batchKinds.push(entry.kind);
          }

          extractedContents.push(batchContents);
          extractedKinds.push(batchKinds);

          return {
            inserted: batchContents.length,
            updated: 0,
            skipped: 0,
          };
        },
      } as unknown as never,
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
      onLegacyFallback: (notice) => {
        fallbackNotices.push(notice.message);
      },
    });

    try {
      const first = await coordinator.ingest(DEFAULT_SESSION_ID);
      expect(first.ran).toBe(true);
      expect(first.processedEntries).toBe(2);
      expect(extractedContents).toEqual([["first", "reply"]]);
      expect(extractedKinds).toEqual([["user_msg", "agent_msg"]]);
      expect(fallbackNotices).toHaveLength(1);
      expect(fallbackNotices[0]).toContain("legacy watermark fallback used");

      const second = await coordinator.flush(DEFAULT_SESSION_ID);
      expect(second.ran).toBe(false);
      expect(second.processedEntries).toBe(0);
      expect(extractedContents).toHaveLength(1);

      const tail = new StreamReader({
        dataDir,
        sessionId: DEFAULT_SESSION_ID,
      }).tail(10);

      expect(tail.map((entry) => entry.kind)).toEqual(["user_msg", "agent_msg"]);
    } finally {
      close();
    }
  });

  it("continues ingestion when legacy fallback reporting throws", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "first", ts: 101 },
      { kind: "agent_msg", content: "reply", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    repo.set("episodic-extractor", DEFAULT_SESSION_ID, {
      lastTs: 100,
      lastEntryId: null,
    });
    const calls: ExtractCall[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor: createFakeExtractor(calls),
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
      onLegacyFallback: () => {
        throw new Error("observer failed");
      },
    });

    try {
      const result = await coordinator.ingest(DEFAULT_SESSION_ID);

      expect(result.ran).toBe(true);
      expect(result.processedEntries).toBe(2);
      expect(calls[0]?.sinceTs).toBe(101);
      expect(calls[0]?.sinceCursor).toBeUndefined();
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(110);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastEntryId).not.toBeNull();
    } finally {
      close();
    }
  });

  it("keeps the watermark unchanged on extractor failure", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "boom", ts: 100 },
      { kind: "agent_msg", content: "oops", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    const extractor = {
      async extractFromStream(): Promise<never> {
        throw new Error("boom");
      },
    } as unknown as never;
    const errors: unknown[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor,
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
      onError: (error) => {
        errors.push(error);
      },
    });

    try {
      const result = await coordinator.ingest(DEFAULT_SESSION_ID);
      expect(result.ran).toBe(false);
      expect(result.error).toBeInstanceOf(Error);
      expect(errors).toHaveLength(1);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).toBeNull();
    } finally {
      close();
    }
  });

  it("drops errored tracked sessions during close", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "boom", ts: 100 },
      { kind: "agent_msg", content: "oops", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    let calls = 0;
    const coordinator = new StreamIngestionCoordinator({
      extractor: {
        async extractFromStream(): Promise<never> {
          calls += 1;
          throw new Error("persistent extractor failure");
        },
      } as unknown as never,
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
    });

    try {
      const result = await coordinator.ingest(DEFAULT_SESSION_ID);
      expect(result.error).toBeInstanceOf(Error);
      expect(
        (
          coordinator as unknown as {
            trackedSessions: Set<string>;
          }
        ).trackedSessions.has(DEFAULT_SESSION_ID),
      ).toBe(true);

      await coordinator.close();

      expect(calls).toBe(2);
      expect(
        (
          coordinator as unknown as {
            trackedSessions: Set<string>;
          }
        ).trackedSessions.has(DEFAULT_SESSION_ID),
      ).toBe(false);
    } finally {
      close();
    }
  });

  it("keeps the watermark unchanged on embedding failure and retries next pass", async () => {
    const dataDir = createTempDir();
    const clock = new ManualClock(1_000);
    await seedStream(dataDir, [
      { kind: "user_msg", content: "candidate one", ts: 100 },
      { kind: "agent_msg", content: "candidate two", ts: 110 },
    ]);

    const store = new LanceDbStore({
      uri: join(dataDir, "lancedb"),
    });
    const db = openDatabase(join(dataDir, "borg.db"), {
      migrations: [
        ...episodicMigrations,
        ...selfMigrations,
        ...retrievalMigrations,
        ...streamWatermarkMigrations,
      ],
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const episodicRepository = new EpisodicRepository({
      table,
      db,
      clock,
    });
    const entityRepository = new EntityRepository({
      db,
      clock,
    });
    const entries = new StreamReader({
      dataDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(2);
    const firstEntry = entries[0];

    if (firstEntry === undefined) {
      throw new Error("Expected seeded stream entry");
    }

    let failed = false;
    const embeddingClient: EmbeddingClient = {
      async embed(): Promise<Float32Array> {
        if (!failed) {
          failed = true;
          throw new EmbeddingError("embedding failed");
        }

        return Float32Array.from([1, 0, 0, 0]);
      },
      async embedBatch(texts): Promise<Float32Array[]> {
        return texts.map(() => Float32Array.from([1, 0, 0, 0]));
      },
    };
    const llm = new FakeLLMClient({
      responses: [
        createEpisodeToolResponse([
          {
            title: "Retry me",
            narrative: "This candidate should retry after the embedding failure.",
            source_stream_ids: [firstEntry.id],
            participants: [],
            location: null,
            tags: [],
            confidence: 0.8,
            significance: 0.7,
          },
        ]),
        createEpisodeToolResponse([
          {
            title: "Retry me",
            narrative: "This candidate should retry after the embedding failure.",
            source_stream_ids: [firstEntry.id],
            participants: [],
            location: null,
            tags: [],
            confidence: 0.8,
            significance: 0.7,
          },
        ]),
      ],
    });
    const extractor = new EpisodicExtractor({
      dataDir,
      episodicRepository,
      embeddingClient,
      llmClient: llm,
      model: "haiku",
      entityRepository,
      clock,
    });
    const repo = new StreamWatermarkRepository({
      db,
      clock,
    });
    const coordinator = new StreamIngestionCoordinator({
      extractor,
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
    });

    try {
      const first = await coordinator.ingest(DEFAULT_SESSION_ID);

      expect(first.ran).toBe(false);
      expect(first.error).toBeInstanceOf(EmbeddingError);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).toBeNull();

      const second = await coordinator.ingest(DEFAULT_SESSION_ID);

      expect(second.ran).toBe(true);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(110);
      expect(await episodicRepository.listAll()).toHaveLength(1);
      expect(llm.requests).toHaveLength(2);
    } finally {
      db.close();
      await store.close();
    }
  });

  it("queues concurrent ingests onto a follow-up pass for the same session", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "a", ts: 100 },
      { kind: "agent_msg", content: "b", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor: createFakeExtractor(calls),
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
    });

    try {
      const [a, b, c] = await Promise.all([
        coordinator.ingest(DEFAULT_SESSION_ID),
        coordinator.ingest(DEFAULT_SESSION_ID),
        coordinator.ingest(DEFAULT_SESSION_ID),
      ]);

      expect(calls).toHaveLength(1);
      expect(a.ran).toBe(true);
      expect(b.ran).toBe(false);
      expect(b.processedEntries).toBe(0);
      expect(c).toBe(b);
    } finally {
      close();
    }
  });

  it("runs a queued follow-up pass when new entries arrive during an active ingest", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "first", ts: 100 },
      { kind: "agent_msg", content: "reply", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    let releaseFirstPass: (() => void) | undefined;
    let notifyFirstPassStarted: (() => void) | undefined;
    const firstPassStarted = new Promise<void>((resolve) => {
      notifyFirstPassStarted = resolve;
    });
    let extractCallCount = 0;
    const coordinator = new StreamIngestionCoordinator({
      extractor: {
        async extractFromStream(options: ExtractCall): Promise<{
          inserted: number;
          updated: number;
          skipped: number;
        }> {
          calls.push(options);
          extractCallCount += 1;

          if (extractCallCount === 1) {
            notifyFirstPassStarted?.();
            await new Promise<void>((resolve) => {
              releaseFirstPass = resolve;
            });
          }

          return {
            inserted: 1,
            updated: 0,
            skipped: 0,
          };
        },
      } as unknown as never,
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
    });

    try {
      const firstPromise = coordinator.ingest(DEFAULT_SESSION_ID);
      await firstPassStarted;

      await seedStream(dataDir, [
        { kind: "user_msg", content: "followup", ts: 200 },
        { kind: "agent_msg", content: "second reply", ts: 210 },
      ]);
      const secondPromise = coordinator.ingest(DEFAULT_SESSION_ID);

      releaseFirstPass?.();

      const first = await firstPromise;
      const firstWatermark = repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      const second = await secondPromise;

      expect(first.ran).toBe(true);
      expect(firstWatermark?.lastTs).toBe(110);
      expect(firstWatermark?.lastEntryId).not.toBeNull();

      expect(second.ran).toBe(true);
      expect(calls).toHaveLength(2);
      expect(calls[1]?.sinceCursor).toEqual({
        ts: firstWatermark?.lastTs,
        entryId: firstWatermark?.lastEntryId,
      });
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(210);
    } finally {
      close();
    }
  });

  it("waits for queued follow-up passes that arrive during close", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "first", ts: 100 },
      { kind: "agent_msg", content: "reply", ts: 110 },
    ]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    let notifyFirstPassStarted: (() => void) | undefined;
    let releaseFirstPass: (() => void) | undefined;
    const firstPassStarted = new Promise<void>((resolve) => {
      notifyFirstPassStarted = resolve;
    });
    let notifySecondPassStarted: (() => void) | undefined;
    let releaseSecondPass: (() => void) | undefined;
    const secondPassStarted = new Promise<void>((resolve) => {
      notifySecondPassStarted = resolve;
    });
    let extractCallCount = 0;
    const coordinator = new StreamIngestionCoordinator({
      extractor: {
        async extractFromStream(options: ExtractCall): Promise<{
          inserted: number;
          updated: number;
          skipped: number;
        }> {
          calls.push(options);
          extractCallCount += 1;

          if (extractCallCount === 1) {
            notifyFirstPassStarted?.();
            await new Promise<void>((resolve) => {
              releaseFirstPass = resolve;
            });
          }

          if (extractCallCount === 2) {
            notifySecondPassStarted?.();
            await new Promise<void>((resolve) => {
              releaseSecondPass = resolve;
            });
          }

          return {
            inserted: 1,
            updated: 0,
            skipped: 0,
          };
        },
      } as unknown as never,
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 2,
    });

    try {
      (
        coordinator as unknown as {
          trackedSessions: Set<string>;
        }
      ).trackedSessions.add(DEFAULT_SESSION_ID);

      let closed = false;
      const closePromise = coordinator.close();
      void closePromise.then(() => {
        closed = true;
      });
      await firstPassStarted;

      await seedStream(dataDir, [
        { kind: "user_msg", content: "followup", ts: 200 },
        { kind: "agent_msg", content: "second reply", ts: 210 },
      ]);
      const queuedIngest = coordinator.ingest(DEFAULT_SESSION_ID);

      releaseFirstPass?.();
      await secondPassStarted;
      await Promise.resolve();
      expect(closed).toBe(false);

      releaseSecondPass?.();

      await Promise.all([closePromise, queuedIngest]);
      expect(closed).toBe(true);
      expect(calls).toHaveLength(2);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(210);
      expect(
        (
          coordinator as unknown as {
            trackedSessions: Set<string>;
          }
        ).trackedSessions.has(DEFAULT_SESSION_ID),
      ).toBe(false);
    } finally {
      close();
    }
  });

  it("flushes below-threshold shutdown tails past the watermark", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [
      { kind: "user_msg", content: "one", ts: 100 },
      { kind: "agent_msg", content: "two", ts: 110 },
      { kind: "user_msg", content: "three", ts: 120 },
      { kind: "agent_msg", content: "four", ts: 130 },
      { kind: "user_msg", content: "five", ts: 140 },
    ]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    let notifyFirstPassStarted: (() => void) | undefined;
    let releaseFirstPass: (() => void) | undefined;
    const firstPassStarted = new Promise<void>((resolve) => {
      notifyFirstPassStarted = resolve;
    });
    let notifyFinalDrainStarted: (() => void) | undefined;
    let releaseFinalDrain: (() => void) | undefined;
    const finalDrainStarted = new Promise<void>((resolve) => {
      notifyFinalDrainStarted = resolve;
    });
    let extractCallCount = 0;
    const coordinator = new StreamIngestionCoordinator({
      extractor: {
        async extractFromStream(options: ExtractCall): Promise<{
          inserted: number;
          updated: number;
          skipped: number;
        }> {
          calls.push(options);
          extractCallCount += 1;

          if (extractCallCount === 1) {
            notifyFirstPassStarted?.();
            await new Promise<void>((resolve) => {
              releaseFirstPass = resolve;
            });
          }

          if (extractCallCount === 2) {
            notifyFinalDrainStarted?.();
            await new Promise<void>((resolve) => {
              releaseFinalDrain = resolve;
            });
          }

          return {
            inserted: 1,
            updated: 0,
            skipped: 0,
          };
        },
      } as unknown as never,
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 5,
    });

    try {
      const initialLastEntry = new StreamReader({
        dataDir,
        sessionId: DEFAULT_SESSION_ID,
      }).tail(5)[4];

      const firstPromise = coordinator.ingest(DEFAULT_SESSION_ID);
      await firstPassStarted;

      await seedStream(dataDir, [
        { kind: "agent_msg", content: "tail one", ts: 200 },
        { kind: "user_msg", content: "tail two", ts: 210 },
        { kind: "agent_msg", content: "tail three", ts: 220 },
      ]);

      let closed = false;
      const closePromise = coordinator.close();
      void closePromise.then(() => {
        closed = true;
      });
      const secondPromise = coordinator.ingest(DEFAULT_SESSION_ID);

      releaseFirstPass?.();

      await finalDrainStarted;

      const [first, second] = await Promise.all([firstPromise, secondPromise]);
      expect(first.ran).toBe(true);
      expect(second.ran).toBe(false);
      expect(second.processedEntries).toBe(3);
      await Promise.resolve();
      expect(closed).toBe(false);

      releaseFinalDrain?.();
      await closePromise;

      expect(calls).toHaveLength(2);
      expect(calls[1]?.sinceCursor).toEqual({
        ts: 140,
        entryId: initialLastEntry?.id,
      });
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(220);
    } finally {
      close();
    }
  });

  it("flush lowers the threshold to 1 for that call", async () => {
    const dataDir = createTempDir();
    await seedStream(dataDir, [{ kind: "user_msg", content: "only", ts: 100 }]);

    const { repo, close } = openRepo();
    const calls: ExtractCall[] = [];
    const coordinator = new StreamIngestionCoordinator({
      extractor: createFakeExtractor(calls),
      watermarkRepository: repo,
      dataDir,
      minEntriesThreshold: 4,
    });

    try {
      const result = await coordinator.flush(DEFAULT_SESSION_ID);
      expect(result.ran).toBe(true);
      expect(calls).toHaveLength(1);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(100);
    } finally {
      close();
    }
  });
});
