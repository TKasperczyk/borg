import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import {
  StreamWatermarkRepository,
  StreamWriter,
  streamWatermarkMigrations,
} from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";

import { StreamIngestionCoordinator } from "./coordinator.js";

type ExtractCall = {
  session?: unknown;
  sinceTs?: number;
  untilTs?: number;
};

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

  async function seedStream(dataDir: string, entries: readonly { kind: "user_msg" | "agent_msg"; content: string; ts: number }[]): Promise<void> {
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

      const watermark = repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      expect(watermark?.lastTs).toBe(110);
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
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(110);

      // Add two more entries -- a new turn.
      await seedStream(dataDir, [
        { kind: "user_msg", content: "followup", ts: 200 },
        { kind: "agent_msg", content: "second reply", ts: 210 },
      ]);

      await coordinator.ingest(DEFAULT_SESSION_ID);
      expect(calls).toHaveLength(2);
      // Second call should have sinceTs = previousLastTs + 1 = 111.
      expect(calls[1]?.sinceTs).toBe(111);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(210);
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

  it("coalesces concurrent ingests for the same session", async () => {
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

      // All three awaits resolve to the same shared result; the extractor
      // was called exactly once because the in-flight promise was shared.
      expect(calls).toHaveLength(1);
      expect(a.ran).toBe(true);
      expect(b).toBe(a);
      expect(c).toBe(a);
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
