import { describe, expect, it } from "vitest";

import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { createSessionId, DEFAULT_SESSION_ID } from "../util/ids.js";

import { streamWatermarkMigrations, StreamWatermarkRepository } from "./watermark.js";

describe("StreamWatermarkRepository", () => {
  function openRepo(nowMs = 1_000): {
    repo: StreamWatermarkRepository;
    clock: ManualClock;
    close: () => void;
  } {
    const db = openDatabase(":memory:", {
      migrations: streamWatermarkMigrations,
    });
    const clock = new ManualClock(nowMs);
    const repo = new StreamWatermarkRepository({ db, clock });
    return { repo, clock, close: () => db.close() };
  }

  it("returns null for unknown (process, session) pairs", () => {
    const { repo, close } = openRepo();

    try {
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).toBeNull();
    } finally {
      close();
    }
  });

  it("upserts watermarks and returns them on subsequent reads", () => {
    const { repo, clock, close } = openRepo(1_000);

    try {
      const first = repo.set("episodic-extractor", DEFAULT_SESSION_ID, {
        lastTs: 123,
        lastEntryId: "strm_aaaaaaaaaaaaaaaa",
      });

      expect(first.lastTs).toBe(123);
      expect(first.lastEntryId).toBe("strm_aaaaaaaaaaaaaaaa");
      expect(first.updatedAt).toBe(1_000);

      clock.advance(50);
      const second = repo.set("episodic-extractor", DEFAULT_SESSION_ID, {
        lastTs: 456,
        lastEntryId: "strm_bbbbbbbbbbbbbbbb",
      });

      expect(second.lastTs).toBe(456);
      expect(second.updatedAt).toBe(1_050);

      const fetched = repo.get("episodic-extractor", DEFAULT_SESSION_ID);
      expect(fetched?.lastTs).toBe(456);
      expect(fetched?.lastEntryId).toBe("strm_bbbbbbbbbbbbbbbb");
    } finally {
      close();
    }
  });

  it("keeps watermarks scoped per (process, session)", () => {
    const { repo, close } = openRepo();
    const otherSession = createSessionId();

    try {
      repo.set("episodic-extractor", DEFAULT_SESSION_ID, { lastTs: 100, lastEntryId: "watermark" });
      repo.set("episodic-extractor", otherSession, { lastTs: 200, lastEntryId: "watermark" });
      repo.set("semantic-extractor", DEFAULT_SESSION_ID, { lastTs: 300, lastEntryId: "watermark" });

      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(100);
      expect(repo.get("episodic-extractor", otherSession)?.lastTs).toBe(200);
      expect(repo.get("semantic-extractor", DEFAULT_SESSION_ID)?.lastTs).toBe(300);
    } finally {
      close();
    }
  });

  it("reset removes the watermark", () => {
    const { repo, close } = openRepo();

    try {
      repo.set("episodic-extractor", DEFAULT_SESSION_ID, { lastTs: 100, lastEntryId: "watermark" });
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).not.toBeNull();

      repo.reset("episodic-extractor", DEFAULT_SESSION_ID);
      expect(repo.get("episodic-extractor", DEFAULT_SESSION_ID)).toBeNull();
    } finally {
      close();
    }
  });
});
