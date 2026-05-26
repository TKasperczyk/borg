import { describe, expect, it } from "vitest";

import { openDatabase } from "../storage/sqlite/index.js";
import { ManualClock } from "../util/clock.js";
import { createEntityId, createSessionId, DEFAULT_SESSION_ID } from "../util/ids.js";

import { operatorAdviceMigrations } from "./migrations.js";
import { OperatorAdviceRepository } from "./repository.js";
import { operatorAdviceStatus } from "./types.js";

function createRepository(clock = new ManualClock(1_000)) {
  const db = openDatabase(":memory:", { migrations: operatorAdviceMigrations });
  const repository = new OperatorAdviceRepository(db, clock);

  return { db, repository, clock };
}

describe("OperatorAdviceRepository", () => {
  it("queues advice with a default ttl and rejects unscoped input", () => {
    const { db, repository } = createRepository();
    try {
      expect(() => repository.queue({ text: "   " as never, session_id: DEFAULT_SESSION_ID })).toThrow(
        "Invalid operator advice input",
      );
      expect(() => repository.queue({ text: "stand firm" })).toThrow(
        "Invalid operator advice input",
      );

      const record = repository.queue({
        text: "  Stand firm if Alice is unfair.  ",
        session_id: DEFAULT_SESSION_ID,
      });

      expect(record).toMatchObject({
        session_id: DEFAULT_SESSION_ID,
        audience_entity_id: null,
        text: "Stand firm if Alice is unfair.",
        created_at: 1_000,
        expires_at: 86_401_000,
        consumed_at: null,
        canceled_at: null,
      });
      expect(record.id).toMatch(/^adv_[a-z0-9]{16}$/);
    } finally {
      db.close();
    }
  });

  it("lists pending advice by session or audience and filters expired rows", () => {
    const { db, repository, clock } = createRepository();
    try {
      const sessionA = createSessionId();
      const sessionB = createSessionId();
      const audience = createEntityId();
      const sessionAdvice = repository.queue({
        text: "session advice",
        session_id: sessionA,
      });
      clock.advance(10);
      const audienceAdvice = repository.queue({
        text: "audience advice",
        audience_entity_id: audience,
      });
      clock.advance(10);
      const otherSessionAdvice = repository.queue({
        text: "other session",
        session_id: sessionB,
      });
      repository.queue({
        text: "already expired",
        session_id: sessionA,
        expires_at: clock.now() - 1,
      });

      expect(repository.list({ pendingOnly: true, session_id: sessionA }).map((r) => r.id)).toEqual([
        sessionAdvice.id,
      ]);
      expect(
        repository
          .list({ pendingOnly: true, session_id: sessionB, audience_entity_id: audience })
          .map((r) => r.id),
      ).toEqual([audienceAdvice.id, otherSessionAdvice.id]);
      expect(repository.list({ session_id: sessionA }).map((r) => r.text)).toEqual([
        "session advice",
        "already expired",
      ]);
    } finally {
      db.close();
    }
  });

  it("cancels idempotently without consuming or changing expired rows", () => {
    const { db, repository, clock } = createRepository();
    try {
      const pending = repository.queue({ text: "pending", session_id: DEFAULT_SESSION_ID });
      const expired = repository.queue({
        text: "expired",
        session_id: DEFAULT_SESSION_ID,
        expires_at: clock.now() - 1,
      });

      const canceled = repository.cancel(pending.id);
      expect(canceled?.canceled_at).toBe(1_000);
      expect(operatorAdviceStatus(canceled!, clock.now())).toBe("canceled");

      clock.advance(10);
      expect(repository.cancel(pending.id)?.canceled_at).toBe(1_000);
      expect(repository.cancel(expired.id)?.canceled_at).toBeNull();
      expect(repository.cancel("adv_aaaaaaaaaaaaaaaa" as never)).toBeNull();
    } finally {
      db.close();
    }
  });

  it("marks only pending advice consumed", () => {
    const { db, repository, clock } = createRepository();
    try {
      const first = repository.queue({ text: "first", session_id: DEFAULT_SESSION_ID });
      const canceled = repository.queue({ text: "canceled", session_id: DEFAULT_SESSION_ID });
      const expired = repository.queue({
        text: "expired",
        session_id: DEFAULT_SESSION_ID,
        expires_at: clock.now() - 1,
      });
      repository.cancel(canceled.id);

      clock.advance(25);
      const consumed = repository.markConsumed([first.id, canceled.id, expired.id], {
        turn_id: "turn-a",
        now: clock.now(),
      });

      expect(consumed.map((record) => record.id)).toEqual([first.id]);
      expect(consumed[0]).toMatchObject({
        consumed_at: 1_025,
        consumed_by_turn_id: "turn-a",
      });
      expect(repository.get(canceled.id)?.consumed_at).toBeNull();
      expect(repository.get(expired.id)?.consumed_at).toBeNull();
      expect(repository.list({ pendingOnly: true, session_id: DEFAULT_SESSION_ID })).toEqual([]);
    } finally {
      db.close();
    }
  });

  it("unconsumes only advice consumed by the supplied turn", () => {
    const { db, repository, clock } = createRepository();
    try {
      const first = repository.queue({ text: "first", session_id: DEFAULT_SESSION_ID });
      const second = repository.queue({ text: "second", session_id: DEFAULT_SESSION_ID });
      repository.markConsumed([first.id], { turn_id: "turn-a", now: clock.now() });
      repository.markConsumed([second.id], { turn_id: "turn-b", now: clock.now() });

      repository.unconsume([first.id, second.id], { turn_id: "turn-a" });

      expect(repository.get(first.id)).toMatchObject({
        consumed_at: null,
        consumed_by_turn_id: null,
      });
      expect(repository.get(second.id)).toMatchObject({
        consumed_by_turn_id: "turn-b",
      });
    } finally {
      db.close();
    }
  });

  it("orders explicit history lists by newest terminal timestamp before applying limit", () => {
    const { db, repository, clock } = createRepository();
    try {
      const consumed: ReturnType<typeof repository.queue>[] = [];
      for (let index = 0; index < 5; index += 1) {
        consumed.push(
          repository.queue({ text: `consumed ${index}`, session_id: DEFAULT_SESSION_ID }),
        );
        clock.advance(10);
        repository.markConsumed([consumed[index]!.id], {
          turn_id: `turn-${index}`,
          now: clock.now(),
        });
        clock.advance(10);
      }

      expect(
        repository
          .list({ pendingOnly: false, session_id: DEFAULT_SESSION_ID, limit: 2 })
          .map((record) => record.text),
      ).toEqual(["consumed 4", "consumed 3"]);
    } finally {
      db.close();
    }
  });
});
