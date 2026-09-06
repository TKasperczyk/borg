import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { createSessionId, createStreamEntryId } from "../../util/ids.js";
import {
  AgentDeliveryRepository,
  agentDeliveryMigrations,
  agentDeliveryIdSchema,
} from "./agent-deliveries.js";

const cleanups: Array<() => void> = [];
afterEach(() => {
  while (cleanups.length) cleanups.pop()!();
});

function harness(initialMigrations = agentDeliveryMigrations) {
  const dir = mkdtempSync(join(tmpdir(), "agent-deliveries-"));
  cleanups.push(() => rmSync(dir, { recursive: true, force: true }));
  const path = join(dir, "borg.db");
  let db = openDatabase(path, { migrations: initialMigrations });
  cleanups.push(() => db.close());
  const clock = new ManualClock(1000);
  const onAvailable = vi.fn();
  const repo = () => new AgentDeliveryRepository({ db, clock, onAvailable });
  const sessionId = createSessionId();
  const input = {
    sessionId,
    terminalEntryId: createStreamEntryId(),
    taskId: "task",
    content: "Result",
    createdAt: 1000,
  };
  return {
    input,
    clock,
    onAvailable,
    repo,
    row: () =>
      db
        .prepare("SELECT * FROM agent_deliveries WHERE terminal_entry_id = ?")
        .get(input.terminalEntryId),
    reopen() {
      db.close();
      db = openDatabase(path, { migrations: agentDeliveryMigrations });
    },
  };
}

describe("AgentDeliveryRepository", () => {
  it("adds acknowledgement receipts without replacing existing delivery rows or active leases", () => {
    const h = harness(agentDeliveryMigrations.slice(0, 1));
    h.repo().create(h.input);
    const [lease] = h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries;
    h.reopen();
    expect(h.row()).toMatchObject({
      delivery_id: lease!.delivery_id,
      state: "leased",
      attempts: 1,
      lease_until: 1100,
    });
    expect(
      h
        .repo()
        .ack({
          delivery_id: lease!.delivery_id,
          claim_generation: lease!.claim_generation,
          outcome: "sent",
        }),
    ).toBe("acknowledged");
    expect(h.row()).toMatchObject({ state: "sent", attempts: 1, lease_until: null });
  });

  it("keeps delivery identity and lease state across recreation and restart", () => {
    const h = harness();
    const repo = h.repo();
    repo.create(h.input);
    repo.create({ ...h.input, content: "must not replace" });
    expect(h.onAvailable).toHaveBeenCalledTimes(1);
    const lease = repo.claim({ sessionIds: [h.input.sessionId], leaseMs: 100 });
    const id = lease.deliveries[0]!.delivery_id;
    expect(lease.deliveries[0]!.claim_generation).toBe(1);
    expect(h.row()).toMatchObject({
      delivery_id: id,
      content: "Result",
      attempts: 1,
      state: "leased",
      lease_until: 1100,
    });
    h.reopen();
    expect(h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 })).toEqual({
      deliveries: [],
      nextLeaseUntil: 1100,
    });
    h.clock.advance(100);
    expect(
      h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries[0]!.delivery_id,
    ).toBe(id);
    expect(h.row()).toMatchObject({ attempts: 2, lease_until: 1200 });
    expect(
      h
        .repo()
        .ack({ delivery_id: id, claim_generation: 2, outcome: "sent", teams_message_id: "teams" }),
    ).toBe("acknowledged");
    h.reopen();
    h.repo().create(h.input);
    expect(
      h.repo().ack({
        delivery_id: id,
        claim_generation: 2,
        outcome: "failed_retryable",
        error: "late error",
      }),
    ).toBe("acknowledged");
    expect(h.row()).toMatchObject({
      state: "sent",
      teams_message_id: "teams",
      last_error: null,
      lease_until: null,
      attempts: 2,
    });
    expect(h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries).toEqual(
      [],
    );
  });

  it("scopes claims to sessions and distinguishes retryable and permanent acknowledgements", () => {
    const h = harness();
    const repo = h.repo();
    repo.create(h.input);
    expect(repo.claim({ sessionIds: [createSessionId()], leaseMs: 10 }).deliveries).toEqual([]);
    const first = repo.claim({ sessionIds: [h.input.sessionId, h.input.sessionId], leaseMs: 10 });
    expect(first.deliveries).toHaveLength(1);
    const id = first.deliveries[0]!.delivery_id;
    expect(
      repo.ack({
        delivery_id: agentDeliveryIdSchema.parse("delivery_0000000000000000"),
        claim_generation: 1,
        outcome: "sent",
      }),
    ).toBeNull();
    expect(
      repo.ack({
        delivery_id: id,
        claim_generation: 1,
        outcome: "failed_retryable",
        error: "transient",
      }),
    ).toBe("acknowledged");
    expect(h.row()).toMatchObject({ state: "pending", last_error: "transient", attempts: 1 });
    expect(h.onAvailable).toHaveBeenCalledTimes(2);
    expect(repo.claim({ sessionIds: [h.input.sessionId], leaseMs: 10 }).deliveries).toHaveLength(1);
    expect(
      repo.ack({
        delivery_id: id,
        claim_generation: 2,
        outcome: "failed_permanent",
        error: "deleted room",
      }),
    ).toBe("acknowledged");
    h.clock.advance(100);
    expect(repo.ack({ delivery_id: id, claim_generation: 2, outcome: "failed_retryable" })).toBe(
      "acknowledged",
    );
    expect(repo.claim({ sessionIds: [h.input.sessionId], leaseMs: 10 }).deliveries).toEqual([]);
    expect(h.row()).toMatchObject({
      state: "failed",
      last_error: "deleted room",
      attempts: 2,
      lease_until: null,
    });
  });

  it("keeps A's retryable ack receipt across restart without releasing B's lease to C", () => {
    const h = harness();
    h.repo().create(h.input);
    const [a] = h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries;
    const ack = {
      delivery_id: a!.delivery_id,
      claim_generation: a!.claim_generation,
      outcome: "failed_retryable" as const,
      error: "try again",
    };
    expect(h.repo().ack(ack)).toBe("acknowledged");
    const [b] = h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries;
    expect(b!.claim_generation).toBe(2);
    h.reopen();
    const notifications = h.onAvailable.mock.calls.length;
    expect(h.repo().ack(ack)).toBe("acknowledged");
    // Even a conflicting replay is bound to the first acknowledgement of A's claim.
    expect(h.repo().ack({ ...ack, outcome: "sent" })).toBe("acknowledged");
    expect(h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries).toEqual(
      [],
    );
    expect(h.onAvailable).toHaveBeenCalledTimes(notifications);
    expect(h.row()).toMatchObject({ state: "leased", attempts: 2, lease_until: 1100 });
    expect(
      h.repo().ack({
        delivery_id: b!.delivery_id,
        claim_generation: b!.claim_generation,
        outcome: "sent",
      }),
    ).toBe("acknowledged");
    expect(h.row()).toMatchObject({ state: "sent", attempts: 2, lease_until: null });
  });

  it("ignores stale acknowledgements after expiry and rejects generations that were never claimed", () => {
    const h = harness();
    h.repo().create(h.input);
    const [a] = h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries;
    h.clock.advance(100);
    expect(
      h.repo().ack({ delivery_id: a!.delivery_id, claim_generation: 1, outcome: "sent" }),
    ).toBe("acknowledged");
    const [b] = h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries;
    expect(b!.claim_generation).toBe(2);
    expect(
      h
        .repo()
        .ack({ delivery_id: a!.delivery_id, claim_generation: 1, outcome: "failed_retryable" }),
    ).toBe("acknowledged");
    expect(
      h.repo().ack({ delivery_id: a!.delivery_id, claim_generation: 3, outcome: "sent" }),
    ).toBeNull();
    expect(h.row()).toMatchObject({ state: "leased", attempts: 2, lease_until: 1200 });
    expect(h.repo().claim({ sessionIds: [h.input.sessionId], leaseMs: 100 }).deliveries).toEqual(
      [],
    );
  });
});
