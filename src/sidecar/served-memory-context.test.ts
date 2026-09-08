import { describe, expect, it } from "vitest";

import { ManualClock } from "../util/clock.js";
import { createEntityId, createSessionId } from "../util/ids.js";
import { ServedMemoryContextRegistry } from "./served-memory-context.js";

describe("served memory contexts", () => {
  it.each([0, -1, Number.NaN, Infinity, -Infinity, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "falls back to bounded defaults for invalid capacities %s",
    (capacity) => {
      const registry = new ServedMemoryContextRegistry({
        maxPerTenant: capacity,
        maxTenants: capacity,
      });
      const snapshot = {
        sessionId: createSessionId(),
        audienceEntityId: createEntityId(),
        senderEntityId: createEntityId(),
        episodes: [],
        commitments: [],
      };
      const first = registry.put("acme", snapshot);
      let latest = first;
      for (let i = 0; i < 16; i++) latest = registry.put("acme", snapshot);
      expect(registry.get("acme", snapshot.sessionId, first)).toBeUndefined();
      expect(registry.get("acme", snapshot.sessionId, latest)).toBeDefined();
      for (let i = 0; i < 64; i++) registry.put(`other-${i}`, snapshot);
      expect(registry.get("acme", snapshot.sessionId, latest)).toBeUndefined();
    },
  );

  it.each([0, -1, Number.NaN, Infinity, -Infinity])(
    "expires snapshots with invalid TTL %s",
    (ttlMs) => {
      const clock = new ManualClock(100);
      const registry = new ServedMemoryContextRegistry({ clock, ttlMs });
      const snapshot = {
        sessionId: createSessionId(),
        audienceEntityId: createEntityId(),
        senderEntityId: createEntityId(),
        episodes: [],
        commitments: [],
      };
      const id = registry.put("acme", snapshot);
      clock.advance(30 * 60_000 - 1);
      expect(registry.get("acme", snapshot.sessionId, id)).toBeDefined();
      clock.advance(1);
      expect(registry.get("acme", snapshot.sessionId, id)).toBeUndefined();
    },
  );

  it("captures valid limits once at construction", () => {
    const clock = new ManualClock();
    const options = { clock, ttlMs: 10, maxPerTenant: 1, maxTenants: 1 };
    const registry = new ServedMemoryContextRegistry(options);
    options.ttlMs = options.maxPerTenant = options.maxTenants = Infinity;
    const snapshot = {
      sessionId: createSessionId(),
      audienceEntityId: createEntityId(),
      senderEntityId: createEntityId(),
      episodes: [],
      commitments: [],
    };
    const first = registry.put("acme", snapshot);
    const second = registry.put("acme", snapshot);
    expect(registry.get("acme", snapshot.sessionId, first)).toBeUndefined();
    const other = registry.put("other", snapshot);
    expect(registry.get("acme", snapshot.sessionId, second)).toBeUndefined();
    clock.advance(10);
    expect(registry.get("other", snapshot.sessionId, other)).toBeUndefined();
  });

  it("keeps independent context IDs across completion order, isolates tenants/sessions, and expires", () => {
    const clock = new ManualClock(100);
    const registry = new ServedMemoryContextRegistry({ clock, ttlMs: 10 });
    const snapshot = {
      sessionId: createSessionId(),
      audienceEntityId: createEntityId(),
      senderEntityId: createEntityId(),
      episodes: [],
      commitments: [],
    };
    const newer = registry.put("acme", snapshot);
    const late = registry.put("acme", snapshot);
    expect(late).not.toBe(newer);
    expect(registry.get("acme", snapshot.sessionId, newer)).toMatchObject(snapshot);
    expect(registry.get("other", snapshot.sessionId, newer)).toBeUndefined();
    expect(registry.get("acme", createSessionId(), newer)).toBeUndefined();
    snapshot.senderEntityId = createEntityId();
    expect(registry.get("acme", snapshot.sessionId, newer)?.senderEntityId).not.toBe(
      snapshot.senderEntityId,
    );
    clock.advance(10);
    expect(registry.get("acme", snapshot.sessionId, newer)).toBeUndefined();
    expect(registry.get("acme", snapshot.sessionId, late)).toBeUndefined();
  });

  it("bounds contexts per tenant and evicts least recently used tenants", () => {
    const registry = new ServedMemoryContextRegistry({ maxPerTenant: 2, maxTenants: 2 });
    const snapshot = {
      sessionId: createSessionId(),
      audienceEntityId: createEntityId(),
      senderEntityId: createEntityId(),
      episodes: [],
      commitments: [],
    };
    const old = registry.put("acme", snapshot);
    const retained = registry.put("acme", snapshot);
    registry.put("acme", snapshot);
    expect(registry.get("acme", snapshot.sessionId, old)).toBeUndefined();
    const other = registry.put("other", snapshot);
    expect(registry.get("acme", snapshot.sessionId, retained)).toBeDefined();
    registry.put("third", snapshot);
    expect(registry.get("other", snapshot.sessionId, other)).toBeUndefined();
    expect(registry.get("acme", snapshot.sessionId, retained)).toBeDefined();
  });
});
