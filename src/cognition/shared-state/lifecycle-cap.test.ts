import { describe, expect, it } from "vitest";

import type {
  SharedStateArtifact,
  SharedStateEntry,
  SharedStateEntryKind,
} from "../../memory/shared-state/index.js";
import {
  createEntityId,
  createSharedStateEntryId,
  createStreamEntryId,
  type EntityId,
} from "../../util/ids.js";
import { applySharedStateArtifactLifecycleCap } from "./lifecycle-cap.js";

function sharedStateEntry(input: {
  audience: EntityId;
  kind: SharedStateEntryKind;
  rank: number;
  updatedAt: number;
  stateKey?: string;
}): SharedStateEntry {
  const streamEntryId = createStreamEntryId();

  return {
    id: createSharedStateEntryId(),
    audience_entity_id: input.audience,
    state_key: input.stateKey ?? `${input.kind}.entry`,
    kind: input.kind,
    text: `${input.kind} entry ${input.rank}`,
    owner_entity_id: null,
    provenance_stream_entry_ids: [streamEntryId],
    last_updated_stream_entry_ids: [streamEntryId],
    created_at: input.updatedAt,
    last_updated_at: input.updatedAt,
    last_updated_turn_global: null,
    superseded_by_id: null,
    rank: input.rank,
    canonicalizes: {
      goal_ids: [],
      commitment_ids: [],
      action_ids: [],
      open_question_ids: [],
    },
  };
}

function sharedStateArtifact(entries: readonly SharedStateEntry[]): SharedStateArtifact {
  return {
    audience_entity_id: entries[0]?.audience_entity_id ?? createEntityId(),
    record_version: 1,
    created_at: 1,
    updated_at: 1,
    last_compiled_at: 1,
    last_compiled_stream_entry_id: null,
    entries: [...entries],
  };
}

describe("applySharedStateArtifactLifecycleCap", () => {
  it("reserves newest state changes while applying kind soft caps", () => {
    const audience = createEntityId();
    const locked = Array.from({ length: 14 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: index,
        updatedAt: 1_000 + index,
      }),
    );
    const live = Array.from({ length: 10 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "live",
        rank: 100 + index,
        updatedAt: 10_000 + index,
      }),
    );
    const invalidated = Array.from({ length: 5 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "invalidated",
        rank: 200 + index,
        updatedAt: 2_000 + index,
      }),
    );
    const entries = [...locked, ...live, ...invalidated];
    const byId = new Map(entries.map((entry) => [entry.id, entry]));

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: [],
      nowMs: 20_000,
      options: {
        maxActiveEntries: 25,
        newestStateChangeReservedSlots: 3,
        kindSoftCaps: {
          locked: 14,
          live: 8,
          invalidated: 3,
          tentative: 2,
        },
      },
    });
    const pruned = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id)?.kind);

    expect(capped.postPlanActiveEntryCount).toBe(25);
    expect(capped.newestReservedEntryCount).toBe(3);
    expect(pruned.filter((kind) => kind === "live")).toHaveLength(2);
    expect(pruned.filter((kind) => kind === "invalidated")).toHaveLength(2);
    expect(pruned.filter((kind) => kind === "locked")).toHaveLength(0);
  });

  it("prunes dormant before low-salience before live when active entries exceed the cap", () => {
    const audience = createEntityId();
    const entries = (
      [
        "dormant_live",
        "low_salience_live",
        "live",
        "tentative",
        "invalidated",
        "locked",
      ] as const
    ).map((kind, index) =>
      sharedStateEntry({
        audience,
        kind,
        rank: index,
        updatedAt: 1_000 + index,
      }),
    );
    const byId = new Map(entries.map((entry) => [entry.id, entry]));

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: [],
      nowMs: 20_000,
      options: {
        maxActiveEntries: 3,
        newestStateChangeReservedSlots: 0,
        kindSoftCaps: {
          locked: 0,
          live: 0,
          low_salience_live: 0,
          dormant_live: 0,
          tentative: 0,
          invalidated: 0,
        },
      },
    });
    const prunedKinds = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id)?.kind);

    expect(prunedKinds).toEqual(["dormant_live", "low_salience_live", "live"]);
  });

  it("skips bands at their soft cap and prunes the last band in the order instead", () => {
    const audience = createEntityId();
    // The prune order is a scan over an over-cap predicate, not a priority queue: a band at or
    // under its soft cap is skipped entirely, however old its entries are, so the band listed
    // last can be the only one ever drawn from.
    const dormant = sharedStateEntry({
      audience,
      kind: "dormant_live",
      rank: 7,
      updatedAt: 1_000,
      stateKey: "oldest.entry.of.the.whole.set",
    });
    const tentative = Array.from({ length: 2 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "tentative",
        rank: 17 + index,
        updatedAt: 2_000 + index,
      }),
    );
    const locked = Array.from({ length: 38 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: 37 + index,
        updatedAt: 3_000 + index,
        stateKey: `locked.entry.${index}`,
      }),
    );
    const entries = [dormant, ...tentative, ...locked];
    const byId = new Map(entries.map((entry) => [entry.id, entry]));

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: [],
      nowMs: 20_000,
    });
    const pruned = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id));

    expect(capped.maxActiveEntries).toBe(40);
    expect(capped.postPlanActiveEntryCount).toBe(40);
    // dormant_live is first in the prune order and holds the oldest entry in the set, but its
    // band sits exactly at its soft cap of 1, so it is never a candidate.
    expect(pruned.map((entry) => entry?.kind)).toEqual(["locked"]);
    expect(pruned.map((entry) => entry?.state_key)).toEqual(["locked.entry.0"]);
    // No live entries exist, and the reservation filters on kind "live", so the three reserved
    // slots protect nothing here.
    expect(capped.newestReservedEntryCount).toBe(0);
  });

  it("prunes an over-cap earlier band before an older floor in a later band", () => {
    const audience = createEntityId();
    const locked = Array.from({ length: 39 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: index,
        updatedAt: 1_000 + index,
        stateKey: `locked.entry.${index}`,
      }),
    );
    const dormant = Array.from({ length: 2 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "dormant_live",
        rank: 100 + index,
        updatedAt: 9_000 + index,
        stateKey: `dormant.entry.${index}`,
      }),
    );
    const entries = [...locked, ...dormant];
    const byId = new Map(entries.map((entry) => [entry.id, entry]));

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: [],
      nowMs: 20_000,
    });
    const pruned = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id));

    expect(capped.postPlanActiveEntryCount).toBe(40);
    // Both bands are over cap. Every locked entry is older than both dormant entries, but band
    // selection runs before the within-band comparator, so the newer band loses.
    expect(pruned.map((entry) => entry?.state_key)).toEqual(["dormant.entry.0"]);
  });
});
