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
  createdAt?: number;
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
    created_at: input.createdAt ?? input.updatedAt,
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

function lifecycleCapFixture(
  audience: EntityId,
  counts: Partial<Record<SharedStateEntryKind, number>>,
): SharedStateEntry[] {
  const entries: SharedStateEntry[] = [];

  for (const [kind, count] of Object.entries(counts) as [SharedStateEntryKind, number][]) {
    for (let index = 0; index < count; index += 1) {
      entries.push(
        sharedStateEntry({
          audience,
          kind,
          rank: entries.length,
          updatedAt: 1_000 + entries.length,
          stateKey: `${kind}.entry.${index}`,
        }),
      );
    }
  }

  return entries;
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

  it("draws from a band the patch's own add pushed over cap before the band already over it", () => {
    const audience = createEntityId();
    // Band protection is recomputed against post-patch counts, not the counts the previous pass
    // ran on. A band sitting exactly at its soft cap is untouchable for as long as it stays
    // there, so its floor can outlast every entry in a band that is far over -- and a single add
    // of that kind, in the same patch, makes that same floor the first thing drawn. "Protected by
    // its band" and "stalest entry in the artifact" are true of the same entry at once, and which
    // one decides is settled by what the patch adds rather than by anything about the entry.
    const tentative = [1_000, 2_000].map((updatedAt, index) =>
      sharedStateEntry({
        audience,
        kind: "tentative",
        rank: index,
        updatedAt,
        stateKey: `tentative.entry.${index}`,
      }),
    );
    const locked = Array.from({ length: 38 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: 2 + index,
        updatedAt: 3_000 + index,
        stateKey: `locked.entry.${index}`,
      }),
    );
    const entries = [...tentative, ...locked];
    const byId = new Map(entries.map((entry) => [entry.id, entry]));

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: (["locked", "locked", "tentative"] as const).map((kind, index) => ({
        type: "add" as const,
        state_key: `${kind}.entry.added.${index}`,
        kind,
        text: `added ${kind} ${index}`,
        provenance_stream_entry_ids: [createStreamEntryId()],
      })),
      nowMs: 20_000,
    });
    const pruned = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id));

    expect(capped.postPlanActiveEntryCount).toBe(40);
    // Three adds, three evictions -- and they do not all come from one band. The first pass finds
    // tentative at 3 over a cap of 2 and takes its floor; that returns the band to its cap and
    // puts it out of reach again, so the remaining two come from locked, which was over its own
    // cap by fourteen the whole time and still lost the first draw.
    expect(pruned.map((entry) => entry?.kind)).toEqual(["tentative", "locked", "locked"]);
    // tentative.entry.1 is older than all 38 locked entries and is evicted by none of them: age
    // is not what exposes an entry here, its band's count is.
    expect(pruned.map((entry) => entry?.state_key)).toEqual([
      "tentative.entry.0",
      "locked.entry.0",
      "locked.entry.1",
    ]);
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

  it("breaks a last_updated_at tie by rank without letting rank outrank a later update", () => {
    const audience = createEntityId();
    // Rank is not a position in the prune queue. It orders entries only when their
    // last_updated_at values are identical -- i.e. within a single write round -- and it is
    // read ascending, so the lowest rank of a tied round dies first.
    const birthRound = [40, 38, 39].map((rank) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank,
        updatedAt: 1_000,
        stateKey: `birth.round.rank${rank}`,
      }),
    );
    const newer = Array.from({ length: 40 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: 41 + index,
        updatedAt: 3_000 + index,
        stateKey: `locked.entry.${index}`,
      }),
    );
    const entries = [...birthRound, ...newer];
    const byId = new Map(entries.map((entry) => [entry.id, entry]));

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: [],
      nowMs: 20_000,
    });
    const pruned = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id)?.state_key);

    expect(capped.postPlanActiveEntryCount).toBe(40);
    // Every entry that ranks above 40 outlives the round tied at 1_000, and inside that round
    // rank decides the order of death.
    expect(pruned).toEqual(["birth.round.rank38", "birth.round.rank39", "birth.round.rank40"]);
  });

  it("lifts an entry out of its birth round when an update moves last_updated_at", () => {
    const audience = createEntityId();
    // Same three entries, except the highest-ranked one was updated in place afterwards. An
    // update rewrites last_updated_at while created_at and rank stay put, and last_updated_at
    // dominates the comparator -- so the touched entry leaves the tie its siblings still die in,
    // and a numerically worse rank survives while lower ranks are pruned.
    const untouched = [38, 39].map((rank) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank,
        updatedAt: 1_000,
        stateKey: `birth.round.rank${rank}`,
      }),
    );
    const touched = sharedStateEntry({
      audience,
      kind: "locked",
      rank: 40,
      updatedAt: 9_000,
      createdAt: 1_000,
      stateKey: "birth.round.rank40",
    });
    const newer = Array.from({ length: 40 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: 41 + index,
        updatedAt: 3_000 + index,
        stateKey: `locked.entry.${index}`,
      }),
    );
    const entries = [...untouched, touched, ...newer];
    const byId = new Map(entries.map((entry) => [entry.id, entry]));

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: [],
      nowMs: 20_000,
    });
    const pruned = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id)?.state_key);

    expect(capped.postPlanActiveEntryCount).toBe(40);
    // The third death is the oldest of the untouched newer entries, not the updated sibling.
    expect(pruned).toEqual(["birth.round.rank38", "birth.round.rank39", "locked.entry.0"]);
    expect(pruned).not.toContain("birth.round.rank40");
  });

  it("leaves a below-cap active set where it is even when a kind sits far over its soft cap", () => {
    const audience = createEntityId();
    // The kind soft caps do not prune on their own -- they only order the draw once the global
    // ceiling is exceeded. locked at 36 against a soft cap of 24 is twelve over and still costs
    // nothing, because the cap loop is never entered.
    const entries = lifecycleCapFixture(audience, { locked: 36, dormant_live: 1, tentative: 2 });

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: [],
      nowMs: 20_000,
    });

    expect(entries).toHaveLength(39);
    expect(capped.operations.filter((operation) => operation.type === "prune")).toHaveLength(0);
    // Nothing tops the set back up either: the cap is a ceiling, not a target, so 39 is a stable
    // resting state and not a transient on the way to 40.
    expect(capped.postPlanActiveEntryCount).toBe(39);
    expect(capped.maxActiveEntries).toBe(40);
    expect(capped.overCapDelta).toBe(0);
  });

  it("starts pruning when the active set exceeds the cap, not when it reaches it", () => {
    const audience = createEntityId();
    const atCap = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(
        lifecycleCapFixture(audience, { locked: 37, dormant_live: 1, tentative: 2 }),
      ),
      operations: [],
      nowMs: 20_000,
    });

    // `while (activeEntries.length > maxActiveEntries)` is strict: equality is not over-cap.
    expect(atCap.operations.filter((operation) => operation.type === "prune")).toHaveLength(0);
    expect(atCap.postPlanActiveEntryCount).toBe(40);

    const overCap = lifecycleCapFixture(audience, { locked: 38, dormant_live: 1, tentative: 2 });
    const byId = new Map(overCap.map((entry) => [entry.id, entry]));
    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(overCap),
      operations: [],
      nowMs: 20_000,
    });
    const pruned = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id));

    // One over the ceiling costs exactly one entry, drawn from the only band above its soft cap
    // -- dormant_live and tentative are each at theirs, so the loop skips both and reaches locked.
    expect(pruned).toHaveLength(1);
    expect(pruned[0]?.kind).toBe("locked");
    expect(pruned[0]?.state_key).toBe("locked.entry.0");
    expect(capped.postPlanActiveEntryCount).toBe(40);
  });

  it("reserves the newest state changes by kind rather than by recency", () => {
    const audience = createEntityId();
    const capForNewestKind = (kind: "tentative" | "live") => {
      const locked = Array.from({ length: 24 }, (_, index) =>
        sharedStateEntry({
          audience,
          kind: "locked",
          rank: index,
          updatedAt: 1_000 + index,
          stateKey: `locked.entry.${index}`,
        }),
      );
      // The three newest entries in the store, and the entirety of an over-cap band.
      const newest = Array.from({ length: 3 }, (_, index) =>
        sharedStateEntry({
          audience,
          kind,
          rank: 100 + index,
          updatedAt: 9_000 + index,
          stateKey: `newest.entry.${index}`,
        }),
      );
      const entries = [...locked, ...newest];
      const byId = new Map(entries.map((entry) => [entry.id, entry]));
      const capped = applySharedStateArtifactLifecycleCap({
        previousArtifact: sharedStateArtifact(entries),
        operations: [],
        nowMs: 20_000,
        options: {
          maxActiveEntries: 26,
          newestStateChangeReservedSlots: 3,
          kindSoftCaps: { locked: 24, live: 2, tentative: 2 },
        },
      });

      return {
        reserved: capped.newestReservedEntryCount,
        pruned: capped.operations
          .filter((operation) => operation.type === "prune")
          .map((operation) => byId.get(operation.id)?.state_key),
      };
    };

    // The reservation filters on `kind === "live"`, so a store with an empty live band reserves
    // nothing however fresh its entries are: the draw reaches past 24 older locked entries and
    // takes the third-newest entry in the store.
    expect(capForNewestKind("tentative")).toEqual({
      reserved: 0,
      pruned: ["newest.entry.0"],
    });

    // Same stamps, same census, one word different in the kind field -- now all three are
    // reserved, the band's effective count falls under its soft cap, and the entry that dies is
    // the oldest in the store rather than one of the newest.
    expect(capForNewestKind("live")).toEqual({
      reserved: 3,
      pruned: ["locked.entry.0"],
    });
  });

  it("carries a transitioned entry's original stamp into its new band, where an update would not", () => {
    const audience = createEntityId();
    const planFor = (move: "transition_kind" | "update") => {
      const locked = Array.from({ length: 6 }, (_, index) =>
        sharedStateEntry({
          audience,
          kind: "locked",
          rank: index,
          updatedAt: 5_000 + index,
          stateKey: `locked.entry.${index}`,
        }),
      );
      const moved = sharedStateEntry({
        audience,
        kind: "live",
        rank: 100,
        updatedAt: 1_000,
        stateKey: "moved.entry",
      });
      const entries = [...locked, moved];
      const byId = new Map(entries.map((entry) => [entry.id, entry]));
      const capped = applySharedStateArtifactLifecycleCap({
        previousArtifact: sharedStateArtifact(entries),
        operations: [
          move === "transition_kind"
            ? { type: "transition_kind", id: moved.id, kind: "locked" }
            : {
                type: "update",
                id: moved.id,
                state_key: "moved.entry",
                kind: "locked",
                last_updated_stream_entry_ids: moved.last_updated_stream_entry_ids,
              },
        ],
        nowMs: 9_000,
        options: {
          maxActiveEntries: 6,
          newestStateChangeReservedSlots: 0,
          kindSoftCaps: { locked: 5 },
        },
      });

      return capped.operations
        .filter((operation) => operation.type === "prune")
        .map((operation) => byId.get(operation.id)?.state_key);
    };

    // The cap planner materializes its own copy of the patch, and its `transition_kind` case
    // rewrites `kind` and nothing else -- so the entry arrives in `locked` still stamped 1_000,
    // oldest in the band it just joined and first out of it.
    expect(planFor("transition_kind")).toEqual(["moved.entry"]);

    // The same band move spelled as an update takes `last_updated_at ?? nowMs`, which puts the
    // entry at the back of the queue and costs a locked entry that never moved.
    expect(planFor("update")).toEqual(["locked.entry.0"]);
  });

  it("takes the top of the rendered index first when only part of a tied round dies", () => {
    const audience = createEntityId();
    // One compile pass stamps every entry it writes with the same `last_updated_at`, so a band
    // sharing one stamp is the settled state of a real artifact rather than an edge case, and
    // rank is then the only discriminator. The two existing tie tests kill whole rounds, where
    // the direction only reorders the plan; this one caps mid-round so the direction decides
    // which entries survive. Ascending rank means the survivors are the tail of the round --
    // i.e. the entries the repository lists last, since it too reads `rank ASC`.
    const tied = Array.from({ length: 4 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: index,
        updatedAt: 1_000,
        stateKey: `locked.tied.${index}`,
      }),
    );
    const byId = new Map(tied.map((entry) => [entry.id, entry]));

    const capped = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(tied),
      operations: [],
      nowMs: 2_000,
      options: {
        maxActiveEntries: 2,
        newestStateChangeReservedSlots: 0,
        kindSoftCaps: { locked: 2 },
      },
    });

    expect(
      capped.operations
        .filter((operation) => operation.type === "prune")
        .map((operation) => byId.get(operation.id)?.state_key),
    ).toEqual(["locked.tied.0", "locked.tied.1"]);
  });

  it("records which scan drew each eviction, because the first one only sees over-cap kinds", () => {
    const audience = createEntityId();
    // The staler entry is the only one of its kind and sits at that kind's cap, so it is never a
    // candidate while another kind is over its own -- the shape a live artifact reaches routinely.
    // Read without the pass, the resulting record looks like the comparator ignored the oldest
    // entry in the artifact.
    const oldestOfCappedKind = sharedStateEntry({
      audience,
      kind: "tentative",
      rank: 0,
      updatedAt: 1_000,
      stateKey: "tentative.oldest",
    });
    const overCapKind = Array.from({ length: 3 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: index + 1,
        updatedAt: 2_000 + index,
        stateKey: `locked.newer.${index}`,
      }),
    );
    const entries = [oldestOfCappedKind, ...overCapKind];
    const byId = new Map(entries.map((entry) => [entry.id, entry]));

    const overSoftCap = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(entries),
      operations: [],
      nowMs: 3_000,
      options: {
        maxActiveEntries: 3,
        newestStateChangeReservedSlots: 0,
        kindSoftCaps: { locked: 2, tentative: 1 },
      },
    });

    expect(
      overSoftCap.operations
        .filter((operation) => operation.type === "prune")
        .map((operation) => byId.get(operation.id)?.state_key),
    ).toEqual(["locked.newer.0"]);
    expect(
      overSoftCap.capEvictions.map((eviction) => [eviction.state_key, eviction.selection_pass]),
    ).toEqual([["locked.newer.0", "over_soft_cap"]]);

    // Drop one locked entry and no kind exceeds its cap, so the same artifact shape now draws on
    // the prune order alone -- which reaches `tentative` first and takes the entry the previous
    // pass could not touch. Same comparator, different pool.
    const withoutOverCapKind = [oldestOfCappedKind, ...overCapKind.slice(0, 2)];

    const anyKind = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact(withoutOverCapKind),
      operations: [],
      nowMs: 3_000,
      options: {
        maxActiveEntries: 2,
        newestStateChangeReservedSlots: 0,
        kindSoftCaps: { locked: 2, tentative: 1 },
      },
    });

    expect(
      anyKind.capEvictions.map((eviction) => [eviction.state_key, eviction.selection_pass]),
    ).toEqual([["tentative.oldest", "any_kind"]]);
  });
});

describe("shipped kind soft caps", () => {
  it("draws the aging ladder's bottom rung before locked entries staler than it", () => {
    const audience = createEntityId();
    // Every other cap case in this file supplies its own kindSoftCaps, so none of them pins what the
    // defaults couple: `dormant_live` is the terminal kind of the lifecycle ladder, and it is also
    // first in the prune order with the smallest cap. A demoted entry therefore reaches the head of
    // the eviction queue while much older locked rows are still nowhere near a draw.
    const dormant = [9_000, 9_100, 9_200].map((updatedAt, index) =>
      sharedStateEntry({
        audience,
        kind: "dormant_live",
        rank: 40 + index,
        updatedAt,
        stateKey: `ladder.bottom.${index}`,
      }),
    );
    const locked = [1_000, 1_001, 1_002].map((updatedAt, index) =>
      sharedStateEntry({
        audience,
        kind: "locked",
        rank: index,
        updatedAt,
        stateKey: `locked.older.${index}`,
      }),
    );

    const result = applySharedStateArtifactLifecycleCap({
      previousArtifact: sharedStateArtifact([...locked, ...dormant]),
      operations: [],
      nowMs: 10_000,
      options: { maxActiveEntries: 4 },
    });

    expect(
      result.capEvictions.map((eviction) => [
        eviction.state_key,
        eviction.kind,
        eviction.selection_pass,
      ]),
    ).toEqual([
      ["ladder.bottom.0", "dormant_live", "over_soft_cap"],
      ["ladder.bottom.1", "dormant_live", "over_soft_cap"],
    ]);
  });
});
