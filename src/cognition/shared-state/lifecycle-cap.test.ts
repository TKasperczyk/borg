import { describe, expect, it } from "vitest";

import type {
  SharedStateArtifact,
  SharedStateEntry,
  SharedStateEntryKind,
} from "../../memory/decision-artifacts/index.js";
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
}): SharedStateEntry {
  const streamEntryId = createStreamEntryId();

  return {
    id: createSharedStateEntryId(),
    audience_entity_id: input.audience,
    state_key: `${input.kind}.entry`,
    kind: input.kind,
    text: `${input.kind} entry ${input.rank}`,
    owner_entity_id: null,
    provenance_stream_entry_ids: [streamEntryId],
    last_updated_stream_entry_ids: [streamEntryId],
    created_at: input.updatedAt,
    last_updated_at: input.updatedAt,
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
    const pending = Array.from({ length: 5 }, (_, index) =>
      sharedStateEntry({
        audience,
        kind: "pending",
        rank: 200 + index,
        updatedAt: 2_000 + index,
      }),
    );
    const entries = [...locked, ...live, ...pending];
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
          pending: 3,
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
    expect(pruned.filter((kind) => kind === "pending")).toHaveLength(2);
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
        "pending",
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
        maxActiveEntries: 4,
        newestStateChangeReservedSlots: 0,
        kindSoftCaps: {
          locked: 0,
          live: 0,
          low_salience_live: 0,
          dormant_live: 0,
          tentative: 0,
          invalidated: 0,
          pending: 0,
        },
      },
    });
    const prunedKinds = capped.operations
      .filter((operation) => operation.type === "prune")
      .map((operation) => byId.get(operation.id)?.kind);

    expect(prunedKinds).toEqual(["dormant_live", "low_salience_live", "live"]);
  });
});
