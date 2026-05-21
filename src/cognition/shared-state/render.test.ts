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
import { summarizeSharedStateArtifactRender } from "./render.js";

function entry(input: {
  audience: EntityId;
  kind: SharedStateEntryKind;
  rank: number;
  updatedAt: number;
}): SharedStateEntry {
  const streamId = createStreamEntryId();

  return {
    id: createSharedStateEntryId(),
    audience_entity_id: input.audience,
    state_key: `${input.kind}.state`,
    kind: input.kind,
    text: `${input.kind} state ${input.rank}`,
    owner_entity_id: null,
    provenance_stream_entry_ids: [streamId],
    last_updated_stream_entry_ids: [streamId],
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

function artifact(entries: readonly SharedStateEntry[]): SharedStateArtifact {
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

describe("summarizeSharedStateArtifactRender", () => {
  it("reports newest reserved live entries in the render summary", () => {
    const audience = createEntityId();
    const entries = [
      ...Array.from({ length: 14 }, (_, index) =>
        entry({ audience, kind: "locked", rank: index, updatedAt: 1_000 + index }),
      ),
      ...Array.from({ length: 10 }, (_, index) =>
        entry({ audience, kind: "live", rank: 100 + index, updatedAt: 10_000 + index }),
      ),
      ...Array.from({ length: 5 }, (_, index) =>
        entry({ audience, kind: "pending", rank: 200 + index, updatedAt: 2_000 + index }),
      ),
    ];

    const summary = summarizeSharedStateArtifactRender(artifact(entries), {
      maxEntries: 25,
      maxTokens: 50_000,
      reservedSlots: {
        live: 8,
        pending: 3,
        invalidated: 3,
      },
      lockedMaxEntries: 14,
      newestStateChangeReservedSlots: 3,
    });

    expect(summary.renderedEntryCount).toBe(25);
    expect(summary.renderedByKind.live).toBe(10);
    expect(summary.newestReservedEntryCount).toBe(3);
  });
});
