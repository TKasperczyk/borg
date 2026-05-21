import { describe, expect, it } from "vitest";

import type {
  SharedStateArtifact,
  SharedStateEntry,
} from "../../memory/decision-artifacts/index.js";
import {
  createEntityId,
  createSharedStateEntryId,
  createStreamEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { EmitSharedStatePatch } from "./schema.js";
import { allowedCanonicalizationIds, normalizePatch } from "./patch-validation.js";

const EMPTY_CANONICALIZES = {
  goal_ids: [],
  commitment_ids: [],
  action_ids: [],
  open_question_ids: [],
};

function makeArtifact(input: {
  audienceEntityId: EntityId;
  entries: readonly SharedStateEntry[];
}): SharedStateArtifact {
  return {
    audience_entity_id: input.audienceEntityId,
    record_version: 1,
    created_at: 1_000,
    updated_at: 1_000,
    last_compiled_at: null,
    last_compiled_stream_entry_id: null,
    entries: [...input.entries],
  };
}

function makeEntry(input: {
  audienceEntityId: EntityId;
  sourceStreamEntryId: StreamEntryId;
  stateKey: string | null;
  kind?: SharedStateEntry["kind"];
  rank?: number;
  text?: string;
}): SharedStateEntry {
  const rank = input.rank ?? 0;

  return {
    id: createSharedStateEntryId(),
    audience_entity_id: input.audienceEntityId,
    state_key: input.stateKey,
    kind: input.kind ?? "live",
    text: input.text ?? `Entry ${rank}`,
    owner_entity_id: null,
    provenance_stream_entry_ids: [input.sourceStreamEntryId],
    last_updated_stream_entry_ids: [input.sourceStreamEntryId],
    created_at: 1_000 + rank,
    last_updated_at: 1_000 + rank,
    superseded_by_id: null,
    rank,
    canonicalizes: EMPTY_CANONICALIZES,
  };
}

function normalizeKeyedPatch(input: {
  previousEntries: readonly SharedStateEntry[];
  operations: EmitSharedStatePatch["operations"];
  audienceEntityId: EntityId;
  sourceStreamEntryId: StreamEntryId;
}) {
  const selfEntityId = createEntityId();
  const speakerEntityId = createEntityId();

  return normalizePatch({
    patch: {
      operations: input.operations,
    },
    previousArtifact: makeArtifact({
      audienceEntityId: input.audienceEntityId,
      entries: input.previousEntries,
    }),
    audienceEntityId: input.audienceEntityId,
    selfEntityId,
    speakerEntityId,
    participants: [],
    allowedSourceStreamEntryIds: new Set([input.sourceStreamEntryId]),
    allowedCanonicalizationIds: allowedCanonicalizationIds(undefined),
    maxLiveEntriesPerKey: 2,
  });
}

function addOperation(input: {
  stateKey: string;
  kind?: SharedStateEntry["kind"];
  sourceStreamEntryId: StreamEntryId;
}): EmitSharedStatePatch["operations"][number] {
  return {
    type: "add",
    state_key: input.stateKey,
    kind: input.kind ?? "live",
    text: "New keyed entry",
    source_stream_entry_ids: [input.sourceStreamEntryId],
  };
}

describe("normalizePatch state_key validation", () => {
  it("accepts a same-key live add below the per-key cap", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 0,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "plan.attendees",
        kind: "live",
      }),
    ]);
  });

  it("rejects a same-key live add when the per-key cap is already reached", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 0,
        }),
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 1,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "live_entry_cap_exceeded_for_key",
        operationType: "add",
        operationIndex: 0,
        stateKey: "plan.attendees",
        currentCount: 2,
        proposedCount: 3,
        maxLiveEntriesPerKey: 2,
      }),
    ]);
  });

  it("does not count legacy null-key entries toward a keyed live add", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: null,
          rank: 0,
        }),
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: null,
          rank: 1,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "plan.attendees",
      }),
    ]);
  });

  it("rejects a live add when a locked entry already owns the state key", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const lockedEntry = makeEntry({
      audienceEntityId,
      sourceStreamEntryId,
      stateKey: "plan.attendees",
      kind: "locked",
      rank: 0,
    });
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [lockedEntry],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.operations).toEqual([]);
    expect(result.rejected).toEqual([
      expect.objectContaining({
        reason: "locked_state_key_collision",
        operationType: "add",
        operationIndex: 0,
        stateKey: "plan.attendees",
        currentCount: 1,
        lockedEntryIds: [lockedEntry.id],
      }),
    ]);
  });

  it("accepts tentative same-key adds when live entries are already at cap", () => {
    const audienceEntityId = createEntityId();
    const sourceStreamEntryId = createStreamEntryId();
    const result = normalizeKeyedPatch({
      audienceEntityId,
      sourceStreamEntryId,
      previousEntries: [
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 0,
        }),
        makeEntry({
          audienceEntityId,
          sourceStreamEntryId,
          stateKey: "plan.attendees",
          rank: 1,
        }),
      ],
      operations: [
        addOperation({
          stateKey: "plan.attendees",
          kind: "tentative",
          sourceStreamEntryId,
        }),
      ],
    });

    expect(result.rejected).toEqual([]);
    expect(result.operations).toEqual([
      expect.objectContaining({
        type: "add",
        state_key: "plan.attendees",
        kind: "tentative",
      }),
    ]);
  });
});
