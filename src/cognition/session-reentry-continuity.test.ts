import { describe, expect, it } from "vitest";

import {
  makeLiveSharedStateEntry,
  makeLockedSharedStateEntry,
  makeSharedStateArtifact,
  makeTentativeSharedStateEntry,
} from "../test-support/factories/shared-state.js";
import { createEntityId, createSharedStateEntryId, createStreamEntryId } from "../util/ids.js";
import {
  SESSION_REENTRY_CONTINUITY_TAG,
  buildSessionReentryContinuityPrompt,
} from "./session-reentry-continuity.js";

describe("buildSessionReentryContinuityPrompt", () => {
  it("does not render after the first user-origin turn", () => {
    const audienceEntityId = createEntityId();
    const artifact = makeSharedStateArtifact([
      makeLockedSharedStateEntry({ audience_entity_id: audienceEntityId }),
    ]);

    const result = buildSessionReentryContinuityPrompt({
      isUserTurn: true,
      priorUserTurnCount: 1,
      audienceEntityId,
      artifact,
    });

    expect(result.promptSection).toBeNull();
    expect(result.summary.status).toBe("not_first_user_turn");
  });

  it("does not render when the audience has no active shared-state entries", () => {
    const audienceEntityId = createEntityId();
    const artifact = makeSharedStateArtifact([], {
      audience_entity_id: audienceEntityId,
      entries: [],
    });

    const result = buildSessionReentryContinuityPrompt({
      isUserTurn: true,
      priorUserTurnCount: 0,
      audienceEntityId,
      artifact,
    });

    expect(result.promptSection).toBeNull();
    expect(result.summary.status).toBe("blank_audience");
    expect(result.summary.activeEntryCount).toBe(0);
  });

  it("renders a compact first-session summary for active audience state", () => {
    const audienceEntityId = createEntityId();
    const firstRef = createStreamEntryId();
    const latestRef = createStreamEntryId();
    const legacyRef = createStreamEntryId();
    const artifact = makeSharedStateArtifact([
      makeLockedSharedStateEntry({
        audience_entity_id: audienceEntityId,
        state_key: "incident.rollback",
        last_updated_at: 1_000,
        last_updated_stream_entry_ids: [firstRef],
      }),
      makeLiveSharedStateEntry({
        audience_entity_id: audienceEntityId,
        state_key: "incident.rollback",
        last_updated_at: 2_000,
        last_updated_stream_entry_ids: [latestRef],
      }),
      makeTentativeSharedStateEntry({
        audience_entity_id: audienceEntityId,
        state_key: "incident.customer-note",
        last_updated_at: 1_500,
      }),
      {
        ...makeLockedSharedStateEntry({
          audience_entity_id: audienceEntityId,
          last_updated_at: 3_000,
          last_updated_stream_entry_ids: [legacyRef],
        }),
        state_key: null,
      },
      makeLockedSharedStateEntry({
        audience_entity_id: audienceEntityId,
        superseded_by_id: createSharedStateEntryId(),
      }),
    ]);

    const result = buildSessionReentryContinuityPrompt({
      isUserTurn: true,
      priorUserTurnCount: 0,
      audienceEntityId,
      artifact,
      nowMs: 183_000,
    });

    expect(result.summary.status).toBe("rendered");
    expect(result.summary.activeEntryCount).toBe(4);
    expect(result.summary.activeKeyedEntryCount).toBe(3);
    expect(result.summary.activeLegacyEntryCount).toBe(1);
    expect(result.summary.activeStateKeyCount).toBe(3);
    expect(result.summary.mostRecentUpdate).toMatchObject({
      stateKey: "legacy",
      kind: "locked",
      lastUpdatedAt: 3_000,
      lastUpdatedStreamEntryId: legacyRef,
    });
    expect(result.promptSection).toContain(`<${SESSION_REENTRY_CONTINUITY_TAG}>`);
    expect(result.promptSection).toContain(
      "This is prior-session carryover for the audience, not evidence that the current speaker remembers, endorsed, or participated in it.",
    );
    expect(result.promptSection).toContain(
      "matched_state_key_buckets=all_active_state_key_buckets active_state_key_bucket_count=3",
    );
    expect(result.promptSection).toContain("active_entry_count=4");
    expect(result.promptSection).toContain("active_keyed_entry_count=3");
    expect(result.promptSection).toContain("active_legacy_unkeyed_entry_count=1");
    expect(result.promptSection).toContain(
      "locked=2 live=1 low_salience_live=0 dormant_live=0 tentative=1 invalidated=0 pending=0",
    );
    expect(result.promptSection).toContain("state_key_bucket=incident.rollback");
    expect(result.promptSection).toContain("state_key_bucket=incident.customer-note");
    expect(result.promptSection).toContain(
      "state_key_bucket=legacy bucket_source=unkeyed_legacy_state entries=1",
    );
    expect(result.promptSection).toContain(
      "most_recent_update_at=1970-01-01T00:00:03.000Z most_recent_relative_age=3m ago",
    );
    expect(result.promptSection).not.toContain("most_recent_update_at=3000");
    expect(result.promptSection).toContain(`most_recent_ref=${legacyRef}`);
  });

  it("renders legacy state when active entries have only null state keys", () => {
    const audienceEntityId = createEntityId();
    const latestRef = createStreamEntryId();
    const legacyEntry = {
      ...makeLiveSharedStateEntry({
        audience_entity_id: audienceEntityId,
        last_updated_at: 4_000,
        last_updated_stream_entry_ids: [latestRef],
      }),
      state_key: null,
    };
    const artifact = makeSharedStateArtifact([legacyEntry]);

    const result = buildSessionReentryContinuityPrompt({
      isUserTurn: true,
      priorUserTurnCount: 0,
      audienceEntityId,
      artifact,
    });

    expect(result.summary.status).toBe("rendered");
    expect(result.summary.activeEntryCount).toBe(1);
    expect(result.summary.activeKeyedEntryCount).toBe(0);
    expect(result.summary.activeLegacyEntryCount).toBe(1);
    expect(result.summary.activeEntriesByKey).toEqual({ legacy: 1 });
    expect(result.promptSection).toContain("state_key_bucket=legacy");
    expect(result.promptSection).toContain("bucket_source=unkeyed_legacy_state");
  });

  it("renders after autonomous-only prior turns when this is the first user turn", () => {
    const audienceEntityId = createEntityId();
    const artifact = makeSharedStateArtifact([
      makeLockedSharedStateEntry({ audience_entity_id: audienceEntityId }),
    ]);

    const result = buildSessionReentryContinuityPrompt({
      isUserTurn: true,
      priorUserTurnCount: 0,
      audienceEntityId,
      artifact,
    });

    expect(result.summary.status).toBe("rendered");
    expect(result.promptSection).toContain(`<${SESSION_REENTRY_CONTINUITY_TAG}>`);
  });
});
