import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase, type SqliteDatabase } from "../../storage/sqlite/index.js";
import {
  collectInactiveStreamEntryRefs,
  filterActiveStreamEntries,
  isAbortedTurnMarker,
  isQuarantinedUserEntryMarker,
  streamEntryIsActive,
} from "../../stream/turn-status.js";
import {
  isEpisodicSourceEntry,
  isNarrativeStreamEntry,
  type StreamEntry,
} from "../../stream/types.js";
import { FixedClock } from "../../util/clock.js";
import {
  createEntityId,
  createObservedEventId,
  createSessionId,
  createStreamEntryId,
  type ObservedEventId,
  type SessionId,
} from "../../util/ids.js";
import { observedEventMigrations } from "./migrations.js";
import { ObservedEventRepository, type ObservedEventRecordInput } from "./repository.js";
import { observedEventSchema } from "./types.js";

const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

function openHarness(prefix: string): {
  db: SqliteDatabase;
  repository: ObservedEventRepository;
} {
  const tempDir = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(tempDir);
  const db = openDatabase(join(tempDir, "observed-events.db"), {
    migrations: observedEventMigrations,
  });

  return {
    db,
    repository: new ObservedEventRepository({
      db,
      clock: new FixedClock(2_000),
    }),
  };
}

function recordInput(
  sessionId: SessionId,
  overrides: Partial<ObservedEventRecordInput> = {},
): ObservedEventRecordInput {
  return {
    occurredAt: 1_000,
    sessionId,
    stance: "rejected_frame",
    taint: "quarantined",
    beliefEffect: "unchanged",
    classificationKind: "frame_assignment_claim",
    disclosureClass: "social_observed",
    interactionText: "The speaker repeatedly pushed a rejected kinship frame.",
    recurrenceKey: "session:speaker:frame_assignment_claim:kinship-frame",
    sourceStreamEntryIds: [createStreamEntryId()],
    ...overrides,
  };
}

describe("ObservedEventRepository", () => {
  it("round-trips observed events and keeps recurrence upserts first-seen stable", () => {
    const { db, repository } = openHarness("borg-observed-events-");
    const sessionId = createSessionId();
    const sourceEntryId = createStreamEntryId();
    const retrySourceEntryId = createStreamEntryId();
    const speakerEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const sourceEntityId = createEntityId();

    const created = repository.record(
      recordInput(sessionId, {
        occurredAt: 1_000,
        interactionText: "Sol rejected the pushed social frame.",
        recurrenceKey: "session:peer:frame_assignment_claim:hidden-family",
        speakerEntityId,
        audienceEntityId,
        sourceEntityId,
        sourceStreamEntryIds: [sourceEntryId, sourceEntryId],
      }),
    );
    const repushed = repository.record(
      recordInput(sessionId, {
        id: createObservedEventId(),
        occurredAt: 5_000,
        stance: "accepted_frame",
        taint: "none",
        beliefEffect: "updated",
        classificationKind: "roleplay_inversion",
        interactionText: "This retry should not overwrite the first rationale.",
        recurrenceKey: "session:peer:frame_assignment_claim:hidden-family",
        speakerEntityId: createEntityId(),
        audienceEntityId: createEntityId(),
        sourceEntityId: createEntityId(),
        sourceStreamEntryIds: [retrySourceEntryId],
        now: 6_000,
      }),
    );

    expect(repository.get(created.id as ObservedEventId)).toEqual(repushed);
    expect(repushed).toMatchObject({
      id: created.id,
      occurred_at: 1_000,
      session_id: sessionId,
      stance: "rejected_frame",
      taint: "quarantined",
      belief_effect: "unchanged",
      classification_kind: "frame_assignment_claim",
      disclosure_class: "social_observed",
      interaction_text: "Sol rejected the pushed social frame.",
      recurrence_key: "session:peer:frame_assignment_claim:hidden-family",
      recurrence_count: 2,
      last_seen_at: 5_000,
      speaker_entity_id: speakerEntityId,
      audience_entity_id: audienceEntityId,
      source_entity_id: sourceEntityId,
      source_stream_entry_ids: [sourceEntryId],
      created_at: 2_000,
      updated_at: 6_000,
    });
    expect(db.prepare("SELECT COUNT(*) AS count FROM observed_events").get()).toEqual({
      count: 1,
    });

    db.close();
  });

  it("keeps distinct recurrence keys separate and lists newest last-seen rows first", () => {
    const { db, repository } = openHarness("borg-observed-events-distinct-");
    const sessionId = createSessionId();

    repository.record(
      recordInput(sessionId, {
        occurredAt: 1_000,
        interactionText: "First distinct frame.",
        recurrenceKey: "session:peer:first-frame",
      }),
    );
    repository.record(
      recordInput(sessionId, {
        occurredAt: 5_000,
        interactionText: "Second distinct frame.",
        recurrenceKey: "session:peer:second-frame",
      }),
    );

    expect(db.prepare("SELECT COUNT(*) AS count FROM observed_events").get()).toEqual({
      count: 2,
    });
    expect(
      repository
        .listRecentForSession({
          sessionId,
          disclosureClass: "social_observed",
          sinceMs: 0,
          limit: 10,
        })
        .map((row) => row.interactionText),
    ).toEqual(["Second distinct frame.", "First distinct frame."]);

    db.close();
  });

  it("validates observed event rows at the read boundary", () => {
    const { db, repository } = openHarness("borg-observed-events-invalid-");
    const created = repository.record(recordInput(createSessionId()));

    db.prepare("UPDATE observed_events SET source_stream_entry_ids = ? WHERE id = ?").run(
      "not json",
      created.id,
    );

    expect(() => repository.get(created.id)).toThrow(/observed event/i);

    db.close();
  });

  it("keeps observed-event ids inert for stream active-set and source-entry gates", () => {
    const sessionId = createSessionId();
    const observedEventId = createObservedEventId();
    const userEntryId = createStreamEntryId();
    const markerEntryId = createStreamEntryId();
    const entries: StreamEntry[] = [
      {
        id: userEntryId,
        timestamp: 1_000,
        kind: "user_msg",
        content: "current user turn",
        session_id: sessionId,
        sender_entity_id: null,
        reply_target_entity_id: null,
        compressed: false,
      },
      {
        id: markerEntryId,
        timestamp: 1_001,
        kind: "internal_event",
        turn_id: "turn-observed-event",
        content: {
          event: "observed_event_recorded",
          observed_event_id: observedEventId,
          cited_stream_entry_ids: [observedEventId],
          interaction_text: "Sol recorded a rejected social frame.",
        },
        session_id: sessionId,
        sender_entity_id: null,
        reply_target_entity_id: null,
        compressed: false,
      },
    ];
    const marker = entries[1]!;
    const refs = collectInactiveStreamEntryRefs(entries);

    expect(refs.streamEntryIds.has(observedEventId)).toBe(false);
    expect(refs.turnIds.has(observedEventId)).toBe(false);
    expect(refs.streamEntryIds.has(markerEntryId)).toBe(false);
    expect(isAbortedTurnMarker(marker)).toBe(false);
    expect(isQuarantinedUserEntryMarker(marker)).toBe(false);
    expect(streamEntryIsActive(marker, refs)).toBe(true);
    expect(filterActiveStreamEntries(entries).map((entry) => entry.id)).toEqual([
      userEntryId,
      markerEntryId,
    ]);
    expect(isNarrativeStreamEntry(marker)).toBe(false);
    expect(isEpisodicSourceEntry(marker)).toBe(false);
  });

  it("stores stance rationale text without exposing a raw-claim field", () => {
    const { db, repository } = openHarness("borg-observed-events-rationale-");
    const sessionId = createSessionId();
    const rationale = "rozmówca wielokrotnie forsował ramę pokrewieństwa, którą odrzuciłem";
    const created = repository.record(
      recordInput(sessionId, {
        interactionText: rationale,
        recurrenceKey: "session:peer:polish-rationale",
      }),
    );
    const rawClaimRejectedByType = {
      ...recordInput(sessionId, {
        recurrenceKey: "session:peer:raw-claim-type-check",
      }),
      // @ts-expect-error raw claims are intentionally not accepted at this boundary.
      rawClaim: "masz ukrytą siostrę",
    } satisfies ObservedEventRecordInput;
    const assertedClaimRejectedByType = {
      ...recordInput(sessionId, {
        recurrenceKey: "session:peer:asserted-claim-type-check",
      }),
      // @ts-expect-error asserted claims are intentionally not accepted at this boundary.
      assertedClaim: "masz ukrytą siostrę",
    } satisfies ObservedEventRecordInput;

    void rawClaimRejectedByType;
    void assertedClaimRejectedByType;

    expect(created.interaction_text).toBe(rationale);
    expect(repository.get(created.id)?.interaction_text).toBe(rationale);
    expect(observedEventSchema.keyof().options).not.toContain("rawClaim");
    expect(observedEventSchema.keyof().options).not.toContain("assertedClaim");
    expect(observedEventSchema.keyof().options).not.toContain("raw_claim");
    expect(observedEventSchema.keyof().options).not.toContain("asserted_claim");

    db.close();
  });
});
