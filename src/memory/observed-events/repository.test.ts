import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
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
import {
  createObservedEventsTableSchema,
  ObservedEventRepository,
  type ObservedEventRecordInput,
} from "./repository.js";
import { observedEventSchema } from "./types.js";

const tempDirs: string[] = [];

class MapEmbeddingClient implements EmbeddingClient {
  constructor(private readonly vectors: Map<string, readonly number[]>) {}

  async embed(text: string): Promise<Float32Array> {
    return Float32Array.from(this.vectors.get(text) ?? [0, 0, 1, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((text) => this.embed(text)));
  }
}

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
  it("treats a repeated fire dedup key as a replay no-op", () => {
    const { db, repository } = openHarness("borg-observed-events-fire-dedup-");
    const sessionId = createSessionId();
    const sourceEntryId = createStreamEntryId();
    const recurrenceKey = `${sessionId}:peer:frame_assignment_claim`;
    const fireDedupKey = `${sessionId}|frame_assignment_claim|${sourceEntryId}`;

    const created = repository.record(
      recordInput(sessionId, {
        recurrenceKey,
        fireDedupKey,
        sourceStreamEntryIds: [sourceEntryId],
      }),
    );
    const replay = repository.record(
      recordInput(sessionId, {
        id: createObservedEventId(),
        occurredAt: 5_000,
        recurrenceKey,
        fireDedupKey,
        interactionText: "Replay text must not overwrite the first rationale.",
        sourceStreamEntryIds: [sourceEntryId],
        now: 6_000,
      }),
    );

    expect(replay).toEqual(created);
    expect(replay.recurrence_count).toBe(1);
    expect(repository.getByFireDedupKey(fireDedupKey)).toEqual(created);
    expect(db.prepare("SELECT COUNT(*) AS count FROM observed_events").get()).toEqual({
      count: 1,
    });

    db.close();
  });

  it("collapses recurring pushes with different fire dedup keys and preserves first-seen fields", () => {
    const { db, repository } = openHarness("borg-observed-events-fire-recurrence-");
    const sessionId = createSessionId();
    const firstSourceEntryId = createStreamEntryId();
    const secondSourceEntryId = createStreamEntryId();
    const recurrenceKey = `${sessionId}:peer:frame_assignment_claim`;
    const firstFireDedupKey = `${sessionId}|frame_assignment_claim|${firstSourceEntryId}`;
    const secondFireDedupKey = `${sessionId}|frame_assignment_claim|${secondSourceEntryId}`;

    const created = repository.record(
      recordInput(sessionId, {
        occurredAt: 1_000,
        recurrenceKey,
        fireDedupKey: firstFireDedupKey,
        interactionText: "Sol rejected the pushed social frame.",
        sourceStreamEntryIds: [firstSourceEntryId],
        now: 2_000,
      }),
    );
    const recurring = repository.record(
      recordInput(sessionId, {
        id: createObservedEventId(),
        occurredAt: 5_000,
        recurrenceKey,
        fireDedupKey: secondFireDedupKey,
        stance: "accepted_frame",
        taint: "none",
        beliefEffect: "updated",
        interactionText: "New push text must not overwrite the first rationale.",
        sourceStreamEntryIds: [secondSourceEntryId],
        now: 6_000,
      }),
    );

    expect(db.prepare("SELECT COUNT(*) AS count FROM observed_events").get()).toEqual({
      count: 1,
    });
    expect(recurring).toMatchObject({
      id: created.id,
      occurred_at: 1_000,
      recurrence_key: recurrenceKey,
      fire_dedup_key: secondFireDedupKey,
      recurrence_count: 2,
      last_seen_at: 5_000,
      stance: "rejected_frame",
      taint: "quarantined",
      belief_effect: "unchanged",
      interaction_text: "Sol rejected the pushed social frame.",
      source_stream_entry_ids: [firstSourceEntryId],
      updated_at: 6_000,
    });
    expect(repository.getByFireDedupKey(secondFireDedupKey)).toEqual(recurring);

    db.close();
  });

  it("preserves legacy recurrence behavior when fire dedup keys are omitted", () => {
    const { db, repository } = openHarness("borg-observed-events-legacy-recurrence-");
    const sessionId = createSessionId();
    const recurrenceKey = `${sessionId}:peer:frame_assignment_claim`;

    repository.record(
      recordInput(sessionId, {
        occurredAt: 1_000,
        recurrenceKey,
      }),
    );
    const recurring = repository.record(
      recordInput(sessionId, {
        occurredAt: 5_000,
        recurrenceKey,
        now: 6_000,
      }),
    );

    expect(db.prepare("SELECT COUNT(*) AS count FROM observed_events").get()).toEqual({
      count: 1,
    });
    expect(recurring.recurrence_count).toBe(2);
    expect(recurring.fire_dedup_key).toBeNull();

    db.close();
  });

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

  it("lists recent rows by speaker across sessions and returns [] for an empty speaker set", () => {
    const { db, repository } = openHarness("borg-observed-events-speaker-recent-");
    const firstSessionId = createSessionId();
    const secondSessionId = createSessionId();
    const speakerEntityId = createEntityId();
    const otherSpeakerEntityId = createEntityId();
    const originAudienceEntityId = createEntityId();

    repository.record(
      recordInput(firstSessionId, {
        occurredAt: 1_000,
        interactionText: "first-session rejection",
        recurrenceKey: "session-one:speaker:frame",
        speakerEntityId,
        audienceEntityId: originAudienceEntityId,
      }),
    );
    repository.record(
      recordInput(secondSessionId, {
        occurredAt: 5_000,
        interactionText: "second-session rejection",
        recurrenceKey: "session-two:speaker:frame",
        speakerEntityId,
        audienceEntityId: createEntityId(),
      }),
    );
    repository.record(
      recordInput(secondSessionId, {
        occurredAt: 6_000,
        interactionText: "other speaker rejection",
        recurrenceKey: "session-two:other-speaker:frame",
        speakerEntityId: otherSpeakerEntityId,
      }),
    );

    expect(
      repository.listRecentBySpeakers({
        speakerEntityIds: [],
        disclosureClass: "social_observed",
        sinceMs: 0,
        limit: 10,
      }),
    ).toEqual([]);
    expect(
      repository
        .listRecentBySpeakers({
          speakerEntityIds: [speakerEntityId],
          disclosureClass: "social_observed",
          sinceMs: 0,
          limit: 10,
        })
        .map((row) => ({
          text: row.interactionText,
          speakerEntityId: row.speakerEntityId,
          audienceEntityId: row.audienceEntityId,
        })),
    ).toEqual([
      {
        text: "second-session rejection",
        speakerEntityId,
        audienceEntityId: expect.any(String),
      },
      {
        text: "first-session rejection",
        speakerEntityId,
        audienceEntityId: originAudienceEntityId,
      },
    ]);

    db.close();
  });

  it("embeds observed-event stance rationale and searches by vector globally", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-vector-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const query = "hidden cousin pressure topic";
    const matchingRationale = "Sol rejected a quarantined hidden cousin pressure frame.";
    const otherRationale = "Sol rejected an unrelated scheduling pressure frame.";
    const embeddingClient = new MapEmbeddingClient(
      new Map([
        [query, [1, 0, 0, 0]],
        [matchingRationale, [1, 0, 0, 0]],
        [otherRationale, [0, 1, 0, 0]],
      ]),
    );
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const table = await store.openTable({
      name: "observed_events",
      schema: createObservedEventsTableSchema(4),
    });
    const repository = new ObservedEventRepository({
      db,
      table,
      embeddingClient,
      clock: new FixedClock(2_000),
    });
    const sessionId = createSessionId();
    const speakerEntityId = createEntityId();

    const matching = repository.record(
      recordInput(sessionId, {
        interactionText: matchingRationale,
        recurrenceKey: "session:speaker:hidden-cousin-pressure",
        speakerEntityId,
      }),
    );
    repository.record(
      recordInput(sessionId, {
        interactionText: otherRationale,
        recurrenceKey: "session:speaker:scheduling-pressure",
        speakerEntityId: createEntityId(),
      }),
    );
    await repository.waitForPendingEmbeddings();

    const results = await repository.searchByVector(await embeddingClient.embed(query), {
      minSimilarity: 0.8,
      limit: 5,
    });

    expect(results.map((result) => result.event.id)).toEqual([matching.id]);
    expect(results[0]?.event.interactionText).toBe(matchingRationale);
    expect(results[0]?.event.speakerEntityId).toBe(speakerEntityId);

    db.close();
  });

  it("backfills missing observed-event embeddings", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-backfill-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const rationale = "Sol rejected a quarantined launch-pressure frame.";
    const embeddingClient = new MapEmbeddingClient(new Map([[rationale, [1, 0, 0, 0]]]));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const table = await store.openTable({
      name: "observed_events",
      schema: createObservedEventsTableSchema(4),
    });
    const sqliteOnlyRepository = new ObservedEventRepository({
      db,
      clock: new FixedClock(2_000),
    });
    const event = sqliteOnlyRepository.record(
      recordInput(createSessionId(), {
        interactionText: rationale,
        recurrenceKey: "session:speaker:backfill",
      }),
    );
    const vectorRepository = new ObservedEventRepository({
      db,
      table,
      embeddingClient,
      clock: new FixedClock(2_000),
    });

    const report = await vectorRepository.backfillMissingEmbeddings();
    const embeddedIds = await vectorRepository.getEmbeddedEventIds([event.id]);

    expect(report).toEqual({
      scanned: 1,
      embedded: 1,
      skipped: 0,
      failed: 0,
    });
    expect(embeddedIds).toEqual(new Set([event.id]));

    db.close();
  });

  it("appends the speaker-recent migration without changing earlier observed-event migrations", () => {
    expect(observedEventMigrations.map((migration) => [migration.id, migration.name])).toEqual([
      [1, "observed_events_baseline"],
      [2, "observed_events_fire_dedup_key"],
      [3, "observed_events_speaker_recent"],
      [4, "observed_events_global_relevance"],
    ]);

    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-migration-"));
    tempDirs.push(tempDir);
    const dbPath = join(tempDir, "observed-events.db");
    const initialDb = openDatabase(dbPath, {
      migrations: observedEventMigrations.slice(0, 2),
    });
    const sessionId = createSessionId();
    const speakerEntityId = createEntityId();

    try {
      const repository = new ObservedEventRepository({
        db: initialDb,
        clock: new FixedClock(2_000),
      });
      repository.record(
        recordInput(sessionId, {
          recurrenceKey: "session:speaker:migration",
          speakerEntityId,
        }),
      );
    } finally {
      initialDb.close();
    }

    const migratedDb = openDatabase(dbPath, {
      migrations: observedEventMigrations,
    });

    try {
      const indexes = migratedDb.prepare("PRAGMA index_list('observed_events')").all() as Record<
        string,
        unknown
      >[];

      expect(indexes.map((row) => row.name)).toContain("idx_observed_events_speaker_recent");
      expect(migratedDb.prepare("SELECT COUNT(*) AS count FROM observed_events").get()).toEqual({
        count: 1,
      });
    } finally {
      migratedDb.close();
    }
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

  it("loads legacy rows with non-enumerated dimension slugs", () => {
    const { db, repository } = openHarness("borg-observed-events-legacy-dimensions-");
    const created = repository.record(recordInput(createSessionId()));

    db.prepare("UPDATE observed_events SET stance = ?, taint = ?, belief_effect = ? WHERE id = ?")
      .run("legacy_frame", "legacy_taint", "legacy_effect", created.id);

    expect(repository.get(created.id)).toMatchObject({
      stance: "legacy_frame",
      taint: "legacy_taint",
      belief_effect: "legacy_effect",
    });

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

  it("still requires at least one source stream entry id", () => {
    const { db, repository } = openHarness("borg-observed-events-source-required-");

    expect(() =>
      repository.record(
        recordInput(createSessionId(), {
          sourceStreamEntryIds: [],
          fireDedupKey: "fire-empty-source",
        }),
      ),
    ).toThrow(/requires at least one source stream entry id/i);

    db.close();
  });
});
