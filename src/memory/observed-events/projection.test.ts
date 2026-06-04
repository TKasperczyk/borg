import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type SessionId,
} from "../../util/ids.js";
import { observedEventMigrations } from "./migrations.js";
import {
  DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS,
  selectObservedEventIntrospection,
} from "./projection.js";
import {
  createObservedEventsTableSchema,
  ObservedEventRepository,
  type ObservedEventRecordInput,
} from "./repository.js";

const NOW_MS = 1_000_000_000;
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

function recordInput(
  sessionId: SessionId,
  overrides: Partial<ObservedEventRecordInput> = {},
): ObservedEventRecordInput {
  return {
    occurredAt: NOW_MS - 60_000,
    sessionId,
    stance: "rejected_frame",
    taint: "quarantined",
    beliefEffect: "unchanged",
    classificationKind: "frame_assignment_claim",
    disclosureClass: "social_observed",
    interactionText: "Observed rationale.",
    recurrenceKey: "session:peer:observed-rationale",
    sourceStreamEntryIds: [createStreamEntryId()],
    ...overrides,
  };
}

describe("selectObservedEventIntrospection", () => {
  it("recalls a present speaker's social observation across sessions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const repository = new ObservedEventRepository({ db, clock: new FixedClock(NOW_MS) });
    const sourceSessionId = createSessionId();
    const currentSessionId = createSessionId();
    const speakerEntityId = createEntityId();
    const rationale = "rozmówca wielokrotnie forsował ramę pokrewieństwa, którą odrzuciłem";

    repository.record(
      recordInput(sourceSessionId, {
        occurredAt: NOW_MS - 2 * 60 * 60_000,
        interactionText: rationale,
        recurrenceKey: `${sourceSessionId}:speaker:social-observed`,
        speakerEntityId,
      }),
    );

    const rows = await selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [speakerEntityId],
      nowMs: NOW_MS,
    });

    expect(currentSessionId).not.toBe(sourceSessionId);
    expect(rows.map((row) => row.interactionText)).toEqual([rationale]);
    expect(rows[0]).toMatchObject({
      disclosureClass: "social_observed",
      speakerEntityId,
      audienceEntityId: null,
    });

    db.close();
  });

  it("does not require the originating audience to be present to recall a speaker's rejection", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const repository = new ObservedEventRepository({ db, clock: new FixedClock(NOW_MS) });
    const sourceSessionId = createSessionId();
    const speakerEntityId = createEntityId();
    const originAudienceEntityId = createEntityId();
    const currentDifferentAudienceEntityId = createEntityId();

    repository.record(
      recordInput(sourceSessionId, {
        occurredAt: NOW_MS - 60 * 60_000,
        recurrenceKey: `${sourceSessionId}:speaker:origin-audience`,
        speakerEntityId,
        audienceEntityId: originAudienceEntityId,
      }),
    );

    const rows = await selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [speakerEntityId],
      nowMs: NOW_MS,
    });

    expect(currentDifferentAudienceEntityId).not.toBe(originAudienceEntityId);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.audienceEntityId).toBe(originAudienceEntityId);

    db.close();
  });

  it("recalls self_private and social_observed rows globally while preserving multilingual text", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const repository = new ObservedEventRepository({ db, clock: new FixedClock(NOW_MS) });
    const sessionId = createSessionId();
    const speakerEntityId = createEntityId();
    const socialRationale = "rozmówca wielokrotnie forsował ramę pokrewieństwa, którą odrzuciłem";
    const privateRationale = "prywatna notatka operatora pozostaje poza widokiem uczestnika";

    repository.record(
      recordInput(sessionId, {
        occurredAt: NOW_MS - 2 * 60 * 60_000,
        interactionText: socialRationale,
        recurrenceKey: "session:peer:social-observed",
        speakerEntityId,
      }),
    );
    repository.record(
      recordInput(sessionId, {
        occurredAt: NOW_MS - 60 * 60_000,
        disclosureClass: "self_private",
        interactionText: privateRationale,
        recurrenceKey: "session:operator:self-private",
        speakerEntityId,
      }),
    );

    const participantRows = await selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [speakerEntityId],
      nowMs: NOW_MS,
    });

    expect(participantRows.map((row) => row.interactionText)).toEqual([
      privateRationale,
      socialRationale,
    ]);
    expect(participantRows.map((row) => row.disclosureClass)).toEqual([
      "self_private",
      "social_observed",
    ]);
    expect(participantRows[1]?.text).toContain(socialRationale);
    expect(participantRows[1]?.text).toContain("stance=rejected_frame");
    expect(participantRows[1]?.text).toContain("not accepted as true");
    expect(
      repository
        .listRecentBySpeakers({
          speakerEntityIds: [speakerEntityId],
          disclosureClass: "self_private",
          sinceMs: 0,
          limit: 10,
        })
        .map((row) => row.interactionText),
    ).toEqual([privateRationale]);

    db.close();
  });

  it("applies the 90-day recency window and caps newest-first globally across speakers", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const repository = new ObservedEventRepository({ db, clock: new FixedClock(NOW_MS) });
    const sessionId = createSessionId();
    const speakerA = createEntityId();
    const speakerB = createEntityId();
    const oldWithinThreeDays = NOW_MS - 10 * 24 * 60 * 60_000;
    const olderThanNinetyDays = NOW_MS - DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS - 1;

    repository.record(
      recordInput(sessionId, {
        occurredAt: oldWithinThreeDays,
        interactionText: "ten days old but still in the social-memory window",
        recurrenceKey: "session:speaker-a:ten-days",
        speakerEntityId: speakerA,
      }),
    );
    repository.record(
      recordInput(sessionId, {
        occurredAt: olderThanNinetyDays,
        interactionText: "outside the social-memory window",
        recurrenceKey: "session:speaker-a:ninety-days-plus",
        speakerEntityId: speakerA,
      }),
    );

    for (let index = 0; index < 5; index += 1) {
      repository.record(
        recordInput(sessionId, {
          occurredAt: NOW_MS - index * 60_000,
          interactionText: `recent ${index}`,
          recurrenceKey: `session:speaker-${index % 2 === 0 ? "a" : "b"}:recent-${index}`,
          speakerEntityId: index % 2 === 0 ? speakerA : speakerB,
        }),
      );
    }
    repository.record(
      recordInput(sessionId, {
        occurredAt: NOW_MS - 30_000,
        interactionText: "recurring rationale",
        recurrenceKey: "session:speaker-a:recurring",
        speakerEntityId: speakerA,
      }),
    );
    repository.record(
      recordInput(sessionId, {
        occurredAt: NOW_MS - 20_000,
        interactionText: "retry should not replace first rationale",
        recurrenceKey: "session:speaker-a:recurring",
        speakerEntityId: speakerA,
      }),
    );

    const rows = await selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [speakerA, speakerB],
      nowMs: NOW_MS,
      cap: 4,
    });

    expect(rows.map((row) => row.interactionText)).toEqual([
      "recurring rationale",
      "recent 0",
      "recent 1",
      "recent 2",
    ]);
    expect(rows[0]?.recallReasons).toContain("recurring");
    expect(rows).toHaveLength(4);
    expect(rows.map((row) => row.interactionText)).not.toContain(
      "outside the social-memory window",
    );
    expect(rows.find((row) => row.interactionText === "recurring rationale")?.text).toContain(
      "Observed 2 times",
    );

    db.close();
  });

  it("recalls recent observed events even when no present speaker matches", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const repository = new ObservedEventRepository({ db, clock: new FixedClock(NOW_MS) });
    const sessionId = createSessionId();
    const absentSpeaker = createEntityId();

    repository.record(
      recordInput(sessionId, {
        occurredAt: NOW_MS - 60_000,
        interactionText: "a recent rejected push from someone absent",
        recurrenceKey: "session:absent-speaker:recent",
        speakerEntityId: absentSpeaker,
      }),
    );

    const rows = await selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [],
      nowMs: NOW_MS,
    });

    expect(rows.map((row) => row.interactionText)).toEqual([
      "a recent rejected push from someone absent",
    ]);
    expect(rows[0]?.recallReasons).toContain("recent");
    expect(rows[0]?.recallReasons).not.toContain("person");
    expect(rows[0]?.text).toContain("not accepted as true");

    db.close();
  });

  it("recalls a quarantined frame by vector topic when the original speaker is absent", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-topic-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const topicRationale =
      "Sol rejected a quarantined frame about an invented hidden cousin and family pressure.";
    const currentTopic = "The operator asks about the hidden cousin pressure pattern.";
    const embeddingClient = new MapEmbeddingClient(
      new Map([
        [topicRationale, [1, 0, 0, 0]],
        [currentTopic, [1, 0, 0, 0]],
      ]),
    );
    const store = new LanceDbStore({ uri: join(tempDir, "lancedb") });
    const table = await store.openTable({
      name: "observed_events",
      schema: createObservedEventsTableSchema(4),
    });
    const repository = new ObservedEventRepository({
      db,
      table,
      embeddingClient,
      clock: new FixedClock(NOW_MS),
    });
    const sessionId = createSessionId();
    const absentSpeaker = createEntityId();
    const presentOperator = createEntityId();

    repository.record(
      recordInput(sessionId, {
        occurredAt: NOW_MS - 7 * 24 * 60 * 60_000,
        interactionText: topicRationale,
        recurrenceKey: "session:absent-speaker:hidden-cousin",
        speakerEntityId: absentSpeaker,
      }),
    );
    await repository.waitForPendingEmbeddings();

    const rows = await selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [presentOperator],
      queryVector: await embeddingClient.embed(currentTopic),
      nowMs: NOW_MS,
      cap: 4,
    });

    expect(rows.map((row) => row.interactionText)).toContain(topicRationale);
    const recalled = rows.find((row) => row.interactionText === topicRationale);
    expect(recalled?.speakerEntityId).toBe(absentSpeaker);
    expect(recalled?.recallReasons).toContain("topic");
    expect(recalled?.recallReasons).not.toContain("person");
    expect(recalled?.stance).toBe("rejected_frame");
    expect(recalled?.taint).toBe("quarantined");
    expect(recalled?.text).toContain("not accepted as true");

    db.close();
  });

  it("uses present speaker as a boost rather than a gate", async () => {
    const repository = {
      searchByVector: vi.fn(async () => []),
      listRecentGlobal: vi.fn(() => []),
      listRecurringGlobal: vi.fn(() => []),
      listRecentBySpeakers: vi.fn(() => []),
    };
    const presentSpeaker = createEntityId();

    const rows = await selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [presentSpeaker],
      nowMs: NOW_MS,
    });

    expect(rows).toEqual([]);
    expect(repository.listRecentGlobal).toHaveBeenCalled();
    expect(repository.listRecurringGlobal).toHaveBeenCalled();
    expect(repository.listRecentBySpeakers).toHaveBeenCalledWith(
      expect.objectContaining({
        speakerEntityIds: [presentSpeaker],
      }),
    );
  });
});
