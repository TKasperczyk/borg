import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

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
import { ObservedEventRepository, type ObservedEventRecordInput } from "./repository.js";

const NOW_MS = 1_000_000_000;
const tempDirs: string[] = [];

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
  it("recalls a present speaker's social observation across sessions", () => {
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

    const rows = selectObservedEventIntrospection({
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

  it("does not require the originating audience to be present to recall a speaker's rejection", () => {
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

    const rows = selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [speakerEntityId],
      nowMs: NOW_MS,
    });

    expect(currentDifferentAudienceEntityId).not.toBe(originAudienceEntityId);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.audienceEntityId).toBe(originAudienceEntityId);

    db.close();
  });

  it("recalls self_private and social_observed rows globally while preserving multilingual text", () => {
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

    const participantRows = selectObservedEventIntrospection({
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
    expect(participantRows[1]?.text).toBe(`Observed rejected_frame 2h ago: ${socialRationale}`);
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

  it("applies the 90-day recency window and caps newest-first globally across speakers", () => {
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

    const rows = selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [speakerA, speakerB],
      nowMs: NOW_MS,
      cap: 4,
    });

    expect(rows.map((row) => row.interactionText)).toEqual([
      "recent 0",
      "recurring rationale",
      "recent 1",
      "recent 2",
    ]);
    expect(rows.map((row) => row.lastSeenAt)).toEqual(
      [...rows.map((row) => row.lastSeenAt)].sort((a, b) => b - a),
    );
    expect(rows).toHaveLength(4);
    expect(rows.map((row) => row.interactionText)).not.toContain(
      "outside the social-memory window",
    );
    expect(rows.find((row) => row.interactionText === "recurring rationale")?.text).toContain(
      "Observed 2 times",
    );

    db.close();
  });

  it("returns nothing for an empty present-entity set without querying the repository", () => {
    const repository = {
      listRecentBySpeakers: vi.fn(),
    };

    const rows = selectObservedEventIntrospection({
      repository,
      speakerEntityIds: [],
      nowMs: NOW_MS,
    });

    expect(rows).toEqual([]);
    expect(repository.listRecentBySpeakers).not.toHaveBeenCalled();
  });
});
