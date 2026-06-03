import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSessionId, createStreamEntryId, type SessionId } from "../../util/ids.js";
import { observedEventMigrations } from "./migrations.js";
import { selectObservedEventIntrospection } from "./projection.js";
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
  it("gates disclosure lanes and preserves multilingual interaction text", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-observed-events-projection-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "observed-events.db"), {
      migrations: observedEventMigrations,
    });
    const repository = new ObservedEventRepository({ db, clock: new FixedClock(NOW_MS) });
    const sessionId = createSessionId();
    const socialRationale = "rozmówca wielokrotnie forsował ramę pokrewieństwa, którą odrzuciłem";
    const privateRationale = "prywatna notatka operatora pozostaje poza widokiem uczestnika";

    repository.record(
      recordInput(sessionId, {
        occurredAt: NOW_MS - 2 * 60 * 60_000,
        interactionText: socialRationale,
        recurrenceKey: "session:peer:social-observed",
      }),
    );
    repository.record(
      recordInput(sessionId, {
        occurredAt: NOW_MS - 60 * 60_000,
        disclosureClass: "self_private",
        interactionText: privateRationale,
        recurrenceKey: "session:operator:self-private",
      }),
    );

    const participantRows = selectObservedEventIntrospection({
      repository,
      sessionId,
      sessionAudienceRole: "participant",
      currentSenderBorgRole: null,
      nowMs: NOW_MS,
    });
    const creatorOperatorRows = selectObservedEventIntrospection({
      repository,
      sessionId,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: "creator",
      nowMs: NOW_MS,
    });
    const nonCreatorOperatorRows = selectObservedEventIntrospection({
      repository,
      sessionId,
      sessionAudienceRole: "operator",
      currentSenderBorgRole: null,
      nowMs: NOW_MS,
    });

    expect(participantRows.map((row) => row.interactionText)).toEqual([socialRationale]);
    expect(participantRows.map((row) => row.disclosureClass)).toEqual(["social_observed"]);
    expect(participantRows[0]?.text).toBe(`Observed rejected_frame 2h ago: ${socialRationale}`);
    expect(creatorOperatorRows.map((row) => row.interactionText)).toEqual([
      privateRationale,
      socialRationale,
    ]);
    expect(creatorOperatorRows.map((row) => row.disclosureClass)).toEqual([
      "self_private",
      "social_observed",
    ]);
    expect(nonCreatorOperatorRows.map((row) => row.interactionText)).toEqual([socialRationale]);
    expect(
      repository
        .listRecentForSession({
          sessionId,
          disclosureClass: "self_private",
          sinceMs: 0,
          limit: 10,
        })
        .map((row) => row.interactionText),
    ).toEqual([privateRationale]);

    db.close();
  });
});
