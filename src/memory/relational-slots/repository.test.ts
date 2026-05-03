import { describe, expect, it, vi } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { createEntityId, createStreamEntryId } from "../../util/ids.js";
import { relationalSlotMigrations } from "./migrations.js";
import { RelationalSlotRepository } from "./repository.js";

describe("relational slot repository", () => {
  it("establishes slots and quarantines on the third distinct value", () => {
    const db = openDatabase(":memory:", {
      migrations: relationalSlotMigrations,
    });
    const clock = new ManualClock(1_000);
    const repo = new RelationalSlotRepository({ db, clock });
    const subject = createEntityId();
    const first = createStreamEntryId();
    const second = createStreamEntryId();
    const third = createStreamEntryId();

    try {
      const established = repo.applyAssertion({
        subject_entity_id: subject,
        slot_key: "partner.name",
        asserted_value: "Sarah",
        source_stream_entry_ids: [first],
      });
      clock.advance(10);
      const contested = repo.applyAssertion({
        subject_entity_id: subject,
        slot_key: "partner.name",
        asserted_value: "Maya",
        source_stream_entry_ids: [second],
      });
      clock.advance(10);
      const quarantined = repo.applyAssertion({
        subject_entity_id: subject,
        slot_key: "partner.name",
        asserted_value: "Clara",
        source_stream_entry_ids: [third],
      });

      expect(established.slot).toMatchObject({
        value: "Sarah",
        state: "established",
        evidence_stream_entry_ids: [first],
      });
      expect(contested.slot).toMatchObject({
        value: "Sarah",
        state: "contested",
        contradicted_by_stream_entry_ids: [second],
      });
      expect(contested.slot.alternate_values).toEqual([
        {
          value: "Maya",
          evidence_stream_entry_ids: [second],
        },
      ]);
      expect(quarantined.slot.state).toBe("quarantined");
      expect(quarantined.values_to_neutralize).toEqual(["Sarah", "Maya", "Clara"]);
      expect(repo.listConstrained()).toEqual([quarantined.slot]);
    } finally {
      db.close();
    }
  });

  it("quarantines a matching slot negation", () => {
    const db = openDatabase(":memory:", {
      migrations: relationalSlotMigrations,
    });
    const repo = new RelationalSlotRepository({ db, clock: new ManualClock(1_000) });
    const subject = createEntityId();
    const assertionEntry = createStreamEntryId();
    const negationEntry = createStreamEntryId();

    try {
      repo.applyAssertion({
        subject_entity_id: subject,
        slot_key: "partner.name",
        asserted_value: "Sarah",
        source_stream_entry_ids: [assertionEntry],
      });
      const result = repo.applyNegation({
        subject_entity_id: subject,
        slot_key: "partner.name",
        rejected_value: "Sarah",
        source_stream_entry_ids: [negationEntry],
      });

      expect(result?.slot.state).toBe("quarantined");
      expect(result?.slot.contradicted_by_stream_entry_ids).toEqual([negationEntry]);
      expect(result?.neutral_phrase).toBe("your partner");
    } finally {
      db.close();
    }
  });

  it("rereads slots inside the write transaction before conflicting assertions", () => {
    const db = openDatabase(":memory:", {
      migrations: relationalSlotMigrations,
    });
    const clock = new ManualClock(1_000);
    const repo = new RelationalSlotRepository({ db, clock });
    const subject = createEntityId();
    const first = createStreamEntryId();
    const second = createStreamEntryId();
    const third = createStreamEntryId();

    try {
      repo.applyAssertion({
        subject_entity_id: subject,
        slot_key: "partner.name",
        asserted_value: "Sarah",
        source_stream_entry_ids: [first],
      });
      const staleEstablished = repo.findBySubjectAndKey(subject, "partner.name");

      expect(staleEstablished).not.toBeNull();

      repo.applyAssertion({
        subject_entity_id: subject,
        slot_key: "partner.name",
        asserted_value: "Maya",
        source_stream_entry_ids: [second],
      });

      const originalFindBySubjectAndKey = repo.findBySubjectAndKey.bind(repo);
      const findSpy = vi
        .spyOn(repo, "findBySubjectAndKey")
        .mockImplementation((subjectEntityId, slotKey) =>
          db.raw.inTransaction
            ? originalFindBySubjectAndKey(subjectEntityId, slotKey)
            : staleEstablished,
        );
      clock.advance(10);
      const quarantined = repo.applyAssertion({
        subject_entity_id: subject,
        slot_key: "partner.name",
        asserted_value: "Clara",
        source_stream_entry_ids: [third],
      });

      expect(findSpy).toHaveBeenCalled();
      expect(quarantined.slot.state).toBe("quarantined");
      expect(quarantined.values_to_neutralize).toEqual(["Sarah", "Maya", "Clara"]);
    } finally {
      db.close();
    }
  });
});
