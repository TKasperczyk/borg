import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { createMigrations } from "../../borg/storage-setup.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { createEntityId } from "../../util/ids.js";
import { OperatorAttentionRepository } from "./repository.js";
import { OPERATOR_ATTENTION_RECENT_LIMIT } from "./types.js";

describe("operator attention index", () => {
  it("migrates an existing bank, persists across reopen, and budgets without losing the total", () => {
    const dir = mkdtempSync(join(tmpdir(), "borg-attention-index-"));
    const path = join(dir, "borg.db");
    const migrations = createMigrations();
    let db = openDatabase(path, { migrations: migrations.slice(0, -1) });
    db.close();
    db = openDatabase(path, { migrations });
    const filer = createEntityId();
    const filing = {
      record_key: "cclink:first",
      filed_at: 1_000,
      filer_entity_id: filer,
      subject: "移行の確認",
    };
    try {
      const repository = new OperatorAttentionRepository({ db });
      expect(repository.snapshot()).toEqual({ total: 0, subjectless: 0, records: [] });
      expect(repository.record(filing)).toEqual({ inserted: true });
      expect(repository.record({ ...filing, subject: null })).toEqual({ inserted: false });
      const label = {
        disclosureClass: "operator_private",
        originAudienceEntityIds: [filer],
        privateToEntityIds: [],
        publicToEntityIds: [],
      };
      expect(repository.snapshot().records).toEqual([{ ...filing, disclosure_label: label }]);
      expect(repository.snapshot().subjectless).toBe(0);
      for (let i = 0; i < 124; i += 1) {
        repository.record({
          ...filing,
          record_key: `legacy:${i}`,
          filed_at: 2_000 + i,
          subject: null,
        });
      }
      db.close();
      db = openDatabase(path, { migrations });
      const snapshot = new OperatorAttentionRepository({ db }).snapshot();
      expect(snapshot.total).toBe(125);
      expect(snapshot.subjectless).toBe(124);
      expect(snapshot.records).toHaveLength(OPERATOR_ATTENTION_RECENT_LIMIT);
      for (const row of snapshot.records) expect(row.disclosure_label).toEqual(label);
      expect(snapshot.records[0]).toMatchObject({
        record_key: "legacy:123",
        filed_at: 2_123,
        subject: null,
      });
      expect(snapshot.records.at(-1)?.filed_at).toBe(2_104);
      expect(
        db
          .prepare("PRAGMA table_info(operator_attention_records)")
          .all()
          .map((row) => row.name),
      ).toEqual(["record_key", "filed_at", "filer_entity_id", "subject"]);
      expect(() =>
        new OperatorAttentionRepository({ db }).record({
          ...filing,
          body: "must stay outside Borg",
        } as typeof filing),
      ).toThrow();
    } finally {
      db.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("counts null subjects outside the newest window independently of record-key prefixes", () => {
    const db = openDatabase(":memory:", { migrations: createMigrations() });
    const repository = new OperatorAttentionRepository({ db });
    const filer = createEntityId();
    try {
      repository.record({
        record_key: "another-connector:older",
        filed_at: 1,
        filer_entity_id: filer,
        subject: null,
      });
      for (let index = 0; index <= OPERATOR_ATTENTION_RECENT_LIMIT; index += 1) {
        repository.record({
          record_key: `cclink:legacy:${index}`,
          filed_at: index + 2,
          filer_entity_id: filer,
          subject: "記録された件名",
        });
      }
      const snapshot = repository.snapshot();
      expect(snapshot.total).toBe(OPERATOR_ATTENTION_RECENT_LIMIT + 2);
      expect(snapshot.subjectless).toBe(1);
      expect(snapshot.records).toHaveLength(OPERATOR_ATTENTION_RECENT_LIMIT);
      expect(snapshot.records.every((record) => record.subject !== null)).toBe(true);
    } finally {
      db.close();
    }
  });
});
