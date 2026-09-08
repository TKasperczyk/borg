import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { createEntityId } from "../src/util/ids.js";
import {
  backfillOperatorAttention,
  readOperatorAttentionBackfill,
} from "./backfill-operator-attention.js";

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});
function fixture(lines: unknown[]) {
  const dir = mkdtempSync(join(tmpdir(), "attention-backfill-"));
  dirs.push(dir);
  const path = join(dir, "operator-attention.jsonl");
  writeFileSync(path, `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`);
  return path;
}

describe("attention backfill", () => {
  it("defaults to read-only dry-run, preserves stored subjects, and keys equal timestamps distinctly", async () => {
    const filer = createEntityId();
    const actualFiler = createEntityId();
    const path = fixture([
      { ts: 1_000, reason: "private historical body" },
      { ts: 1_000, reason: "another private body" },
      {
        ts: 2_000,
        record_key: "cclink:stored-key",
        filer_entity_id: actualFiler,
        subject: "Stored subject",
        reason: "body",
      },
    ]);
    const before = readFileSync(path, "utf8");
    const fetchImpl = vi.fn<typeof fetch>();
    const result = await backfillOperatorAttention({ path, filerEntityId: filer, fetchImpl });
    expect(fetchImpl).not.toHaveBeenCalled();
    expect(readFileSync(path, "utf8")).toBe(before);
    expect(result.records).toEqual(readOperatorAttentionBackfill(path, filer));
    expect(result.records.map((record) => record.subject)).toEqual([null, null, "Stored subject"]);
    expect(new Set(result.records.map((record) => record.record_key)).size).toBe(3);
    expect(result.records[2]).toMatchObject({
      record_key: "cclink:stored-key",
      filer_entity_id: actualFiler,
    });
    expect(JSON.stringify(result)).not.toContain("private");
    expect(result.records[2]).toMatchObject({ filed_at: 2_000, subject: "Stored subject" });
    // Keys are based on filing metadata, never on body contents.
    writeFileSync(path, before.replace("private historical body", "changed local body"));
    expect(readOperatorAttentionBackfill(path, filer)).toEqual(result.records);
  });

  it("applies only envelopes and can be resent without duplicate records", async () => {
    const filer = createEntityId();
    const stored = {
      ts: 1_000,
      record_key: "cclink:failed-report",
      filer_entity_id: filer,
      subject: "Stored subject",
      reason: "SECRET",
    };
    const path = fixture([stored]);
    const seen = new Set<string>();
    const fetchImpl = vi.fn<typeof fetch>(async (_url, init) => {
      const record = JSON.parse(String(init?.body));
      expect(Object.keys(record).sort()).toEqual([
        "filed_at",
        "filer_entity_id",
        "record_key",
        "subject",
      ]);
      expect(init?.body).not.toContain("SECRET");
      expect(record).toEqual({
        record_key: stored.record_key,
        filed_at: stored.ts,
        filer_entity_id: filer,
        subject: stored.subject,
      });
      const inserted = !seen.has(record.record_key);
      seen.add(record.record_key);
      return Response.json({ inserted });
    });
    const input = {
      path,
      filerEntityId: createEntityId(),
      apply: true,
      borgUrl: "http://borg.test:7740",
      fetchImpl,
    };
    expect(await backfillOperatorAttention(input)).toMatchObject({ inserted: 1, duplicates: 0 });
    expect(await backfillOperatorAttention(input)).toMatchObject({ inserted: 0, duplicates: 1 });
    expect(String(fetchImpl.mock.calls[0]?.[0])).toBe(
      "http://borg.test:7740/api/operator-attention",
    );
  });

  it("validates all rows before writing and does not echo malformed bodies", async () => {
    const path = fixture([{ ts: 1_000, reason: "valid" }]);
    writeFileSync(path, `${readFileSync(path, "utf8")}malformed SECRET BODY\n`);
    const fetchImpl = vi.fn<typeof fetch>();
    await expect(
      backfillOperatorAttention({
        path,
        filerEntityId: createEntityId(),
        apply: true,
        borgUrl: "http://borg.test",
        fetchImpl,
      }),
    ).rejects.toThrow("Invalid attention envelope on line 2");
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
