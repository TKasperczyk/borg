import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

const fsFailure = vi.hoisted(() => ({ enabled: false }));

vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return {
    ...actual,
    writeFileSync: ((file, data, options) => {
      if (fsFailure.enabled && String(data).includes('"force_partial_failure":true')) {
        fsFailure.enabled = false;
        actual.writeSync(file as number, Buffer.from('{"partial":'));
        throw new Error("injected append failure");
      }
      return actual.writeFileSync(file, data, options as never);
    }) as typeof actual.writeFileSync,
  };
});

import { appendDurableJsonl } from "./durable-jsonl.js";

describe("appendDurableJsonl", () => {
  const tempDirectories: string[] = [];

  afterEach(() => {
    fsFailure.enabled = false;
    for (const directory of tempDirectories.splice(0)) {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("creates capture directories as 0700 and result files as 0600 under umask 0022", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-new-private-jsonl-"));
    tempDirectories.push(root);
    const captures = join(root, "captures");
    const path = join(captures, "planner-ab-results.jsonl");
    const previousUmask = process.umask(0o022);

    try {
      await appendDurableJsonl(path, { result: true }, { privateDirectory: captures });
    } finally {
      process.umask(previousUmask);
    }

    expect(statSync(captures).mode & 0o777).toBe(0o700);
    expect(statSync(path).mode & 0o777).toBe(0o600);
  });

  it("creates and repairs private capture permissions under a 0022 umask", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-private-jsonl-"));
    tempDirectories.push(root);
    const captures = join(root, "captures");
    const path = join(captures, "records.jsonl");
    mkdirSync(captures, { mode: 0o755 });
    writeFileSync(path, `${JSON.stringify({ existing: true })}\n`, { mode: 0o644 });
    chmodSync(captures, 0o755);
    chmodSync(path, 0o644);
    const previousUmask = process.umask(0o022);

    try {
      await appendDurableJsonl(path, { appended: true }, { privateDirectory: captures });
    } finally {
      process.umask(previousUmask);
    }

    expect(statSync(captures).mode & 0o777).toBe(0o700);
    expect(statSync(path).mode & 0o777).toBe(0o600);
    expect(readFileSync(path, "utf8").trim().split("\n")).toHaveLength(2);
  });

  it("repairs an unterminated tail before the next locked append", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-repair-jsonl-"));
    tempDirectories.push(root);
    const captures = join(root, "captures");
    const path = join(captures, "records.jsonl");
    mkdirSync(captures, { mode: 0o700 });
    writeFileSync(path, '{"kept":1}\n{"partial":', { mode: 0o600 });

    const result = await appendDurableJsonl(path, { next: 2 }, { privateDirectory: captures });

    expect(result).toMatchObject({ status: "appended", repairedTailBytes: 11 });
    expect(readFileSync(path, "utf8")).toBe('{"kept":1}\n{"next":2}\n');
  });

  it("applies the file cap after discarding an incomplete tail", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-cap-jsonl-"));
    tempDirectories.push(root);
    const captures = join(root, "captures");
    const path = join(captures, "records.jsonl");
    mkdirSync(captures, { mode: 0o700 });
    writeFileSync(path, '{"kept":1}\npartial', { mode: 0o600 });
    const next = `${JSON.stringify({ next: 2 })}\n`;
    const cap = Buffer.byteLength('{"kept":1}\n') + Buffer.byteLength(next);

    await expect(
      appendDurableJsonl(path, { next: 2 }, { maxFileBytes: cap, privateDirectory: captures }),
    ).resolves.toMatchObject({ status: "appended", repairedTailBytes: 7 });
  });

  it("truncates back to the record start when an append fails after a partial write", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-rollback-jsonl-"));
    tempDirectories.push(root);
    const captures = join(root, "captures");
    const path = join(captures, "records.jsonl");
    await appendDurableJsonl(path, { kept: 1 }, { privateDirectory: captures });
    fsFailure.enabled = true;

    await expect(
      appendDurableJsonl(path, { force_partial_failure: true }, { privateDirectory: captures }),
    ).rejects.toThrow("injected append failure");

    expect(readFileSync(path, "utf8")).toBe('{"kept":1}\n');
    await appendDurableJsonl(path, { next: 2 }, { privateDirectory: captures });
    expect(readFileSync(path, "utf8")).toBe('{"kept":1}\n{"next":2}\n');
  });

  it("serializes concurrent result appends through the shared file lock", async () => {
    const root = mkdtempSync(join(tmpdir(), "borg-concurrent-jsonl-"));
    tempDirectories.push(root);
    const captures = join(root, "captures");
    const path = join(captures, "results.jsonl");

    await Promise.all(
      Array.from({ length: 8 }, (_, index) =>
        appendDurableJsonl(path, { index }, { privateDirectory: captures }),
      ),
    );

    const indexes = readFileSync(path, "utf8")
      .trim()
      .split("\n")
      .map((line) => (JSON.parse(line) as { index: number }).index)
      .sort((left, right) => left - right);
    expect(indexes).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });
});
