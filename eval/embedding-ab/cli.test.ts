import { mkdirSync, mkdtempSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { assertScratchOutsideBank, parseCliArgs } from "./cli.js";

describe("embedding A/B batch budgets", () => {
  const args = ["--data-dir", "copied-bank", "--models", "embedding-model", "--out", "scratch"];

  it("defaults the byte budget and accepts an override beside the count cap", () => {
    expect(parseCliArgs(args)).toMatchObject({ batchSize: 8, batchBytes: 48_000 });
    expect(parseCliArgs([...args, "--batch-size", "32", "--batch-bytes", "24000"])).toMatchObject({
      batchSize: 32,
      batchBytes: 24_000,
    });
  });

  it.each(["0", "-1", "1.5", "NaN", "Infinity", "9007199254740992"])(
    "rejects the invalid byte budget %s",
    (value) => {
      expect(() => parseCliArgs([...args, "--batch-bytes", value])).toThrow(/--batch-bytes/);
    },
  );
});

describe("embedding A/B scratch-directory guard", () => {
  let root: string;
  let bank: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "borg-embedding-ab-path-"));
    bank = join(root, "bank");
    mkdirSync(bank);
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it("rejects the exact bank directory", () => {
    expect(() => assertScratchOutsideBank(bank, bank)).toThrow(/outside --data-dir/);
  });

  it("rejects an ordinary child of the bank", () => {
    expect(() => assertScratchOutsideBank(bank, join(bank, "scratch"))).toThrow(
      /outside --data-dir/,
    );
  });

  it("rejects an in-bank child whose name begins with two dots", () => {
    expect(() => assertScratchOutsideBank(bank, join(bank, "..scratch"))).toThrow(
      /outside --data-dir/,
    );
  });

  it("accepts a sibling of the bank", () => {
    expect(() => assertScratchOutsideBank(bank, join(root, "scratch"))).not.toThrow();
  });

  it("rejects a child reached through a symlink to the bank", () => {
    const bankAlias = join(root, "bank-alias");
    symlinkSync(bank, bankAlias, process.platform === "win32" ? "junction" : "dir");

    expect(() => assertScratchOutsideBank(bank, join(bankAlias, "scratch"))).toThrow(
      /outside --data-dir/,
    );
  });
});
