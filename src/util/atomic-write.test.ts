import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { readJsonFile, writeFileAtomic, writeJsonFileAtomic } from "./atomic-write.js";

describe("atomic-write", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("writes bytes atomically and overwrites existing content", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const filePath = join(tempDir, "nested", "file.txt");
    writeFileAtomic(filePath, "first");
    writeFileAtomic(filePath, Buffer.from("second"));

    expect(readFileSync(filePath, "utf8")).toBe("second");
  });

  it("reads and writes JSON files", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const filePath = join(tempDir, "config.json");
    writeJsonFileAtomic(filePath, {
      answer: 42,
      nested: { ok: true },
    });

    expect(readJsonFile<{ answer: number; nested: { ok: boolean } }>(filePath)).toEqual({
      answer: 42,
      nested: { ok: true },
    });
    expect(readJsonFile(join(tempDir, "missing.json"))).toBeUndefined();
  });
});
