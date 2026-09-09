import { appendFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { openCaptureSnapshot } from "./capture-snapshot.js";

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

it.each(["", '{"capture":1}\n'])(
  "pins a read-only snapshot of %j, including initially empty files",
  async (initial) => {
    const dir = mkdtempSync(join(tmpdir(), "borg-capture-snapshot-"));
    dirs.push(dir);
    const path = join(dir, "captures.jsonl");
    writeFileSync(path, initial);
    const snapshot = openCaptureSnapshot(path);
    appendFileSync(path, '{"capture":2}\n');
    const lines: string[] = [];
    for await (const line of snapshot.lines) lines.push(line);
    expect(snapshot.snapshotBytes).toBe(Buffer.byteLength(initial));
    expect(lines).toEqual(initial === "" ? [] : ['{"capture":1}']);
    expect(readFileSync(path, "utf8")).toBe(initial + '{"capture":2}\n');
  },
);
