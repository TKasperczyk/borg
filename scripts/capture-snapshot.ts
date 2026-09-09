import { createReadStream, statSync } from "node:fs";
import { createInterface } from "node:readline";
import { Readable } from "node:stream";

/** Read-only JSONL cohort: later appends belong to the next snapshot. */
export function openCaptureSnapshot(path: string): {
  snapshotBytes: number;
  lines: ReturnType<typeof createInterface>;
} {
  const snapshotBytes = statSync(path).size;
  const input =
    snapshotBytes === 0
      ? Readable.from([])
      : createReadStream(path, { encoding: "utf8", start: 0, end: snapshotBytes - 1 });
  return { snapshotBytes, lines: createInterface({ input, crlfDelay: Infinity }) };
}
