import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";

import { afterEach, describe, expect, it, vi } from "vitest";

const pipelineStall = vi.hoisted(() => ({
  gate: null as Promise<void> | null,
  release: null as (() => void) | null,
  stalledCalls: 0,
}));

vi.mock("node:stream/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:stream/promises")>();
  return {
    ...actual,
    pipeline: (async (...streams: unknown[]) => {
      const gate = pipelineStall.gate;
      if (gate !== null) {
        pipelineStall.stalledCalls += 1;
        await gate;
      }
      return Reflect.apply(actual.pipeline, undefined, streams);
    }) as typeof actual.pipeline,
  };
});

import {
  appendBoundedContextCapture,
  commitStagedContentAddressedCaptureSidecars,
  createContentAddressedCaptureSidecar,
  discardStagedContentAddressedCaptureSidecars,
  readContentAddressedCaptureSidecar,
  stageContentAddressedCaptureSidecars,
  waitForContextCaptureMaintenance,
} from "./context-capture-storage.js";

const tempDirs: string[] = [];

afterEach(async () => {
  const release = pipelineStall.release;
  pipelineStall.gate = null;
  pipelineStall.release = null;
  release?.();
  await waitForContextCaptureMaintenance();
  pipelineStall.stalledCalls = 0;
  for (const directory of tempDirs.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

function temporaryDirectory(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(directory);
  return directory;
}

function readJsonlRecords(path: string): Array<{ sequence: number }> {
  const bytes = readFileSync(path);
  const text = path.endsWith(".gz") ? gunzipSync(bytes).toString("utf8") : bytes.toString("utf8");
  return text
    .trim()
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as { sequence: number });
}

describe("bounded context capture rotation", () => {
  it("keeps the triggering record, names and compresses the rotation, and prunes raw and gzip archives together", async () => {
    const dataDir = temporaryDirectory("borg-capture-rotation-");
    const captureDirectory = join(dataDir, "captures");
    const fileName = "test-contexts.jsonl";
    const path = join(captureDirectory, fileName);
    mkdirSync(captureDirectory, { recursive: true });
    writeFileSync(`${path}.rotated-20260101T000000.000Z`, '{"sequence":0}\n');
    writeFileSync(`${path}.rotated-20260201T000000.000Z.gz`, gzipSync('{"sequence":1}\n'));
    const existing = { sequence: 2 };
    const triggering = { sequence: 3 };
    writeFileSync(path, `${JSON.stringify(existing)}\n`);
    const logger = { info: vi.fn(), error: vi.fn() };

    const result = await appendBoundedContextCapture({
      dataDir,
      fileName,
      record: triggering,
      maxFileBytes: Buffer.byteLength(`${JSON.stringify(existing)}\n`),
      rotationKeep: 2,
      rotationTimestampMs: Date.UTC(2026, 2, 1),
      logger,
    });
    await waitForContextCaptureMaintenance(path);

    expect(result).toMatchObject({
      status: "rotated",
      path,
      rotatedPath: `${path}.rotated-20260301T000000.000Z`,
    });
    expect(readJsonlRecords(path)).toEqual([triggering]);
    const archives = readdirSync(captureDirectory)
      .filter((name) => name.startsWith(`${fileName}.rotated-`))
      .sort();
    expect(archives).toEqual([
      `${fileName}.rotated-20260201T000000.000Z.gz`,
      `${fileName}.rotated-20260301T000000.000Z.gz`,
    ]);
    expect(readJsonlRecords(join(captureDirectory, archives[1]!))).toEqual([existing]);
    expect(logger.info).toHaveBeenCalledWith("Rotated deliberation context capture", {
      capturePath: path,
      rotatedPath: `${path}.rotated-20260301T000000.000Z`,
    });
    expect(logger.info).toHaveBeenCalledWith(
      "Pruned rotated deliberation context capture",
      expect.objectContaining({
        capturePath: path,
        prunedPath: `${path}.rotated-20260101T000000.000Z.gz`,
        rotationKeep: 2,
      }),
    );
    expect(logger.error).not.toHaveBeenCalled();
  });

  it("allocates after the greatest retained timestamp when a fixed clock leaves pruned holes", async () => {
    const dataDir = temporaryDirectory("borg-capture-fixed-clock-");
    const fileName = "test-contexts.jsonl";
    const path = join(dataDir, "captures", fileName);
    const timestampMs = Date.UTC(2026, 8, 4, 12);
    const records = Array.from({ length: 5 }, (_, sequence) => ({ sequence }));
    const maxFileBytes = Buffer.byteLength(`${JSON.stringify(records[0])}\n`);
    const logger = { info: vi.fn(), error: vi.fn() };

    await appendBoundedContextCapture({
      dataDir,
      fileName,
      record: records[0],
      maxFileBytes,
      rotationKeep: 2,
      rotationTimestampMs: timestampMs,
      logger,
    });
    for (const record of records.slice(1)) {
      await appendBoundedContextCapture({
        dataDir,
        fileName,
        record,
        maxFileBytes,
        rotationKeep: 2,
        rotationTimestampMs: timestampMs,
        logger,
      });
      await waitForContextCaptureMaintenance(path);
    }

    const archives = readdirSync(join(dataDir, "captures"))
      .filter((name) => name.startsWith(`${fileName}.rotated-`))
      .sort();
    expect(archives).toEqual([
      `${fileName}.rotated-20260904T120000.002Z.gz`,
      `${fileName}.rotated-20260904T120000.003Z.gz`,
    ]);
    expect(archives.flatMap((name) => readJsonlRecords(join(dataDir, "captures", name)))).toEqual([
      records[2],
      records[3],
    ]);
    expect(readJsonlRecords(path)).toEqual([records[4]]);
    expect(logger.error).not.toHaveBeenCalled();
  });

  it("coalesces a burst of rotations while the active gzip claim is stalled", async () => {
    const dataDir = temporaryDirectory("borg-capture-stalled-gzip-");
    const fileName = "test-contexts.jsonl";
    const captureDirectory = join(dataDir, "captures");
    const path = join(captureDirectory, fileName);
    const records = Array.from({ length: 16 }, (_, sequence) => ({
      sequence,
      padding: "fixed-width",
    }));
    const maxFileBytes = Math.max(
      ...records.map((record) => Buffer.byteLength(`${JSON.stringify(record)}\n`)),
    );
    const logger = { info: vi.fn(), error: vi.fn() };
    await appendBoundedContextCapture({
      dataDir,
      fileName,
      record: records[0],
      maxFileBytes,
      rotationKeep: records.length,
      rotationTimestampMs: Date.UTC(2026, 8, 4, 12),
      logger,
    });
    pipelineStall.gate = new Promise<void>((resolve) => {
      pipelineStall.release = resolve;
    });

    await appendBoundedContextCapture({
      dataDir,
      fileName,
      record: records[1],
      maxFileBytes,
      rotationKeep: records.length,
      rotationTimestampMs: Date.UTC(2026, 8, 4, 12),
      logger,
    });
    await vi.waitFor(() => expect(pipelineStall.stalledCalls).toBe(1));
    await Promise.all(
      records.slice(2).map((record) =>
        appendBoundedContextCapture({
          dataDir,
          fileName,
          record,
          maxFileBytes,
          rotationKeep: records.length,
          rotationTimestampMs: Date.UTC(2026, 8, 4, 12),
          logger,
        }),
      ),
    );

    expect(pipelineStall.stalledCalls).toBe(1);
    expect(
      readdirSync(captureDirectory).filter((name) => name.endsWith(".gz.partial")),
    ).toHaveLength(1);
    const release = pipelineStall.release;
    pipelineStall.gate = null;
    pipelineStall.release = null;
    release?.();
    await waitForContextCaptureMaintenance(path);

    expect(readdirSync(captureDirectory).some((name) => name.endsWith(".partial"))).toBe(false);
    const stored = readdirSync(captureDirectory)
      .filter((name) => name === fileName || name.startsWith(`${fileName}.rotated-`))
      .flatMap((name) => readJsonlRecords(join(captureDirectory, name)))
      .map((record) => record.sequence)
      .sort((left, right) => left - right);
    expect(stored).toEqual(records.map((record) => record.sequence));
    expect(logger.error).not.toHaveBeenCalled();
  });

  it("protects a plain archive while another process owns its partial gzip claim", async () => {
    const dataDir = temporaryDirectory("borg-capture-partial-claim-");
    const captureDirectory = join(dataDir, "captures");
    const fileName = "test-contexts.jsonl";
    const path = join(captureDirectory, fileName);
    mkdirSync(captureDirectory, { recursive: true });
    const claimedPath = `${path}.rotated-20260101T000000.000Z`;
    const olderGzipPath = `${path}.rotated-20260201T000000.000Z.gz`;
    writeFileSync(claimedPath, '{"sequence":0}\n');
    writeFileSync(`${claimedPath}.gz.partial`, "in progress elsewhere");
    writeFileSync(olderGzipPath, gzipSync('{"sequence":1}\n'));
    const existing = { sequence: 2 };
    const triggering = { sequence: 3 };
    writeFileSync(path, `${JSON.stringify(existing)}\n`);
    const logger = { info: vi.fn(), error: vi.fn() };

    await appendBoundedContextCapture({
      dataDir,
      fileName,
      record: triggering,
      maxFileBytes: Buffer.byteLength(`${JSON.stringify(existing)}\n`),
      rotationKeep: 1,
      rotationTimestampMs: Date.UTC(2026, 2, 1),
      logger,
    });
    await waitForContextCaptureMaintenance(path);

    expect(existsSync(claimedPath)).toBe(true);
    expect(existsSync(`${claimedPath}.gz.partial`)).toBe(true);
    expect(existsSync(olderGzipPath)).toBe(false);
    expect(existsSync(`${path}.rotated-20260301T000000.000Z.gz`)).toBe(true);
    expect(logger.error).not.toHaveBeenCalled();
  });

  it("serializes concurrent cap rotations without losing or duplicating records", async () => {
    const dataDir = temporaryDirectory("borg-capture-concurrent-rotation-");
    const fileName = "test-contexts.jsonl";
    const path = join(dataDir, "captures", fileName);
    const records = Array.from({ length: 12 }, (_, sequence) => ({
      sequence,
      padding: "fixed-width",
    }));
    const lineBytes = Buffer.byteLength(`${JSON.stringify(records[0])}\n`);
    const logger = { info: vi.fn(), error: vi.fn() };

    await Promise.all(
      records.map((record) =>
        appendBoundedContextCapture({
          dataDir,
          fileName,
          record,
          maxFileBytes: lineBytes * 2,
          rotationKeep: records.length,
          rotationTimestampMs: Date.UTC(2026, 8, 4, 12),
          logger,
        }),
      ),
    );
    await waitForContextCaptureMaintenance(path);

    const stored = readdirSync(join(dataDir, "captures"))
      .filter((name) => name === fileName || name.startsWith(`${fileName}.rotated-`))
      .flatMap((name) => readJsonlRecords(join(dataDir, "captures", name)))
      .map((record) => record.sequence)
      .sort((left, right) => left - right);
    expect(stored).toEqual(records.map((record) => record.sequence));
    expect(logger.error).not.toHaveBeenCalled();
  });
});

describe("content-addressed context capture sidecars", () => {
  it("stages and commits private, hash-verified bytes under a 0022 umask", () => {
    const dataDir = temporaryDirectory("borg-capture-sidecar-");
    const pending = createContentAddressedCaptureSidecar({
      subdirectory: "test-sidecars",
      bytes: Buffer.from("private payload"),
    });
    const previousUmask = process.umask(0o022);
    try {
      const staged = stageContentAddressedCaptureSidecars(dataDir, [pending]);
      expect(statSync(join(dataDir, "captures")).mode & 0o777).toBe(0o700);
      expect(statSync(join(dataDir, "captures", "test-sidecars")).mode & 0o777).toBe(0o700);
      expect(statSync(staged[0]!.stagedPath!).mode & 0o777).toBe(0o600);

      commitStagedContentAddressedCaptureSidecars(staged);
      const finalPath = join(dataDir, "captures", pending.relativePath);
      expect(statSync(finalPath).mode & 0o777).toBe(0o600);
      expect(
        Buffer.from(
          readContentAddressedCaptureSidecar({
            dataDir,
            relativePath: pending.relativePath,
            sha256: pending.sha256,
            byteSize: pending.byteSize,
          }),
        ).toString("utf8"),
      ).toBe("private payload");
    } finally {
      process.umask(previousUmask);
    }
  });

  it("rejects a sidecar symlink that escapes the capture directory", () => {
    const dataDir = temporaryDirectory("borg-capture-sidecar-");
    const outside = temporaryDirectory("borg-capture-outside-");
    const pending = createContentAddressedCaptureSidecar({
      subdirectory: "test-sidecars",
      bytes: Buffer.from("private payload"),
    });
    const sidecarDirectory = join(dataDir, "captures", "test-sidecars");
    mkdirSync(sidecarDirectory, { recursive: true });
    const outsidePath = join(outside, "payload");
    writeFileSync(outsidePath, pending.bytes);
    symlinkSync(outsidePath, join(dataDir, "captures", pending.relativePath));

    expect(() =>
      readContentAddressedCaptureSidecar({
        dataDir,
        relativePath: pending.relativePath,
        sha256: pending.sha256,
      }),
    ).toThrow(/resolve below the captures directory/);
  });

  it("rejects a content-addressed file whose bytes no longer match its hash", () => {
    const dataDir = temporaryDirectory("borg-capture-sidecar-");
    const pending = createContentAddressedCaptureSidecar({
      subdirectory: "test-sidecars",
      bytes: Buffer.from("original"),
    });
    const staged = stageContentAddressedCaptureSidecars(dataDir, [pending]);
    commitStagedContentAddressedCaptureSidecars(staged);
    writeFileSync(join(dataDir, "captures", pending.relativePath), "tampered");

    expect(() =>
      readContentAddressedCaptureSidecar({
        dataDir,
        relativePath: pending.relativePath,
        sha256: pending.sha256,
      }),
    ).toThrow(/hash mismatch/);
  });

  it("removes staged bytes without committing them", () => {
    const dataDir = temporaryDirectory("borg-capture-sidecar-");
    const pending = createContentAddressedCaptureSidecar({
      subdirectory: "test-sidecars",
      bytes: Buffer.from("rollback"),
    });
    const staged = stageContentAddressedCaptureSidecars(dataDir, [pending]);
    const stagedPath = staged[0]!.stagedPath!;
    discardStagedContentAddressedCaptureSidecars(staged);

    expect(existsSync(stagedPath)).toBe(false);
    expect(existsSync(join(dataDir, "captures", pending.relativePath))).toBe(false);
  });
});
