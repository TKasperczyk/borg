import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  commitStagedContentAddressedCaptureSidecars,
  createContentAddressedCaptureSidecar,
  discardStagedContentAddressedCaptureSidecars,
  readContentAddressedCaptureSidecar,
  stageContentAddressedCaptureSidecars,
} from "./context-capture-storage.js";

const tempDirs: string[] = [];

afterEach(() => {
  for (const directory of tempDirs.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

function temporaryDirectory(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempDirs.push(directory);
  return directory;
}

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
