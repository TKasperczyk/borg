import { randomUUID } from "node:crypto";
import {
  chmodSync,
  closeSync,
  constants,
  existsSync,
  fsyncSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  realpathSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, isAbsolute, join } from "node:path";

import { appendDurableJsonl } from "../../util/durable-jsonl.js";
import { syncDirectory, writeJsonFileAtomic } from "../../util/atomic-write.js";
import { isPathWithin, resolveRealPathForCreation } from "../../util/path.js";
import { sha256Bytes } from "./request-fingerprint.js";

const PRIVATE_DIRECTORY_MODE = 0o700;
const PRIVATE_FILE_MODE = 0o600;

export type PendingContentAddressedCaptureSidecar = {
  sha256: string;
  byteSize: number;
  relativePath: string;
  bytes: Uint8Array;
};

export type StagedContentAddressedCaptureSidecar = {
  sha256: string;
  stagedPath: string | null;
  finalPath: string;
  directory: string;
};

export function resolveContextCaptureStoragePath(
  dataDir: string,
  fileName: string,
): { path: string; captureDirectory: string } {
  const dataDirectory = realpathSync(dataDir);
  const captureDirectory = resolveRealPathForCreation(join(dataDirectory, "captures"));
  if (!isPathWithin(dataDirectory, captureDirectory) || captureDirectory === dataDirectory) {
    throw new Error("Context capture directory must resolve below the Borg data dir");
  }
  const path = resolveRealPathForCreation(join(captureDirectory, fileName));
  if (!isPathWithin(captureDirectory, path)) {
    throw new Error("Context capture file must resolve below the captures directory");
  }
  return { path, captureDirectory };
}

export function resolveContextCaptureSubdirectory(dataDir: string, name: string): string {
  const { captureDirectory } = resolveContextCaptureStoragePath(dataDir, "containment-check.jsonl");
  const directory = resolveRealPathForCreation(join(captureDirectory, name));
  if (!isPathWithin(captureDirectory, directory) || directory === captureDirectory) {
    throw new Error("Context capture sidecar directory must resolve below the captures directory");
  }
  return directory;
}

function ensurePrivateCaptureDirectory(dataDir: string, subdirectory: string): string {
  const { captureDirectory } = resolveContextCaptureStoragePath(dataDir, "containment-check.jsonl");
  const directory = resolveContextCaptureSubdirectory(dataDir, subdirectory);
  mkdirSync(captureDirectory, { recursive: true, mode: PRIVATE_DIRECTORY_MODE });
  chmodSync(captureDirectory, PRIVATE_DIRECTORY_MODE);
  mkdirSync(directory, { recursive: true, mode: PRIVATE_DIRECTORY_MODE });
  chmodSync(directory, PRIVATE_DIRECTORY_MODE);
  return directory;
}

function sidecarSubdirectory(relativePath: string): string {
  if (isAbsolute(relativePath)) {
    throw new Error("Context capture sidecar path must be relative");
  }
  const subdirectory = dirname(relativePath);
  if (subdirectory === "." || subdirectory === "..") {
    throw new Error("Context capture sidecar path must include a subdirectory");
  }
  return subdirectory;
}

function verifySidecarFile(path: string, sha256: string, byteSize?: number): Uint8Array {
  const stats = lstatSync(path);
  if (stats.isSymbolicLink() || !stats.isFile()) {
    throw new Error(`Context capture sidecar is not a regular file: ${path}`);
  }
  const fileDescriptor = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const bytes = readFileSync(fileDescriptor);
    if (byteSize !== undefined && bytes.byteLength !== byteSize) {
      throw new Error(`Context capture sidecar byte length mismatch: ${sha256}`);
    }
    if (sha256Bytes(bytes) !== sha256) {
      throw new Error(`Context capture sidecar hash mismatch: ${sha256}`);
    }
    return bytes;
  } finally {
    closeSync(fileDescriptor);
  }
}

function resolveExistingSidecarPath(
  dataDir: string,
  sidecar: Pick<PendingContentAddressedCaptureSidecar, "relativePath" | "sha256">,
): string {
  const subdirectory = sidecarSubdirectory(sidecar.relativePath);
  if (
    basename(sidecar.relativePath) !== sidecar.sha256 ||
    join(subdirectory, sidecar.sha256) !== sidecar.relativePath
  ) {
    throw new Error("Context capture sidecar path must end in its content hash");
  }
  const { captureDirectory } = resolveContextCaptureStoragePath(dataDir, "containment-check.jsonl");
  const candidate = join(captureDirectory, sidecar.relativePath);
  const resolved = realpathSync(candidate);
  if (!isPathWithin(captureDirectory, resolved) || resolved === captureDirectory) {
    throw new Error("Context capture sidecar must resolve below the captures directory");
  }
  return resolved;
}

export function createContentAddressedCaptureSidecar(input: {
  subdirectory: string;
  bytes: Uint8Array;
}): PendingContentAddressedCaptureSidecar {
  const bytes = Buffer.from(input.bytes);
  const sha256 = sha256Bytes(bytes);
  return {
    sha256,
    byteSize: bytes.byteLength,
    relativePath: join(input.subdirectory, sha256),
    bytes,
  };
}

function validatePendingSidecar(sidecar: PendingContentAddressedCaptureSidecar): void {
  if (
    sha256Bytes(sidecar.bytes) !== sidecar.sha256 ||
    sidecar.bytes.byteLength !== sidecar.byteSize
  ) {
    throw new Error(`Context capture sidecar staging metadata mismatch: ${sidecar.sha256}`);
  }
}

function writePrivateStagedSidecar(stagedPath: string, directory: string, bytes: Uint8Array): void {
  let fileDescriptor: number | undefined;
  try {
    fileDescriptor = openSync(
      stagedPath,
      constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | constants.O_NOFOLLOW,
      PRIVATE_FILE_MODE,
    );
    writeFileSync(fileDescriptor, bytes);
    fsyncSync(fileDescriptor);
    closeSync(fileDescriptor);
    fileDescriptor = undefined;
    chmodSync(stagedPath, PRIVATE_FILE_MODE);
  } catch (error) {
    if (fileDescriptor !== undefined) {
      try {
        closeSync(fileDescriptor);
      } catch {
        // Preserve the staging failure; cleanup below remains best effort.
      }
    }
    if (existsSync(stagedPath)) {
      unlinkSync(stagedPath);
      syncDirectory(directory);
    }
    throw error;
  }
}

function stageContentAddressedCaptureSidecar(
  dataDir: string,
  sidecar: PendingContentAddressedCaptureSidecar,
): StagedContentAddressedCaptureSidecar {
  validatePendingSidecar(sidecar);
  const subdirectory = sidecarSubdirectory(sidecar.relativePath);
  const directory = ensurePrivateCaptureDirectory(dataDir, subdirectory);
  const finalPath = join(directory, basename(sidecar.relativePath));
  if (existsSync(finalPath)) {
    verifySidecarFile(finalPath, sidecar.sha256, sidecar.byteSize);
    chmodSync(finalPath, PRIVATE_FILE_MODE);
    return {
      sha256: sidecar.sha256,
      stagedPath: null,
      finalPath,
      directory,
    };
  }

  const stagedPath = join(directory, `.staged-${randomUUID()}-${sidecar.sha256}`);
  writePrivateStagedSidecar(stagedPath, directory, sidecar.bytes);
  return {
    sha256: sidecar.sha256,
    stagedPath,
    finalPath,
    directory,
  };
}

export function stageContentAddressedCaptureSidecars(
  dataDir: string,
  pendingSidecars: readonly PendingContentAddressedCaptureSidecar[],
): StagedContentAddressedCaptureSidecar[] {
  const uniqueSidecars = [
    ...new Map(pendingSidecars.map((sidecar) => [sidecar.relativePath, sidecar])).values(),
  ];
  const staged: StagedContentAddressedCaptureSidecar[] = [];
  try {
    for (const sidecar of uniqueSidecars) {
      staged.push(stageContentAddressedCaptureSidecar(dataDir, sidecar));
    }
    return staged;
  } catch (error) {
    discardStagedContentAddressedCaptureSidecars(staged);
    throw error;
  }
}

export function discardStagedContentAddressedCaptureSidecars(
  stagedSidecars: readonly StagedContentAddressedCaptureSidecar[],
): void {
  const syncedDirectories = new Set<string>();
  for (const sidecar of stagedSidecars) {
    if (sidecar.stagedPath !== null && existsSync(sidecar.stagedPath)) {
      unlinkSync(sidecar.stagedPath);
      syncedDirectories.add(sidecar.directory);
    }
  }
  for (const directory of syncedDirectories) syncDirectory(directory);
}

export function commitStagedContentAddressedCaptureSidecars(
  stagedSidecars: readonly StagedContentAddressedCaptureSidecar[],
): void {
  const syncedDirectories = new Set<string>();
  for (const sidecar of stagedSidecars) {
    if (sidecar.stagedPath === null) continue;
    if (existsSync(sidecar.finalPath)) {
      verifySidecarFile(sidecar.finalPath, sidecar.sha256);
      unlinkSync(sidecar.stagedPath);
    } else {
      renameSync(sidecar.stagedPath, sidecar.finalPath);
      chmodSync(sidecar.finalPath, PRIVATE_FILE_MODE);
    }
    syncedDirectories.add(sidecar.directory);
  }
  for (const directory of syncedDirectories) syncDirectory(directory);
}

export function contentAddressedCaptureSidecarStorageBytes(
  dataDir: string,
  subdirectories: readonly string[],
): number {
  let bytes = 0;
  for (const subdirectory of new Set(subdirectories)) {
    const directory = resolveContextCaptureSubdirectory(dataDir, subdirectory);
    if (!existsSync(directory)) continue;
    for (const name of readdirSync(directory)) {
      const path = join(directory, name);
      const stats = lstatSync(path);
      if (stats.isSymbolicLink()) {
        throw new Error(`Context capture sidecar must not be a symlink: ${path}`);
      }
      if (stats.isFile()) bytes += stats.size;
    }
  }
  return bytes;
}

export function pendingNewContentAddressedCaptureSidecarBytes(
  dataDir: string,
  pendingSidecars: readonly PendingContentAddressedCaptureSidecar[],
): number {
  const uniqueSidecars = new Map(pendingSidecars.map((sidecar) => [sidecar.relativePath, sidecar]));
  let bytes = 0;
  for (const sidecar of uniqueSidecars.values()) {
    const subdirectory = sidecarSubdirectory(sidecar.relativePath);
    const directory = resolveContextCaptureSubdirectory(dataDir, subdirectory);
    const finalPath = join(directory, basename(sidecar.relativePath));
    if (!existsSync(finalPath)) bytes += sidecar.byteSize;
  }
  return bytes;
}

export function readContentAddressedCaptureSidecar(input: {
  dataDir: string;
  relativePath: string;
  sha256: string;
  byteSize?: number;
}): Uint8Array {
  const path = resolveExistingSidecarPath(input.dataDir, {
    relativePath: input.relativePath,
    sha256: input.sha256,
  });
  return verifySidecarFile(path, input.sha256, input.byteSize);
}

export async function appendBoundedContextCapture(input: {
  dataDir: string;
  fileName: string;
  record: unknown;
  maxFileBytes: number;
}): Promise<
  | { status: "appended"; path: string; bytes: number }
  | { status: "file_full"; path: string; bytes: number }
> {
  const { path, captureDirectory } = resolveContextCaptureStoragePath(
    input.dataDir,
    input.fileName,
  );
  const bytes = Buffer.byteLength(`${JSON.stringify(input.record)}\n`);
  const result = await appendDurableJsonl(path, input.record, {
    maxFileBytes: input.maxFileBytes,
    privateDirectory: captureDirectory,
  });
  return { status: result.status, path, bytes };
}

export function writePrivateContextCaptureJson(input: {
  dataDir: string;
  fileName: string;
  value: unknown;
}): string {
  const { path, captureDirectory } = resolveContextCaptureStoragePath(
    input.dataDir,
    input.fileName,
  );
  mkdirSync(captureDirectory, { recursive: true, mode: 0o700 });
  chmodSync(captureDirectory, 0o700);
  writeJsonFileAtomic(path, input.value, { mode: 0o600 });
  chmodSync(path, 0o600);
  return path;
}
