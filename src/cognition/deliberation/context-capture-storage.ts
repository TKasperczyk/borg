import { randomUUID } from "node:crypto";
import {
  chmodSync,
  closeSync,
  constants,
  createReadStream,
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
import {
  lstat as lstatAsync,
  open as openAsync,
  readdir as readdirAsync,
  rename as renameAsync,
  unlink as unlinkAsync,
} from "node:fs/promises";
import { basename, dirname, isAbsolute, join } from "node:path";
import { pipeline } from "node:stream/promises";
import { createGzip } from "node:zlib";

import { appendDurableJsonl } from "../../util/durable-jsonl.js";
import { syncDirectory, writeJsonFileAtomic } from "../../util/atomic-write.js";
import { isNodeError } from "../../util/guards.js";
import { isPathWithin, resolveRealPathForCreation } from "../../util/path.js";
import { DEFAULT_CONTEXT_CAPTURE_ROTATION_KEEP } from "./constants.js";
import { sha256Bytes } from "./request-fingerprint.js";

const PRIVATE_DIRECTORY_MODE = 0o700;
const PRIVATE_FILE_MODE = 0o600;

export type ContextCaptureLogger = Pick<Console, "error" | "info">;

const DIRECTORY_FSYNC_UNSUPPORTED = new Set(["EBADF", "EINVAL", "ENOTSUP"]);
const GZIP_SUFFIX = ".gz";
const GZIP_PARTIAL_SUFFIX = ".gz.partial";

type RotatedCaptureFile = {
  archivePath: string;
  filePath: string;
  kind: "gzip" | "partial" | "plain";
  timestampMs: number;
};

type ContextCaptureMaintenanceWorker = {
  keep: number;
  logger: ContextCaptureLogger;
  pending: boolean;
  worker?: Promise<void>;
};

const contextCaptureMaintenanceWorkers = new Map<string, ContextCaptureMaintenanceWorker>();

function emitCaptureLog(
  logger: ContextCaptureLogger,
  level: keyof ContextCaptureLogger,
  message: string,
  details: Record<string, unknown>,
): void {
  try {
    logger[level](message, details);
  } catch {
    // Capture maintenance is observational and must not become a turn dependency.
  }
}

function rotationTimestamp(timestampMs: number): string {
  return new Date(timestampMs).toISOString().replaceAll("-", "").replaceAll(":", "");
}

function parseRotationTimestamp(value: string): number | null {
  if (value.length !== 20 || value[8] !== "T" || value[15] !== "." || value[19] !== "Z") {
    return null;
  }
  const timestampMs = Date.parse(
    `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)}T${value.slice(9, 11)}:${value.slice(11, 13)}:${value.slice(13)}`,
  );
  return Number.isFinite(timestampMs) && rotationTimestamp(timestampMs) === value
    ? timestampMs
    : null;
}

function parseRotatedCaptureFile(path: string, name: string): RotatedCaptureFile | null {
  const prefix = `${basename(path)}.rotated-`;
  if (!name.startsWith(prefix)) return null;

  const suffix = name.slice(prefix.length);
  const kind = suffix.endsWith(GZIP_PARTIAL_SUFFIX)
    ? "partial"
    : suffix.endsWith(GZIP_SUFFIX)
      ? "gzip"
      : "plain";
  const extensionLength =
    kind === "partial" ? GZIP_PARTIAL_SUFFIX.length : kind === "gzip" ? GZIP_SUFFIX.length : 0;
  const timestamp = suffix.slice(0, suffix.length - extensionLength);
  const timestampMs = parseRotationTimestamp(timestamp);
  if (timestampMs === null) return null;

  return {
    archivePath: join(dirname(path), `${prefix}${timestamp}`),
    filePath: join(dirname(path), name),
    kind,
    timestampMs,
  };
}

function nextRotatedCapturePath(path: string, timestampMs: number): string {
  let greatestExistingTimestampMs: number | null = null;
  for (const name of readdirSync(dirname(path))) {
    const rotated = parseRotatedCaptureFile(path, name);
    if (
      rotated !== null &&
      (greatestExistingTimestampMs === null || rotated.timestampMs > greatestExistingTimestampMs)
    ) {
      greatestExistingTimestampMs = rotated.timestampMs;
    }
  }
  const allocatedTimestampMs =
    greatestExistingTimestampMs === null
      ? timestampMs
      : Math.max(timestampMs, greatestExistingTimestampMs + 1);
  return `${path}.rotated-${rotationTimestamp(allocatedTimestampMs)}`;
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await lstatAsync(path);
    return true;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return false;
    throw error;
  }
}

async function syncDirectoryAsync(path: string): Promise<void> {
  let directoryHandle: Awaited<ReturnType<typeof openAsync>> | undefined;
  try {
    directoryHandle = await openAsync(path, "r");
    await directoryHandle.sync();
  } catch (error) {
    if (!isNodeError(error) || !DIRECTORY_FSYNC_UNSUPPORTED.has(error.code)) {
      throw error;
    }
  } finally {
    await directoryHandle?.close();
  }
}

async function removeOwnedPartialCapture(partialPath: string): Promise<void> {
  try {
    await unlinkAsync(partialPath);
    await syncDirectoryAsync(dirname(partialPath));
  } catch (error) {
    if (!isNodeError(error) || error.code !== "ENOENT") throw error;
  }
}

async function compressRotatedCapture(rotatedPath: string): Promise<void> {
  let stats: Awaited<ReturnType<typeof lstatAsync>>;
  try {
    stats = await lstatAsync(rotatedPath);
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") return;
    throw error;
  }
  if (stats.isSymbolicLink() || !stats.isFile()) {
    throw new Error(`Rotated context capture is not a regular file: ${rotatedPath}`);
  }

  const directory = dirname(rotatedPath);
  const compressedPath = `${rotatedPath}${GZIP_SUFFIX}`;
  const partialPath = `${rotatedPath}${GZIP_PARTIAL_SUFFIX}`;
  if (await pathExists(compressedPath)) return;

  let partialHandle: Awaited<ReturnType<typeof openAsync>> | undefined;
  try {
    partialHandle = await openAsync(
      partialPath,
      constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | constants.O_NOFOLLOW,
      PRIVATE_FILE_MODE,
    );
  } catch (error) {
    if (isNodeError(error) && error.code === "EEXIST") return;
    throw error;
  }

  try {
    await pipeline(createReadStream(rotatedPath), createGzip(), partialHandle.createWriteStream());
    partialHandle = undefined;
    const completedHandle = await openAsync(partialPath, constants.O_RDWR | constants.O_NOFOLLOW);
    try {
      await completedHandle.chmod(PRIVATE_FILE_MODE);
      await completedHandle.sync();
    } finally {
      await completedHandle.close();
    }
    await renameAsync(partialPath, compressedPath);
    await syncDirectoryAsync(directory);
    await unlinkAsync(rotatedPath);
    await syncDirectoryAsync(directory);
  } catch (error) {
    try {
      await partialHandle?.close();
    } catch {
      // Preserve the compression failure and continue owned-claim cleanup.
    }
    await removeOwnedPartialCapture(partialPath);
    throw error;
  }
}

async function rotatedCaptureFiles(path: string): Promise<RotatedCaptureFile[]> {
  const files: RotatedCaptureFile[] = [];
  for (const name of await readdirAsync(dirname(path))) {
    const rotated = parseRotatedCaptureFile(path, name);
    if (rotated === null) continue;
    try {
      const stats = await lstatAsync(rotated.filePath);
      if (!stats.isSymbolicLink() && stats.isFile()) files.push(rotated);
    } catch (error) {
      if (!isNodeError(error) || error.code !== "ENOENT") throw error;
    }
  }
  return files.sort(
    (left, right) =>
      left.timestampMs - right.timestampMs || left.filePath.localeCompare(right.filePath),
  );
}

async function pruneRotatedCaptures(input: {
  path: string;
  keep: number;
  logger: ContextCaptureLogger;
}): Promise<void> {
  const allFiles = await rotatedCaptureFiles(input.path);
  const claimedArchivePaths = new Set(
    allFiles.filter((file) => file.kind === "partial").map((file) => file.archivePath),
  );
  const files = allFiles.filter(
    (file) =>
      file.kind !== "partial" &&
      !(file.kind === "plain" && claimedArchivePaths.has(file.archivePath)),
  );
  const pruneCount = Math.max(0, files.length - input.keep);
  for (const pruned of files.slice(0, pruneCount)) {
    if (
      pruned.kind === "plain" &&
      (await pathExists(`${pruned.archivePath}${GZIP_PARTIAL_SUFFIX}`))
    ) {
      continue;
    }
    try {
      await unlinkAsync(pruned.filePath);
    } catch (error) {
      if (isNodeError(error) && error.code === "ENOENT") continue;
      throw error;
    }
    await syncDirectoryAsync(dirname(pruned.filePath));
    emitCaptureLog(input.logger, "info", "Pruned rotated deliberation context capture", {
      capturePath: input.path,
      prunedPath: pruned.filePath,
      rotationKeep: input.keep,
    });
  }
}

async function maintainRotatedCaptures(input: {
  path: string;
  keep: number;
  logger: ContextCaptureLogger;
}): Promise<void> {
  try {
    const files = await rotatedCaptureFiles(input.path);
    const claimedArchivePaths = new Set(
      files.filter((file) => file.kind === "partial").map((file) => file.archivePath),
    );
    const compressedArchivePaths = new Set(
      files.filter((file) => file.kind === "gzip").map((file) => file.archivePath),
    );
    for (const file of files) {
      if (
        file.kind !== "plain" ||
        claimedArchivePaths.has(file.archivePath) ||
        compressedArchivePaths.has(file.archivePath)
      ) {
        continue;
      }
      try {
        await compressRotatedCapture(file.archivePath);
      } catch (error) {
        emitCaptureLog(input.logger, "error", "Failed to compress rotated context capture", {
          capturePath: input.path,
          rotatedPath: file.archivePath,
          cause: error instanceof Error ? error.message : String(error),
        });
      }
    }
  } catch (error) {
    emitCaptureLog(input.logger, "error", "Failed to scan rotated context captures", {
      capturePath: input.path,
      cause: error instanceof Error ? error.message : String(error),
    });
  }

  try {
    await pruneRotatedCaptures(input);
  } catch (error) {
    emitCaptureLog(input.logger, "error", "Failed to prune rotated context captures", {
      capturePath: input.path,
      cause: error instanceof Error ? error.message : String(error),
    });
  }
}

async function drainContextCaptureMaintenance(
  path: string,
  state: ContextCaptureMaintenanceWorker,
): Promise<void> {
  try {
    while (state.pending) {
      state.pending = false;
      await maintainRotatedCaptures({ path, keep: state.keep, logger: state.logger });
    }
  } finally {
    if (contextCaptureMaintenanceWorkers.get(path) === state) {
      contextCaptureMaintenanceWorkers.delete(path);
    }
  }
}

function scheduleRotatedCaptureMaintenance(input: {
  path: string;
  keep: number;
  logger: ContextCaptureLogger;
}): void {
  const existing = contextCaptureMaintenanceWorkers.get(input.path);
  if (existing !== undefined) {
    existing.keep = input.keep;
    existing.logger = input.logger;
    existing.pending = true;
    return;
  }

  const state: ContextCaptureMaintenanceWorker = {
    keep: input.keep,
    logger: input.logger,
    pending: true,
  };
  state.worker = Promise.resolve().then(async () =>
    drainContextCaptureMaintenance(input.path, state),
  );
  contextCaptureMaintenanceWorkers.set(input.path, state);
  void state.worker.catch((error: unknown) => {
    emitCaptureLog(input.logger, "error", "Context capture maintenance worker failed", {
      capturePath: input.path,
      cause: error instanceof Error ? error.message : String(error),
    });
  });
}

export async function waitForContextCaptureMaintenance(path?: string): Promise<void> {
  while (true) {
    const pending =
      path === undefined
        ? [...contextCaptureMaintenanceWorkers.values()].flatMap((state) =>
            state.worker === undefined ? [] : [state.worker],
          )
        : [contextCaptureMaintenanceWorkers.get(path)?.worker].filter(
            (worker): worker is Promise<void> => worker !== undefined,
          );
    if (pending.length === 0) return;
    await Promise.all(pending);
  }
}

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
  rotationKeep?: number;
  rotationTimestampMs?: number;
  logger?: ContextCaptureLogger;
}): Promise<
  | { status: "appended"; path: string; bytes: number }
  | { status: "rotated"; path: string; rotatedPath: string; bytes: number }
> {
  const rotationKeep = input.rotationKeep ?? DEFAULT_CONTEXT_CAPTURE_ROTATION_KEEP;
  if (!Number.isInteger(rotationKeep) || rotationKeep < 0) {
    throw new Error("Context capture rotation keep must be a non-negative integer");
  }
  const { path, captureDirectory } = resolveContextCaptureStoragePath(
    input.dataDir,
    input.fileName,
  );
  const bytes = Buffer.byteLength(`${JSON.stringify(input.record)}\n`);
  const result = await appendDurableJsonl(path, input.record, {
    maxFileBytes: input.maxFileBytes,
    privateDirectory: captureDirectory,
    rotatedFilePath: () => nextRotatedCapturePath(path, input.rotationTimestampMs ?? Date.now()),
  });
  if (result.status === "file_full") {
    throw new Error(`Context capture rotation was not applied at ${path}`);
  }
  if (result.status === "rotated") {
    const logger = input.logger ?? console;
    emitCaptureLog(logger, "info", "Rotated deliberation context capture", {
      capturePath: path,
      rotatedPath: result.rotatedPath,
    });
    scheduleRotatedCaptureMaintenance({
      path,
      keep: rotationKeep,
      logger,
    });
    return { status: "rotated", path, rotatedPath: result.rotatedPath, bytes };
  }
  return { status: "appended", path, bytes };
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
