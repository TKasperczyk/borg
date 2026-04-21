import {
  closeSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";

const DIRECTORY_FSYNC_UNSUPPORTED = new Set(["EBADF", "EINVAL", "ENOTSUP"]);

function isNodeError(error: unknown): error is NodeJS.ErrnoException & { code: string } {
  return error instanceof Error && typeof (error as NodeJS.ErrnoException).code === "string";
}

function syncDirectory(path: string): void {
  let directoryFd: number | undefined;

  try {
    directoryFd = openSync(path, "r");
    fsyncSync(directoryFd);
  } catch (error) {
    if (isNodeError(error) && DIRECTORY_FSYNC_UNSUPPORTED.has(error.code)) {
      return;
    }

    throw error;
  } finally {
    if (directoryFd !== undefined) {
      closeSync(directoryFd);
    }
  }
}

function createTempFilePath(targetPath: string): string {
  return join(
    dirname(targetPath),
    `.${basename(targetPath)}.${process.pid}.${Date.now()}.${Math.random()
      .toString(16)
      .slice(2)}.tmp`,
  );
}

export function writeFileAtomic(filePath: string, data: string | Buffer | Uint8Array): void {
  const targetDirectory = dirname(filePath);
  const tempPath = createTempFilePath(filePath);
  const bytes =
    typeof data === "string" ? Buffer.from(data) : Buffer.isBuffer(data) ? data : Buffer.from(data);

  mkdirSync(targetDirectory, { recursive: true });

  let tempFd: number | undefined;

  try {
    tempFd = openSync(tempPath, "wx");
    writeFileSync(tempFd, bytes);
    fsyncSync(tempFd);
    closeSync(tempFd);
    tempFd = undefined;

    renameSync(tempPath, filePath);
    syncDirectory(targetDirectory);
  } catch (error) {
    if (tempFd !== undefined) {
      closeSync(tempFd);
    }

    try {
      unlinkSync(tempPath);
    } catch (cleanupError) {
      if (!isNodeError(cleanupError) || cleanupError.code !== "ENOENT") {
        throw cleanupError;
      }
    }

    throw error;
  }
}

export function readJsonFile<T>(filePath: string): T | undefined {
  try {
    const raw = readFileSync(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return undefined;
    }

    throw error;
  }
}

export function writeJsonFileAtomic(
  filePath: string,
  value: unknown,
  options: { space?: number } = {},
): void {
  const space = options.space ?? 2;
  const serialized = `${JSON.stringify(value, null, space)}\n`;
  writeFileAtomic(filePath, serialized);
}
