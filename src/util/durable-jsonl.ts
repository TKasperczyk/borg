import {
  chmodSync,
  closeSync,
  constants,
  existsSync,
  fchmodSync,
  fsyncSync,
  fstatSync,
  ftruncateSync,
  mkdirSync,
  openSync,
  readSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";

import { withFileLock } from "../stream/file-lock.js";
import { syncDirectory } from "./atomic-write.js";

const PRIVATE_DIRECTORY_MODE = 0o700;
const PRIVATE_FILE_MODE = 0o600;
const TAIL_SCAN_CHUNK_BYTES = 64 * 1024;

export type DurableJsonlAppendResult =
  | {
      status: "appended";
      bytes: number;
      startOffset: number;
      repairedTailBytes: number;
    }
  | {
      status: "file_full";
      bytes: number;
      repairedTailBytes: number;
    };

export type DurableJsonlAppendOptions = {
  maxFileBytes?: number;
  /** Repair this owned directory to 0700 while the append lock is held. */
  privateDirectory?: string;
};

function lastCompleteRecordOffset(fileDescriptor: number, size: number): number {
  if (size === 0) {
    return 0;
  }

  const lastByte = Buffer.allocUnsafe(1);
  readSync(fileDescriptor, lastByte, 0, 1, size - 1);
  if (lastByte[0] === 0x0a) {
    return size;
  }

  let scanEnd = size;
  while (scanEnd > 0) {
    const scanStart = Math.max(0, scanEnd - TAIL_SCAN_CHUNK_BYTES);
    const buffer = Buffer.allocUnsafe(scanEnd - scanStart);
    readSync(fileDescriptor, buffer, 0, buffer.length, scanStart);
    const newlineIndex = buffer.lastIndexOf(0x0a);
    if (newlineIndex >= 0) {
      return scanStart + newlineIndex + 1;
    }
    scanEnd = scanStart;
  }

  return 0;
}

function ensureParentDirectory(filePath: string, privateDirectory: string | undefined): void {
  const parent = dirname(filePath);
  mkdirSync(parent, { recursive: true, mode: PRIVATE_DIRECTORY_MODE });
  if (privateDirectory !== undefined) {
    mkdirSync(privateDirectory, { recursive: true, mode: PRIVATE_DIRECTORY_MODE });
  }
}

export async function appendDurableJsonl(
  filePath: string,
  record: unknown,
  options: DurableJsonlAppendOptions = {},
): Promise<DurableJsonlAppendResult> {
  const line = `${JSON.stringify(record)}\n`;
  const bytes = Buffer.byteLength(line);
  const parent = dirname(filePath);
  ensureParentDirectory(filePath, options.privateDirectory);

  return withFileLock(`${filePath}.lock`, () => {
    if (options.privateDirectory !== undefined) {
      chmodSync(options.privateDirectory, PRIVATE_DIRECTORY_MODE);
    }

    const existedBeforeOpen = existsSync(filePath);
    const fileDescriptor = openSync(
      filePath,
      constants.O_APPEND | constants.O_CREAT | constants.O_RDWR | constants.O_NOFOLLOW,
      PRIVATE_FILE_MODE,
    );
    let startOffset = 0;
    let repairedTailBytes = 0;

    try {
      fchmodSync(fileDescriptor, PRIVATE_FILE_MODE);
      const originalSize = fstatSync(fileDescriptor).size;
      startOffset = lastCompleteRecordOffset(fileDescriptor, originalSize);
      repairedTailBytes = originalSize - startOffset;
      if (repairedTailBytes > 0) {
        ftruncateSync(fileDescriptor, startOffset);
        fsyncSync(fileDescriptor);
      }

      if (options.maxFileBytes !== undefined && startOffset + bytes > options.maxFileBytes) {
        return { status: "file_full", bytes, repairedTailBytes };
      }

      try {
        writeFileSync(fileDescriptor, line);
        fsyncSync(fileDescriptor);
        if (!existedBeforeOpen) {
          syncDirectory(parent);
        }
      } catch (error) {
        ftruncateSync(fileDescriptor, startOffset);
        fsyncSync(fileDescriptor);
        throw error;
      }

      return { status: "appended", bytes, startOffset, repairedTailBytes };
    } finally {
      closeSync(fileDescriptor);
    }
  });
}
