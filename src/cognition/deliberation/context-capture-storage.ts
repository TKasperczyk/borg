import { realpathSync } from "node:fs";
import { join } from "node:path";

import { appendDurableJsonl } from "../../util/durable-jsonl.js";
import { isPathWithin, resolveRealPathForCreation } from "../../util/path.js";

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
