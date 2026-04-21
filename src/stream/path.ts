import { join } from "node:path";

import type { SessionId } from "./types.js";

export function getStreamDirectory(dataDir: string): string {
  return join(dataDir, "stream");
}

export function getSessionStreamPath(dataDir: string, sessionId: SessionId): string {
  return join(getStreamDirectory(dataDir), `${sessionId}.jsonl`);
}
