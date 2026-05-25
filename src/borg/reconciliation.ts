// Startup reconciliation for stores that derive state from the stream or LanceDB.

import { existsSync, readdirSync } from "node:fs";

import type { AttachmentRepository } from "../attachments/index.js";
import {
  getStreamDirectory,
  StreamReader,
  type StreamEntry,
  type StreamEntryIndexRepository,
} from "../stream/index.js";
import { parseSessionId, type SessionId } from "../util/ids.js";

export async function backfillStreamEntryIndex(options: {
  dataDir: string;
  entryIndex: StreamEntryIndexRepository;
  attachmentRepository?: Pick<
    AttachmentRepository,
    "reconcileFromStreamEntries" | "reconcileActiveStateFromStreamIndex"
  >;
}): Promise<void> {
  const streamDir = getStreamDirectory(options.dataDir);

  if (!existsSync(streamDir)) {
    return;
  }

  const sessionIds = readdirSync(streamDir)
    .map((filename) => {
      if (!filename.endsWith(".jsonl")) {
        return null;
      }

      try {
        return parseSessionId(filename.slice(0, -".jsonl".length));
      } catch {
        return null;
      }
    })
    .filter((sessionId): sessionId is SessionId => sessionId !== null);

  for (const sessionId of sessionIds) {
    await options.entryIndex.backfillSession(sessionId);

    if (options.attachmentRepository !== undefined) {
      const entries: StreamEntry[] = [];
      for await (const entry of new StreamReader({
        dataDir: options.dataDir,
        sessionId,
      }).iterate({ kinds: ["user_image_attachment"] })) {
        entries.push(entry);
      }
      options.attachmentRepository.reconcileFromStreamEntries(entries);
    }
  }

  options.attachmentRepository?.reconcileActiveStateFromStreamIndex();
  options.entryIndex.warnLegacyRowsMissingKind();
}
