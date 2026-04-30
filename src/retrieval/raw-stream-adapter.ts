import {
  StreamReader,
  type StreamEntry,
  type StreamEntryIndexRepository,
} from "../stream/index.js";
import type { StreamEntryId } from "../util/ids.js";

import { CitationResolver } from "./citations.js";

export type RawStreamAdapterOptions = {
  dataDir: string;
  entryIndex?: StreamEntryIndexRepository;
};

export class RawStreamAdapter {
  private readonly citationResolver: CitationResolver;

  constructor(private readonly options: RawStreamAdapterOptions) {
    this.citationResolver = new CitationResolver({
      dataDir: options.dataDir,
      entryIndex: options.entryIndex,
    });
  }

  async resolveSourceIds(ids: readonly StreamEntryId[]): Promise<Map<string, StreamEntry>> {
    return this.citationResolver.resolveCitationEntries(ids);
  }

  recent(options: { limit?: number } = {}): StreamEntry[] {
    const limit = Math.max(1, options.limit ?? 4);
    const entries: StreamEntry[] = [];

    for (const sessionId of this.citationResolver.listSessionIds()) {
      const reader = new StreamReader({
        dataDir: this.options.dataDir,
        sessionId,
      });

      entries.push(...reader.tail(limit));
    }

    return entries.sort(compareStreamEntriesDescending).slice(0, limit);
  }

  async searchText(_query: string): Promise<StreamEntry[]> {
    throw new Error("Raw stream text search is unavailable; resolve by source id or recency only");
  }
}

function compareStreamEntriesDescending(left: StreamEntry, right: StreamEntry): number {
  if (left.timestamp !== right.timestamp) {
    return right.timestamp - left.timestamp;
  }

  return right.id.localeCompare(left.id);
}
