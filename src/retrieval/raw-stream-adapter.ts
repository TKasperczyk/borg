import {
  StreamReader,
  collectAbortedTurnRefs,
  filterActiveStreamEntries,
  streamEntryIsActive,
  type StreamEntry,
  type StreamEntryIndexRepository,
} from "../stream/index.js";
import type { SessionId, StreamEntryId } from "../util/ids.js";

import { CitationResolver } from "./citations.js";

const RECENT_TAIL_MULTIPLIER = 4;

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
    const entries = await this.citationResolver.resolveCitationEntries(ids);
    const sessionIds = [...new Set([...entries.values()].map((entry) => entry.session_id))];
    const markerEntries: StreamEntry[] = [];

    for (const sessionId of sessionIds) {
      const reader = new StreamReader({
        dataDir: this.options.dataDir,
        sessionId,
      });

      for await (const entry of reader.iterate({ kinds: ["internal_event"] })) {
        markerEntries.push(entry);
      }
    }

    const refs = collectAbortedTurnRefs(markerEntries);
    return new Map([...entries].filter(([, entry]) => streamEntryIsActive(entry, refs)));
  }

  recent(options: { limit?: number; sessionId?: SessionId } = {}): StreamEntry[] {
    const limit = Math.max(1, options.limit ?? 4);
    const tailLimit = limit * RECENT_TAIL_MULTIPLIER;
    const entries: StreamEntry[] = [];
    const sessionIds =
      options.sessionId === undefined
        ? this.citationResolver.listSessionIds()
        : [options.sessionId];

    for (const sessionId of sessionIds) {
      const reader = new StreamReader({
        dataDir: this.options.dataDir,
        sessionId,
      });

      entries.push(...filterActiveStreamEntries(reader.tail(tailLimit)));
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
