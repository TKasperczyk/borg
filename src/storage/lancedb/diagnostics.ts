// Plain diagnostic shapes: no native handles, Arrow values, or row payloads.
export type LanceDbQueryCounters = {
  listQueries: number;
  searchQueries: number;
  upserts: number;
  rowsRead: number;
};

export type LanceDbMemoryStats = {
  cache: {
    sizeBytes: number;
    approxItems: number;
    indexMaxBytes: number;
    metadataMaxBytes: number;
  } | null;
  tables: Array<LanceDbQueryCounters & { table: string; openHandles: number }>;
  lastOptimization: {
    completedAt: number;
    durationMs: number;
    fragmentsRemoved: number;
    fragmentsAdded: number;
    versionsPruned: number;
    errors: number;
  } | null;
};

export type LanceDbTableStats = {
  table: string;
  fragments: number | null;
  smallFragments: number | null;
  version: number | null;
  versions: number | null;
  durationMs: number;
  error?: string;
};
