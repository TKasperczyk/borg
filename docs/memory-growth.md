# Demo service memory growth

Investigation: 2026-09-09, `borg-demo-sol`, Node 25, `@lancedb/lancedb` 0.27.2.
The service runs `demo/server/src/index.ts` through tsx, with the Sol connector
in the same process. All live inspection was read-only; SQLite was opened with
`?mode=ro`. No service restart, allocator change, live optimization, or live data
write was performed.

The operator's measurements show RSS around 4.43 GB while V8 reports only
147 MB heap used, 278 MB heap total, 20 MB external and 15 MB array buffers.
The cgroup had about 4.37 GB anonymous memory and 0.39 GB file cache, with growth
around 1 GB/hour during activity. This points to native allocations or allocator
retention; JavaScript objects and Arrow IPC buffers alone do not explain it.
`arrayBuffers` is included in `external`: do not add them together. RSS, swap,
cgroup usage and V8 counters measure different things and are not an exact
partition of memory.

The changes bound demonstrated retention paths and expose the native cache.
They do **not** establish which fraction of the old process's RSS was live
native data versus freed memory retained by glibc. A comparable live run after
the operator's restart is still required to verify the growth rate is fixed.

## Ranked findings

| Rank | Suspect and evidence | Lifetime / growth | Disposition |
| --- | --- | --- | --- |
| 1 | Oversized native Lance caches. The old `LanceDbStore` constructor called `connect(uri)` without session/cache settings. The installed SDK documents **6 GiB index + 1 GiB metadata** defaults (`node_modules/@lancedb/lancedb/dist/native.d.ts:116`). Connection and table handles live for the Borg runtime. | Native caches fill with reads/writes across turns and versions. These are capacity defaults, not a measurement of the live cache's occupancy. The independent reviewer found no `_indices` directory in the live tables: there is currently no vector index, so the index cache budget is not the current constraint. Metadata cache occupancy remains measurable; no per-turn byte slope can be established statically. | [Constructor and cache telemetry](../src/storage/lancedb/index.ts): explicitly share one Session across every table, with **256 MiB index + 64 MiB metadata**. Expose `sizeBytes()` and `approxNumItems()`. |
| 2 | glibc arena fragmentation/high-water retention. Operator observed many anonymous 64/128 MB mappings, a large main heap, 12 Tokio workers and four libuv workers, while V8/external stayed small. | Repeated native allocation bursts can leave resident free pages across arenas. Mapping sizes are suggestive, not an allocator allocation/free census; a mapping is not necessarily one arena. | Restart trial with `MALLOC_ARENA_MAX=2`, described below. Caches and allocator retention can coexist. |
| 3 | Fragment/version churn increases native scan and write work. [Upsert](../src/storage/lancedb/index.ts) performs a merge insert per call. [Curator](../src/offline/curator/index.ts:605) loads effectively visible episodes and applies significance updates individually; [consolidator](../src/offline/consolidator/index.ts:579) loads visible episodes and family members. | Each pass reads O(episodes), including embeddings; consolidation holds candidates across awaited LLM calls. Each write creates a table version and can add fragments. At 4096 dimensions, each final Float32Array owns 16 KiB of buffer payload; 3,000 own 46.875 MiB, plus JS wrappers and the current Arrow batch. The initial materializer also allocated boxed arrays, substantially increasing transient heap use; see the correction below. This is pass-scoped allocation pressure, not a proven ever-growing JS collection. | Query/upsert/returned-row counters and slow table metadata samples make this measurable. Dense float vectors now copy directly into their final hydration buffers. Preserve existing maintenance/cognition behavior. Optimization is working; see the trace evidence below. |
| 4 | Arrow row ownership at the wrapper boundary. SDK `dist/query.js:165` collects all batches, and `toArray()` returns Arrow StructRows. `apache-arrow/row/struct.mjs:24` stores the parent batch. The old `normalizeRows()` only cast each row. | Retaining one raw row pins the batch; nested vector cells also retain Arrow data. Previously, repository hydration usually copied vectors (`src/storage/codecs.ts:47`, `src/memory/episodic/repository.ts:517`), so this is not evidence of 4 GB retained across turns. JS-visible buffers normally affect `external`/`arrayBuffers`. | [Materialization](../src/storage/lancedb/materialize.ts) consumes the async batch iterator to completion and copies rows, nested structures and vectors. Dense float vectors become independently owned Float32Arrays, which hydration reuses by identity. No batches or vector views escape. Mutation tests verify independence from source buffers. There is no public query-result `dispose()` API to call in this SDK. |
| 5 | Demo diagnostic retention. [Live bridge](../demo/server/src/live.ts) stored a full ledger under every turn ID until reset. Its frame buffers limited each session to 64 frames, but never removed inactive sessions globally. | One ledger per turn indefinitely; up to 64 final frames per historical session indefinitely. These are mostly JS text/objects, so they do not fit the dominant native gap, but they are real growth paths. | Keep the latest **64 ledgers**; cap frame-buffer sessions at **128**, prune frames older than 60 seconds on activity/sampling. Evicted historical ledger requests use the existing unavailable/404 behavior. |
| 6 | Native handle cleanup on reset/open failure. The old store `close()` closed only the connection; repositories and `openTablesByName` still held native table handles. Schema/refresh failure paths also lacked cleanup. | Seven runtime handles, not one new handle per turn. Old Borg instances or partial opens can retain them until GC, especially around reset; this does not explain steady growth in a runtime without resets by itself. | Close every registered table before the connection, clear the registry and release the Session reference; release temporary handles on failed schema/checkout. Tests retain JS references and verify the underlying tables are closed. |

The SDK's Session API is documented [here](https://lancedb.github.io/lancedb/js/classes/Session/).
Code anchors for the ranked findings: `src/storage/lancedb/index.ts:467`
(Session construction), `:292` (merge-insert writes), `:344` / `:381`
(query materialization), `:634` (startup handles), `:745` (shutdown);
`src/storage/lancedb/materialize.ts` (`materializeRows`, batch ownership), and
`demo/server/src/live.ts:234` / `:571` (frame and ledger retention).
The 0.27.2 JS implementation does not forward the positional `connect(uri,
options, session)` argument. Use **`connect({ uri, session })`**, as the wrapper
now does. The native integration test checks that queries populate the observed
Session, rather than merely checking a configured but unused object. Cache
capacities are eviction budgets, not a hard process RSS limit: query scratch
space, allocator overhead and non-Lance allocations remain outside them.

The current **256 MiB index / 64 MiB metadata** defaults remain unchanged.
Because the reviewer found no vector indices, lowering the index budget
cannot explain an improvement on the current data. If vector indices are added,
use **512 MiB for the index cache as a benchmark starting point**, then measure
cache occupancy, query latency, RSS and maintenance duration under representative
concurrency before choosing a production budget.

### Review correction: scan vector allocation

The first materializer in `420c3334` detached vectors through `number[]`, and
repository hydration then copied them to Float32Array. The reviewer measured
about **133 MiB of additional JS heap before hydration** for 3,000 × 4,096
vectors, and conversion plus hydration rose from **856 ms to 1,006 ms** in that
probe. The earlier 16 KiB-per-row estimate described only the final float32
payload and understated peak allocation in that implementation.

Fixed-size lists of float32/float64 whose numeric elements contain no nulls
now copy their Arrow numeric view directly into a **fresh Float32Array**. This
does not alias the Arrow buffer, and hydration accepts that same owned array
without a second copy. LanceDB query schemas mark even dense vectors' elements
nullable, so eligibility uses each vector's actual null count. Lists containing
nulls and nested lists retain recursive conversion so nulls and structure survive.
The regression scan covers **3,000 × 4,096** float32 vectors across sliced
batches, verifies 3,000 distinct owned buffers
totalling **49,152,000 bytes (46.875 MiB)**, and verifies hydration identity.
Separate tests cover float64 conversion, nulls, nested lists and actual episodic
repository hydration. This payload count excludes JS wrappers, other row fields,
the live Arrow batch and native scanner/cache/allocator memory; it is not a
peak-RSS claim or a fresh wall-clock benchmark.

### Other long-lived objects examined

- [Storage composition](../src/borg/storage-setup.ts:283) creates **one store /
  connection per Borg runtime**, then seven table handles. Repositories receive
  those handles; normal queries do not reopen tables. Schema validation does an
  open/create, close and reopen **at startup**. Optimization reuses registered
  handles; temporary handles for other tables close in `finally`.
- [Embedding cache](../src/embeddings/cache.ts:5) holds up to 5,000 settled
  `Float32Array`s (78.1 MiB of vector payload at 4096 dimensions). Pending work
  may temporarily exceed that cap, reported by `pending_overflow`, and failed
  entries are deleted. Returned vectors are copies. This is real cross-turn
  retention, but bounded and visible in `arrayBuffers`; the reported 15 MB
  array-buffer total excludes it as the explanation for the 4 GB native gap.
- [Retrieval](../src/retrieval/pipeline.ts:536) creates query-vector memos,
  candidate maps, source-label maps and dedupe sets per search. The batched
  source-disclosure resolver's maps are per `resolve()` call
  (`src/memory/common/source-stream-audience-disclosure.ts:175`). They do not
  accumulate Arrow batches across turns. The semantic-revision cache is capped
  at 1,000 verdicts containing IDs/hashes/statuses, not embeddings.
- [Turn execution metrics](../src/cognition/turn-execution-metrics.ts:22) uses
  weak maps and an async-local record of numbers. Live phase tracking deletes
  entries on `turn.terminal`; active-turn snapshots store phases/IDs, not rows.
  Ingestion's pending/in-flight/settling maps are drained and deleted after
  completion (`src/cognition/ingestion/coordinator.ts:669`). A stalled task can
  retain its current inputs, but no completed-turn native-row memo was found.
- [JSONL tracing](../src/tracing/tracer.ts) writes and fsyncs each event directly;
  it has no accumulating event queue. Context captures await durable appends.
  [Capture maintenance](../src/cognition/deliberation/context-capture-storage.ts:324)
  coalesces compression work per file into one worker plus a boolean pending
  flag, streams gzip, and deletes the worker entry when done. It does not queue
  captured payloads in that map.
- The manual demo `dreamPlans` map (`demo/server/src/app.ts`, `createDemoServerApp`)
  retains plan/apply results until reset and is unbounded by request count.
  Plans can include serialized embeddings. This is a remaining JS-heap growth
  path under repeated manual preview requests, not the automatic maintenance
  scheduler's storage. No evidence ties it to the observed native growth.
- Read-only inspection of the configured external plugin
  `/home/luth/Programming/sol-connector/src/plugin.ts` confirmed it uses the
  provided `getBorg()` handle. Its principal directory is replaced on refresh;
  poller warning/dedupe maps contain thread/room IDs and counters, not Lance
  handles or embeddings. No connector code was changed. This was a targeted
  lifetime review, not a complete audit of every external link.

## Why files/versions reach the thousands

The live config had no `maintenance` overrides. [Defaults](../src/config/index.ts:691)
enable maintenance and optimization, but run **light every four hours** and
**heavy every 24 hours**. Consolidator, semantic extractor and curator are light
processes. [Scheduler](../src/offline/scheduler.ts:345) optimizes only after a
heavy pass (inside its maintenance lock), and can defer while busy. A process
restart preserves successful-cadence watermarks; it does not force compaction.

The read-only SQLite watermark for heavy maintenance was
`1788824253544` (`2026-09-07T23:37:33.544Z`). Light maintenance subsequently
completed at `2026-09-08T20:55:24.137Z`. The active trace had rotated: the
optimization event was in **`traces/turns.jsonl.20260909-000356.gz`**, not the
current `turns.jsonl`. It reported:

| Table | Fragments removed | Fragments added | Versions pruned |
| --- | ---: | ---: | ---: |
| episodes | 1300 | 1 | 1362 |
| semantic_nodes | 212 | 1 | 230 |
| action_records | 132 | 1 | 177 |
| open_questions | 27 | 1 | 33 |
| other three tables | 0 | 0 | 0 |

`storage.optimize.completed`: **errors=0**, total removed=1671, added=4,
versions pruned=1802, duration=3429 ms. Compaction is demonstrably working.
At inspection, `episodes.lance/data` contained 2651 files and `_versions`
contained 1352 entries; the corresponding semantic/action counts were 448/239
and 260/132. **Data-file count is not active fragment count.** Old versions
keep obsolete files reachable until pruning, and each fragment can have multiple
files. The wrapper deliberately preserves a 15-minute version-cleanup grace for
in-flight readers; that grace does not disable fragment compaction. New writes
then accumulate until the next daily heavy pass.

Use `lanceTables.tables[].fragments` for the actual fragment count and
`versions` for retained versions. `version` is the current monotonically
increasing version ID, which need not decrease after pruning. Changing the
heavy interval also changes cognitive maintenance frequency; this patch leaves
those cadences and prompts unchanged.

## Reading `/api/admin/memory`

The server samples immediately after startup and every five minutes thereafter.
Each sample writes one JSON line to stdout/journald and emits the trace event
`process.memory.sample` to the live tracer and `BORG_TRACE` when configured.
The trace uses a synthetic `turnId` of `process_memory_<pid>_<ts>`.

`GET /api/admin/memory` returns `{ config, latest, history }`, with history ordered
oldest to newest. Before sampling, or when disabled, `latest` is null and history
is empty. The default ring holds **288 samples** (24 hours at five minutes), and
all history is process-local. GET returns the existing snapshot, sets
`Cache-Control: no-store`, and performs **no storage reads**. It remains available
during reset. Like the other demo admin routes, expose it only through the
deployment's existing admin access controls.

Illustrative stdout line (all memory sizes in **bytes**, uptime in **seconds**,
timestamps/durations in **milliseconds**):

```json
{"event":"process.memory.sample","ts":1788908400000,"pid":1661258,"rss":4500000000,"heapUsed":147000000,"heapTotal":278000000,"external":20000000,"arrayBuffers":15000000,"uptime":22000,"activeTurns":1,"ledgerCacheEntries":64,"bufferedSessions":2,"clients":1,"lance":{"cache":{"sizeBytes":50000000,"approxItems":800,"indexMaxBytes":268435456,"metadataMaxBytes":67108864},"tables":[{"table":"episodes","openHandles":1,"listQueries":2500,"searchQueries":300,"upserts":900,"rowsRead":120000}],"lastOptimization":null},"lanceTables":{"sampledAt":1788907800000,"durationMs":200,"tables":[{"table":"episodes","fragments":800,"smallFragments":799,"version":2700,"versions":1352,"durationMs":190}]},"tableStatsPending":false}
```

`lance.cache` reports the native Session's approximate cache occupancy and its
configured budgets. Injected/custom connections without an owned Session report
null. Query counters count attempts, and `rowsRead` counts successfully returned
rows; they are cumulative for the current table handles. `openHandles` should
normally be one per table. `lastOptimization` summarizes the last pass in this
runtime; null after restart does not mean historical maintenance never ran.
Use the durable watermark/trace for that history.

Table metadata refreshes independently every **30 minutes**, sequentially over
existing table handles. `stats()` reads fragment/file metadata and `listVersions()`
enumerates version metadata; neither is a full row/vector query, but both have
cost proportional to fragments/versions. They are deliberately excluded from
the fast sample and HTTP request paths. `durationMs` makes that cost visible.
These independent metadata calls are not a transactional snapshot of concurrent writes.
The first sample normally has `lanceTables: null`; subsequent samples include
the last completed refresh with its own timestamp. `tableStatsPending` marks
an outstanding refresh. Only one refresh can run: even a stalled read cannot
stop process sampling or accumulate queued reads. Errors appear only as stable
codes: `storageErrorCode: "LANCEDB_MEMORY_STATS_FAILED"`,
`lanceTables.errorCode: "LANCEDB_TABLE_STATS_FAILED"`, or the affected table's
`errorCode: "LANCEDB_TABLE_STATS_FAILED"` with null counts. Detailed causes,
including native messages and filesystem paths, go to the server's error log;
they are excluded from API history, structured sample lines and sample traces. Reset
discards old-handle metadata; shutdown clears the timer and awaits its sole read.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `DEMO_MEMORY_SAMPLE_INTERVAL_MS` | 300000 | Sampling interval; 0 disables automatic sampling/logging. |
| `DEMO_MEMORY_TABLE_INTERVAL_MS` | 1800000 | Minimum time between metadata attempts, scheduled on sample ticks; 0 disables them. |
| `DEMO_MEMORY_HISTORY_LIMIT` | 288 | Ring capacity, integer 1–2016. |

Native cache capacities are wrapper defaults, overridable with
`LanceDbStoreOptions.indexCacheBytes` / `metadataCacheBytes` for custom library
construction. They are not allocator environment variables. No dependency
upgrade or additional package is required.

For correlation:

1. Compare equal-activity windows using deltas in `listQueries`, `searchQueries`,
   `upserts`, `rowsRead` and elapsed time. Include both turns and four-hour
   maintenance passes; `activeTurns` counts observed turn phases, not background
   maintenance. Use `offline_process.started/completed` traces for the latter.
2. If native cache bytes rise with RSS, cache occupancy contributes. If cache,
   heap and external bytes plateau but RSS keeps rising, investigate allocator
   retention or other native allocations. These counters cannot distinguish
   allocated native bytes from free pages inside an arena.
3. Compare fragment counts before/after `storage.optimize.completed`, its
   `errors`, and the metadata refresh timestamp. A cached count is not a fresh
   post-compaction measurement. Check rotated traces as well as the active file.
4. Capture at least one daily heavy cycle under comparable activity before
   claiming the daily OOM is resolved. An explicit GC/heap snapshot is neither
   required by this instrumentation nor an explanation of untracked native RSS.

## Allocator recommendation (operator applies at restart)

Start with **`MALLOC_ARENA_MAX=2`** for the Node/tsx process, alongside the bounded
Lance Session. This limits glibc arenas and should reduce fragmented resident
free space in this workload. The 12 Tokio and four libuv workers make multiple
arenas plausible; sharing two can increase allocator-lock contention, so compare
turn latency, maintenance duration, CPU and RSS per amount of work. It is a
mitigation to test, not a memory cap and not a cure for retained native objects.
Set it in the operator-managed service environment **before Node starts**.
No systemd files or running-process settings were changed here. See the
[glibc allocator parameters](https://sourceware.org/glibc/manual/latest/html_node/Malloc-Tunable-Parameters.html).

If the new cache occupancy is stable but RSS remains high after bursts, a
separate trial can add **`MALLOC_TRIM_THRESHOLD_=131072`**. It fixes the trimming
threshold at 128 KiB rather than allowing dynamic adjustment, trading more
release/reallocation work for potentially lower residency. It only helps
releasable free space and does not evict live Lance caches. `M_TRIM_THRESHOLD`
is a `mallopt` constant, not an environment variable; there is no `M_TRIM` switch
to put in the service environment. Do not add per-turn FFI calls or periodic
`malloc_trim()` to this TypeScript service to conceal unexplained growth.

**jemalloc via `LD_PRELOAD` is a second controlled trial**, particularly if two
glibc arenas hurt concurrent query performance. Use the host's actual compatible
`libjemalloc.so.2` path; no library was installed as part of this work. Verify it
is loaded in `/proc/<node-pid>/maps`, then compare the same workloads. A possible
trial configuration is `MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000`
to purge unused pages without depending on continued foreground allocation.
Faster decay spends more CPU; test native-addon startup and normal reads/writes
in a disposable data directory first. The upstream
[integration guide](https://github.com/jemalloc/jemalloc/wiki/getting-started) and
[allocator manual](https://jemalloc.net/jemalloc.3.html) describe preloading and
decay. Preloading affects dynamically resolved allocation, not any allocator
statically linked into an addon. Once jemalloc owns malloc, glibc tuning variables
are not the controls for those allocations. jemalloc also cannot free live cached
objects; preserve the Lance cache bounds whichever allocator is used.

Record each restart's settings with the sampler history. Separate allocator
trials where practical so a lower RSS is attributable, and avoid treating a
lower idle RSS as proof that native allocations no longer grow.

## Validation

Tests exercise actual Lance reads/writes, compaction and version pruning,
native Session wiring, closing live table references, detached Arrow rows and
nested lists, the 3,000-vector scan and hydration ownership, bounded demo buffers,
sampler timing/history/reset behavior, diagnostic error sanitization and logging,
and the HTTP/startup integration. Anthropic and embedding calls are mocked;
Lance fixtures use disposable directories. Live data was never opened through
LanceDB or a write-capable SQLite connection.

Review follow-up checks on Node **v25.3.0**:

```sh
choom -n 800 -- pnpm typecheck
choom -n 800 -- pnpm heuristics:guard
choom -n 800 -- pnpm exec tsc --noEmit -p demo/server/tsconfig.dev.json
choom -n 800 -- pnpm vitest run \
  src/storage/lancedb/index.test.ts \
  src/storage/lancedb/materialize.test.ts \
  src/storage/codecs.test.ts \
  src/memory/episodic/repository.test.ts \
  demo/server/src/memory.test.ts \
  demo/server/src/live.test.ts \
  demo/server/src/__tests__/server.test.ts \
  --maxWorkers=2 --cache=false --configLoader=runner
git diff --check
```

All passed; Vitest reported **7 files, 157 tests passed**. Dependencies were
symlinked from the existing installation as instructed; nothing was installed.
