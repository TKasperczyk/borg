The Borg observatory producer is implemented and disabled unless `OBSERVATORY_BASE_URL` is set.
This work was done on local `dev` using file reads, searches, and edits only. No npm, Node,
TypeScript, Vitest, tsx, formatter, build, or other executable verification was run. Tests are
written but unverified. No commit was made, and `node_modules`, the untracked pnpm files, and
`.memory-audit/` were not changed. The pre-existing untracked `docs/observatory-events.md` was
read and preserved.

**Current review fixes**

The reviewer confirmed the shutdown deadline and receipt entry-ID fixes. This pass changes
the three regressions in degradation notification and trace filtering:

| Finding | Exact change | Regression tests added or extended (written, not run) |
| --- | --- | --- |
| 1. Awaited degradation observer blocked fallback. | `pipeline.ts` calls the observer synchronously so lane-error bookkeeping is immediate, then detaches a returned promise with a guarded rejection handler. Synchronous exceptions, async rejections and tracer failures are logged without content and cannot replace the fallback result. | `pipeline.test.ts`: “preserves open-question fallback and rejected lanes with a %s degradation observer” now covers throwing, permanently unresolved and rejecting observers. The unresolved promise is never resolved by the test; fallback must still complete. All cases retain rejected lane/error metadata and unaffected episodic results. |
| 2. Recursive filtering did unbounded work before queue admission. | Removed the standalone recursive copier. The trace sink now calls `ObservatoryEmitter.emitLazy`; closing/capacity checks run before its factory reads trace data. Redaction runs inside the existing clamp traversal, before reading gated values. The traversal stops at depth 6 and uses a global 200-visit budget for object keys and array elements, with three slots reserved for metadata. Strings/keys are sliced before serialization, byte scans or policy lookup. Large correlation IDs skip lookup. Partial data is marked `truncated`; `truncated_lists` survives byte-size summaries. | `observatory/tracer.test.ts`: a full-queue Proxy asserts zero data reads/enumeration; oversized/deep/wide input uses unread getters to check redaction, depth and list limits, plus at most 200 field reads; nested array fanout verifies the global budget and retained list metadata in a 24-KiB summary. Mixed-composition tests now inspect serialized emitter output and retain sibling payloads. |
| 3. Shared `entityTerms` gating changed legacy sinks. | Restored the original contents of `PAYLOAD_GATED_TRACE_KEYS`, exposing it as a read-only set for reuse. The extra `entityTerms` rule lives only in the observatory sink's field predicate. Existing `CallbackTracer`, ring and JSONL filtering behavior is unchanged. | `memory-trace.test.ts` and `tracing/tracer.test.ts` each assert top-level `entityTerms` and nested options remain in payload-disabled legacy output while the existing top-level `query` gate still applies. The observatory mixed-composition test checks top-level and nested `entityTerms` are omitted there. |

The amended `docs/observatory-events.md` was read and left untouched. Section 6's snapshot
bounds, `updated_at`, waiter counts, `recall.lanes_detail`, and reducer enums remain consumer
responsibilities; the section 4.2 query-planner field names need no producer change. Source
inspection confirms `startLaneTrace` emits `intent_kind`, `intent_source`, `candidate_count`,
`duration_ms` and `outcome`, and the sink retains all five in `recall.lane.data` (duration also
appears on the envelope). The new “preserves lane detail fields on the wire for %s” test
asserts the serialized fields for both fulfilled and rejected outcomes.

**Earlier findings, current implementation**

| Finding | Exact change | Regression test added (written, not run) |
| --- | --- | --- |
| 1. Observatory transport extended shutdown beyond the drain deadline. | `memory-sidecar-main.ts` now supplies `shutdownObservatory` to `drainMemorySidecar`. The drain passes its absolute deadline and an abort signal, cancelled before the deadline race returns. `ObservatoryEmitter.shutdown({ deadlineAt, signal })` aborts its active transport and drops the queue at that cutoff; transport races cancellation even if fetch ignores its signal, and cannot retry or start a final batch after expiry. | `src/sidecar/observatory/emitter.test.ts`: “aborts a stalled batch within the shared drain deadline after %s ms of HTTP drain” covers a 100-ms drain with HTTP finishing at 60 ms or exhausting the deadline. It asserts completion and transport abort at 100 ms, no retry/final send, and no remaining timers. “bounds standalone shutdown by its absolute deadline even when fetch ignores abort” also exercises the emitter's own deadline timer. |
| 2. Nested payload-gated fields leaked through mixed tracer composition. | Observatory-only redaction uses the shared legacy key set plus a local `entityTerms` rule during the emitter's bounded clamp pass, after capacity checks. It preserves the conditional `spans` rule and leaves sibling inputs intact. | `src/sidecar/observatory/tracer.test.ts`: “removes nested retrieval payloads in mixed composition without changing sibling data” checks serialized `retrieval.started` data, nested prompts/entity terms, preserved counts, unchanged sibling/input data, and payload-enabled retention. |
| 3. Coalesced enqueue receipts used the first pending entry's ID. | `onPendingSession` now forwards `entryId` to `schedule(sessionId, "append", "user", entryId)`, matching `onAppend`. The first pending timestamp still controls maximum wait age. | `src/cognition/ingestion/chat-response-catch-up-worker.test.ts`: “attributes consecutive receipt settle events to the entry that resets the timer” sends two receipts with distinct IDs and checks each settle observer notification names its own ID while both retain the first pending timestamp. |
| 4. Open-question search failures appeared as fulfilled lanes. | The synchronous degradation notification records lane failure and detaches any async observer. Each collector emits `outcome: "rejected"` with the original `error_name` while retaining fallback candidate count/results. `open-questions.ts` and its fallback scoring remain unchanged; the sink maps rejected lanes to `recall.lane` status `error`. | `src/retrieval/pipeline.test.ts`: the parameterized open-question fallback test injects a vector-search `TypeError` and throwing, unresolved or rejecting observers. It checks rejected lanes, relation-backed fallback results, unaffected episodes and content-free logs. |

All four findings are wired. Compiler acceptance, fake-timer scheduling, actual fetch cancellation,
and regression-test results remain unverified because executing code is prohibited for this task.

Opus 5.0: in-scope because this exposes harness operations without changing language
interpretation, recall eligibility, model judgments, or durable memory formats.

**Exploration and hook placement**

The wire contract, observatory design, AGENTS.md, CLAUDE.md, WORKFLOW.md, Teams inbox contract,
and memory context contract informed the implementation. The relevant existing paths were:

| Finding | Implementation and reason |
| --- | --- |
| `scripts/memory-sidecar-main.ts` owns the tenant pool, trace registry, waiters and maintenance coordinator. | It owns one emitter and the observatory adapters. Pool tracers compose the existing ring tracer with the new sink; HTTP handlers receive both explicitly. Observatory shutdown follows HTTP and pool draining within their shared absolute deadline and aborts transport at expiry. |
| `MemoryTraceRegistry` uses `CallbackTracer` but retains only selected events. | The second sink independently receives every trace event. Its payload setting comes from the existing registry's `includePayloads`; enabling the observatory alone keeps payloads disabled. |
| `CallbackTracer` copies top-level fields and strips legacy payload keys; mixed composition can also supply nested payloads. | The observatory implements `TurnTracer` directly and defers trace reads to the emitter's capacity-checked factory, avoiding a pre-admission `CallbackTracer` copy. Its field predicate reuses the shared key set with local `entityTerms` gating during bounded clamping. `data.trace_turn_id` retains the Borg trace identifier. |
| `memory-handler.ts` has one router and multiple early degradation/error responses. | A request-local response observer ends `recall.request` from the actual response projection, including the existing deadline, embedding and activity degradation reasons. Its start timestamp includes parsing time; no event is emitted before a tenant is known. |
| `raceRecallDeadline` races the caller's budget while abandoned work settles independently. | Optional stage metadata wraps identity, context sections, episodes and autobiographical races without cancelling or awaiting abandoned work. Budgets describe the remaining phase allowance. |
| Recall trace IDs were generated only when the local trace ring was enabled. | They are now also generated when the observatory sink exists. Context and legacy recall register correlations before retrieval. Guard calls use a fresh trace ID per request, avoiding collisions when a served context is reused. Append traces register before writing. |
| Enqueue commits under the tenant's exclusive chain; duplicate receipts return an existing entry. | `inbox.enqueued` is emitted only for a new durable enqueue, with transport actor/conversation fields and source references. Duplicate receipts do not create a new pending dashboard message. |
| The worker's delay decision combines quiet time, max age, immediate continuation and backoff. | `delayFor` returns the winning bound with its existing delay. The guarded observer emits `inbox.settle` after installing that timer. Receipt/append source IDs are carried mechanically in existing lane state. |
| `drain` hands a concrete prefix to the runner, then validates durable watermark coverage. | `inbox.batch` starts before that handoff and ends with the resulting `DrainResult`, including busy/error outcomes and the original error name. Its watermark is the prior durable response watermark. |
| The runner validates metadata and staleness before HTTP and selects a terminal from the response. | It mints a fresh `turn_` ID for every `run`, forwards it in `X-Observatory-Turn-Id`, and reports fetch/response-validation timing and terminal choice. The call span ends before terminal persistence; persistence failures are visible in the batch result. Skipped calls describe stale or metadata sealing. |
| `onTerminalCommitted` is guarded and also called by reconciliation. | A second optional callback argument carries the known seal decision from the actual commit path. The observatory adapter deduplicates repeated notifications and joins terminals to attempts by source entry IDs, without persisting observatory IDs. |
| Waiters have immediate remembered-generation, registration, terminal, timer, cancellation and shutdown paths. | Their guarded observer reports registration, generating, resolution, timeout and cancellation. Generating is observable even with no current waiter; shutdown is cancellation. Existing leases and acknowledgement behavior remain intact. |
| Episodic intents settle independently; semantic/open-question/image collectors catch some failures internally. | Timings surround each episodic intent and each actual context intent collector. Open-question degradation callbacks carry caught search failures into lane settlement without changing fallback retrieval. The shared commitment collector reports its aggregate result, including rejected embedding work. The new trace event is in both the closed union and taxonomy. |
| Maintenance log writers run only at completion; their serialized projection omits state and timestamps. | A guarded typed `observeReport` sink receives accepted starts and the complete final report, including error and forced-abort paths. The root projects it into `job.maintenance`; existing `writeReport`/`writeError` output remains intact. This avoids reconstructing incomplete reports from log strings or double-emitting error completions. |
| `scheduleIngestion` already isolates append-turn ingestion from the response and returns `ran`, `processedEntries`, and optional `error`. | It supplies `job.ingestion` start/end with both counters, including returned errors as well as thrown failures. |

**Utilities searched and reused**

| Search / inspection before adding helpers | Result |
| --- | --- |
| `rg` for `truncate`, `preview`, `clamp`, ID functions, `newId`, and `randomUUID` under `src/util` and `src/sidecar`; read `src/util/ids.ts`. | Reused `createIdHelpers` for event, span and runner turn IDs. No new random-ID implementation or dependency. |
| Read `src/util/clock.ts` and the existing injected-clock/timer patterns in the worker, trace registry and waiters. | Reused `Clock`, `SystemClock`, and `ManualClock` in tests. The emitter accepts injected clock, fetch and timer functions; existing handler deadline timekeeping stays intact. |
| Read `src/util/errors.ts`, especially `describeError`, and the guarded observer call sites. | Reused `describeError` for wire stage errors. New observer logs contain an object with an error name/hook or cumulative drop count, never event content, response bodies or tokens. |
| `rg` for truncation/clamp/preview function declarations across `src`; read `text-excerpt.ts`, `utf16-boundary.ts`, and `math.ts`. | Reused `utf16SafePrefix` for both preview and ordinary string cuts, and `clampPositiveIntegerOrFallback` for configurable trace-map bounds. The existing head/tail excerpt and specialized prompt/error truncators do not implement the contract's prefix plus `…[+N chars]` suffix. |
| `rg` for `class.*(Ttl\|TTL\|Lru)`, `expiresAt`, and `maxEntries` in `src/util`; inspect the trace, served-context and waiter registries. | No shared TTL-map utility matched. Small local maps follow the existing registry pattern; no general cache framework was introduced. |
| `rg` for `stripPayload`, `PAYLOAD_GATED`, recursive filter/omit and deep filter/omit helpers across `src/`; follow-up searches for `maxDepth`, `maxKeys`, redaction and clamp helpers in `src/util`, `src/tracing` and `src/sidecar/observatory`. | Reused the original trace key set and the existing observatory clamp. Removed the standalone recursive copier; no suitable shared bounded redaction utility existed. The observatory adds only its local `entityTerms` condition. |
| Re-read `utf16-boundary.ts`, JSON utilities, open-question callbacks and existing detached `.catch` observer patterns. | `utf16SafePrefix` slices without scanning whole strings, inspecting only the boundary code units. Reused it before serialization/byte scans. The JSON validator's whole-object traversal was unsuitable for bounded telemetry. Detached observer rejection handling follows existing fire-and-forget patterns. |

The new contract-specific recursive clamp is necessary to combine prefix truncation, 280-character
previews, 50-item lists with `data.truncated_lists`, and the 24-KiB serialized-data summary.
It uses UTF-8 byte counts and preserves UTF-16 boundaries. Depth-limited branches become
`"[truncated]"`; exhausted traversal budgets stop reading further fields/elements and mark the
data truncated. List truncation metadata is retained when data becomes a byte-size summary,
subject to the same list/string/data limits. Cyclic or non-JSON telemetry is dropped.
The small guarded emit/start/end
helpers centralize failure isolation for injected observers; their equivalents previously
existed only as individual try/catch blocks at call sites. The lane timer is private to retrieval.

**Decisions and limits**

- Transport batches honor both the configured event count (at most 500) and the 1-MiB body cap.
  The queue drops new events on overflow. A full batch advances the timer to the next timer
  turn; HTTP never runs synchronously from `emit`. There is at most one in-flight batch and one
  retry on network failure or 5xx, with the same event IDs. No 4xx is retried. Redirects are not
  followed. Enabled configurations require a token and an HTTP(S) base URL without embedded
  credentials, a query or a fragment; invalid settings fail startup like the inbox config.
- Trace admission is lazy only until the synchronous queue-capacity check; accepted events
  are still snapshotted immediately, using one bounded redaction/clamp traversal. It permits
  depth 6 from each input root, at most 50 items per list, and a global 200-visit budget across
  object keys and array elements (three slots reserved for metadata). Only bounded copies
  reach JSON serialization and byte scans. Truncation may therefore reduce large telemetry
  even below the 24-KiB summary threshold; sibling tracers retain their original input.
- Shutdown finishes any in-flight batch and sends at most one final bounded batch without retry
  only while the shared absolute drain deadline allows it. Expiry aborts active transport,
  suppresses further attempts, and drops the remaining queue. The drain's abort signal fires
  before its deadline race returns; the emitter also accepts a standalone absolute cutoff.
  This preserves "flush once" without adding observatory timeouts after the process drain budget.
- Trace correlations are tenant-scoped, capped at 10,000 and expire after one hour by default.
  Inbox attempt/source correlations and terminal dedup share a capped 10,000-entry, one-hour
  map. Runner attempt counters are capped at 5,000 per runner with the same TTL. Pruning occurs
  on access; evictions/restarts may lose optional correlation or restart attempt numbering.
- A batch start precedes `run`, where the turn ID must be minted. Its start carries source IDs;
  its paired end carries the minted turn ID. The shared span ID links the pair. The runner and
  terminal events carry that turn ID directly when the correlation is retained.
- Reconciled historical terminals have no durable seal bit. Their `sealed` field is omitted,
  rather than inferred from reason text. Live commit notifications always supply the bit.
  A `has_more` continuation can have only a session reference until its next source prefix is
  known; no extra stream scan was added solely to fill an observatory reference.
- Empty scans and repair-only drains do not hand a batch to the runner, so they do not create
  `inbox.batch` spans. Context collectors intentionally skipped by episodes-only projection
  likewise emit no fictitious lane. Autobiographical work skipped before a deadline race remains
  visible in the request's existing degradation reason.
- `collector` is an additive lane data field distinguishing episodic, semantic, open-question,
  image and commitment collection. Actual intent IDs/kinds/sources are retained; shared
  commitment collection uses the structural identity `context:commitments` / `commitment` /
  `context`. Counts are collector outputs before final fusion; semantic counts include matched
  nodes and relation hits. No new semantic interpretation is performed.
- Successful append-turns use the new `sidecar.append_turn.completed` tracer event, forwarded
  as `trace.sidecar.append_turn.completed`; no additional non-catalogue Borg stage kind was
  introduced. Existing append-projection degradation traces are also correlated.
- `job.ingestion` covers the requested sidecar scheduling boundary. Terminal-triggered ingestion
  inside `BacklogTerminalService` is not given another job hook. Existing background/offline
  traces still pass through the second sink. Task-event runners retain their existing trace
  events and are outside the Teams chat-runner correlation change.

**Files changed**

| File | Change |
| --- | --- |
| `src/sidecar/observatory/config.ts` | Disabled/enabled environment configuration and validation. |
| `src/sidecar/observatory/config.test.ts` | Defaults, overrides, disablement and invalid-setting cases. |
| `src/sidecar/observatory/emitter.ts` | Capacity-first lazy admission, one bounded redaction/clamp pass, transport, stages and deadline-aborted shutdown. |
| `src/sidecar/observatory/emitter.test.ts` | Transport/clamp coverage plus stalled shutdown under shared and standalone absolute deadlines. |
| `src/sidecar/observatory/tracer.ts` | Lazy trace factory, local field policy, lane translation and tenant-scoped TTL correlation. |
| `src/sidecar/observatory/tracer.test.ts` | Serialized catalogue/correlation/gating, queue admission, bounded traversal, metadata and lane-detail coverage. |
| `src/sidecar/observatory/observers.ts` | Inbox and maintenance adapters with terminal dedup/correlation. |
| `src/sidecar/observatory/observers.test.ts` | Batch/terminal joining and maintenance start/abort projection. |
| `scripts/memory-sidecar-main.ts` | Producer composition, observers and tracers; flush runs inside the shared drain deadline. |
| `src/borg/types.ts` | Optional inbox observer and terminal seal metadata types. |
| `src/borg/open.ts` | Passes the inbox observer and known pending receipt source IDs to the worker. |
| `src/borg/__tests__/enqueue-message.test.ts` | Updates the receipt notification assertion to require its source entry ID. |
| `src/cognition/ingestion/index.ts` | Exports the worker observer type. |
| `src/cognition/ingestion/chat-response-catch-up-worker.ts` | Guarded settle/batch notifications, attempt handoff and current receipt entry correlation. |
| `src/cognition/ingestion/chat-response-catch-up-worker.test.ts` | Settle/backoff coverage plus distinct entry IDs for consecutive coalesced receipts. |
| `src/cognition/ingestion/backlog-terminal.ts` | Reports the actual seal decision through the existing guarded commit observer. |
| `src/cognition/ingestion/backlog-terminal.test.ts` | Checks live seal metadata even when post-commit advancement fails. |
| `src/sidecar/team-agent-turn-runner.ts` | Turn header, bounded attempt counters, call timing and terminal-choice reporting. |
| `src/sidecar/team-agent-turn-runner.test.ts` | Fresh IDs across retries, HTTP outcomes and throwing observers. |
| `src/sidecar/response-waiter-registry.ts` | Guarded registration/generating/resolution/timeout/cancellation events. |
| `src/sidecar/response-waiter-registry.test.ts` | Tenant-separated lifecycle events, resolution and lease cleanup under observer failure. |
| `src/sidecar/memory-handler.ts` | Enqueue, recall request/phases, guard, append trace correlation and ingestion job events. |
| `src/sidecar/memory-handler.test.ts` | Header propagation, correlation without the trace ring, response parity and failed hooks. |
| `src/sidecar/memory-maintenance.ts` | Typed second report observer for accepted starts and final reports. |
| `src/sidecar/memory-maintenance.test.ts` | Start/forced-abort notifications remain harmless when the observer throws. |
| `src/sidecar/memory-sidecar-shutdown.ts` | Passes the shared absolute deadline and cancellation signal to observatory shutdown. |
| `src/sidecar/memory-trace.ts` | Exposes the existing immutable payload setting for sink composition. |
| `src/sidecar/memory-trace.test.ts` | Legacy payload-disabled ring output retains top-level entity terms and nested options. |
| `src/retrieval/pipeline.ts` | Collector timing/outcomes and synchronous failure bookkeeping with detached async observers. |
| `src/retrieval/pipeline.test.ts` | Lane/fallback assertions with throwing, unresolved and rejecting degradation observers. |
| `src/tracing/tracer.ts` | Closed event union additions and read-only export of the unchanged legacy payload key set. |
| `src/tracing/tracer.test.ts` | Legacy payload-disabled JSONL output retains top-level entity terms and nested options. |
| `src/tracing/taxonomy.ts` | Registers both new trace names in the existing taxonomy. |
| `docs/memory-context.md` | Links the event contract and documents all six environment variables. |
| `docs/observatory-implementation.md` | Exploration, utility searches, implementation decisions and validation limits. |

**First CI run**

Only source and diff inspection was performed. There is no compiler or test result to claim.
The first `npm run typecheck` should pay particular attention to optional observer callback
types, the optional observatory shutdown result, lazy emission/field-filter signatures, the closed trace
catalogue, and Node fetch/timer mock signatures in `tsconfig.test.json`.
The main sidecar script is not included by that command's current tsconfig include lists, so
passing it alone would not verify the production composition root. Runtime checks still need
to exercise unresolved/rejected observers, bounded trace traversal, legacy sink compatibility,
fake timer/unref behavior, deadline races, queue shutdown and the consumer's
joining of initially uncorrelated batch starts. Formatting has been edited manually and has
not been checked with a formatter. No new dependency, migration or persisted observatory
state was added.

**Current diff against HEAD**

This is the literal `git diff HEAD --stat` after the current three regression fixes. It includes the earlier
observatory changes, not just this review pass. Git does not include the still-untracked
`src/sidecar/observatory/` files or this report in this stat; nothing was staged.

```text
 docs/memory-context.md                             |   8 +
 scripts/memory-sidecar-main.ts                     |  49 ++++-
 src/borg/__tests__/enqueue-message.test.ts         |   4 +-
 src/borg/open.ts                                   |   7 +-
 src/borg/types.ts                                  |   4 +-
 src/cognition/ingestion/backlog-terminal.test.ts   |   1 +
 src/cognition/ingestion/backlog-terminal.ts        |   4 +-
 .../chat-response-catch-up-worker.test.ts          |  56 +++++
 .../ingestion/chat-response-catch-up-worker.ts     | 127 +++++++++--
 src/cognition/ingestion/index.ts                   |   1 +
 src/retrieval/pipeline.test.ts                     |  79 +++++++
 src/retrieval/pipeline.ts                          | 242 ++++++++++++++-------
 src/sidecar/memory-handler.test.ts                 | 101 +++++++++
 src/sidecar/memory-handler.ts                      | 190 ++++++++++++++--
 src/sidecar/memory-maintenance.test.ts             |  20 ++
 src/sidecar/memory-maintenance.ts                  |  21 ++
 src/sidecar/memory-sidecar-shutdown.ts             |  14 ++
 src/sidecar/memory-trace.test.ts                   |  13 ++
 src/sidecar/memory-trace.ts                        |   2 +-
 src/sidecar/response-waiter-registry.test.ts       |  45 +++-
 src/sidecar/response-waiter-registry.ts            |  34 ++-
 src/sidecar/team-agent-turn-runner.test.ts         |  41 +++-
 src/sidecar/team-agent-turn-runner.ts              | 174 +++++++++++----
 src/tracing/taxonomy.ts                            |   2 +
 src/tracing/tracer.test.ts                         |  14 ++
 src/tracing/tracer.ts                              |   4 +-
 26 files changed, 1080 insertions(+), 177 deletions(-)
```
