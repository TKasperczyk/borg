# Sprint 4 focused re-review fixes

## Narrow re-review: exact event ownership and actual rendered generations

Worktree verification again returned `/home/luth/tmp/borg-wt-batch9d` from both
`pwd` and `git rev-parse --show-toplevel` before edits. No live tree/store or
`.borg-data` directory was accessed. All tests below run one file and one worker
at a time; the single final typecheck uses the requested 5,120 MB heap.

### Searches before helpers

`rg` searches for `semantic_edge_invalidation_events`, `processed_at`,
`invalidation`, `restoreEdge`, `reversal`, and event-ID ownership found the
existing trigger-written outbox, audited restoration transaction, and reviser's
event-existence recheck. These are reused; no new graph helper was added.

Scheduling searches for `GoalPresentation`, `goalPresentations`,
`onPromptPresented`, `shownOpenQuestionIds`, `TurnResult`,
`snapshotTurnExecutionMetrics`, and `applyPerGoalEmptyWakeBackoff` found the real
self-context snapshots and the existing AsyncLocal turn-metrics transport.
The new `recordShownGoalPresentations` helper only copies/deduplicates known goal
snapshots into that transport. Both existing prompt callbacks call it; there is
no new snapshot source, semantic interpretation, or scheduler-side inference.
Tests reuse the real `TurnSelfContextBuilder`, existing `openTestBorg` fixture,
renderer test contexts, and scheduler construction patterns.

Migration searches covered `goal_last_interaction_at`, numeric local/composed ID
forms, ordered migration lists, and documentation pins. Only the registration,
Sol-ledger pending-ID pin, and documentation encoded this migration's old ID.
Name-based tests and historical ledger fixtures remain unchanged.

### Finding 1: retirement owns an exact invalidation event

`src/memory/semantic/contradiction-retirement.ts` records each edge's exact new
outbox event ID in `reversal.invalidation_events`. Inside the existing transaction,
the operation reads the prior outbox maximum ID, closes the edge through the
existing path, and requires exactly one newly inserted event for that edge.
This avoids relying on either timestamps or connection-wide `last_insert_rowid`
after the closure's other audit writes.

The registered reverser passes the recorded ID to `restoreEdge` in
`src/memory/semantic/repository.ts`. Restoration deletes only that pending ID,
also scoped to its edge, in the same transaction as restoring the edge and audit
finalization. Other same-timestamp closures and processed history survive.
Dependency preservation and the stale-edge-snapshot fence remain in force.
Legacy retirement metadata without event IDs is rejected before writes instead
of guessing ownership. No migration or backfill of ambiguous old audit payloads
was introduced.

`contradiction-retirement.test.ts` adds the reproduced sequence: ordinary
invalidate, ordinary restore, audited retirement, audited reversal, all at one
clock tick. It asserts two identical timestamp tuples, distinct IDs, the second
ID in the audit, exactly the first event remaining after reversal, and repeated
reversal idempotency. Another regression verifies legacy metadata rejection is
non-mutating. Existing rollback, dependency, and belief-reviser regressions remain.

Opus 5.0: in scope; undo must identify and cancel only its own structural work.

### Finding 2: completion uses what the model actually saw

`src/cognition/deliberation/deliberator.ts` records the actual frozen snapshots
selected by each planner/finalizer renderer's shown-goal IDs. The existing
`src/cognition/turn-execution-metrics.ts` transport deduplicates per goal and
copies snapshots. `src/cognition/turn-orchestrator.ts` exposes them on `TurnResult`.
`src/autonomy/scheduler.ts` removes the pre-run capture and uses the returned
per-goal generations for completion backoff. Real turns report an empty array
when no goals were presented; an unpresented goal receives no backoff. Legacy
custom runners omitting the field retain timestamp-only accounting without an
invented generation. Source ownership, headway attribution, and budget/brake
rules are unchanged.

`scheduler.test.ts` replaces the early `buildTurn()` capture regression with
before/after timing around the real self-context builder, for single-goal and
two-goal wakes. Input before capture is generation 2 in both presentation and
backoff, with normal dormancy. Input after capture remains eligible immediately;
the next wake can then apply backoff to generation 2.

`turn-orchestrator.test.ts` also exercises both orders through a real turn,
renderer, and scheduler completion, asserting the returned snapshots, actual
rendered goal, stored generation, next scan, next due time, and dormancy.
`deliberator.test.ts` checks rendered membership and frozen generations for both
planner/finalizer paths and compact/legacy surfaces, including failure before
presentation. `turn-execution-metrics.test.ts` checks turn isolation, deduplication,
and immutable copies. `executive-focus-due.test.ts` supplies the actual shown
snapshots to its simulated runner so shared-backoff ordering remains covered.

Opus 5.0: in scope; scheduling state must describe the input actually evaluated.

### Integration migration renumber

`src/memory/self/migrations.ts` registers `goal_last_interaction_at` as self ID 15;
the name and migration body are unchanged. `src/storage/sqlite/migrations.test.ts`
uses composed ID `1_000_015` in the sorted pending list. Ten pending migrations,
83 total ledger rows, historical timestamps, and no-op repeat-run assertions are
unchanged. Sprint 5's `autobiographical_narrative_coverage` definition is not
copied into this branch. No configuration keys or defaults changed this round.

Opus 5.0: in scope; independently integrated migrations must not collide.

### Narrow re-review validation

All 11 focused files passed (454 tests), one file/worker at a time.
The single requested typecheck exited 2 with three diagnostics in the new
integration-test fixture: the scheduler exposes only `Pick<GoalsRepository, "get">`,
while the fixture needs the full repository. The fixture now obtains it through
the existing typed `BorgDependencies` access pattern. The orchestration file
passed all 105 tests again after that correction.

The typecheck was **not rerun**, honoring the one-run limit; a clean full
typecheck after the fixture correction is therefore not verified. The assessor
configuration includes source tests, so the command stopped at that configuration
(the earlier progress update describing the fifth configuration was incorrect).
Targeted Prettier checks and `git diff --check` passed. No full suite or parallel
heavy command was run.

#### Single typecheck (verbatim)

Command: `NODE_OPTIONS=--max-old-space-size=5120 pnpm typecheck`

Exit code: `2`.

```text

> borg@0.1.0 typecheck /home/luth/tmp/borg-wt-batch9d
> tsc --noEmit && tsc --noEmit -p tsconfig.eval.json && tsc --noEmit -p tsconfig.assessor.json && tsc --noEmit -p tsconfig.simulator.json && tsc --noEmit -p tsconfig.test.json

src/cognition/turn-orchestrator.test.ts(4646,27): error TS2339: Property 'presentation' does not exist on type 'Pick<GoalsRepository, "get">'.
src/cognition/turn-orchestrator.test.ts(4647,26): error TS2339: Property 'add' does not exist on type 'Pick<GoalsRepository, "get">'.
src/cognition/turn-orchestrator.test.ts(4672,9): error TS2740: Type 'Pick<GoalsRepository, "get">' is missing the following properties from type 'GoalsRepository': presentation, clock, reconcilingBlocks, options, and 20 more.
 ELIFECYCLE  Command failed with exit code 2.
```

#### Focused runs before typecheck (verbatim)

Command: `pnpm exec vitest run --maxWorkers=1 src/autonomy/scheduler.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:785580) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  124 passed (124)
   Start at  00:01:56
   Duration  10.65s (transform 2.26s, setup 0ms, import 3.12s, tests 7.39s, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/memory/semantic/contradiction-retirement.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:785892) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  16 passed (16)
   Start at  00:02:12
   Duration  3.98s (transform 1.84s, setup 0ms, import 2.66s, tests 1.20s, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/offline/belief-reviser/index.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:786489) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  40 passed (40)
   Start at  00:02:55
   Duration  8.38s (transform 2.01s, setup 0ms, import 2.82s, tests 5.43s, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/cognition/turn-execution-metrics.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d


 Test Files  1 passed (1)
      Tests  3 passed (3)
   Start at  00:03:13
   Duration  212ms (transform 53ms, setup 0ms, import 71ms, tests 7ms, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/cognition/deliberation/deliberator.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:787004) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  74 passed (74)
   Start at  00:03:17
   Duration  3.95s (transform 2.57s, setup 0ms, import 3.55s, tests 269ms, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/cognition/turn-orchestrator.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:787282) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}

 Test Files  1 passed (1)
      Tests  105 passed (105)
   Start at  00:03:33
   Duration  16.33s (transform 3.53s, setup 0ms, import 4.79s, tests 11.41s, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/memory/self/migrations.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:787638) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  5 passed (5)
   Start at  00:03:54
   Duration  1.29s (transform 686ms, setup 0ms, import 1.10s, tests 74ms, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/storage/sqlite/migrations.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:787949) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  8 passed (8)
   Start at  00:04:10
   Duration  2.89s (transform 1.81s, setup 0ms, import 2.62s, tests 155ms, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/memory/semantic/repository.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:788207) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  32 passed (32)
   Start at  00:04:26
   Duration  1.33s (transform 197ms, setup 0ms, import 426ms, tests 781ms, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 scripts/contradiction-edge-repair.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:788536) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  6 passed (6)
   Start at  00:04:41
   Duration  6.19s (transform 1.78s, setup 0ms, import 2.60s, tests 3.46s, environment 0ms)

```

Command: `pnpm exec vitest run --maxWorkers=1 src/autonomy/triggers/executive-focus-due.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:788736) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  41 passed (41)
   Start at  00:04:53
   Duration  5.14s (transform 1.86s, setup 0ms, import 2.72s, tests 2.30s, environment 0ms)

```

#### Focused rerun after correcting fixture typing (verbatim)

Command: `pnpm exec vitest run --maxWorkers=1 src/cognition/turn-orchestrator.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:790208) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}
borg open: similarity model="fake-embed" profile="generative-apis/qwen3-embedding-8b" fallback=true overrides={}

 Test Files  1 passed (1)
      Tests  105 passed (105)
   Start at  00:06:34
   Duration  14.82s (transform 3.31s, setup 0ms, import 4.50s, tests 10.19s, environment 0ms)

```


## Earlier focused re-review (historical)

The implementation descriptions below document the previous round. The narrow
re-review above replaces its timestamp event matching and pre-run generation
capture. Its validation output is retained as historical evidence.

Verified `pwd` and `git rev-parse --show-toplevel` both returned
`/home/luth/tmp/borg-wt-batch9d` before edits. No live store or `.borg-data`
directory was accessed. Validation uses focused files one at a time with
`--maxWorkers=1`, followed by one five-config typecheck with a 5 GB heap.

## Searches and reuse

Before editing, `rg` searches followed `GoalPresentation`, `goalPresentations`,
`preparedEvent`, `turnOrchestrator.run`, `applyPerGoalEmptyWakeBackoff`, and
`evaluateGoalWakeOutcome` through the scheduler, trigger builders, deliberator,
self-context builder, and presentation store. The existing
`GoalPresentationStore.snapshot`, per-goal batch grouping, and completion
transaction provide everything needed; no new scheduling helper was introduced.

Graph searches covered `semantic_edge_invalidation_events`, `processed_at`,
`restoreEdge`, `invalidation`, `reversal`, and existing pending-event helpers in
semantic storage, the belief reviser, correction, and audit code. The outbox is
written by the existing closure trigger. There was no cancellation helper; the
audited branch of `restoreEdge` is already transactional, and the reviser already
rechecks event existence before applying a planned event. Those paths were reused
without adding a new helper or a new reviser guard. The existing test `addEdge`
helper now accepts the full relation type for the contradiction regression.

## 1. Backoff belongs to the generation captured with the wake input

`src/autonomy/scheduler.ts` captures each selected goal's input generation after
source preparation, alongside building the wake input and before running the turn.
The `PreparedGoalWake` data shape carries that immutable generation through
completion accounting. Backoff no longer reads the current generation after the
turn. Current progress/status checks still determine headway and blocked-goal
handling. Source caps, the fleet brake, and presentation ownership are unchanged.

`src/autonomy/scheduler.test.ts` adds single-goal and two-goal regressions. The
second case associates new evidence with the secondary goal after prompt build
and before completion. Completion happens later than the association, proving
that the generation comparison, not timestamp ordering, preserves eligibility.
The tests verify the stored older generation, `hasNewInput`, immediate scan and
`nextDueAt` admission, unaffected-goal dormancy, and a real second scheduler tick.
After that second wake evaluates the new generation without further input, its
backoff applies normally.

Opus 5.0: in-scope because scheduling must not treat unpresented input as already evaluated.

## 2. Reversal atomically cancels the pending invalidation

`src/memory/semantic/repository.ts` extends the existing audited `restoreEdge`
branch. When reopening a closed edge, it deletes only pending outbox rows matching
that closure's edge ID, `valid_to`, and `invalidated_at`. The latter uses the same
null fallback as the insertion trigger. Already processed events and other
closures remain untouched. Dependency preservation continues to apply.

The cancellation and edge restoration share `AuditLog.revert`'s transaction with
audit finalization. The reversal registration in
`src/memory/semantic/contradiction-retirement.ts` documents that behavior. Atomic
cancellation was chosen because undoing the closure should also remove its
unconsumed work. It also protects a previously prepared reviser plan through the
existing event-existence recheck, without adding another production guard.

Regressions in `src/memory/semantic/contradiction-retirement.test.ts` verify exact
pending-event matching, retained processed/history rows, repeated undo without
new writes, preservation of a later invalidation, and rollback of both event
cancellation and restoration when audit finalization fails.

`src/offline/belief-reviser/index.test.ts` adds review and supersession cases:
retire an edge with a `derived_from` dependent at confidence `0.8`, prepare a
reviser plan, reverse twice, then apply the old plan and run a fresh reviser pass.
The edge/dependency are restored, no belief revision is queued, confidence stays
`0.8`, and no vector-sync work remains.

Opus 5.0: in-scope because reversing a structural mutation must cancel its pending downstream work.

No configuration, schema migration, or migration renumbering was added.

## Validation

All six focused files passed: 255 tests, one file and one worker at a time.

All five typecheck configurations passed in the single requested run, after the
focused tests. Targeted Prettier and `git diff --check` also passed. No full suite
or parallel heavy command was run.

### Typecheck (verbatim output)

Command: `NODE_OPTIONS=--max-old-space-size=5120 pnpm typecheck`

Exit code: `0`.

```text

> borg@0.1.0 typecheck /home/luth/tmp/borg-wt-batch9d
> tsc --noEmit && tsc --noEmit -p tsconfig.eval.json && tsc --noEmit -p tsconfig.assessor.json && tsc --noEmit -p tsconfig.simulator.json && tsc --noEmit -p tsconfig.test.json
```

### Focused tests (verbatim outputs)

Command: `pnpm exec vitest run --maxWorkers=1 src/autonomy/scheduler.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:767272) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  122 passed (122)
   Start at  23:43:59
   Duration  12.97s (transform 2.99s, setup 0ms, import 4.13s, tests 8.67s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/memory/semantic/contradiction-retirement.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:767956) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  14 passed (14)
   Start at  23:44:27
   Duration  4.25s (transform 2.15s, setup 0ms, import 3.00s, tests 1.12s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/offline/belief-reviser/index.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:768847) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  40 passed (40)
   Start at  23:44:58
   Duration  8.55s (transform 1.89s, setup 0ms, import 2.75s, tests 5.68s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/memory/semantic/repository.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:769111) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  32 passed (32)
   Start at  23:45:08
   Duration  1.44s (transform 211ms, setup 0ms, import 459ms, tests 841ms, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 scripts/contradiction-edge-repair.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:769763) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  6 passed (6)
   Start at  23:45:35
   Duration  5.84s (transform 1.87s, setup 0ms, import 2.71s, tests 3.00s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/autonomy/triggers/executive-focus-due.test.ts`

Exit code: `0`.

```text

 RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:770683) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  41 passed (41)
   Start at  23:46:20
   Duration  5.65s (transform 2.21s, setup 0ms, import 3.16s, tests 2.37s, environment 0ms)
```
