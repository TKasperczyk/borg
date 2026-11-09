# Sprint 4 independent-review fixes

The subsequent two-finding review and current validation are recorded in
[sprint-4-rereview-fixes.md](sprint-4-rereview-fixes.md).

All work stayed in `/home/luth/tmp/borg-wt-batch9d`; `pwd` and
`git rev-parse --show-toplevel` were verified before edits. No live store or
`.borg-data` directory was accessed. Focused tests ran one file at a time with
`pnpm exec vitest run --maxWorkers=1 <file>`. The full suite was not run.

## Searches and reuse

Before editing, searches followed `hasNewExecutiveInput`, `hasNewInput`,
`goalFollowupAdmission`, `isGoalStaleBackedOff`, `backoffConcern`,
`GoalWakeSnapshot`, `GoalWakeOutcome`, `inputGeneration`, `associateInbound`,
`presented`, and `last_interaction_at` through the executive store, both triggers,
the scheduler, and their tests. The shared backoff module and goal inactivity
helper already supplied the needed primitives; no parallel admission helper was
introduced.

The graph searches followed `restoreEdge`, `source_edge_id`, `listBySourceEdge`,
`resolver_finalized_without_handler`, `__borg_review_resolver_diagnostic`,
`needs_manual`, and `apply_failed` through semantic repositories, audit reversal,
the review resolver, and repair scripts/tests. They confirmed that invalidation
does not mutate dependencies, that `restoreEdge` normally deletes source-edge
dependencies, and that diagnostic refs can survive later judgments.

Migration searches checked `goal_last_interaction_at` and the two pending composed
IDs against registration, name-based migration tests, and the Sol-ledger pin.
No new helper, configuration key, migration, or dependency was needed for these
review fixes. Existing helper APIs and watermark metadata were extended.

## Findings addressed

### 1. Inactivity cannot fall through to executive stale selection

`src/autonomy/triggers/executive-focus-due.ts` applies `isGoalInactive` directly
before cooldown/backoff/delegation checks. An excluded goal cannot be selected by
negating failed follow-up admission. The shared configured inactivity limit also
applies when follow-up is disabled. Explicit due-step policy and scoring are
unchanged.

Regression: the old test expecting an inactive goal to wake was replaced by two
cases (follow-up enabled/disabled) covering an overdue deadline, changed action
topology, disabled backoff respect, inert rendering, unchanged status, and a fresh
association restoring eligibility.

Opus 5.0: in-scope because source arbitration must preserve a structural age exclusion.

### 2. Shared backoff compares against structural input, not presentation ownership

`src/autonomy/executive-focus-stale-backoff.ts` accepts the current input generation
and interaction timestamp. `src/autonomy/scheduler.ts` records `input_generation`
when writing shared empty-wake backoff. Following the subsequent review, this is
the generation captured with the wake input, before the turn runs. Both triggers use this shared comparison,
including executive delegation and dormant-goal reporting. Missing or older
executive acknowledgements only reopen executive presentation eligibility; they
cannot bypass newer shared backoff. Generations distinguish writes within the
same clock tick. Legacy rows retain timestamp-based release behavior.

Regressions in `executive-focus-due.test.ts` cover both input/backoff orders,
cooldown/dormancy, missing/older executive acknowledgements, actual scheduler
backoff writing after a follow-up presentation, association replay, release after
later input even when another source acknowledged it, and legacy timestamp rows.
`goal-followup-due.test.ts` seeds generation-bearing shared backoffs for its
same-timestamp reopening cases. `scheduler.test.ts` keeps exact metadata assertions
for rollback, stale-slot reservation, and duplicate-lane accounting, including the
new generation field.

Opus 5.0: in-scope because shared scheduling brakes must follow structural input order.

### 3. Earlier diagnostics do not disqualify a later judged dismissal

`scripts/contradiction-edge-repair.ts` excludes only the structural
`resolver_finalized_without_handler` marker. It no longer excludes every resolved
dismissal carrying diagnostic refs. Relation/endpoints, deduplication, dry-run
default, and audited apply/reversal remain unchanged.

`scripts/contradiction-edge-repair.test.ts` covers both `needs_manual` and
`apply_failed` followed by a successful judged dismissal with retained refs. Each
fixture has one judged candidate, one actual handler bypass, and one mismatched
reference. The formatted dry run reports exactly:

```json
{ "changed": 1, "skipped_diagnostic": 1, "skipped_mismatched_refs": 1 }
```

The test also applies the plan and verifies the judgment's review ID and resolution
in invalidation provenance/audit. The existing `skipped_diagnostic` output key is
preserved for compatibility; it counts marked handler-bypass finalizations.

Opus 5.0: in-scope because repair eligibility depends on structural finalization provenance.

### 4. Audit reversal preserves source-edge dependencies

`src/memory/semantic/repository.ts` adds `preserveDependencies` to the existing
`restoreEdge` options. `src/memory/semantic/contradiction-retirement.ts` uses it
alongside the audited reversal exception after checking the expected edge shape.
The existing `AuditLog.revert` transaction still covers restoration and audit
finalization. Ordinary restore behavior is unchanged.

Preservation was chosen over copying dependencies into the reversal payload:
invalidation never changes those rows, so undo need not delete/recreate them or
overwrite dependencies added later. Two regression cases in
`contradiction-retirement.test.ts` reverse review and supersession retirement with
`derived_from` dependencies targeting a node and another edge. They check exact
rows and timestamps, including a dependency added after retirement.

Opus 5.0: in-scope because undo must not delete unrelated structural dependency records.

### 5. The Sol-ledger pin includes both sprint migrations

`src/storage/sqlite/migrations.test.ts` now expects ten pending migrations and
83 ledger rows, with the two added IDs in sorted order and ten identical new
application timestamps. Historical timestamps remain unchanged. The repeat-run
assertions still require no new executions, no ledger changes, and no SQLite
changes.

At this review stage, `goal_last_interaction_at` kept its name and registration;
the integration renumbering was deferred as requested. The later narrow review
authorized the move to self ID 15, recorded in
[sprint-4-rereview-fixes.md](sprint-4-rereview-fixes.md). No fixture ledger was edited.

Opus 5.0: in-scope because migration accounting must match the registered schema.

## Validation

The initial scheduler run had eight assertion failures at three expectation sites,
all due to the added `input_generation` metadata field. Those exact expectations
were updated; behavior assertions were retained. Final outputs follow below.


All 8 focused files passed (242 tests). The five-config typecheck passed once
with an 8 GB heap, after the focused checks finished. No heavy commands ran in
parallel. Targeted Prettier and `git diff --check` also passed.

### Typecheck (verbatim output)

Command: `NODE_OPTIONS=--max-old-space-size=8192 pnpm typecheck`

Exit code: `0`.

```text
> borg@0.1.0 typecheck /home/luth/tmp/borg-wt-batch9d
> tsc --noEmit && tsc --noEmit -p tsconfig.eval.json && tsc --noEmit -p tsconfig.assessor.json && tsc --noEmit -p tsconfig.simulator.json && tsc --noEmit -p tsconfig.test.json
```

### Focused tests (verbatim output)

Command: `pnpm exec vitest run --maxWorkers=1 src/autonomy/triggers/executive-focus-due.test.ts`

Exit code: `0`.

```text
RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:741243) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  41 passed (41)
   Start at  23:24:11
   Duration  5.78s (transform 2.28s, setup 0ms, import 3.22s, tests 2.43s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/autonomy/triggers/goal-followup-due.test.ts`

Exit code: `0`.

```text
RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:737013) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  20 passed (20)
   Start at  23:20:58
   Duration  3.96s (transform 1.83s, setup 0ms, import 2.67s, tests 1.15s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 scripts/contradiction-edge-repair.test.ts`

Exit code: `0`.

```text
RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:737247) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  6 passed (6)
   Start at  23:21:07
   Duration  5.78s (transform 1.90s, setup 0ms, import 2.75s, tests 2.91s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/memory/semantic/contradiction-retirement.test.ts`

Exit code: `0`.

```text
RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:737750) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  12 passed (12)
   Start at  23:21:22
   Duration  3.89s (transform 1.89s, setup 0ms, import 2.74s, tests 1.02s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/storage/sqlite/migrations.test.ts`

Exit code: `0`.

```text
RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:738476) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  8 passed (8)
   Start at  23:21:46
   Duration  3.45s (transform 2.21s, setup 0ms, import 3.14s, tests 165ms, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/autonomy/scheduler.test.ts`

Exit code: `0`.

```text
RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:742026) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  120 passed (120)
   Start at  23:24:35
   Duration  11.31s (transform 2.46s, setup 0ms, import 3.39s, tests 7.77s, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/memory/semantic/repository.test.ts`

Exit code: `0`.

```text
RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:743478) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  32 passed (32)
   Start at  23:25:18
   Duration  1.38s (transform 211ms, setup 0ms, import 447ms, tests 805ms, environment 0ms)
```

Command: `pnpm exec vitest run --maxWorkers=1 src/offline/audit-log.test.ts`

Exit code: `0`.

```text
RUN  v4.1.9 /home/luth/tmp/borg-wt-batch9d

(node:744076) ExperimentalWarning: SQLite is an experimental feature and might change at any time
(Use `node --trace-warnings ...` to show where the warning was created)

 Test Files  1 passed (1)
      Tests  3 passed (3)
   Start at  23:25:36
   Duration  3.68s (transform 2.43s, setup 0ms, import 3.35s, tests 198ms, environment 0ms)
```

