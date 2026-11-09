# Sprint 4: graph hygiene and the executive lane

Implemented in `/home/luth/tmp/borg-wt-batch9d` on `batch9-edges`, starting at
`171832c40b17c3d0580569d5a97aeeec55db0c4d`. The working directory and Git top-level
were verified before edits. No live store was opened, no backfill was applied to
deployment data, and no full test suite was run.

Independent-review fixes and current serial validation are recorded in
[sprint-4-review-fixes.md](sprint-4-review-fixes.md).
The subsequent focused review is recorded in
[sprint-4-rereview-fixes.md](sprint-4-rereview-fixes.md).

## Changes and files by item

### 1. Judged non-conflicts retire their contradiction edge

`keep_both` and `dismiss` now close the contradiction item's own `edge_id` after
checking the relation and both endpoint handles. Review finalization, edge
invalidation, its outbox event, the existing identity event, and maintenance audit
share the review repository's SQLite transaction. Handler-bypass dismissals remain
non-mutating. `loser_edge_id` retains its existing semantics.

Files: `src/memory/review-queue/review-handlers/semantic-pair.ts`,
`src/memory/semantic/contradiction-retirement.ts`, `src/borg/offline-setup.ts`,
`src/offline/test-support.ts`, and the judged backfill files listed below.
Behavioral tests: `semantic-pair.test.ts`, `contradiction-retirement.test.ts`,
`src/offline/review-resolver/index.test.ts`, and `scripts/contradiction-edge-repair.test.ts`.

Invariant: only a structurally referenced, judged edge is retired. The audit
records the review ID, resolution, process provenance, and complete before/after
edge snapshots. Invalidation uses `by_process: "review"` and `by_review_id`.
Reversal metadata also records the exact invalidation event ID for each edge.
Question-closing paths have no edge verdict/reference that would authorize the
same mutation; they remain unchanged. No new configuration.

### 2. Supersession retires incident contradictions

`SemanticNodeRepository.markSuperseded` and the common SQL writer used by node
insert/update/restore call the same retirement operation. It gathers both
directions, deduplicates edge IDs, closes only open `contradicts` edges, and audits
the node ID, correction reference, writer process, and edge snapshots. Node SQL
lifecycle changes and retirement/audit commit together. The existing vector
write compensation remains in place for generic updates/restores.

Files: `src/memory/semantic/repository.ts`, the shared retirement/reverser module,
and the supersession backfill files. Behavioral tests:
`src/memory/semantic/contradiction-retirement.test.ts` and the shared backfill test.

Invariant: ordinary insertion, evidence merge, and restoration cannot create an
open contradiction touching a node whose status is `superseded`. Historical
closed edges remain admissible. Non-contradiction relations retain their existing
behavior; no edges are automatically re-pointed. `corrected_by` is not necessarily
a successor node ID, so it cannot establish a meaningful replacement relation.

The registered maintenance reverser uses `restoreEdge` with an explicit internal
audit-undo exception. This deliberately restores the recorded pre-repair graph,
including an old stale edge, while rejecting undo if the edge changed after the
repair. Without this exception the supersession backfill would not be reversible.
It does not restore the superseded node's status or unresolve the review.
Invalidation leaves belief dependencies untouched, and reversal preserves those
rows (including `derived_from` rows and their original creation timestamps).
Reopening the edge cancels only the pending invalidation event ID recorded by
that retirement, within the audit-reversal transaction. Matching timestamps do
not identify a closure. Legacy retirement audits lacking event IDs are refused
before any writes because their pending work cannot be safely identified.

Adjacent decision left out of scope: `contradicted` nodes also carry
`superseded_at`; they are not superseded nodes for this policy. Neither a timestamp
alone nor archival causes retirement. Consolidation currently supersedes episodes,
not semantic nodes. No new configuration.

### 3. Executive presentation ownership

Executive acknowledgements use a separate per-goal watermark,
`autonomy:goal-presentation:executive:<goalId>`, in the global watermark session.
Its metadata records input generation, progress anchor and turn ID; `updatedAt`
is the presentation timestamp. The S2 planner and finalizer acknowledge the same
turn once. Each mixed-batch row keeps its own source owner. Other presentations
can still update the existing general presentation latch, but cannot update the
executive acknowledgement. Old global presentation rows are not imported.

Files: `src/executive/goal-presentation.ts`,
`src/cognition/autonomy-trigger.ts`, `src/cognition/deliberation/deliberator.ts`,
`src/autonomy/triggers/executive-focus-due.ts`, `src/autonomy/scheduler.ts`,
`src/borg/autonomy-setup.ts`, and their co-located tests.

Invariant: the same generation becomes eligible again after the existing
`autonomy.executiveFocus.wakeCooldownSec` (default 3,600 seconds), while structural
input reopens eligibility. Existing empty-wake backoff/dormancy, due-step policy,
source budget caps, and fleet brake remain operative. Score computation and
executive payload are unchanged. Cross-source delegation now consults the actual
follow-up admission rule, including its age and presentation latches. Coalescing
an executive event under a follow-up-owned row no longer stamps that executive
event's cooldown. Source budget accounting is unchanged.

Shared empty-wake backoff records each goal's frozen snapshot actually rendered
by the planner/finalizer, carried through turn execution metrics to `TurnResult`.
Input arriving after run starts but before self-context capture is included;
input associated after capture remains newer than that completion's backoff.
Goals absent from the rendered snapshot list do not acquire backoff. Legacy
custom runners that omit presentation metadata retain timestamp-only accounting.
Both sources compare structural input against that shared record, independently
of executive/global presentation acknowledgements. This preserves input/backoff
ordering even within one clock tick. Legacy backoff rows still use progress and
interaction timestamps; rendering cannot refresh either timestamp.

`nextDueAt` now chooses the earliest eligible due step after per-goal cooldowns;
the earliest raw due step may itself still be cooling down. No new configuration.

### 4. Follow-up inactivity limit

The effective activity anchor is the latest of `created_at`, actual
`last_progress_ts`, and nullable durable `last_interaction_at`. Qualifying goal
writes and first-time model-associated inbound evidence stamp interaction and
reopen the generation. Association replay, rendering, scoring and reads do not.
Status transitions and named block/unblock operations qualify; automatic unblock
uses the structural release timestamp rather than the later observation time.
Restore carries the old timestamp instead of manufacturing fresh interaction.
Episode-backed model progress uses the existing goal mutation path; vector
similarity alone is not an interaction.

Files: `src/memory/self/goals-repository.ts`, `goal-inactivity.ts`, `types.ts`,
`shared/sql-mapping.ts`, `migrations.ts`,
`src/autonomy/triggers/goal-followup-due.ts`, the executive cross-source admission
call, `src/borg/autonomy-setup.ts`, `src/config/index.ts`, and associated tests.

Invariant: an inactive goal is excluded before deadline, topology and backoff
bypasses. SQL filters inactive candidates before the bounded next-due scan; a
predicted future wake beyond the inactivity boundary is also excluded. Paused
block scheduling clocks and global recall remain unchanged. The same exclusion
applies before executive stale-goal fallback, even if follow-up delegation is
disabled or unavailable. It does not retire, block, or hide a goal. Existing
explicit due-step scheduling and score computation remain unchanged.

Configuration: `autonomy.triggers.goalFollowupDue.maxInactivityMs`, default
`2_592_000_000` (30 days), with environment override
`BORG_AUTONOMY_TRIGGER_GOAL_FOLLOWUP_DUE_MAX_INACTIVITY_MS`. The default is shared by
configuration and direct trigger construction. The accepted exploration baseline
had 49 of 70 active goals beyond 30 days and a gap between 22.51 and 49.15 days;
30 days separates that old cohort without changing status. Those figures were
not re-queried during implementation.

### 5. Declined action state

`declined` has nullable `declined_at`, `active=false`, `terminal=true`, and
`terminal_closure_guard=true`. The extractor and reflector choose the classification
from model output. Prompts distinguish voluntary abstention from failure or
non-completion; the harness does not inspect wording. Legacy `not_done` rows keep
their state. The nullable insert default is deliberately removed from the partial
patch field so lifecycle updates stamp, and later unrelated patches preserve,
`declined_at`.

Files: `src/memory/actions/{types,repository,migrations,open-question-resolution}.ts`,
`src/memory/lifecycle-ops/{complete,index}.ts`,
`src/cognition/actions/action-state-extractor.ts`,
`src/cognition/prompts/{action-extraction,reflector}.ts`,
`src/cognition/reflection/reflector.ts`,
`src/cognition/evidence-ledger/action-threads.ts`,
`simulator/{metrics,memory-snapshot}.ts`, `src/tools/internal/outbound-post.ts`,
`src/test-support/factories/memory.ts`, and tests/typed fixtures listed below.

Invariant: declining closes the action without completion callbacks or completed
question resolution. Later lifecycle completion cannot overwrite the terminal
choice. Rendered action disposition explicitly says `closed by choice to decline`;
terminal salience and state counts include declined separately. Simulator terminal
state lists include it. The commitment reconciler at this revision has no action
record consumer or `not_done` failure count to modify. No new configuration.

## Migrations

| Module | ID | Name | Effect |
| --- | --- | --- | --- |
| Self | 15 | `goal_last_interaction_at` | Add nullable `goals.last_interaction_at`; no historical timestamp inference. ID 14 is reserved for Sprint 5's earlier integration. |
| Actions | 2 | `action_declined_state` | Add nullable `declined_at`, then rebuild the action table with the expanded state CHECK. Preserve rows, deployment columns, references, indexes and triggers. |

The action migration follows `goal_superseded_status`; it changes machine-authored
DDL, not model-authored content. Neither graph backfill runs at boot.

## Backfills

Entry points:

- `scripts/backfill-judged-contradiction-edges.ts`
- `scripts/backfill-superseded-contradiction-edges.ts`

Both share `scripts/contradiction-edge-repair.ts` and the existing
`scripts/audited-data-repair.ts` driver. The former selects resolved contradiction
items with `keep_both`/`dismiss` and matching edge/endpoints, excludes finalizations
marked `resolver_finalized_without_handler`, and deduplicates edge IDs. Earlier
`needs_manual`/`apply_failed` diagnostics do not exclude a later judged dismissal.
The latter
selects open contradiction edges incident to `status='superseded'`, deduplicated
even when both endpoints are superseded. Counts are discovered from the selected
store; the supplied planning counts (220 keep-both + 51 dismiss, and 51 superseded
incident edges) are not hardcoded. The repair sets can overlap, so applying one can reduce the other
script's next dry-run count.

Example dry runs on a stopped copy, after applying the current schema migrations:

```sh
pnpm tsx scripts/backfill-judged-contradiction-edges.ts --data-dir /path/to/stopped-copy
pnpm tsx scripts/backfill-superseded-contradiction-edges.ts --data-dir /path/to/stopped-copy
```

Add `--apply` to execute a plan, and `--apply --resume` to resume a pending journal.
The existing explicit `--live-store-confirmed /path/to/backup` convention is retained
for the operator's stop-window run. No deployment backfill was executed here.

Dry-run JSON shape:

```text
version, kind, dataDir, plannedAt, runId,
summary: { changed, skipped_diagnostic, skipped_mismatched_refs },
rows: [{ before, after, provenance, review, nodes }],
live_store_override, completion_audit, journal: { pending, archive }
```

For no changes, `runId`, `completion_audit` and `journal` are null. Planning opens
SQLite read-only and writes neither data, audit nor journal. Apply rechecks edge
and judgment/node snapshots, commits the batch's invalidations and audits in one
SQLite transaction, then records completion and archives the existing journal.
An interrupted commit can resume without double-applying the batch. Repeat plans
exclude closed edges. Use `borg.audit.list({runId})` and `borg.audit.revert(id)`
(or the existing `audit list --run-id` / `audit revert` CLI commands configured for
the same bank) to undo each `correction:retire_contradiction_edges` record. Completion
summary reversal is a registered no-op; data reversal restores the edge snapshots.

## Search and reuse notes

Before introducing helpers, searches covered these symbols and their call sites:

```text
rg -n 'invalidateEdge|closeSemanticEdgeFromReview|restoreEdge|maintenance_audit|ReverserRegistry' src scripts
rg -n 'markSuperseded|superseded_by|superseded_at|corrected_by' src
rg -n 'GoalPresentationStore|presented\(|associateInbound|progressWritten|goalFollowupDue|goalStaleBackoff' src
rg -n 'goal_input_associations|advanced_goals|updateProgress|updateGoal' src
rg -n 'not_done|ACTION_STATE_METADATA|actionMigrations|goal_superseded_status' src simulator scripts
```

Reused: `invalidateEdge`, `closeSemanticEdgeFromReview`, review resolution's SQLite
transaction, `AuditLog`/`ReverserRegistry`/`restoreEdge`, the audited repair driver
and journal/preflight primitives, watermark transactions/metadata, idempotent goal
association handles, existing goal CAS and block scheduling, executive scoring,
action state metadata/timestamp mapping, and the preserving goal table rebuild.

New shared helpers: `retireContradictionEdges` and its reverser registration;
`isContradictionForNodePair`; `goalActivityAnchor`/`isGoalInactive` with one default;
`goalFollowupAdmission` shared by both sources; `executivePresentationGoalIds`;
and `declineAction`, sharing `closeUncompletedAction` with `markActionNotDone`.
The two repair wrappers share one planning/apply implementation. New script tests
are included in `tsconfig.test.json`.

Repository instructions `CLAUDE.md`, `WORKFLOW.md`, and `AGENTS.md` informed the
implementation. All new decisions in harness code depend on IDs, enum states,
timestamps, source ownership, or model-supplied associations. No content matching
or deterministic semantic judgment was added.

## Original implementation validation (before independent review)

Root and demo/server dependencies were installed from their existing npm lockfiles
inside this worktree using `npm ci --ignore-scripts`. No lockfile changes were
needed. An initial `pnpm typecheck` could not start without dependencies; a later
run identified the missing demo server dependencies used by the test configuration.
The existing root/server lockfiles select incompatible Zod minor versions (4.4.3
and 4.5.4). Local validation aligned the server install with the root using
`npm install --prefix demo/server --no-save --package-lock=false --ignore-scripts zod@4.4.3`.
Neither lockfile nor dependency manifest was changed; this environment alignment
is needed to reproduce the five-config check from these two lockfiles.

The default Node heap was insufficient for the final test typecheck configuration
(exit 134 at the 4 GB heap limit). The same five-config command was rerun with
`NODE_OPTIONS=--max-old-space-size=12288`; no typecheck configuration was removed.

Focused validation covers every changed test fixture and the affected graph,
review/audit, goal, scheduler, action, rendering, and simulator areas. The one TODO
is pre-existing: stale plan-branch commitment retirement in
`src/cognition/shared-state/reconciliation.test.ts`.

Final command outputs and the exact focused file list follow.


### Focused tests (verbatim result)

Command: `pnpm vitest run --maxWorkers=4` with the 48 files below.

```text
 Test Files  48 passed (48)
      Tests  1572 passed | 1 todo (1573)
   Start at  23:03:13
   Duration  34.12s (transform 9.83s, setup 0ms, import 54.05s, tests 72.82s, environment 7ms)

EXIT_CODE=0
```

```text
scripts/contradiction-edge-repair.test.ts
simulator/memory-snapshot.test.ts
simulator/metrics.test.ts
simulator/overseer.test.ts
src/autonomy/scheduler.test.ts
src/autonomy/triggers/executive-focus-due.test.ts
src/autonomy/triggers/goal-followup-due.test.ts
src/borg/__tests__/outbound.test.ts
src/cognition/actions/action-state-extractor.test.ts
src/cognition/actions/turn-action-state-service.test.ts
src/cognition/autobiographical-recall.test.ts
src/cognition/autonomy-trigger.test.ts
src/cognition/deliberation/deliberator.test.ts
src/cognition/deliberation/prompt/system-prompt.test.ts
src/cognition/evidence-ledger/action-threads.test.ts
src/cognition/evidence-ledger/builder.test.ts
src/cognition/generation/post-generation-guards-shadow.test.ts
src/cognition/lifecycle/turn-lifecycle-tracker.test.ts
src/cognition/lifecycle/turn-phase-coordinator.test.ts
src/cognition/lifecycle/turn-phase/post-generation-phase.test.ts
src/cognition/lifecycle/turn-phase/shared-state-phase.test.ts
src/cognition/reflection/reflector.test.ts
src/cognition/shared-state/compiler.test.ts
src/cognition/shared-state/reconciliation.test.ts
src/cognition/turn-orchestrator.test.ts
src/config/index.test.ts
src/correction/service.test.ts
src/executive/goal-presentation.test.ts
src/memory/actions/migrations.test.ts
src/memory/actions/open-question-resolution.test.ts
src/memory/actions/repository.test.ts
src/memory/lifecycle-ops/action-session.test.ts
src/memory/lifecycle-ops/lifecycle-ops.test.ts
src/memory/review-queue/review-handlers/semantic-pair.test.ts
src/memory/review-queue/review-queue.test.ts
src/memory/self/migrations.test.ts
src/memory/self/repository.test.ts
src/memory/semantic/contradiction-retirement.test.ts
src/memory/semantic/repository.test.ts
src/memory/working/store.test.ts
src/offline/audit-log.test.ts
src/offline/lived-experience-day-summarizer/index.test.ts
src/offline/review-resolver/index.test.ts
src/offline/ruminator/index.test.ts
src/retrieval/semantic-retrieval.test.ts
src/stream/watermark.test.ts
src/tools/internal/goals-block.test.ts
src/tools/internal/outbound-post.test.ts
```

### Item 5: additional typed fixtures and count expectations

These files carry the nullable timestamp or expanded state counts; all were included
in focused validation. Behavioral test files are listed with each item above.

```text
simulator/memory-snapshot.test.ts
simulator/metrics.test.ts
simulator/overseer.test.ts
src/borg/__tests__/outbound.test.ts
src/cognition/actions/turn-action-state-service.test.ts
src/cognition/autobiographical-recall.test.ts
src/cognition/deliberation/prompt/system-prompt.test.ts
src/cognition/evidence-ledger/builder.test.ts
src/cognition/generation/post-generation-guards-shadow.test.ts
src/cognition/lifecycle/turn-lifecycle-tracker.test.ts
src/cognition/lifecycle/turn-phase-coordinator.test.ts
src/cognition/lifecycle/turn-phase/post-generation-phase.test.ts
src/cognition/lifecycle/turn-phase/shared-state-phase.test.ts
src/cognition/shared-state/compiler.test.ts
src/cognition/shared-state/reconciliation.test.ts
src/cognition/turn-orchestrator.test.ts
src/memory/lifecycle-ops/action-session.test.ts
src/memory/working/store.test.ts
src/offline/lived-experience-day-summarizer/index.test.ts
src/offline/ruminator/index.test.ts
src/tools/internal/outbound-post.test.ts
```

### Typecheck (verbatim output)

Command: `NODE_OPTIONS=--max-old-space-size=12288 pnpm typecheck`.
All five configurations completed; process exit code: `0`.

```text
> borg@0.1.0 typecheck /home/luth/tmp/borg-wt-batch9d
> tsc --noEmit && tsc --noEmit -p tsconfig.eval.json && tsc --noEmit -p tsconfig.assessor.json && tsc --noEmit -p tsconfig.simulator.json && tsc --noEmit -p tsconfig.test.json
```

Targeted Prettier check passed for all changed/new TypeScript, JSON and Markdown
files. `git diff --check` passed. No full suite or deployment backfill was run.
