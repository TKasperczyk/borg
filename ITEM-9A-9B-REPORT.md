Implemented items 9a and 9b on `codex-swarm/cc4b94ab`, in `/tmp/codex-swarm-worktrees/cc4b94ab`.

Opus 5.0: in-scope because named goal state, paused scheduling clocks, and exact response-boundary evidence are harness information and structural correctness gaps.

`tool.goals.block` and `tool.goals.unblock` follow `tool.goals.retire` registration, origin permissions, and autonomous finalizer availability. Blocking requires an active goal, `attempt_status: "attempted_unavailable"`, a nonempty reason, and exactly one strict tagged blocker: another existing goal, an existing entity, or a Unix-millisecond timestamp. Optional attempt evidence uses the shared reflection artifact-reference schema, with record existence, timestamp, and applicable state checks. Generic creation/status updates cannot create bare blocked goals or silently reactivate them.

Block and unblock transitions persist complete history and identity events in the same SQLite transaction. Reconciliation runs at startup, after indexed stream appends, before scheduler ticks, and when building turn goal context. It releases blocks when the named goal is done/abandoned (retirement uses abandoned), the timestamp passes, or an active inbound `user_msg`, `user_image_attachment`, or `agent_observed` from the named entity has a strictly later timestamp. Entity arrivals can come from any session. Release reasons identify the timestamp or source event and its time. Manual unblock requires a reason. Blocking preserves open executive steps; terminal closure still abandons them.

Blocked goals remain in unfinished-goal cognition and retain blocker, time, reason, evidence handle, and history on the goal digest, ledger, reflection, autonomous prompt, wake goal payload, autobiographical recall, operator UI, and planner captures. Executive competition and due-goal selection exclude blocked goals. Closed block intervals pause progress-debt and deadline clocks, including read-only due queries, without rewriting real progress timestamps or target dates.

`AnsweredStreamWindow` now has a shared documented definition. Model-facing evidence resolves the persisted terminal `response_to` stamp through the stream index and records the response/turn, last answered entry ID/time, and actual source-record count. Outside counts partition recorded session `user_msg` inputs outside that exact source-ID set into arrivals after the edge, unselected entries inside the response span, and entries before the span. The render distinguishes `none`, no recorded edge, and unavailable basis records. Both the detailed mechanism evidence and compact suppression diagnostics expose this evidence. Wake rows join their recorded session to a current observation of its edge. Planner capture/replay preserves these labels. No response decision, output policing, or recall gate depends on them.

Decisions and specification boundaries:

- No requested feature shape was rejected. An attempt declaration is structurally enforceable; proving that an attempt was adequate from the text would violate the cardinal rules. Evidence remains optional as requested, and its meaning is left to the model.
- A goal cannot name itself as its blocker: reaching its own terminal state cannot return that same goal to active. Already-satisfied blockers release immediately, retaining both audited transitions.
- The invalid existing shape was a legacy bare `blocked` row. No blocker or block time can be recovered from that row. The migration reactivates such rows and preserves their old values in identity events rather than inventing history. New named blocks have full history. A blocker goal becoming blocked is not terminal and does not release its dependents.
- Automatic release uses the structural event time as the end of the pause, with observation time also recorded in the reason. This avoids charging a goal for a delayed reconciliation of a timestamp that already passed.
- The answered window is an exact set, not a contiguous prefix or the ingestion watermark. Outside counts describe membership in the latest response's scope; they are not pending-response counts. Earlier entries can have been answered by a previous response. `user_msg` is the existing backlog input kind; thoughts and internal context records do not count as extra inbound messages.
- Wake evidence is explicitly a current read of the wake's session, not a reconstruction of what was known when the wake fired. A row with no recorded session is marked not applicable; a caller without an index provider leaves evidence unavailable.
- The operator's previous bare-block action was updated to the same invariant. Extra demo typechecking also required a missing `counterparty_entity_id` in goal test fixtures and the existing day-summarizer process's missing description-map entry.

Final validation results:

| Check | Result |
| --- | --- |
| Root `pnpm typecheck` (all five tsconfigs in the root script) | Passed, exit 0 |
| Source/server Vitest command below | 30 files, 933 tests passed |
| Operator UI Vitest file | 1 file, 15 tests passed |
| Demo web `pnpm typecheck` | Passed, exit 0 |
| Demo server typecheck against this worktree's source | Passed, exit 0 |
| `pnpm heuristics:guard` | Passed, exit 0 |
| Prompt fixtures and size envelopes | Passed in the final regression run |
| `git diff --check` | Passed |

The final regression runs passed 948 tests in 31 files. No real LLM or embedding API was used. Tests emitted Node's existing experimental SQLite warning.

Required dependency preparation was completed before checks, without installing packages:

```sh
ln -s /home/luth/Programming/borg/node_modules node_modules && ln -s /home/luth/Programming/borg/demo/server/node_modules demo/server/node_modules
ln -s /home/luth/Programming/borg/demo/web/node_modules demo/web/node_modules
```

The second link command enabled checking the changed operator UI against its existing dependencies. Heavy commands used `choom -n 800 --`; every Vitest invocation used `--maxWorkers=2`. The final source/server regression command is:

```sh
choom -n 800 -- pnpm vitest run --maxWorkers=2 --no-cache --configLoader runner \
  src/tools/internal/goals-block.test.ts \
  src/tools/internal/goals-retire.test.ts \
  src/stream/answered-window.test.ts \
  src/memory/self/migrations.test.ts \
  src/memory/self/repository.test.ts \
  src/memory/identity/service.test.ts \
  src/executive/goal-competition.test.ts \
  src/autonomy/triggers/goal-followup-due.test.ts \
  src/autonomy/triggers/executive-focus-due.test.ts \
  src/autonomy/scheduler.test.ts \
  src/cognition/self/turn-self-context.human-mind-invariants.test.ts \
  src/cognition/mechanism-evidence.test.ts \
  src/cognition/deliberation/prompt/planner-context.test.ts \
  src/cognition/deliberation/prompt/system-prompt.test.ts \
  src/borg/tools-setup.test.ts \
  src/cognition/evidence-ledger/builder.test.ts \
  src/cognition/autobiographical-recall.test.ts \
  src/cognition/ingestion/coordinator.test.ts \
  src/cognition/ingestion/backlog-terminal.test.ts \
  src/cognition/turn-orchestrator.test.ts \
  src/cognition/reflection/reflector.test.ts \
  src/stream/entry-index.test.ts \
  src/cognition/prompts/prompt-surface-fixtures.test.ts \
  src/cognition/goals/goal-promotion-extractor.test.ts \
  src/cognition/goals/turn-goal-promotion-service.test.ts \
  src/offline/overseer/index.test.ts \
  src/offline/reflector/index.test.ts \
  src/offline/ruminator/index.test.ts \
  src/cognition/deliberation/planner-context-capture.test.ts \
  demo/server/src/__tests__/server.test.ts
```

Other check and fixture commands:

```sh
# Worktree root; includes root, eval, assessor, simulator, and test tsconfigs.
choom -n 800 -- pnpm typecheck
choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm heuristics:guard
choom -n 800 -- env UPDATE_PROMPT_SURFACE_FIXTURES=1 pnpm vitest run --maxWorkers=2 --no-cache --configLoader runner src/cognition/prompts/prompt-surface-fixtures.test.ts src/cognition/deliberation/prompt/planner-context.test.ts

# Worktree demo/web.
choom -n 800 -- pnpm typecheck
choom -n 800 -- pnpm vitest run --maxWorkers=2 --no-cache --configLoader runner src/pages/Mind.test.tsx

# Worktree root; temporary config extends demo/server/tsconfig.json,
# sets rootDir='.' and paths.borg=['./src/index.ts'], and includes server sources.
# This checks against this worktree's library rather than the live package's dist.
choom -n 800 -- pnpm exec tsc --noEmit -p tsconfig.9ab-server-check.json

git diff --check
```

Prettier was run under `choom -n 800 -- pnpm exec prettier --write` on changed TypeScript files. The temporary server-check config is removed before commit. Read/review commands included `rg`, `sed`, `cat`, `git status`, and `git diff`.

During implementation, earlier check runs found invalid system provenance fields, a wait-step test without `due_at`, stale migration/column assertions, stale prompt expectations, prompt-size overruns, a server test using wall time with a fixed clock, and the demo type declarations noted above. These were corrected. Prompt-size limits were preserved. Fixture regeneration and the paired planner tests passed 58 tests; the first clean broad run passed 917 tests in 29 files; the additional reflection/capture/fixture run passed 103 tests in three files. The final checks above supersede those iterations.

Workspace-boundary incident: the first three Vitest invocations had default caching enabled. Vitest followed the dependency symlink and wrote its results cache into `/home/luth/Programming/borg/node_modules/.vite/vitest/da39a3ee5e6b4b0d3255bfef95601890afd80709/results.json`. This violated the no-live-tree-write constraint and was disclosed during the task. The results-cache write was confirmed; any temporary config-bundling writes were not independently verified. Subsequent runs used `--no-cache --configLoader runner`, and the TypeScript guard used `TSX_DISABLE_CACHE=1`. The live cache was left untouched after discovery. No live source or entity database was edited, no packages were installed, and no live migrations, service restart, or deployment was performed.

Changed files, grouped by purpose (paths are relative to the worktree):

- Goal storage, artifact handles, and migration: `src/memory/common/artifact-reference.ts`, `src/memory/common/artifact-reference-validation.ts`, `src/memory/self/goal-blocks.ts`, `src/memory/self/goals-repository.ts`, `src/memory/self/types.ts`, `src/memory/self/shared/sql-mapping.ts`, `src/memory/self/migrations.ts`, `src/memory/self/migrations.test.ts`, `src/memory/self/repository.test.ts`, `src/memory/self/index.ts`, `src/memory/identity/migrations.ts`.
- Tools and public wiring: `src/tools/internal/goals-block.ts`, `src/tools/internal/goals-block.test.ts`, `src/tools/internal/index.ts`, `src/tools/index.ts`, `src/borg/tools-setup.ts`, `src/borg/tools-setup.test.ts`, `src/borg/repositories.ts`, `src/borg/facade.ts`, `src/borg/facade-types.ts`, `src/borg/public-facade.ts`, `src/borg/open.ts`, `src/borg/autonomy-setup.ts`, `src/index.ts`.
- Scheduling: `src/executive/goal-competition.ts`, `src/autonomy/scheduler.ts`, `src/autonomy/types.ts`, `src/autonomy/triggers/goal-followup-due.ts`, `src/autonomy/triggers/executive-focus-due.ts`.
- Goal cognition and recall: `src/cognition/self/active-goals.ts`, `src/cognition/self/turn-self-context.ts`, `src/cognition/goals/goal-promotion-extractor.ts`, `src/cognition/goals/turn-goal-promotion-service.ts`, `src/cognition/evidence-ledger/builder.ts`, `src/cognition/evidence-ledger/audience-standing.ts`, `src/cognition/evidence-ledger/sections/attribution.ts`, `src/cognition/evidence-ledger/sections/group-channel-memory.ts`, `src/cognition/autobiographical-recall.ts`, `src/cognition/reflection/reflector.ts`, `src/cognition/reflection/reflector.test.ts`, `src/offline/reflector/index.ts`, `src/offline/reflector/index.test.ts`, `src/offline/overseer/index.ts`, `src/offline/ruminator/index.ts`.
- Answered edge and prompt/capture rendering: `src/stream/answered-window.ts`, `src/stream/answered-window.test.ts`, `src/stream/entry-index.ts`, `src/stream/entry-index.test.ts`, `src/cognition/ingestion/coordinator.ts`, `src/cognition/mechanism-evidence.ts`, `src/cognition/lifecycle/turn-phase/retrieval-phase.ts`, `src/cognition/deliberation/autonomous-finalizer-tools.ts`, `src/cognition/deliberation/prompt/planner-context.ts`, `src/cognition/deliberation/prompt/system-prompt.ts`, `src/cognition/deliberation/prompt/system-prompt.test.ts`, `src/cognition/deliberation/planner-context-capture.ts`, `src/cognition/deliberation/planner-context-capture.test.ts`, `src/cognition/prompts/prompt-surface-fixtures.test.ts`.
- Prompt fixtures under `src/cognition/prompts/__fixtures__/prompt-surface/`: `base-system-autonomous-dm-relational.txt`, `base-system-user-group-problem-solving.txt`, `cacheable-base-dynamic-content.txt`, `finalizer-system-blocks-s2.txt`, `s2-planner-system-prompt-autonomous.txt`, `s2-planner-system-prompt-compact.txt`, `s2-planner-system-prompt.txt`, and new `goal-block-history.txt`.
- Operator surface: `demo/server/src/app.ts`, `demo/server/src/__tests__/server.test.ts`, `demo/web/src/api/types.ts`, `demo/web/src/pages/Mind.tsx`, `demo/web/src/pages/Mind.test.tsx`.
- This report: `ITEM-9A-9B-REPORT.md`.
