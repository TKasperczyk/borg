# Shared-state correction diagnosis, 2026-09-06

The compiler had the false claim and its target ID available. The applied patch
added a correction as a new entry and updated the original while preserving its
false assertion. No structural path converted a supersede into an add.

Opus 5.0: in-scope because explicit entry presentation, correction and evidence
guidance, and a dedicated model slot address the compiler's input and interpretation
without introducing deterministic semantic judgments.

## Evidence

Source: the unrotated
`/home/luth/Programming/borg/demo/server/.borg-data/demo/traces/turns.jsonl`,
read without modifying the live tree. Line numbers below identify JSONL records;
all timestamps are UTC on 2026-09-06. The relevant turn is
`4ce91ab9-a84f-4edb-ba5e-4a190b8f38a1`, session
`sess_41uuw7ztyzh00kjc`, audience `ent_u95mmrw3eg6rt3oi`.

| Record | Time | Fact |
| --- | --- | --- |
| 19710, `shared_state.compile.completed` | 08:56:56.672 | The earlier **pre-answer** compile applied nothing: six citation rejections, artifact version 178 retained. Its preceding repair was a separate call, before the assistant's correction existed. |
| 19715, `evidence_ledger.built` | 08:56:56.779 | Contains the complete version-178 artifact. `dart_gxh8aj853rum5dmg` is active and locked, with key `observation.infrastructure.goal_standing_dispositions_unbounded_accumulation` and the false assertion “Two have been retired per Sol's judgment.” |
| 19723, `llm_call.started`, `system_1_finalizer` | 08:56:57.099 | The recorded finalizer prompt also renders that row's ID, key, locked kind, citations, and false text. |
| 19747, `shared_state.compile.degraded` | 09:02:35.575 | Post-response input estimate 60,696 exceeds the 35,000-token warning budget. Breakdown: previous-artifact summary 5,926; key registry 3,284. This warning does not truncate the request or skip compilation. |
| 19748, `llm_call.started`, `shared_state_compiler` | 09:02:35.577 | Model `claude-haiku-4-5-20251001`, attempt 1, `schema_repair=false`, 224,365 prompt characters; advertises `EmitSharedStatePatch` and its alias. |
| 19749, `llm_call.completed`, `shared_state_compiler` | 09:02:55.695 | Tool call `toolu_019spQfKuUF9bPwJThvstVYH`, `EmitSharedStatePatch`, stop reason `tool_use`. |
| 19750, `shared_state.compile.empty_update_dropped` | 09:02:55.720 | Drops an unrelated empty update targeting `dart_wolyzn2e0idt83on`. |
| 19761, `shared_state.reconcile.completed` | 09:04:01.537 | Zero current canonicalization IDs, no duplicate canonicalizations dropped, no reconciliation errors. |
| 19762, `shared_state.compile.completed` | 09:04:01.571 | **Post-response** compile applied version 179: two adds, one update, zero supersedes, two lifecycle cap prunes, one salience transition; zero rejections. The citation allowlist includes the correction `strm_ntw0djyoxtgasgvv`. |
| 20984, `evidence_ledger.built` | 10:34:01.105 | Contains version 179. Both the original and correcting entry `dart_m31a0dl9g19wt122` remain active, with null `superseded_by_id`. Both were written at 09:02:55.722. |

Version 179 preserves the original false opening and appends an explanation that
retirement happens during an autonomous wake. The sibling uses the different key
`observation.infrastructure.assistant_fabricated_shared_state_intention_true_state_false`
and records that the goals were still active. Both changes cite
`strm_ntw0djyoxtgasgvv`.

## What the compiler could see

Compiler `llm_call` records save request/response metadata, not the complete
request or emitted tool arguments. The exact raw payload therefore cannot be
quoted from those records. Visibility was reconstructed using the version-178
artifact from record 19715 and the compiler's pre-change summary/registry builders
at commit `719fbeca`.
The live config has no shared-state summary override. Reconstruction reproduces
both recorded token estimates exactly: **5,926 and 3,284**.

The reconstructed summary includes the original's **entire text**, ID, and key
in `active_entries.locked` and `active_entries_by_state_key`. The registry includes
that ID, `kinds: ["locked"]`, and the ID in `text_visible_entry_ids`. The five
rendered `shared_state_recall` results did not include this row; compiler artifact
presentation is independent of that recall slice. The compiler prefilter skips
whole calls, not individual artifact entries, and this call ran.

The tool schema already offered `supersede` with the existing entry's `id` and a
`replacement`. The system prompt already distinguished update from supersede and
recommended supersede for wrong/retracted text. The actual failure therefore was
not a missing target or missing operation.

## Structural paths and fix

`normalizePatch` validates a supersede against the full previous artifact and
preserves its operation type. Locked kind does not block supersession. Same-key
collision handling rejects sibling adds; it does not rewrite supersedes. Different
keys do not establish contradiction in the harness. Canonicalization filtering
only affects linked goal/action/commitment/question IDs. The store inserts the
replacement and links the old row through `superseded_by_id` in its transaction.
No repair or rejection occurred in the post-response call that produced the
sibling. The model-selected operations were not converted into this outcome.

The fix makes `kind` explicit on each summary row, including the keyed view,
clarifies that locked entries are correctable and which ID a supersede targets,
and explicitly requires correction of the original claim rather than a sibling
or commentary appended to a still-false assertion. Intentions, plans, and decisions
must retain their timing; performed acts require evidence of the act itself.
Both compile passes use `anthropic.models.sharedStateCompiler`, default
`claude-sonnet-5`, overridden by `BORG_MODEL_SHARED_STATE_COMPILER`.

Contradiction, choice of target, update versus supersede, and whether a source
evidences a performed act remain model judgments. Existing summary budgets remain;
this incident's target was already fully visible. Regression tests verify the
explicit row metadata and that a supplied same-key supersede of a locked entry
works without canonicalization IDs, leaving one active replacement and linked
history in both compile passes. They do not claim to measure live model judgment.

## Files changed

- `src/cognition/shared-state/summary.ts`: explicit kind on each presented entry.
- `src/cognition/prompts/shared-state.ts` and `src/cognition/shared-state/schema.ts`:
  correction, target-ID, locked-kind, and decision-versus-act guidance.
- `src/config/index.ts` and `src/cognition/lifecycle/turn-phase/retrieval-phase.ts`:
  dedicated model default, environment override, and routing for both compile passes.
- `src/cognition/shared-state/compiler.test.ts`: prompt rules, target presentation,
  and same-key locked supersession through the compiler and SQLite repository.
- `src/config/index.test.ts` and
  `src/cognition/lifecycle/turn-phase/retrieval-phase.test.ts`: model defaults,
  file/environment precedence, independence from recall expansion, and both passes.
- `src/cognition/evidence-ledger/renderer.test.ts`: fixture token budget accounts
  for the explicit kind field in both summary views.
- `src/cli/correction.test.ts` and `src/correction/service.test.ts`: complete
  model-map fixtures include the new required slot.
- `.env.example`, `README.md`, and this report: configuration and diagnosis docs.

Prompt-surface fixtures do not contain the compiler prompt; their existing pins
passed unchanged. No live entity data was modified.

## Validation

Dependencies were linked before checks, without installing:

```sh
ln -s /home/luth/Programming/borg/node_modules node_modules && ln -s /home/luth/Programming/borg/demo/server/node_modules demo/server/node_modules
```

The complete root typecheck passed across all five tsconfigs:

```sh
choom -n 800 -- pnpm typecheck
```

The relevant suite passed: **15 files, 414 tests passed, one existing TODO**:

```sh
choom -n 800 -- pnpm vitest run --maxWorkers=2 --cache=false --configLoader=runner src/config/index.test.ts src/cognition/shared-state src/memory/shared-state src/cognition/lifecycle/turn-phase/retrieval-phase.test.ts src/cognition/evidence-ledger/renderer.test.ts src/cognition/prompts/prompt-surface-fixtures.test.ts src/cli/correction.test.ts src/correction/service.test.ts
```

Cache writes and bundled config output were disabled to avoid writes through the
linked live `node_modules` directory. Checks ran sequentially with OOM score 800.
The first test run exposed the renderer fixture's tight token budget; the first
typecheck exposed four complete model maps missing the new slot. Both were fixed
before the successful runs above.

The architectural guard also passed:

```sh
choom -n 800 -- pnpm heuristics:guard
```

Scoped `choom -n 800 -- pnpm exec prettier --check` over all 11 changed TypeScript
files and `git diff --check` also passed.
