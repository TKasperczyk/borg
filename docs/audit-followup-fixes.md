# Memory tool audit follow-up

Opus 5.0: in-scope because pagination, reference integrity, and access to durable
dispositions are harness responsibilities that a stronger model cannot repair.

Reviewed commits: `254b3350`, `246b4e70`, `51f30732`, `003c1847`, and `3d23646f`.
Also read operator-attention record 145 without modifying its source file.

## Changes

1. **Rumination reader.** The repository's 50-row clamp consumed the tool's
   lookahead. The range reader now honors a validated positive limit, including
   the tool's 51-row request. The first-row budget fallback drops every unbounded
   payload field, including tensions, connected handles, and the process name;
   unbounded disclosure metadata becomes an explicit unknown label. The serialized
   fallback is checked against the same budget. Per-question recorded-note counts
   remain present. Composed regressions cover exactly 50 and 51 notes and oversized
   retained fields.
2. **Semantic edges.** Normalizing an entity-id label removed the key used by its
   incident edges. An extraction-batch map now remembers the original model handle
   and the successfully resolved node. Endpoint resolution consults it without
   persisting the handle as a label or alias. The regression includes both an
   off-roster person and a self anchor addressed by ids, joined by an edge.
3. **Citation integrity.** Well-shaped missing episode and stream ids now produce
   a tool error naming the missing references before any resolution write. The
   composed regression replaces the old repository test that endorsed dangling
   references. Permissive episode reads accept archived records, and stream-index
   existence accepts inactive records. Existence establishes neither relevance nor
   disclosure permission; stream-only resolution labels remain unknown. The
   existing requirement for at least one citation remains.
4. **Goal dispositions.** `tool.ownRecords.list` accepts `goal_disposition`, reads
   existing identity-event snapshots globally, and recounts retire/block/unblock
   transitions with historical descriptions, reasons, and recorded times. A
   distinct effective unblock time is stated when present. Prose omits internal
   handles; paging uses opaque cursors and existing disclosure metadata. The
   reader includes automatic unblocks and does not append anything to the journal.
   Tests cover mixed-source pagination at equal timestamps, explicit kind/range
   selection, cross-audience recall, no-op exclusion, and effective versus recorded
   time. Goal events have no session affiliation; the optional session filter
   continues to apply to thoughts and journal records.

## Search and reuse

- Reused rumination `withoutPayload`, `resultFitsBudget`, `countRuminations`,
  `estimatePromptTokens`, and the existing budget and disclosure-label utilities.
- Reused semantic `upsertNode`, `resolveEdgeNode`, and active-node checks. The new
  batch map moves exact source handles; it does not interpret language.
- Reused `EpisodicRepository.getMany`, `StreamEntryIndexRepository.lookupEntriesById`,
  `ToolError`, and existing identity-governance resolution. The searched
  `artifactReferenceExists` is unsuitable here: it has no episode arm and its
  stream arm applies activity restrictions beyond existence.
- Reused identity events, their row mapper, and
  `identityEventDisclosureLabelsForCollection`; no new persistence or migration.
  Own-record ordering, cursor encoding/decoding, budget handling, and relative
  timestamps also remain shared with thoughts and journal entries.
- **New range method:** searches for goal disposition readers and identity-event
  range listing found only `list`, which has no time cursor or status-transition
  selection. `listGoalDispositionsInRange` adds that SQL query over existing audit
  snapshots, with an `IdentityService` forwarding method.
- **New prose renderer:** searches in the goal, identity-event, and own-record
  renderers found no disposition prose renderer. `renderGoalDisposition` uses
  `goalSchema` and the stored transition fields; it does not interpret reason text.
- **New ID-scrubbing wrapper:** searches found the directive-only scrubber and the
  host guard's durable-ID grammar. `scrubMemoryRecordInternalIds` reuses both.
  The guard's grammar moved unchanged to `util/ids.ts`; its behavior is unchanged.

## Prompt bytes

Each changed fixture grew by 39 UTF-8 bytes (234 bytes total):

| Fixture | Before | After | Delta |
| --- | ---: | ---: | ---: |
| base-system-autonomous-dm-relational.txt | 54,361 | 54,400 | +39 |
| cacheable-base-static-prefix.txt | 14,306 | 14,345 | +39 |
| finalizer-system-blocks-s2-compact.txt | 91,348 | 91,387 | +39 |
| finalizer-system-blocks-s2.txt | 65,214 | 65,253 | +39 |
| s2-planner-system-prompt-autonomous.txt | 57,166 | 57,205 | +39 |
| s2-planner-system-prompt-compact.txt | 60,540 | 60,579 | +39 |

Other prompt fixtures are unchanged. The changed portion of the own-record tool
description grew by 210 bytes (its shared page-end claim is unchanged); the
resolution tool description shrank by 61 bytes. The own-record input kind enum
now includes `goal_disposition`. Rumination fallback `source_process` can be null.

## Validation

- 170 tests passed across 11 focused files, with the requested environment unsets,
  `--maxWorkers=2 --cache=false --configLoader=native`.
- `pnpm heuristics:guard` passed.
- Prompt fixtures were regenerated in their focused test and then passed without
  regeneration in the final focused run.
- `pnpm typecheck` passed, run once after implementation and focused testing.
- No live-store writes, schema changes, or new dependencies.

Focused test command:

```sh
env -u BORG_TRACE -u BORG_TRACE_PROMPTS -u BORG_DATA_DIR -u EXTRA_CONNECTOR_MODULE \
  pnpm vitest run \
  src/tools/internal/internal-tools.test.ts \
  src/tools/internal/open-questions-ruminations.test.ts \
  src/tools/internal/own-records-list.test.ts \
  src/tools/internal/inner-life-tools.test.ts \
  src/memory/semantic/extractor.test.ts \
  src/memory/self/open-questions.test.ts \
  src/tools/internal/goals-block.test.ts \
  src/tools/internal/goals-retire.test.ts \
  src/borg/tools-setup.test.ts \
  src/cognition/generation/post-generation-guards-shadow.test.ts \
  src/cognition/prompts/prompt-surface-fixtures.test.ts \
  --maxWorkers=2 --cache=false --configLoader=native
```

Open questions: none. Reference existence remains distinct from semantic evidence
judgment, and disposition reads depend on the durable audit events already written
by production repositories.
