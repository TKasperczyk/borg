Finalizer cache tiers, 2026-09-06 — reviewed follow-up

Opus 5.0: in-scope because cache boundaries, excerpt inputs and repeated structural attributes are harness presentation concerns.

The compact finalizer sends five system blocks with exactly four cache breakpoints. Static framing shares the durable-global 1h breakpoint. Audience context retains its 1h breakpoint. The slow tier has two 5m breakpoints, standing first and relative-age overlay second, so an overlay change can reuse the standing prefix. Fast turn context and regeneration instructions follow the last breakpoint without cache control.

The review of `913fbdff` found that relational standing excerpts still depended on assembled relative ages, subject roles and participant names. Even an age hidden in the elided middle changed the excerpt's source-length marker. Commitment entity labels also fell back to current participant/sender/audience context inside the slow overlay. The old test changed `context.nowMs` while a higher-precedence prompt option pinned the rendered clock; it did not establish clock stability. Replaying frozen ledger entries alone also cannot detect reassembly dependencies.

Relational metadata is now partitioned before serialization and excerpting. Slow rows contain stored state and use ID order. Fast `relational_standing_turn_row` records join by ID and retain original ledger order as `ordinal`, clock-relative ages, assembled subject identity/name/role, current audience metadata, participant-name value projections, and ledger projection attributes. Shared structural attributes are hoisted separately into each tier's interpretation header only when every row has the exact same value. Fast projection fields cannot affect slow excerpt lengths. No row membership or upstream selection changes.

All four assembled commitment entity labels now live in fast `commitment_entity_labels` rows joined to canonical or ledger-only overlay rows by commitment ID. Each carries the same fail-closed resolved disclosure as its overlay row. Relational slow and fast rows likewise retain identical, complete disclosure labels; private labels and origin audiences are never hoisted. Stored text keeps its existing excerpt budget, and critical canonical directives remain exact. Fast metadata carries the moved fields in full.

Chars below are JavaScript string lengths from frozen user capture `df50268b-8001-4c6c-a827-16678d0c104f`. The first comparison is the previous same-input replay at `913fbdff`; the second is the reviewed fix at `615849d3`.

| Tier | Cache | Before review fix | After review fix |
| --- | --- | ---: | ---: |
| Static + durable global | 1h | 284,228 | 284,228 |
| Durable audience | 1h | 59,798 | 59,798 |
| Slow standing | 5m | 219,067 | 204,646 |
| Slow overlay | 5m | 116,563 | 89,580 |
| Fast turn | Uncached | 382,860 | 541,414 |

The original four-block baseline (`719fbeca`) was static 26,610, durable global 257,527, durable audience 59,798 and turn 741,625 chars. Its two stable global blocks were merged and the turn block split as above.

The review fix reduces the two slow blocks from 335,630 to 294,226 chars, down 41,404. Total system text including separators grows from 1,062,524 to 1,179,674 chars, up 117,150: ID-keyed projections repeat required disclosure labels and expose the moved metadata in full. The uncached tail grows by 158,554 chars. This fixes cache invalidation rather than reducing total input size. Actual token counts, cache hits and latency require a subsequent live run; these are renderer measurements.

Measurement used the first six records frozen from the read-only `demo/server/.borg-data/demo/captures/finalizer-contexts.jsonl`. The archived surfaces predate the baseline renderer and are not byte-for-byte reproductions of its output. Replays use the captured projected context, evidence ledger, static head and captured additional sections, retrieval/semantic budgets fixed at 32,000/32,768, and each captured clock. No live model call or SQLite access was required. Private source captures and measurement helpers were temporary worktree artifacts and are not committed.

The existing cache report with `--same-session` found two consecutive pairs within those six captures. For `eb377199-01c5-413e-9850-460fd1728e8b` → `2ee890e6-93d5-4eee-9436-4aaea8f101ec`, after the fix:

| Tier | Previous chars | Current chars | Common UTF-8 prefix bytes | Byte-stable |
| --- | ---: | ---: | ---: | --- |
| Static + durable global | 284,715 | 284,715 | 285,520 (entire block) | Yes |
| Durable audience | 126,913 | 126,913 | 127,026 (entire block) | Yes |
| Slow standing | 33,712 | 33,712 | 33,735 (entire block) | Yes |
| Slow overlay | 89,553 | 89,553 | 89,553 (entire block) | Yes |
| Fast turn | 805,855 | 852,296 | 270 | No |

All four cached blocks, and therefore the cumulative prefixes through both slow breakpoints, are identical in this pair. In the second pair (`2ee890e6-93d5-4eee-9436-4aaea8f101ec` → `eca5f48a-f315-4147-add3-b4b19e209572`), standing remains identical at 33,712 chars, but overlay changes from 89,553 to 89,560 chars with 47,184 common bytes. One commitment's `ledger_scope` and `ledger_trust_rank` changed. Those projection fields remain in the slow overlay and still invalidate that breakpoint; they do not invalidate the preceding standing breakpoint. Exact record timestamps and state also remain there and must reflect actual changes.

A mechanical capture audit checked all 322 relational rows across six inputs against their slow and fast ID sets, original order, states, metadata partition and identical disclosure labels. All 1,026 commitment-label rows matched their overlay IDs and resolved disclosures. Every replay retained five blocks and four cache markers. The new regression test calls the production standing assembler afresh for unchanged stored slots, goals and commitments, advances time by three seconds across an age-length boundary, and changes the speaker, roster order and display name. It asserts changed rendered `current_time_ms`, changed source metadata/excerpts, byte-identical slow blocks and differing fast blocks. A separate test covers non-null commitment entity references with missing label-map entries, including canonical and fail-closed ledger-only rows.

Files changed in this follow-up:

- `src/cognition/deliberation/prompt/finalizer-context.ts`: partition before excerpts; stable row order; disclosed fast projections and entity labels; shared attributes retained per tier.
- `src/cognition/deliberation/prompt/finalizer-context.test.ts`: real clock propagation, production-assembled standing regression, entity-label fallback and disclosure/attribute coverage.
- `src/cognition/prompts/__fixtures__/prompt-surface/finalizer-system-blocks-s2-compact.txt`: updated compact surface fixture.
- `docs/finalizer-cache-tiers.md`: corrected stability claims and updated measurements without host-specific paths.

Tier bookkeeping in `finalizer.ts`, request fingerprints, planner/finalizer captures and A/B tooling remains on the five-block/four-marker layout. Their tests are included below.

Commands and results (existing dependency symlinks; no install):

- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm typecheck`: passed all five root-script tsconfigs.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm heuristics:guard`: passed, including same-object disclosure-label checks.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm vitest run src/cognition/deliberation/prompt/finalizer-context.test.ts src/cognition/deliberation/finalizer.test.ts src/cognition/deliberation/finalizer-context-capture.test.ts src/cognition/deliberation/request-fingerprint.test.ts src/cognition/deliberation/planner-context-capture.test.ts src/cognition/deliberation/finalizer-ab-judge.test.ts scripts/finalizer-cache-report.test.ts scripts/finalizer-ab-judge.test.ts src/cognition/prompts/prompt-surface-fixtures.test.ts src/cognition/evidence-ledger/builder.test.ts --maxWorkers=2 --cache=false --configLoader=runner`: 10 files, 201 tests passed.
- Fixture update: `choom -n 800 -- env TSX_DISABLE_CACHE=1 UPDATE_PROMPT_SURFACE_FIXTURES=1 pnpm vitest run src/cognition/prompts/prompt-surface-fixtures.test.ts --maxWorkers=2 --cache=false --configLoader=runner -t 'pins compact finalizer cache tiers'`: one selected test passed.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm exec tsx .finalizer-cache-review/measure.ts`: six captured inputs rendered; per-tier sizes and byte comparisons recorded.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm --silent finalizer:cache-report .finalizer-cache-review/after.jsonl --same-session`: passed, two same-session pairs reported.
- `choom -n 800 -- python3 .finalizer-cache-review/audit.py`: all 322 relational rows and 1,026 commitment-label joins passed the checks above.
- Prettier on changed TypeScript files and `git diff --check`: passed.

Every Vitest invocation in this follow-up used `--configLoader=runner` as well as `--cache=false`, avoiding the shared-cache config-loader write encountered during the original commit. No live source, dependency or data files were changed during this fix.
