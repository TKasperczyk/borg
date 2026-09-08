Finalizer cache tiers, 2026-09-06 — full-turn caching and compact projections

Opus 5.0: in-scope because cache boundaries, excerpt inputs and repeated structural attributes are harness presentation concerns.

The compact finalizer now emits exactly four system blocks, each with `cache_control.type="ephemeral"`. The breakpoint/TTL sequence is:

| System block | Trace tier | Contents | TTL |
| --- | --- | --- | --- |
| 0 | terminal_durable_global | Static framing + durable global | 1h |
| 1 | terminal_durable_audience | Durable audience | 1h |
| 2 | terminal_slow_turn | Standing memory + relative-age overlay | 5m |
| 3 (last) | terminal_fast_turn | Fast turn context, including any regeneration suffix | 5m |

The earlier five-block design spent both 5m markers on separate slow sections and left fast text uncached. The corrected fourth marker covers the full fast tier, allowing later finalizer rounds with an unchanged system prompt to reuse it. Standing and overlay share the third marker. A stored-state or ledger-projection change in either invalidates the combined slow prefix. Every tier is freshly rendered; caching never substitutes stale rows. Regeneration suffix bytes are appended verbatim inside the last block, and block/section/total telemetry includes them.

Fast projections use one line per record ID with tab-separated JSON cells. Column names and meanings appear once per table; a bare `-` distinguishes absent fields from explicit JSON `null`. Cells are XML-escaped as text, so names containing tabs, newlines, quotes or tag-like text cannot change table structure. Shared ledger attributes still appear once in the group header. The tables cover relational turn fields, commitment entity labels and self-record counters. Clock/roster fields are partitioned before slow excerpts are computed, and slow relational rows retain stable ID order; fast ordinal preserves the original ledger order.

Rows containing names, values or additional source/audience handles retain complete disclosure cells: `[disclosure_class, origin_audience, private-to, public-to]`, including every audience ID and fail-closed unknown labels. Only projections containing the record ID plus numeric/enumerated fields omit a duplicate disclosure cell and join the labeled slow row. The heuristics guard passes without changes or exemptions. Membership, stored rows, exact directives and excerpt budgets are unchanged.

Measurements use the same six frozen capture inputs as the prior report. The selected same-session pair is A = `eb377199-01c5-413e-9850-460fd1728e8b`, B = `2ee890e6-93d5-4eee-9436-4aaea8f101ec`. Chars are JavaScript string lengths. Before is `8ccab397` (renderer `615849d3`); after is this correction.

| Tier | Before A | Before B | After A | After B |
| --- | ---: | ---: | ---: | ---: |
| Static + global | 284,715 | 284,715 | 284,715 | 284,715 |
| Audience | 126,913 | 126,913 | 126,913 | 126,913 |
| Slow | 33,712 standing + 89,553 overlay | 33,712 standing + 89,553 overlay | 123,302 combined | 123,302 combined |
| Fast | 805,855 uncached | 852,296 uncached | 767,446 cached | 813,887 cached |
| Total including block separators | 1,340,756 | 1,387,197 | 1,302,382 | 1,348,823 |

The original pre-change four-block renderer (`719fbeca`) produced static 27,097, global 257,527, audience 126,913 and turn 890,781 / 937,222 chars on A / B. Both the original renderer and `913fbdff` were re-rendered for this report to make the baseline explicit:

| Total comparison | A | B |
| --- | ---: | ---: |
| Original pre-change (`719fbeca`) | 1,302,324 | 1,348,765 |
| Initial split (`913fbdff`) | 1,298,439 | 1,344,880 |
| After this correction | 1,302,382 | 1,348,823 |
| Delta vs original | +58 (+0.0045%) | +58 (+0.0043%) |
| Delta vs initial split | +3,943 (+0.304%) | +3,943 (+0.293%) |

Both captures meet the requested +2% limit against either baseline. Total text falls by 38,374 chars per prompt versus the reviewed five-block version. The separate user capture `df50268b-8001-4c6c-a827-16678d0c104f`, which previously showed +117,150 chars versus `913fbdff`, now totals 1,093,159: +30,635 (+2.88%) versus that initial split, or +7,593 (+0.70%) versus the original four-block renderer. The +2% target is met on the requested pair; that separate capture remains above +2% of the initial split.

The existing capture stability report with `--same-session` found two pairs. For A → B:

| Emitted block | Common UTF-8 prefix bytes | Byte-stable |
| --- | ---: | --- |
| Static + global | 285,520 (entire block) | Yes |
| Audience | 127,026 (entire block) | Yes |
| Combined slow | 123,325 (entire block) | Yes |
| Fast | 270 | No |

The cumulative prefix through breakpoint 3 is identical. The production-assembly regression also advances the actual rendered clock by three seconds across an age-length boundary, changes speaker, display name and roster order, and verifies an identical combined slow block with a differing fast block. A separate fallback-label test uses non-null commitment entity references. Five successive mocked autonomous requests with the same turn context have identical four-block system content and retain the final 5m marker. Fingerprint tests detect both fast-content changes and removal of its marker; finalizer capture/replay round-trips all four blocks. A/B neutralization includes the new tier/table names and retains historical names.

The second captured pair, B → `eca5f48a-f315-4147-add3-b4b19e209572`, has stable standing rows but changed commitment ledger scope/trust. Its combined slow block changes from 123,302 to 123,309 chars, with 80,956 common UTF-8 bytes. This correctly invalidates the combined slow breakpoint; the first two tiers remain identical. This is the accepted tradeoff for retaining a fast-tier breakpoint.

An independent mechanical audit compared all six before/after renders. All 1,540 slow rows were unchanged, including state, payload and labels. All 322 relational projections preserved moved fields and original order; 102 ID/numeric/enum-only rows omitted duplicate labels, while all content-bearing projections retained their full labels. All 1,026 commitment-label rows and 192 self-counter rows round-tripped exactly by ID. Trace row totals matched on every capture. No membership or upstream selection changed.

Replays use read-only `demo/server/.borg-data/demo/captures/finalizer-contexts.jsonl`, freezing the first six records into the worktree. Archived surfaces predate the baseline renderer, so comparisons re-render identical projected context, evidence ledger, static head and additional sections with retrieval/semantic budgets 32,000/32,768 and each captured clock. They measure presentation and request shape, not live token usage, cache hits or latency. No live LLM or SQLite access was used. Temporary private captures and historical renderer copies were removed before commit.

Files changed:

- `src/cognition/deliberation/prompt/finalizer-context.ts`: four cached tiers and compact disclosed projection tables.
- `src/cognition/deliberation/finalizer.ts`: cached regeneration suffix and accurate tier/total telemetry.
- `src/cognition/deliberation/finalizer-ab-judge.ts`: neutralization for the combined tier and table names.
- Context, finalizer, fingerprint, finalizer capture/replay and cache-report tests; compact prompt-surface fixture.
- `docs/finalizer-cache-tiers.md`: current layout, measurements and validation.

`request-fingerprint.ts` already fingerprints actual content and cache controls; planner-context capture delegates to it. Both remain schema-compatible with historical layouts, with their tests included in validation.

Commands and final results (existing dependency symlinks; no install):

- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm typecheck`: passed all five root-script tsconfigs.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm heuristics:guard`: passed unchanged, including disclosure-label checks.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm vitest run src/cognition/deliberation/prompt/finalizer-context.test.ts src/cognition/deliberation/finalizer.test.ts src/cognition/deliberation/finalizer-context-capture.test.ts src/cognition/deliberation/request-fingerprint.test.ts src/cognition/deliberation/planner-context-capture.test.ts src/cognition/deliberation/finalizer-ab-judge.test.ts scripts/finalizer-cache-report.test.ts scripts/finalizer-ab-judge.test.ts src/cognition/prompts/prompt-surface-fixtures.test.ts src/cognition/evidence-ledger/builder.test.ts --maxWorkers=2 --cache=false --configLoader=runner`: 10 files, 202 tests passed.
- Fixture update: `choom -n 800 -- env TSX_DISABLE_CACHE=1 UPDATE_PROMPT_SURFACE_FIXTURES=1 pnpm vitest run src/cognition/prompts/prompt-surface-fixtures.test.ts --maxWorkers=2 --cache=false --configLoader=runner -t 'pins compact finalizer cache tiers'`: one selected test passed.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm exec tsx .finalizer-cache-revision/measure.ts original`, repeated with `split`, `before` and `after`: six identical captured inputs rendered on each version.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm --silent finalizer:cache-report .finalizer-cache-revision/after.jsonl --same-session`: passed; two same-session pairs.
- `choom -n 800 -- python3 .finalizer-cache-revision/audit.py`: all row/label/field comparisons passed; four markers in the required TTL order.
- Prettier on changed TypeScript files and `git diff --check`: passed.

Every Vitest run used `--maxWorkers=2 --cache=false --configLoader=runner`; TSX caching was disabled. No live source, dependency or data files were changed.
