Finalizer cache tiers, 2026-09-06

Opus 5.0: in-scope because cache boundaries and repeated structural attributes are harness presentation concerns.

The compact finalizer now sends five system blocks with exactly four cache breakpoints. Static framing shares the durable-global 1h breakpoint. Audience context retains its 1h breakpoint. The slow tier has two 5m breakpoints, standing first and relative-age overlay second, so an overlay change can reuse the standing prefix. Fast turn context and regeneration instructions follow the last breakpoint without cache control.

Chars below are JavaScript string lengths from one frozen user capture, `df50268b-8001-4c6c-a827-16678d0c104f`, rendered with the baseline (`719fbeca`) and changed renderer using identical inputs and options.

| Before block | Chars | Cache | After block | Chars | Cache |
| --- | ---: | --- | --- | ---: | --- |
| Static | 26,610 | 1h | Static + durable global | 284,228 | 1h |
| Durable global | 257,527 | 1h | Durable audience | 59,798 | 1h |
| Durable audience | 59,798 | 1h | Slow standing | 219,067 | 5m |
| Turn | 741,625 | 5m | Slow overlay | 116,563 | 5m |
| | | | Fast turn | 382,860 | Uncached |

Total system text including block separators: 1,085,566 → 1,062,524 chars, down 23,042. The 741,625-char turn prefix is replaced by 335,630 chars of reusable slow context; 382,860 fast chars are outside cache creation. Actual token counts, cache hits, latency and billing require a subsequent live run; these are presentation measurements.

All 180 relational standing rows remain. Their row attributes total 243,177 → 217,077 chars. Including its shared interpretation header, the relational group is 244,234 → 218,653 chars. Shared structural fields are hoisted only when every row has the exact same rendered value. Differing structural fields, IDs, states, text/value/metadata, complete disclosure labels and origin audiences stay on individual rows. No membership selection or excerpt budget changed.

Measurement used the first six records frozen from the read-only `demo/server/.borg-data/demo/captures/finalizer-contexts.jsonl` in the live checkout. The archived surfaces predate the baseline renderer and are not byte-for-byte reproductions of its output. Both comparison sides were therefore rendered from the same captured projected context, evidence ledger, static head and captured additional sections, with retrieval/semantic budgets fixed at 32,000/32,768 and each captured clock. This isolates this change from intervening prompt changes; it does not rerun the user's separate 34-call trace census. Private source captures were temporary worktree artifacts and are not committed.

The existing cache report, extended with `--same-session`, found two consecutive pairs within those six captures. For `eb377199-01c5-413e-9850-460fd1728e8b` → `2ee890e6-93d5-4eee-9436-4aaea8f101ec`:

| Surface | Previous chars | Current chars | Common UTF-8 prefix bytes |
| --- | ---: | ---: | ---: |
| Before turn | 890,781 | 937,222 | 270 |
| After slow standing | 33,298 | 33,298 | 33,321 (entire block) |
| After slow overlay | 116,536 | 116,536 | 116,536 (entire block) |
| After fast turn | 736,969 | 783,410 | 270 |

An independent mechanical comparison expanded the shared relational attributes and joined moved self counters by ID: all 3,362 rows checked across ten row tags in six captures retained every original attribute and their order. Trace row counts also matched. The unit stability test changes the clock, roster, cadence, working state, ledger estimate and self record counters, then verifies both slow blocks remain identical and the fast block differs. Row-state and exact-timestamp changes correctly invalidate the affected slow block.

Fields kept out of slow prefixes: current clock, tool/action availability, sender/roster/authority, session state, evidence ledger, mechanism evidence, working state, goal/lived-experience digests, social and cross-session entries. Raw relational slots retain their changing `updated_age` and assembled subject labels in fast context. Group counts, draw scope and cadence also stay fast; self record versions and support/contradiction counters use ID-keyed fast rows. Exact record timestamps and commitment ledger projections remain in the freshly rendered slow overlay and invalidate it whenever their values change. Disclosure labels and private metadata were unsafe to hoist and remain on each private-bearing row, including fail-closed unknown labels.

Changed files:

- `src/cognition/deliberation/prompt/finalizer-context.ts`: tier assembly, counter placement, relational row diet and truthful trace totals.
- `src/cognition/deliberation/finalizer.ts`: legacy tier report follows actual markers, without fictitious empty tiers.
- `src/cognition/deliberation/finalizer-ab-judge.ts`: neutralizes new presentation labels, retaining historical labels.
- `scripts/finalizer-cache-report.ts`: optional same-session comparisons, including user turns.
- Co-located finalizer, context, capture, fingerprint, A/B and cache-report tests; `prompt-surface-fixtures.test.ts` and `finalizer-system-blocks-s2-compact.txt`.

`request-fingerprint.ts` already counts actual cache markers independently of system blocks; `planner-context-capture.ts` delegates to it. Tests verify five blocks/four markers through both, preserve the uncached tail in fingerprints, and round-trip/replay the new captured layout without changing capture schemas.

Commands and results (no dependency installation or live LLM calls):

- Required `ln -s /home/luth/Programming/borg/node_modules node_modules && ln -s /home/luth/Programming/borg/demo/server/node_modules demo/server/node_modules`: passed.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm typecheck`: passed, all five root-script tsconfigs. Initial test-only typing errors were fixed before the passing run.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm heuristics:guard`: passed, including disclosure-label checks.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm vitest run src/cognition/deliberation/prompt/finalizer-context.test.ts src/cognition/deliberation/finalizer.test.ts src/cognition/deliberation/finalizer-context-capture.test.ts src/cognition/deliberation/request-fingerprint.test.ts src/cognition/deliberation/planner-context-capture.test.ts src/cognition/deliberation/finalizer-ab-judge.test.ts scripts/finalizer-cache-report.test.ts scripts/finalizer-ab-judge.test.ts src/cognition/prompts/prompt-surface-fixtures.test.ts --maxWorkers=2 --cache=false --configLoader=runner`: 9 files, 142 tests passed.
- Fixture update: the same Vitest options with `UPDATE_PROMPT_SURFACE_FIXTURES=1`, restricted to `pins compact finalizer cache tiers`; passed.
- Temporary worktree measurement: `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm exec tsx .finalizer-cache-analysis/measure.ts before` and `after`; six captured inputs rendered on each side. `choom -n 800 -- python3 .finalizer-cache-analysis/validate.py`: all 3,362 checked rows preserved.
- `choom -n 800 -- env TSX_DISABLE_CACHE=1 pnpm finalizer:cache-report .finalizer-cache-analysis/before.jsonl --same-session`, repeated for `after.jsonl`: passed, two same-session pairs each.
- Prettier on changed TypeScript files and `git diff --check`: passed.

Workspace constraint incident: the initial Vitest invocation used Vite's default bundled config loader, which briefly wrote and removed its config under the shared `node_modules/.vite-temp` despite `--cache=false`. Subsequent runs used `--configLoader=runner` to prevent this. No live source or live data files were changed; no live SQLite database was opened.
