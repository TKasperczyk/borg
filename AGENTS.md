# AGENTS.md -- Conventions for Code Agents (Claude, Codex, etc.)

## Project

**Borg** is a cognitive memory architecture for autonomous AI beings. It is a
TypeScript library (plus thin CLI and optional headless daemon) that provides
primitives for agent memory, cognition, and evolving identity.

The code is the authoritative reference for the memory bands, cognitive loop,
offline processes, and retrieval pipeline. Start from `src/borg/facade.ts`
and follow the imports.

## Stack

- **Runtime:** Node >= 22, ESM-only (`"type": "module"`)
- **Language:** TypeScript, strict mode, `moduleResolution: "Bundler"`
- **Vectors:** LanceDB (`@lancedb/lancedb`) embedded, per-table
- **Structured state:** SQLite (`better-sqlite3`) for goals, commitments,
  graph edges, skills, stats
- **Stream log:** JSONL atomic append on disk
- **LLM:** Anthropic SDK (`@anthropic-ai/sdk`). The cognition,
  extraction, and background slots default to Opus 4.7. borg runs
  under OAuth subscription -- there is no per-token cost to optimize
  for, so we pay the latency for consistent quality across reasoning-heavy
  work. Recall expansion has its own `recallExpansion` slot, defaulting
  to Haiku, because it is a small structured fanout task rather than a
  background reasoning pass.
- **Embeddings:** OpenAI-compatible endpoint (default LM Studio, qwen3-embedding-8b,
  4096 dims).
- **Validation:** Zod.
- **CLI:** `cac`.
- **Tests:** Vitest.
- **Build:** tsup.

## Architectural invariant: LLM-first interpretation

Borg may use deterministic code to **move already-known source handles
around**. Borg may not use deterministic code to **interpret language**.

This rule applies to `src/cognition/`, `src/retrieval/`, `src/memory/`,
`src/offline/`, and `src/simulator/overseer*` whenever code is inferring:

- entities, topics, relationships, facts
- intent, salience, memory relevance
- topic continuity, corrections, belief changes
- user / audience identity

Interpretation goes through LLMs.

### Prohibited in semantic / interpretive paths

- regex over user-authored text
- `.includes()` / `.indexOf()` / `.startsWith()` / `.endsWith()` for matching
- string splits or tokenization on user content
- capitalization heuristics (`\p{Lu}`, `[A-Z]`, `toUpperCase`)
- wordlists / `Set`s of phrase patterns
- hardcoded topic / entity / relationship labels
- n-grams, token-shape inference for entity extraction
- hand-rolled topic-fingerprinting or change-detection logic
- substring or lexical matching that decides whether two records
  "are about the same thing"

### Acceptable -- mechanical parsing only

Regex and structural code are fine for non-interpretive parsing:

- ID validation: `/^ep_[a-z0-9]+$/.test(episodeId)`
- config / env value validation
- machine-generated tags
- log line splitting
- migration helpers
- test snapshot normalization
- protocol-level formatting

The test: are you parsing **machine-generated structure** (allowed) or
trying to decide **what the user meant** (forbidden)?

### Why

Every language-specific heuristic embeds assumptions that fail across user
populations. A regex like `/[A-Z][a-z]+/` extracts English-style names but
misses Chinese names. A wordlist like `{"thanks", "thank you"}` catches
English gratitude but not French, Spanish, Japanese. The
`pnpm heuristics:guard` CI catches known patterns, but it is reactive --
new variants slip through.

LLM interpretation handles every language with the same code. The latency
cost is acceptable under our OAuth subscription. Failure modes are
explicit (degrade-with-observability via `onDegraded` hooks) rather than
silent wrong answers for half the user population.

The Maya gaslight scenario surfaced this concretely: perception's LLM-only
entity extractor missed "Maya" in a multi-topic message, and Borg
capitulated. The fix was a second LLM call (recall expansion) that emits
explicit `named_terms`, plus a deterministic union with the perception
output -- moving already-LLM-identified handles around, not interpreting
language deterministically. Reaching for regex would have just shifted the
bug from English to non-English users.

## Production posture (post v82-v87 arc)

The v82-v87 development arc closed out correctness, hygiene, and per-turn stream scaling. The remaining notes here document architectural invariants and operational caveats that landed during that arc and that callers should understand.

### `StreamEntryIndex` is a production invariant, not an optimization

After v86 P8 and v87 P0-P3, the index participates in:

- source-trust validation (`lookupEntriesById`)
- prior user-turn count (`countSessionEntriesByKind`)
- cross-session quarantine refs (`quarantinedSharedStateArtifactRefs`)
- citation status-marker filtering (`lookupSessionEntriesByKind`)
- active/inactive entry facts (the `active` column)
- the post-generation guard's transcript window (no -- that one uses scanReverse, not the index)

Production Borg must construct `StreamWriter` and the retrieval/turn coordinator with the shared `StreamEntryIndexRepository`. The no-index fallbacks in retrieval and citations are for test setups and custom harnesses only. They preserve correctness but lose the scaling guarantees.

### Stream append and index update are not atomic

`StreamWriter` writes the JSONL stream first (fsync'd) and then updates SQLite. If the SQLite update fails after the append commits, the writer throws `StreamError` with code `STREAM_INDEX_UPDATE_FAILED`. The startup backfill (`reconciliation.ts`) repairs the index by re-reading committed stream files.

Operationally: a `STREAM_INDEX_UPDATE_FAILED` is an index-consistency incident, not a normal soft degradation. The committed turn is on disk; the failed turn should be treated as needing a backfill before the next turn runs.

### Citation status-marker lookup is index-backed, not constant-time

`readStatusMarkerEntries` no longer reads the whole session stream file. It queries the `(session_id, kind)` index for `internal_event` rows and reads only those entries by byte offset. That's a large improvement and the per-turn cost stays bounded with marker volume, not session length.

If marker density ever becomes a hot path, a dedicated `inactive_refs` / `status_refs` table parallel to `stream_quarantine_refs` would let the citation filter ask `inactiveRefsForSession(sessionId)` directly. Filed as future work, not closeout-critical.

### Cap telemetry: nonzero cap-hit is a review trigger

`EvidenceLedgerBuilder` uses `scanReverse({ maxEntries: 1024, maxBytes: 8MiB })` and emits `ledger_reverse_scan_entry_cap_hit_total` / `_byte_cap_hit_total` cumulative counters plus the per-turn `evidence_ledger.reverse_scan` trace event.

Through v87.1 family + compaction, both cap-hit counters stayed at 0 (14 entries/turn avg in family, 33/turn in compaction). The cap-hit branches are unit-tested but never sim-exercised.

Operational rule: nonzero cap-hit is not automatically bad, but it should trigger review of whether the ledger is leaning on recent transcript instead of retrieval / shared-state / durable memory.

### Sparse indexed source-trust facts are not a turn sequence

`compileSharedStateArtifactForEvidenceLedgerResult()` builds `lastUpdatedSourceTrustEntries` for the indexed source-trust path. That input is now scoped to the current user entry only -- sparse indexed facts (visible ledger IDs, off-limits IDs, relational-slot evidence IDs, prior-session source IDs) are valid for trust allow/reject lookups but NOT for reconstructing a turn-age sequence.

The principle from v86 P0 still applies: durable `last_updated_turn_global` is the only authoritative source of shared-state age. Legacy or in-flight entries with null durable age stay "unknown age" by design -- they do not get a pseudo-age inferred from sparse indexed facts.

### Backlog (not closeout blockers)

These are real maintainability items but do not extend the v82-v87 arc:

- split `src/retrieval/pipeline.ts` (2200+ lines)
- split `simulator/metrics.ts` (4365 lines)
- migration helper or test/checklist for `_next` table swaps
- broader pure schema/type extraction beyond the v89 public-consumer cleanup
- possible dedicated `StreamFactsRepository` shell that wraps `StreamEntryIndex`

Future cycles may pull any of these, but none should be treated as required.

## Conventions

### File layout

```
src/
  stream/         append-only JSONL log
  memory/
    episodic/
    semantic/
    procedural/
    affective/
    self/         Self-band data (values, goals, traits, autobiographical, ...)
    identity/     governance over identity-bearing mutations
    commitments/
    social/
    working/
    common/       shared memory primitives (provenance, identity-events, ...)
  cognition/      perception, attention, deliberation, turn-action, reflection
  offline/        consolidator, reflector, curator, overseer, ruminator,
                  self-narrator, procedural-synthesizer, belief-reviser
  retrieval/      unified context-aware retrieval pipeline
  correction/     `borg.correction.forget` / why / invalidate-edge service
  executive/      executive focus selection (goal stickiness, step rendering)
  autonomy/       autonomy scheduler + wake-source triggers
  auth/           Claude Code OAuth credential helpers
  tools/          internal tool dispatcher (episodic.search, semantic.walk, ...)
  storage/        lancedb + sqlite abstractions
  embeddings/     embedding client
  llm/            Anthropic client wrapper
  config/         config loader
  util/           cross-cutting helpers (atomic file ops, ids, clocks, ...)
  borg/           composition root (open.ts), facade, lifecycle, repositories
  cli/            `borg` CLI entry
  index.ts        library entry
scripts/
  daemon.ts       headless daemon helper (not part of the shipped CLI surface)
  chat.ts         developer-only interactive REPL
```

Tests co-located with source as `*.test.ts`.

### Naming

- Files: `kebab-case.ts`
- Types/interfaces: `PascalCase`
- Functions/variables: `camelCase`
- Constants: `UPPER_SNAKE_CASE` only for true compile-time constants
- Modules export named symbols; avoid default exports

### Types

- Prefer types over interfaces for data shapes
- Prefer Zod schemas at I/O boundaries (stream entries, config, tool args),
  derive TS types from schemas via `z.infer`
- Use branded types (`type EpisodeId = string & { __brand: "EpisodeId" }`) for
  IDs that should not be mixed

### Error handling

- Throw typed errors (subclasses of a project `BorgError` base)
- Validate at system boundaries only; trust internal code
- No swallowing errors; log + rethrow or surface to caller
- Async functions should always be `async/await`, not `.then` chains

### Persistence

- **Atomic writes everywhere.** Use temp file + fsync + rename pattern
- **Crash-safe.** Append-only logs preferred over in-place updates where possible
- **Citation anchors.** Every derived memory carries the source stream IDs that
  produced it
- **Provenance + confidence.** Every claim carries `confidence` and `source_*`
  fields -- no silent trust

### Testing

- Vitest. Unit tests co-located.
- Use a temp dir for any filesystem-touching test (`mkdtempSync`, clean up in `afterEach`)
- Mock Anthropic and embedding calls; do not hit the real API in tests
- Aim for fast tests; mark slow integration tests explicitly

### Dependencies

- Before adding a new dependency, check if an existing one covers the use case
- Prefer small, focused libraries
- Document non-obvious choices in the module README or a code comment

### No MCP, no shipped interactive TUI

Borg is a library. The CLI is a thin operational shell. There is no MCP server
and no shipped interactive TUI -- keep the library and CLI free of those
concerns. `scripts/chat.ts` is a developer-only helper for local interactive
sessions, not part of the distributed CLI surface.

## Common operations

- Run one test file: `pnpm vitest run path/to/file.test.ts`
- Typecheck: `pnpm typecheck`
- Format: `pnpm format`
- Build: `pnpm build`

## When implementing a sprint

1. Read the relevant module to understand its current shape before changing it.
2. Before writing a new utility, search for an existing one.
3. Match the existing patterns (naming, error handling, file layout).
4. Add tests for new behavior.
5. Keep scope tight -- don't ship unrelated refactors.
