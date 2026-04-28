# AGENTS.md -- Conventions for Code Agents (Claude, Codex, etc.)

## Project

**Borg** is a cognitive memory architecture for autonomous AI beings. It is a
TypeScript library (plus thin CLI and optional headless daemon) that provides
primitives for agent memory, cognition, and evolving identity.

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design. It is the
authoritative reference for the current memory bands, cognitive loop, offline
processes, and retrieval pipeline.

## Stack

- **Runtime:** Node >= 22, ESM-only (`"type": "module"`)
- **Language:** TypeScript, strict mode, `moduleResolution: "Bundler"`
- **Vectors:** LanceDB (`@lancedb/lancedb`) embedded, per-table
- **Structured state:** SQLite (`better-sqlite3`) for goals, commitments,
  graph edges, skills, stats
- **Stream log:** JSONL atomic append on disk
- **LLM:** Anthropic SDK (`@anthropic-ai/sdk`). All three slots
  (cognition, extraction, background) default to Opus 4.7. borg runs
  under OAuth subscription -- there is no per-token cost to optimize
  for, so we pay the latency for consistent quality across the stack.
  The three slots still exist in config so deployments CAN split if
  they ever need to.
- **Embeddings:** OpenAI-compatible endpoint (default LM Studio, qwen3-embedding-8b,
  4096 dims).
- **Validation:** Zod.
- **CLI:** `cac`.
- **Tests:** Vitest.
- **Build:** tsup.

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
  cognition/      perception, attention, deliberation, action, reflection
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

1. Read `ARCHITECTURE.md` for what the sprint is building.
2. Before writing a new utility, search for an existing one.
3. Match the existing patterns (naming, error handling, file layout).
4. Add tests for new behavior.
5. Keep scope tight -- don't ship unrelated refactors.
