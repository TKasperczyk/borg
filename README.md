# Borg

Cognitive memory architecture for autonomous AI beings.

A TypeScript library, CLI, and optional headless daemon for building agents
with persistent memory, explicit cognition, and identity that evolves over
time.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the design document.

## Status

Feature-complete against the design in `ARCHITECTURE.md` (18 sprints delivered).
Not yet used in anger. No stability guarantees; schemas, storage layout, and
public API may still shift as the library gets exercised.

## What it does

- **Append-only stream.** JSONL log of every user/agent message,
  perception, thought, tool call/result, and internal event. The stream
  is an audit log and extraction input; authoritative state lives in
  the typed repositories (SQLite rows + LanceDB vectors).
- **Eight memory bands.** Episodic (what happened), semantic (what I know),
  procedural (how I do things, as Bayesian skills), affective (how I felt / feel),
  self (values, goals, traits, autobiographical arc, growth markers,
  open questions), commitments (scoped promises + guard), social (per-person
  trust/history), plus ephemeral working memory per turn.
- **Explicit cognitive loop.** Perception → Attention → Deliberation → Action →
  Reflection. System 1 / System 2 branching with internal monologue persisted
  as stream entries.
- **Graph-aware retrieval.** Typed semantic edges (supports, contradicts,
  causes, prevents, is_a, ...) walked during retrieval. Mode-conditioned
  weights (problem-solving, relational, reflective, idle). Mood-congruent
  ranking when mood is non-neutral. Reason-tagged per-session suppression.
- **Commitment awareness, not filter.** Rules injected into the prompt so
  the agent knows them before speaking. Post-hoc check with Sonnet revision,
  refusal-aware (compliant refusals don't falsely trigger).
- **Offline dream cycle.** Eight processes (consolidator, reflector, curator,
  overseer, ruminator, self-narrator, procedural-synthesizer, belief-reviser)
  with plan/apply parity, budget caps, append-only audit log, and reversible
  maintenance where a reverser is registered (a few destructive prunes of
  transient observability data, e.g. retrieval-log trim, are recorded as
  `no_reverser` audit rows).
- **Bayesian procedural memory.** Skills are Beta(α, β) posteriors; Thompson
  sampling selects an approach; outcomes update the posterior atomically.

Currently 778 tests across 120 files, typecheck clean. The being targets
Opus 4.7; the substrate is not designed to be portable across arbitrary
LLMs (see ARCHITECTURE.md Part 7 for why).

## Install

```
pnpm install
pnpm build
```

Requires:
- Node >= 22
- An OpenAI-compatible embeddings endpoint (defaults to LM Studio on
  `localhost:1234` with `text-embedding-qwen3-embedding-8b`, 4096 dims)
- Anthropic credentials for cognition. `anthropic.auth` defaults to `auto`,
  which prefers a Claude Code OAuth token (see `borg auth status`) and
  falls back to `ANTHROPIC_API_KEY` if no OAuth token is available. Set
  `BORG_ANTHROPIC_AUTH=api-key` (and provide `ANTHROPIC_API_KEY`) or
  `BORG_ANTHROPIC_AUTH=oauth` to pin the mode.

Copy `.env.example` if you want to run with an API key; OAuth users can
skip it and rely on the shared Claude credentials file (run `claude
/login` once, then `borg auth status` to confirm).

## Development

```
pnpm test         # run tests once
pnpm test:watch   # watch mode
pnpm typecheck    # tsc --noEmit
pnpm build        # tsup build (dist/)
pnpm dev          # tsx watch on src/cli/index.ts
pnpm format       # prettier
```

### Scripts

`pnpm chat` runs `scripts/chat.ts`, a developer-only interactive helper for
local sessions. It is not part of the shipped `borg` CLI surface.

## CLI surface

```
borg version
borg config show
borg auth status|refresh

borg stream tail [--n 20] [--session <id>]
borg stream append --kind <kind> --content <text>

borg episode search <query> [--limit] [--since <rel>]
borg episode show <id>
borg episode extract [--since 1h]

borg goal add|list|done|block|progress
borg value add|list|affirm|bind
borg trait show

borg turn "<message>" [--session] [--audience] [--stakes low|medium|high]
borg workmem show|clear [--session]

borg semantic node add|show|search|list
borg semantic edge add|list
borg semantic walk <node-id> [--depth 2]
borg commitment add|list|revoke [--audience]
borg review list|resolve

borg dream [--process ...] [--dry-run] [--budget N] [--output plan.json]
borg dream {consolidate,reflect,curate,oversee,ruminate,narrate}
borg dream apply --plan plan.json
borg audit list|revert

borg period current|list|open|close|show
borg growth list|add
borg question list|add|resolve|abandon|bump

borg skill add|list|show|select
borg mood current|history
borg social profile|upsert|adjust-trust
```

## Library surface (sketch)

```ts
import { Borg } from "borg";

const borg = await Borg.open();

// per-turn cognition
const result = await borg.turn({ userMessage: "...", audience: "alice" });

// memory access
await borg.episodic.search({ query: "...", limit: 5 });
await borg.self.goals.add({ description: "..." });
await borg.skills.select("debugging pgvector similarity");
await borg.mood.current(sessionId);
await borg.social.getProfile(entityId);

// offline
await borg.dream.run({ processes: ["consolidator", "reflector"] });

await borg.close();
```

## License

Unlicensed (personal project). Ask before reusing substantively.
