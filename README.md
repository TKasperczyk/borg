# Borg

Cognitive memory architecture for autonomous AI beings.

A library, CLI, and optional headless daemon for building agents with
persistent memory, explicit cognition, and identity that evolves over time.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the design document.

## Status

Early development. Not ready for use.

## Install

```
pnpm install
pnpm build
```

## Development

```
pnpm test         # run tests once
pnpm test:watch   # watch mode
pnpm typecheck    # tsc --noEmit
pnpm dev          # tsx watch on src/cli/index.ts
```

## Requirements

- Node >= 22
- An OpenAI-compatible embeddings endpoint (defaults to LM Studio on `localhost:1234`)
- Anthropic API key for cognition

See `.env.example`.
