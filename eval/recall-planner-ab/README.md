# Recall query planner A/B evaluator

This tool evaluates Borg's production recall query planner against a copied tenant bank. It runs
the real episodes-only disclosure path (`RetrievalPipeline.searchEpisodesForDisclosure`) for every
case and requested semantic-variant count, captures planner and per-intent candidate traces in
memory, and reports expected-episode rank after production score fusion and MMR.

The optional baseline is intentionally labelled and narrower than Borg's current degraded path:
it embeds the raw FOCUS blob and uses that vector lane only. It has no LLM expansion, exact-term,
typed-query, or recent candidate lane. The implementation still uses `RetrievalPipeline`; a bound
repository proxy returns no candidates for the pipeline's unavoidable recent intent.

## Run

The repository intentionally has no package script for this evaluator. Run it directly:

```bash
NODE_EXTRA_CA_CERTS=/path/to/corporate-ca.pem \
KRATOS_BASE_URL=https://kratos.example/v1 \
LLM_API_KEY=... \
BORG_MODEL_RECALL_EXPANSION=generative-apis/your-recall-model \
npx tsx eval/recall-planner-ab/cli.ts \
  --data-dir /data/copied-team-agent-bank \
  --cases eval/recall-planner-ab/seed-cases.json \
  --out /tmp/borg-recall-planner-ab \
  --variant-counts 1,3 \
  --baseline \
  --judge-model
```

`--embedding-model <id>` overrides the model ID in the copied bank's `config.json`; otherwise that
configured bank model is used. Its returned vector dimension must match the existing episode table.
`--judge-model` without an ID uses the same strong-model selection policy as `eval/embedding-ab`;
an explicit ID selects that model. Judging is disabled when the flag is absent.

Additional referential Polish cases can be synthesized from a deterministic sample of active bank
episodes:

```bash
KRATOS_BASE_URL=https://kratos.example/v1 \
LLM_API_KEY=... \
npx tsx eval/recall-planner-ab/cli.ts \
  --data-dir /data/copied-team-agent-bank \
  --cases eval/recall-planner-ab/seed-cases.json \
  --out /tmp/borg-recall-planner-generated \
  --variant-counts 1,3 \
  --baseline \
  --generate-cases 6
```

Successful generated cases are evaluated in the same run and written to
`generated-cases.json`. Generation failures stay in `results.json` and do not stop other cases.
The generator emits exactly two context turns (user, then assistant) and a Polish FOCUS whose
reference should only resolve with that context. It uses structured tool output and does not infer
referential language with deterministic text rules.

## Case format

The `--cases` file is a non-empty JSON array:

```json
[
  {
    "id": "referential-example",
    "focus": "A co on o tym powiedział?",
    "context_turns": [
      { "role": "user", "content": "Przypomnij rozmowę z Jackiem o Atlasie." },
      { "role": "assistant", "content": "Jacek omawiał z nami rollback Atlasa." }
    ],
    "identity": {
      "memory_owner_name": "team-agent",
      "current_sender_name": "Tomasz",
      "current_audience_name": "Tomasz",
      "current_venue": { "type": "personal", "name": "Tomasz" },
      "entity_terms": ["Jacek", "Atlas"]
    },
    "owner_recent_activity": [
      {
        "excerpt": "Jacek opisał rollback Atlasa.",
        "occurred_at": 1788508500000,
        "venue": { "type": "groupChat", "name": "AI Ninjas" },
        "counterparty_name": "Jacek"
      }
    ],
    "expected_episode_ids": ["ep_aaaaaaaaaaaaaaaa"],
    "now": "2026-09-05T12:00:00+02:00",
    "notes": "Optional operator note."
  }
]
```

Conversation types are `personal`, `groupChat`, and `channel`; `occurred_at` is epoch
milliseconds. `now` is optional and pins the pipeline clock for that case as an ISO-8601 instant
with an explicit offset; the planner reads NOW from that clock, so a case phrased around "yesterday"
is only reproducible (and planner-cacheable across runs) when it is pinned. Without it the wall
clock is used, as in production. Expected IDs absent from the active/effectively-visible corpus remain explicit
misses. For disclosure parity, the evaluator resolves already-supplied audience and activity-venue
names through Borg's entity repository. Unresolved handles are recorded per run rather than silently
granting cross-audience access.

## Read-only and failure behavior

The source bank is opened with SQLite `readOnly` and `query_only`; LanceDB opens only the existing
`episodes` table. Retrieval accounting and recall-state persistence are disabled. Before any search,
the tool requires the current `lance_backfilled_at` episode-index marker, preventing the repository's
normal compatibility backfill from attempting a write. If the marker is absent, open the copied
bank once with the current Borg build and rerun the evaluator. Never do that against the source bank.
For production-path parity it also requires the current `entities` and `stream_entry_index` tables;
the no-index repository fallback is intentionally not used by this evaluator.

The output directory is rejected when it is the bank or any descendant of it. The guard uses
`resolveRealPathForCreation` and `isPathWithin` through the exported sibling evaluator guard, so
symlinked descendants are rejected too.

Planner, embedding-lane, and judge failures are bounded to their case where the production pipeline
can degrade safely. A thrown case error becomes a failed run and later cases continue. Provider
error summaries are bounded; API keys are never written or logged. The report stores a sanitized
gateway base URL without credentials, query parameters, or fragments.

## Outputs and metrics

All artifacts under `--out` use private file/directory modes:

- `results.json` contains cases, configurations, parsed planner outputs, each lane's query and full
  ranked candidate list plus expected ranks, post-MMR final ranks and top ten, degradation events,
  audience resolution, embedding logical calls, every physical embedding HTTP attempt, optional
  judge ratings, and aggregate metrics.
- `report.md` contains Recall@1/3/10, MRR, a per-case comparison table, planner latency p50/p95,
  logical embedding calls per query, gateway attempts and p95 latency, errors/timeouts, optional
  judged top-five relevance, and an intentionally blank verdict template.
- `cache/planner-outputs.jsonl` caches successful structured planner responses by the planner model
  and complete planner request inputs.
- `cache/vectors/*.jsonl` is the imported `VectorCache`; model, dimensions, and text hash form the
  identity. Concurrent duplicate requests share a pending vector.
- `cache/generated-cases.jsonl` and `cache/judge-ratings.jsonl` cache successful structured LLM
  work by model, prompt version, and content identity.

Recall treats the best rank among a case's expected IDs as its rank. A failed run or an expected ID
missing from the active corpus is a miss. Planner latency spans the traced LLM start through parsed
plan completion (or recall-expansion degradation). Logical embedding-call counts include disk and
pending cache hits; gateway-attempt counts include transport retries. Therefore a warm-cache rerun
can have the same logical load with fewer physical calls and much lower planner latency.

## Production pieces reused

Implementation started with repository searches through `eval/embedding-ab`,
`src/retrieval/recall-expansion.ts`, `src/retrieval/pipeline.ts`, the sidecar recall-options block,
`src/memory/episodic/repository.ts`, and the assessor scenario runner. Rather than copying the
sibling evaluator, this tool imports its:

- active-bank loader and corpus fingerprint;
- instrumented OpenAI-compatible gateway, retry/timeout classification, and model discovery;
- durable `VectorCache` and `JsonlValueCache`;
- deterministic episode sampler and structured top-five relevance judge (given the focus, context,
  identity handles, and owner recent activity, but not the expected IDs);
- Recall@K/MRR summarizer; and
- canonical scratch-directory path guard.

Planner context uses the sidecar's camel-case production shape, including separately labelled
context turns, identity handles, owner-authored recent activity, and `semanticVariantCount`.
`CallbackTracer` exposes the production `recall_expansion.completed` and
`retrieval.intent_candidates` payloads without a temporary trace file.

## Sol assessor scenario

The accompanying Polish/English conversational scenario is runnable without real APIs:

```bash
pnpm assess --mock --scenario recall-resolution
```

Run the real assessor and Borg model paths with configured Anthropic credentials:

```bash
pnpm assess --real --scenario recall-resolution --keep
```

The real scenario enables trace payloads and asks the assessor to verify both that the expected
labelled-venue exchange was retrieved and that `recall_expansion.completed.resolved_query` names the
pronoun/ellipsis referent.

## Verification

```bash
npx vitest run eval/recall-planner-ab assessor
npm run typecheck
npx prettier --check eval/recall-planner-ab assessor/scenarios/recall-resolution.ts assessor/scenarios/index.ts assessor/scenarios/index.test.ts
git diff --check
pnpm assess --mock --scenario recall-resolution
```
