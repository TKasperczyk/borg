# Offline embedding-model A/B evaluator

This standalone evaluator compares embedding models against a copied Borg tenant bank. It lives
under `eval/embedding-ab/` because this repository already has a top-level `eval/` tooling area and
an eval-specific TypeScript configuration.

The source bank is opened with SQLite `readOnly` plus `query_only`, and LanceDB is used only to open
and query the existing `episodes` table. Episodes are decoded and filtered through Borg's own
`EpisodicRepository`; no production `Borg.open()` wiring, migrations, reconciliation, retrieval
logging, or source-bank writes run. `--out` is rejected when it is inside `--data-dir`.
The guard resolves symlinks in existing ancestors and uses path-segment-aware containment, so the
bank itself, ordinary descendants, and descendants named with a `..` prefix are all rejected.

## Run

```bash
NODE_EXTRA_CA_CERTS=/path/to/corporate-ca.pem \
KRATOS_BASE_URL=https://kratos.example/v1 \
LLM_API_KEY=... \
npm run embedding:ab -- \
  --data-dir /data/team-agent-ai \
  --models generative-apis/qwen3-embedding-8b,scw/bge-m3,generative-apis/bge-multilingual-gemma2 \
  --out /tmp/borg-embedding-ab \
  --queries /tmp/eval-queries.json \
  --gold-size 60 \
  --judge-model
```

`--queries` accepts either a file containing a JSON array of strings or the JSON array itself. Each
query string is embedded byte-for-byte as supplied: role prefixes and
`<untrusted_observation>` wrappers are neither trimmed nor parsed. `--judge-model` without a value
uses the same automatically selected strong model as gold generation; pass a model ID after the flag
to choose one explicitly. Judging is disabled when the flag is absent.

Other options:

- `--gold-size N` defaults to 60 and may be 0 to skip synthetic gold generation.
- `--batch-size N` defaults to 8. Every embedding HTTP attempt has a fixed 15-second timeout.
- `--help` prints the complete CLI synopsis.

The tool calls `GET /v1/models`. For gold generation it prefers
`generative-apis/qwen3-235b-a22b-instruct-2507`, then an advertised GLM model, then an advertised
`mistral-small` model. The known embedding dimensions are 4096 for
`generative-apis/qwen3-embedding-8b`, 1024 for `scw/bge-m3`, and 3584 for
`generative-apis/bge-multilingual-gemma2`. An unknown embedding model uses advertised dimension
metadata when available, otherwise one dimension-probe operation whose HTTP attempts are recorded.

The instrumented evaluator transport is the single retry owner. Each embedding operation makes at
most three total attempts: the initial attempt, then up to two retries after 500 ms and 1,500 ms for
transient connection, 409, 429, and 5xx errors. The OpenAI SDK retry loop and Borg embedding
client's model-reload retry ladder are disabled. A local timeout or HTTP 408 is never retried; the
failed batch remains missing so intermittent stalls are visible in coverage, latency, and error
counts. Attempt numbers in `results.json` are therefore the monotonic 1, 2, and 3 from this one
loop. A later invocation retries only missing vectors. Use a new output directory when you want a
fresh latency measurement rather than a cache-heavy rerun.

If a model cannot be initialized, including when its dimension probe exhausts the retry policy, the
failure is recorded on that model and evaluation continues for the remaining models. The CLI still
writes partial `results.json` and `report.md` outputs.

## Outputs

All output is private-mode scratch data under `--out`:

- `results.json` contains inputs, discovered models, bank/corpus identity, the exact common-corpus
  episode IDs and excluded IDs, full per-question ranks, per-query top-10 results, every pairwise
  model comparison, optional judge ratings, per-model coverage or initialization errors, and every
  embedding HTTP attempt's latency.
- `report.md` contains compact synthetic-gold, real-query overlap/cross-rank, optional relevance,
  and p50/p95/max latency tables plus an intentionally unfilled verdict template.
- `cache/vectors/*.jsonl` stores content-addressed float32 vectors as little-endian binary encoded in
  base64. The model ID and dimensions are part of the cache identity.
- `cache/gold-questions.jsonl` and `cache/judge-ratings.jsonl` cache successful structured LLM
  outputs. Cache keys include the model, prompt version, and relevant content hashes.

Every successfully initialized model is ranked over the same candidate corpus: the intersection of
active episodes embedded successfully by all participating models. Per-model episode coverage and
the common-corpus size are reported. If that intersection is smaller than the active bank, the
output sets `comparative_metrics_comparable_to_full_bank_recall` to `false` and the report labels
recall and other comparative metrics as not comparable to full-bank recall. A gold source excluded
from the intersection has a `null` rank for every model; a failed gold-query embedding is also
`null` and remains a miss in recall/MRR. Gold-question generation failures are reported separately
because no valid query exists to rank.

## Repository behavior reused

The implementation was based on searches through `src/memory/episodic`, `src/borg/open.ts`,
`src/borg/storage-setup.ts`, `src/storage/lancedb`, `src/embeddings`, and `src/retrieval`:

- Episode text exactly matches `EpisodicExtractor`: `title + "\n" + narrative + "\n" +
  tags.join(" ")`.
- Episode decoding and active/effective-visibility filtering use `EpisodicRepository.listAll()` and
  `isEpisodeEffectivelyVisible()`.
- Gateway vectors use `OpenAICompatibleEmbeddingClient`, including its float encoding and dimension
  validation. The evaluator adds instrumentation and the evaluation-specific timeout policy around
  its transport.
- Question generation and relevance grading use `OpenAICompatibleLLMClient`, forced structured tool
  calls, and Zod validation. These are the only semantic interpretation steps.
- Ranking uses the existing `cosineSimilarity` helper. It already divides by vector norms, so no
  second normalization implementation is needed.
- Episode decay and production retrieval signal fusion were found but intentionally are not applied:
  this experiment isolates raw embedding-model quality with cosine over the same active corpus.
- The production embedding stall guard was found but intentionally is not used because it retries
  timed-out calls, contrary to this experiment's requirement to expose stalls.

## Verification

Run the network-free evaluator tests and typechecks with:

```bash
npx vitest run eval/embedding-ab
npm run typecheck
```

An end-to-end dry run still needs a reachable OpenAI-compatible gateway because both candidate
vectors and synthetic questions are intentionally real model calls. Against a small copied bank, use
the main command above with `--gold-size 3` and a fresh scratch directory. For a query-only smoke run,
use `--gold-size 0`; model discovery and embeddings still exercise the live gateway.
