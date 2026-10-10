# Similarity distribution measurements (phase A)

Measurement tooling for the qwen3-embedding-8b (4096-d) → scw/bge-m3 (1024-d)
migration. Inventory baseline: `fb25a5c8`. Nothing here supplies runtime defaults,
changes thresholds, calls an embedding gateway, or opens the Borg facade.

## Run inside the pod

From the Borg checkout, with Node 22 and the existing dependencies (including tsx):

```sh
export PATH=/layers/paketo-buildpacks_node-engine/node/bin:$PATH
mkdir -p /tmp/borg-measure-cache/tmp
export TMPDIR=/tmp/borg-measure-cache/tmp
export XDG_CACHE_HOME=/tmp/borg-measure-cache

node --import tsx scripts/measure-similarity-distributions.ts \
  --bank /tmp/borg-bank-copies/team-agent-ai \
  --vectors current --sample 10000 --seed borg-similarity-v1 \
  --cache-dir /tmp/borg-measure-cache \
  --out /tmp/borg-similarity-results

node --import tsx scripts/measure-similarity-distributions.ts \
  --bank /tmp/borg-bank-copies/team-agent-ai \
  --vectors prev --sample 10000 --seed borg-similarity-v1 \
  --cache-dir /tmp/borg-measure-cache \
  --out /tmp/borg-similarity-results
```

Substitute the location of the **quiescent tenant copy**, not the tenant root
containing multiple banks. The copy must contain `lancedb/` and the old
`lancedb.prev-<N>/` alongside it. Include `borg.db`, its WAL if present, and
`embedding-profile.json`. The script does not take a live-system snapshot.

Both commands measure the intersection of valid current/previous IDs in sorted
order, regardless of physical table order. Run them sequentially, in either
order, with identical inputs. The first creates a single-model report with
pending proposals; the second adds the comparison. A rerun replaces that model's
artifact. Changed vectors/titles, family metadata, bank identity, source directory,
seed, or sample prevent stale reports from contributing proposals.

`prev` selects `lancedb.prev-<profile.generation>` when present. The migration uses
the **target** generation in this directory name. Otherwise the largest numeric
suffix is selected and reported. Metadata supplies model attribution when
available; otherwise migration model names are explicitly marked as assumptions.
Numeric proposals require current 1024-d BGE-M3 and previous 4096-d Qwen rows.

`--vectors prev` fails with a clear error if there is no previous directory.
`--vectors current` can still measure a bank without previous vectors, but those
rows are marked unpaired and cannot produce cross-model proposals. Missing tables
are reported; they are never created. Duplicate IDs abort the run. Zero, nonfinite,
or wrong-dimension vectors are excluded on **both** sides and listed with reasons.

## Tables and read-only behavior

The default `--tables` is:

```text
episodes,semantic_nodes,open_questions,action_records,skills,observed_events
```

The seventh table from `src/borg/storage-setup.ts:293` is
`image_perception_embeddings`. Include it explicitly in the comma-separated list
to measure it too. It uses `payload_id`; the other six use `id`. Display fields are
episode title, semantic label/description, question text, action description,
skill name/applies_when, observed interaction text, and image embedding text.
These fields are only displayed for human review, never interpreted.

The reader follows `eval/embedding-ab/bank.ts`: direct existing-table opens and
LanceDB queries, bypassing production schema evolution and backfills. It measures
all stored rows, including inactive/archived rows, to expose the stored geometry.
It does not apply retrieval visibility, eligibility, temporal, or status filters.

Family metadata comes from SQLite
`episode_index(episode_id, consolidation_family_id)`. Read-only SQLite connections
can still write WAL shared-memory sidecars, so the script first copies the main
SQLite file and WAL to a disposable directory under
`<cache>/borg-similarity-distributions/`. Cache precedence is `--cache-dir`, then
`XDG_CACHE_HOME`, then `borg-cache` under the OS temporary directory. It never
derives this path from HOME (which can be `/` and read-only in the pod).
It opens that scratch copy with
`openReadOnlyDatabase` and removes it afterward. The bank itself receives no
writes. Reports must be outside the bank; output paths through symlinks into the
bank are rejected. Each report file is written with temp/fsync/rename and mode
0600. Writes are atomic per file, not transactional across all report files; rerun
the same command if interrupted during publication.

## What is measured

- **Nearest neighbor:** exact maximum cosine for every measured row, excluding its
  own ID. Distinct IDs with identical vectors remain valid neighbors. JSON keeps
  each row ID, neighbor ID, and cosine, plus p50/p90/p95/p99/max/min. An empty or
  singleton table has no nearest-neighbor samples; singleton rows carry nulls.
- **Random pairs:** uniform seeded sampling without replacement of unordered,
  non-self pairs, capped at the available pair count. Both models sample identical
  pair IDs. JSON retains sorted cosine samples and a hash of sampled pair IDs.
- **Duplicate sweep:** exact counts of all unordered pairs with cosine **>=**
  0.80, 0.85, 0.90, 0.95, and 0.97. Each cutoff includes the first five qualifying
  title/ID pairs in sorted ID order, rather than only the highest-scoring pairs.
  These are candidates for human judgment, not duplicate verdicts.
- **Episode families:** group IDs and member IDs from SQLite, within-family and
  across-family cosine distributions, and population/sample counts. Only pairs
  with two non-null family IDs enter these strata; missing IDs are not treated as
  singleton families. Each stratum uses an independent deterministic reservoir
  capped at `--sample`, so small strata are exhaustive and large strata are sampled.
  A missing DB or family column produces an explicit unavailable reason. Family
  labels describe the existing consolidation structure, not independent truth.

All vectors are validated and cosines use their actual norms (normalization is
not assumed). Nearest-neighbor and duplicate sweeps share one exhaustive pair pass:
**O(rows² × dimensions)** time, with no ANN approximation and no pair matrix.
`--sample` limits retained distribution samples, not the exhaustive work. Progress
goes to stderr at table boundaries and roughly every five seconds during long
passes. Memory includes both model vector tables, per-row nearest neighbors, and
bounded samples; only one table pair is loaded at a time.

## Percentile proposals

The inventory assigns each gate a stated stored-table distribution. Gates asking
whether *any* close duplicate exists use per-row nearest-neighbor maxima. Pair
clustering and candidate score floors use random pairs. Every assignment and its
limitation is included in JSON and Markdown. Stored survivors exclude candidates
that old thresholds already rejected, and record-to-record geometry is only a
proxy for query-to-record geometry. Fresh goals, commitments, tags, procedural
snapshots, compact messages, and pending actions have no paired LanceDB vectors;
their proposals use explicitly labeled, weaker proxies.

Quantiles use linear interpolation at `h = (n - 1) * p`. For a Qwen cutoff `t`,
the script inverts that quantile curve to find its rank `p`, then evaluates the
BGE curve at `p`. Ties use their midpoint rank. The full retained distributions
are used, not interpolation between just p50 and p99. Values outside the observed
Qwen range clip to p0 or p100 and carry a support warning; those tails need more
data. Empty distributions produce null proposals. A single observation is marked
as such. For `maxClusterDiameter`, it maps the cosine floor `1 - 0.18` and converts
the result back to a distance.

**`BORG_RECALL_ABSTAIN_THRESHOLD` is not a cosine cutoff.** At this HEAD,
`src/sidecar/memory-handler.ts:2888` compares the maximum included episode
`rawScore`; `src/retrieval/scoring.ts:399` and `src/retrieval/pipeline.ts:1903` build
that fused score from cosine plus other signals/boosts. Its default is 0 (disabled)
and deployment environment may override it. It is inventoried but emits
`not_a_cosine_threshold` and a null proposal. Calibrating it needs real query replay
and scoring signals; applying a cosine percentile conversion would invent a number.

No proposal is automatically adopted. Phase B chooses settings after reviewing
the distributions, family strata, and example pairs on real banks.

## Output layout

```text
<out>/
  <tenant-basename>-<sha256-of-canonical-bank-path-first-12>/
    current.json   # full current-model measurements, pairing and provenance
    prev.json      # full previous-model measurements, pairing and provenance
    report.json    # both runs (available ones) and all 23 threshold proposals
    summary.md     # distributions, sweeps, pairing, proposals and full inventory
```

The path hash prevents banks with identical basenames from overwriting each other.
Moving the copy produces a different report directory. Raw vectors are not exported.
Example text and row IDs are included for human inspection. The tool does not read
runtime threshold overrides; inventory values are source defaults at `fb25a5c8`.

## Threshold inventory

Paired config/module fallbacks for the same gate share one row. The extra distance
gate is included because it constrains the same cosine geometry. The search also
finds `src/retrieval/scoring.ts:36` (`similarity: 0.7`), which is a scoring weight,
not a threshold; test fixtures, confidence gates, mood/valence thresholds,
posterior divergence, and unconfigured optional floors are not cosine cutoffs.

| Gate / file:line | Default | Compared items | Vector source |
|---|---:|---|---|
| action_persistence_duplicate<br>[src/cognition/actions/action-state-extractor.ts:55](../../src/cognition/actions/action-state-extractor.ts) | 0.85 | New action description vs active and newly accepted actions on the same dedup axis | Fresh embeddings of SQLite action_records descriptions (not stored Lance vectors) |
| open_question_recall<br>[src/retrieval/open-questions.ts:6](../../src/retrieval/open-questions.ts) | 0.01 | Retrieval query vs open questions; vector candidate floor | LanceDB open_questions |
| commitment_evidence<br>[src/retrieval/pipeline.ts:394](../../src/retrieval/pipeline.ts) | 0.3 | Recall intent query vs active commitment directive; evidence admission | Fresh embeddings of SQLite commitments directives; no persistent vector table |
| action_thread<br>[src/config/index.ts:207](../../src/config/index.ts)<br>[src/cognition/evidence-ledger/action-threads.ts:26](../../src/cognition/evidence-ledger/action-threads.ts) | 0.85 | Two action descriptions with the same non-null goal and actor; thread union | LanceDB action_records (fresh description embeddings for missing vectors) |
| skill_selection<br>[src/config/index.ts:393](../../src/config/index.ts)<br>[src/memory/procedural/selector.ts:67](../../src/memory/procedural/selector.ts) | 0.5 | Current context vs skill applies_when; selection candidate floor | LanceDB skills |
| consolidation_similarity<br>[src/config/index.ts:498](../../src/config/index.ts) | 0.82 | Two episode embeddings; complete-link consolidation pair eligibility | LanceDB episodes |
| consolidation_diameter<br>[src/config/index.ts:499](../../src/config/index.ts) | 0.18 | Two episode embeddings; maximum 1-cosine cluster diameter (cosine floor 0.82) | LanceDB episodes |
| consolidation_temporal_bypass<br>[src/config/index.ts:505](../../src/config/index.ts) | 0.97 | Two episode embeddings; bypass soft temporal proximity within the maximum gap | LanceDB episodes |
| reflection_goal_and_tag_grouping<br>[src/config/index.ts:520](../../src/config/index.ts) | 0.82 | Episode vs goal vector AND two freshly embedded episode tags; reflection clustering | LanceDB episodes plus in-memory goal/tag vectors |
| skill_synthesis_duplicate<br>[src/config/index.ts:554](../../src/config/index.ts) | 0.88 | Proposed skill applies_when vs existing skill contexts; dedup candidate | LanceDB skills |
| ruminator_duplicate<br>[src/config/index.ts:625](../../src/config/index.ts) | 0.9 | Two open questions; similarity backstop for duplicate-merge planning | LanceDB open_questions |
| semantic_recall<br>[src/retrieval/semantic-retrieval.ts:76](../../src/retrieval/semantic-retrieval.ts) | 0.01 | Retrieval query vs semantic node; vector candidate floor | LanceDB semantic_nodes |
| semantic_revision<br>[src/cognition/shared-state/semantic-revision.ts:61](../../src/cognition/shared-state/semantic-revision.ts) | 0.01 | Shared-state reconciliation query vs semantic nodes; revision candidate floor | LanceDB semantic_nodes |
| generation_repeated_input<br>[src/cognition/generation/generation-gate.ts:18](../../src/cognition/generation/generation-gate.ts) | 0.96 | Current compact user message vs last four user messages; repeated-exchange signal | Fresh embeddings of stream/recency text; no stored message vectors |
| goal_promotion_duplicate<br>[src/cognition/goals/turn-goal-promotion-service.ts:33](../../src/cognition/goals/turn-goal-promotion-service.ts) | 0.9 | New goal description vs active and newly accepted goal descriptions | Fresh embeddings of SQLite goals descriptions; no LanceDB goals table |
| open_question_duplicate_backstop<br>[src/memory/self/open-question-duplicates.ts:12](../../src/memory/self/open-question-duplicates.ts) | 0.9 | New open question vs nearest existing open question; duplicate backstop | LanceDB open_questions |
| observed_event_topic<br>[src/memory/observed-events/projection.ts:12](../../src/memory/observed-events/projection.ts) | 0.45 | Turn topic/query vector vs observed interaction text; topic recall floor | LanceDB observed_events |
| semantic_duplicate_review<br>[src/memory/semantic/review-service.ts:110](../../src/memory/semantic/review-service.ts) | 0.9 | Proposition node vs other proposition nodes; contradiction/duplicate review candidates | LanceDB semantic_nodes |
| procedural_evidence_cluster<br>[src/offline/procedural-synthesizer/index.ts:53](../../src/offline/procedural-synthesizer/index.ts) | 0.85 | Two procedural evidence snapshots; synthesis clustering | Fresh embeddings of SQLite procedural evidence; no stored snapshot vectors |
| semantic_extraction_duplicate<br>[src/memory/semantic/extractor.ts:133](../../src/memory/semantic/extractor.ts) | 0.88 | Extracted semantic node vs existing nodes; dedup candidate | LanceDB semantic_nodes |
| pending_action_merge<br>[src/memory/working/store.ts:31](../../src/memory/working/store.ts) | 0.85 | Incoming pending action vs existing pending actions (description plus next_action) | Working-memory pending-action vectors (JSON/in-memory), fresh embeddings as needed |
| reflection_insight_duplicate<br>[src/offline/reflector/index.ts:137](../../src/offline/reflector/index.ts) | 0.88 | Reflected insight embedding vs existing proposition nodes; dedup candidate | LanceDB semantic_nodes |
| BORG_RECALL_ABSTAIN_THRESHOLD<br>[scripts/memory-sidecar-main.ts:72](../../scripts/memory-sidecar-main.ts)<br>[src/sidecar/memory-handler.ts:1396](../../src/sidecar/memory-handler.ts) | 0 | Top included episode rawScore vs abstention floor; zero disables the gate | LanceDB episodes plus retrieval scoring signals; rawScore is a fused score, NOT cosine |

## Validation

Targeted Vitest tests use synthetic 1024-d/4096-d tables and no real API calls.
They cover quantiles/ranks, proposal conversion, deterministic sampling, exact
self-excluding nearest neighbors, sweep counts, family strata, ID pairing,
missing previous directories, stale runs, CLI invocation, and unchanged input
file hashes/mtimes (including an uncheckpointed SQLite WAL copy).

The scripts and tests are included in `tsconfig.test.json`, so the repository's
`npm run typecheck` checks them. Do not run the full suite for phase A.

Validated on Node **22.23.2**: **19 tests passed in 2 files**
(`bank.test.ts`, `statistics.test.ts`); **`npm run typecheck` passed** across all
five configured TypeScript projects. Prettier and `git diff --check` passed.
Test/Vite/npm/tsx scratch and caches were under `$HOME`. The full test suite was
not run. These results validate the tooling with synthetic banks; no real-bank
calibration values are claimed here.
