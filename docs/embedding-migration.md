# Embedding model migration

## Preconditions

Deploy this version of the sidecar **with the old embedding environment and an explicit legacy source assertion first**:

```text
EMBEDDING_MODEL=generative-apis/qwen3-embedding-8b
EMBEDDING_DIMS=4096
EMBEDDING_LEGACY_SOURCE_MODEL=generative-apis/qwen3-embedding-8b
```

The assertion allows an existing unlabelled bank to be adopted only when the effective shared client has that model and every existing vector table has the expected float32 vector type and dimensions. Missing or mismatched assertions leave legacy banks unavailable. New banks with no SQLite/LanceDB storage are initialized with the effective client profile. Restart the sidecar so every open bank holds the new access lease. A sidecar from before this change does not honor migration fences; stop that process before migrating. Stop other direct bank-writing tools as well.

The pod app directory is `/workspace/workspace/repos/app` and is read-only. Node is at `/layers/paketo-buildpacks_node-engine/node/bin/node` and is not initially on PATH. Only `/data` is writable. Run this setup in each new pod shell; the writable `TMPDIR` is required by the tsx cache. All migration commands use `nice -n 19` because the live sidecar shares the pod.

```sh
# Pod shell setup
export PATH=/layers/paketo-buildpacks_node-engine/node/bin:$PATH
export TMPDIR=/data/tmp
mkdir -p "$TMPDIR"
cd /workspace/workspace/repos/app
nice -n 19 node --import tsx scripts/migrate-embeddings.ts --help
```

Use Node >= 22.18.0, the installed `tsx`, and the pod's gateway credentials/CA environment. No `pnpm` is needed. The CLI uses `LLM_API_KEY` and `KRATOS_BASE_URL`, including the sidecar's existing default gateway URL. `BORG_EMBEDDING_API_KEY` and `BORG_EMBEDDING_BASE_URL` are fallbacks. It uses a 60-second request timeout and at most four attempts per batch (backoff 1, 4, 10 seconds). Batch size defaults to 32 and embedding concurrency to 2. Tenant processing is always sequential.

Do not change the shared sidecar model while tenants have mixed profiles. After each tenant migrates, the old-configured sidecar returns HTTP 503 for that tenant until the final environment cutover. Other tenants can continue serving. Plan for this per-tenant downtime.

All commands below name the source model explicitly, which is required for unlabelled legacy banks. The source dimension is read from **every existing table** and must be consistent; production should report 4096. No text is sent to an LLM. The migration uses the production embedding client directly.

## Inventory and capacity check

```sh
nice -n 19 node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --tenant team-agent-ai \
  --source-model generative-apis/qwen3-embedding-8b \
  --target-model scw/bge-m3 --target-dims 1024 --dry-run
```

Repeat for `team-agent-esb`, `team-agent-rtm`, `team-agent-tn`, and `team-agent-tni`, or use `--all-tenants` instead of `--tenant team-agent-ai`. `--tenant` is repeatable. Discovery requires a valid tenant directory containing `borg.db`; backups, quarantines, and non-bank directories are excluded. CLI discovery is strict: unreadable candidates, unreadable bank files, or symlink candidates cause a nonzero exit before any tenant is migrated. `--all-tenants` cannot silently skip an inaccessible bank.

The JSON report includes all seven tables: `episodes`, `semantic_nodes`, `skills`, `open_questions`, `action_records`, `image_perception_embeddings`, and `observed_events`. It includes archived/superseded rows, source dimensions, exact row identities, SQL-only/vector-only ids, text disagreements, and serialized-vector locations in review refs, audit reversals/targets, and JSON files under the tenant. Saved plans outside the tenant cannot be inventoried; their application is still protected by the runtime payload guard.

`report.embedding_text_unrecoverable` contains `count`, `ids`, and `rows` for **every** episode whose input cannot be reconstructed. Each row has `episode_id`, `reason`, and `candidate_count`: the number of distinct rendered embedding inputs consistent with the persisted narrative and raw lineage (zero when sources are missing or inconsistent). Inventory continues across all rows and tables. Such rows retain their id and non-vector field hash, with `text_hash: null`. A completed dry-run exits **0**, even when `complete: false` means a real migration would be blocked; inspect the report instead of treating exit 0 as migration approval. `--all-tenants --dry-run` continues through every selected bank. Configuration, discovery, storage-read, integrity, or capacity errors still exit nonzero.

A dry-run writes no profile, fence, backup, checkpoint, or staging table and makes no gateway request. SQLite's read-only WAL handling can create `borg.db-wal`/`borg.db-shm` coordination files. A live dry-run is **provisional**; the authoritative inventory runs after draining. SQL/vector discrepancies and unknown LanceDB tables block migration rather than silently discarding data. Resolve them with the old sidecar/appropriate repair operation before proceeding.

The capacity check is conservative: whole-bank backup bytes plus twice the source LanceDB bytes of unfinished tables, three copies of remaining target vector bytes, and 64 MiB for staging/metadata. It checks available blocks on both filesystems when the backup is elsewhere. Previous generations also count toward the whole-bank backup. A table retains its full staging allowance until complete, since individual row sizes and fragment rewrite costs can vary. Capacity is checked before every backup and staging attempt, including resume. Resume uses the destination recorded in the journal, excludes completed backup work, and accounts for staging rows with valid committed checkpoints. Changing `--backup-dir` does not relocate an existing attempt. Use `--backup-dir /data/embedding-backups` to choose a different backup directory in this pod; the destination must be outside every tenant directory. Symlinks and special files in a bank are refused. Incomplete backups remain as `.partial-*` directories for inspection; they are never silently reused or removed.

## Fence, drain, back up, migrate

Start the migration for one tenant:

```sh
nice -n 19 node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --tenant team-agent-ai \
  --source-model generative-apis/qwen3-embedding-8b \
  --target-model scw/bge-m3 --target-dims 1024
```

The first action installs `embedding-migration.lock`. If the sidecar has that bank open, the CLI exits nonzero with `EMBEDDING_BANK_BUSY` and **leaves the fence in place**. Drain it through the authenticated route (Node's fetch works in the slim pod without curl):

```sh
nice -n 19 node --input-type=module - <<'JS'
const tenant = 'team-agent-ai';
const response = await fetch(`http://127.0.0.1:8088/memory/admin/evict?tenant=${tenant}`, {
  method: 'POST',
  headers: { 'x-borg-token': process.env.BORG_MEMORY_TOKEN },
});
console.log(response.status, await response.text());
if (!response.ok) process.exitCode = 1;
JS
```

Wait for HTTP 200 with `status: "closed"`. The route waits for in-flight work, startup backfills, and pending embedding writes before closing the bank. Then repeat the migration command with **`--resume`**. If no process owns the bank, the original command continues directly.

Under exclusive bank access, the CLI:

1. Inventories every current row without calling `Borg.open`, schema migrations, reconciliation, or backfills.
2. Copies the tenant data, excluding migration locks, journals, checkpoints, reports, and prior backup manifests, into `/data/backups/<tenant>/generation-<N>-<timestamp>`. It copies `borg.db` with SQLite's backup API, not a live file copy. It checks SQLite integrity and table counts, exact inventory equality, file checksums, and fsyncs the copy before marking it complete.
3. Builds `lancedb.staging-<N>` beside `lancedb`, preserving all existing non-vector schema fields and row values, with the explicit consolidation-input labelling exception described below. Embedding inputs use the production recipe for each table, including consolidation episodes and semantic observation metadata. Image `model` fields describe perception models and remain unchanged. New consolidation episodes retain `consolidation_embedding_input`, containing the synthesized narrative and protected source lines used by the writer. Existing rows recover the input from their persisted raw lineage in source order, including archived rows, only when all possible append boundaries yield byte-identical input, unless the operator selects the fallback below.
4. Commits bounded batches and fsyncs `.embedding-migration-g<N>.jsonl` checkpoints keyed by table/id/text hash (also recording non-vector field hashes and the target profile). Resume validates the staging rows before skipping them. A table commit interrupted before its checkpoint is safely replayed by id.
5. Verifies exact id coverage and non-vector fields/schema against the expected target inventory (including only the planned input labels, if opted in), SQLite integrity/counts, finite nonzero target-dimensional vectors, and up to three nearest-neighbour self-probes per nonempty table. Every fallback-written input must reproduce the stored narrative with the runtime preservation function. These are storage/retrieval sanity checks, not a semantic-quality benchmark; use `eval/embedding-ab` for quality comparisons.
6. Journals cutover intent, renames `lancedb` to `lancedb.prev-<N>`, renames staging to `lancedb`, atomically writes `embedding-profile.json`, and removes the fence only after verification succeeds. It keeps the previous directory and writes `.embedding-migration-report-<N>.json`.

Without an opt-in, `EMBEDDING_TEXT_UNRECOVERABLE` blocks a real run after the full inventory is reported, before backup or embedding calls. The error's `report` contains the full list, and `blocked_embedding_inputs` identifies the rows still blocking that attempt. The fence remains in place. Inline legacy protocol lines can make the original synthesized prose impossible to recover from the final narrative alone; no separate original synthesis is required when using the following recorded resolution policy.

### Resolving ambiguous legacy consolidations

For a re-embed to a **new model**, where byte parity with the old embedding input is not required, explicitly choose `--legacy-consolidation-input longest-prefix`. Preview its effect first:

```sh
nice -n 19 node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --tenant team-agent-ai \
  --source-model generative-apis/qwen3-embedding-8b \
  --target-model scw/bge-m3 --target-dims 1024 \
  --legacy-consolidation-input longest-prefix --dry-run
```

The policy applies only to ambiguous legacy consolidation rows with recoverable raw sources. It selects the zero-appended-lines candidate: **the entire persisted narrative is treated as synthesized prose**, with protected source lines collected from the raw lineage in production order. The production builder renders the new embedding input from that choice. This is a deterministic new input choice, not a claim to have recovered the old input byte-for-byte.

`report.legacy_consolidation_resolutions` lists `count`, `ids`, and `rows` (`episode_id`, `candidate_count`, `policy`) for every fallback choice. During dry-run, these are planned choices and `embedding_text_unrecoverable` still describes the original source rows. `complete` reflects whether the chosen policy clears all input blockers and other inventory discrepancies. Dry-run writes no labels. Missing raw sources, inconsistent recorded inputs, and narratives that cannot satisfy exact preservation remain blocked even with the flag.

Remove `--dry-run` to perform the migration. If the earlier attempt installed a fence, drain as above and replace `--dry-run` with `--resume`, keeping `--legacy-consolidation-input longest-prefix` to authorize the initial plan. Each selected row receives `consolidation_embedding_input` with `synthesized_narrative` equal to its entire narrative and `protected_source_lines` equal to the collected lines, in the **same LanceDB batch as its new vector**. Verification checks:

```text
preserveProtectedEpisodeTokenLines(input.synthesized_narrative, input.protected_source_lines) === narrative
```

If the old episodes schema lacks the input column, staging adds a nullable string column and leaves it null on other rows. All existing non-vector fields, including narratives, remain unchanged. Unambiguous legacy rows keep their exact reconstruction and receive no label.

The fallback is **one-way: it labels the rows**. Subsequent runs use the persisted choice and find those rows unambiguous; omitting the flag does not undo it. Reverting the choice requires the matching pre-migration snapshot/directory and profile as described under rollback. The journal retains both the original source inventory and expected labelled target inventory, so backups verify against the original bank and resume detects changed sources. Once journaled, `--resume` reuses the recorded policy and choices even if the flag is omitted; an existing migration's policy cannot be changed. `--verify-only` needs no fallback flag and checks the recorded inputs. Completion and verification reports retain the resolved ids and report zero remaining unrecoverable inputs.

Progress and the final report are JSON lines on stdout; failures are JSON on stderr with a distinct code and a nonzero exit status. Keep the report and backup location with the deployment record. Backup files and reports contain private bank data; preserve their access controls.

For the five tenants in one sequential command:

```sh
nice -n 19 node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --all-tenants \
  --source-model generative-apis/qwen3-embedding-8b \
  --target-model scw/bge-m3 --target-dims 1024
```

A real run stops at the first failing tenant, after reporting all of that bank's unrecoverable inputs. Add `--legacy-consolidation-input longest-prefix` when the preview calls for the recorded fallback. For a busy tenant, drain that tenant and rerun with `--resume`; already completed tenants are skipped. During a planned outage, stop/drain all five before starting this command. Do not terminate the container that is executing the migration.

## Verify, then switch the sidecar

Before restoring writes, verify the migrated banks independently:

```sh
nice -n 19 node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --all-tenants \
  --target-model scw/bge-m3 --target-dims 1024 --verify-only
```

`--verify-only` uses the journal and backup to repeat exact migration verification without embedding calls or cutover. It needs the bank's access lease; evict an open tenant first. A verified staging directory with cutover still pending returns an incomplete result/nonzero exit; use `--resume` to finish. Exact baseline comparison is intended before new production writes; legitimate later changes need normal operational checks instead.

Confirm that all five reports succeeded and all five profiles record `scw/bge-m3`, 1024 dimensions, and the expected generation. Then set the sidecar deployment environment and restart:

```text
EMBEDDING_MODEL=scw/bge-m3
EMBEDDING_DIMS=1024
```

Remove `EMBEDDING_LEGACY_SOURCE_MODEL` from the deployment environment. Remove any temporary `BORG_EMBEDDING_LEGACY_SOURCE_MODEL` or tenant `embedding.legacySourceModel` fallback as well. Keep the assertion absent after the cutover; every existing bank must now carry its persisted target profile.

The sidecar's default constants remain unchanged. Validate through the actual sidecar for **each** tenant: authenticated `GET /memory/episodes?tenant=<tenant>&limit=3` should open the bank successfully; `POST /memory/recall` with a representative query should return HTTP 200 without an embedding/profile degradation. For example:

```sh
nice -n 19 node --input-type=module - <<'JS'
for (const tenant of ['team-agent-ai', 'team-agent-esb', 'team-agent-rtm', 'team-agent-tn', 'team-agent-tni']) {
  const response = await fetch('http://127.0.0.1:8088/memory/recall', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-borg-token': process.env.BORG_MEMORY_TOKEN },
    body: JSON.stringify({ tenant, query: 'What were our recent decisions?', limit: 3 }),
  });
  const body = await response.json();
  console.log(tenant, response.status, JSON.stringify(body));
  if (!response.ok || body.degraded) process.exitCode = 1;
}
JS
```

`/healthz` is liveness only and does not prove tenant compatibility. Profile mismatches (including a different model with the same dimensions), unverifiable schemas, and migration fences produce tenant-unavailable/degraded responses. Injected client identity is checked; editing `config.json` cannot bypass the effective-client guard. Legacy adoption requires the explicit source assertion as well as matching table dimensions/types. The library equivalents are `embedding.legacySourceModel` in config, `BORG_EMBEDDING_LEGACY_SOURCE_MODEL`, and the `embeddingLegacySourceModel` open option. An `embeddingProfile` option is an additional assertion, never a substitute for an injected client's own profile.

Queued reviews, saved plans, and semantic audit reversals keep their historical data. When materialized, their serialized vectors are recomputed from stored text using the effective client even if dimensions happen to match. A payload without recoverable text is rejected with `SERIALIZED_EMBEDDING_INCOMPATIBLE`; regenerate it or reject/dismiss the review. The migration does not automatically resolve or delete such items.

## Interrupted runs and rollback

Do not remove `embedding-migration.lock` to clear an error. Use the same tenant and target arguments with `--resume`. `.embedding-migration.json` records the source/target profiles, immutable backup location, source inventory, any consolidation policy/choices and expected target inventory, and phase. A crash between renames leaves either old+staging, previous+staging with no live directory, or previous+new live; the fence prevents any sidecar open until recovery verifies and completes cutover. Corrupted checkpoints, changed source data, backup checksum failures, or unexpected directory combinations stop recovery for investigation.

Ordinary owner/access locks are reaped automatically for dead local PIDs. If a pod restart changes the hostname, their ownership cannot safely be inferred. After confirming the former pod is gone and no process is using the tenant, remove only its stale `.embedding-migration-owner.lock` / `embedding-bank-access.lock` (and a stale checkpoint append `.lock`, if present), then resume. Keep the persistent migration fence. Never delete a live process's ownership lock.

For rollback, stop the sidecar and other bank writers. Preserve the failed target directories and journals for diagnosis. For each tenant, with `<N>` taken from its migration report:

```sh
MIGRATION_GENERATION=1 # Set this to N from the report.
mv /data/team-agent-ai/lancedb "/data/team-agent-ai/lancedb.failed-$MIGRATION_GENERATION"
mv "/data/team-agent-ai/lancedb.prev-$MIGRATION_GENERATION" /data/team-agent-ai/lancedb
```

Restore **the matching source profile**, not just the old vector directory: copy `embedding-profile.json` from the verified backup if it existed. For a legacy source, the journal's `source` object is a valid profile to write atomically using `writeJsonFileAtomic`. Move the migration journal, checkpoints, and target report outside the tenant directory so a later run cannot mistake rollback for a completed migration; remove the fence only once the restored bank/profile pair has been checked. Restore `EMBEDDING_MODEL=generative-apis/qwen3-embedding-8b` and `EMBEDDING_DIMS=4096`, restart, and repeat per-tenant sidecar verification.

Directory-only rollback is appropriate before new production writes. If target-model serving has already changed SQLite, stream data, or other memory state, restore the **entire verified tenant backup**, including its SQLite snapshot, to avoid joining old vectors to new metadata. Retain the newer bank separately; restoring a snapshot loses post-backup writes unless they are separately recovered. If a restored full snapshot has no profile, temporarily restore the legacy source assertion from the preconditions while the old-model sidecar adopts it. Do not overlay backup SQLite onto an open database or leave old WAL/SHM files beside the restored snapshot.

## Sidecar embedding environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `generative-apis/qwen3-embedding-8b` | Actual model used by the one shared injected client and its explicit profile. |
| `EMBEDDING_DIMS` | `4096` | Positive integer dimension expected from that client and required in every bank table. |
| `EMBEDDING_LEGACY_SOURCE_MODEL` | unset | Explicit source model assertion for adopting existing unlabelled banks; must equal the effective client model. Remove after all tenants migrate. |

The gateway URL and credential for the shared sidecar client come from `KRATOS_BASE_URL` and `LLM_API_KEY`; TLS uses `NODE_EXTRA_CA_CERTS`. The three `BORG_EMBEDDING_STALL_*` controls affect transport timing, not bank identity: `BORG_EMBEDDING_STALL_TIMEOUT_MS` defaults to 1000 per single attempt, `BORG_EMBEDDING_STALL_BATCH_TIMEOUT_MS` to 20000 per batch attempt, and `BORG_EMBEDDING_STALL_RETRIES` to 1 retry. Library `BORG_EMBEDDING_MODEL`/`BORG_EMBEDDING_DIMS` settings do not override the sidecar's injected client identity.

If `EMBEDDING_LEGACY_SOURCE_MODEL` is unset, the library source assertion (`BORG_EMBEDDING_LEGACY_SOURCE_MODEL` or tenant `embedding.legacySourceModel`) is the fallback. The library also parses `BORG_EMBEDDING_BASE_URL` and `BORG_EMBEDDING_API_KEY`; those do not replace the shared sidecar gateway/credential.
