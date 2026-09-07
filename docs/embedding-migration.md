# Embedding model migration

Opus 5.0: in-scope because embedding compatibility and crash-safe migration are structural storage correctness requirements.

## Implementation utility searches

- Searched for atomic file writes -> found `src/util/atomic-write.ts`; reuse `writeJsonFileAtomic` and `syncDirectory`.
- Searched for durable JSONL and interruption recovery -> found `src/util/durable-jsonl.ts` and `eval/embedding-ab/cache.ts`; reuse the append helper and committed-record pattern.
- Searched for cross-process locks -> found `src/stream/file-lock.ts` and the lifetime lease pattern in `src/cognition/session-lock.ts`; use those ownership rules for bank access. A separate persistent migration fence is necessary because ordinary locks are reaped after crashes.
- Searched for parsing -> found Zod and `src/util/parse.ts`; reuse those and Node's CLI parser rather than another argument parser.
- Searched for hashing -> found `sha256Bytes` and `fingerprintCanonicalValue` in `src/cognition/deliberation/request-fingerprint.ts`; reuse those for text, row and schema identities.
- Searched for retries/backoff -> found transport-specific loops and `src/util/clock.ts:sleep`, but no suitable shared migration retry policy; use the existing sleep with bounded embedding batch attempts.
- Searched for progress -> found `eval/embedding-ab/embed-items.ts:onBatch`; use the same completed/total reporting pattern, with durable checkpoints after LanceDB commits.
- Searched for backup and disk headroom -> found the manual recipe in `WORKFLOW.md`, no coordinated Node bank backup; use Node's SQLite backup API and filesystem APIs.
- Searched for streaming file checksums -> found buffer/text hashes but no streaming file hash; the migration's `fileChecksum` uses `createReadStream` + `createHash` to verify large files with bounded memory.
- Searched for filesystem walkers -> found unrelated snapshot/capture scripts, no suitable whole-bank walker; `bankFiles` counts bank bytes and rejects symlinks and special files before copying.
- Searched for tenant discovery -> found `BorgPool.listTenantIds`; extracted it to `src/borg/tenant-directories.ts` for the pool and CLI. The default `backups` directory is always excluded.
- Searched for read-only SQLite opens -> found `eval/embedding-ab/bank.ts:openReadOnlyDatabase`; moved it into the SQLite module and reused it from the harness and migration. `SqliteRawDatabase.backup` exposes the Node backup API without running SQL migrations.
- Searched for JSONL readers -> found the A/B cache's interrupted-tail reader; moved `parseJsonLines` into `durable-jsonl.ts`. Checkpoints store committed batch identities, not a second in-memory vector cache.
- Searched for Arrow/vector conversion -> found `toFloat32Array`, schema factories, and `LanceDbTable.upsert`; reuse them. The new vector validator additionally requires the expected length and nonzero norm. All-null Boolean batches expose an Arrow 18 IPC bug: staging alone temporarily writes Boolean values and restores nulls using Lance SQL before checkpointing. It never exposes those intermediate rows.

## Preconditions

Deploy this version of the sidecar **with the old embedding environment first**. Restart it so every open bank holds the new access lease. A sidecar from before this change does not honor migration fences; stop that process before migrating. Stop other direct bank-writing tools as well.

Run from `/app` in the pod, with Node >= 22.18.0, installed `tsx`, access to `/data`, and the pod's gateway credentials/CA environment. No `pnpm` is needed. The CLI uses `LLM_API_KEY` and `KRATOS_BASE_URL`, including the sidecar's existing default gateway URL. `BORG_EMBEDDING_API_KEY` and `BORG_EMBEDDING_BASE_URL` are fallbacks. It uses a 60-second request timeout and at most four attempts per batch (backoff 1, 4, 10 seconds). Batch size defaults to 32 and embedding concurrency to 2. Tenant processing is always sequential.

Do not change the shared sidecar model while tenants have mixed profiles. After each tenant migrates, the old-configured sidecar returns HTTP 503 for that tenant until the final environment cutover. Other tenants can continue serving. Plan for this per-tenant downtime.

All commands below name the source model explicitly, which is required for unlabelled legacy banks. The source dimension is read from **every existing table** and must be consistent; production should report 4096. No text is sent to an LLM. The migration uses the production embedding client directly.

## Inventory and capacity check

```sh
node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --tenant team-agent-ai \
  --source-model generative-apis/qwen3-embedding-8b \
  --target-model scw/bge-m3 --target-dims 1024 --dry-run
```

Repeat for `team-agent-esb`, `team-agent-rtm`, `team-agent-tn`, and `team-agent-tni`, or use `--all-tenants` instead of `--tenant team-agent-ai`. `--tenant` is repeatable. Discovery requires a valid tenant directory containing `borg.db`; backups, quarantines, and non-bank directories are excluded.

The JSON report includes all seven tables: `episodes`, `semantic_nodes`, `skills`, `open_questions`, `action_records`, `image_perception_embeddings`, and `observed_events`. It includes archived/superseded rows, source dimensions, exact row identities, SQL-only/vector-only ids, text disagreements, and serialized-vector locations in review refs, audit reversals/targets, and JSON files under the tenant. Saved plans outside the tenant cannot be inventoried; their application is still protected by the runtime payload guard.

A dry-run writes no profile, fence, backup, checkpoint, or staging table and makes no gateway request. SQLite's read-only WAL handling can create `borg.db-wal`/`borg.db-shm` coordination files. A live dry-run is **provisional**; the authoritative inventory runs after draining. SQL/vector discrepancies and unknown LanceDB tables block migration rather than silently discarding data. Resolve them with the old sidecar/appropriate repair operation before proceeding.

The capacity check is conservative: whole-bank backup bytes plus twice the bank's current size, three copies of target vector bytes, and 64 MiB for staging/metadata. It checks available blocks on both filesystems when the backup is elsewhere. Previous generations also count. Use `--backup-dir /mounted-backups/borg` to move backup storage; the destination must be outside every tenant directory. Symlinks and special files in a bank are refused. Incomplete backups remain as `.partial-*` directories for inspection; they are never silently reused or removed.

## Fence, drain, back up, migrate

Start the migration for one tenant:

```sh
node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --tenant team-agent-ai \
  --source-model generative-apis/qwen3-embedding-8b \
  --target-model scw/bge-m3 --target-dims 1024
```

The first action installs `embedding-migration.lock`. If the sidecar has that bank open, the CLI exits nonzero with `EMBEDDING_BANK_BUSY` and **leaves the fence in place**. Drain it through the authenticated route (Node's fetch works in the slim pod without curl):

```sh
node --input-type=module - <<'JS'
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
2. Copies the whole tenant into `/data/backups/<tenant>/generation-<N>-<timestamp>`. It copies `borg.db` with SQLite's backup API, not a live file copy. It checks SQLite integrity and table counts, exact inventory equality, file checksums, and fsyncs the copy before marking it complete.
3. Builds `lancedb.staging-<N>` beside `lancedb`, preserving all non-vector schema fields and row values. Embedding inputs use the production recipe for each table, including consolidation episodes and semantic observation metadata. Image `model` fields describe perception models and remain unchanged.
4. Commits bounded batches and fsyncs `.embedding-migration-g<N>.jsonl` checkpoints keyed by table/id/text hash (also recording non-vector field hashes and the target profile). Resume validates the staging rows before skipping them. A table commit interrupted before its checkpoint is safely replayed by id.
5. Verifies exact id coverage and unchanged non-vector fields/schema, SQLite integrity/counts, finite nonzero target-dimensional vectors, and up to three nearest-neighbour self-probes per nonempty table. These are storage/retrieval sanity checks, not a semantic-quality benchmark; use `eval/embedding-ab` for quality comparisons.
6. Journals cutover intent, renames `lancedb` to `lancedb.prev-<N>`, renames staging to `lancedb`, atomically writes `embedding-profile.json`, and removes the fence only after verification succeeds. It keeps the previous directory and writes `.embedding-migration-report-<N>.json`.

Progress and the final report are JSON lines on stdout; failures are JSON on stderr with a distinct code and a nonzero exit status. Keep the report and backup location with the deployment record. Backup files and reports contain private bank data; preserve their access controls.

For the five tenants in one sequential command:

```sh
node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --all-tenants \
  --source-model generative-apis/qwen3-embedding-8b \
  --target-model scw/bge-m3 --target-dims 1024
```

This stops at the first failure. For a busy tenant, drain that tenant and rerun with `--resume`; already completed tenants are skipped. During a planned outage, stop/drain all five before starting this command. Do not terminate the container that is executing the migration.

## Verify, then switch the sidecar

Before restoring writes, verify the migrated banks independently:

```sh
node --import tsx scripts/migrate-embeddings.ts \
  --data-root /data --all-tenants \
  --target-model scw/bge-m3 --target-dims 1024 --verify-only
```

`--verify-only` uses the journal and backup to repeat exact migration verification without embedding calls or cutover. It needs the bank's access lease; evict an open tenant first. A verified staging directory with cutover still pending returns an incomplete result/nonzero exit; use `--resume` to finish. Exact baseline comparison is intended before new production writes; legitimate later changes need normal operational checks instead.

Confirm that all five reports succeeded and all five profiles record `scw/bge-m3`, 1024 dimensions, and the expected generation. Then set the sidecar deployment environment and restart:

```text
EMBEDDING_MODEL=scw/bge-m3
EMBEDDING_DIMS=1024
```

The sidecar's default constants remain unchanged. Validate through the actual sidecar for **each** tenant: authenticated `GET /memory/episodes?tenant=<tenant>&limit=3` should open the bank successfully; `POST /memory/recall` with a representative query should return HTTP 200 without an embedding/profile degradation. For example:

```sh
node --input-type=module - <<'JS'
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

`/healthz` is liveness only and does not prove tenant compatibility. Profile mismatches (including a different model with the same dimensions), unverifiable schemas, and migration fences produce tenant-unavailable/degraded responses. Injected client identity is checked; editing `config.json` cannot bypass the effective-client guard. Legacy adoption happens only after actual table dimensions match the effective client.

Queued reviews, saved plans, and semantic audit reversals keep their historical data. When materialized, their serialized vectors are recomputed from stored text using the effective client even if dimensions happen to match. A payload without recoverable text is rejected with `SERIALIZED_EMBEDDING_INCOMPATIBLE`; regenerate it or reject/dismiss the review. The migration does not automatically resolve or delete such items.

## Interrupted runs and rollback

Do not remove `embedding-migration.lock` to clear an error. Use the same tenant and target arguments with `--resume`. `.embedding-migration.json` records the source/target profiles, immutable backup location, inventory, and phase. A crash between renames leaves either old+staging, previous+staging with no live directory, or previous+new live; the fence prevents any sidecar open until recovery verifies and completes cutover. Corrupted checkpoints, changed source data, backup checksum failures, or unexpected directory combinations stop recovery for investigation.

Ordinary owner/access locks are reaped automatically for dead local PIDs. If a pod restart changes the hostname, their ownership cannot safely be inferred. After confirming the former pod is gone and no process is using the tenant, remove only its stale `.embedding-migration-owner.lock` / `embedding-bank-access.lock` (and a stale checkpoint append `.lock`, if present), then resume. Keep the persistent migration fence. Never delete a live process's ownership lock.

For rollback, stop the sidecar and other bank writers. Preserve the failed target directories and journals for diagnosis. For each tenant, with `<N>` taken from its migration report:

```sh
mv /data/team-agent-ai/lancedb /data/team-agent-ai/lancedb.failed-<N>
mv /data/team-agent-ai/lancedb.prev-<N> /data/team-agent-ai/lancedb
```

Restore **the matching source profile**, not just the old vector directory: copy `embedding-profile.json` from the verified backup if it existed. For a legacy source, the journal's `source` object is a valid profile to write atomically using `writeJsonFileAtomic`. Archive the migration journal and target report so a later run cannot mistake rollback for a completed migration; remove the fence only once the restored bank/profile pair has been checked. Restore `EMBEDDING_MODEL=generative-apis/qwen3-embedding-8b` and `EMBEDDING_DIMS=4096`, restart, and repeat per-tenant sidecar verification.

Directory-only rollback is appropriate before new production writes. If target-model serving has already changed SQLite, stream data, or other memory state, restore the **entire verified tenant backup**, including its SQLite snapshot, to avoid joining old vectors to new metadata. Retain the newer bank separately; restoring a snapshot loses post-backup writes unless they are separately recovered. Do not overlay backup SQLite onto an open database or leave old WAL/SHM files beside the restored snapshot.

## Implementation validation

Validation used Node 22.23.2 and `TMPDIR=$HOME/.cache/borg-bge-m3/tmp`, with logs under `$HOME/.cache/borg-bge-m3/logs`; no test cache used the small `/tmp` tmpfs. Embeddings were mocked. No production bank or gateway was accessed.

- `npm run typecheck`: all five TypeScript projects pass.
- Focused guard, pool, sidecar, payload, repository, migration, and eval coverage: 19 files / 377 tests passed. The final migration suite separately passed all 16 tests, including actual 4096-to-1024 schemas, all seven tables, complete row preservation, empty tables, dry-run, bounded retries/concurrency, interrupted batches, and recovery after either cutover rename. The final embedding client suite passed 13 tests.
- The required full run, `npx vitest run --maxWorkers=2`, ran once: 369 files passed, 6 failed; 4,443 tests passed, 6 failed, 1 todo (1,104.45 seconds). Two failures were Borg fixtures that needed the new opening order/model identity; those fixtures were corrected and both complete files passed (28 tests). Three failures exceeded the default 15-second test timeout; all three passed individually with `--maxWorkers=1 --testTimeout=60000`.
- The remaining full-run failure is an exact floating-point comparison in `src/retrieval/recall-core.test.ts`, "maps N=3 variants to semantic lanes without changing episode fusion": `0.0038047635709747476` versus `0.0038047635709747467`. Running that test on the original `dev` commit `b1e37f5a` with the same Node version reproduces the identical failure. It was left unchanged. The full suite was not rerun after the fixture corrections.

The CLI help invocation was checked under Node 22. Production gateway credentials, BGE-M3 semantic quality/latency, actual pod volume capacity, and live Kubernetes drain/restart/rollback remain operator verification steps. Fault injection tests exercise process interruption around commits/renames; they cannot establish the persistence guarantees of the production storage device during power loss.

## Implementation file inventory

Paths are relative to the repository root.

| Change | File |
| --- | --- |
| Modified | `README.md` |
| Created | `docs/embedding-migration.md` |
| Modified | `eval/embedding-ab/bank.ts` |
| Modified | `eval/embedding-ab/cache.ts` |
| Created | `scripts/embedding-migration/backup.ts` |
| Created | `scripts/embedding-migration/inventory.ts` |
| Created | `scripts/embedding-migration/migrate.ts` |
| Modified | `scripts/memory-sidecar-main.ts` |
| Created | `scripts/migrate-embeddings.test.ts` |
| Created | `scripts/migrate-embeddings.ts` |
| Modified | `src/borg/__tests__/config-and-facade.test.ts` |
| Modified | `src/borg/__tests__/turn-ingestion.test.ts` |
| Modified | `src/borg/lifecycle.ts` |
| Modified | `src/borg/offline-setup.ts` |
| Modified | `src/borg/open.ts` |
| Modified | `src/borg/pool.test.ts` |
| Modified | `src/borg/pool.ts` |
| Modified | `src/borg/repositories.ts` |
| Created | `src/borg/tenant-directories.ts` |
| Modified | `src/borg/types.ts` |
| Created | `src/embeddings/bank-profile.test.ts` |
| Created | `src/embeddings/bank-profile.ts` |
| Modified | `src/embeddings/cache.ts` |
| Modified | `src/embeddings/index.test.ts` |
| Modified | `src/embeddings/index.ts` |
| Created | `src/embeddings/serialized.test.ts` |
| Created | `src/embeddings/serialized.ts` |
| Modified | `src/embeddings/stall-guard.ts` |
| Modified | `src/memory/episodic/extractor.ts` |
| Modified | `src/memory/episodic/protected-lines.test.ts` |
| Modified | `src/memory/episodic/protected-lines.ts` |
| Modified | `src/memory/observed-events/repository.ts` |
| Modified | `src/memory/review-queue/review-queue.test.ts` |
| Modified | `src/memory/review-queue/review-queue.ts` |
| Modified | `src/memory/self/open-questions.ts` |
| Created | `src/memory/semantic/embedding-text.ts` |
| Modified | `src/memory/semantic/extractor.ts` |
| Modified | `src/offline/audit-log.test.ts` |
| Modified | `src/offline/audit-log.ts` |
| Modified | `src/offline/orchestrator.test.ts` |
| Modified | `src/offline/orchestrator.ts` |
| Created | `src/sidecar/gateway-config.ts` |
| Modified | `src/sidecar/memory-handler.test.ts` |
| Modified | `src/sidecar/memory-handler.ts` |
| Modified | `src/storage/sqlite/index.ts` |
| Modified | `src/util/durable-jsonl.ts` |
| Modified | `tsconfig.test.json` |
