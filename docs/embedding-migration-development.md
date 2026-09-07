# Embedding migration development notes

Opus 5.0: in scope because embedding compatibility and crash recovery are structural storage correctness requirements.

## Implementation utility searches

- Searched for atomic file writes -> found `src/util/atomic-write.ts`; reuse `writeJsonFileAtomic` and `syncDirectory`.
- Searched for durable JSONL and interruption recovery -> found `src/util/durable-jsonl.ts` and `eval/embedding-ab/cache.ts`; reuse the append helper and committed-record pattern.
- Searched for cross-process locks -> found `src/stream/file-lock.ts` and the lifetime lease pattern in `src/cognition/session-lock.ts`; share `acquireFileLockLease` from `src/stream/file-lock.ts` between the bank lease and session lock, retaining those ownership rules. A separate persistent migration fence is necessary because ordinary locks are reaped after crashes.
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

## Initial implementation record

The following records validation before the review fixes; current verification is recorded separately below.

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

## Review corrections

- The injected client must supply its own valid model/dimensions profile. Cache and stall wrappers preserve the inner identity; a cache cannot manufacture it from its cache-key options. Configuration/open-option profiles are assertions only.
- Existing unlabelled banks require an explicit matching legacy source model. Brand-new banks initialize a generation-zero profile. Schema validation requires float32 FixedSizeList vectors in all registered stores before storage migrations or reconciliation.
- Consolidation persists `consolidation_embedding_input` (synthesized narrative and protected source lines). The migration preserves the source schema, including legacy tables without that column. Legacy recovery loads archived raw lineage in its persisted writer order and inverts the append operation only when every possible prefix renders identical embedding text. Ambiguous inline protocol prose fails with `EMBEDDING_TEXT_UNRECOVERABLE`; the persisted narrative alone cannot establish the original input. The production writer, outcome-corpus writer, saved-plan preparation, and migration all retain this provenance.
- Backup copying excludes migration bookkeeping and previous backup manifests. Resume reuses its journal destination, checks capacity before any new backup/staging attempt, and validates committed rows before accounting for remaining work. Each unfinished table retains a full fragment allowance rather than assuming average row sizes.
- CLI tenant discovery is strict, while the pool keeps its existing best-effort discovery behavior.
- Regression coverage includes abrupt child-process exits after backup, each rename, and profile/journal completion; source changes; rollback then remigration; and the runbook shell setup in a simulated read-only pod app with a writable tsx cache.
