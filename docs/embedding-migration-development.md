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

### Initial validation

Validation used Node 22.23.2 and `TMPDIR=$HOME/.cache/borg-bge-m3/tmp`, with logs under `$HOME/.cache/borg-bge-m3/logs`; no test cache used the small `/tmp` tmpfs. Embeddings were mocked. No production bank or gateway was accessed.

- `npm run typecheck`: all five TypeScript projects pass.
- Focused guard, pool, sidecar, payload, repository, migration, and eval coverage: 19 files / 377 tests passed. The final migration suite separately passed all 16 tests, including actual 4096-to-1024 schemas, all seven tables, complete row preservation, empty tables, dry-run, bounded retries/concurrency, interrupted batches, and recovery after either cutover rename. The final embedding client suite passed 13 tests.
- The required full run, `npx vitest run --maxWorkers=2`, ran once: 369 files passed, 6 failed; 4,443 tests passed, 6 failed, 1 todo (1,104.45 seconds). Two failures were Borg fixtures that needed the new opening order/model identity; those fixtures were corrected and both complete files passed (28 tests). Three failures exceeded the default 15-second test timeout; all three passed individually with `--maxWorkers=1 --testTimeout=60000`.
- The remaining full-run failure is an exact floating-point comparison in `src/retrieval/recall-core.test.ts`, "maps N=3 variants to semantic lanes without changing episode fusion": `0.0038047635709747476` versus `0.0038047635709747467`. Running that test on the original `dev` commit `b1e37f5a` with the same Node version reproduces the identical failure. It was left unchanged. The full suite was not rerun after the fixture corrections.

The CLI help invocation was checked under Node 22. Production gateway credentials, BGE-M3 semantic quality/latency, actual pod volume capacity, and live Kubernetes drain/restart/rollback remain operator verification steps. Fault injection tests exercise process interruption around commits/renames; they cannot establish the persistence guarantees of the production storage device during power loss.

### Initial file inventory

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
- Existing banks require a persisted profile. Restored pre-profile backups use the explicit `label-source-profile` operator subcommand after stored-dimension verification. Brand-new banks initialize a generation-zero profile. Schema validation requires float32 FixedSizeList vectors in all registered stores before storage migrations or reconciliation.
- Consolidation persists `consolidation_embedding_input` (synthesized narrative and protected source lines). The migration preserves the source schema, including legacy tables without that column. Legacy recovery loads archived raw lineage in its persisted writer order and inverts the append operation only when every possible prefix renders identical embedding text. Ambiguous inline protocol prose fails with `EMBEDDING_TEXT_UNRECOVERABLE`; the persisted narrative alone cannot establish the original input. The production writer, outcome-corpus writer, saved-plan preparation, and migration all retain this provenance.
- Backup copying excludes migration bookkeeping and previous backup manifests. Resume reuses its journal destination, checks capacity before any new backup/staging attempt, and validates committed rows before accounting for remaining work. Each unfinished table retains a full fragment allowance rather than assuming average row sizes.
- CLI tenant discovery is strict, while the pool keeps its existing best-effort discovery behavior.
- Regression coverage includes abrupt child-process exits after backup, each rename, and profile/journal completion; source changes; rollback then remigration; and the runbook shell setup in a simulated read-only pod app with a writable tsx cache.

## Files changed by the review fixes

- `docs/embedding-migration-development.md`
- `docs/embedding-migration.md`
- `eval/recall-planner-ab/instrumentation.ts`
- `eval/support/embedding.ts`
- `scripts/embedding-migration-runbook.test.ts`
- `scripts/embedding-migration/backup.ts`
- `scripts/embedding-migration/inventory.ts`
- `scripts/embedding-migration/migrate.ts`
- `scripts/memory-sidecar-main.ts`
- `scripts/migrate-embeddings.test.ts`
- `scripts/migrate-embeddings.ts`
- `scripts/migrate-outcome-corpus.ts`
- `src/autonomy/integration.test.ts`
- `src/borg-sprint7.test.ts`
- `src/borg.sprint-28.test.ts`
- `src/borg/__tests__/episodic-api.test.ts`
- `src/borg/__tests__/test-helpers.ts`
- `src/borg/open.ts`
- `src/borg/tenant-directories.test.ts`
- `src/borg/tenant-directories.ts`
- `src/borg/types.ts`
- `src/cli/app.test.ts`
- `src/cli/correction.test.ts`
- `src/cognition/session-lock.ts`
- `src/config/index.ts`
- `src/correction/service.test.ts`
- `src/embeddings/bank-profile.test.ts`
- `src/embeddings/bank-profile.ts`
- `src/embeddings/cache.ts`
- `src/embeddings/index.ts`
- `src/embeddings/serialized.test.ts`
- `src/embeddings/serialized.ts`
- `src/memory/episodic/protected-lines.test.ts`
- `src/memory/episodic/protected-lines.ts`
- `src/memory/episodic/repository.ts`
- `src/memory/episodic/types.ts`
- `src/offline/consolidator/index.test.ts`
- `src/offline/consolidator/index.ts`
- `src/offline/test-support.ts`
- `src/sidecar/memory-handler.test.ts`
- `src/stream/file-lock.test.ts`
- `src/stream/file-lock.ts`

## Regression coverage by finding

| Finding | Covering tests |
| --- | --- |
| 1. Effective injected identity | `src/embeddings/bank-profile.test.ts`: unprofiled clients (bare and cached), config-independent identity, same-dimension model mismatch; `scripts/migrate-embeddings.test.ts`: client identity before fencing; `src/sidecar/memory-handler.test.ts`: typed HTTP 503 mapping. |
| 2. Required bank profiles | `src/embeddings/bank-profile.test.ts`: missing-profile refusal and fresh initialization; `scripts/embedding-migration/source-profile.test.ts`: explicit restored-backup labelling, dimensions, lease/fence enforcement, and overwrite refusal; sidecar 503 mapping. |
| 3. Complete vector schema validation | `src/embeddings/bank-profile.test.ts`: missing embedding column in all seven tables, float64/int32 elements, variable lists, dimension mismatch, and no profile/SQLite writes. |
| 4. Consolidation text parity | `src/offline/consolidator/index.test.ts`: real writer with copied/appended inline legacy lines; `src/memory/episodic/protected-lines.test.ts`: ambiguity and stale provenance; `scripts/migrate-embeddings.test.ts`: archived lineage order and unchanged legacy schema; `src/embeddings/serialized.test.ts`: saved-plan provenance. |
| 5. Restored backup manifests | `scripts/migrate-embeddings.test.ts`: rollback to a verified snapshot followed by a new complete migration and backup verification. |
| 6. Headroom on every attempt | `scripts/migrate-embeddings.test.ts`: no progress callback, persisted backup destination on resume, staging interruption/remaining work, and a large final row. |
| 7. Strict tenant discovery | `src/borg/tenant-directories.test.ts`: inaccessible directories/databases and symlink candidates; `scripts/migrate-embeddings.test.ts`: actual all-tenants CLI exits nonzero without migrating another bank. |
| 8. Real pod runbook | `scripts/embedding-migration-runbook.test.ts`: actual documented setup under a simulated read-only app, non-PATH Node, writable data cache, and nice; old unwritable TMPDIR reproduces EACCES. |
| 9. Shared file-lock lease | `src/stream/file-lock.test.ts`: contention, idempotent release, dead owner, I/O failure; existing session, bank lease, and pool drain tests. |

Additional recovery regressions in `scripts/migrate-embeddings.test.ts` use abrupt process exits after backup, both cutover renames, and completion before fence removal, and reject a source text change after a checkpoint.

## Final review-fix verification

Runtime: Node 22.23.2, Vitest 4.1.9. `TMPDIR=$HOME/.cache/borg-bge-fixes/tmp` and `npm_config_cache=$HOME/.cache/borg-bge-fixes/npm`; complete output logs are under `$HOME/.cache/borg-bge-fixes/logs`. All embedding calls used test doubles; no production bank or provider was accessed. Runtime changes were tested at commit `361a1977`; the subsequent commit records documentation only.

`npm run typecheck` exited 0. Exact output (`typecheck-final.log`):

```text
> borg@0.1.0 typecheck
> tsc --noEmit && tsc --noEmit -p tsconfig.eval.json && tsc --noEmit -p tsconfig.assessor.json && tsc --noEmit -p tsconfig.simulator.json && tsc --noEmit -p tsconfig.test.json
```

Focused Vitest command (exit 0; `touched-tests-final.log`):

```sh
npx vitest run scripts/migrate-embeddings.test.ts scripts/embedding-migration-runbook.test.ts scripts/migrate-outcome-corpus.test.ts src/borg/tenant-directories.test.ts src/memory/episodic src/offline/consolidator src/embeddings src/offline/orchestrator.test.ts src/offline/audit-log.test.ts src/memory/review-queue/review-queue.test.ts src/stream/file-lock.test.ts src/cognition/session-lock.test.ts src/borg/__tests__ src/borg/pool.test.ts src/borg/lifecycle.test.ts src/sidecar/memory-handler.test.ts src/config/index.test.ts src/cli/app.test.ts src/cli/correction.test.ts src/borg-sprint7.test.ts src/borg.sprint-28.test.ts src/autonomy/integration.test.ts src/correction/service.test.ts eval/recall-planner-ab eval/embedding-ab eval/support --maxWorkers=2
```

```text
 Test Files  55 passed (55)
      Tests  687 passed (687)
   Start at  13:03:40
   Duration  171.89s (transform 13.82s, setup 0ms, import 71.32s, tests 262.38s, environment 8ms)
```

One full `npx vitest run --maxWorkers=2` was run (exit 1; `full-vitest.log`). Exact summary:

```text
 Test Files  5 failed | 372 passed (377)
      Tests  5 failed | 4484 passed | 1 todo (4490)
   Start at  13:07:17
   Duration  987.97s (transform 18.24s, setup 0ms, import 204.41s, tests 1714.79s, environment 48ms)
```

Four failures were `Error: Test timed out in 15000ms.` They were each rerun sequentially with `npx vitest run <file> -t <name> --maxWorkers=1 --testTimeout=60000`; every rerun exited 0. The default suite timeout remains unchanged. These reruns verify the assertions; they do not make the original full run green.

`assessor/scenarios/index.test.ts` — "runs every scenario through the scripted mock path and produces a report" (`rerun-assessor.log`):

```text
 Test Files  1 passed (1)
      Tests  1 passed | 12 skipped (13)
   Start at  13:24:46
   Duration  25.95s (transform 4.20s, setup 0ms, import 5.49s, tests 20.30s, environment 1ms)
```

`demo/server/src/__tests__/server.test.ts` — "caps activity rows for a day and marks the response truncated" (`rerun-activity.log`):

```text
 Test Files  1 passed (1)
      Tests  1 passed | 88 skipped (89)
   Start at  13:25:13
   Duration  12.75s (transform 4.42s, setup 0ms, import 5.86s, tests 6.74s, environment 0ms)
```

`src/autonomy/triggers/executive-focus-due.test.ts` — "returns null instead of scanning beyond the due-step observability candidate cap" (`rerun-executive.log`):

```text
 Test Files  1 passed (1)
      Tests  1 passed | 27 skipped (28)
   Start at  13:25:26
   Duration  8.05s (transform 2.55s, setup 0ms, import 3.50s, tests 4.41s, environment 0ms)
```

`src/cognition/evidence-ledger/builder.test.ts` — "traces reverse-scan count, bytes, and cap hits" (`rerun-ledger.log`):

```text
 Test Files  1 passed (1)
      Tests  1 passed | 56 skipped (57)
   Start at  13:25:35
   Duration  12.02s (transform 2.88s, setup 0ms, import 3.85s, tests 8.02s, environment 0ms)
```

The remaining assertion failure is `src/retrieval/recall-core.test.ts:411`, "maps N=3 variants to semantic lanes without changing episode fusion": expected `0.0038047635709747476`, received `0.0038047635709747467`. The identical test and difference were reproduced on the archived original `dev` commit `b1e37f5a7c8ad38acfe2c106126c06c67c29e41c`, with the same Node version (exit 1; `dev-floating-point.log`):

```sh
npx vitest run src/retrieval/recall-core.test.ts -t 'maps N=3 variants to semantic lanes without changing episode fusion' --maxWorkers=1
```

```text
 Test Files  1 failed (1)
      Tests  1 failed | 40 skipped (41)
   Start at  13:06:51
   Duration  4.26s (transform 2.78s, setup 0ms, import 3.68s, tests 430ms, environment 0ms)
```

## Complete sidecar embedding environment contract

| Variable | Default | Effective behavior |
| --- | --- | --- |
| `EMBEDDING_MODEL` | required | Model sent by the shared OpenAI-compatible client and declared in its explicit profile. |
| `EMBEDDING_DIMS` | required | Positive integer returned-vector dimension and bank schema requirement. |
| `BORG_EMBEDDING_STALL_TIMEOUT_MS` | `1000` | Timeout in milliseconds per single-text embedding attempt. |
| `BORG_EMBEDDING_STALL_BATCH_TIMEOUT_MS` | `20000` | Timeout in milliseconds per batch attempt. |
| `BORG_EMBEDDING_STALL_RETRIES` | `1` | Number of retries after an attempt stalls. |
| `BORG_EMBEDDING_MODEL` | config value; library default `text-embedding-qwen3-embedding-8b` | Parsed library setting; does not replace the sidecar's injected model/profile. |
| `BORG_EMBEDDING_DIMS` | config value; library default `4096` | Parsed library setting; does not replace the sidecar's explicit injected dimensions. |
| `BORG_EMBEDDING_BASE_URL` | config value; library default `http://localhost:1234/v1` | Parsed library setting; the shared sidecar client uses `KRATOS_BASE_URL` instead. Also a CLI fallback when `KRATOS_BASE_URL` is absent. |
| `BORG_EMBEDDING_API_KEY` | config value; library default `lm-studio` | Parsed library setting; the shared sidecar client requires `LLM_API_KEY` instead. Also a CLI fallback when `LLM_API_KEY` is absent. |

All five production banks now carry `scw/bge-m3` / 1024 generation-one profiles. Existing banks must retain their persisted profile; restored pre-profile backups must be labelled offline through the operator subcommand documented in [the migration runbook](embedding-migration.md#restored-pre-profile-backups).


## Legacy consolidation resolution after the production dry-run

The production dry-run exposed ambiguous legacy rows with no retained original synthesis. This change supersedes the review-fix requirement to recover that original synthesis before migrating. Strict reconstruction remains the default; the operator can now explicitly choose `--legacy-consolidation-input longest-prefix` for re-embedding to a new model.

- The runtime builder enumerates every valid append boundary and counts distinct rendered inputs. Its typed error exposes the zero-appended-lines candidate only when it exactly reproduces the stored narrative. Runtime callers remain strict.
- Inventory records every unrecoverable episode, its reason and candidate count, without stopping the scan. An unresolved row has a null text hash and retains its non-vector field hash. Completed dry-run inventories exit zero even when the report marks a migration blocked. Real runs emit the complete inventory and error report before stopping.
- The opt-in selects the entire persisted narrative as synthesized prose, plus protected raw source lines in writer order. The source inventory remains immutable for source-change checks and backup verification. A separate expected target inventory accounts for the chosen input labels and, where necessary, the added nullable column.
- The durable journal pins the policy, per-row choices and target identities. Staging writes each label with its new vector, and checkpoints hash the resulting target fields and text. Resume reuses that plan, including when the flag is omitted. A completed strict migration can be skipped when an all-tenant resume opts in for remaining banks.
- Every cutover/verification path uses the expected target inventory and checks the exact runtime narrative-preservation invariant for labelled rows. Unambiguous rows retain their original reconstruction; other non-vector fields remain unchanged. Missing sources and inconsistent recorded inputs still block a migration.
- Optional inventory/journal properties are omitted when unused so existing strict journals and backup fingerprints remain compatible.

### Files changed for this request

Paths are relative to the repository root; baseline is `ccded9f3`.

- `src/memory/episodic/protected-lines.ts`
- `src/memory/episodic/protected-lines.test.ts`
- `scripts/embedding-migration/inventory.ts`
- `scripts/embedding-migration/migrate.ts`
- `scripts/migrate-embeddings.ts`
- `scripts/migrate-embeddings.test.ts`
- `docs/embedding-migration.md`
- `docs/embedding-migration-development.md`

### Regression coverage

| Requirement | Covering test |
| --- | --- |
| Full dry-run inventory, counts, ids, distinct candidate counts, exit zero across tenants | `scripts/migrate-embeddings.test.ts`: reports every ambiguous legacy input in a mixed dry-run inventory and exits zero across tenants |
| Real run reports the full list then fails before backup or staging | Same file: fails a real run only after reporting the full unrecoverable list, including missing raw sources |
| Flag selection, nullable legacy column, unchanged other fields, vector/input equality, verify-only, subsequent dry-run without ambiguity | Same file: labels only ambiguous legacy inputs with their vectors and verifies them (both schema layouts) |
| Atomic label/vector persistence and checkpoint replay/skip | Same file: resumes labelled vectors after a checkpointed/uncheckpointed interruption without changing the recorded policy |
| Recovery after both renames with labelled target inventory | Same file: recovers labelled rows after the previous/live cutover rename |
| Source changes, strict remaining blockers, completed-tenant resume | Same file: rejects a changed source after recording fallback choices; keeps missing or inconsistent inputs blocked even with the longest-prefix opt-in; skips completed strict migrations when an all-tenant resume opts in for remaining banks |
| Real runtime open and input rendering | Same file: opens a migrated bank through Borg and renders the persisted fallback with the runtime recipe |
| Every candidate counted; fallback preserves the exact narrative | `src/memory/episodic/protected-lines.test.ts`: counts every distinct legacy candidate and exposes only a narrative-preserving longest prefix |

### Verification

Node 22.23.2, Vitest 4.1.9. `TMPDIR=$HOME/.cache/borg-bge-longest-prefix/tmp`, `npm_config_cache=$HOME/.cache/borg-bge-longest-prefix/npm`; logs are under `$HOME/.cache/borg-bge-longest-prefix/logs`. All embedding/LLM clients in the regressions are fakes. No production bank or gateway was accessed.

`npm run typecheck` exited 0; exact output (`typecheck-final.log`):

```text
> borg@0.1.0 typecheck
> tsc --noEmit && tsc --noEmit -p tsconfig.eval.json && tsc --noEmit -p tsconfig.assessor.json && tsc --noEmit -p tsconfig.simulator.json && tsc --noEmit -p tsconfig.test.json
```

The touched tests and related writer/repository/runtime/runbook coverage ran with:

```sh
npx vitest run scripts/migrate-embeddings.test.ts scripts/embedding-migration-runbook.test.ts src/memory/episodic/protected-lines.test.ts src/memory/episodic/repository.test.ts src/memory/episodic/extractor.test.ts src/offline/consolidator/index.test.ts src/embeddings/serialized.test.ts src/embeddings/bank-profile.test.ts --maxWorkers=2 --testTimeout=60000 --hookTimeout=60000
```

Exit 0; exact summary (`touched-tests-final.log`):

```text
 Test Files  8 passed (8)
      Tests  168 passed (168)
   Start at  13:58:54
   Duration  71.15s (transform 5.41s, setup 0ms, import 10.95s, tests 85.97s, environment 1ms)
```

The full suite ran once, on `eda019f2`, with:

```sh
npx vitest run --maxWorkers=2 --testTimeout=60000 --hookTimeout=60000
```

Exit 1; exact summary (`full-vitest.log`):

```text
 Test Files  2 failed | 375 passed (377)
      Tests  2 failed | 4500 passed | 1 todo (4503)
   Start at  14:00:28
   Duration  1138.19s (transform 19.22s, setup 0ms, import 209.73s, tests 2007.96s, environment 50ms)
```

The two failures were:

- `assessor/scenarios/index.test.ts:32`, "runs every scenario through the scripted mock path and produces a report": `Test timed out in 60000ms` (60,379 ms). This same test timed out in the previous full review run.
- `src/retrieval/recall-core.test.ts:411`, "maps N=3 variants to semantic lanes without changing episode fusion": expected `decayedSalience: 0.0038047635709747476`, received `0.0038047635709747467`. These values exactly match the prior `dev` reproduction in `$HOME/.cache/borg-bge-fixes/logs/dev-floating-point.log`; that test and its implementation were not changed for this request.

The assessor timeout passed in isolation with the same 60-second test limit:

```sh
npx vitest run assessor/scenarios/index.test.ts -t 'runs every scenario through the scripted mock path and produces a report' --maxWorkers=1 --testTimeout=60000 --hookTimeout=60000
```

Exit 0; exact summary (`assessor-isolated.log`):

```text
 Test Files  1 passed (1)
      Tests  1 passed | 12 skipped (13)
   Start at  14:20:00
   Duration  24.42s (transform 4.11s, setup 0ms, import 5.42s, tests 18.85s, environment 0ms)
```

All requested regressions passed in both focused and full validation. The full run remains non-green because of the two failures above; it was not repeated. The pre-existing untracked `pnpm-lock.yaml` was left untouched. No sidecar environment variable was added or changed; the opt-in is a CLI flag, persisted per attempt in the migration journal and per resolved row in LanceDB.
