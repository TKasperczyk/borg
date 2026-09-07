# Embedding model migration

Opus 5.0: in-scope because embedding compatibility and crash-safe migration are structural storage correctness requirements.

## Implementation utility searches

- Searched for atomic file writes -> found `src/util/atomic-write.ts`; reuse `writeJsonFileAtomic` and `syncDirectory`.
- Searched for durable JSONL and interruption recovery -> found `src/util/durable-jsonl.ts` and `eval/embedding-ab/cache.ts`; reuse the append helper and committed-record pattern.
- Searched for cross-process locks -> found `src/stream/file-lock.ts` and the lifetime lease pattern in `src/cognition/session-lock.ts`; use those ownership rules for bank access. A separate persistent migration fence is necessary because ordinary locks are reaped after crashes.
- Searched for parsing -> found Zod and `src/util/parse.ts`; reuse those and Node's CLI parser rather than another argument parser.
- Searched for hashing -> found `node:crypto.createHash` throughout the A/B harness; use it for migration inventories and checkpoint identities.
- Searched for retries/backoff -> found transport-specific loops and `src/util/clock.ts:sleep`, but no suitable shared migration retry policy; use the existing sleep with bounded embedding batch attempts.
- Searched for progress -> found `eval/embedding-ab/embed-items.ts:onBatch`; use the same completed/total reporting pattern, with durable checkpoints after LanceDB commits.
- Searched for backup and disk headroom -> found the manual recipe in `WORKFLOW.md`, no coordinated Node bank backup; use Node's SQLite backup API and filesystem APIs.

The runbook below will describe the storage-only CLI, live-sidecar drain, verification, and rollback.
