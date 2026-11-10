# Contradiction edge backfills

Stop all writers and use a backed-up rehearsal bank with current migrations.
No default data directory is opened; the live demo path is refused by default.
Pending/unknown migrations and other pending repairs are refused.

`backfill-judged-contradiction-edges.ts` selects open `contradicts` edges referenced
by resolved contradiction reviews with `keep_both` or `dismiss`. The review's
`edge_id` must match both `node_ids`. It excludes handler-bypass finalizations
marked `resolver_finalized_without_handler`; earlier failed-attempt diagnostics
do not exclude a later judgment. Duplicate edge IDs are retired once.

`backfill-superseded-contradiction-edges.ts` selects open `contradicts` edges with
at least one endpoint whose status is `superseded`, deduplicating edges when both
endpoints qualify. Archival or a supersession timestamp alone does not qualify.
The two repair sets can overlap, so applying one can reduce the other's count.

Both scripts default to a read-only dry run:

```sh
pnpm tsx scripts/backfill-judged-contradiction-edges.ts --data-dir /home/luth/tmp/bank-copy
pnpm tsx scripts/backfill-superseded-contradiction-edges.ts --data-dir /home/luth/tmp/bank-copy
pnpm tsx scripts/backfill-judged-contradiction-edges.ts --data-dir /home/luth/tmp/bank-copy --apply
pnpm tsx scripts/backfill-judged-contradiction-edges.ts --data-dir /home/luth/tmp/bank-copy --apply --resume
```

The same `--apply` and `--apply --resume` flags work for the superseded-edge script.
For a stopped-service live invocation, add `--live-store-confirmed <backup path>`
to the dry-run/apply commands with the live `--data-dir`. The backup must be an
existing directory containing `borg.db` or `borg.db-clean`, distinct from the
target database (including symlink/hard-link aliases). The flag does not stop
writers or verify backup freshness. It records the canonical backup directory in
`live_store_override` in the plan, journal and audits. Repeat the same flag and
backup path on `--resume`; a recorded override alone grants no later access.

The JSON plan includes before/after edge snapshots and judgment or endpoint
provenance. Its summary fields are:

- `changed`: distinct open edges selected for retirement.
- `skipped_diagnostic`: judged-review candidates finalized without their handler.
- `skipped_mismatched_refs`: invalid/missing review references, missing edges, or
  edges whose relation/endpoints do not match the review. Both skip counts are
  zero for the superseded-edge script.

Apply prints the plan, creates an fsync'd journal in
`audited-data-repairs/<script-name-without-extension>/`, rechecks snapshots, and commits edge
invalidations with their maintenance audits in one SQLite transaction. A completion
audit precedes journal archival. Resume uses the journal and durable audit markers
without repeating changes or audits. Reruns exclude closed edges; an empty plan
creates no journal or audit. Dry runs write neither data, journals nor audits.

For reversal, list the run with `borg.audit.list({ runId })` (CLI: `audit list --run-id <run-id>`)
and revert each `correction:retire_contradiction_edges` record with
`borg.audit.revert(id)` (CLI: `audit revert <id>`) for the same bank. Reversal restores
the recorded edge snapshots, preserves dependencies and cancels only the recorded
pending invalidation events. It refuses if an edge changed after repair or the
audit lacks invalidation event IDs. Reviews stay resolved and node statuses stay
unchanged. Reverting the completion summary is a registered no-op.
