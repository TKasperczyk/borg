# Orphan action actor repair (E3)

Stop all writers and rehearse on a backed-up copy. Both entity IDs are explicit operator
inputs; the script performs no spelling/name inference and does not create redirects.

```sh
pnpm tsx scripts/repair-action-actor.ts --data-dir /home/luth/tmp/bank-copy --from ent_aaaaaaaaaaaaaaaa --to ent_bbbbbbbbbbbbbbbb
pnpm tsx scripts/repair-action-actor.ts --data-dir /home/luth/tmp/bank-copy --from ent_aaaaaaaaaaaaaaaa --to ent_bbbbbbbbbbbbbbbb --apply
pnpm tsx scripts/repair-action-actor.ts --data-dir /home/luth/tmp/bank-copy --apply --resume
```

The default full JSON plan lists every exact matching action, including terminal/history
rows, before/after SQL records and existing Lance mirror records/vectors. It refuses if
`--from` has any entities row (even a redirect), if `--to` has none, if migrations are
pending/unknown, or if another repair is pending. It also refuses the live demo path.

Apply flushes the plan, fsyncs a complete journal, and updates SQL through the actions
repository in the same transaction as one `maintenance_audit` correction row per action.
The audit contains the old actor and complete SQL/Lance before/after snapshots. Each SQL
row is re-read immediately before its write inside the transaction. ID and actor are the
SQL preconditions; other metadata can advance without invalidating an actor-only repair.
The fresh fields are carried into the repository patch so Zod defaults cannot clear omitted
archive/session/reference metadata. The audit records these actual before/after values,
and explicitly scopes reversal to `actor`. Lifecycle state, timestamps, description,
provenance and vectors are preserved by the repair. Mirror timestamps are not resynchronized.
Existing mirrors are then updated through the awaited actions repository repair method;
failures propagate and leave the journal pending. No embeddings are requested, and absent
mirrors are reported rather than created. A run-summary audit precedes journal archival.

Resume rechecks entity preconditions, SQL IDs/actors and every affected mirror/vector. It
accepts only the planned actor states and exact mirror states, and avoids duplicate SQL
writes/audits. Metadata changes to sibling rows do not cause stale snapshot failures; a changed
actor still refuses, including a change during the transaction. Keep writers
stopped until it completes. A zero-change rerun adds nothing. Audit snapshots support an
operator-reviewed reversal; no automatic undo command is added. Source streams, entity rows,
review rows and action lifecycle decisions are untouched.

Opus 5.0: in-scope because an orphan machine actor handle is a stored-data integrity defect.
