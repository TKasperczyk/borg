# Commitment source-sender backfill (L)

Stop all writers and use a backed-up rehearsal bank. No default data directory is opened;
the live demo path is refused by default. Pending/unknown migrations and other pending repairs are refused.

```sh
pnpm tsx scripts/backfill-commitment-committers.ts --data-dir /home/luth/tmp/bank-copy
pnpm tsx scripts/backfill-commitment-committers.ts --data-dir /home/luth/tmp/bank-copy --apply
pnpm tsx scripts/backfill-commitment-committers.ts --data-dir /home/luth/tmp/bank-copy --apply --resume
```

For a stopped-service live apply, explicitly add `--live-store-confirmed <verified-backup-directory>`
to the dry-run/apply commands with the live `--data-dir`. The backup path must exist, be a
directory and contain a `borg.db` or `borg.db-clean` file. This confirmation skips both the
lexical and resolved live-path refusals for that invocation only; it does not stop writers
or verify backup freshness/consistency on the operator's behalf.

The plan, apply summary and journal record `live_store_override: { "backup_path": "<canonical directory>" }`.
Every correction audit, including the run summary, records it in `targets.live_store_override`.
Without the flag the field is `null`. On `--resume`, supply the flag with the same backup
directory again; omitting it, changing it, or adding it to an unconfirmed journal refuses.
A recorded override alone never grants access on a later invocation.

The default read-only plan lists **every NULL row**, all source entry IDs, indexed/missing
joins, distinct non-NULL senders, raw stored before/after rows, identity events and audit inputs.
Snapshots retain legacy NULLs and the exact stored JSON strings instead of decoded defaults.
Every source entry must be indexed with a non-NULL sender, and all must resolve to the
same sender. A missing entry or NULL sender leaves the commitment NULL with status
`unresolved_source_entry` and reason `unresolved source entry`, even when other entries
have one or more known senders. The summary's `unresolved_source_rows` counts these
commitments separately; `zero_senders` counts rows without source entries, and
`multiple_senders` counts fully resolved rows with conflicting senders. No entity redirect,
speaker role, audience, directive text or authorship inference is used. The field means
the sender of the entry from which the directive was extracted.

Apply emits the complete plan before creating an fsync'd journal in
`audited-data-repairs/backfill-commitment-committers/`. The commitments repository updates
only `committed_by_entity_id` through a dedicated backfill method; every other column,
including timestamps, record version, legacy NULLs and JSON text, is preserved. The identity
repository records the manual maintenance event with raw before/after rows. Original
commitment provenance is retained.
Updates and one `maintenance_audit` row per change commit in one SQLite transaction.
Each audit has process `correction`, the prior NULL, the full prior record and expected
after record for guarded reversal. A separate run-summary audit precedes journal archival.
Resume uses the journal and durable audit markers, rechecking source joins and affected
records and the complete-source attribution rule; it does not duplicate updates or audits.
Older plans that accepted partial joins refuse. An empty rerun creates no journal or audit.

No automatic reversal command is provided. The audit snapshots support an operator-reviewed
repository-based reversal; restoring a verified backup is the alternative. No expiry sweep,
stream rewrite, schema migration, LLM call or ambiguous attribution is performed.

Opus 5.0: in-scope because missing source-sender handles are a stored-data integrity defect.
