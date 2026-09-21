# Commitment source-sender backfill (L)

Stop all writers and use a backed-up rehearsal bank. No default data directory is opened;
the live demo path is refused. Pending/unknown migrations and other pending repairs are refused.

```sh
pnpm tsx scripts/backfill-commitment-committers.ts --data-dir /home/luth/tmp/bank-copy
pnpm tsx scripts/backfill-commitment-committers.ts --data-dir /home/luth/tmp/bank-copy --apply
pnpm tsx scripts/backfill-commitment-committers.ts --data-dir /home/luth/tmp/bank-copy --apply --resume
```

The default read-only plan lists **every NULL row**, all source entry IDs, indexed/missing
joins, distinct non-NULL senders, full before/after records, identity events and audit inputs.
Exactly one distinct sender is sufficient even if some other source IDs are missing or
have NULL senders. Zero or multiple senders leave the row NULL. No entity redirect,
speaker role, audience, directive text or authorship inference is used. The field means
the sender of the entry from which the directive was extracted.

Apply emits the complete plan before creating an fsync'd journal in
`audited-data-repairs/backfill-commitment-committers/`. The commitments repository updates
only the requested field plus its normal timestamp/version metadata; the identity repository
records the manual maintenance event. Original commitment provenance is retained.
Updates and one `maintenance_audit` row per change commit in one SQLite transaction.
Each audit has process `correction`, the prior NULL, the full prior record and expected
after record for guarded reversal. A separate run-summary audit precedes journal archival.
Resume uses the journal and durable audit markers, rechecking source joins and affected
records; it does not duplicate updates or audits. An empty rerun creates no journal or audit.

No automatic reversal command is provided. The audit snapshots support an operator-reviewed
repository-based reversal; restoring a verified backup is the alternative. No expiry sweep,
stream rewrite, schema migration, LLM call or ambiguous attribution is performed.

Opus 5.0: in-scope because missing source-sender handles are a stored-data integrity defect.
