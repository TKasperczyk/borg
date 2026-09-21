# Operator-applied autobiographical rollover

Stop every writer, take and verify a fresh independent backup, then rehearse on a copy.
The script opens only an explicit bank, refuses the live demo path, pending/unknown migrations,
and other pending repair journals. It makes no LLM calls and does not select a review for you.

```sh
pnpm tsx scripts/apply-period-rollover.ts --data-dir /home/luth/tmp/bank-copy --review-id 123 --dismiss 124 --dismiss 125 --operator 'Operator name'
pnpm tsx scripts/apply-period-rollover.ts --data-dir /home/luth/tmp/bank-copy --review-id 123 --dismiss 124 --dismiss 125 --operator 'Operator name' --backup-verified /home/luth/tmp/verified-snapshot --apply
pnpm tsx scripts/apply-period-rollover.ts --data-dir /home/luth/tmp/bank-copy --backup-verified /home/luth/tmp/verified-snapshot --apply --resume
```

`--operator` defaults to the OS account name. Supply the responsible person's name when
that differs. `--backup-verified` is the operator's attestation that the supplied backup is
fresh and consistent. The path must exist, be independent of the bank, and be supplied again
on resume; the tool does not create a backup or infer its freshness from filesystem timestamps.
The applied plan and audit record its canonical path. A directory or snapshot archive is accepted.

Only an **open identity_inconsistency review** whose validated refs target the sole open
autobiographical period, include a numeric `patch.end_ts`, and include an open
`next_period_open_payload` is eligible. The successor ID must be unused. Its `start_ts` is
the single boundary; the proposed predecessor `end_ts` is replaced with that boundary.
The boundary must follow the predecessor start. Shifted predecessor starts, empty intervals,
multiple open periods, existing timeline gaps/overlaps, and a closed successor are refused.
Existing historical gaps are reported for separate review, never silently repaired.

The closed period patch, patch application provenance and successor payload all use
`{ kind: "manual" }`. Narrative content remains exactly the selected proposal's content.
Proposal creation stamps on the successor retain the handler's existing behavior; closing
the predecessor advances its update time/version through the identity service.

Each explicit dismissal must target the same predecessor and be open:

- A numeric `patch.end_ts` identifies another closure: `dismissed as duplicate of <id>`.
- A narrative/label/themes/key-episodes patch with no time-boundary or successor change:
  `superseded by the operator-applied rollover <id>; content after the boundary belongs to the successor`.

Classification examines schema fields only, never proposal text. Unrelated and reopening
reviews are refused. Dismissals are explicit; other pending reviews remain untouched.

The review queue's `resolution` column is an enum. Its durable resolution **text** is stored
on the same row in `refs.__borg_review_resolver_manual_resolution.text`, alongside `operator`
and `boundary`. The accepted row names the operator and numeric/ISO boundary. The enum remains
`accept` or `dismiss`, preserving normal queue consumers. The accepted row also records the
actual applied refs; the audit retains the full original proposal. The reserved resolver
metadata prefix is stripped by the queue's existing strict-handler parsing path.

Dry run executes the standard identity-inconsistency handler and `ReviewQueueRepository.resolve`
on an in-memory database clone and prints the complete boundary, both period records
before/after, review records, dismissal texts, identity events, correction audit inputs and
journal paths. Apply flushes that plan before fsyncing its journal. An offline handler adapter
supplies the manual refs and records a reversal audit **inside the normal queue transaction**;
the standard handler/identity service/repositories perform all period, identity-event and
review writes. The rollover and its review/audit are atomic. Each subsequent dismissal and
its audit are atomic. A run-summary audit precedes journal archival.

After interruption keep writers stopped and resume with the recorded backup path. Resume
checks the complete period timeline and affected reviews against journaled snapshots and
audit markers; it never repeats a completed rollover or dismissal. A new invocation targeting
an already-resolved review refuses. Full before/expected-after snapshots are in correction
audits for operator-reviewed reversal; no automatic undo command is added. Identity and
maintenance audit history should remain append-only during any subsequent correction.

Opus 5.0: in-scope because period boundaries and manual application provenance are structural
memory integrity, independent of model intelligence.
