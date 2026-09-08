# Operator attention index

Borg stores filing metadata only. Operator-facing bodies stay in the connector's
`operator-attention.jsonl`. The index is presentation only and does not gate action.
The autonomous mechanism-evidence prompt shows the total count and latest 20 rows,
with date, filer, subject (or `subject unavailable`), and each row's disclosure label.

## API contract

`POST /api/operator-attention` accepts exactly:

```json
{
  "record_key": "cclink:<uuid>",
  "filed_at": 1788739200000,
  "filer_entity_id": "ent_aaaaaaaaaaaaaaaa",
  "subject": "A one-line subject"
}
```

- `record_key`: stable opaque key, 1–200 characters, persisted by the connector for replay.
- `filed_at`: Unix milliseconds, UTC.
- `filer_entity_id`: the counterpart's provisioned Borg entity ID.
- `subject`: one line, at most 240 UTF-16 code units, or `null`. Body and unknown fields are rejected.

HTTP 200 returns `{ "inserted": true }` for a new key or `{ "inserted": false }`
for a duplicate. First filing wins; replay does not change an accepted row. Invalid
payloads return HTTP 400. The route uses the demo API's CORS, request/reset gate,
and error handling. This API has no application-level authentication middleware.

`borg.operatorAttention.record(envelope)` writes metadata;
`borg.operatorAttention.snapshot()` returns the total and latest rows. Every returned
row carries `disclosure_label` in the existing `MemoryDisclosureLabel` shape, using
`operator_private`, `originAudienceEntityIds: [filer_entity_id]`, and empty
`privateToEntityIds` / `publicToEntityIds` (recipient authorization is unknown).
The filer is not assumed to be an operator. These labels permit global internal
cognition; disclosure remains the model's judgment given audience/authority context.
Capture/replay retains labels, and combining with an unknown label stays `unknown`.

## Configuration and marker protocol

The connector sends attention metadata over HTTP to `BORG_API_BASE`, defaulting to
`http://127.0.0.1:${PORT:-7740}` when embedded in the demo process. Its other Borg
operations use the in-process handle. Filing and operator notification happen before
the five-second HTTP reporting attempt. Failures are logged; replay uses the local
JSONL record. A failed local append does not create an index entry.

The exact first-line sentinel is `CCLINK_OPERATOR_ATTENTION_V1`:

```text
CCLINK_OPERATOR_ATTENTION_V1
A one-line subject
Operator-facing body, which may span multiple lines.
```

Only that complete versioned header enables subject extraction. Unversioned markers,
including multiline bodies, remain body-only with `subject: null`. LF and CRLF are
accepted. Subject clipping preserves code-point boundaries within 240 UTF-16 code
units. The existing flattened local `reason` representation and 64 KiB marker cap
remain in place.

The counterpart's filing instruction is:

> ALSO write $CLAUDE_CONFIG_DIR/operator-attention as a tool action. Use this exact marker format: first line CCLINK_OPERATOR_ATTENTION_V1, second line a short, one-line subject (at most 240 UTF-16 code units), and the operator-facing body starting on the third line. Unversioned markers are treated as body-only filings with no subject. Sol receives only the filing's existence, date, your provisioned counterpart entity id as filer, and that subject. The body stays in the connector's operator log and is never sent to Borg. Write the subject for this limited visibility; do not copy the body into it.

## Backfill and replay

Dry-run performs no HTTP calls or writes and prints the exact envelopes to review:

```sh
choom -n 800 -- pnpm exec tsx scripts/backfill-operator-attention.ts --file /path/to/operator-attention.jsonl --filer-entity-id ent_aaaaaaaaaaaaaaaa
```

To submit through a running Borg API, add `--apply --borg-url http://127.0.0.1:7740`.
The source is read-only; all envelopes are validated before any HTTP write. The
script never opens Borg storage directly. Stored subjects are forwarded when present;
missing/null subjects stay null. A subject is never inferred from `reason` or body,
and neither body field enters the payload.

Stored keys, dates, and filers are preserved. `--filer-entity-id` supplies the known
counterpart ID for old records lacking it. Legacy keys are `cclink:legacy:<sha256>`,
derived only from filer ID, timestamp, and the occurrence among rows with that
filer/timestamp in file order. Copying, renaming, or appending to the source preserves
prior keys; identical timestamps remain distinct. HTTP failures stop the import with
a nonzero exit. Rerun the same command to retry; accepted keys are not duplicated.
