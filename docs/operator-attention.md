# Operator attention index

An external counterpart can file an operator attention record. Borg retains only
the filing envelope; the operator-facing body stays in the connector's JSONL log.
The index is presentation only and has no effect on wake admission or action policy.

`POST /api/operator-attention` accepts exactly:

```json
{
  "record_key": "cclink:<uuid>",
  "filed_at": 1788739200000,
  "filer_entity_id": "ent_aaaaaaaaaaaaaaaa",
  "subject": "A one-line subject"
}
```

- `record_key`: stable opaque key, 1–200 characters. The connector creates it once
  per consumed marker and persists it in `operator-attention.jsonl` for resends.
- `filed_at`: Unix milliseconds, UTC.
- `filer_entity_id`: the counterpart's provisioned Borg entity ID, not the entity
  the filing is about and not the human operator's ID.
- `subject`: one line, at most 240 characters (UTF-16 code units), or `null` when
  unavailable. Unknown fields, including `body` and `reason`, are rejected.

The response is HTTP 200 with `{ "inserted": true }` for a new key or
`{ "inserted": false }` for a duplicate. First filing wins: resends do not modify
the existing date, filer, or subject. Validation errors return HTTP 400. The route
uses the demo API's existing CORS, request/reset gate, and error handling. This
checkout has no application-level HTTP authentication middleware.

The facade is `borg.operatorAttention.record(envelope)` and
`borg.operatorAttention.snapshot()`. SQLite stores only the four envelope columns
in `operator_attention_records`; its migration is appended as a new migration band.
The mechanism-evidence prompt, beside the wake-window rows, renders the total and
the latest 20 records ordered by filing date, newest first, with the record key as
a deterministic tie-breaker. Legacy rows render `subject unavailable`. Subjects
are quoted and XML-escaped for the enclosing prompt. The index is labeled private
to the entity and operator, with contextual disclosure guidance.

## Backfill

Use the counterpart's provisioned entity ID as the fallback filer for historical
rows, which have only `ts` and `reason`. Dry-run performs no HTTP calls or writes
and prints the exact envelopes to review:

```sh
choom -n 800 -- pnpm exec tsx scripts/backfill-operator-attention.ts \
  --file /path/to/operator-attention.jsonl \
  --filer-entity-id ent_aaaaaaaaaaaaaaaa
```

To import those envelopes through a running Borg API, add both:

```sh
--apply --borg-url http://127.0.0.1:7740
```

The source is opened read-only. All rows are validated before any HTTP write.
Borg storage is never opened directly by this script. Subjects are always `null`
on import, even if a newer source row contains a subject; no historical subject is
inferred from the body. A row's stored `record_key` and `filer_entity_id` are reused
when present. Legacy keys are `cclink:legacy:<sha256>` derived only from filer ID,
timestamp, and the zero-based occurrence among rows with that filer/timestamp in
file order. Copied or renamed files and repeat runs retain the same keys; rows
sharing a timestamp remain distinct. An append preserves earlier keys. HTTP
failures stop the import with a nonzero exit; rerunning safely skips accepted keys.

## Connector integration

The `sol-connector` checkout uses an in-process Borg handle for its existing
operations. Attention reporting uses its new `BorgHttpClient`, targeting
`BORG_API_BASE`, defaulting to `http://127.0.0.1:${PORT:-7740}` for the embedding
demo process. It writes the local JSONL record and notifies the operator before
making the HTTP request. Reporting has a five-second timeout; failures are logged
and do not undo filing. There is no automatic resend queue. The local record key
supports explicit replay, including the existence-only backfill above.

New markers have a subject on the first line and the body below. The connector
retains the existing flattened `reason` representation and 64 KiB marker cap for
operator storage. A single content line, with or without a trailing newline, is treated as a legacy
body with no subject. A failed local append still logs the operator notification, but does not
report an index record for a filing that was not persisted.

## Counterpart instruction added

The filing paragraph in `sol-connector/src/config.ts` now includes this exact text:

> ALSO write $CLAUDE_CONFIG_DIR/operator-attention as a tool action. The first line must be a short, one-line subject (at most 240 characters); put the operator-facing body on the lines below. Sol receives only the filing's existence, date, your provisioned counterpart entity id as filer, and that subject. The body stays in the connector's operator log and is never sent to Borg. Write the subject for this limited visibility; do not copy the body into it.

## Changed files

Borg:

```text
demo/server/src/__tests__/server.test.ts
demo/server/src/app.ts
docs/operator-attention.md
scripts/backfill-operator-attention.test.ts
scripts/backfill-operator-attention.ts
src/borg.ts
src/borg/facade-types.ts
src/borg/facade.ts
src/borg/open.ts
src/borg/public-facade.ts
src/borg/repositories.ts
src/borg/storage-setup.ts
src/borg/turn-setup.ts
src/borg/types.ts
src/cognition/deliberation/planner-context-capture.test.ts
src/cognition/deliberation/planner-context-capture.ts
src/cognition/deliberation/prompt/system-prompt.test.ts
src/cognition/deliberation/prompt/system-prompt.ts
src/cognition/lifecycle/turn-phase/retrieval-phase.test.ts
src/cognition/lifecycle/turn-phase/retrieval-phase.ts
src/cognition/lifecycle/turn-phase/types.ts
src/cognition/mechanism-evidence.ts
src/cognition/turn-orchestrator.ts
src/index.ts
src/memory/operator-attention/index.ts
src/memory/operator-attention/migrations.ts
src/memory/operator-attention/repository.test.ts
src/memory/operator-attention/repository.ts
src/memory/operator-attention/types.ts
```

sol-connector:

```text
.env.example
README.md
src/borg-client.test.ts
src/borg-client.ts
src/cclink/runtime.test.ts
src/cclink/runtime.ts
src/config.ts
```

## Verification (2026-09-06)

No dependency installation, live API writes, live-tree source edits, or writes
under `.borg-data` / `.sol-state` were performed. Backfill was exercised with
synthetic temporary input and mocked apply requests; it was not applied to the
existing operator log. The provided Borg dependency symlinks were created before
checks. The connector's pre-existing `node_modules` symlink was left untouched.
Vitest used two workers, disabled caching, and loaded config through the runner
so it would not write caches/config bundles into symlinked live dependencies.

Final results:

| Scope | Command | Result |
| --- | --- | --- |
| Borg | `choom -n 800 -- pnpm typecheck` | Passed all five root-script tsconfigs |
| Connector | `choom -n 800 -- pnpm typecheck` | Passed |
| Demo API | `choom -n 800 -- pnpm exec tsc --noEmit -p demo/server/tsconfig.dev.json` | Passed |
| Backfill | `choom -n 800 -- pnpm exec tsc --noEmit --strict --skipLibCheck --target ES2023 --module ESNext --moduleResolution Bundler scripts/backfill-operator-attention.ts scripts/backfill-operator-attention.test.ts` | Passed |
| Borg | `choom -n 800 -- pnpm heuristics:guard` | Passed |
| Borg targeted tests | Command below | 269 passed in 6 files |
| Connector targeted tests | Command below | 50 passed, 4 excluded, in 5 files |
| Backfill CLI | `choom -n 800 -- pnpm exec tsx scripts/backfill-operator-attention.ts --file <temporary fixture> --filer-entity-id ent_aaaaaaaaaaaaaaaa` (under a temporary-fixture wrapper) | Passed: dry-run default, two distinct metadata-only records, source unchanged |
| Both repos | `git diff --check` | Passed |

Borg test command:

```sh
choom -n 800 -- pnpm exec vitest run   src/memory/operator-attention/repository.test.ts   scripts/backfill-operator-attention.test.ts   src/cognition/deliberation/prompt/system-prompt.test.ts   src/cognition/deliberation/planner-context-capture.test.ts   src/cognition/lifecycle/turn-phase/retrieval-phase.test.ts   demo/server/src/__tests__/server.test.ts   --maxWorkers=2 --no-cache --configLoader=runner
```

Connector test command:

```sh
choom -n 800 -- pnpm exec vitest run   src/borg-client.test.ts src/cclink/runtime.test.ts   src/cclink/provision.test.ts src/cclink/thread.test.ts src/plugin.test.ts   --maxWorkers=2 --no-cache --configLoader=runner   --testNamePattern='^(?!.*(?:moves the legacy in-repository DB|refuses a failed deploy gate|refuses a deploy gate)).*$'
```

The four excluded cases test legacy state migration and deploy gates and create
directories named `.sol-state`. They were excluded to honor the write restriction;
the attention, client, provisioning, prompt/thread, plugin, and remaining runtime
cases all ran. These were targeted suites, not full repository test runs.

Initial typechecks caught missing attention composition connections and a test
tracer fixture field; both were corrected. The supplemental demo check also
exposed a pre-existing missing `lived-experience-day-summarizer` description in
`app.ts`; the single missing map entry was added so the API typecheck passes.
Changed Borg TypeScript files and the new client files were formatted with
`pnpm exec prettier --write`; connector runtime formatting was restricted to the
changed attention sections. Heavy commands, including formatting, ran under
`choom -n 800`.
