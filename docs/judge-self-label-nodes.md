# Judged self-label node repair

Opus 5.0: in-scope because identity merges require explicit evidence, a recorded judgment, and crash-safe persistence independently of model intelligence.

`scripts/judge-self-label-nodes.ts` is an offline second pass after the ID-backed entity-graph repair. Stop **every writer** for judging, recording, planning, applying and recovery. The script opens existing repositories without starting Borg or running migrations. Pending repairs and pending/unknown migrations refuse.

A matching label selects a candidate; it never proves identity. The self entity comes from `EntityRepository.getSelf()` (`entities.kind = 'self'`). The comparison target must be active, unarchived, unsuperseded, have kind `self`, `entity` or `person`, and its trimmed, case-insensitive label must equal the self entity's canonical name or one of its aliases. An alias on the node itself does not qualify a nonmatching label. Use `--target-node <semantic-node-id>` to select the operator-approved merge destination, such as the entity-kind hub used by the first repair pass. When multiple eligible nodes exist across these kinds, the flag is required in every phase, including resume. Refusal lists all eligible targets with `id`, `kind`, `label`, `created_at`, `edge_count` (incoming and outgoing records), and `source_count` (stored source episode IDs). The flag is optional when exactly one eligible target exists; an explicit ID must still be eligible. Missing targets refuse. The script does not choose among eligible nodes by kind, date, label or description.

`--self-node` remains a deprecated CLI alias with the same eligibility rules as `--target-node`. Supplying both flags with different IDs refuses. New verdict files, plans and audits use `target_node: { id, kind, label }`.

The selected target is excluded from judging. The remaining candidates have kind `entity` or `person`, are neither archived nor superseded, and their trimmed, case-insensitive label equals the self canonical name or an alias. No entity name is hardcoded. All other active `kind = 'self'` nodes appear with the same inventory fields in the plan's `other_self_nodes_report_only` section, including nodes whose labels do not match a self name/alias. They are never judged or merged by this script. Other entity rows whose canonical name matches the self canonical name appear in `matching_entities_report_only`; they are never merged.

## Judge: read the store, write a verdict file

```sh
pnpm tsx scripts/judge-self-label-nodes.ts judge \
  --data-dir /path/to/stopped-bank --out /path/to/new-verdicts.json \
  --target-node '<semantic-node-id>'
```

`judge` is the default when no verdict file or apply flag is supplied. Without `--out`, a unique JSON file is written beneath the bank's `audited-data-repairs/` directory. Rerun with the same `--out` path and selected target to resume an existing verdict file: successful judgments with unchanged dossier hashes are reused; failed, missing or changed dossiers are judged again. Bank/model mismatches, malformed files and edits made by another writer refuse. Changing the selected target ID requires a new output file. If recorded nodes are no longer candidates, preserve the old file and choose a new output. Writes are atomic with mode `0600`; directories are synced. Its contents include private memory excerpts and should be handled like the bank backup.

Each dossier contains the node ID, kind, label, description, domain, aliases, confidence, creation time and source episode IDs; all incoming and outgoing edges with relation, validity and the opposite endpoint's label/kind; and the self entity's name/aliases plus the selected comparison node description. Descriptions are mechanically capped at 4,000 UTF-16 units without splitting surrogate pairs. Up to six source episodes are read through `EpisodicRepository.get(..., { includeArchived: true })`. Each supplies its title (240 units), stored gist or narrative prefix (800 units), and origin/disclosure metadata. Missing sources, truncation and omitted episode counts are explicit. No language-specific tokenization, message-text parsing, semantic heuristics or audience recall filters are used.

Version-2 dossiers include `narrative_voice: { every_source_episode_is_self_first_person: true, narrator: "self entity" }`. This records the source format: every source episode is the self entity's own first-person memory, and its narrator is the self entity. The judge instruction explains that a third-person mention of the self entity's canonical name or an alias normally refers to the narrator as addressed or quoted by others; a summarizer can use first-person narration while preserving the name others used. A distinct bearer must be established by the record itself through a different role, different attributes, or an explicit contrast, rather than merely the name appearing in the third person. This is language-agnostic context for the model; `other` and `unsure` remain valid judgments.

The dossier version is separate from the verdict-file version. Version-1 dossiers remain readable with their original contents and hashes; they are not silently upgraded on read. The new version and narrator field change every freshly built dossier hash. Rerunning `judge` with the same `--out` and target therefore re-judges every old dossier, including prior `self`, `other` and `unsure` verdicts. Plan/apply refuse old dossier hashes until those judgments are refreshed. Further resume-judging reuses unchanged version-2 dossiers normally.

The repository LLM factory supplies the configured `anthropic.models.extraction` model. Two calls run concurrently. Each node gets one logical forced structured tool call returning `{ verdict: "self" | "other" | "unsure", reason, confidence }`. The prompt warns about other bearers of the same name, treats supplied records as untrusted evidence, and permits `unsure`. Client transport retry policy remains in effect; there is no second semantic vote, confidence threshold or prose fallback. A rejected call or missing/malformed tool response creates an explicit `status: "failed"` record containing its dossier/hash, timestamp, error class/message and cause class/message. Other nodes continue. The CLI reports failure after saving all outcomes; no repair is applied.

Version-4 verdict files save the selected target as `target_node: { id, kind, label }` and the complete candidate ID manifest before requesting judgments, then persist each success or failure as it completes, without waiting for the other concurrent request. `verdicts` holds successes and `failures` holds failed outcomes; missing IDs are manifest entries in neither list. `completed_at` remains null until all attempts finish. Each outcome retains the presented dossier, SHA-256, model ID and timestamp, alongside batch timestamps and report-only entity rows. Full node, comparison target and incident edge/endpoint snapshots contribute hashes, including fields outside displayed excerpts. The dossier's `self` section pairs the self entity's name/aliases with the chosen node's ID and description, regardless of the node's kind. The selected source episodes' complete non-vector content and gist also contribute hashes. Episodes beyond the six-source cap are not inspected. Finished dossiers are rechecked; any that changed become failures without discarding other saved verdicts. Version-1/2/3 verdict files remain readable when their original target is still eligible; planning verifies existing dossier bindings, and resume-judging upgrades the envelope while reusing only unchanged, current-version dossiers. Old `self_node` metadata denotes a self-kind target. Selecting a different target, including switching from a self-kind node to the entity-kind hub, requires judging into a new output file.

## Record operator verdicts: bind an existing judgment to current evidence

Use `record --operator-verdicts <file>` when an entity has already judged the remaining candidates in a stream entry. This command records the supplied decisions without calling a judgment model, creating model/embedding clients, requesting embeddings or loading model configuration. It uses the same read-only SQLite/Lance evidence path, inventory, target eligibility, dossier builder, path guards and pending-repair/migration refusals as `judge`.

`record` performs no application-level SQLite/Lance mutation, writes no audit or repair journal, runs no migrations and makes no model or embedding calls. Opening a WAL-mode SQLite database read-only can still create `borg.db-wal`/`borg.db-shm` side files. The `--out` writer creates missing parent directories and an atomic temporary file before publishing the verdict file.

The UTF-8 JSON input has this strict schema (`operatorVerdictsSchema` in the dossier module). Unknown keys are rejected at every object level; fields are required unless marked optional:

| Field | Type and validation |
| --- | --- |
| `target_node_id` | Semantic node ID. Must equal the target selected with the existing `--target-node` rules. |
| `judged_by.entity_id` | Non-empty string identifying an existing row in `entities`, of **any** kind. This is the judging entity. |
| `judged_by.evidence.stream_entry_id` | Non-empty string, retained as an opaque reference. |
| `judged_by.evidence.session_id` | Optional non-empty string, retained as an opaque reference. |
| `judged_by.evidence.timestamp` | Optional nonnegative integer, epoch milliseconds. |
| `recorded_by` | Optional strict object containing only `entity_id`: a non-empty string identifying an existing `entities` row of any kind. This identifies the transcriber. |
| `verdicts` | Array containing exactly one judgment for every current candidate. Empty only when there are no candidates. |
| `verdicts[].node_id` | Semantic node ID; duplicates and IDs outside the current candidate set refuse. |
| `verdicts[].verdict` | `"self"`, `"other"` or `"unsure"`. |
| `verdicts[].reason` | Non-empty string, preserved verbatim, including whitespace. |
| `verdicts[].confidence` | Required number in `[0, 1]` or explicit `null`. No default or inferred confidence. |

Example (replace the placeholder IDs, and include **all** current candidates):

```json
{
  "target_node_id": "<selected-semantic-node-id>",
  "judged_by": {
    "entity_id": "<judging-entity-id>",
    "evidence": {
      "stream_entry_id": "<judgment-stream-entry-id>",
      "session_id": "<operator-session-id>",
      "timestamp": 1789992000000
    }
  },
  "recorded_by": { "entity_id": "<transcribing-entity-id>" },
  "verdicts": [
    { "node_id": "<candidate-node-id>", "verdict": "self", "reason": "The recorded judgment, unchanged.", "confidence": null }
  ]
}
```

```sh
npx tsx scripts/judge-self-label-nodes.ts record \
  --data-dir /path/to/stopped-bank --target-node '<semantic-node-id>' \
  --operator-verdicts /path/to/operator-input.json --out /path/to/recorded-verdicts.json

npx tsx scripts/judge-self-label-nodes.ts plan \
  --data-dir /path/to/stopped-bank --target-node '<semantic-node-id>' \
  --verdicts /path/to/recorded-verdicts.json

npx tsx scripts/judge-self-label-nodes.ts apply \
  --data-dir /path/to/stopped-bank --target-node '<semantic-node-id>' \
  --verdicts /path/to/recorded-verdicts.json --backup-verified /path/to/verified-backup
```

The input target does not bypass selection: when multiple eligible targets exist, repeat `--target-node` (or its deprecated `--self-node` alias) in every phase. `--operator-verdicts` is accepted only by `record`; `plan`/`apply` consume the resulting verdict file through `--verdicts`. `record` accepts neither apply/resume flags nor `--backup-verified`; the existing `--live-store-confirmed` path override remains available.

Recording validates that both supplied entity IDs exist in the current store. It does not require either entity to be the self entity, inspect their names, infer who authored the reason, or reinterpret the verdict/confidence. Stream and session references are **not** looked up, authenticated or checked for authorship, content, session membership or chronology. Timestamp validation checks only the numeric shape. The operator is responsible for faithfully transcribing the judgment and choosing evidence references that support it. The command binds those supplied judgments to dossiers built from the **current** store; it does not reconstruct the store as of the referenced stream entry. Attribution is preserved in the resulting file; planning and recovery do not resolve those opaque references.

Missing and extra candidate IDs are listed together on refusal. No output is written until the complete input, entities, target and current dossiers validate, including a second dossier/candidate check. Each verdict receives the unchanged current dossier and `fingerprint(dossier)` SHA-256. The default output directory, unique filename, atomic `0600` write, directory syncing and checked-path/live-store protections are shared with `judge`. Recording is a single complete write. An identical retry with the same `--out` preserves the saved verdicts and batch timestamps; changed input, store, target or dossiers require a new output path. Edits by another writer refuse. It cannot overwrite a model file. Conversely, `judge --out` refuses an operator file instead of converting it into a model judgment.

Recorded verdict files use **version 5**, with file-level attribution:

```ts
type OperatorJudge = {
  kind: "operator";
  judged_by: {
    entity_id: string;
    evidence: { stream_entry_id: string; session_id?: string; timestamp?: number };
  };
  recorded_by?: { entity_id: string };
};
```

Each record is exactly `{ node_id, verdict, reason, confidence, dossier, dossier_hash, judged_at }`. `judged_at` equals the supplied evidence timestamp, or `null` when absent; batch `started_at`/`completed_at` describe recording time. The file retains `kind`, `data_dir`, `target_node`, `candidate_node_ids`, `verdicts`, empty `failures` and `matching_entities_report_only`. No operator envelope, judge or verdict record contains `model` or `model_role`. Versions 1–4 retain their original model envelopes, record shapes, dossier hashes and parsing/planning behavior; the existing `judge` command still writes version 4. Internally, attribution discriminates `kind: "model"` (implicit in legacy files) from `kind: "operator"`, without rewriting old files or journals.

For an operator judgment, the exact `targets.judgment` shape is:

```ts
type OperatorJudgment = {
  verdict: "self" | "other" | "unsure";
  reason: string;
  confidence: number | null;
  judged_by: {
    entity_id: string;
    evidence: { stream_entry_id: string; session_id?: string; timestamp?: number };
  };
  recorded_by?: { entity_id: string };
  dossier_hash: string;
};
```

`recorded_by` is omitted when not supplied; `model`, `model_role` and the internal `kind` discriminator are absent from this audit object. The enclosing audit remains `process: "correction"`, `action: "judge-self-label:<verdict>"`, with the existing `targets.node_id`, `canonical_node_id`, `judged_at`, `target_node`, `backup_verified`, `live_store_override`, and `judgment_id = fingerprint({ node_id, dossier_hash, verdict })`. Reasons and confidence are copied without rewriting. The same explicit merge path, audit-only non-merges, report-only lists, reversal snapshots and tuple-based `already_recorded` lookup apply. Attribution, reason and confidence are not part of judgment identity: an existing unreverted audit for the same tuple is retained, including one originally recorded from a model file. Recording does not replace its historical attribution.

Recovery uses the unchanged version-2 journal and version-1 `selfLabelJudgments` payload containing the version-5 verdict file. Use the existing `apply --resume` command with the same target, backup and live override, and omit both `--verdicts` and `--operator-verdicts`. Resume repeats the original/precommit or projected/postcommit dossier checks and never requests another judgment.

## Plan, then apply with a verified backup

A response must contain exactly one `JudgeSelfLabelNode` tool call. Multiple calls, including conflicting `self` and `unsure` calls, fail the node instead of selecting the first. This checks protocol cardinality only; it does not reinterpret the judgment text.

Plan/apply refuse any file with failed or missing judgments and list every affected node ID with its status. The current candidate inventory is also checked, so a missing node cannot be hidden by removing it from the saved manifest. Resume-judging uses `judge --out /path/to/existing-verdicts.json`; `--apply --resume` remains reserved for repair journal recovery.

```sh
pnpm tsx scripts/judge-self-label-nodes.ts plan \
  --data-dir /path/to/stopped-bank --verdicts /path/to/new-verdicts.json \
  --target-node '<semantic-node-id>'

pnpm tsx scripts/judge-self-label-nodes.ts \
  --data-dir /path/to/stopped-bank --verdicts /path/to/new-verdicts.json \
  --target-node '<semantic-node-id>' \
  --apply --backup-verified /path/to/verified-backup
```

`apply` is also accepted as a command in place of `--apply`. Otherwise a verdict file selects a dry run. Repeat the same `--target-node` whenever multiple active targets exist; the chosen ID/kind/label must match the verdict file and plan. Apply requires `--backup-verified` naming an existing independent file or directory; the operator is responsible for verifying its freshness and integrity. This uses the same independent-backup check as the period-rollover repair. Paths inside the bank, containing it, or naming the bank database itself refuse. Supply the flag during a dry run too if reviewing a plan that records the intended backup.

The complete JSON plan reaches stdout before store mutation. It contains every SQL/Lance change, vectors and reversal audits, per-node judgments, `selfLabelJudgments.target_node`, report-only self nodes/entities, backup/override metadata, and pending/archive journal paths. Planning a merge uses the existing configured embedding client to obtain actual replacement vectors; judging, recording and plans containing only non-merges do not request embeddings. Plan/apply never call the judgment model.

| Verdict | Data effect | Audit |
| --- | --- | --- |
| `self` | Explicit merge into the selected comparison node through `planEntityGraphRepair` / `applyEntityGraphRepair`. | `correction`, `judge-self-label:self`, with `targets.judgment`. Existing graph audit rows carry full reversal snapshots. |
| `other` | No merge or node modification. | `correction`, `judge-self-label:other`, reversal `{ kind: "none", data_changes: false }`. |
| `unsure` | No merge or node modification. | `correction`, `judge-self-label:unsure`, reversal `{ kind: "none", data_changes: false }`. |

Every judgment is recorded, including low-confidence `self`, null-confidence operator judgments and high-confidence non-merges; confidence is evidence metadata, not a deterministic override. For model files, `targets.judgment` contains exactly `verdict`, `reason`, `confidence`, `model`, and `dossier_hash`; operator files use the entity-attributed shape above. Judgment audit rows additionally retain the source node ID, target ID and judgment time. Every new audit row, including graph reversal snapshots and completion markers, carries `targets.target_node: { id, kind, label }`, the backup path and explicit live-store override.

Each judgment audit carries `targets.judgment_id`, a stable SHA-256 of `{ node_id, dossier_hash, verdict }`. Planning skips matching unreverted audits, including older rows without the explicit ID field by checking the same tuple. It reports each judgment as `record` or `already_recorded`. An unchanged, still-current non-merge file produces a null run ID and `no changes`; apply creates no audit, completion marker or journal. Reapplying an audit-only plan also rechecks these identities before writing, so it remains a no-op if another completed apply already recorded them. A partially outdated audit plan refuses for re-planning. Changed dossiers require refreshed model judgments or a new operator recording; a changed dossier hash or verdict has a distinct identity. Existing audit text is retained, not overwritten by a new reason/confidence for the same identity.

The graph planner's `explicit-node-merges` scope disables general alias cleanup, entity merging and goal-history backfills. It retains the original node-merge mechanics: alias/source unions, archive/supersession pointers, reference and edge re-aiming, duplicate-edge evidence transfer, closing edges whose endpoints coalesce, re-embedding and generation-guarded outbox drains. Changed aliases use the repository decoder's normalization so the planned mirror agrees with its eventual outbox sync. Historical judgment audits remain immutable under later reference projection.

If those reference rewrites would alter a node judged `other`/`unsure` or a report-only self node (for example its `corrected_by` pointer), the whole plan refuses for separate operator review. It never silently modifies a non-merge node or an entity row. The plan refreshes both report-only lists from the current store.

The verdict file is bound to its canonical bank path. Duplicate IDs, malformed structures, inconsistent model IDs or dossier hashes, a changed candidate set, or changed evidence refuse. Apply rechecks the dossiers before creating its journal. Stop writers throughout; this is an offline repair, not a concurrent-store transaction protocol.

## Recovery and live-path guard

The journal envelope records `kind: "judge-self-label-nodes"`. The shared recovery engine rejects this kind through the legacy CLI and accepts it only through the owning script's delegated path after its preflight. Kind/origin mismatches refuse; pre-kind judged journals are recognized from their plan metadata and have the same restriction.

The script uses the **same** version-2 `entity-graph-repair/pending.json` journal and `<run-id>.json` archives as the first repair, with an additional versioned, checksummed judged-plan payload. Judgments and graph SQL changes commit together; Lance commits, outbox drains and completion/archive follow the existing machinery. Resume through this script so backup, confirmation, checksum, migrations and dossier guards are repeated:

```sh
pnpm tsx scripts/judge-self-label-nodes.ts \
  --data-dir /path/to/stopped-bank --apply --resume \
  --target-node '<semantic-node-id>' \
  --backup-verified /path/to/verified-backup
```

Omit `--verdicts` on resume: the journal retains the exact reviewed verdicts and vectors. Before the SQL commit, resume requires the original dossier hashes; afterwards, it requires the hashes of the planned resulting dossiers. This permits interrupted merges without accepting unrelated changes. Completion/archive recovery does not duplicate judgments. An old verdict file is not a new approval for a changed or already merged node. Re-applying an unchanged non-merge file reports `no changes`; use resume to finish an interrupted run.

Paths lexically or through symlinks under `demo/server/.borg-data` refuse by default. As with the other audited repairs, an explicit `--live-store-confirmed /path/to/backup-directory` permits an operator-selected guarded store only when that directory contains a regular `borg.db` or `borg.db-clean`. This flag names the **backup**, not the target store, and does not replace `--backup-verified`. The canonical override is stored in the plan and all audits; repeat the identical backup and override paths on resume. The implementation and tests use disposable fixtures only; they do not open the real live directory.

Graph data can be reversed using the existing offline entity-graph audit-based reversal, subject to its unchanged-row checks. Judgment history is retained. No new reversal dispatcher, schema migration, automatic backup, entity-row merge, live-store run, service restart or online interpretation change is included.
