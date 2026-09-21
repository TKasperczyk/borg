# Judged self-label node repair

Opus 5.0: in-scope because identity merges require explicit evidence, a recorded judgment, and crash-safe persistence independently of model intelligence.

`scripts/judge-self-label-nodes.ts` is an offline second pass after the ID-backed entity-graph repair. Stop **every writer** for judging, planning, applying and recovery. The script opens existing repositories without starting Borg or running migrations. Pending repairs and pending/unknown migrations refuse.

A matching label selects a candidate; it never proves identity. The self entity comes from `EntityRepository.getSelf()` (`entities.kind = 'self'`), and the comparison node must be the single active, unarchived, unsuperseded semantic node with `kind = 'self'`. Missing or ambiguous anchors refuse rather than falling back to a name. Candidates have kind `entity` or `person`, are neither archived nor superseded, and their trimmed, case-insensitive label equals the self canonical name or an alias. No entity name is hardcoded. Other entity rows whose canonical name matches the self canonical name appear in `matching_entities_report_only`; they are never merged.

## Judge: read the store, write a verdict file

```sh
pnpm tsx scripts/judge-self-label-nodes.ts judge \
  --data-dir /path/to/stopped-bank --out /path/to/new-verdicts.json
```

`judge` is the default when no verdict file or apply flag is supplied. Without `--out`, a unique JSON file is written beneath the bank's `audited-data-repairs/` directory. Existing output paths refuse, avoiding accidental overwrite. The file is written atomically with mode `0600`; directories are synced. Its contents include private memory excerpts and should be handled like the bank backup.

Each dossier contains the node ID, kind, label, description, domain, aliases, confidence, creation time and source episode IDs; all incoming and outgoing edges with relation, validity and the opposite endpoint's label/kind; and the self entity's name/aliases plus the canonical self node description. Descriptions are mechanically capped at 4,000 UTF-16 units without splitting surrogate pairs. Up to six source episodes are read through `EpisodicRepository.get(..., { includeArchived: true })`. Each supplies its title (240 units), stored gist or narrative prefix (800 units), and origin/disclosure metadata. Missing sources, truncation and omitted episode counts are explicit. No language-specific tokenization, message-text parsing, semantic heuristics or audience recall filters are used.

The repository LLM factory supplies the configured `anthropic.models.extraction` model. Two calls run concurrently. Each node gets one logical forced structured tool call returning `{ verdict: "self" | "other" | "unsure", reason, confidence }`. The prompt warns about other bearers of the same name, treats supplied records as untrusted evidence, and permits `unsure`. Client transport retry policy remains in effect; there is no second semantic vote, confidence threshold or prose fallback. A missing/malformed tool call fails the batch without writing a verdict file or applying anything.

The file stores each complete presented dossier, its SHA-256, model ID, judgment, and timestamp, plus batch timestamps and the report-only entity rows. Full node, self target and incident edge/endpoint snapshots contribute hashes, including fields outside the displayed excerpts. The selected source episodes' complete non-vector content and gist also contribute hashes. Episodes beyond the six-source cap are not inspected. Dossiers are rechecked before the verdict file is written.

## Plan, then apply with a verified backup

```sh
pnpm tsx scripts/judge-self-label-nodes.ts plan \
  --data-dir /path/to/stopped-bank --verdicts /path/to/new-verdicts.json

pnpm tsx scripts/judge-self-label-nodes.ts \
  --data-dir /path/to/stopped-bank --verdicts /path/to/new-verdicts.json \
  --apply --backup-verified /path/to/verified-backup
```

`apply` is also accepted as a command in place of `--apply`. Otherwise a verdict file selects a dry run. Apply requires `--backup-verified` naming an existing independent file or directory; the operator is responsible for verifying its freshness and integrity. This uses the same independent-backup check as the period-rollover repair. Paths inside the bank, containing it, or naming the bank database itself refuse. Supply the flag during a dry run too if reviewing a plan that records the intended backup.

The complete JSON plan reaches stdout before store mutation. It contains every SQL/Lance change, vectors and reversal audits, per-node judgments, report-only entities, backup/override metadata, and pending/archive journal paths. Planning a merge uses the existing configured embedding client to obtain actual replacement vectors; judging and plans containing only non-merges do not request embeddings. Plan/apply never call the judgment model.

| Verdict | Data effect | Audit |
| --- | --- | --- |
| `self` | Explicit merge into the canonical self node through `planEntityGraphRepair` / `applyEntityGraphRepair`. | `correction`, `judge-self-label:self`, with `targets.judgment`. Existing graph audit rows carry full reversal snapshots. |
| `other` | No merge or node modification. | `correction`, `judge-self-label:other`, reversal `{ kind: "none", data_changes: false }`. |
| `unsure` | No merge or node modification. | `correction`, `judge-self-label:unsure`, reversal `{ kind: "none", data_changes: false }`. |

Every judgment is recorded, including low-confidence `self` and high-confidence non-merges; confidence is evidence metadata, not a deterministic override. `targets.judgment` contains exactly `verdict`, `reason`, `confidence`, `model`, and `dossier_hash`. Audit rows additionally retain the source node ID, target ID, judgment time, backup path and explicit live-store override.

The graph planner's `explicit-node-merges` scope disables general alias cleanup, entity merging and goal-history backfills. It retains the original node-merge mechanics: alias/source unions, archive/supersession pointers, reference and edge re-aiming, duplicate-edge evidence transfer, closing edges whose endpoints coalesce, re-embedding and generation-guarded outbox drains. Changed aliases use the repository decoder's normalization so the planned mirror agrees with its eventual outbox sync. Historical judgment audits remain immutable under later reference projection.

If those reference rewrites would alter a node judged `other`/`unsure` (for example its `corrected_by` pointer), the whole plan refuses for separate operator review. It never silently modifies a non-merge node or an entity row. The plan refreshes the report-only entity list from the current store.

The verdict file is bound to its canonical bank path. Duplicate IDs, malformed structures, inconsistent model IDs or dossier hashes, a changed candidate set, or changed evidence refuse. Apply rechecks the dossiers before creating its journal. Stop writers throughout; this is an offline repair, not a concurrent-store transaction protocol.

## Recovery and live-path guard

The script uses the **same** version-2 `entity-graph-repair/pending.json` journal and `<run-id>.json` archives as the first repair, with an additional versioned, checksummed judged-plan payload. Judgments and graph SQL changes commit together; Lance commits, outbox drains and completion/archive follow the existing machinery. Resume through this script so backup, confirmation, checksum, migrations and dossier guards are repeated:

```sh
pnpm tsx scripts/judge-self-label-nodes.ts \
  --data-dir /path/to/stopped-bank --apply --resume \
  --backup-verified /path/to/verified-backup
```

Omit `--verdicts` on resume: the journal retains the exact reviewed verdicts and vectors. Before the SQL commit, resume requires the original dossier hashes; afterwards, it requires the hashes of the planned resulting dossiers. This permits interrupted merges without accepting unrelated changes. Completion/archive recovery does not duplicate judgments. An old verdict file is not a new approval for a changed or already merged node. A fresh apply of a still-current file containing only non-merges records a new audited run; use resume to finish the original run without duplicate records.

Paths lexically or through symlinks under `demo/server/.borg-data` refuse by default. As with the other audited repairs, an explicit `--live-store-confirmed /path/to/backup-directory` permits an operator-selected guarded store only when that directory contains a regular `borg.db` or `borg.db-clean`. This flag names the **backup**, not the target store, and does not replace `--backup-verified`. The canonical override is stored in the plan and all audits; repeat the identical backup and override paths on resume. The implementation and tests use disposable fixtures only; they do not open the real live directory.

Graph data can be reversed using the existing offline entity-graph audit-based reversal, subject to its unchanged-row checks. Judgment history is retained. No new reversal dispatcher, schema migration, automatic backup, entity-row merge, live-store run, service restart or online interpretation change is included.
