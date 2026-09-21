# Judged self-label node repair

Opus 5.0: in-scope because identity merges require explicit evidence, a recorded judgment, and crash-safe persistence independently of model intelligence.

`scripts/judge-self-label-nodes.ts` is an offline second pass after the ID-backed entity-graph repair. Stop **every writer** for judging, planning, applying and recovery. The script opens existing repositories without starting Borg or running migrations. Pending repairs and pending/unknown migrations refuse.

A matching label selects a candidate; it never proves identity. The self entity comes from `EntityRepository.getSelf()` (`entities.kind = 'self'`). The comparison node must be active, unarchived, unsuperseded and have `kind = 'self'`. When multiple such nodes exist, `--self-node <semantic-node-id>` is required in every phase, including resume. Refusal lists all eligible anchors with `id`, `label`, `created_at`, `edge_count` (incoming and outgoing records), and `source_count` (stored source episode IDs). The flag is optional when exactly one anchor exists; an explicit ID must still name an eligible anchor. Missing anchors refuse. Labels, aliases, dates and descriptions never choose an anchor automatically.

Candidates have kind `entity` or `person`, are neither archived nor superseded, and their trimmed, case-insensitive label equals the self canonical name or an alias. No entity name is hardcoded. Other active `kind = 'self'` nodes appear with the same inventory fields in the plan's `other_self_nodes_report_only` section; they are never judged or merged by this script, even when their labels match a self name/alias. Other entity rows whose canonical name matches the self canonical name appear in `matching_entities_report_only`; they are never merged.

## Judge: read the store, write a verdict file

```sh
pnpm tsx scripts/judge-self-label-nodes.ts judge \
  --data-dir /path/to/stopped-bank --out /path/to/new-verdicts.json \
  --self-node '<semantic-node-id>'
```

`judge` is the default when no verdict file or apply flag is supplied. Without `--out`, a unique JSON file is written beneath the bank's `audited-data-repairs/` directory. Rerun with the same `--out` path and selected anchor to resume an existing verdict file: successful judgments with unchanged dossier hashes are reused; failed, missing or changed dossiers are judged again. Bank/model mismatches, malformed files and edits made by another writer refuse. Changing the selected anchor ID requires a new output file. If recorded nodes are no longer candidates, preserve the old file and choose a new output. Writes are atomic with mode `0600`; directories are synced. Its contents include private memory excerpts and should be handled like the bank backup.

Each dossier contains the node ID, kind, label, description, domain, aliases, confidence, creation time and source episode IDs; all incoming and outgoing edges with relation, validity and the opposite endpoint's label/kind; and the self entity's name/aliases plus the canonical self node description. Descriptions are mechanically capped at 4,000 UTF-16 units without splitting surrogate pairs. Up to six source episodes are read through `EpisodicRepository.get(..., { includeArchived: true })`. Each supplies its title (240 units), stored gist or narrative prefix (800 units), and origin/disclosure metadata. Missing sources, truncation and omitted episode counts are explicit. No language-specific tokenization, message-text parsing, semantic heuristics or audience recall filters are used.

The repository LLM factory supplies the configured `anthropic.models.extraction` model. Two calls run concurrently. Each node gets one logical forced structured tool call returning `{ verdict: "self" | "other" | "unsure", reason, confidence }`. The prompt warns about other bearers of the same name, treats supplied records as untrusted evidence, and permits `unsure`. Client transport retry policy remains in effect; there is no second semantic vote, confidence threshold or prose fallback. A rejected call or missing/malformed tool response creates an explicit `status: "failed"` record containing its dossier/hash, timestamp, error class/message and cause class/message. Other nodes continue. The CLI reports failure after saving all outcomes; no repair is applied.

Version-3 verdict files save the selected anchor as `self_node: { id, label }` and the complete candidate ID manifest before requesting judgments, then persist each success or failure as it completes, without waiting for the other concurrent request. `verdicts` holds successes and `failures` holds failed outcomes; missing IDs are manifest entries in neither list. `completed_at` remains null until all attempts finish. Each outcome retains the presented dossier, SHA-256, model ID and timestamp, alongside batch timestamps and report-only entity rows. Full node, self target and incident edge/endpoint snapshots contribute hashes, including fields outside displayed excerpts. The selected source episodes' complete non-vector content and gist also contribute hashes. Episodes beyond the six-source cap are not inspected. Finished dossiers are rechecked; any that changed become failures without discarding other saved verdicts. Version-1/2 files remain readable; planning verifies their existing dossier bindings before recording the selected anchor, and resume-judging upgrades unchanged judgments without repeated model calls.

## Plan, then apply with a verified backup

A response must contain exactly one `JudgeSelfLabelNode` tool call. Multiple calls, including conflicting `self` and `unsure` calls, fail the node instead of selecting the first. This checks protocol cardinality only; it does not reinterpret the judgment text.

Plan/apply refuse any file with failed or missing judgments and list every affected node ID with its status. The current candidate inventory is also checked, so a missing node cannot be hidden by removing it from the saved manifest. Resume-judging uses `judge --out /path/to/existing-verdicts.json`; `--apply --resume` remains reserved for repair journal recovery.

```sh
pnpm tsx scripts/judge-self-label-nodes.ts plan \
  --data-dir /path/to/stopped-bank --verdicts /path/to/new-verdicts.json \
  --self-node '<semantic-node-id>'

pnpm tsx scripts/judge-self-label-nodes.ts \
  --data-dir /path/to/stopped-bank --verdicts /path/to/new-verdicts.json \
  --self-node '<semantic-node-id>' \
  --apply --backup-verified /path/to/verified-backup
```

`apply` is also accepted as a command in place of `--apply`. Otherwise a verdict file selects a dry run. Repeat the same `--self-node` whenever multiple active anchors exist; the chosen ID/label must match the verdict file and plan. Apply requires `--backup-verified` naming an existing independent file or directory; the operator is responsible for verifying its freshness and integrity. This uses the same independent-backup check as the period-rollover repair. Paths inside the bank, containing it, or naming the bank database itself refuse. Supply the flag during a dry run too if reviewing a plan that records the intended backup.

The complete JSON plan reaches stdout before store mutation. It contains every SQL/Lance change, vectors and reversal audits, per-node judgments, `selfLabelJudgments.self_node`, report-only self nodes/entities, backup/override metadata, and pending/archive journal paths. Planning a merge uses the existing configured embedding client to obtain actual replacement vectors; judging and plans containing only non-merges do not request embeddings. Plan/apply never call the judgment model.

| Verdict | Data effect | Audit |
| --- | --- | --- |
| `self` | Explicit merge into the canonical self node through `planEntityGraphRepair` / `applyEntityGraphRepair`. | `correction`, `judge-self-label:self`, with `targets.judgment`. Existing graph audit rows carry full reversal snapshots. |
| `other` | No merge or node modification. | `correction`, `judge-self-label:other`, reversal `{ kind: "none", data_changes: false }`. |
| `unsure` | No merge or node modification. | `correction`, `judge-self-label:unsure`, reversal `{ kind: "none", data_changes: false }`. |

Every judgment is recorded, including low-confidence `self` and high-confidence non-merges; confidence is evidence metadata, not a deterministic override. `targets.judgment` contains exactly `verdict`, `reason`, `confidence`, `model`, and `dossier_hash`. Judgment audit rows additionally retain the source node ID, target ID and judgment time. Every new audit row, including graph reversal snapshots and completion markers, carries `targets.self_node: { id, label }`, the backup path and explicit live-store override.

Each judgment audit carries `targets.judgment_id`, a stable SHA-256 of `{ node_id, dossier_hash, verdict }`. Planning skips matching unreverted audits, including older rows without the explicit ID field by checking the same tuple. It reports each judgment as `record` or `already_recorded`. An unchanged, still-current non-merge file produces a null run ID and `no changes`; apply creates no audit, completion marker or journal. Reapplying an audit-only plan also rechecks these identities before writing, so it remains a no-op if another completed apply already recorded them. A partially outdated audit plan refuses for re-planning. Changed dossiers still require re-judging; a changed dossier hash or verdict has a distinct identity. Existing audit text is retained, not overwritten by a new reason/confidence for the same identity.

The graph planner's `explicit-node-merges` scope disables general alias cleanup, entity merging and goal-history backfills. It retains the original node-merge mechanics: alias/source unions, archive/supersession pointers, reference and edge re-aiming, duplicate-edge evidence transfer, closing edges whose endpoints coalesce, re-embedding and generation-guarded outbox drains. Changed aliases use the repository decoder's normalization so the planned mirror agrees with its eventual outbox sync. Historical judgment audits remain immutable under later reference projection.

If those reference rewrites would alter a node judged `other`/`unsure` or a report-only self node (for example its `corrected_by` pointer), the whole plan refuses for separate operator review. It never silently modifies a non-merge node or an entity row. The plan refreshes both report-only lists from the current store.

The verdict file is bound to its canonical bank path. Duplicate IDs, malformed structures, inconsistent model IDs or dossier hashes, a changed candidate set, or changed evidence refuse. Apply rechecks the dossiers before creating its journal. Stop writers throughout; this is an offline repair, not a concurrent-store transaction protocol.

## Recovery and live-path guard

The journal envelope records `kind: "judge-self-label-nodes"`. The shared recovery engine rejects this kind through the legacy CLI and accepts it only through the owning script's delegated path after its preflight. Kind/origin mismatches refuse; pre-kind judged journals are recognized from their plan metadata and have the same restriction.

The script uses the **same** version-2 `entity-graph-repair/pending.json` journal and `<run-id>.json` archives as the first repair, with an additional versioned, checksummed judged-plan payload. Judgments and graph SQL changes commit together; Lance commits, outbox drains and completion/archive follow the existing machinery. Resume through this script so backup, confirmation, checksum, migrations and dossier guards are repeated:

```sh
pnpm tsx scripts/judge-self-label-nodes.ts \
  --data-dir /path/to/stopped-bank --apply --resume \
  --self-node '<semantic-node-id>' \
  --backup-verified /path/to/verified-backup
```

Omit `--verdicts` on resume: the journal retains the exact reviewed verdicts and vectors. Before the SQL commit, resume requires the original dossier hashes; afterwards, it requires the hashes of the planned resulting dossiers. This permits interrupted merges without accepting unrelated changes. Completion/archive recovery does not duplicate judgments. An old verdict file is not a new approval for a changed or already merged node. Re-applying an unchanged non-merge file reports `no changes`; use resume to finish an interrupted run.

Paths lexically or through symlinks under `demo/server/.borg-data` refuse by default. As with the other audited repairs, an explicit `--live-store-confirmed /path/to/backup-directory` permits an operator-selected guarded store only when that directory contains a regular `borg.db` or `borg.db-clean`. This flag names the **backup**, not the target store, and does not replace `--backup-verified`. The canonical override is stored in the plan and all audits; repeat the identical backup and override paths on resume. The implementation and tests use disposable fixtures only; they do not open the real live directory.

Graph data can be reversed using the existing offline entity-graph audit-based reversal, subject to its unchanged-row checks. Judgment history is retained. No new reversal dispatcher, schema migration, automatic backup, entity-row merge, live-store run, service restart or online interpretation change is included.
