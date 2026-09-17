# Offline entity graph repair

Opus 5.0: in-scope because entity duplication, divergent audience bindings, and stale stored references are structural persistence errors.

Run `scripts/repair-entity-graph.ts` with all writers stopped and a verified full backup. The default is a read-only plan; `--apply` performs the repair. The script opens the existing SQLite database without migrations and inspects existing Lance tables without creating any. It never opens Borg, starts cognition, or calls an LLM. Only apply calls the configured embedding service.

```sh
pnpm tsx scripts/repair-entity-graph.ts \
  --data-dir /path/to/stopped-bank \
  --merge <named-person-id>=<self-id> --rename-into '<merged-canonical-name>' \
  --merge <transport-entity-id>=<participant-id> \
  --merge-node <duplicate-node-id>=<survivor-node-id>
```

Repeat `--merge` and `--merge-node` as needed. `--rename-into` applies to the preceding entity merge, must equal that source's canonical name, and transfers its name provenance. Without it, the survivor keeps its canonical name and provenance. Old canonical names and aliases become survivor aliases. The self entity must survive. Chains, cycles, absent IDs, ambiguous primary-key collisions, and incompatible node kinds are rejected.

Node merges require explicit operator-selected IDs. Equal names, leaked ID aliases, and similar descriptions are never treated as evidence of identity. This prevents a wrongly attached ID alias from driving a wrong-person merge. The survivor unions aliases and source episodes; duplicate nodes retain their original descriptions as archived, superseded records pointing to the survivor. Edges and belief dependencies are redirected. Colliding open edge triples union evidence with the same confidence/time policy as `mergeOpenEdgeEvidence`; redundant edge/dependency rows are removed with reversal snapshots.

The plan prints every scanned SQL table and column, per-table update/delete counts, detailed entity/node/edge before/after rows, reconstructed goal events, file hashes and record counts, Lance counts, outbox/audit writes, and unresolved action/history reports. No real store names or IDs are compiled into the script or its tests.

## Why resolution diverged

The demo's `resolveTurnAudienceEntityId` searched with `kind: "person"`, while `EpisodicExtractor.resolveStreamAudience` searched names/aliases across kinds and inferred self from a literal label. An explicit session binding also did not register its transport label as an alias. Consequently, a session could bind an existing participant while later extraction minted a room identity from the label. The journal tool separately looked up self by name, making renames unsafe. Ordinary label resolution could also promote the provenance of the canonical name just because a transport alias appeared.

`EntityRepository.resolveAudienceLabel` now resolves existing IDs, canonical names, and aliases without a kind filter. Explicit bindings register their labels and reject conflicting bindings. Production `SessionsRepository` uses that resolver for every binding caller, including ingestion and activity projection; extraction, turn coordination, and the public facade use the same method. Existing canonical-name provenance is preserved.

Composition seeds self only when no self-kind row exists. Self resolution uses kind, rejects multiple self rows, and preserves the ID across renames. `ensureSelf` preserves previous canonical names as aliases instead of continually injecting a literal name.

## Coverage and recovery

The planner discovers all SQLite tables/columns and all Lance tables, then follows exact IDs, participant terms, JSON values, JSON map keys, and nested serialized JSON. This covers sessions; episode audience/origin/participant indexes and mirrors; all four commitment entity fields; goal ownership/audience/counterparty/block history; identity-event snapshots; actions; open questions; disclosure labels; social state; external mappings; shared state; train-of-thought and lived-experience records; semantic graph references; review and existing maintenance records; attachment/perception metadata; and extension tables. Traits/values currently have episode evidence rather than direct entity columns, but are included in the scan.

Structured stream metadata and working-memory files are included. Stream text and prose retain their historical wording. Modified stream files get matching byte offsets in `stream_entry_index`. Trace logs, binary attachment blobs, configuration backups, and this repair's reversal records remain historical artifacts. Raw entity-ID tokens in semantic labels resolve through entity records; ID-bearing aliases are discarded, including incorrect bindings. Descriptions are retained.

There is no atomic transaction spanning SQLite, Lance, and JSONL files. Before modifying the store, apply generates embeddings and fsyncs a journal at `entity-graph-repair/pending.json`. SQLite changes, outbox entries, and maintenance audits commit together. External writes check before/after snapshots and can resume idempotently. Completion archives the journal under its run ID.

After an interruption, keep every writer stopped and run:

```sh
pnpm tsx scripts/repair-entity-graph.ts --data-dir /path/to/stopped-bank --apply --resume
```

Do not restart until completion. If the SQLite phase rejected a stale plan, no store changes from that run committed: preserve the pending journal elsewhere and create a fresh plan. If it committed, use resume or restore the verified full backup. Audit reversal payloads and the completed journal contain SQL row snapshots, external row metadata, replaced embeddings, and file-line originals; these offline operations are not registered with the running service's `audit.revert` API.

`syncPendingVectorUpdates` is called from belief revision, not boot, and retains existing embeddings. Consequently the script cannot merely enqueue work. It generates new vectors using the configured client and the extractor's embedding-text builder, writes touched semantic mirrors (including archived duplicates), then drains those node IDs with the repository. A failed drain leaves the repair pending.

Missing goal audit versions are reconstructed only when adjacent stored snapshots exist. `IdentityEventRepository.record` writes manual `repair_missing_update` events without changing goal contents or versions. Their reasons record the actual repair time and identify the historical ordering timestamp as a placeholder; neither the original author nor exact write time is invented. Gaps without adequate snapshots are reported.

Scheduled actions are reported with their complete stored rows. `classifyActionArchiveCandidate` explicitly excludes scheduled actions; their schedule stamps are state-entry timestamps, not due dates. That explains accumulation but cannot establish completion, cancellation, or continued validity of an individual action. This repair does not infer lifecycle outcomes from descriptions or deployed commit names.

## Existing utilities and new code

Searches covered the entity and semantic repositories, consolidator/review resolver, identity repositories, offline audit log, embedding setup, and existing `scripts/repair-*` utilities. Reused:

- `getSelf`, name/alias lookup, and existing entity persistence conventions.
- `SqliteDatabase`/`SqliteRawDatabase`; the existing repair scripts' read-only, no-migration open pattern.
- `GoalsRepository.get`, `IdentityEventRepository.record`, and `AuditLog.record`.
- Semantic lifecycle fields and open-edge evidence-union SQL conventions. The consolidator merges episodes, not entity graphs.
- `SemanticNodeRepository.syncPendingVectorUpdates` and its outbox generation UPSERT.
- `buildNodeEmbeddingText` (now exported), `createEmbeddingClient`, and `LanceDbTable` materialization/upserts.
- Branded-ID validators, participant-term codecs, and fsync/rename atomic-file helpers.

No existing utility merges entity references across SQL, Lance, and files. The new planner, structural ID walker, collision checks, and resumable journal are script-local to support that operation. Existing goal repair scripts handle different defects and do not reconstruct missing versions. No dependencies were added.

## Backup validation (2026-09-17)

The required backup `demo-20260916-074852` was copied to `/tmp/repair-dryrun-data`. The real dry run inspected 67 SQLite tables and seven Lance tables. Full output is at `/tmp/repair-entity-graph-dryrun.jsonl`. The database, WAL, and three affected JSON/JSONL files were byte-identical to the backup afterward; SQLite `quick_check` returned `ok`. No live-store writes were made.

| SQLite table | Updates | Deletes |
| --- | ---: | ---: |
| action_records | 1,935 | 0 |
| activity_events | 959 | 0 |
| commitments | 6 | 0 |
| entities | 2 | 2 |
| episode_index | 793 | 0 |
| episode_participants | 1 | 1 |
| goals | 6 | 0 |
| growth_markers | 29 | 0 |
| identity_events | 319 | 0 |
| lived_experience_day_summaries | 37 | 0 |
| maintenance_audit | 139 | 0 |
| observed_events | 4 | 0 |
| open_questions | 104 | 0 |
| procedural_evidence | 2 | 0 |
| review_queue | 207 | 0 |
| semantic_belief_dependencies | 0 | 16 |
| semantic_edges | 80 | 17 |
| semantic_nodes | 16 | 0 |
| shared_state_artifacts | 1 | 0 |
| shared_state_entries | 45 | 0 |
| social_events | 405 | 0 |
| social_profiles | 1 | 0 |
| stream_entry_index | 6,674 | 0 |

Other scanned tables have zero changes, including sessions: the supplied session already points at the selected survivor. Of the 793 affected episode rows, 785 change the primary audience and 793 change origin audiences. The totals are 11,801 existing SQLite row changes, plus five identity-event inserts, 16 outbox enqueues/drains, and 53 new maintenance-audit rows.

Lance changes: 1,935 action records, 793 episodes, four observed events, and 16 semantic nodes (2,748 total). Five semantic duplicates are archived; 17 colliding edges and 16 duplicate belief dependencies are removed. The semantic cleanup changes five ID-bearing labels and removes IDs from nine stored alias lists; merging contributes the remaining alias change.

Files: 755 structured records in `stream/default.jsonl`, 163 in the supplied counterpart session stream, and one working-memory file. Their changed byte lengths account for the 6,674 stream index offset updates.

All five reported goal gaps are reconstructable: `goal_rgj15to5kj4b2zge` version 2, `goal_2cg2qiub7x8yz2tx` version 2, `goal_2g6jc2cjri8745mc` version 19, `goal_om4jgqz4rafgj2pr` version 8, and `goal_805c16bdu93n8fy6` version 7. No unreconstructable gap was found.

The backup contains 245 scheduled actions, predating the later 247-row census. `act_n3drktjsrrip1c49` remains scheduled and retains its historical deployment claim. Correcting that claim or retiring those actions requires outcome evidence or an explicit lifecycle-policy decision; their audience references are still repaired.

Focused validation: nine test files, 222 passing tests. Tests use temporary stores constructed through repositories and fake embeddings, covering cross-kind resolution, renamed self, session aliases, external IDs, commitment fields, nested/extension JSON, stream offsets, duplicate graph dependencies, existing outbox work, goal backfills, idempotence, stale-plan rejection, and interrupted-apply recovery.

The single required `pnpm typecheck` run passed. A separate strict TypeScript check of the repair scripts and tests passed, as did `pnpm heuristics:guard` and `git diff --check`. The full test suite was not run.

## File inventory

- Resolution and binding: `src/memory/commitments/repository.ts`, `src/sessions/repository.ts`, `src/memory/episodic/extractor.ts`, `src/cognition/lifecycle/turn-phase-coordinator.ts`, `demo/server/src/app.ts`.
- Composition and public API: `src/borg/repositories.ts`, `src/borg/tools-setup.ts`, `src/borg/facade.ts`, `src/borg/facade-types.ts`, `src/borg/public-facade.ts`.
- Existing embedding-text builder export: `src/memory/semantic/extractor.ts`.
- Offline repair: `scripts/repair-entity-graph.ts`, `scripts/repair-entity-graph-plan.ts`, `scripts/repair-entity-graph-storage.ts`.
- Tests: `scripts/repair-entity-graph.test.ts`, `src/memory/commitments/entity-resolution.test.ts`, `src/memory/episodic/extractor.test.ts`, `src/borg/__tests__/sessions.test.ts`.
- Operator report and instructions: `docs/repair-entity-graph.md`.
