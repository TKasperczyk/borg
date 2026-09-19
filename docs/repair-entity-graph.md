# Offline entity graph repair

Opus 5.0: in-scope because duplicate identities and stale stored references are structural persistence errors.

Stop every writer and retain a verified backup before applying. `scripts/repair-entity-graph.ts` defaults to a read-only plan. `--apply` persists that plan. It opens existing SQLite and Lance stores without starting Borg, cognition, or an LLM. Planning calls the configured embedding endpoint when semantic rows change, so the complete plan contains the actual replacement vectors before any store mutation.

```sh
pnpm tsx scripts/repair-entity-graph.ts \
  --data-dir /path/to/stopped-bank \
  --merge <named-person-id>=<self-id> --rename-into '<merged-canonical-name>' \
  --merge <transport-entity-id>=<participant-id> \
  --merge-node <duplicate-node-id>=<survivor-node-id>
```

Repeat the merge flags as needed. `--rename-into` applies to its preceding entity merge, must equal the source's canonical name, and transfers its name provenance. Otherwise the survivor keeps its name and provenance. Both entities' other names become aliases. The survivor must be a live handle; the self entity must survive. Entity and node identities are selected explicitly, never inferred from names, descriptions, or leaked ID aliases.

## Redirects and readers

Merged entity rows remain in `entities`, with `merged_into` pointing to the survivor. The normal `entity_identity_redirects` migration (7,000,004 in the composed migration list) adds that nullable foreign key, its index, a unique live-self index, and the recursive `resolved_entities` view. The offline planner permits this reviewed additive migration only; other pending migrations require a separate review. Its exact DDL and migration row are included in the plan.

`EntityRepository.get` follows redirect chains and rejects cycles or dangling destinations. Name/alias searches and entity listings exclude redirected rows; raw-ID and external-ID lookups resolve the survivor. `resolveAudienceLabel` uses the same mechanism for session binding and episode stamping, across entity kinds. Explicit bindings retain their transport labels as aliases. `rename` retains the prior canonical name. Self lookup uses kind; the literal seed name is used only when creating an absent self identity.

Read-time projection leaves historical files untouched:

- `StreamEntryIndexRepository` resolves returned entity handles without changing its stored rows or byte offsets.
- `StreamReader` and indexed hydration resolve structured handles through the repository/index. Composition supplies the resolver to production readers. Episodic and offline semantic extraction also supply it directly.
- Recent raw-stream recall, indexed and fallback citation lookup, live ingestion scans, and thought-record tools use the same projection.
- `WorkingMemoryStore.load` projects persisted handles in memory. Production composition injects the entity repository.
- `SocialRepository` canonicalizes incoming entity handles for profile/event reads and attribution writes. Activity display joins use `resolved_entities`.
- Existing facade, participant, speaker, and disclosure-label consumers already call `get`, `findByNames`, or external/name resolution; these now follow redirects.

The original divergence was between the demo's person-only audience lookup and extraction's across-kind lookup. An explicit session binding did not register its transport alias. The shared resolver closes both paths. Looking up self by a literal name was a separate source of rename instability.

## Store repair and validation

The planner discovers all SQLite tables/columns and Lance schemas. It moves exact machine handles, participant terms, JSON values, JSON keys, and nested serialized JSON. It never substitutes inside prose. Changed records pass the repositories' existing decoders/Zod schemas; an unknown changed table fails closed until its adapter is supplied. A private SQLite projection runs actual constraints, triggers, and foreign-key checks before target writes.

Encoded-key searches covered the repositories, retrieval, cognition, autonomy, and offline processes:

| Key family | Treatment |
| --- | --- |
| Observed-event recurrence | Rebuilt with `buildObservedEventRecurrenceKey` from session, speaker and classification fields; collisions rejected. |
| Recall suppression `semantic_node:` / `semantic_edge:` | Handle keys are redirected; colliding expiry turns retain the later value. Scope key and serialized state must agree. |
| Episode participant `entity_id:` | Rebuilt with the existing participant-term codec. |
| Open-question dedupe | Rebuilt with `buildOpenQuestionDedupeKey` when audience or related nodes change; conflicting rows rejected. |
| Commitment reconciliation scope/detection keys | Stored JSON objects or serialized JSON; the structural walker handles these. |
| Procedural context keys | Derived from problem kind, domain tags and audience category, with no entity/node/edge ID input. Unchanged. |
| Observed-event fire keys, action/goal wake keys, lived-experience bucket keys | Stream, goal/step, day and session handles are unchanged by this repair. |
| Semantic revision verdict cache and retrieval evidence keys | Process-local caches/derived render objects, rebuilt after the stopped service restarts. No persisted cache to rewrite. |
| Open-question merge/suppression keys | Open-question IDs are not merged. |

Source streams, working files, and `stream_entry_index` are protected, with zero planned writes. Every file under `stream/` and `working/` has a SHA-256 in the plan. Their bytes and the entire stream index are checked before and after persistence. Trace logs, configuration, and binary attachments remain unchanged.

Semantic node merges union aliases/source episodes and archive duplicates as superseded, preserving their descriptions. Raw entity IDs are removed from every alias; ID-bearing labels use the referenced entity's canonical name. A wrong ID alias is discarded without treating it as identity evidence.

Open-edge collisions transfer evidence through `SemanticEdgeRepository.mergeOpenEdgeEvidence` and retain the redundant edge as closed. Edges whose endpoints coalesce are closed while retaining their historical endpoints, preventing self-edges. Pair-review arrays retain their two-element shape; obsolete pairs are explicitly dismissed. Arrays are not generically deduplicated.

Parent collisions are rejected before any deletion, even if deleting the parent would satisfy a later foreign-key check through cascades. Only identical leaf references may coalesce. Every deletion, trigger insertion, changed field, and affected row count is checked and represented in the plan and reversal payloads.

## Plan, persistence and recovery

The JSONL plan contains every table's scanned columns and counts, every changed SQLite row/key/value, every changed Lance row/value, replacement and prior vectors, exact schema changes, identity-event inserts, outbox enqueues/drains and mirror upserts, audit rows, protected file hashes, journal paths, and embedding model/dimensions/endpoint. CLI apply waits until the whole plan has been emitted before mutation. Plans can be large because audit payloads contain complete reversal snapshots.

Persistence boundaries are:

1. Fsync the complete version-2 journal at `entity-graph-repair/pending.json`.
2. Commit the reviewed migration, SQLite changes, goal events, outbox enqueues and audit records in one transaction. Assert before/after rows and affected counts.
3. Commit each changed Lance table, preserving all untouched vectors. Before/after comparisons make completed commits idempotent on resume.
4. Drain the touched semantic nodes with the repository's generation-guarded outbox path. The service drains this outbox from belief revision, not boot, so the offline repair completes it itself.
5. Commit the completion marker; atomically archive and fsync the journal as `<run-id>.json`.

```sh
pnpm tsx scripts/repair-entity-graph.ts --data-dir /path/to/stopped-bank --apply --resume
```

Keep writers stopped until resume completes. An interruption after journal archival is recognized from the durable completion marker. Outbox errors retain the pending operation for retry. A zero-change rerun creates no journal, audit, migration, embedding request, SQLite write, or Lance commit.

Reversal is an offline operation built from `maintenance_audit` rows, independently of the forward journal:

```sh
pnpm tsx scripts/repair-entity-graph.ts --data-dir /path/to/stopped-bank --reverse <run-id>
pnpm tsx scripts/repair-entity-graph.ts --data-dir /path/to/stopped-bank --reverse <run-id> --apply
```

Reversal refuses intervening modifications to affected rows. It restores prior SQLite data, deleted leaf rows, original Lance metadata/vectors and prior outbox state, then marks the original audit records reversed. It retains the additive schema migration, monotonic SQLite sequence counters, and repair/reversal audit history. The generic online `audit.revert` dispatcher does not implement this cross-store operation.

Missing goal events are reconstructed only from adjacent stored snapshots. `IdentityEventRepository.record` writes manual `repair_missing_update` events without changing goal content or versions. Reasons identify the repair time and historical ordering timestamp as a placeholder, since the original author and exact write time are unknown. Unreconstructable gaps are reported.

Scheduled actions are reported unchanged except for structural references. The existing archive policy excludes scheduled actions; their stamps record state entry, not a deadline. Neither a stale deployment claim nor an old stamp proves completion or cancellation. Lifecycle changes require outcome evidence or an explicit policy decision.

## Reused utilities and additions

Searches covered entity/semantic repositories, consolidator and review handlers, existing `scripts/repair-*` code, identity/audit repositories, key builders, embedding setup, and atomic-file helpers.

Reused directly: `EntityRepository` name/alias/self resolution; `runMigrations`; `SqliteDatabase`/`SqliteRawDatabase`; repository row decoders and schemas; `buildObservedEventRecurrenceKey`; `buildOpenQuestionDedupeKey`; participant-term codecs; `GoalsRepository.get`; `IdentityEventRepository.record`; `AuditLog.record`; `SemanticEdgeRepository.mergeOpenEdgeEvidence`; `SemanticNodeRepository.enqueueVectorSyncRow` and `syncPendingVectorUpdates`; `buildNodeEmbeddingText`; configured embedding client; `LanceDbTable`; branded IDs; and atomic fsync/rename helpers.

No existing utility performed an entity merge through redirects or coordinated a reversible SQLite/Lance repair. The added repository redirect mechanism serves ordinary readers. Script-local inventory/projection, validation adapters, complete-plan formatting, resume journal and audit-based reversal coordinate the offline operation. No dependency was added. The consolidator's episode merge is not an entity-graph merge; its archive/supersession conventions are retained here.

## Fresh backup rehearsal

The operator prepared `/home/luth/tmp/repair-dryrun-data` from `demo-20260917-224047`. It was neither recopied nor deleted. A separate interruption copy was created with the requested `cp -r` command at `/home/luth/tmp/repair-resume-d2aaf222`.

The fresh dry run scanned 67 SQLite tables and seven Lance tables; all 7,290 target files retained their hashes. Machine-readable plans, per-row checksum manifests, verification results and logs are under `/home/luth/tmp/entity-graph-rehearsal-d2aaf222/`.

| SQLite table | Updates | Inserts | Deletes |
| --- | ---: | ---: | ---: |
| action_records | 2,034 | 0 | 0 |
| activity_events | 993 | 0 | 0 |
| commitments | 6 | 0 | 0 |
| entities | 4 | 0 | 0 |
| episode_index | 830 | 0 | 0 |
| episode_participants | 1 | 0 | 1 |
| goals | 6 | 0 | 0 |
| growth_markers | 29 | 0 | 0 |
| identity_events | 325 | 0 | 0 |
| lived_experience_day_summaries | 38 | 0 | 0 |
| maintenance_audit | 141 | 0 | 0 |
| observed_events | 4 | 0 | 0 |
| open_questions | 106 | 0 | 0 |
| procedural_evidence | 2 | 0 | 0 |
| review_queue | 210 | 0 | 0 |
| semantic_belief_dependencies | 0 | 0 | 16 |
| semantic_edge_invalidation_events | 0 | 17 | 0 |
| semantic_edges | 97 | 0 | 0 |
| semantic_nodes | 16 | 0 | 0 |
| shared_state_artifacts | 1 | 0 | 0 |
| shared_state_entries | 44 | 0 | 0 |
| social_events | 418 | 0 | 0 |
| social_profiles | 1 | 0 | 0 |
| stream_entry_index | 0 | 0 | 0 |

All other scanned tables, including sessions, have zero base changes. Total: 5,340 base SQLite changes. Additional writes: five identity events, 16 outbox enqueues followed by 16 drains, 31 maintenance-audit rows, and one migration row. Lance: 2,034 actions, 830 episodes, four observed events, 16 semantic nodes; 2,884 rows total. Both entity sources remain as redirects. Five node duplicates are archived, 17 redundant open edges are closed, and 17 duplicate leaf references are deleted with reversal snapshots.

The embedding configuration is `text-embedding-qwen3-8b-text-embedding`, 4,096 dimensions, `http://localhost:1234/v1`. There are five reconstructable goal gaps and zero unsupported gaps. The fresh backup has 256 scheduled actions whose outcomes remain unresolved.

The preserved outbox-failure test was already present and passed. This continuation finished missing reader wiring, remaining built-in reference validation, bounded-memory reversal planning, stdout completion before writes, migration/sequence tests, documentation, and the full backup rehearsal.

Both copies verified 5,392 planned SQL row operations and 2,884 Lance changes. Independent scans of 218,103 post-apply SQLite rows and all 12,932 Lance rows found zero unexpected source handles. Two entity definitions remain as redirects and five node definitions remain archived. All 12,916 untouched vectors matched their baseline hashes; 16 semantic vectors matched the replacement vectors in the plan. All 18 protected files and all 44,034 index rows remained unchanged. `quick_check` returned `ok`; `foreign_key_check` returned zero violations.

The normal no-op checked all 7,370 files; the interrupted copy's no-op checked all 7,373 files. Both formatted plans had a null run/journal and zero changes, and applying them wrote nothing. The extra three files are Lance history from replaying the interrupted outbox mirror commit.

The interruption copy deliberately exited with code 86 at 12 boundaries: after journaling; after SQLite; after each of four Lance commits; inside the first outbox item after its Lance commit but before SQLite acknowledgement; after its acknowledgement; before and after completion; before and after journal archival. Every resume succeeded. Pending outbox counts progressed 16 → 15 → 0; the audit count progressed 30 → 31 with exactly one completion marker. The final resume exited 0 and left no pending journal.

Audit-based reversal restored all 218,066 original SQLite rows and all 12,932 Lance rows, including original vectors, metadata, deleted leaf references and affected historical audit payloads. Reversal performs 5,345 base SQL row operations and 2,884 Lance restorations. The final store has 218,127 rows: the original population plus one migration row and 60 repair/reversal audit rows (31 forward, 29 reverse). `maintenance_audit` therefore has 18,244 rows, including all 18,184 original rows unchanged. No entity redirect remains after reversal. The additive schema and sequence high-water marks intentionally remain.

## Review finding closeout

| Finding | Result and verification |
| --- | --- |
| 1. Immutable provenance / redirects | Normal migration; redirect chains, live-only name lookup and reader projection. All source hashes and 44,034 index rows unchanged through apply, 12 interruptions and reversal. |
| 2. Encoded keys | Four real recurrence keys rebuilt. Suppression handles and open-question keys tested with collisions; key-builder inventory above. |
| 3. Cascades and row counts | Parent collisions reject before mutation. Leaf deletions, trigger inserts and every update have exact plans/reversal data and affected-row assertions. Shared-state child-loss regression passes. |
| 4. Tuple/self-edge validation | Arrays preserve shape; obsolete pairs dismissed; collapsing edges retained closed with historical endpoints. Existing repository schemas and SQLite constraints validate projected records. |
| 5. Rename aliases | Old label retained and tested through shared audience resolution and episode extraction. Redirected handles also rename the survivor. |
| 6. Complete machine plan | Every affected row/value and extra write included, with vectors/configuration. CLI waits for stdout before applying. |
| 7. Failure/reversal coverage | Outbox write-before-ack failure test, all persistence boundaries, whole-copy checks and audit-based reversal all pass. |
| 8. Zero-write reruns / reuse | Both physical file manifests unchanged on rerun; no journal/audits. Existing outbox enqueue and open-edge evidence merge methods reused directly. |

## Checksums and restored counts

Checksums below are SHA-256 over deterministic JSON row-hash manifests; entity rows normalize the newly added nullable `merged_into` field, and the additive migration/new audit rows are excluded from the original-row comparison. Each individual protected file uses SHA-256 of its raw bytes. Complete per-row baselines and after-state manifests are retained with the rehearsal artifacts.


| Manifest | Before and after SHA-256 |
| --- | --- |
| All original SQLite rows | `19fd4b1ac5048d5455f294f868c227aaea49344768b6949d92f6a441e511c796` |
| Affected SQLite tables | `b3ffbf19fe4f23143c51e329fdea732b6aa78ea0e500d3155e2e776bdd4ad8fb` |
| All Lance rows and vectors | `42b57fe461434d74590b05900eb33ed1a85055faebf2fec0b180cf30f606e7c7` |
| Protected file manifest | `c74f9cbb96a2ceebcd6f6df14d70aa470da44b8b6b0775e5e4ebe0d0a7222b98` |
| Stream entry index | `6313bd6a38ef9e70c59b11b77edfe3629011154d30077877789c8b8d47482434` |

| Affected table | Original / restored rows | Original / restored checksum |
| --- | ---: | --- |
| action_records | 4,448 / 4,448 | `62ebaa2f23d9eefc02ab13fa0d15303d664bc702d1b3ddfdbf88aba536e166af` |
| activity_events | 8,853 / 8,853 | `b73063c85063fef234d808acf350b8ab3c1f1d96056bfdaf25765b7f6ea4c2c1` |
| commitments | 251 / 251 | `6f76a6584f162e1a427406bed089f9920c461ffade796cbb32221fd658712e25` |
| entities | 37 / 37 | `a574041ec3343c7818cf580f1ce244f0a3e93bc8444e17cdf93c1d5d54c65e22` |
| episode_index | 2,310 / 2,310 | `bf04402e8ba40a0a0c8c36ee4946fbebfd8fe1bae5c44e624b63284ff3ee44c4` |
| episode_participants | 6,640 / 6,640 | `31d464f54904749ba4f773f0d612c21416d69590900354d772859cf4f7532a91` |
| goals | 238 / 238 | `5367499eb701b79a4c7d616c028093a144a9fbb3b6c0960b4bf8e47c213660ce` |
| growth_markers | 104 / 104 | `6a506be9e1faa41284489079fa186dd89a2b0e066f857a71a33e06f56b386108` |
| identity_events | 19,188 / 19,188 | `f0946f40ce96f0cee9b3668549abaa5db220256e174be914941fd888c8c6018d` |
| lived_experience_day_summaries | 92 / 92 | `e866524f16fbc1b4c2731c911ce64aa721e697949fbccababf7bcaaf00e623f5` |
| maintenance_audit | 18,184 / 18,184 | `45eb7eb98b41f69273d65d4d8d0d29aaf1d49db506978370ae0faee806eb5ca6` |
| observed_events | 22 / 22 | `5c3552631e1972ac71ef971d0b080dbd08cf983b8ae4da1ecfe11cbbcb72a006` |
| open_questions | 417 / 417 | `ad0a7c2c3ec9ee473ff76dbc4df0624ccff2cb28ca88a38a2bee457e3c2ecab4` |
| procedural_evidence | 49 / 49 | `ec2e0d01bfa973b86fb1d3fe96ba433acbfe2c61b54029211fc62b89531a35d4` |
| review_queue | 631 / 631 | `c08076da92db5b793655297167103961aea3981e40f1cdc4769a5c11a608ae45` |
| semantic_belief_dependencies | 9,401 / 9,401 | `5be14d7e498c9d96e8fbd67aa1ab693443a2c43b6d9bff9acecd156023e8a0c6` |
| semantic_edge_invalidation_events | 0 / 0 | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| semantic_edges | 12,225 / 12,225 | `779477d0bfdc00e92e1cf25c7a609ee8b0097ca83bce26e10fb21cf6fc8e180b` |
| semantic_nodes | 5,733 / 5,733 | `3a22af7319d93863fd6fb45caa38721fa2f5f63e10759353fdde3d5ccefab95e` |
| shared_state_artifacts | 10 / 10 | `d5aac7df7240b5330163ba822c2384c0a18357e9f2a783fab609270b6f74c4d1` |
| shared_state_entries | 324 / 324 | `1f389d88bf9d4bfbd6ab6856016350ec07dc2baa4f6bd20930f1e27160acfdee` |
| social_events | 1,642 / 1,642 | `15ebed15d863f02b599f3dc7ac698861ead1f9e31b025e05170ef0736d32b3e7` |
| social_profiles | 16 / 16 | `513acbbc20ba9f1c92ff556ad366620883b0d93df7cc253ced3d33bd7f4092d4` |

| Protected file | Unchanged SHA-256 |
| --- | --- |
| `stream/default.jsonl` | `b92de3a409566ed36f6e36f1641ec1ab93973a3bee8db1af90941fd1c49ccf0d` |
| `stream/sess_1i51d9mjoclmehci.jsonl` | `14fe3beed4632ab9e8300cd7a7614ee00e5ee2a4efc638b8d4e96d15c12f4a8f` |
| `stream/sess_41uuw7ztyzh00kjc.jsonl` | `8244531eef8993b98d567ae2c48b073c72fe40bb0b3036890a3354c19088ce89` |
| `stream/sess_82frs56mddjq0nid.jsonl` | `51945ee426ba00ea67c7f6a159c869f0189bd15a382df5cccc02a0fa0ac9bfee` |
| `stream/sess_8kaz38dcaprbvr94.jsonl` | `2464dc6e2a827afa4045fd40d93bbd03482eda5e1f29d0403fbe862e076b8345` |
| `stream/sess_fxpnxgatoohv3zr5.jsonl` | `e77a340e59715faf1ee3c5c422cd7cdb58846518611ee29a4111d89031b1236c` |
| `stream/sess_o389mso29rjpd7dq.jsonl` | `c6eb08c57b76e7176a34912ea98fc66504a66f9b1ea501072cf9cc1511af8f78` |
| `stream/sess_pszptntooo5ak910.jsonl` | `2ebd4ca6f3f56ad61ec3f1121b3d178dc2de6a7a07926de45cd2c96111b2f604` |
| `stream/sess_xi61nkgcuaakdnqc.jsonl` | `c979c76463a0deb1978958246bf8109a1b703693660ca37cb9ed7e7b47074905` |
| `working/default.json` | `dc196bb3c91a4ecac961a5a4eb450b1f0dc4cdbde7d85e89b08716d0cbdcd4d6` |
| `working/sess_1i51d9mjoclmehci.json` | `fe599d63348ae400131bc22b1e53af49ccd16c59895ca730c24283a158bc0460` |
| `working/sess_41uuw7ztyzh00kjc.json` | `d471cb5bab6e6320f57af7485ad371db9d143aab78c20358acdc17a0dd858e9e` |
| `working/sess_82frs56mddjq0nid.json` | `6501c4878279ed437f8d8aea9f7465085e2d82e72037630a6947265c54ec92bd` |
| `working/sess_8kaz38dcaprbvr94.json` | `b41efe9b552274b9a43a193e2b9387ec9e71bb21560041ed6797d5e52c980f92` |
| `working/sess_fxpnxgatoohv3zr5.json` | `356bd5b4fa16482aace05b507e0dd4e9cefbcc9a1a9871e2046f2dacbbe3c311` |
| `working/sess_o389mso29rjpd7dq.json` | `3ed0f672f795f18fd95dacdb0cc023c767269464e730594d449e73d90e235c6b` |
| `working/sess_pszptntooo5ak910.json` | `5546092d012c264184bd96ac17bb876152a8a582864d365ac2eae83fa1feffb7` |
| `working/sess_xi61nkgcuaakdnqc.json` | `eeb9b5a7f31ae36f04f4e44e798e19223c6ebf6e3372a7ebc1d347e5b7959d35` |

## Validation and artifact paths

The final focused run passed 259 tests in 16 files. The subsequent repair-file run passed all 21 tests, including the additional attachment/perception/operator-attention case (260 distinct tests overall). Tests build temporary stores through repositories and use fake embeddings. The real rehearsal uses the configured local embedding endpoint. The single final `pnpm typecheck` passed. A separate strict TypeScript check of scripts/tests passed after fixing two fixture literal types; those files are excluded by the package typecheck. `pnpm heuristics:guard` and `git diff --check` passed. The full test suite was not run.

Created top-level paths under `/home/luth/tmp`:

- `/home/luth/tmp/entity-graph-rehearsal-d2aaf222/`: helper, complete plans, baseline/after-state row hashes, summaries, logs, checksum manifests, and `test-tmp/` for test fixtures and tsx cache. `normal/` and `interrupted/` hold the two runs. Tests remove their own temporary fixtures.
- `/home/luth/tmp/repair-resume-d2aaf222/`: the one additional backup copy, retained after reversal.

The operator-provided `/home/luth/tmp/repair-dryrun-data/` was used in place and retained after reversal. Its new journal/Lance-history files are listed separately in `created-paths.txt` alongside a recursive inventory of the created paths. No files were written under `/tmp`, no disposable copy was deleted, and the live data directory was never opened.

Open operational question: the 256 scheduled actions, including the reported stale action, need outcome evidence or an explicit lifecycle decision. The repair preserves their state/stamps and reports their complete rows. No unresolved structural repair finding remains in the rehearsed backup.

## Changed files in this continuation

- `docs/repair-entity-graph.md`
- `scripts/repair-entity-graph-plan.ts`
- `scripts/repair-entity-graph-reversal.ts`
- `scripts/repair-entity-graph-validation.ts`
- `scripts/repair-entity-graph.test.ts`
- `scripts/repair-entity-graph.ts`
- `src/borg/ingestion-setup.ts`
- `src/borg/tools-setup.ts`
- `src/cognition/ingestion/coordinator.ts`
- `src/memory/commitments/entity-resolution.test.ts`
- `src/memory/commitments/repository.ts`
- `src/retrieval/citations.ts`
- `src/retrieval/raw-stream-adapter.test.ts`
- `src/retrieval/raw-stream-adapter.ts`

The preserved redirect migration, core readers, repair planner/storage and repository decoder exports remain in `b86869c4`. The complete review-to-final file inventory is in `review-changed-files.txt` under the artifact directory.
