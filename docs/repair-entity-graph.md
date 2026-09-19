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

`EntityRepository.get` follows redirect chains and rejects cycles or dangling destinations. Name/alias searches and entity listings exclude redirected rows; raw-ID and external-ID lookups resolve the survivor. `resolveAudienceLabel` uses the same mechanism for session binding and episode stamping, across entity kinds. Established session bindings win over incoming IDs and label lookups. A label that resolves to another entity emits `entity.audience_binding_mismatch` (or the repository callback) without rebinding or rejecting ingress. Unclaimed transport labels become aliases; a conflicting label is never silently stolen. Every dry run preflights all sessions before projected merges, listing both entity records, primary/origin/session episode counts, and proposed survivor IDs. `rename` retains the prior canonical name. Self lookup uses kind; the literal seed name is used only when creating an absent self identity.

Read-time projection leaves historical files untouched:

- `StreamEntryIndexRepository` resolves returned entity handles without changing its stored rows or byte offsets.
- `StreamReader` and indexed hydration resolve structured handles through the repository/index. Composition supplies the resolver to production readers. Episodic and offline semantic extraction also supply it directly.
- Recent raw-stream recall, indexed and fallback citation lookup, live ingestion scans, and thought-record tools use the same projection.
- `WorkingMemoryStore.load` projects persisted handles in memory. Production composition injects the entity repository.
- `SocialRepository` canonicalizes incoming entity handles for profile/event reads and attribution writes. Activity display joins use `resolved_entities`.
- Demo explicit sender/audience parsing, enqueue persistence and turn orchestration canonicalize identity inputs. Session `get`/`getMany`/`list` return canonical audience IDs; connector reads use these methods.
- Episodic reads and disclosure filters, facade get/search options, retrieval participant handles and active/source participant deduplication canonicalize IDs before comparison. Production composition supplies entity resolution to episodic and activity repositories.
- Activity filters compare resolved speakers/audiences and return resolved identity fields. Entity-blocked goals resolve both the blocker and indexed sender through `resolved_entities`; no index row is rewritten.
- Runtime and offline repair share `projectMachineReferences`, whose field contracts distinguish handles and serialized machine envelopes from authored scalar fields. The participant codec and semantic handle keys are shared. User/assistant content remains opaque, including ID-only messages and pasted JSON. Machine internal-event objects can project declared identity fields. Typed arrays and non-plain objects remain intact.

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

Open-edge collisions transfer evidence through `SemanticEdgeRepository.mergeOpenEdgeEvidence` and retain the redundant edge as closed. Edges whose endpoints coalesce are closed while retaining their historical endpoints, preventing self-edges. Pair-review arrays retain their two-element shape; only obsolete `duplicate`/`contradiction` pairs are explicitly dismissed. A `new_insight` review remains active when references collapse. Arrays are not generically deduplicated.

Parent collisions are rejected before any deletion, even if deleting the parent would satisfy a later foreign-key check through cascades. Only identical leaf references may coalesce. Every deletion, trigger insertion, changed field, and affected row count is checked and represented in the plan and reversal payloads.

## Plan, persistence and recovery

The JSONL plan contains every table's scanned columns and counts, every changed SQLite row/key/value, every changed Lance row/value, replacement and prior vectors, before/after vector hashes for metadata-only changes, exact schema changes, identity-event inserts, outbox enqueues/drains and mirror upserts, audit rows, protected file hashes, journal paths, and embedding model/dimensions/endpoint. CLI apply waits until the whole plan has been emitted before mutation. Plans can be large because audit payloads contain complete reversal snapshots.

Persistence boundaries are:

1. Preflight every affected Lance row and vector. Create the journal directory and fsync its parent bank directory, then fsync the complete version-2 journal at `entity-graph-repair/pending.json`.
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

Reversal preflights every affected Lance row (including metadata-only vector hashes) before any persistent write. Apply repeats that preflight before creating the journal, so a modification after reversal planning is also rejected without changing SQLite or creating a pending operation. Reversal refuses intervening modifications to affected rows. It restores prior SQLite data, deleted leaf rows, original Lance metadata/vectors and prior outbox state, then marks the original audit records reversed. It retains the additive schema migration, monotonic SQLite sequence counters, and repair/reversal audit history. The generic online `audit.revert` dispatcher does not implement this cross-store operation.

Missing goal events are reconstructed only from adjacent stored snapshots. `IdentityEventRepository.record` writes manual `repair_missing_update` events without changing goal content or versions. Reasons identify the repair time and historical ordering timestamp as a placeholder, since the original author and exact write time are unknown. Unreconstructable gaps are reported.

Scheduled actions are reported unchanged except for structural references. The existing archive policy excludes scheduled actions; their stamps record state entry, not a deadline. Neither a stale deployment claim nor an old stamp proves completion or cancellation. Lifecycle changes require outcome evidence or an explicit policy decision.

## Reused utilities and additions

Searches covered entity/semantic repositories, consolidator and review handlers, existing `scripts/repair-*` code, identity/audit repositories, key builders, embedding setup, and atomic-file helpers.

Reused directly: `EntityRepository` name/alias/self resolution; `runMigrations`; `SqliteDatabase`/`SqliteRawDatabase`; repository row decoders and schemas; `buildObservedEventRecurrenceKey`; `buildOpenQuestionDedupeKey`; participant-term codecs; `GoalsRepository.get`; `IdentityEventRepository.record`; `AuditLog.record`; `SemanticEdgeRepository.mergeOpenEdgeEvidence`; `SemanticNodeRepository.enqueueVectorSyncRow` and `syncPendingVectorUpdates`; `buildNodeEmbeddingText`; configured embedding client; `LanceDbTable`; branded IDs; and atomic fsync/rename helpers.

No existing utility performed an entity merge through redirects or coordinated a reversible SQLite/Lance repair. The added repository redirect mechanism serves ordinary readers. Script-local inventory/projection, validation adapters, complete-plan formatting, resume journal and audit-based reversal coordinate the offline operation. No dependency was added. The consolidator's episode merge is not an entity-graph merge; its archive/supersession conventions are retained here.

## Second review follow-up

| Finding | Change and evidence |
| --- | --- |
| 1. Existing room ingress | Stored bindings win; mismatches are observable. All nine backup sessions pass direct ensure and HTTP ingress before and after repair. The operator-specified third merge resolves the second mismatch. |
| 2. Identity boundaries | Canonical inputs/outputs at demo, ingestion, turn orchestration, sessions, episode/disclosure APIs, activity, participant dedupe and retrieval handles. Tests use old/survivor IDs interchangeably. |
| 3. Entity blockers | The blocker and immutable index sender resolve to the same identity in SQL. Both directions tested with unchanged index rows. |
| 4. Projection contracts | One shared machine-field projection replaces both recursive walkers. Tests cover object keys, encoded participants, serialized machine JSON, tuple preservation and authored text in both directions. |
| 5. Reversal preflight | All affected Lance rows checked before writes; metadata-only vectors have hashes in plans/audits. Metadata and vector-only stale plans reject without SQLite/journal mutations, including an edit after planning. |
| 6. Journal reachability | Parent bank fsync precedes journaling and SQLite. Injected parent-fsync failure leaves SQLite and journal contents untouched. |
| 7. Review semantics | Only semantic-pair handlers retire collapsed pairs; `new_insight` stays active and validates. |

The shared field-contract projection is new because searches found two divergent walkers and existing participant codecs, but no schema-aware projector. The Lance vector hash/preflight helpers replace the applier's metadata-only comparison and serve planning, apply, reversal and resume. Session preflight is read-only and belongs to the offline planner. The existing autobiographical row decoder is now exported/reused: the third merge touches one period that the earlier merge set did not reach. No dependency was added.

## Fresh backup rehearsal for this follow-up

Two fresh copies were created only with `cp -r /home/luth/borg-demo-backups/demo-20260917-224047 <destination>`. Each dry run scanned 67 SQLite and seven Lance tables and left all 7,290 original files byte-identical. Plans used the three explicit entity merges and five explicit node merges from the operator's command, with the configured real embedding client. Live data was never opened.

The preflight checked nine sessions and found exactly two mismatches:

| Session | Bound entity | Label entity | Bound primary/origin episodes | Label primary/origin/session episodes |
| --- | --- | --- | ---: | ---: |
| `sess_82frs56mddjq0nid` | `ent_g9a1j5imbhrgbo1q` | `ent_k9p3xw428ufe4sk7` | 0 / 0 | 822 / 830 / 822 |
| `sess_1i51d9mjoclmehci` | `ent_9ixum375kymki7tu` | `ent_n2bpaqyqcypdzqtv` | 0 / 0 | 151 / 155 / 151 |

These are measured SQLite episode-index counts in this backup; primary and origin counts are distinct. The complete machine plan contains the labels, full entity records/provenance, and every session, including matching sessions. Both mismatches project to the established binding with the supplied merges.

Per-table base changes (additional writes are below):

| SQLite table | Updates | Inserts | Deletes |
| --- | ---: | ---: | ---: |
| action_records | 2,309 | 0 | 0 |
| activity_events | 1,351 | 0 | 0 |
| autobiographical_periods | 1 | 0 | 0 |
| commitments | 37 | 0 | 0 |
| entities | 6 | 0 | 0 |
| episode_index | 984 | 0 | 0 |
| episode_participants | 1 | 0 | 1 |
| goals | 50 | 0 | 0 |
| growth_markers | 78 | 0 | 0 |
| identity_events | 890 | 0 | 0 |
| lived_experience_day_summaries | 63 | 0 | 0 |
| maintenance_audit | 224 | 0 | 0 |
| observed_events | 7 | 0 | 0 |
| open_questions | 186 | 0 | 0 |
| procedural_evidence | 4 | 0 | 0 |
| review_queue | 273 | 0 | 0 |
| semantic_belief_dependencies | 0 | 0 | 16 |
| semantic_edge_invalidation_events | 0 | 17 | 0 |
| semantic_edges | 97 | 0 | 0 |
| semantic_nodes | 16 | 0 | 0 |
| shared_state_artifacts | 2 | 0 | 0 |
| shared_state_entries | 89 | 0 | 0 |
| social_events | 536 | 0 | 0 |
| social_profiles | 2 | 0 | 0 |
| Other 43 tables, including stream_entry_index | 0 | 0 | 0 |

Base total: **7,240 SQLite operations**. Additional writes are five identity-event backfills, 16 outbox enqueues followed by 16 drains, 33 forward audit rows (including completion), and one normal redirect migration. Entity source rows remain as three redirects; five redundant nodes remain archived.

| Lance table | Changed rows |
| --- | ---: |
| action_records | 2,309 |
| episodes | 984 |
| observed_events | 7 |
| open_questions | 2 |
| semantic_nodes | 16 |
| image_perception_embeddings | 0 |
| skills | 0 |

Lance total: **3,318 rows**, including 16 re-embeddings and 3,302 metadata-only changes with vector hashes. Embeddings use `text-embedding-qwen3-8b-text-embedding`, 4,096 dimensions, `http://localhost:1234/v1`. Stream files, working files and stream_entry_index each have **zero changes**.

Each copy verified 7,294 planned SQLite rows, 3,318 Lance rows, all 12,916 untouched vectors and all 16 replacement vectors. Independent scans of the resumed copy covered all 218,105 post-apply SQLite rows and 12,932 Lance rows, finding zero unexpected source handles. The retained handles are three redirect definitions, five archived node definitions, and the new audit reversal payloads. Foreign-key checks returned zero violations and quick_check returned `ok`.

HTTP ingress used the real `/api/turn` route against an in-memory SQLite clone of each store state and captured enqueue inputs instead of appending messages. Before and after apply: 27 explicit `sessions.ensure` calls, 27 successful HTTP requests, and nine connector-facing session reads. Each phase also executes 27 route-internal ensure calls and one default-session bootstrap ensure. Before repair the two mismatches emit 18 observations without rejected turns; after repair there are zero mismatches. Unit tests separately exercise real enqueue persistence with redirected inputs.

The normal zero-change rerun checked all 7,374 files; the interrupted copy checked 7,377 files. Both formatted plans have null run/journal and zero writes, and applying them leaves every physical file hash unchanged. The extra three Lance history files reflect the repeated outbox mirror commit.

The interruption copy exited with code 86 at 13 boundaries: after journaling, after SQLite, after each of five Lance commits, inside the first outbox item after Lance commit/before acknowledgement, after its acknowledgement, before/after completion, and before/after archival. All resumes succeeded, including the final no-pending-journal resume. Every checkpoint rechecked all 18 protected file hashes. Outbox counts progressed 16 → 15 → 0; forward audits progressed 32 → 33 exactly once.

The normal copy's stale-vector rehearsal changed an affected metadata-only row's embedding. Both reversal planning and applying a previously prepared reversal rejected it: **zero SQLite mutations, zero journal mutations**. The vector was then restored before the successful audit-based reversal.

Audit-based reversal restores all 218,066 original SQLite rows and all 12,932 Lance rows/vectors. It performs 7,245 base SQL reversals and 3,318 Lance restorations. The remaining additive state is one migration plus 64 forward/reversal audit rows (33 + 31), for 218,131 total SQLite rows; monotonic sequence counters and Lance/journal history remain by design.

## Hashes, validation and retained paths

SHA-256 below hashes deterministic JSON manifests of row hashes. Entity baselines normalize the newly nullable `merged_into`; original-row comparisons exclude the additive migration and new repair/reversal audit rows. Each protected file's entry hashes its raw bytes.

| Manifest | Before / restored SHA-256 |
| --- | --- |
| All original SQLite rows | `19fd4b1ac5048d5455f294f868c227aaea49344768b6949d92f6a441e511c796` |
| Affected SQLite tables | `56762cb96682efcec473106e73050ba68cb47ab9e8704c7c3c9a8da5dddabef9` |
| All Lance rows and vectors | `42b57fe461434d74590b05900eb33ed1a85055faebf2fec0b180cf30f606e7c7` |
| Protected stream/working files | `c74f9cbb96a2ceebcd6f6df14d70aa470da44b8b6b0775e5e4ebe0d0a7222b98` |
| Stream entry index | `6313bd6a38ef9e70c59b11b77edfe3629011154d30077877789c8b8d47482434` |

The protected manifest contains nine stream and nine working files. The stream index contains 44,034 rows. Complete per-file hashes, all 67 tables' before/restored row counts and checksums, plans with per-row changes, and interruption counters are retained in the artifact directories below.

Focused validation: **357 tests passed in 19 files** using the required four `env -u` settings, `--maxWorkers=2 --cache=false --configLoader=native`. Following the nullable-ID type correction, the two affected focused files passed again: **37 tests**. The final machine-key codec checks and repair tests also passed: **30 tests**. `pnpm typecheck` was invoked once; it initially found that nullable-ID mismatch. After correction, all four remaining compiler projects (which include production sources) and the strict repair-script/demo-import compilation passed. The script check also caught and corrected a type-only import path. No full test suite was run.

Created paths for this follow-up:

- `/home/luth/tmp/repair-review2-d2aaf222/` — fresh normal rehearsal copy, retained after reversal.
- `/home/luth/tmp/repair-review2-resume-d2aaf222/` — fresh interruption/resume copy, retained after reversal.
- `/home/luth/tmp/entity-graph-review2-d2aaf222/` — helpers, logs, compiler configs, `normal/` and `interrupted/` artifacts, and `test-tmp/` for test fixtures/cache. A recursive created-path inventory is retained here; test fixtures clean up within this root.
- The new tracked worktree files are `src/memory/common/reference-projection.ts` and `src/memory/common/reference-projection.test.ts`.

No path under `/tmp` was written, no disposable copy was deleted, and no live-store operation occurred. The exact live invocation is provided in the final report, with the additional operator-specified merge; production/test code contains no live names.

Open data decisions: 256 scheduled actions still need outcome evidence or an explicit lifecycle policy, including the stale deployment-claim action. Five missing goal events are backfilled from adjacent snapshots; the original actor/time cannot be recovered and is not invented. There are no unreconstructable goal version gaps in this backup. Existing label/binding mismatches in any other store remain explicit operator decisions; the resolver does not infer merges.
