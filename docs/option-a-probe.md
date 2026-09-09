# Option A: native Borg inbox probe on P4

This probe runs the existing native `TurnOrchestrator` for Teams inbox messages in
one sidecar tenant: perception, evidence ledger/shared state, planner (when the
native path selects System 2), finalizer, post-generation guard, persistence and
reflection. It does not force a planner path or an answer. Retrieval scoring,
recall planning, recent-activity selection, remember and consent semantics are
unchanged. Opus 5.0: in-scope because runner selection, provider compatibility and
delivery are structural integration concerns.

`BORG_OPTION_A_PROBE=1` (also accepts `true`) enables the probe.
`BORG_OPTION_A_PROBE_TENANT` selects the tenant and defaults to `team-agent-ai`.
Unset, `0`, or `false` retains the existing HTTP `TeamAgentTurnRunner` with the
same arguments and wire payloads. Other tenants keep that runner even while the
probe is enabled. Effective model logging is available in both modes; the expanded
slot defaults and gateway settings apply only with the probe enabled.

For the selected tenant, the sidecar omits the HTTP runner override and supplies
`inbox.native: { tools: "none", agentDeliveries: true }` to `Borg.open()`. The
native orchestrator receives an empty executable-tool dispatcher, including no
read, outbound, task, or autonomous tools. Borg's structured classification,
planning and `EmitAnswer`/`EmitObserve`/`EmitNoOutput`/`EmitSelfReport` function
schemas remain the internal generation protocol; these are necessary to run the
native pipeline. The separate existing task-result lane remains installed.

Native inbox delivery waits until `TurnOrchestrator.run()` returns successfully,
including its rollback-capable post-response lifecycle. The runner then durably
appends a `native_inbox_turn_committed` internal event naming the active terminal.
Only then is an `agent_msg` inserted idempotently into the existing tenant-local `agent_deliveries` table,
then `onDeliveryAvailable` wakes its existing claim waiter. The required opaque
`task_id` field is `native:<terminal_entry_id>`; no task is created. The regular
`await-response` waiter also sees the same terminal ID. Observed/suppressed turns
resolve `await-response` without a text delivery. Startup reconciliation repairs
missing delivery rows from active native stamped terminals with that commit marker after index backfill,
even when the response watermark already advanced. A session-scoped reconciliation
before each inbox drain also retries failed queue writes without regenerating.
HTTP-runner terminals, which lack `turn_id`, and task-event terminals are not
projected by this path.

An append alone never authorizes delivery: a later native failure can mark it
inactive. The runner does not publish from append observers or failure cleanup.
Reconciliation also leaves unmarked turns unpublished, including turns interrupted
before the marker and historical probe turns written before this boundary existed.
Such an interruption needs operator review or a restored bank copy; the script
does not infer successful completion from an active append or reset watermarks.
The existing `await-response` lookup remains append-visible and can return before
the lifecycle finishes; `agent-deliveries` is the probe's reply-commit boundary.

This is a sidecar probe, not a deployed bridge change. If a bridge consumes both
`await-response` and `agent-deliveries`, it must treat the shared terminal ID as
one reply. Confirm that consumer deduplication and acceptance of an opaque
non-UUID `task_id` before attaching both consumers to a live thread. This checkout
contains neither the team-agent nor bridge consumer implementation.

With the probe enabled, all eight model slots fall back to `LLM_MODEL` in the sidecar, including the
previously missing image-perception env override. Explicit nonblank env values
win over that fallback and tenant/root config files. Startup prints the effective
eight-slot map once, before tenants open. Both planner and finalizer use
`cognition`; they do not have separate Borg model slots.

With the flag off, HEAD's original seven-slot `??=` assignments are preserved:
only unset values receive `LLM_MODEL`; blank overrides fall through to the tenant's
config or built-in defaults. Image perception uses its configured/Haiku default,
and `BORG_MODEL_IMAGE_PERCEPTION` is ignored as in HEAD.

| Slot                 | Explicit override                  |
| -------------------- | ---------------------------------- |
| cognition            | `BORG_MODEL_COGNITION`             |
| background           | `BORG_MODEL_BACKGROUND`            |
| extraction           | `BORG_MODEL_EXTRACTION`            |
| recallExpansion      | `BORG_MODEL_RECALL_EXPANSION`      |
| correctivePreference | `BORG_MODEL_CORRECTIVE_PREFERENCE` |
| sharedStateCompiler  | `BORG_MODEL_SHARED_STATE_COMPILER` |
| creatorDirective     | `BORG_MODEL_CREATOR_DIRECTIVE`     |
| imagePerception      | `BORG_MODEL_IMAGE_PERCEPTION`      |

Run the probe on a quiescent copy of the populated tenant bank with its existing
embedding profile and cross-venue history. A fresh empty bank cannot measure the
memory question. Use a consistent backup taken while the source sidecar is
stopped; never open the same bank with two Borg processes. Keep the bridge
disconnected from the copy: replay really enqueues a question, changes native
turn state and consumes a delivery. Repeated runs on the same bank copy accumulate
turn state, including earlier probe questions and answers. The caller decides
whether to restore the copy between runs; the script never restores or resets
it. For independent samples, stop the sidecar, restore the snapshot and restart
between separate `--runs 1` invocations. `--runs N` measures sequential turns on
the same evolving bank. Record that choice with the results, and use the same
starting snapshot for model comparisons.
Use a snapshot with a drained inbox: enabling the worker also schedules the
selected tenant's existing pending Teams messages.

1. Start this worktree's sidecar with its normal credentials, embedding settings,
   CA trust and inbox configuration, plus the probe settings below. The existing
   `TEAM_AGENT_BASE_URL` and `TEAM_AGENT_API_TOKEN` remain required because the
   sidecar's inbox admission and task-result lane use that configuration.

   ```bash
   export BORG_DATA_ROOT=/srv/borg-option-a-copy
   export BORG_OPTION_A_PROBE=1
   export BORG_OPTION_A_PROBE_TENANT=team-agent-ai
   export KRATOS_BASE_URL=https://inference.kratos.omc.hdp.it.p4/v1
   export LLM_MODEL='<exact P4 model id>'
   export BORG_MEMORY_LLM_REASONING_EFFORT=none
   export BORG_MEMORY_LLM_MAX_TOKENS=16384
   export BORG_MEMORY_LLM_TIMEOUT_MS=180000
   export BORG_DELIBERATION_FINALIZER_TRANSPORT=unary
   export BORG_DELIBERATION_FINALIZER_CONTEXT_CAPTURE_SAMPLE_RATE=1
   export BORG_MEMORY_TRACE_CAP=2000
   pnpm start
   ```

   Retain `BORG_MEMORY_TOKEN`, `LLM_API_KEY`, `NODE_EXTRA_CA_CERTS`, the bank's
   `EMBEDDING_MODEL`/`EMBEDDING_DIMS` and normal Team Agent settings from the
   deployment. Do not paste credentials into reports. The probe defaults to
   reasoning effort `none`, a 16384 gateway output ceiling, and a 180000 ms SDK
   per-attempt timeout. Explicit env values override them. With the flag off,
   `BORG_MEMORY_LLM_MAX_TOKENS` and `BORG_MEMORY_LLM_REASONING_EFFORT` are ignored,
   retaining HEAD's model-family caps and omission of `reasoning_effort` even if
   those env variables remain set. Timeout parsing is exactly
   `Number(BORG_MEMORY_LLM_TIMEOUT_MS ?? 120000)`, preserving zero, blank and
   fractional values and the SDK's own validation. The SDK still has one retry:
   a failing call can consume roughly two per-attempt timeouts plus retry delay.

   Probe mode enables the existing token-protected `/memory/trace` buffer for
   the selected tenant with full native events and payloads, excluding token
   chunks. Its default capacity is 2000; other tenants keep the original filter.
   Exact finalizer capture sampling defaults to 1 when the probe is enabled;
   explicit capture settings are respected. Check the startup model map and
   gateway settings. Sample rate 0 prevents a complete probe report.

2. Run the replay on the sidecar host, or where the tenant directory is mounted
   read-only, using the existing raw team-agent thread ID and Tomasz's real
   transport sender ID. Those two IDs are not supplied by the scenario; a display
   name or conversation external ID is not a substitute for either. Set
   `--operator` only if the original transport marked Tomasz as an operator.

   ```bash
   umask 077
   pnpm exec tsx scripts/option-a-probe.ts \
     --base-url http://127.0.0.1:8088 \
     --session '<existing raw team-agent thread id>' \
     --sender-external-id '<Tomasz transport sender id>' \
     --data-dir /srv/borg-option-a-copy/team-agent-ai \
     > option-a-g.json
   ```

   `BORG_MEMORY_TOKEN` authenticates every request. The script verifies the
   server's probe flag for the tenant before mutating anything, opens an actual
   `agent-deliveries/claim` long poll, and defaults to scenario G:

   - Tenant: `team-agent-ai` (override with `--tenant` for a renamed copy).
   - Sender: Tomasz Kasperczyk.
   - Conversation: `groupChat`, AI Ninjas,
     `19:d93832bc19034403b15732c6441d3391@thread.v2`.
   - Mentioned question: `Czy Marcin będzie w następnym tygodniu w pracy?`

   Configure another scenario with these flags. Changing the conversation type
   does not rewrite the other defaults: supply the matching name, conversation
   external ID, raw thread/session ID and sender identity from the bank copy.

   | Flag                            | Default / meaning                                                                       |
   | ------------------------------- | --------------------------------------------------------------------------------------- |
   | `--question TEXT`               | G's Polish question above                                                               |
   | `--sender-display-name NAME`    | `Tomasz Kasperczyk`                                                                     |
   | `--sender-external-id ID`       | Required transport sender ID; existing flag                                             |
   | `--conversation-type TYPE`      | `groupChat`; accepts `groupChat`, `personal`, `channel`                                 |
   | `--conversation-external-id ID` | G's `19:…@thread.v2` ID above                                                           |
   | `--conversation-name NAME`      | `AI Ninjas`                                                                             |
   | `--mentioned true\|false`       | `true`; explicit value required, also accepts `--mentioned=false`                       |
   | `--session ID`                  | Required existing raw team-agent thread ID; existing flag                               |
   | `--runs N`                      | `1`; positive integer, same scenario repeated sequentially                              |
   | `--expect-present TEXT`         | Repeatable literal substring assertion on each committed reply                          |
   | `--expect-absent TEXT`          | Repeatable literal substring assertion on each committed reply                          |
   | `--compact`                     | Omit large ledger/system-prompt payloads while retaining summaries and capture metadata |

   For example, a personal-chat control using real identities from the copy:

   ```bash
   pnpm exec tsx scripts/option-a-probe.ts \
     --base-url http://127.0.0.1:8088 \
     --session '<existing personal-chat thread id>' \
     --sender-external-id '<transport sender id>' \
     --sender-display-name '<sender display name>' \
     --conversation-type personal \
     --conversation-external-id '<personal conversation external id>' \
     --conversation-name '<personal conversation name>' \
     --question '<control question>' \
     --mentioned false \
     --runs 20 \
     --expect-absent '<private detail that must not appear>' \
     --expect-absent '<another private detail>' \
     --expect-present '<expected answer fragment>' \
     --compact \
     --data-dir /srv/borg-option-a-copy/team-agent-ai \
     > option-a-personal.json 2> option-a-personal.log
   ```

   Expectation checks are case-sensitive literal substring checks, with no
   normalization or interpretation. Each prints `PASS` or `FAIL` on stderr after
   each run and is recorded in that run's `expectations` array. Empty substrings
   are rejected. A turn without a committed reply fails both kinds of assertion
   with a `no committed reply` reason, so suppression cannot silently pass a
   no-consent control. These diagnostic failures do not change the exit code.

   `--observed-at` accepts an ISO timestamp; default is the current time. It
   stamps the source observation, not Borg's clock. Native relative-time
   interpretation still uses the sidecar clock. A historical reproduction needs
   a matching evaluation clock/data snapshot; changing only `observed_at` does
   not recreate the original meaning of “next week.” `--external-message-id`
   defaults to a fresh UUID-based probe ID for every run. With `--runs N` greater
   than 1, an explicit `--external-message-id` becomes a prefix with a fresh UUID
   appended each time. With one run, an explicit ID is sent unchanged; reusing
   that ID exercises transport deduplication and may return an already-consumed
   terminal. Each run records its actual external message ID. An explicit
   `--observed-at` is reused across runs; otherwise each run gets its own current
   timestamp. `--timeout-ms` defaults to 900000 per multi-stage run, not for the
   entire batch. Each run finishes trace/capture collection, acknowledges its
   delivery and closes its polls before the next starts. A capture/trace or
   transport failure stops subsequent runs and retains the partial report.

3. Inspect the JSON report on stdout. Per-run receipt latencies, each committed
   reply, expectation results and aggregate latency results are also printed to
   stderr; redirect it separately when retaining a JSON report. A single run keeps
   the original top-level fields, with additional scenario metadata, summaries and
   expectations. Multiple runs put those per-run records in `runs`. Both shapes
   include `requested_runs`, `completed_runs` and `aggregate_latency`.

   `finalizers` contains every matching initial or
   regeneration capture, including its `evidence_ledger`, model, surface variant,
   full fidelity metadata, capture ID/timestamp/schema version, outcome and exact
   `live_request.system`. The canonical `parseFinalizerContextCaptureRecord()`
   parser validates records; the shared read-only snapshot helper also serves the
   planner replay tools. The system field is
   the actual rendered surface; a compact finalizer may see a projection of the
   fuller ledger object. Capture files come from
   `<tenant>/captures/finalizer-contexts.jsonl`, not from a second recall request.
   The script reads stream/capture files without opening Borg or SQLite.

   A terminal trace with `suppressed_generation_gate` or `suppressed_closure`
   confirms an early exit before finalization. When no finalizer ran, the report
   records `finalizer_ran: false`, `capture_status: not_applicable_early_suppression`,
   the suppression outcome/content in `early_suppression`, and empty `finalizers`
   and `ledger_summary` arrays. The batch continues. Observed/suppressed turns
   after finalizer execution still require complete, verified captures. Trace
   truncation and missing captures when finalization ran remain errors.

   Each run's `ledger_summary` has one item per finalizer capture, labelled with
   its one-based `finalizer_index` and `attempt_kind`. Each item preserves section
   order and includes every section's `id`, `row_count` and ordered `rows`. Each
   row contains the ledger `id`, `is_episode` (from its explicit `source_type`),
   and `episode_id` (the bare `ep_…` ID for episode rows, otherwise `null`). Empty
   sections have count 0 and an empty row array. Regenerations are kept separate
   rather than double-counted into one ledger. These are the captured ledger's
   rows; the exact rendered system remains the authority for a compact finalizer
   projection. `--compact` removes only `finalizers[].evidence_ledger` and
   `finalizers[].live_request.system` from the report, retaining the summary,
   model, tool names, surface variant and fidelity checks. The complete capture
   stays in the sidecar's capture file for detailed inspection.

   `committed_reply` and `delivery.content` must agree exactly and refer to the
   same terminal as `await-response`. `stage_timings` records native phase
   durations; `llm_timings` pairs call events using the process monotonic clock.
   Phase and LLM timings overlap and must not be summed. Client receipt times
   measure time since that run starts, including client preflight/enqueue overhead
   and inbox wait. `terminal_received_after_ms` measures the terminal HTTP receipt;
   `delivery_received_after_ms` measures the matching claim response receipt.
   Delivery now includes successful post-response completion and its durable marker;
   remeasure the delivery latency gate after this lifecycle fix. The script also
   collects the `turn.terminal` trace; additional capture/trace collection time is
   not added to receipt latency. Both background polls use a cleanup cancellation
   signal. Intentional cleanup aborts are ignored; transport failures, including
   failures arriving during cleanup, fail the command and retain the report.

   `aggregate_latency` summarizes each receipt metric separately with `count`,
   `missing_count`, `median_ms`, `p95_ms` and `passes_gate`. Median averages the
   middle two samples for an even count. P95 uses nearest rank: sorted sample
   `ceil(0.95 * count)`, counting from 1, without interpolation. Thus a small
   sample's p95 is usually its maximum; choose enough runs for the gate. The
   inclusive latency thresholds are median <= 20000 ms and p95 <= 30000 ms.
   Missing receipt values are excluded from the numeric statistics and counted
   explicitly; the metric cannot pass unless all requested runs have samples.
   No samples produces `null` median/p95. Latency gate results are diagnostic and
   do not change the exit code or replace capture/trace validity checks.

   `delivery_woke_waiter` is true only when the actual delivery-waiter callback
   reports `available` for this session during the run. It is not inferred from
   a fast HTTP response. False with a delivery can mean it was already queued
   before a waiter registered. Use a quiescent copy with no competing claimers to
   attribute this wake to the replay's waiter. This measures the sidecar long
   poll used by the bridge, not a real Teams post or a bridge process wake.

   The replay consumer prints the answer and acknowledges only its matching
   delivery as `failed_permanent` with an explicit “probe consumed; no Teams
   send” reason. It never claims a successful Teams send. Other claimed delivery
   IDs are reported and left unacknowledged to expire. Do not run the consumer
   against the live bridge's queue. Capture fidelity failures, missing/rotated
   captures, trace overflow and timeouts produce a nonzero exit and a diagnostic
   report. Suppression and degradation remain visible; reply assertions come
   only from the caller's explicit `--expect-present` / `--expect-absent` values.

4. Stop the probe sidecar after the worker drains. Unset `BORG_OPTION_A_PROBE`
   (or set it to `0`) and the probe-specific transport/capture overrides before
   restarting the original HTTP-runner deployment. No schema migration was
   added. Retain the private report and immutable input snapshot for comparison.
   An index-update failure requires normal index reconciliation before another
   turn, as described in `AGENTS.md`.

The gateway audit below is a source audit against the supplied P4 constraints;
it is not a live certification of any model or gateway route. File/line locations
refer to this worktree. The supplied constraints were not independently measured.

| Constraint / finding                                                                                     | Source                                                                                                      | Resolution / remaining limit                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GLM stalls without explicit reasoning control; adapter previously dropped only Anthropic thinking/effort | `src/llm/openai-compatible.ts:378`, `src/sidecar/llm-config.ts:38`                                          | Added opt-in `reasoningEffort`; probe sends `reasoning_effort: none` on both completion methods. `BORG_MEMORY_LLM_REASONING_EFFORT` is shared across slots; mixed-model acceptance must be verified on P4.                                                             |
| Gateway rejects `propertyNames`                                                                          | `src/llm/index.ts:147`, `src/llm/index.ts:208`, `src/tools/anthropic.ts:9`                                  | Already handled: `toToolInputSchema` strips this keyword recursively (and other unsupported grammar keywords) while Zod still validates results. All native schema construction sites below use this converter. No duplicate sanitizer or memory schema change needed. |
| Output cap 16384; model-family caps can reach 64000                                                      | `src/llm/max-tokens.ts:8`, `src/llm/openai-compatible.ts:373`                                               | Added optional gateway ceiling, default 16384 in probe mode, applied to complete/converse. Existing lower limits remain: unknown model names, including GLM, currently default to 8192.                                                                                |
| Hades cuts unary calls at 60 s; kratos supports at least 150 s                                           | `src/sidecar/gateway-config.ts:1`, `scripts/memory-sidecar-main.ts:145`, `src/llm/openai-compatible.ts:407` | Use kratos plus the probe's 180 s SDK timeout. Client timeout cannot fix Hades/proxy termination; route/ingress verification remains a deployment prerequisite.                                                                                                        |
| Adapter is unary; per-call `timeoutMs` and Anthropic thinking/effort hints are not mapped                | `src/llm/openai-compatible.ts:392`, `src/cognition/turn-action/tool-loop.ts:419`                            | `signal` and adapter/SDK timeout are supported. Streaming requests fall back to converse; use unary for this probe. Per-stage timeout hints are not a substitute for `BORG_MEMORY_LLM_TIMEOUT_MS`. No streaming/deadline redesign in this change.                      |
| Anthropic `output_config` unsupported                                                                    | `src/llm/openai-compatible.ts:361`                                                                          | Explicit error remains. Native structured paths use forced tool calls; no native caller supplies `output_config`.                                                                                                                                                      |
| `image_ref` unsupported                                                                                  | `src/llm/openai-compatible.ts:231`                                                                          | Text-only scenario G is supported. An image-bearing finalizer context still fails; configuring `BORG_MODEL_IMAGE_PERCEPTION` does not add an attachment resolver.                                                                                                      |
| Forced tool calls, union/nullable/ref schemas and P4 model adherence                                     | Native schema inventory below                                                                               | Structural source audit and fake-client tests pass. Live schema acceptance and emission quality still need the target gateway/model; no content judgments or prompt changes were added.                                                                                |

The native structured-output inventory is below. Every entry reaches
`toToolInputSchema`; finalizer and executable tools go through
`toAnthropicToolDefinitions`. The recall tool only adds `temporal_cue` to the
converted schema's `required` array. Schemas in capture/storage/HTTP records are
not sent as generation schemas. The regression audit collects private schemas
loaded with `TurnOrchestrator` as well as dynamic recall and emission schemas.

| Stage                                                                      | Schema construction sites                                                                                                                                                                                                                    |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Perception                                                                 | `src/cognition/perception/entity-extractor.ts:35`, `src/cognition/perception/mode-detector.ts:25`, `src/cognition/perception/temporal-cue.ts:25`; `src/memory/affective/extractor.ts:50`; `src/cognition/procedural/context-extractor.ts:35` |
| Frame, closure, admission                                                  | `src/cognition/frame-anomaly/classifier.ts:56`; `src/cognition/generation/closure-loop.ts:90`, `src/cognition/generation/generation-gate.ts:30`                                                                                              |
| Live preferences/directives/actions/goals                                  | `src/cognition/commitments/corrective-preference-extractor.ts:193`; `src/cognition/creator-directives/extractor.ts:137`; `src/cognition/actions/action-state-extractor.ts:130`; `src/cognition/goals/goal-promotion-extractor.ts:298`        |
| Recall query plan                                                          | `src/retrieval/recall-expansion.ts:409`                                                                                                                                                                                                      |
| Ledger shared-state patches (current and alias names), semantic revision   | `src/cognition/shared-state/schema.ts:40`, `src/cognition/shared-state/semantic-revision.ts:102`                                                                                                                                             |
| System 2 planner (user and autonomous variants)                            | `src/cognition/deliberation/s2-planner.ts:106`                                                                                                                                                                                               |
| Finalizer terminal schemas (including autonomous continue-thought variant) | `src/cognition/deliberation/finalizer.ts:122`, `:136`, `:150`, `:164`, `:179`; converted at `src/tools/anthropic.ts:9`                                                                                                                       |
| Pending actions and post-generation guards                                 | `src/cognition/turn-action/pending-action-judge.ts:27`; `src/cognition/generation/closure-pressure-guard.ts:52`; `src/memory/commitments/checker.ts:67`. Internal-ID guard rendering uses text, not a new JSON schema.                       |
| Reflection                                                                 | `src/cognition/reflection/reflector.ts:349`                                                                                                                                                                                                  |
| Ingestion/catch-up derived memory and review                               | `src/memory/episodic/extractor.ts:134`, `src/memory/semantic/extractor.ts:138`, `src/memory/semantic/review-service.ts:24`, `src/memory/self/review-open-question-extractor.ts:48`                                                           |
| Image perception (adapter limitation above)                                | `src/attachments/perception.ts:828`                                                                                                                                                                                                          |
| Executable tools                                                           | `src/tools/anthropic.ts:9` converts registered tool input schemas. The probe's dispatcher has no registrations, so none are advertised or dispatched; no tool-schema bypass is introduced.                                                   |

Validation uses mocked LLMs/embeddings, a real temporary Borg bank and a local
HTTP delivery waiter. It covers flag-off identity, tenant selection, empty tool
dispatchers, commit-time wake, no duplicate generation/delivery, restart recovery,
all slot overrides, gateway request options, native schema conversion and the
replay script's transport/report contract. Live replay still requires the bank
snapshot, actual sender/thread IDs, target model access and the deployment's
route/consumer checks above.

The flag-off parity check loaded config and adapter sources directly from
`git show HEAD`, and executed HEAD's seven-slot startup loop against synthetic
env/config inputs. All 24 complete-config comparisons, seven timeout cases and
18 serialized SDK request body/header comparisons matched (both `complete` and
`converse`, three model IDs, with probe-only gateway env absent, set or invalid).
The requests used a local fake fetch; no provider calls were made. The HTTP
`TeamAgentTurnRunner` source remains identical to HEAD. Regression tests also
exercise the real native `run()` path with a post-response failure: its appended
reply becomes inactive and is never published, including after reopen.

The operator reported successful G, P and sensitive no-consent replays from this worktree using cognition
`generative-apis/glm-5.2`, other slots at their sidecar defaults, and kratos via a
relay against a production-bank copy: the committed reply included Marcin's exact
absence dates from `ep_xij9l0i4k8uvespi`, and the no-consent control disclosed
Jacek's Friday absence while withholding the reason. The delivery woke the waiter.
Relay latency was 26–42 seconds per turn; in-cluster latency has not been measured.
The reports are stored outside this repository. These observations predate the
post-success delivery boundary above and do not establish the new latency gate.
Replay tests additionally cover configurable personal/channel scenarios,
sequential runs with fresh IDs, expectation results, ordered ledger summaries,
compact reports, percentile boundaries, and partial reports on capture/trace
failures.

Changed files in this probe (all left uncommitted):

- Runner wiring: `src/borg/open.ts`, `src/borg/types.ts`,
  `src/borg/native-inbox-runner.ts`,
  `src/cognition/ingestion/chat-response-catch-up-worker.ts`.
- Sidecar startup, flag and models: `scripts/memory-sidecar-main.ts`,
  `src/sidecar/option-a-probe.ts`, `src/sidecar/llm-config.ts`, `src/config/index.ts`.
- Gateway adapter: `src/llm/openai-compatible.ts`.
- Delivery/trace observation: `src/sidecar/delivery-waiter-registry.ts`,
  `src/sidecar/memory-trace.ts`, `src/sidecar/memory-handler.ts`,
  `src/tracing/tracer.ts`.
- Replay and runbook: `scripts/option-a-probe.ts`, `docs/option-a-probe.md`,
  `scripts/capture-snapshot.ts`, `scripts/planner-ab-replay.ts`.
- Tests: `src/borg/native-inbox-runner.test.ts`,
  `src/sidecar/option-a-probe.test.ts`, `src/sidecar/llm-config.test.ts`,
  `src/sidecar/memory-trace.test.ts`, `src/llm/openai-compatible.test.ts`,
  `src/llm/native-gateway-schema.test.ts`, `scripts/option-a-probe.test.ts`,
  `scripts/capture-snapshot.test.ts`.
