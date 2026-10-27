# Observatory event contract (v1)

Shared between `team-agent` (consumer and producer; see `docs/observatory.md`) and `borg`
(producer). Keep both copies identical; bump `v` only for incompatible changes.

## 1. Transport

The consumer is team-agent itself (`team_agent/observatory*.py`). Events cross replicas through
an UNLOGGED Postgres table (`observatory_event`) that every team-agent replica writes to and
polls; the table is pruned to the retention window (default 60 min) and is never backed up or
WAL-logged. That is the only place events are ever stored.

- **team-agent's own events** never leave the process over HTTP: the emitter batches them
  straight into the bus table.
- **borg** (and any other external producer) posts to team-agent:

```
POST {OBSERVATORY_BASE_URL}/ingest/events        # OBSERVATORY_BASE_URL = http://team-agent:8080/v1/observatory
Authorization: Bearer <team-agent API key>         # the same key borg already holds (TEAM_AGENT_API_TOKEN)
Content-Type: application/json

{ "events": [ <Event>, ... ] }                     # 1..500 events, body <= 1 MiB
```

Responses: `202 {"accepted": n, "dropped": m}` on success, `401` bad key, `403` when an event's
`tenant` is outside the key's tenant scope, `413` too large, `400` unparsable body. A producer
never retries on 4xx, retries at most once on 5xx / network error, and drops afterwards.
Producers must not block a request on this call.

## 2. Envelope

```jsonc
{
  "v": 1,
  "id": "evt_9f3c…",                  // unique per event (producer-generated)
  "ts": 1726300000123,                // epoch milliseconds, producer clock
  "source": "team-agent",             // "team-agent" | "borg"
  "instance": "team-agent-7ccccf65d4-b4txs",   // pod / host name
  "tenant": "team-agent-ai",          // borg tenant id == team-agent tenant id
  "kind": "turn.observe",             // dotted kind, catalogue below
  "phase": "start",                   // "start" | "end" | "point"
  "span_id": "sp_1a2b",               // required for start/end; the pair shares it
  "parent_span_id": "sp_0000",        // optional nesting
  "duration_ms": 1234,                // end only
  "status": "ok",                     // end/point: "ok" | "error" | "degraded" | "skipped"
  "turn_id": "turn_7d1e…",            // optional, see correlation
  "refs": {                           // all optional
    "thread_id": "team-agent-ai::shared::groupChat::19:abc@thread.v2",
    "sidecar_session_id": "sess_0123456789abcdef",
    "entry_id": "ent_…",
    "entry_ids": ["ent_…"],
    "external_message_id": "1726300000000",
    "context_id": "ctx_…",
    "run_id": "run_…"
  },
  "conversation": { "external_id": "19:abc@thread.v2", "type": "groupChat", "name": "AI Ninjas" },
  "actor": { "external_id": "4ea458f9-…", "display_name": "Tomasz Kasperczyk" },
  "data": { }                         // kind-specific, clamped (see 5)
}
```

Rules:

- `start` and `end` of one stage share `span_id`; an `end` without a `start` is still
  accepted (the consumer synthesises the start from `ts - duration_ms`).
- `status` on `end` defaults to `ok`; an exception in the stage produces `error` with
  `data.error_type` and `data.error` (message clamped).
- Unknown `kind` values are accepted and shown as generic spans; the consumer only
  needs the catalogue for the stage columns and the detail cards.
- `tenant` is required. Events without a tenant are dropped.

## 3. Correlation

- `turn_id` = one processed turn (one coalesced batch, one attempt). borg's
  `TeamAgentTurnRunner` mints `turn_<id>` per `run()` attempt and sends it to
  team-agent as the request header `X-Observatory-Turn-Id` on `POST /v1/chat/observe`.
  team-agent stores it for the request (ContextVar), stamps it on every event, and
  forwards the same header on every sidecar call so borg can stamp `/memory/context`,
  `/memory/guard-reply`, `/memory/append-turn` events with it.
- team-agent mints `turn_<id>` itself on `POST /v1/chat/completions` (no borg runner
  in front) and when the header is absent.
- Retries of the same batch produce new `turn_id`s with the same `refs.entry_ids`;
  the consumer groups them as attempts of one batch.
- Message-level events emitted before a turn exists (`teams.enqueue`, `teams.await`,
  `inbox.enqueued`, `inbox.settle`, `inbox.waiter`) carry `refs.entry_id` and
  `refs.sidecar_session_id` and no `turn_id`. The consumer joins them to the turn via
  `refs.entry_ids` on `inbox.batch` / `turn.observe`.

## 4. Kind catalogue

Stage column mapping used by the flow board is in the last column.

### 4.1 team-agent

| kind | phase | data | column |
|---|---|---|---|
| `teams.enqueue` | start/end | `text_preview`, `mentioned`, `quotes_bot`, `seeded_own_post`; end: `entry_id`, `sidecar_session_id` | Intake |
| `teams.await` | start/end | `wait_ms`, `seen_generating`; end: `outcome` (`answered`/`observed`/`generating`/`pending`) | Intake |
| `turn.observe` | start/end | `source`, `message_count`, `full_turn`, `replayed`; end: `action` (`reply`/`silent`), `reason`, `content_preview` | (whole turn) |
| `turn.receipt_lock` | start/end | - | Coalesce |
| `presence.classify` | start/end | `policy`, `venue`, `message_count`, `model`; end: `speak`, `reason`, `note` | Classify |
| `turn.prepare` | start/end | - (parent of the four below) | Recall |
| `turn.prepare.assets` | start/end | `mode`; end: `error` | Recall |
| `turn.prepare.binding_context` | start/end | `sections`; end: `degraded`, `degraded_reason`, `context_id` | Recall |
| `turn.prepare.knowledge_index` | start/end | end: `chars` | Recall |
| `turn.prepare.graph` | start/end | `variant`, `cached` | Recall |
| `turn.thread_lock` | start/end | - | Reason |
| `turn.invoke` | start/end | end: `reply_preview`, `model_calls`, `tokens` `{input,output,total,max_prompt}`, `tools_used`, `guard_blocked` | Reason |
| `llm.call` | start/end | `node`, `model`, `message_count`; end: `input_tokens`, `output_tokens`, `tool_calls` (names), `finish_reason` | Reason |
| `tool.call` | start/end | `name`, `args_preview`; end: `result_preview` | Tools |
| `memory.recall` | end | `focus`, `context_turns`, `cache_hit`, `include_episodes`, `limit`, `episodes` `[{id, score, excerpt}]`, `recalled_chars`, `degraded`, `degraded_reason`, `partial`, `unavailable_reason`, `context_id` | Recall |
| `memory.call` | end | `path`, `http_status`, `breaker_state`, `degraded` | Recall / Remember |
| `guard.reply` | end | `verdict`, `reason`, `skipped` | Guard |
| `reply.stored` | point | `action`, `content_preview` | Reply |
| `memory.append_turn` | end | `retry` | Remember |
| `turn.error` | point | `error_type`, `error`, `stage` | - |
| `debug.<type>` | point | passthrough of `DebugTrace` payloads (`recall_query`, `recall_candidates`, `recall_injected`, `recall_degraded`, `recall_partially_degraded`, `recall_unavailable`, `recall_expansion`, `binding_context_degraded`, `assistant_assets_context_injected`, `playbook_route`, `playbook_prompt`, `group_presence_decision`, `memory_guard_completed`, `memory_guard_skipped`, `tool_progress`, `tool_result`, `tools_available`, `final`) | by type |

### 4.2 borg

| kind | phase | data | column |
|---|---|---|---|
| `inbox.enqueued` | point | `external_message_id`, `mentioned`, `quotes_bot`, `sender_bot`, `text_preview` | Intake |
| `inbox.settle` | point | `reason` (`append`/`startup`/`in_flight_settled`/`has_more`/`retry`), `delay_ms`, `bound` (`quiet`/`max`/`immediate`/`backoff`), `oldest_pending_at`, `quiet_window_ms`, `max_wait_ms` | Coalesce |
| `inbox.batch` | start/end | `included_count`, `has_more`, `watermark`; end: `result` (`drained`/`busy`/`error`/`empty`), `error_name` | Coalesce |
| `inbox.runner.call` | start/end | `runs_full_turn`, `message_count`, `attempt`; end: `http_status`, `action`, `terminal_kind`, `sealed_reason` | Coalesce |
| `inbox.terminal` | point | `kind` (`agent_msg`/`agent_observed`), `reason`, `sealed`, `content_preview` | Reply |
| `inbox.waiter` | point | `action` (`registered`/`generating`/`resolved`/`timeout`/`cancelled`) | Intake |
| `recall.request` | start/end | `route`, `sections`, `focus_present`, `deadline_ms`; end: `degraded`, `degraded_reason`, `episode_count`, `abstained`, `hidden_episode_count` | Recall |
| `recall.phase` | start/end | `phase` (`identity`/`context_sections`/`episodes`/`autobiographical`), `budget_ms` | Recall |
| `recall.lane` | end | `intent_id`, `intent_kind`, `intent_source`, `candidate_count`, `outcome` (`fulfilled`/`rejected`), `error_name` | Recall |
| `guard.reply` | end | `verdict`, `reason` | Guard |
| `job.maintenance` | start/end | `mode`, `run_id`, `processes`; end: `state`, `total_duration_ms`, `errors_total`, `process_reports` `[{name,status,duration_ms,changes,errors,tokens_used}]` | Borg strip |
| `job.ingestion` | start/end | end: `processed_entries`, `ran` | Borg strip |
| `trace.<event>` | point | passthrough of `TurnTracer` events with their fields (`recall_expansion.completed`, `retrieval.started`, `retrieval.intent_candidates`, `retrieval.degraded`, `retrieval.completed`, `llm_call.*`, `offline_process.started`, `offline_process.completed`, `task_event.*`, `sidecar.guard_reply.completed`, `episodic_extractor.skipped`, …); `data.trace_turn_id` keeps borg's own turn id | by event |

Payload-gated tracer fields (query text, prompts, focus, context turns) are forwarded
only when the tracer runs with `includePayloads` (the sidecar's `/memory/trace`
setting); otherwise they are absent, never blanked.

`trace.recall_expansion.completed` (the query planner) carries `semantic_variant_count`,
`named_term_count`, `typed_query_count`, `temporal_cue_present` always, and `resolved_query`,
`semantic_variants`, `named_terms`, `typed_queries`, `focus`, `context_turns` only with payloads.
Consumers must not invent other field names for it.

## 5. Clamps (producer side)

- any string field above 4 096 chars → truncated to 4 096 + `…[+N chars]`;
- `episodes`, `tool_calls`, `process_reports`, and any other list above 50 items →
  first 50 + `data.truncated_lists: ["episodes"]`;
- serialized `data` above 24 KiB → replaced by `{ "truncated": true, "keys": [...],
  "bytes": N }`;
- `*_preview` fields are 280 chars.

The consumer applies the same clamps defensively on ingest.

## 6. Consumer WebSocket protocol

`GET /v1/observatory/ws` (starlette WebSocket on team-agent; the UI is served from
`/observatory/`). The server owns the reducer: clients never see the raw stream except for
turns they explicitly follow. Client speaks first:

```jsonc
{ "type": "hello", "token": "<viewer token>", "tenant": "team-agent-ai" }
```

Server answers `{ "type": "unauthorized" }` and closes with code 4401, or:

```jsonc
{ "type": "snapshot", "server_ts": 1726300000123, "retention_ms": 3600000,
  "tenants": [ { "id": "team-agent-ai", "turns": 42, "pending": 1, "jobs": 2, "waiters": 1, "last_ts": … } ],
  "turns": [ <TurnSummary> ], "pending": [ <PendingMessage> ], "jobs": [ <JobSummary> ],
  "truncated": false }
```

Bounds: `turns` holds at most the 400 most recently updated summaries (`truncated: true` when
the tenant holds more; the timeline then shows the rest as counts only), `pending` at most 200,
`jobs` at most 50. The server enforces these before materialising the frame, so a snapshot is
always deliverable under the 4 MiB send limit. `tenants[].waiters` is the number of inbox
waiters currently registered or generating for that tenant (from `inbox.waiter` events).

Then, live for the selected tenant only (every frame carries the full object, upsert semantics,
ordered by the bus sequence):

- `{ "type": "turn", "turn": <TurnSummary> }` whenever a turn summary changes;
- `{ "type": "pending", "pending": <PendingMessage> }` and
  `{ "type": "pending.remove", "entry_id": "…", "turn_id": "…" }` when a queued message
  joins a turn or expires;
- `{ "type": "job", "job": <JobSummary> }`;
- `{ "type": "event", "event": <Event> }` only for turns the client follows;
- `{ "type": "tenants", "tenants": [...] }` when the tenant list changes;
- `{ "type": "evict", "before_ts": … }` every 10 s: the client drops every turn, pending
  message and job whose `updated_at` is older than `before_ts`. Eviction is at whole-object
  granularity on the server too (a turn leaves with all its events; live turns are never
  partially trimmed), so a replaying replica and a long-running replica hold the same summaries.

Client requests:

- `{ "type": "switch", "tenant": "…" }` → a fresh `snapshot`;
- `{ "type": "turn.events", "turn_id": "…" }` → `{ "type": "turn.events", "turn_id",
  "events": [ <Event> ], "truncated": false }` (raw events of one turn including the
  message-level events joined to it, oldest first; at most the newest 2 000 events, then
  `truncated: true`);
- `{ "type": "turn.follow", "turn_id" }` / `{ "type": "turn.unfollow", "turn_id" }` to
  receive live `event` frames for an open turn (at most 4 followed turns per socket);
- `{ "type": "ping" }` → `{ "type": "pong" }`; the server also pings every 20 s and
  drops sockets that miss two.

Slow consumers (send queue above 4 MiB or 2 000 frames) are dropped; they reconnect and get a
fresh snapshot. Any replica can serve any viewer: all replicas hold the same derived state
because they all consume the same bus.

### Derived types (server reducer)

```ts
type TurnStatus = "queued" | "running" | "replied" | "silent" | "error";
type StageKey = "intake" | "coalesce" | "classify" | "recall" | "reason" | "tools" | "guard" | "reply" | "remember";
interface StageSummary { key: StageKey; started_at?: number; ended_at?: number; status: "idle" | "running" | "ok" | "degraded" | "error" | "skipped"; duration_ms?: number; note?: string }
interface TurnSummary {
  turn_id: string; tenant: string; attempt: number; seq: number; updated_at: number;
  entry_ids: string[]; sidecar_session_id?: string; thread_id?: string;
  conversation?: Conversation; actors: Actor[];
  started_at: number; ended_at?: number; status: TurnStatus;
  current_stage?: StageKey; stages: StageSummary[];      // always all nine, in order
  llm_calls: number; tool_calls: number; tokens?: { input: number; output: number; total: number };
  recall?: { active: boolean; episodes: number; lanes: number; degraded: boolean; cache_hit: boolean;
             lanes_detail: { intent_kind: string; intent_source?: string; candidate_count: number;
                             duration_ms?: number; outcome: "fulfilled" | "rejected" | "running" }[] };  // <= 12, newest
  degraded: boolean; degraded_reasons: string[];
  classifier?: { speak: boolean; reason: string };
  reply_preview?: string; sources: ("team-agent" | "borg")[]; event_count: number;
}
interface PendingMessage {
  entry_id: string; tenant: string; sidecar_session_id: string; seq: number; updated_at: number;
  enqueued_at: number; settle_deadline?: number; settle_bound?: "quiet" | "max";
  text_preview?: string; actor?: Actor; conversation?: Conversation;
  waiter?: "registered" | "generating";
}
interface JobSummary {
  run_id: string; tenant: string; seq: number; updated_at: number;
  kind: "maintenance" | "ingestion" | "offline_process";
  name: string; started_at: number; ended_at?: number;
  state: "running" | "completed" | "completed_with_errors" | "aborted" | "error";
  processes?: { name: string; status: "pending" | "running" | "completed" | "skipped" | "error"; duration_ms?: number }[];
}
```

`seq` is the bus sequence of the last event applied, so a client can discard stale frames.
`updated_at` is the server receive time of that event and is what `evict.before_ts` compares
against. Producer `status`/`state` strings outside the enums above are mapped by the server
(borg maintenance `pending`/`completed`/`skipped` pass through; a process with `error_details`
or `error` becomes `error`).
