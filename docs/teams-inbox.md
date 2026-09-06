# Teams inbox: coalesced turns through borg's durable inbox

Status: v1.9 contract, 2026-09-05 (v1.9: an `agent_msg` terminal records the owner's `borg_replied` activity). All OPEN points are settled; v1.3 added the implementers'
clarifications (transport envelope, await tenant mapping, response union, terminal reclaim,
recovery settings, delivery guarantees); v1.4 adds the review outcomes (seal-on-claim, stale
batches, connector failures park, batch rendering order, atomic post-send bookkeeping); v1.5
makes bridge intake refuse-proof after the first production burst; v1.6 adds the `generating`
interim status so the bridge can show a typing indicator exactly while the agent composes, and
resolves the classifier's praise contradiction; v1.7 makes the interim signal acknowledgement-aware
(review found a re-await hot loop) and gives the generating state a bounded lifecycle; v1.8
ties the stale threshold to the runner timeout and records the production timeouts.
Identical copies live in `team-agent/docs/teams-inbox.md` and `borg/docs/teams-inbox.md`;
every amendment must land in both. Implementers report needed amendments; they do not edit
this file.

## Why

Today every Teams mention is one turn and every unmentioned message is debounced by the
bridge into an `/v1/chat/observe` call. Bursts get one reply per mention, a mention
force-flushes pending observations (two upstream calls, possibly two bot messages), and a
message that arrives while a turn runs is never seen by that turn.

Sol (borg on ivory) does not have this problem because its connector only enqueues, and
borg's durable inbox (`ChatResponseCatchUpWorker`) coalesces: quiet window + max wait,
oldest-first contiguous prefix of the unanswered backlog (max 16), one turn per batch,
per-session serialisation, responded-through watermark, startup scan after restart.

This port reuses that machinery unchanged and swaps only the cognition: the worker's turn
runner calls team-agent instead of borg's deliberation. Delivery stays in the bridge, which
keeps its request to team-agent open until the coalesced reply exists, exactly as it does
today for a single mention.

## Flow

```
Teams -> bridge /api/messages (Bot Framework auth, ledger claim, conversation reference stored)
      -> [per-thread ingress FIFO] team-agent POST /v1/teams/enqueue  -> sidecar POST /memory/enqueue
      -> [FIFO released]           team-agent POST /v1/teams/await    -> sidecar POST /memory/await-response
sidecar worker: quiet 3 s / max 15 s, per session, startup scan, backoff   [borg, unchanged]
      -> TeamAgentTurnRunner: read batch -> team-agent POST /v1/chat/observe (source "inbox")
      -> append terminal: agent_msg (reply) | agent_observed (silent), stamped response_to
      -> borg.episodic.ingest({session}); resolve waiters
team-agent returns the terminal to the bridge; the bridge posts it once and completes.
An `agent_msg` terminal also records the memory owner's reply as a `borg_replied` activity event
(speaker and actor = the self entity, audience = the session's audience entity, participants =
self + the batch senders + audience, source = the terminal entry) and touches the session, exactly
as `/memory/append-turn`'s reply-only mode does; an `agent_observed` terminal records nothing. A
missing session, self entity or audience, or a projection error, leaves the committed terminal
intact and logs a redacted warning. Until 2026-09-05 the inbox path skipped this projection, so
Teams replies never reached `recent_activity` or the recall planner's owner rows.
Terminals written before that fix are repaired by `POST /memory/maintenance/inbox-reply-activity`
(`{ tenant, dry_run?: true, since?: ISO-8601, until?: ISO-8601, limit?: <=5000 }`, token-protected,
runs under the tenant's exclusive chain, so it is bounded: at most 1000 sessions per pass, a scan
cap of 20000 terminals, and `limit` inserts). It scans `teams_inbox` sessions for active
`agent_msg` terminals stamped `response_to.kind = "stream_backlog"` in the window (unstamped
legacy entries are ignored), counts those already carrying a `borg_replied` event, and projects the
missing ones with the same builder the live runner uses, reconstructing the batch senders in stamp
order; `dry_run` (the default) only counts. The response reports `sessions_scanned`,
`sessions_truncated`, `terminals_scanned`, `inactive_skipped`, `already_recorded`, `inserted`,
per-reason `skipped` counts (including `malformed_stamp` and `projection_failed`), up to 50
`failed_terminal_ids`, `truncated`, and `complete`. Re-running is safe: kind + source dedupe makes
every insert idempotent, and a partial pass is simply run again.
```

Nothing in the cluster ever calls the bridge: direct egress has no DNS and the corporate
proxy returns 403 for the Azure host. Pull/hold is the only shape that works.

Enqueue happens under the bridge's existing per-thread ingress FIFO, so stream order equals
Teams arrival order; the long await runs outside the FIFO so waits overlap and join batches.

## Identity and sessions

- `thread_id`: the raw team-agent thread key (`<tenant>::shared::<type>::<conversation>`
  for groups and channels, `<tenant>::<user>::<conversation>` for personal chats; see
  `team_agent/conversations.py`). It is what the append-turn path already sends as
  `session`, so history, recall and venue recency stay continuous.
- `sidecar_session_id`: borg's internal session id, derived by the sidecar from `thread_id`
  (`sessionFromCaller`, today `sess_<sha256 prefix>`). Callers never derive it; they read it
  from the enqueue response and pass it back to await.
- Sender and audience entities: the same resolution the append-turn path uses
  (`resolveTeamAgentIdentity`). No second resolver.
- Inbox-managed sessions must be distinguishable from append-turn-managed ones so the worker
  never drains a console or REPL session. The enqueue route claims the session with
  `source_type: "teams_inbox"` and `source_external_id: <Teams conversation external_id>`
  (`sessions.ensure` replaces both on conflict; recall, venue recency and participants do
  not key on them). The worker's session filter admits only `teams_inbox`, applied at
  startup scan, append notification, pending notification and immediately before a drain.
- Ownership is sticky: once a session is `teams_inbox`, the `/memory/context` and
  `/memory/append-turn` identity refreshes may update labels, audience and conversation kind
  but must preserve `source_type` and `source_external_id`. Only the enqueue route claims a
  session for the inbox.
- Seal on claim: append-turn wrote its `user_msg` entries without `turn_id`, so a claimed
  session's whole history looks like unanswered backlog to the worker, and those entries carry
  no `metadata.teams_inbox`. The first enqueue for a session therefore, inside the same
  exclusive chain and before appending the new message, seals every pending user entry up to
  the current tail with one `agent_observed` terminal (ordinary catch-up ingestion, since it
  straddles ingested history). The runner additionally treats a prefix containing any entry
  without the envelope as a legacy remnant and seals it as observed with a note, never as a
  repeated failure.
- Source message key: `{ source_type: "teams_inbox", source_external_id: <Teams conversation
external_id>, external_message_id: <Teams activity id> }`, matching the session's own
  source fields as the enqueuer requires. Redelivery by Teams or a bridge retry yields
  `status: "duplicate"` and must not enqueue again.
- `observed_at` is persisted as the entry's top-level `observed_at` (epoch ms), via a new
  `observedAt` on `enqueueMessage`; the existing `arrivedAt` only touches activity and
  session times. Staleness and ordering use `observed_at ?? timestamp`.
- Participation policy (active / observing / muted) governs whether the agent speaks and what
  the graph history persists. It does not govern the durable stream: observed messages are
  memory regardless, as they are for Sol.

## Per-message flags

Each inbound message carries `mentioned` (the bot was @mentioned) and `quotes_bot` (the
message quotes one of the bot's own posts). The bridge reports facts only; it never sets
`mentioned` for a personal chat to force a reply. Transport to the runner: a generic
optional `metadata: Record<string, JsonValue>` on the stream entry schema, passed through
`enqueueMessage` onto the `user_msg` entry (JSONL only; the SQLite index ignores it, so no
migration). No structural header line in the content. The envelope is

```json
"metadata": { "teams_inbox": { "thread_id": "...", "sender": { "external_id": "...", "display_name": "...", "bot": false }, "mentioned": false, "quotes_bot": false } }
```

because `sidecar_session_id` is a one-way hash of `thread_id` and entities have no reverse
external-id lookup: after a restart the runner rebuilds the team-agent request from the
entries alone.

## Sidecar routes (borg-memory, per tenant; auth: existing `x-borg-token`)

### POST /memory/enqueue

```json
{
  "tenant": "team-agent-ai",
  "session": "team-agent-ai::shared::groupChat::19_d93...",
  "conversation": {
    "external_id": "19:d93...@thread.v2",
    "type": "groupChat",
    "name": "AI Ninjas"
  },
  "sender": {
    "external_id": "4ea458f9-...",
    "display_name": "Kowalski, Tomasz",
    "bot": false,
    "operator": false
  },
  "text": "Zrozumiał Pan?",
  "external_message_id": "1788450205707",
  "observed_at": "2026-09-03T15:43:25.707Z",
  "flags": { "mentioned": false, "quotes_bot": false }
}
```

`session` is the raw `thread_id`; `conversation.external_id` is required. Identity
resolution, the session claim and `borg.enqueueMessage` run under the tenant's
`{ exclusive: true }` chain. Response
`200 { "status": "enqueued" | "duplicate", "sidecar_session_id": "sess_...", "entry_id": "..." }`.
Errors: 400 validation, 401 auth, 503 tenant unavailable.

### POST /memory/await-response

```json
{
  "tenant": "team-agent-ai",
  "sidecar_session_id": "sess_...",
  "entry_id": "...",
  "timeout_ms": 90000
}
```

Response is a discriminated union:
`{ "status": "answered", "terminal_id", "entry_ids", "reply" }`,
`{ "status": "observed", "terminal_id", "entry_ids" }`, `{ "status": "generating" }`, or
`{ "status": "pending" }`.

`generating` is an interim wake-up, not a terminal: it means the agent has decided to reply to
a batch covering this entry and is composing. The caller shows a typing indicator and re-awaits
the same entry with `"seen_generating": true`; the next answer is the terminal (or `pending` on
timeout). The registry remembers the generating state per covered entry so a waiter that
registers after the signal, without `seen_generating`, receives `generating` immediately; a
request carrying `seen_generating: true` never receives `generating` and holds for a terminal
or its timeout. Lifecycle of the remembered state: set by a progress call, cleared when the
entry's terminal commits, ignored once a terminal has committed (a bounded tombstone of at
least 10 minutes absorbs a late progress call), and expired unconditionally after
`TEAM_AGENT_TIMEOUT_MS` plus 10 s, since no generation outlives the runner's request.

- `answered`: an `agent_msg` whose `response_to` stamp covers `entry_id`; `reply` is its
  content. `observed`: an `agent_observed` or `agent_suppressed` stamp covers it.
  `pending`: `timeout_ms` elapsed; the caller re-issues, nothing is lost.
- Lookup order is scan, register, scan again: if a covering stamp exists on arrival, answer
  from the stream; otherwise register an in-process waiter (many waiters per entry allowed,
  keyed by tenant, session and entry) that the runner resolves after the durable append. A
  waiter is removed on timeout, client disconnect and shutdown; shutdown resolves the rest
  as `pending`.
- `timeout_ms` default 90000, cap 120000. Unknown entry or session mismatch: 404. Await is
  read-only and does not take the tenant's exclusive write chain.
- `terminal_id` is the terminal entry's stream id; callers key their once-only delivery on
  `(sidecar_session_id, terminal_id)`.

### POST /memory/inbox-progress

```json
{
  "tenant": "team-agent-ai",
  "sidecar_session_id": "sess_...",
  "entry_ids": ["..."],
  "phase": "generating"
}
```

Marks the covered entries as generating and wakes their waiters with `{ "status": "generating" }`.
Response `200 { "ok": true }`; 404 on unknown session; idempotent. Best-effort by design: a
failed or late progress call must never affect the turn or the terminal.

Who calls it: the runner itself, right before its team-agent request, for a batch that will run
as a full turn (personal conversation, or any message with `mentioned` or `quotes_bot`), since it
already knows a reply will follow; team-agent, for a classifier batch, right after the
classifier decides to speak and the guardrails admit the chip-in, before generation begins.
Silent batches never signal.

## Runner (sidecar, TypeScript)

Constructed per tenant and injected as the worker's turn runner. For each drained batch:

1. Hydrate the full source entries of the worker-supplied prefix in durable order (content,
   sender entity and display name, `observed_at`, `metadata` flags, conversation) and
   reconcile first; an already-stamped batch is a no-op.
2. Stale batch: only when every entry of the prefix is older than `TEAMS_INBOX_STALE_MS`
   (default 600000) by `observed_at ?? timestamp` (equivalently, the maximum is stale), seal
   without calling team-agent. The seal covers the maximal stale contiguous prefix of the
   session's unanswered backlog from the watermark, uncapped, stopping before the first
   `receipt_pending` entry and before the first fresh entry. A fresh tail is never sealed. A
   prefix mixing stale and fresh entries is handed to team-agent whole. After a run the
   worker reconciles: watermark at or beyond the prefix is success; advanced but short of the
   prefix is partial progress and re-drains immediately; unchanged is a failure.
3. Otherwise `POST <team-agent>/v1/chat/observe` (below) with native `fetch`, no redirects,
   `AbortSignal.timeout(TEAM_AGENT_TIMEOUT_MS)` (default 120000). Transport error, 5xx or a
   malformed 2xx body: throw so the worker retries with its backoff. 4xx: seal the prefix as
   `agent_observed` with a note so a bad batch cannot wedge the session.
4. Append the terminal through the shared inbox terminal service with a valid
   `stream_backlog` stamp: `agent_msg` with the reply, or `agent_observed` when silent.
   Advance the watermark. Live terminals then run exact answered-window ingestion; a stale
   seal that may straddle already-ingested history runs ordinary catch-up ingestion
   instead. Waiter resolution keys off the durable stamp, not off the runner's success
   path: a committed terminal resolves waiters even if the watermark advance or the
   ingestion after it fails, and a stamp the worker discovers during reconcile (a previous
   run crashed after the append) resolves them too.
5. Background inbox work holds a pool-visible lease on its tenant's Borg instance while a
   drain is scheduled or in flight and while waiters are registered, so pool eviction cannot
   cancel a drain under a live-poll and a graceful close never blocks an unrelated tenant.
6. Never deliver anything itself. Delivery belongs to the bridge.

The runner's success is the durable stamp, not its return value: after a run the worker
reconciles and asserts the watermark is at or beyond the supplied prefix. The runner binds
to the currently open Borg's inbox operations and never re-enters the tenant pool; the
worker's own per-session in-flight map is the only serialisation it needs, since the sidecar
runs no borg turns, autonomy or maintenance schedulers.

Configuration (sidecar env): `TEAM_AGENT_BASE_URL` (in-cluster, e.g.
`http://team-agent:8080`), `TEAM_AGENT_API_TOKEN` (sent as `Authorization: Bearer`),
`TEAM_AGENT_TIMEOUT_MS` (120000), `TEAMS_INBOX_SETTLE_MS` (3000), `TEAMS_INBOX_MAX_SETTLE_MS`
(15000), `TEAMS_INBOX_STALE_MS` (600000). `TEAMS_INBOX_STALE_MS` must exceed
`TEAM_AGENT_TIMEOUT_MS` plus `TEAMS_INBOX_MAX_SETTLE_MS`, and the sidecar refuses to start
otherwise: a batch retried after a runner timeout is at least that old when it is claimed
again, and a lower threshold would seal it unanswered as stale. Production runs 600000 /
900000 since 2026-09-04, after a code-exploring turn measured 399 s against the 120 s default.
The inbox settle values are separate from the generic stream-ingestion settle config. Session filter: `source_type === "teams_inbox"`.
The team-agent tenant id equals the sidecar tenant name (true today: `team-agent-ai`); the
runner sends it as `model`.

## team-agent endpoints

### POST /v1/teams/enqueue (bridge -> team-agent; existing Bearer auth and Teams headers)

```json
{
  "conversation": { "external_id": "...", "type": "groupChat", "name": "AI Ninjas" },
  "sender": { "external_id": "...", "display_name": "...", "bot": false },
  "text": "...",
  "external_message_id": "1788450205707",
  "observed_at": "2026-09-03T15:43:25.707Z",
  "mentioned": false,
  "quotes_bot": false
}
```

Resolves the tenant with the existing Teams routing (personal: `X-User-Email` against
`tenant_member`; groups: `teams_conversation_binding`), computes `thread_id`, calls sidecar
enqueue (idempotent), and returns

`200 { "status": "enqueued" | "duplicate" | "unbound", "thread_id"?: "...",
"sidecar_session_id"?: "...", "entry_id"?: "...", "reply"?: "..." }`.

`unbound`: no tenant binding. Nothing is enqueued. `reply` carries the pairing text when the
message was solicited (mentioned, quotes the bot, or a personal chat), matching what
`/v1/chat/completions` returns today; an unmentioned group message stays silent, matching
`/v1/chat/observe` today. The bridge delivers an `unbound` reply once (Activity dedup) with
no terminal claim.

The legacy `TEAM_AGENT_TENANT` pin is retired; the bridge sends no `model`.

### POST /v1/teams/await (bridge -> team-agent; same auth)

```json
{ "sidecar_session_id": "sess_...", "entry_id": "...", "wait_ms": 90000 }
```

Proxies sidecar await-response and returns its body unchanged, passing `seen_generating`
through. `wait_ms` default 90000, cap 120000. The HTTP client read timeout on both hops is `wait_ms` plus a 10 s margin. The body
carries no tenant: team-agent durably records `sidecar_session_id -> tenant` at enqueue and
re-applies the caller's Bearer tenant scope on await; an unknown id is 404.

### POST /v1/chat/observe (runner -> team-agent; existing Bearer auth, tenant from `model`)

Extends the existing batch-shaped body: `source: "inbox"`, `thread_id`, `sidecar_session_id`,
`conversation {external_id, type, name}`, and per message `entry_id`, `mentioned`,
`quotes_bot`, `sender {external_id, display_name, bot}`, `observed_at`. No Teams routing
headers are sent; the tenant is the request's `model`, checked against the token's scope.

- Full turn (tools, no classifier, no chip-in guardrails) when the conversation is personal
  or any message has `mentioned` or `quotes_bot`. The batch is rendered in `observed_at`
  order (entry order as tie-break) with sender attribution, because a transient bridge
  enqueue failure can land a retried message later in the stream than a younger one; the
  raw texts, not the attributed renderings, feed the assistant-mode Assets precollection; each message is stored in the graph history as its own
  `HumanMessage` with `id = entry_id`. The acting sender is the newest message carrying a
  trigger flag (or the sole sender in a personal chat). Operator authority is granted only
  when every message in the batch is from that same sender; a mixed-sender batch has no
  operator, mirroring borg.
- Otherwise the existing classifier path, judging the whole batch.
- `source: "inbox"` disables `schedule_append_turn` for both sides: the user entries are
  already in the stream and the runner appends the reply.
- Idempotency: the batch key is `(tenant, thread_id, newest entry_id)`, derived server-side
  from the request. team-agent stores a receipt with the response under that key and
  returns it for a repeated request instead of running the model or tools again, so a
  runner retry after a crash between the turn and the stamp cannot repeat tool side effects.
  Receipts are retained 24 h, and a cross-replica lock keyed the same way serialises
  overlapping runner retries. Guarantee boundary: a team-agent process death after a tool's
  external side effect but before the receipt commit can still repeat that side effect on
  retry; only tool-level idempotency keys would close that, and they are out of scope here.
- An empty model reply is `silent`, never an error.

Response unchanged: `{ "action": "reply" | "silent", "content"?: "...", "reason"?: "..." }`.

### Classifier changes (same change set)

- New solicited reason `directed_at_agent`: formal or second-person address, praise, thanks,
  banter, or a follow-up inside an exchange the agent is already part of, when the message
  itself shows it is aimed at the agent (second person, the agent's name or role, or it
  answers or reacts to the agent's last content). Chronology alone is not evidence: a message
  that names or answers another human right after the agent's reply stays silent. Treated like
  `addressed_by_name`: no gap, no hourly cap. This emulates a coworker: anything said to you
  gets at least a brief acknowledgement, whether or not it contains a question.
- Humour between humans stays silent; humour aimed at the agent is a message to answer
  briefly. The "statements without a question or request stay silent" rule applies only to
  speech not aimed at the agent; the first production run showed the model letting that rule
  override rule 2 for "Piknie, brawo TA".
- `unanswered_domain_question` requires an actual question or request, not a statement
  that might be one.
- The reply-only step may return nothing for a pure reaction; treat that as silent.

## Bridge

- Ingress unchanged: Bot Framework auth, ledger claim, admission, per-thread ingress FIFO,
  mention detection, quoted-bot detection from `bot_messages`. The receipt timestamp is
  captured on arrival and the activity joins its thread FIFO before any asynchronous lookup
  (the quoted-bot ledger read included), so arrival order is the enqueue order. New: strip
  every `<quoted messageId="..."/>` marker from the text after detection; persist the
  conversation reference (aliased Pydantic JSON) with each pending activity.
- Admission gates only the await, never the intake. A valid Activity is always claimed,
  enqueued in FIFO order and answered 202; only then does it need a slot for the long await.
  When the global or per-thread cap (`TEAMS_BRIDGE_MAX_IN_FLIGHT_TURNS`,
  `TEAMS_BRIDGE_MAX_IN_FLIGHT_TURNS_PER_THREAD`, now meaning concurrent awaits) is exhausted
  the row is parked due-now with one warning log line and recovery re-awaits it as capacity
  frees. 503 with Retry-After is reserved for a ledger that cannot record the claim at all.
  (First production burst: five messages in four seconds against a per-thread cap of 4 refused
  the fifth, the mention; Teams retries a 503 once and then drops the message.)
- Per message: enqueue under the ingress FIFO, then await outside it. Typing indicator from the
  start of the await for a message that is personal, mentions or quotes the bot; for any other
  message only once the await returns `generating`, after which the bridge re-awaits the same
  entry with `seen_generating: true` (and keeps sending it for that delivery) and keeps typing
  until the terminal arrives. Recovery never types and always sends `seen_generating: true`. On `answered`: claim
  `(sidecar_session_id, terminal_id)` in `delivered_terminals`; the winner posts, records
  `sent_at` and the bot message id, losers complete silently. The owning activity may reclaim
  a terminal it claimed but never marked sent (a crash or cancellation between claim and
  post), so recovery can finish that delivery; any other activity always loses. A crash after
  Teams accepted the post but before `sent_at` was recorded is an unavoidable duplicate
  window: the Bot Connector offers neither an idempotency key nor a transactional
  acknowledgement. On `observed`: complete. On `pending`: re-issue the await; after the
  foreground deadline (default 5 min) park the activity. `unbound`: deliver the pairing reply
  once when present, complete.
- Errors: transport, 429, 5xx and any other unexpected status (3xx included) or malformed
  body from team-agent retry with backoff then park; 4xx other than 429 is a permanent
  failure that, for solicited messages only, sends the existing generic failure reply and
  only then finalises the activity. A Bot Connector failure while posting an answered or
  unbound reply parks the activity with its pending row and terminal ownership intact, so
  recovery finishes the delivery; it is never turned into a failed activity. The adapter's
  automatic generic reply is suppressed on processor-managed paths.
- The await client requires a non-empty `entry_ids` on answered/observed responses and that
  the awaited `entry_id` is among them; anything else is a malformed response and retryable.
- After a successful post, one atomic ledger operation verifies terminal ownership, records
  `sent_at` and the bot message id, completes the activity and deletes its pending row.
- Durable state: `pending_activities` (activity key, request JSON without credentials,
  `sidecar_session_id`, `entry_id`, conversation reference, state active/parked/recovering,
  attempts, `next_attempt_at`) written in the same transaction as the ledger claim, so a
  crash during the first await loses nothing. `delivered_terminals` claims are kept for 30
  days. Completion deletes the pending row and completes the ledger row in one transaction.
- Recovery: on startup, leftover active/recovering rows become parked. A worker wakes at
  startup and whenever a row is parked, re-awaits due rows oldest-first under the same
  admission limits, applies the same status handler, sends no typing indicators, and sleeps
  when nothing is parked. Replies are addressed with the activity's own stored reference.
- Removed: observation debounce and batching, forced flush before a mention, whole-turn
  locks, `/v1/chat/completions` and `/v1/chat/observe` calls,
  `TEAMS_BRIDGE_OBSERVE_DEBOUNCE_SECONDS`, `TEAMS_BRIDGE_OBSERVE_TIMEOUT_SECONDS`,
  `TEAM_AGENT_TENANT`. `TEAMS_BRIDGE_OBSERVE_UNMENTIONED` remains the gate for forwarding
  unmentioned messages. New settings: `TEAMS_BRIDGE_WAIT_MS` (90000),
  `TEAMS_BRIDGE_FOREGROUND_DEADLINE_SECONDS` (300; production 600 since 2026-09-04 so the
  foreground hold roughly matches the sidecar's 600 s runner timeout),
  `TEAMS_BRIDGE_RECOVERY_BACKOFF_MIN_SECONDS` (1), `TEAMS_BRIDGE_RECOVERY_BACKOFF_MAX_SECONDS`
  (60). Rolling the bridge back to the previous build pauses recovery of rows parked by this
  build until it is rolled forward again.

- Phase B agent task deliveries are pulled by the bridge; nothing in the cluster calls
  Azure. At enqueue, `conversation_registry` atomically stores the returned
  `sidecar_session_id`, conversation/channel IDs, bot/tenant IDs when available, the
  request's aliased conversation reference, and first/last-seen timestamps. Each
  validated Activity refreshes an existing registration before lifecycle, local-duplicate,
  or dropped-unmentioned returns, even if enqueue fails. Refreshes stay inside the ingress
  FIFO; persisted receipt timestamps prevent an older enqueue retry or queued Activity
  from overwriting a newer reference. The initial session mapping still comes from enqueue.
  Registry rows survive at least `TEAMS_BRIDGE_DEDUP_RETENTION_SECONDS` after last seen,
  and longer while pending Activities or deliveries still need them.
- With delivery polling enabled, startup discovers `GET /v1/teams/capabilities` in the
  background. Only `agent-task-v1: true` starts delivery work; an absent/false capability
  or 404 quietly disables it until restart. Discovery failures back off without delaying
  intake. One poll loop refreshes all registered sessions each sweep and calls
  `POST /v1/teams/agent-deliveries/claim` in consecutive chunks of at most 200, sending
  `sidecar_session_ids`, `wait_ms` (default/max 60000), and `lease_ms` (default 120000).
  Empty chunks can hold the request for the full wait before the next chunk is polled.
  Transient failures back off before the next claim, preserving progress through chunks.
  All calls use the existing Bearer token and CA bundle, without redirects; the token
  implies the tenant (one tenant per token). Claim read timeout is wait plus 10 seconds.
- Before issuing an HTTP claim, durable `delivery_claims` rows pin its registered sessions.
  Lost responses, persistence failures and interrupted requests keep pins across restart.
  An uncertain request can still lease work during its poll wait; pins survive for wait
  plus lease duration after failure or restart, and until a successful retry starts after
  that interval. Earlier empty responses and elapsed time alone cannot release them.
  Earlier uncertainty also survives changes to the lease setting. Cleanup cannot remove
  those references even while there are no local pending deliveries.
- Leased batches are persisted before posting in typed `pending_deliveries`: delivery,
  session, terminal and task IDs, content, timestamps, active/parked state, attempts,
  next-attempt time, final outcome and Teams message ID. Pending rows, acquisition of
  available terminal claims and resolution of eligible HTTP claim pins share one SQLite
  transaction, so a rollback leaves all three unchanged. A separate coordinator takes
  due rows oldest-first with bounded concurrency across new and recovered deliveries;
  it uses the existing backoff calculation and demotes leftover active rows on startup.
  Excess leased work stays parked locally until capacity is available.
- Agent deliveries reuse the existing `(sidecar_session_id, terminal_entry_id)` terminal
  claim and post through `TeamsConnector.run_in_conversation` using the registry reference.
  They never type or emit the generic failure reply. Before acking success, one transaction
  records terminal sent-at, the Teams message ID (including quoted-bot detection), and the
  pending row's final outcome. Redelivery or restart then retries only the ack. A different
  owner of an unsent terminal parks and retries; it cannot post that terminal.
- `POST /v1/teams/agent-deliveries/ack` sends `delivery_id`, `outcome`, and optional
  `teams_message_id` or a content-free `error` code. Success acks `sent`; Connector failures
  park with backoff and ack `failed_retryable`; missing/invalid references and Connector
  403/404 record a permanent result and ack `failed_permanent` with a warning. A successful
  final ack deletes the pending row; an ack failure keeps the final outcome for recovery.
  Any HTTP 2xx is a successful ack, regardless of response body. Upstream normally answers
  `200 {"status": "acknowledged"}`; this JSON shape is informational and is not required.
  Terminal results retain the existing 30-day retention, extended while a pending delivery
  needs them. The same unavoidable post-accepted/before-SQLite-receipt crash window as inbox
  replies applies: Bot Connector has no idempotency key or transactional acknowledgement.
- Outcome logs contain identifiers and outcomes, never message content. Shutdown stops the
  poll and recovery coordinators, drains active sends within the existing shutdown budget,
  then cancels and parks unfinished work before closing the shared HTTP clients.

Delivery polling settings (defaults):

```text
TEAMS_BRIDGE_DELIVERY_POLLING=1
TEAMS_BRIDGE_DELIVERY_POLL_WAIT_MS=60000
TEAMS_BRIDGE_DELIVERY_LEASE_MS=120000
TEAMS_BRIDGE_DELIVERY_MAX_IN_FLIGHT=4
```

Existing recovery/timeout settings also used by agent deliveries (defaults):

```text
TEAMS_BRIDGE_RECOVERY_BACKOFF_MIN_SECONDS=1
TEAMS_BRIDGE_RECOVERY_BACKOFF_MAX_SECONDS=60
TEAMS_BRIDGE_CONNECTOR_SEND_TIMEOUT_SECONDS=15
TEAMS_BRIDGE_SHUTDOWN_DRAIN_SECONDS=30
TEAMS_BRIDGE_DEDUP_RETENTION_SECONDS=604800
```

## Extension: agent task events and deliveries

Phase B adds a separate lane for deferred task results. A task accepted during a Teams turn
finishes outside that turn, reports its outcome to borg, and produces a normal durable reply in
the originating inbox session. Delivery remains pull-based: team-agent claims the reply and
arranges delivery through the bridge. Borg never calls the bridge. This extension is implemented
in borg; the consumer changes and matching documentation must land separately in team-agent.

The task-event lane is **disabled by default**, independently of the existing user inbox.
Enable it with `TEAMS_INBOX_TASK_EVENTS_ENABLED=true` in the sidecar, or
`Borg.open({ inbox: { taskEventsEnabled: true, taskEventRunner: ... } })` in a library host.
A runner factory alone does not enable it. When disabled, event enqueue returns 503, startup
does not drain task events, and stream writers reject `task_event` terminal stamps. Delivery
claim/ack remain available for already-created deliveries.

**The first deploy of this version must run with the lane disabled.** Upgrade every Borg
reader before enabling task terminal writes. From this version onward, `StreamReader`, indexed
reads and index backfill retain entries with an unknown response stamp kind: the stamp is
preserved as read-only `opaque_response_to`, with no interpreted `response_to`. Known malformed
stamps remain invalid, and writers remain strict. The existing `stream_backlog` schema and
watermark semantics are unchanged; the SQLite migrations are additive and need no index rebuild.

After the lane has written terminals, rolling back to a **pre-task_event Borg makes its older
readers skip those entire terminals**, because they reject the unfamiliar stamp kind. The
new tolerant reader cannot change old binaries. Disabling the lane stops new writes but does
not make existing task terminals readable by those old versions. Use a version with tolerant
readers for rollback after enabling the lane.

All three routes require the existing `x-borg-token`, take `tenant` in the body, and use that
tenant's exclusive chain for stream/SQLite operations. A long poll releases the chain while
waiting. Validation errors are 400, authentication failures 401, and an unavailable tenant or
disabled Teams inbox is 503.

### POST /memory/agent-events

```json
{
  "tenant": "team-agent-ai",
  "sidecar_session_id": "sess_...",
  "event_id": "event-123",
  "task_id": "task-456",
  "task_version": 1,
  "kind": "task_completed",
  "occurred_at": "2026-09-06T12:30:00+02:00",
  "outcome": {
    "status": "succeeded",
    "summary": "The report is ready.",
    "artifacts": [{ "label": "Report", "url": "https://example.com/report" }]
  },
  "origin": { "source_entry_ids": ["strm_..."] }
}
```

`kind` is `task_completed` or `task_failed`; outcome `status` is `succeeded`, `failed`,
`timed_out`, or `cancelled`. `task_version` must be an integer. `occurred_at` must include a
UTC designator or offset. The summary is a string of at most 8000 characters; `artifacts` and
an outcome `error` string are optional. These envelopes and response stamps are Zod-validated.

Returns `200 { "status": "enqueued" | "duplicate", "entry_id": "strm_..." }`. The session
must already exist with `source_type: "teams_inbox"`; otherwise 404. The route appends one
`internal_event`, with a short plain-text content line and
`metadata.task_event = { schema_version: 1, event_id, task_id, task_version, kind,
occurred_at, outcome, origin }`. `(tenant, sidecar_session_id, event_id)` is the idempotency
key: repeat submissions return the original entry id, preserve its original payload, and
append nothing. Both new and duplicate submissions notify the session's task-event lane.
A poisoned stream index is repaired before checking for duplicates. If repair is unavailable,
the request fails without another append; a retry returns the original committed event after
repair succeeds.

### Task-event runner and terminal

`TeamAgentTaskEventRunner` shares `ChatResponseCatchUpWorker`'s per-session in-flight gate,
tenant lifetime lease and startup scan with the user-turn runner. Task retry state and timers
are independent of user settling and retry state. Pending user work is selected before task
terminal repair, so a failing task ingestion or delivery repair does not block user messages.
User backlog batches retain priority and coalescing. Neither task notifications nor continuation
after a task can shorten or reset a settling user batch's quiet window. Events drain individually
in stream append order, oldest first;
their reported `occurred_at` does not reorder them or subject them to the user's stale policy.
They never enter the user prefix builder and never advance the chat-response watermark.

For one unanswered event, the runner calls `POST <TEAM_AGENT_BASE_URL>/v1/chat/task-result`
with the existing `TEAM_AGENT_API_TOKEN` Bearer authentication and `TEAM_AGENT_TIMEOUT_MS`:

```json
{
  "model": "team-agent-ai",
  "sidecar_session_id": "sess_...",
  "conversation": {
    "external_id": "19:origin@thread.v2",
    "type": "channel",
    "name": "Origin room"
  },
  "event": {
    "event_id": "event-123",
    "event_entry_id": "strm_...",
    "task_id": "task-456",
    "task_version": 1,
    "kind": "task_completed",
    "occurred_at": "2026-09-06T12:30:00+02:00",
    "outcome": { "status": "succeeded", "summary": "The report is ready." }
  },
  "requester": { "external_id": "requester-id", "display_name": "Requester" }
}
```

Conversation identity comes from the stored session (`dm` maps to `personal`, `thread` to
`groupChat`). Requester identity comes only from the original source entries' Teams sender
envelopes in that same session. Missing provenance, bots, or multiple different senders yield
`requester: null`; no identity is inferred from the task summary or the latest user turn.

The expected response is exactly `200 { "action": "reply", "content": "..." }`, with
non-whitespace content. Transport errors, non-200 responses outside 4xx, malformed JSON,
and invalid or empty replies retry through the worker. Every 4xx, including 429, instead
commits the deterministic fallback `Task <task_id> finished: <summary>`.

The result is an `agent_msg` with this stamp, answering exactly one event:

```json
"response_to": {
  "kind": "task_event",
  "event_id": "event-123",
  "event_entry_id": "strm_...",
  "task_id": "task-456",
  "task_version": 1
}
```

The existing `stream_backlog` stamp remains unchanged. Task replies use the same ingestion
coordinator's explicit answered-window mechanism with exactly the event entry and terminal.
A separate durable ingestion receipt per task terminal leaves the user ingestion cursor
untouched, including when those entries straddle a user turn. The runner then uses
`buildInboxReplyActivityProjection`: speaker and actor are self, audience is the session's
audience entity, participants are self + audience, and the source is the terminal. Projection
failures log a redacted warning; the maintenance inbox-reply-activity route also repairs task
terminals. Finally the runner creates an idempotent delivery row for the terminal.

With live extraction enabled, the episodic extractor receives the validated task metadata,
including event id and outcome summary, as context attached to its matching terminal. Both
are labelled with the session audience. The LLM interprets the result, and derived episodes
carry both source entry ids with the normal reply audience/disclosure restrictions. Unrelated
unanswered user entries are excluded from this exact window; mismatched or invalid task
metadata is not supplied as extraction context.

On reopening a tenant, startup scans include sessions with internal events even when there is
no user backlog. A stamped event is never generated again. A terminal missing its delivery row
is reconciled through ingestion, activity projection, and delivery creation. Delivery rows
and ingestion receipts survive restart. The consumer should persist task-result response
receipts keyed by `(tenant, sidecar_session_id, event_id)` to cover a sidecar crash after the
HTTP response but before the terminal append, as observe receipts do for user turns.

### POST /memory/agent-deliveries/claim

```json
{
  "tenant": "team-agent-ai",
  "sidecar_session_ids": ["sess_..."],
  "wait_ms": 60000,
  "lease_ms": 120000
}
```

At most 200 session ids; an empty list returns immediately. `wait_ms` defaults to 0 and is
bounded to 0–60000. `lease_ms` is a positive integer, default 120000. Returns:

```json
{
  "deliveries": [
    {
      "delivery_id": "delivery_...",
      "claim_generation": 1,
      "sidecar_session_id": "sess_...",
      "terminal_entry_id": "strm_...",
      "task_id": "task-456",
      "content": "The report is ready.",
      "created_at": "2026-09-06T10:30:05.000Z"
    }
  ]
}
```

All currently pending rows for the listed sessions are leased atomically, in creation order;
two claimers cannot receive the same unexpired lease. `attempts` increments on each claim.
That monotonically increasing value is returned as `claim_generation`; consumers must send
the generation from that claim with its acknowledgement.
Expired leases become pending on the next claim. Without an available row, scan/register/scan
closes the notification race and waits until delivery creation, retryable failure, lease expiry,
or the deadline. A deadline or shutdown returns `{ "deliveries": [] }`. Disconnects remove
waiters; each waiter holds a tenant lifetime lease while it is registered. Unknown sessions
simply have no deliveries. Rows are stored in the tenant's own SQLite bank, uniquely keyed by
`(sidecar_session_id, terminal_entry_id)`, with state `pending | leased | sent | failed`.

### POST /memory/agent-deliveries/ack

```json
{
  "tenant": "team-agent-ai",
  "delivery_id": "delivery_...",
  "claim_generation": 1,
  "outcome": "sent",
  "teams_message_id": "teams-message-id"
}
```

`outcome` is `sent`, `failed_retryable`, or `failed_permanent`; `teams_message_id` and `error`
are optional strings. `claim_generation` is a required positive integer from the claim response.
Success is **HTTP 200 with JSON `{ "status": "acknowledged" }`**; the bridge may treat any 2xx
as success. Unknown delivery ids or never-issued generations in this tenant return 404.
For the current unexpired lease, a retryable failure releases the lease, records the error,
and wakes claimers; permanent failures become `failed`, and successful delivery becomes `sent`.

An acknowledgement receipt is stored durably per `(delivery_id, claim_generation)`. Repeating
that acknowledgement returns the same success without changing state or waking claimers,
even after restart or if the payload conflicts with the first ack. A stale generation or
expired lease is acknowledged as a no-op. It cannot release or complete a newer consumer's
lease. Sent and permanently failed rows remain terminal.

The protocol provides at-least-once delivery. A crash after Teams accepts a message but before
the ack can still duplicate it on reclaim. Claimers must finish and ack within their lease;
generation checks protect Borg's lease state but cannot undo an external Teams send by a stale
consumer. The consumer owns protection against duplicate external sends. There is no bridge
callback or push route.

## Agent tasks (phase B)

This section describes the team-agent implementation. Borg and the bridge are
separate changes; team-agent never contacts the bridge. The completion path is
`task_start -> Postgres worker -> outbox -> borg /memory/agent-events -> borg runner
-> /v1/chat/task-result -> borg terminal/delivery queue -> bridge claim/ack`.

### Tools and origin

Ops chat exposes `task_start(spec: str, system: str | null = null)` and
`task_status(task_id: str | null = null)`. They are absent from every autonomous
role and from the existing read-only assistant mode. Personal, mentioned, quoted,
and classifier-admitted inbox turns use the same tool surface. Silent batches and
completion composition never execute tools.

`task_start` commits one queued task and returns `{task_id, state: "queued"}`;
it does not wait for OpenCode. A replay of the same runtime tool call returns
the existing task id and its current state. The submission key hashes the saved
thread id and LangGraph-injected tool call id and is unique within the tenant;
the tool call id is absent from the model-visible argument schema. Distinct tool
calls can submit several tasks in one turn. Tenant, requester, conversation, original thread,
opaque sidecar session id and source entry ids come only from runtime contexts.
The inbox origin is bound while `observe_chat` processes the batch and is cleared
afterward. Legacy/console turns without an inbox origin cannot submit a task.
The prompt requires the confirmed task id and says the result will be posted here.

Specs are self-contained instructions (1–32000 characters). Repository/MR work
can use a JSON string containing `task`, `repo`, `source_branch`, `target_branch`;
those are forwarded to the existing OpenCode delegate, preserving its repository
lock, branch/publication guards and cleanup. No identity, credential, agent or
delivery override is accepted in this JSON form. `system` is the existing logical
OpenCode system and is checked against the tenant's current assignment by the
delegate. An MR without the required branch coordinates fails its existing guard.

`task_status` only reads: the latest 20 tasks, or one UUID, scoped to the tenant
and the same raw conversation id **and type**. A requester's personal chat may
also see their own tasks from other conversations. Inaccessible and unknown ids
both return an empty `tasks` list. Each result includes state, attempt, elapsed
seconds, last heartbeat, summary, artifacts, error and delivery state.
`delivery_state: sent` means the sidecar accepted the event; it is not confirmation
that Teams received the eventual message. `not_ready` means no terminal event yet.

### Persistence and loops

`config_store.setup_config_store()` creates these tables under its existing
transaction and startup advisory lock:

- `agent_tasks`: immutable submission/origin coordinates, spec/system,
  `queued|running|succeeded|failed|timed_out|cancelled`, transition version,
  submission key, attempt and repository-deferral counts, next admission time,
  owner/lease/heartbeat/deadline, result summary/artifacts/error,
  creation/update timestamps. Conversation/origin fields are stored as columns;
  source entry ids and artifacts are JSONB.
- `agent_task_events`: UUID event id, task/version, `task_completed|task_failed`,
  immutable JSONB payload, `pending|sent|failed` relay state, attempts, next due
  time, creation and sent timestamps. `(task_id, task_version)` is unique.
- `agent_task_replies`: first composed completion or fallback per
  `(tenant_id, event_id)`, durable across replicas/restarts. Phase B retains task,
  event and reply records indefinitely; it adds no retention or cancellation API.

The lifespan starts worker and relay jobs with `_spawn_background_task` and cancels
them on shutdown. Workers claim with `FOR UPDATE SKIP LOCKED`. An additional
transaction advisory lock enforces `agent.task_concurrency` (default **2 per
replica**, across all tenants and Gunicorn processes); hostname identifies the
replica and a UUID identifies each process owner. Process-shared kernel file
locks additionally reserve replica slots from before claim until the delegate
thread exits, including after lease loss or during shutdown. Unlike a database
lease, this admission survives a database outage while work drains, and process
death releases it automatically. Gunicorn processes share the replica's temporary
directory; the admission files are zero-byte locks and must not be unlinked while
processes are running. Different replicas can claim
different tasks concurrently. State writes are fenced by version, owner and a
live lease. Heartbeats run every **30 s**, renewing a **90 s** lease. Each attempt
has a **900 s** wall-clock deadline and its own OpenCode session scope, separate
from the chat and all other attempts.

The blocking delegate runs in an executor thread with copied runtime contexts.
A stopped failed attempt may
be requeued once (**at most two attempts**); only the final outcome emits an event.
The delegate's host-produced repository-busy refusal instead defers admission
for 5, 10, 20, … seconds, capped at 300 s, without consuming an execution attempt
or emitting an event. The due time and deferral count survive restart.
Timeouts are terminal and are not retried: the host requests session abort, ignores
late success, and retains its concurrency slot until the Python thread exits.
While the lease remains valid it is renewed during draining. The local stop flag
is set synchronously before persistence; remote abort runs in a separate executor
and shares the bounded, sanitized abort transport with interactive delegates.
Unsuccessful abort HTTP statuses are reported without response bodies or credentials.
HTTP request deadlines also cover repository preparation and shell calls.
Remote abort is best-effort and cannot undo external effects. At startup and every
30 s, expired running leases become failed with an explicit “OpenCode session was
not resumed” error; live peer leases are preserved. Terminal state and outbox event
commit atomically. Completed delegate outcomes stay in memory while fenced
finalization retries with continuing heartbeats; persistence failures never rerun
the completed delegate. Each finalization write is bounded to 5 s. This recovery
requires a live process and lease; loss of either retains the reconciliation rule
above. Execution retries do not guarantee exactly-once external effects.

Task-store operations have a 5 s total budget, a 3 s connection timeout, a 4 s
statement timeout and a 3 s lock timeout.

The relay claims each due event for 60 s using a locked row and an incremented
attempt fence, then performs the HTTP call outside the transaction, with a 30 s
timeout. It uses the existing memory async POST helper, configured URL/CA and
`x-borg-token`; credentials and transport/delegate error bodies are never logged
by these loops. Failures retry after 1, 2, 4, … seconds, capped at 300 s, including
after a restart. A late acknowledgement cannot overwrite a newer relay attempt.

`POST /memory/agent-events` receives:

```json
{
  "tenant": "acme",
  "sidecar_session_id": "sess_...",
  "event_id": "<uuid>",
  "task_id": "<uuid>",
  "task_version": 3,
  "kind": "task_completed",
  "occurred_at": "2026-09-06T12:00:00+00:00",
  "outcome": {
    "status": "succeeded",
    "summary": "Completed the requested work.",
    "artifacts": [{"label": "Report", "url": "https://example.org/report"}],
    "error": null
  },
  "origin": {"source_entry_ids": ["entry-1"]}
}
```

Summary is capped at 8000 characters and times include an offset. Non-success
terminal states use `kind: task_failed`. A validated
`200 {status: "enqueued"|"duplicate", entry_id}` marks the event sent; both statuses
are success. Event id and payload remain identical on every relay retry.

### Completion and delivery endpoints

All three POST endpoints use the existing `/v1` Bearer authentication.

1. **`POST /v1/chat/task-result`** — borg's runner sends
   `{model, sidecar_session_id, conversation: {external_id,type,name},
   event: {event_id,event_entry_id,task_id,task_version,kind,occurred_at,outcome},
   requester: {external_id,display_name}|null}`. `model` is checked against the
   token scope; the event is checked against the local ledger's task/version and
   origin. The saved requester and thread work even when requester is null.
   A cross-replica event lock serializes retries and the existing conversation
   lock protects a tools-disabled chat graph invocation (recursion limit 4, no
   playbook routing). One 45 s endpoint deadline covers registry refresh, locks,
   ledger reads, composition and reply writes, reserving the final 5 s for fallback
   persistence. A fixed runtime instruction requires
   one outcome message, task id and artifact citations; event text is quoted
   untrusted data. The response is `200 {action: "reply", content}`. Empty model
   text and composition failures use deterministic non-empty fallback text built
   from the saved event. Outcome replies are cached only after ledger origin
   validation succeeds; all outcome text and event metadata used for composition
   come from that ledger (the sidecar event entry id remains transport metadata).
   The first validated reply/fallback is cached without expiry; a retry returns
   the same content. Before validation succeeds, availability failures return a
   neutral non-empty message without claiming an outcome or writing a cache entry.
   After validation, a database outage still returns the saved outcome's fallback,
   or completed composition, but cannot guarantee durable caching. Authentication,
   malformed requests and mismatched/unknown events
   retain 4xx errors. Borg alone appends the canonical message; no append-turn
   call or direct Teams post is made here.
2. **`POST /v1/teams/agent-deliveries/claim`** — accepts
   `{sidecar_session_ids, wait_ms?: 60000, lease_ms?: 120000}`. `wait_ms` must be
   between 0 and 60000. Proxies `/memory/agent-deliveries/claim`, with a ten-second
   client timeout margin, returning `{deliveries: [{delivery_id,
   sidecar_session_id,terminal_entry_id,task_id,content,created_at}]}` unchanged.
3. **`POST /v1/teams/agent-deliveries/ack`** — accepts
   `{delivery_id, outcome: "sent"|"failed_retryable"|"failed_permanent",
   teams_message_id?, error?}` and proxies `/memory/agent-deliveries/ack`.
   The sidecar's acknowledgement object is passed through unchanged.

Claim and ack require a single authorized enabled tenant in the Bearer scope
(or the existing single-tenant configuration). A multi-tenant token is ambiguous
and receives 400; neither body accepts `tenant` or `model`. Sidecar errors retain
the existing sanitized proxy status handling.

**`GET /v1/teams/capabilities`** uses the same Bearer authentication and returns
`{"agent-task-v1": true}`. Deploy borg's phase B routes/runner before enabling
bridge claim/ack polling. End-to-end delivery requires that separate rollout.

## Rollout order

1. borg (sidecar): routes and runner ship inert; no session has `source_type: "teams_inbox"`
   until something enqueues.
2. team-agent: new endpoints unused until the bridge switches.
3. bridge: switch. Bridge rollback is a redeploy of the previous zip. For the task-event
   extension, first deploy Borg with the lane disabled and follow the reader compatibility
   restrictions above before enabling it or rolling Borg back.
