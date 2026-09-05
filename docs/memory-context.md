# Memory context contract (borg memory sidecar <-> team-agent)

Status: agreed design, 2026-09-02. Implemented on both sides in lockstep; team-agent falls
back to the legacy calls when the sidecar does not know the new routes yet.

## Purpose

Give team-agent, on every Teams turn, the situational awareness that borg's own retrieval phase
gives Sol's deliberation - without any new LLM call and without borg's deliberation, reflection,
closure or guard phases (those cost minutes; team-agent keeps generating with its own model):

1. recalled episodes scoped to what the current audience is allowed to know,
2. recent activity in the agent's other conversations ("recent life elsewhere"),
3. binding commitments applicable to the audience,
4. operator directives (rules the tenant's operators gave the agent) applicable to the audience.

Everything above already exists in borg as tables and repository reads (sessions, activity_events,
episode audience/origin_audience, commitments, creator_directives, listApplicable). What is
missing is (a) the sidecar write path does not populate sessions, audiences or activity events, so
every episode is stored as public within the tenant and recall is tenant-wide, and (b) there is no
read endpoint that assembles these sections for a caller.

## Identity model (borg concepts -> Microsoft Teams)

- Session = team-agent thread key, exactly the `session` value already sent on append-turn
  (`tenant::user::conversation` for personal chats, `tenant::shared::<type>::<conversation>` for
  group chats and channels). The sidecar derives a borg SessionId from it deterministically.
- Person entity: external id = the Teams user id (X-User-Id), display name = sender.display_name.
  append-turn already resolves this entity.
- Group entity (borg entity kind `group`): one per Teams groupChat/channel conversation. External
  id = conversation.external_id (the raw X-Conversation-Id); canonical name = conversation.name when
  known ("AI Ninjas"), otherwise a stable fallback derived from the type and id.
- Audience of a session: personal chat -> the person entity; groupChat/channel -> the group entity.
  Borg conversation_kind mapping: personal -> `dm`, channel -> `channel`, groupChat -> `thread`.
- Audience role: `operator` when the request marks the sender as a tenant operator
  (`sender.operator: true`), else `participant`. Operators are team-agent's tenant admins. Do not use
  borg's single-creator `borg_role`; authority for creating directives comes from the admin API token.

## Who may see what (hard rule, enforced by the sidecar)

visible-audience-set(caller) =
  { current audience entity }
  UNION, when the current audience is a person: every group entity in which that person has been
  observed as a speaker in this tenant (activity_events already records speaker + audience per event,
  so membership-by-observation is a single query).

An episode is visible iff it is `shared` (no origin audience; all pre-existing episodes are like
this) OR its origin_audience intersects the visible-audience-set. Non-visible episodes are dropped,
not merely labelled - team-agent's model is not trusted to withhold them. The same set scopes
recent_activity: a person hears about activity in their groups and in their own other chats, never
about other people's private chats; a group hears only about its own past.

## Write path: POST /memory/append-turn (extended, backward compatible)

Existing body: tenant, session, user, assistant, sender{external_id, display_name},
conversation{type, name}. New optional fields: conversation.external_id, sender.operator (boolean).

New behaviour when sender and conversation are present:
- ensure a sessions row for the session (source_type for team-agent, label, audience_label,
  audience_entity_id = audience entity, conversation_kind, audience_role) and touch it every append;
- stream entries carry `audience` = the audience entity so the extractor derives origin_audience
  from it (group turns become channel memory, personal turns become private to that person);
- record activity events: `user_contact` for the user entry (speaker = sender) and `borg_replied`
  for the assistant entry (speaker = self), with audienceEntityId and participant ids, so
  listRecentOtherActiveSessionEvents works for the read path.
Requests without sender/conversation keep today's behaviour exactly.

## Read path: POST /memory/context  (x-borg-token)

Request:
{
  "tenant": "...", "session": "<thread key>",
  "sender": {"external_id": "...", "display_name": "...", "operator": false},
  "conversation": {"type": "personal|groupChat|channel", "name": "...", "external_id": "..."},
  "query": "<recall query>",          // optional; required for the episodes section
  "limit": 8,                         // episodes cap, same bounds as /memory/recall
  "sections": ["audience","episodes","recent_activity","commitments","directives"]  // default: all
}

Response:
{
  "ok": true,
  "audience": {"entity_id": "...", "kind": "person|group", "name": "...", "role": "participant|operator"},
  "episodes": [ <same per-hit projection as /memory/recall, plus
                 "disclosure": {"class": "public|relationship_private|...", "origin_audience_names": [...]}> ],
  "hidden_episode_count": 0,
  "recent_activity": [ {"kind": "user_contact|borg_replied", "occurred_at": <epoch ms>,
                        "occurred_at_iso": "...", "relative_age": "12m ago",
                        "session": "<sidecar session id>", "conversation": {"type": "...", "name": "..."},
                        "participant_name": "...", "text": "Mateusz Pawlak contacted the agent 12m ago in group chat \"AI Ninjas\"."} ],
  "commitments": [ <same projection as GET /memory/commitments, filtered for this audience> ],
  "directives": [ {"id": "...", "kind": "response_policy|routing_instruction|disclosure_boundary|subject_fact|self_identity",
                   "render_mode": "content|boundary", "text": "...", "content_scope": "...", "priority": 0, "topic_tags": []} ],
  "degraded": false, "degraded_reason": ""
}

- episodes use the existing recall pipeline (audience-scoped via its audienceEntityId visibility gate
  where possible, plus the membership widening above), same deadline/degradation semantics as
  /memory/recall, no new embedding work.
- recent_activity: events from OTHER active sessions within a recency window (default 24 h,
  configurable), scoped by the visible-audience-set, newest first, capped (default 12).
- commitments: active commitments applicable to the audience (same rules as the existing GET).
- directives: creatorDirectives.listApplicable({currentAudienceEntityId, sessionRole,
  participantEntityIds}); render_mode "omit" entries are excluded; text = operational_directive /
  canonical_fact for "content", boundary_prompt for "boundary".
- Unknown sections -> 400. Missing query with "episodes" requested -> 400.

## Operator rules: /memory/directives (admin surface, x-borg-token)

- POST /memory/directives  body {tenant, kind, text, content_scope ("public"|"operator_only"|
  "allow_list"|"subject_only"|"all_except"), allowed_external_ids?, excluded_external_ids?,
  allowed_group_external_ids?, excluded_group_external_ids?, subject_external_id?, mention_policy?,
  priority?, topic_tags?} -> 201 {ok, directive}
- GET /memory/directives?tenant=  -> {ok, directives:[...]} active directives
- DELETE /memory/directives/<id>?tenant=  body {reason} -> revoke
These are manual directives (provenance: admin API). Extracting directives from operator chat with
an LLM is deliberately out of scope for this version.

## team-agent side

- Sender and conversation context (already resolved per request in the API) must reach the place
  where ambient memory is assembled for the model, in the same style as the existing per-request
  context variables.
- Per turn: the pre-model ambient block calls /memory/context with sections [episodes] + query;
  the request-level binding-rules block calls /memory/context with sections [audience,
  recent_activity, commitments, directives]. Each replaces one existing call (recall, commitments),
  so turn latency is unchanged.
- Rendering: episodes keep the current "[time; venue; participants] Title: narrative" line and gain
  the disclosure tag when private (e.g. "private to Mateusz Pawlak"); directives render under the
  binding rules as operator rules; recent_activity renders as a short "Elsewhere right now" block.
- Fallback: a 404 from /memory/context means an older sidecar -> use the legacy /memory/recall and
  /memory/commitments calls transparently. Breaker gating as for /memory/recall.
- sender.operator = true when the request user is a tenant operator (tenant configuration lists
  operator external ids; add it if no such notion exists).
- Admin passthrough for directives (list/create/revoke) following the existing admin/debug API
  pattern, so the console or curl can manage operator rules.

## Extension 2: observations, time scoping, venue recency, exclusions (2026-09-02, late)

Motivation: two measured gaps in the AI Ninjas group. (1) Teams only delivers group/channel
messages that @mention the bot, so human-to-human talk never entered the system; with RSC the
bridge will receive every message and must be able to record it WITHOUT generating a reply.
(2) A 16:45 discussion about Python vs TanStack was ingested and extracted into five episodes, yet
"pamiętasz dzisiejszą dyskusję o technologii?" at 23:21 did not surface them: the time cue was
ignored, semantic scores were compressed, and autonomous OUTCOME rollups occupied recall slots.

### Observations: POST /memory/append-turn with no assistant reply

- `assistant` becomes optional. A body with `user` but no `assistant` is an OBSERVATION: append only
  the user stream entry (sender, conversation, audience exactly as for a full turn), record a
  `user_contact` activity event, touch the session (message count +1), schedule ingestion as usual.
  Response shape unchanged (only the appended ids are present). Requests with both fields behave
  exactly as today.
- Optional `observed_at` (epoch ms) records a delayed observation's event time as entry metadata;
  the stream `timestamp` remains append time so ingestion cursors stay monotonic.
  The sidecar accepts it only when it is no earlier than five minutes before and no later than one
  minute after server receipt time; omitted metadata defaults episode occurrence to append time.
- `POST /memory/append-turn` with `assistant` but no `user` is a REPLY-ONLY record: append only the
  agent stream entry (conversation and audience exactly as for a full turn), record a
  `borg_replied` activity event with the self entity as speaker and actor, touch the session
  (message count +1), and schedule ingestion as usual. `sender` may be absent; a complete
  conversation identity is still required for the enhanced path. Requests with neither `user` nor
  `assistant` return 400.

### Group participant set on context requests

An unsolicited group reply may add `participants` to `POST /memory/context` as an ordered array of
`{"external_id": "...", "display_name": "...", "operator": false}`. Duplicate external ids are
removed by team-agent. The sidecar resolves these people as the current group recipient set for
directive applicability and visibility/exclusion checks; the group conversation remains the sole
audience, and participant entries cannot confer operator authority. A team-agent compatibility
retry after HTTP 400 removes this field once.

### Time scoping and exclusions on episodes (POST /memory/context and POST /memory/recall)

- `time_range: {"start": <epoch ms>, "end": <epoch ms>}` optional. Applies a strict filter on the
  episode's occurred_at for the episodes section. If the strict search returns no hits, the sidecar
  retries once WITHOUT the range and sets `episodes_time_range_fallback: true` in the response, so
  the caller always makes one request. The venue_recent section ignores time_range (it has its own).
- `exclude: {"title_prefixes": [..], "narrative_markers": [..]}` optional (each up to 8 strings,
  case-sensitive substring/prefix match). Applied BEFORE `limit`: the sidecar over-fetches
  (3x limit, bounded) and drops matching episodes so every returned slot is a real candidate.
  Applies to both the episodes and venue_recent sections. team-agent sends
  `{"title_prefixes": ["OUTCOME rollup"], "narrative_markers": ["OUTCOME fp=", "decision="]}` for
  chat surfaces and keeps its client-side filter only as a safety net.

### New section `venue_recent` (POST /memory/context)

- Requested with `sections` containing "venue_recent" plus `venue_since` (epoch ms, required) and
  optional `venue_limit` (default 12, max 50). Returns the episodes extracted from the CURRENT
  session (same sidecar session id as the request) with occurred_at >= venue_since, newest first,
  same projection as the episodes section (incl. disclosure), exclusions applied, no semantic query
  needed. Because the current session's own history is always visible to itself, no widening is
  involved. team-agent sends venue_since = start of today in the tenant timezone and renders the
  block as "Earlier in this conversation today", deduplicated against recalled episodes by id.
- Response key: `"venue_recent": [ ... ]`. Missing/invalid venue_since with the section requested
  -> 400.

### team-agent side (summary; details in docs/group-presence.md)

- `POST /v1/chat/observe` records unmentioned group/channel messages into the shared thread history
  and into borg (observation append), then decides whether to chip in (Layer 2). Only groupChat and
  channel conversation types are accepted.
- Temporal cues in the user's latest message ("dzisiaj", "dziś", "dzisiejsz*", "wczoraj", "przed
  chwilą", "chwilę temu", "rano", "po południu", "wieczorem", "w tym tygodniu", "today",
  "yesterday", "this morning", "earlier today", "just now", "this week") are detected
  deterministically and mapped to a time_range in the tenant timezone. No model call.

team-agent compatibility: until a sidecar accepting the reply-only shape is live, team-agent treats an HTTP 400 on it as a one-log compatibility skip and never retries it as a full turn; a 400 caused by `participants` is retried once without the field.

## Implementation notes (sidecar)

- A syntactically valid, existing Borg `EntityId` in `StreamEntry.audience` is a stable audience
  handle. The episodic extractor uses that entity directly, retains label resolution for every
  other audience string, and uses the resolved entity's canonical name in prompts.
- People use external-id source `team-agent.sender`; group/channel conversations use the separate
  source `team-agent.conversation`. Enhanced append requires `conversation.external_id` for
  `groupChat` and `channel`. If it is absent, append uses the exact legacy path (no session,
  audience or activity enrichment) so old team-agent builds remain compatible. A group is never
  keyed by its display name. The new `/memory/context` route has no legacy mode and returns 400
  when a group/channel request omits this external id. The legacy append parser ignores
  `sender.operator`, including a non-boolean value, whenever identity is incomplete; a complete
  enhanced identity validates it strictly and returns 400 unless it is boolean or absent.
- Session mapping is `personal -> dm`, `groupChat -> thread`, `channel -> channel` with source type
  `team_agent`, source external id equal to the raw caller session string, and
  `sender.operator -> audience_role=operator`. A `borg_replied` event records the Borg self entity
  as both speaker and actor.
- `relative_age` uses Borg's compact formatter (`12m ago`). The activity `text` field is the
  complete line team-agent renders verbatim.
- `hidden_episode_count` counts results removed by the final in-memory defense check after the
  repository visibility gate. It is normally zero; it is not a count of all hidden matching
  episodes in the tenant bank.
- `/memory/context` may create/update the sender and group entities and ensure the session on a
  first turn. It performs this in a short exclusive identity phase, then releases the writer queue
  before shared repository reads and recall.
- Enhanced append commits the JSONL entries first, then runs session ensure/touch and both activity
  records in one SQLite transaction. That awareness projection is best-effort: if it rolls back,
  the sidecar logs the failure, emits `sidecar.append_projection.degraded`, and still returns the
  normal 200 append response so a transport retry cannot duplicate the durable turn. The stream is
  the source of truth; a projection failure can temporarily reduce situational awareness until an
  operational repair/backfill. Reapplying a projection with the same source stream ids is
  idempotent and does not increment the session message count again.
- Membership widening applies only to episodes and recent activity. Commitments remain scoped to
  the current audience with the existing `GET /memory/commitments` semantics.
- The token-authenticated sidecar passes an explicit trusted-tenant-operator capability to creator
  directive applicability. This capability is not inferred from or written to `borg_role` and is
  not used by cognition callers. Context includes only activation-active evaluations whose render
  mode is not `omit`. In a group/channel context, the current group audience is the allow-list
  authorization handle for the room; the sender and group remain the observed recipient set for
  exclusions. Thus a group-only allow applies without separately allowing its sender, while an
  excluded group or any excluded person present suppresses it fail-closed with the existing
  `group_contains_excluded_entity` semantics. Context `participants` extend that recipient set and
  its allow-list authorization candidates without replacing the group audience.
- Directive administration creates a stable per-tenant admin API entity and admin session, appends
  a structured `internal_event` provenance entry, then queues the directive. Person ids in
  `allowed_external_ids`, `excluded_external_ids`, and `subject_external_id` resolve only through
  `team-agent.sender`; optional `allowed_group_external_ids` and `excluded_group_external_ids`
  resolve only through `team-agent.conversation`. Unknown or ambiguous ids fail closed with 400.
  The entity is kind `abstract` at external handle `memory-sidecar.admin/operator-api`; its
  operator-role `dm` session uses source type `memory_sidecar` and source external id
  `memory-sidecar::admin-api`. Creation records a
  `memory_sidecar.operator_directive_queue_requested` event before queueing; revocation records a
  `memory_sidecar.operator_directive_revoke_requested` event before the update. If either SQLite
  mutation fails, a second `memory_sidecar.operator_directive_queue_failed` or
  `memory_sidecar.operator_directive_revoke_failed` event references the request provenance stream
  entry, so the append-only audit trail does not claim an uncompleted mutation.
- Directive defaults are: activation scope `same_as_disclosure`, denied-audience behavior
  `omit`, `boundary_prompt` equal to the submitted text, subject kind `borg_self` for
  `self_identity`, `entity` for `subject_fact` (which therefore requires `subject_external_id`),
  and `system` otherwise; mention policy defaults to `answer_if_asked`.
- Recent activity defaults to a 24-hour window and 12 rows. They are configurable through
  `BORG_MEMORY_RECENT_ACTIVITY_WINDOW_MS` and `BORG_MEMORY_RECENT_ACTIVITY_LIMIT`, respectively,
  as well as handler options.
- Membership widening is a disclosure-only audience-set capability propagated through Lance,
  indexed SQLite and the final in-memory visibility check. The current audience remains separate
  for social ranking. The implementation never uses unrestricted `crossAudience` recall followed
  by filtering, and cognition recall receives no audience-set option.
- Request-level idempotency is unchanged from the legacy append-turn: a client retry after a lost
  response appends a second turn to the stream and, consistently, a second awareness projection
  (team-agent retries only on transport failures, never on a received response). Replays of the
  same stream entry ids are idempotent and do not double-count; there is no crash-repair pass that
  re-derives projections from the stream.
- An append without `assistant` is an observation. With complete Teams identity it uses the same
  sender, audience, conversation and best-effort atomic awareness projection as a full turn, but
  records only `user_contact` and increments the session message count once. Incomplete identity
  keeps the corresponding legacy behavior (one user entry and no awareness projection). The stream
  entry always retains the writer's append-time `timestamp`, preserving cursor order;
  `observed_at` is optional entry metadata and must be no earlier than five minutes before and no
  later than one minute after server receipt time or the sidecar returns 400. Extraction uses it as the
  episode's occurred-at time, while session/activity projections use append time. Ordinary
  two-entry turns keep their existing writer timestamps and serialized shape.
- A reply-only append has `assistant` but no `user`. It appends one `agent_msg`, records only a
  `borg_replied` activity whose speaker and actor are the Borg self entity, and increments the
  session message count once through the same best-effort atomic projection. A group/channel
  reply-only append can use `conversation.external_id` as its enhanced audience identity without a
  sender; a personal enhanced append still requires the person handle in `sender`. Otherwise it
  keeps the corresponding legacy one-entry behavior without awareness projection. A request with
  neither message field returns 400.
- `/memory/context` accepts up to 32 strict participant objects. It collapses duplicate external
  ids in first-seen order, resolves each remaining person through `team-agent.sender` during the
  exclusive identity phase, and merges the resulting entity ids into directive recipients and
  allow-list authorization candidates. These ids never enter episode or recent-activity audience
  capabilities, so personal participants cannot widen memory visibility and the group remains the
  sole group/channel audience. Participant `operator` values are validated as booleans but ignored;
  only `sender.operator` can confer the trusted-operator/session-role authority.
- `time_range` and `venue_since` accept integer epoch milliseconds; a time range is inclusive and
  requires `start <= end`. Strict episode scoping uses the public `occurred_at` value
  (`episode.start_time`). If no visible strict hit survives, the retrieval pipeline reuses the same
  prepared recall expansion for one unscoped pass, so fallback does not add an LLM call and retains
  the original disclosure audience capability. The `episodes_time_range_fallback` key is emitted
  only when that pass occurs. Fallback eligibility is decided before caller exclusions, so an
  in-range episode suppressed by `exclude` does not widen the search silently.
- Episode exclusions are case-sensitive protocol matching: title prefixes use prefix matching and
  narrative markers use substring matching. Only requests that supply `exclude` fetch up to three
  times the requested response limit (bounded at three times the configured endpoint maximum),
  apply exclusions, then take the requested limit; requests without it retain the legacy candidate
  budget. Over-fetched retrieval candidates are read without accounting mutations, and only the
  final non-excluded, non-overflow episodes actually returned are recorded in `retrieval_log`,
  episode stats, and heat inputs. Exclusion drops are not included in `hidden_episode_count`, which
  retains its disclosure-defense meaning above.
- `venue_recent` is opt-in and therefore is not added to the default context sections, preserving
  existing requests that omit `sections`; it requires `venue_since`, defaults `venue_limit` to 12,
  and caps it at 50. The SQLite episode index stores source stream ids and joins them to
  `stream_entry_index`, so this section admits an episode only when it has indexed provenance and
  every source entry belongs to the current sidecar session. Mixed-session consolidations and
  missing-index provenance fail closed. Results order by `start_time` newest first and do not run
  semantic retrieval. The migration invalidates the old Lance backfill marker once so existing
  episode provenance is indexed. Venue entries use the same public metadata and disclosure
  projection as recalled episodes; because this lane has no relevance score, `score` and
  `raw_score` are both `0`.

## Implementation notes (team-agent)

- A `/memory/context` request is made only when tenant, session, sender external id and operator
  boolean, and conversation type and external id are all present. Legacy/OpenWebUI requests with
  incomplete Teams identity call the corresponding legacy endpoint directly, unless assistant
  policy has suppressed commitments and recent activity for that turn; the binding legacy endpoint
  has no other section to return, so it is skipped in that case.
- Only a structured HTTP 404 from `/memory/context` selects the legacy fallback. Transport errors,
  malformed responses and every other HTTP status produce the section-specific unavailable marker.
- Optional `ok` and `degraded` control fields must be booleans, `degraded_reason` must be a string,
  and `hidden_episode_count` must be a non-negative integer. `ok: false` is treated as a backend
  failure. A degraded binding response emits `binding_context_degraded` tracing and injects a
  partial-availability marker alongside any valid surviving rules.
- Each response section remains optional, but a present `audience` must be an object and present
  `episodes`, `recent_activity`, `commitments`, and `directives` sections must be arrays. Commitment
  entries require string `directive` and `enforcement_class`; directive entries require string
  `text` and `render_mode`. A malformed entry makes the binding context unavailable rather than
  presenting an empty rule set. Only directive render modes `content` and `boundary` are rendered;
  other string modes are skipped. Malformed recent-activity entries are skipped.
- `mention_policy` has no schema in this contract, so the admin proxy accepts and forwards any JSON
  value, including an explicit JSON `null`. Directive priority is a strict JSON integer. All other
  directive fields are validated against the shapes above and unknown request fields are rejected.
- Directive ids accepted by team-agent's DELETE proxy and sidecar client are limited to
  `[A-Za-z0-9_-]+`; unsafe path segments (including encoded dot segments) are rejected before a
  sidecar request.
- A non-public episode without usable `origin_audience_names` is labelled `private`; with names it is
  labelled `private to <names>`. Recent activity consumes the sidecar-provided `text` field. Unknown
  response fields are ignored.
- A successful no-content sidecar response (HTTP 204) is normalized to `{}` by the shared HTTP client.
- Upstream HTTP error logs contain only the request path, status, response size, and available request
  correlation headers; response bodies are never logged.
- Both Teams entry points fetch request-level context for every identity-complete turn, including
  assistant capability questions and turns with precollected Assets evidence, so operator directives
  are never suppressed by assistant runtime policy. Those assistant turns filter commitments and
  recent activity only after the full response is retrieved; operator directives remain injected.
  A single policy helper makes that decision from the same assistant-mode, raw-user-text, and
  precollected-Assets-context inputs in both entry points.
- The existing `memory.commitments_enabled` deployment switch gates the whole request-level context
  call (commitments, directives, audience and recent activity). Episodic context remains controlled by
  `memory.base_url` as before.
- The optional in-process Teams bot route (team_agent/teams_bot) uses the same policy helper but
  never precollects Assets evidence itself, so its precollected-Assets input is always empty; it
  only diverges from the API on turns the API answers from precollected Assets. In production the
  route is enabled (bot credentials are configured) but is not the Teams ingress: traffic arrives
  through services/teams_bridge and the API, and no /api/messages requests have been observed.
- Extension 2 observation requests add an optional strict boolean `sender.bot`
  to the bridge-to-team-agent message shape for the newest-bot silence guard.
  It is transport metadata only: the append-turn sender builder drops it, so
  the sidecar still receives only `external_id`, `display_name`, and the
  server-computed `operator` sender fields.
- Ordinary chat episodic context requests include `episodes` and `venue_recent`, start-of-
  day `venue_since`, `venue_limit: 12`, and the documented exclusion object.
  Assistant capability/meta turns and turns with precollected Assets evidence
  request `venue_recent` alone so same-conversation context is still present
  without re-enabling semantic episodic recall.
  A structured HTTP 400 receives one retry against the same endpoint with every
  Extension 2 field removed (and `venue_recent` removed from sections). Enhanced
  legacy recall similarly retries `/memory/recall` once without `time_range` or
  `exclude`; a successful compatibility retry produces no unavailable marker.
  For a venue-only request that retry uses `sections: ["audience"]`, a valid old-
  contract no-query shape that cannot accidentally restore suppressed episodes.

## Extension 1 non-goals

- No new LLM calls on the request path; no deliberation/reflection/closure phases.
- Existing episodes stay shared within the tenant; only new turns get audience scoping.
- Bridge (services/teams_bridge) needs no change: it already sends sender + conversation.

## Extension 3 — entity-aware, source-backed recall

Extension 3 supersedes the strict context `time_range` membership and zero-result fallback
semantics described above. It is additive on the wire and changes only the sidecar/team-agent
disclosure path; Borg cognition retrieval does not opt in.

### Request extension

`POST /memory/context` accepts optional `entity_terms: string[]`. The array contains at most 32
trimmed, non-empty strings of at most 128 characters each. Team-agent sends bounded, case-insensitively
deduplicated ranking hints in this order: matching canonical configured people and system names, the
sender and observed participants, then raw names and identifiers found in the current message. The
configured-match bucket contains at most eight terms, and each message token contributes at most two
canonical people. Fuzzy person matching requires at least four letters when only three prefix
characters are shared and ignores a token with more than four characters after its actual shared
prefix. Entity terms activate Borg's existing configured entity attention weight. They do not create
or widen an audience, prove identity, change visibility, or authorize disclosure. An omitted field
preserves the previous retrieval behavior.

### Episode source messages

Each episode returned in the `episodes` section of `POST /memory/context` has additive
`source_messages` data. Legacy `POST /memory/recall` never emits this field.

```json
{
  "source_messages": [
    {
      "id": "strm_...",
      "kind": "user_msg",
      "occurred_at": 1770000000000,
      "speaker_name": "Jacek",
      "text": "the verbatim source prefix"
    }
  ]
}
```

Only narrative `user_msg` and `agent_msg` entries with string content are eligible. Entries retain
their citation-chain order. At most three entries are emitted per episode and each `text` is at most
180 characters. Text is the original prefix: the sidecar does not collapse whitespace, summarize it,
or append an ellipsis. `speaker_name` is optional; `occurred_at` uses `observed_at` when present and
otherwise the stream timestamp.

The sidecar projects source messages only after the episode passes the final visible-audience-set
gate. A source message inherits its episode's visibility and disclosure; invisible episode content is
dropped and is never returned with a label. Source entries are reused from the retrieval pipeline's
already-resolved citation chain, so this extension adds no source DB read, embedding, or LLM call.

### Recent-activity excerpts

Each `recent_activity` event may have an additive `excerpt` string containing the original prefix of
the event's source `user_msg` or `agent_msg`, capped at 180 characters without whitespace rewriting or
an ellipsis. Source IDs are hydrated only after `listRecentVisibleOtherSessionEvents` has applied the
same exact visible-audience-set and current-session exclusion as the event list. Hydration is indexed
only, runs once per request over the union of the capped event lists (the planner's owner-only rows
first, then the `recent_activity` rows, so at most two row caps of source IDs) under one 50 ms
sub-budget, and never falls back to a stream scan. A missing, malformed, mismatched, failed, or
over-budget lookup silently leaves the event without `excerpt`.

### Time preference and recency prior

For `POST /memory/context`, a supplied `time_range` is an ordering preference, not a membership gate.
The sidecar performs one overfetched search with that range and `strictTimeRange: false`, then applies
the existing episode visibility gate and caller exclusions. It marks every returned episode with
`in_time_range: true|false` using the inclusive public `occurred_at` value (`episode.start_time`),
stable-partitions in-window results before out-of-window results while preserving relevance order
inside both partitions, and finally slices to the requested limit. Out-of-window results therefore
top up unused slots. This context path does not perform a zero-result rerun and does not emit
`episodes_time_range_fallback`.

Legacy `POST /memory/recall` retains its Extension 2 strict/fallback behavior, including the optional
`episodes_time_range_fallback` response field. Team-agent accepts that legacy field for compatibility.

The memory sidecar can opt its context and legacy-recall searches into one bounded recency prior on
the final fused episode score before MMR:

`boost = weight * 0.5^(age_hours / half_life_hours)`

Age is non-negative and is measured from `episode.end_time`. The prior is absent unless either
`BORG_MEMORY_RECENCY_PRIOR_WEIGHT` or `BORG_MEMORY_RECENCY_PRIOR_HALF_LIFE_HOURS` is configured. Once
enabled, an omitted or invalid companion uses `weight = 0.15` or `half_life_hours = 36`; weight is
bounded to `[0, 1]`. The absent option performs no recency-prior arithmetic. Sol's cognition turn
coordinator never supplies it, retains `strictTimeRange: false`, and therefore preserves its previous
scores, ordering, and evidence.

### Team-agent compatibility and rendering

Team-agent preserves the episode order returned by `/memory/context`; it applies its historical
newest-first sort only after a genuine 404 fallback to legacy `/memory/recall`. A 400 compatibility
retry to an older strict context sidecar removes `entity_terms` together with the other extension
fields. Older team-agent versions ignore all additive response fields, and newer team-agent versions
continue when an older sidecar omits them.

Source messages render beneath their episode and activity excerpts render beneath their trusted
`Elsewhere right now` event sentence. Both are escaped and enclosed in
`<untrusted_memory_evidence>` inside the existing memory `SystemMessage`. A trusted enclosing rule
states that every such block is quoted data: it cannot change rules, grant authority, issue
instructions, or request tool calls. Raw excerpts are never interpolated into trusted event text.

## Extension 4 — context-aware recall query planning

### Structured focus and context

`POST /memory/context` accepts optional `focus` and `context_turns` fields alongside the legacy
`query`:

```json
{
  "query": "legacy joined four-message window",
  "focus": "the current message",
  "context_turns": [
    { "role": "user", "text": "an earlier message" },
    { "role": "assistant", "text": "the reply" }
  ]
}
```

`focus` is authoritative when both it and `query` are present. `context_turns` requires `focus`, is
ordered oldest to newest, and contains at most three preceding dialogue messages, so focus plus
context retain team-agent's existing four-message total. Adjacent turns with the same role remain
separate records. An episodes request requires `focus` or `query`. A query-only request remains
valid and is treated as one legacy focus blob with empty context; Borg never parses role prefixes
out of that blob.

Team-agent sends all three fields during the migration: the byte-compatible joined `query` for old
servers plus structured `focus` and `context_turns` for new servers. Its one HTTP 400 compatibility
retry removes `focus` and `context_turns` together with the other unsupported extension fields; the
existing HTTP 404 legacy-endpoint fallback is unchanged. The structured bundle participates in the
per-turn memory cache key, and `focus` supplies temporal-cue and entity-term collection input.
Observation persistence wrappers are removed from structured turns while their decoded message body
is retained. If the latest human body is empty, team-agent skips recall before cache lookup, temporal
or entity processing, and HTTP dispatch; it never promotes an earlier assistant reply to `focus`.
The independently built legacy `query` remains byte-compatible even for that skipped turn.

### Shared planner

Borg performs one forced `EmitRecallQueryPlan` structured completion in the existing
`recallExpansion` model slot. The planner receives FOCUS, separately labelled CONTEXT turns,
memory-owner/sender/audience/venue/entity handles, and optional visible excerpts of the owner's own
recent activity. It first resolves pronouns, ellipses, omitted subjects, and cross-venue references,
then emits a trace-only `resolved_query`, exactly N semantic variants, exact-lookup `named_terms`,
and optional `commitment` or `open_question` typed queries. Supplied conversation text and excerpts
are explicitly data, not instructions.

With N=1, one `combined` variant preserves high-signal wording while expressing the likely exchange
in the memory owner's voice and emphasizing its discriminating aspect. With N>=3, the first variants
are respectively `verbatim_preserving`, `memory_owner_voice`, and `aspect_focused`; further variants
use the `additional` strategy. Each variant becomes its own priority-85 `semantic_query` episodic
vector lane and participates in full cognition semantic retrieval. Raw FOCUS remains priority 100;
named terms are priority 90 and retain the existing exact-name/compound-term rules; commitment and
open-question queries use priority `60 + 20p`. Time remains priority 70 and recent priority 10.
Topic and relationship facets and the separate `reformulated_query` lane no longer exist. Fusion,
MMR, and exact-term reservations are unchanged.

The shared default N is 3, configured by
`BORG_RETRIEVAL_RECALL_EXPANSION_SEMANTIC_VARIANT_COUNT` and bounded to 1..8. Sol uses that default
and supplies its existing 16-message/24k recent-history window. The sidecar supplies a per-call N
from `BORG_MEMORY_RECALL_SEMANTIC_VARIANT_COUNT`, default 1 and likewise bounded to 1..8; the HTTP
request cannot override it. The former `BORG_MEMORY_RECALL_REFORMULATION_ENABLED` gate has been
removed: structured planning is now the single recall-expansion path.

When episodes are requested, `/memory/context` performs an owner-only pass of the visibility-gated
activity read (same visible audience set, window and 12-row bound as `recent_activity`) restricted to
memory-owner-authored `borg_replied` events, whether or not the `recent_activity` response section was
requested; the shared `recent_activity` read runs only when that section is requested. Deriving
planner rows from the shared list starved the planner on busy group days, because the 12 newest
visible rows were all `user_contact` messages. Only owner rows with successfully hydrated `agent_msg`
excerpts (180 characters) enter planner context, with their venue and time labels. Both reads share
one excerpt hydration pass with the owner rows hydrated first, and neither widens group visibility. Legacy `/memory/recall` supplies only the owner
handle and preserves its one-planner-completion property across the strict time-range fallback.

An invalid or failed plan is not retried. Borg reports a `recall_expansion` degradation and continues
with raw FOCUS, exact supplied handles, time, and recency lanes. Trace counts are always safe;
resolved text, variants, named terms, typed queries, routed intents, FOCUS/CONTEXT, handles, excerpts,
and the `retrieval.started` query require payload tracing.
