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
  `group_contains_excluded_entity` semantics.
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
  only diverges from the API on turns the API answers from precollected Assets. That route is
  disabled in production (the deployment carries no bot connection credentials), where Teams
  traffic arrives through services/teams_bridge and the API.

## Non-goals

- No new LLM calls on the request path; no deliberation/reflection/closure phases.
- Existing episodes stay shared within the tenant; only new turns get audience scoping.
- Bridge (services/teams_bridge) needs no change: it already sends sender + conversation.
