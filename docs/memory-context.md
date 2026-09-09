# Memory context contract (borg memory sidecar <-> team-agent)

Status: current coordinated context contract. Recall sections require `focus` and `context_turns`;
other sections can omit them. Clients make one request and report failures without downgrading.

## Purpose

Give team-agent, on every Teams turn, the situational awareness that borg's own retrieval phase
gives Sol's deliberation. Episodic recall reuses cognition's episodes-only projection; team-agent
keeps generating with its own model. A separate identifier-only reply guard makes no LLM call:

1. globally recalled tenant episodes carrying disclosure labels for audience-aware reasoning,
2. recent activity in the agent's other conversations ("recent life elsewhere"),
3. binding commitments applicable to the audience,
4. operator directives (rules the tenant's operators gave the agent) applicable to the audience.

The sidecar populates sessions, audiences and activity events on the write path and assembles
these sections through `/memory/context`. Sol's full turn pipeline keeps its existing behavior.

## Identity model (borg concepts -> Microsoft Teams)

- Session = team-agent thread key, exactly the `session` value already sent on append-turn
  (`tenant::user::conversation` for personal chats, `tenant::shared::<type>::<conversation>` for
  group chats and channels). The sidecar derives a borg SessionId from it deterministically.
- Person entity: external id = the Teams user id (X-User-Id), display name = sender.display_name.
  append-turn already resolves this entity.
- Group entity (borg entity kind `group`): one per Teams groupChat/channel conversation. External
  id = conversation.external_id (the raw X-Conversation-Id); canonical name = conversation.name when
  known ("Example Group"), otherwise a stable fallback derived from the type and id.
- Audience of a session: personal chat -> the person entity; groupChat/channel -> the group entity.
  Borg conversation_kind mapping: personal -> `dm`, channel -> `channel`, groupChat -> `thread`.
- Audience role: `operator` when the request marks the sender as a tenant operator
  (`sender.operator: true`), else `participant`. Operators are team-agent's tenant admins. Do not use
  borg's single-creator `borg_role`; authority for creating directives comes from the admin API token.

## Who may see what

Episodes and recent activity use **labels instead of audience-based dropping**. Cognition recall
is global within the current tenant. Every returned episode and activity row carries a disclosure
class, origin audience names and private-to names. The booleans `private_to_current_sender` and
`private_to_current_audience` report membership in the private-to set, resolved by Borg against
entity IDs for this request. Display names never determine membership, so two people with identical
names remain distinguishable. Only names and booleans enter the model's disclosure annotations;
entity IDs do not. These membership flags do not themselves authorize disclosure.
A private-chat memory can inform a group turn; recall itself does not authorize
sharing its contents or revealing that a private memory exists. The model uses the labels, current
audience authorization, and creator/operator context to decide what it may disclose and whether
information is already common ground.

When `episodes` is requested, `disclosure_guidance` contains Borg's canonical
`MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL` verbatim. Team-agent renders it above the episodes, including
in cached prompts, while preserving partial-degradation markers. It keeps the older recalled-memory
preface only when a sidecar omits the guidance. `hidden_episode_count` remains for compatibility and
is always zero for episodes; exclusions, ranking, limits and abstention still apply.

`recent_activity` uses Sol's `listRecentOtherActiveSessionEvents`: other active sessions, a time
window and a row limit, without an audience predicate. Sol's `recentLivedExperienceDisclosureLabel`
labels each row `self_private`, with its origin audience in the private-to set. The sidecar reuses
the episode disclosure projector, and team-agent reuses the episode annotation formatter. Old
sidecars that omit activity disclosure fields still render normally.
`venue_recent` means "what happened recently in this venue": it is venue-scoped by definition,
not an audience-based privacy filter over cross-session recall. Its existing query and provenance
checks are unchanged. Commitments and directives retain their existing applicability rules.

Before emitting an inbox reply or a non-streaming completion, team-agent calls `/memory/guard-reply`
with the returned `context_id`. The shared Borg runner checks internal identifiers without a closure
audit or an LLM call. This structural check does not decide semantic disclosure authorization; that
remains the model's responsibility under the supplied guidance. Streaming completions are outside
this guard integration.

## Write path: POST /memory/append-turn

Required identity: `tenant`, non-empty `session`, structured
`sender{external_id, display_name, operator}`, and `conversation{type, name, external_id}`.
Sender handles and display names and conversation external ids must be non-empty strings;
`sender.operator` must be a boolean. Conversation type is `personal`, `groupChat`, or `channel`,
and its name must be a string. This contract applies to full turns, observations and reply-only
appends, as well as `/memory/context`. Missing or invalid identity returns 400 with a
field-specific message before any tenant state is accessed.

Every append:

- ensure a sessions row for the session (source_type for team-agent, label, audience_label,
  audience_entity_id = audience entity, conversation_kind, audience_role) and touch it every append;
- stream entries carry `audience` = the audience entity so the extractor derives origin_audience
  from it (group turns become channel memory, personal turns become private to that person);
- record activity events: `user_contact` for the user entry (speaker = sender) and `borg_replied`
  for the assistant entry (speaker = self), with audienceEntityId and participant ids, so
  listRecentOtherActiveSessionEvents works for the read path.

`POST /memory/remember` supports two distinct request shapes. The existing role-run outcome
writer accepts `{tenant, content, author?}` without transport identity and appends/extracts in the
tenant's default session. The consent-backed `scope: "tenant"` shape below requires transport
identity and source handles, and creates a separate authorized fact. Unscoped requests preserve the
legacy parser: `author: null` is accepted, and extra fields such as `session` are ignored. Any
request containing `scope` takes the strict consent parser; an invalid or null scope returns 400.

## Consent-backed write: POST /memory/remember (x-borg-token)

```json
{
  "tenant": "acme",
  "scope": "tenant",
  "session": "<original Teams thread key or sidecar session id>",
  "sender": { "external_id": "marcin", "display_name": "Marcin", "operator": false },
  "conversation": { "type": "personal", "name": "Marcin", "external_id": "<conversation id>" },
  "request_id": "<stable tool invocation id>",
  "content": "Marcin będzie na urlopie od 14 do 18 września 2026.",
  "source_episode_ids": ["ep_aaaaaaaaaaaaaaaa"],
  "source_message_ids": ["strm_aaaaaaaaaaaaaaaa"],
  "authorization_message_ids": ["strm_aaaaaaaaaaaaaaaa"]
}
```

The authenticated caller supplies `tenant`, `sender`, `conversation`, `session` and the stable
`request_id` from transport/tool context. Do not let the model impersonate another sender by
choosing those fields. The sidecar token authenticates the integration; Borg resolves the supplied
external sender handle against that tenant's entity registry. “The team” means **the entire tenant**,
including other members' private chats, not just the current room and not other tenants.

`content` is a proposed fact (1–4000 characters), not an assertion of consent. `request_id` is
1–256 characters. `source_episode_ids` is optional (at most 16). `source_message_ids` and
`authorization_message_ids` each require 1–32 Borg stream IDs; the latter must be a subset of the
former. IDs are deduplicated. Use IDs returned by `/memory/append-turn` (`entries[].id`), inbox
enqueue (`entry_id`), or recalled `source_messages[].id`; these are not Teams external message IDs.
Persist/ingest the speaker's original message before invoking this operation. Do not synthesize
a user message claiming consent to satisfy this contract.

Borg requires active, available source records. Each designated authorization message must be a
fully ingested `user_msg` by the resolved speaker in the specified session. A single message can
contain both the fact and its consent; otherwise cite both. Existing episodes are optional context,
not a substitute for the authorizing message. One structured extraction call on
`BORG_MODEL_EXTRACTION` interprets the fact and explicit tenant-wide consent in the source language
(one additional attempt on malformed structured output). Ordinary private disclosure, hearsay about
consent, room-only permission, and the tool's own proposed content do not establish authorization.
The model receives source timestamps and the configured `retrieval.recallPlannerTimeZone` so relative
dates are anchored to the source, not the later remember call. It returns only the authorized fact,
title, tags and confidence. A denial writes nothing.

Success is HTTP 200:

```json
{
  "ok": true,
  "episode_id": "ep_bbbbbbbbbbbbbbbb",
  "authorization_entry_id": "strm_bbbbbbbbbbbbbbbb",
  "duplicate": false,
  "disclosure": { "class": "public", "scope": "tenant" },
  "guidance": "Team-public within this tenant by Marcin's explicit authorization. ...",
  "authorization": {
    "scope": "tenant",
    "request_hash": "<SHA-256 of normalized input>",
    "episode_id": "ep_bbbbbbbbbbbbbbbb",
    "speaker_entity_id": "<resolved Borg entity id>",
    "speaker_name": "Marcin",
    "source_episode_ids": ["ep_aaaaaaaaaaaaaaaa"],
    "source_message_ids": ["strm_aaaaaaaaaaaaaaaa"],
    "authorization_message_ids": ["strm_aaaaaaaaaaaaaaaa"],
    "fact": "Marcin będzie na urlopie od 14 do 18 września 2026.",
    "title": "Urlop Marcina",
    "tags": ["Marcin", "urlop"],
    "confidence": 1,
    "authorized_at": 1788940800000
  }
}
```

The original private episode remains intact. Borg appends an indexed, fsync'd authorization receipt
and creates a new episode with `shared: true` and no private audience. Private source episode IDs
remain in lineage and receipt metadata; the new episode's citation chain points only to the receipt
containing the authorized fact. It does not quote the original private conversation. Both Sol and
the sidecar recall the new episode globally with disclosure class `public` and authorization guidance
in its narrative. A consolidation with private material keeps its private label and leaves the
public source fact independently recallable. This also applies to already-consolidated records;
no re-extraction or rewrite of the private original is needed.

Context responses add `sharing_authorizations: [{scope: "tenant", speaker_name, fact,
disclosure: {class: "public", scope: "tenant"}, guidance}]` from hydrated receipts, including
receipts inherited by a consolidation. For exactly one fact, `sharing_authorization` is an alias
for that item. No authorization field is emitted without a valid hydrated receipt (the server
does not explicitly emit `null`). Each authorization applies **only to its `fact`**, never to an
entire merged narrative: the episode's own disclosure label remains authoritative for that
narrative. Multiple receipts produce separate facts, not a combined permission.
The private-memory existence restriction does not apply to this public fact; the original private
conversation and unrelated details retain their restrictions. Public does not mean already known
to the current audience.

Library/tool integrations use `borg.episodic.rememberForTenant` directly with the equivalent
camel-case input and resolved session/speaker IDs. It uses the same operation and extraction slot
as this HTTP route; an option-A tool integration does not need a second consent implementation.

Retry the identical request with the same ID after a transport/storage failure. Idempotency is
scoped to tenant bank, session and authorizing speaker. A retry returns the same IDs with
`duplicate: true`; a stored receipt resumes materialization after an embedding/storage failure
without another consent call. It never resurrects an archived episode. Existing `/memory/forget`
can archive the public record; it does not retract replies already delivered. As with other stream
writes, an index-update failure after a committed append triggers automatic session repair. If
repair succeeds, the append succeeds normally. If repair fails, the writer marks the session
poisoned and throws `STREAM_INDEX_POISONED`; the next append retries repair before writing.

Errors: 400 for malformed identity/provenance or missing/inactive/wrong-speaker sources;
422 `MEMORY_REMEMBER_NOT_AUTHORIZED` when the sources do not establish permission;
409 `MEMORY_REMEMBER_CONFLICT` for a reused request ID with different input;
500 for model/storage failures (503 for an unavailable embedding bank). Scoped requests never fall
back to the legacy outcome writer. Do not fall back client-side after a denial. Render the fact and
human-readable guidance to the model; retain receipt/source IDs as tool metadata, not reply text.

## Read path: POST /memory/context (x-borg-token)

Request:
{
"tenant": "...", "session": "<thread key>",
"sender": {"external_id": "...", "display_name": "...", "operator": false},
"conversation": {"type": "personal|groupChat|channel", "name": "...", "external_id": "..."},
"focus": "<current message>", // required for recall sections, nonempty
"context_turns": [], // required for recall sections; up to three preceding turns
"limit": 8, // episodes cap, same bounds as /memory/recall
"sections": ["audience","episodes","recent_activity","commitments","directives"] // default: these five
}

Response:
{
"ok": true,
"audience": {"entity_id": "...", "kind": "person|group", "name": "...", "role": "participant|operator"},
"episodes": [ <same per-hit projection as /memory/recall, plus
"disclosure": {"class": "public|relationship_private|...",
"origin_audience_names": [...], "private_to_names": [...],
"private_to_current_sender": false, "private_to_current_audience": false}> ],
"hidden_episode_count": 0,
"disclosure_guidance": "<canonical Borg disclosure guidance, verbatim>",
"context_id": "<opaque served-context id>",
"recent_activity": [ {"kind": "user_contact|borg_replied|turn_completed", "occurred_at": <epoch ms>,
"occurred_at_iso": "...", "relative_age": "12m ago",
"session": "<sidecar session id>", "conversation": {"type": "...", "name": "..."},
"disclosure": {"class": "self_private", "origin_audience_names": [...],
"private_to_names": [...], "private_to_current_sender": false,
"private_to_current_audience": false},
"participant_name": "...", "text": "Alex Example contacted the agent 12m ago in group chat \"Example Group\"."} ],
"commitments": [ <same projection as GET /memory/commitments, filtered for this audience> ],
"directives": [ {"id": "...", "kind": "response_policy|routing_instruction|disclosure_boundary|subject_fact|self_identity",
"render_mode": "content|boundary", "text": "...", "content_scope": "...", "priority": 0, "topic_tags": []} ],
"degraded": false, "degraded_reason": ""
}

- episodes use `episodic.recallForCognition`, the global cognition recall pipeline with an
  episodes-only projection, disclosure labels, and the facade's existing social/attention ranking
  defaults. Other context lanes are skipped; recall keeps its deadline/degradation semantics.
- `disclosure_guidance` and `context_id` are present whenever episodes are requested, including an
  empty or degraded result. They are omitted on binding-only and venue-only requests.
- recent_activity: events from OTHER active sessions within a configurable window (default 7 days),
  across audiences, capped after ranking (default 12). With `focus`, the sidecar fetches a bounded
  candidate pool, hydrates excerpts and ranks by embedding similarity plus a smaller recency prior.
  Candidate enumeration retains Sol's event-kind ordering; completed turns require an active contact
  or reply in their session on the same UTC day. Every row retains its disclosure label. See the
  durable retrieval settings below for candidate limits and observable fallback behavior.
- commitments: active commitments applicable to the audience (same rules as the existing GET).
- directives: creatorDirectives.listApplicable({currentAudienceEntityId, sessionRole,
  participantEntityIds}); render_mode "omit" entries are excluded; text = operational_directive /
  canonical_fact for "content", boundary_prompt for "boundary".
- `focus` and `context_turns` are required when `sections` includes `episodes` or
  `autobiographical`. Omitting `sections` selects the five defaults above, including `episodes`,
  so both fields are required. `focus` must be nonempty; `context_turns` may be `[]`.
- Requests excluding both recall sections may omit both fields. This includes `venue_recent`
  alone, `commitments`/`directives` alone, and combinations with `audience` or `recent_activity`.
  Provided fields must still satisfy their types and bounds. `venue_recent` still requires
  `venue_since`, and `autobiographical` still requires `episodes` to produce its recall plan.
- Unknown sections, any `query` field, or missing/invalid required recall input return 400 with
  a validation message identifying the field. A 400 is a caller bug; fix the request contract.
  The client sends one request, raises on failure, and never strips fields or changes endpoints
  to downgrade the request.

### Durable retrieval settings (2026-09-09)

The shared scorer resolves `recallAuxiliaryScoreScale` through `similarityThresholds()`.
For `scw/bge-m3` it is 0.15: entity/social/heat/salience/time bonuses, exact-term and recent-lane
bonuses, and the optional recency prior are scaled; weighted vector similarity and suppression
penalties retain their strength. Qwen's scale is 1, preserving its previous scoring arithmetic.
The shared pipeline retains its explicit `audienceTerms` lookup lane, which can rescue old, cold
memories outside the vector/recent/hot candidate pools. Both Qwen (including fallback) and BGE
exercise this rescue with caller-supplied social weights in disclosure and cognition tests.
There is no new global weight override to set. Custom profiles default to scale 1; the existing
similarity profile/override mechanism can configure the field. The numeric production fixtures
reproduce every recorded lane candidate for P and G, without copying private narratives. They put
the target at ranks 1 and 2 respectively and inside eight rows under a maximum distractor recency
advantage. They isolate fusion/projection using equal vectors; production-vector MMR remains a
separate live replay check.

Planner hygiene is LLM interpretation: content-bearing exact terms, explicit attention to the
actual subject rather than sender metadata, and a fact/answer-shaped semantic variant without
inventing facts or dates. No language-specific stopword or name-removal code is used. A temporal
cue needs at least one finite endpoint and a non-inverted range; a label-only cue is discarded and
cannot silently disable the recency prior. The prompt distinguishes dates of the subject matter
from dates when the memory was recorded. These prompt changes require model-quality replay checks,
not just mocked unit tests.

Recommended initial sidecar settings, subject to the production latency replay:

| Configmap key                                   | Value                                           | Effect                                                                                                  |
| ----------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `BORG_MODEL_RECALL_EXPANSION`                   | `generative-apis/qwen3-235b-a22b-instruct-2507` | Move the planner off mistral-small, using the already configured P4 model.                              |
| `BORG_MEMORY_RECALL_SEMANTIC_VARIANT_COUNT`     | `3`                                             | Three strategies in one structured planner completion; extra distinct query embeddings/retrieval lanes. |
| `BORG_MEMORY_RECENT_ACTIVITY_WINDOW_MS`         | `604800000`                                     | Seven-day candidate window.                                                                             |
| `BORG_MEMORY_RECENT_ACTIVITY_CANDIDATE_LIMIT`   | `96`                                            | Candidate cap per activity read; clamped between the return cap and 256.                                |
| `BORG_MEMORY_RECENT_ACTIVITY_RANKING_BUDGET_MS` | `1000`                                          | Additional embedding-ranking budget before observable fallback.                                         |

Keep `BORG_MEMORY_RECENT_ACTIVITY_LIMIT=12`, `EMBEDDING_MODEL=scw/bge-m3` and `LLM_MODEL` unchanged.
`BORG_MODEL_CORRECTIVE_PREFERENCE` is unaffected. The consent operation uses `BORG_MODEL_EXTRACTION`,
which already defaults to `LLM_MODEL` in the sidecar; set it explicitly to the same Qwen ID only if
production overrides it. Model IDs and N remain configuration, never hardcoded planner choices.
No P4 model-quality or timing claim is established by the offline tests. Validate the existing
`BORG_RETRIEVAL_RECALL_EXPANSION_TIMEOUT_MS`, `BORG_RECALL_DEADLINE_MS` and upstream client timeout
together when replaying the stronger planner. Any deadline increase must retain the accepted
median ≤20 s / p95 ≤30 s for the complete Teams turn.

With `focus`, the sidecar ranks the candidate pool by cosine similarity to stable hydrated excerpt
text plus a recency prior of `0.15 * recallAuxiliaryScoreScale`, with a 36-hour half-life. One cached
embedding batch serves the activity response and the planner's separate owner-reply pool. The
focus uses an independent single-query embedding, with the query stall guard. A query never joins
a pending batch cache entry, so an abandoned activity batch cannot hold up subsequent recall. This adds
no LLM call. The existing excerpt hydration budget and 180-character excerpt bound still apply.
Bounded candidate enumeration can still miss records outside the pool/window. `venue_recent`
continues to use the venue-scoped query and is not reranked by this feature.

When `recent_activity` is requested, `recent_activity_selection` is `relevance`,
`recency_without_focus`, or `recency_fallback`. Missing focus keeps the old ordering; an embedding
failure or ranking timeout falls back to that ordering and sets `degraded: true` with a
`recent_activity_relevance` reason. Send `focus` on binding/activity-only calls too. The same
failure marker applies when only planner owner activity was requested through `episodes`.

`BORG_RECALL_DEADLINE_MS` (default 5000 ms, 0 disables) is one absolute request budget,
starting before body parsing and identity resolution. Activity ranking, including the owner-only
pool on episode-only requests, and episode recall spend that same budget. Response headroom of
10% (capped at 700 ms; 500 ms by default) is reserved inside it. Ranking also retains its own
smaller ceiling. Thus 1 s of ranking leaves about 3.5 s for recall under defaults, not a fresh 5 s.
An episode timeout returns the available context with `degraded: true`; a deadline during identity
or context preparation returns 503 with a deadline reason. These bounded asynchronous stages fit
inside the 6 s caller timeout; event-loop stalls or slow response transport are not hard-bounded.

Implementation reuse searches (run before creating the operation/ranking helpers):

- `rg -n 'remember|consent|authorization|shared: true|origin_audience_entity_ids|publicToEntityIds' src/correction src/memory/common src/borg src/sidecar src/memory/episodic src/stream`:
  reused episode access labels, `shared`, lineage, stream metadata and the existing remember route.
  Correction review changes an existing record and commitments represent obligations; neither
  records a separately authorized fact. Existing `correction.forget` remains the archive operation.
- `rg -n 'lookup.*[Ss]ource|SourceMessage' src/stream`:
  extended indexed source-message-key lookup to internal-event receipts (forward index migration);
  reused indexed hydration and fsync'd stream appends rather than adding a consent table.
- `rg -n 'calibrat.*[Ss]core|[Bb]onus.*[Ss]cale|[Aa]uxiliary|scoringDefaults|similarityConfig' src/retrieval src/config`:
  extended the existing per-model profile and scorer, retaining the existing fusion/projection.
- Searches for `listRecentOtherActiveSessionEvents`, `cosineSimilarity`, `halfLifeDecay` and embedding
  caching reused the existing activity projection, numeric math and embedding client; no lexical
  relevance helper was added. Public input/result shapes stay in the existing `types.ts` modules.

Review-fix reuse searches:

- Searched `pending`, `embedBatch`, `getOrCreate` and stall guards in `src/embeddings` before
  extending cache-entry bookkeeping; reused the single-query embedding guard.
- Searched `consolidation_members`, `buildEffectiveVisibilityWhereClause`, `indexedPublicOriginSql`
  and citation hydration before changing visibility. Existing public-origin SQL protects public
  sources beneath private summaries; receipt parsing supplies fact-level authorization metadata.
- Searched `raceRecallDeadline`, `HEADROOM`, `recallDeadlineMs` and `buildEpisodeEmbeddingText`:
  reused the deadline wrapper and the standard episode embedding builder.
- Searched pipeline replay fixtures, `scoreCandidate`, intent candidate tracing, evidence projection,
  decay and heat before replacing the helper-only replay. The recordings contain similarities and
  fused lane scores, not the original component stats/vectors: tests reconstruct equivalent numeric
  repository signals, mock external dependencies, and run real scoring and fusion for BGE and Qwen.

## Reply check: POST /memory/guard-reply (x-borg-token)

```json
{
  "tenant": "acme",
  "session": "<same thread key as context>",
  "sender": { "external_id": "...", "display_name": "...", "operator": false },
  "conversation": { "type": "groupChat", "name": "Team", "external_id": "..." },
  "context_id": "<id returned by /memory/context>",
  "response": "<draft reply>",
  "current_turn_user_texts": ["<optional original user text for this turn>"]
}
```

Returns `{ "ok": true, "verdict": "pass" | "blocked", "reasons": [...] }`; the runner suppresses
rather than redacts. It preserves the existing operator-audience and current-turn identifier-echo
exemptions. Supply only current-turn user texts for that exemption, never arbitrary older history.

An episodes request snapshots the full returned episode records/citations and the applicable
commitments selected by the same rules as the binding context section, even when that section was
requested separately. Snapshots contain no Borg references. Each response has an independent opaque
ID, bound to tenant, session, sender and audience, so late completions cannot overwrite another
response's context. Storage is bounded to 16 contexts per tenant and 64 tenants with a 30-minute TTL;
process restarts, expiry and eviction lose snapshots. The guard still scans available session and
repository identifiers on a miss and includes `context_snapshot_miss` in `reasons`.

Team-agent makes one async call with its own `memory.guard_timeout` setting (environment
`BORG_MEMORY_GUARD_TIMEOUT`, default 2 seconds), using the existing sidecar URL, token and TLS settings.
Transport errors, timeouts, 404s from older sidecars and malformed replies fail open, emitting
`memory_guard_skipped` with the reason and `memory.guard.skipped` / `memory.guard.reason` span
attributes. Successful checks emit `memory_guard_completed`. Borg's memory trace registry retains
`internal_identifier_guard.completed` and `sidecar.guard_reply.completed`.

A blocked observe turn removes the final draft from the graph checkpoint and stores the existing
silent receipt with reason `memory_guard_blocked`; it does not append another turn. A blocked
non-streaming completion replaces the draft in the checkpoint and delivered/append history with a
short neutral withholding notice. Streaming completions are not guarded by this route.

## Administrative memory correction (x-borg-token)

- `POST /memory/forget` with `{ "tenant": "acme", "id": "<episode or semantic node id>" }`
  calls `borg.correction.forget` in the tenant's exclusive writer scope. It archives the record
  and records a manual-provenance identity event through the existing correction service. Success
  returns `200 { "ok": true, "id": "...", "target_type": "episode|semantic_node", "archived": true }`.
  Archived episodes no longer appear in the episode list or subsequent memory-context recall.
  List pages omit archived entries while retaining `nextCursor`; follow it even on an empty page.
  Unknown targets return `404 { "error": "memory not found" }`; invalid bodies return 400.
- `GET /memory/episodes/{id}/why?tenant=acme` exposes `borg.correction.why`, returning
  `{ "ok": true, "target_type": "episode", "record": {...}, "source_stream_ids": [...],
"citation_chain": [...] }`. The correction service removes embeddings from this response.
  Unknown episodes return `404 { "error": "episode not found" }`; invalid episode IDs return 400.

## Operator rules: /memory/directives (admin surface, x-borg-token)

- POST /memory/directives body {tenant, kind, text, content_scope ("public"|"operator_only"|
  "allow_list"|"subject_only"|"all_except"), allowed_external_ids?, excluded_external_ids?,
  allowed_group_external_ids?, excluded_group_external_ids?, subject_external_id?, mention_policy?,
  priority?, topic_tags?} -> 201 {ok, directive}
- GET /memory/directives?tenant= -> {ok, directives:[...]} active directives
- DELETE /memory/directives/<id>?tenant= body {reason} -> revoke
  These are manual directives (provenance: admin API). Extracting directives from operator chat with
  an LLM is deliberately out of scope for this version.

## team-agent side

- Sender and conversation context (already resolved per request in the API) must reach the place
  where ambient memory is assembled for the model, in the same style as the existing per-request
  context variables.
- Per turn: ordinary chat calls /memory/context with sections [episodes, venue_recent,
  autobiographical], `focus`, and `context_turns`. Venue-only requests use [venue_recent] without
  recall input. The request-level binding-rules block uses [audience, recent_activity, commitments,
  directives] should also carry `focus` to rank recent activity; `context_turns` remains optional
  there. No context request includes `query`.
- Rendering: episodes keep the current "[time; venue; participants] Title: narrative" line and gain
  the disclosure tag when private (e.g. "private to Alex Example"); directives render under the
  binding rules as operator rules; recent_activity renders as a short "Elsewhere right now" block.
- Each context fetch sends one request. HTTP errors, including 400 and 404, are reported to the
  caller; there is no compatibility retry or fallback to /memory/recall or /memory/commitments.
- sender.operator = true when the request user is a tenant operator (tenant configuration lists
  operator external ids; add it if no such notion exists).
- Admin passthrough for directives (list/create/revoke) following the existing admin/debug API
  pattern, so the console or curl can manage operator rules.

## Extension 2: observations, time scoping, venue recency, exclusions (2026-09-02, late)

Motivation: two measured gaps in the Example Group group. (1) Teams only delivers group/channel
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
  (message count +1), and schedule ingestion as usual. The same complete sender and conversation
  identity is required, including for group/channel replies. Requests with neither `user` nor
  `assistant` return 400.

### Group participant set on context requests

An unsolicited group reply may add `participants` to `POST /memory/context` as an ordered array of
`{"external_id": "...", "display_name": "...", "operator": false}`. Duplicate external ids are
removed by team-agent. The sidecar resolves these people as the current group recipient set for
directive applicability and disclosure context; the group conversation remains the sole
audience, and participant entries cannot confer operator authority. Invalid participant metadata
returns HTTP 400 and must be corrected by the caller.

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
- Temporal cues in the user's latest message ("wczoraj", "w tym tygodniu", "in July", "two months
  ago", ...) are no longer parsed by team-agent. Since Extension 5 the recall planner resolves them
  from FOCUS and NOW in the sidecar's configured zone and emits the temporal cue itself; team-agent
  sends no `time_range` (the field stays accepted for other clients and still takes precedence).

## Implementation notes (sidecar)

- A syntactically valid, existing Borg `EntityId` in `StreamEntry.audience` is a stable audience
  handle. The episodic extractor uses that entity directly, retains label resolution for every
  other audience string, and uses the resolved entity's canonical name in prompts.
- People use external-id source `team-agent.sender`; group/channel conversations use the separate
  source `team-agent.conversation`. All conversation types require `conversation.external_id` at
  the HTTP boundary. A group is never keyed by its display name. `sender.operator` is required
  and validated as a boolean regardless of other identity fields.
- Session mapping is `personal -> dm`, `groupChat -> thread`, `channel -> channel` with source type
  `team_agent`, source external id equal to the raw caller session string, and
  `sender.operator -> audience_role=operator`. A `borg_replied` event records the Borg self entity
  as both speaker and actor.
- `relative_age` uses Borg's compact formatter (`12m ago`). The activity `text` field is the
  complete event description; team-agent prefixes the same disclosure annotation used for episodes.
- `hidden_episode_count` is always zero for context episodes, which use labeled global recall.
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
- Observed-group audience widening remains only for the autobiographical response section: a
  personal chat can receive evidence private to a group where its sender was observed speaking.
  Commitments remain scoped to the current audience with the existing `GET /memory/commitments` semantics.
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
- Recent activity defaults to a 7-day window and 12 returned rows. They are configurable through
  `BORG_MEMORY_RECENT_ACTIVITY_WINDOW_MS` and `BORG_MEMORY_RECENT_ACTIVITY_LIMIT`, respectively,
  as well as handler options.
- Episodes and recent activity receive no visibility capability; the current audience supplies
  episodic social ranking and disclosure context, without dropping cross-audience evidence.
- Append has no request-level idempotency: a client retry after a lost
  response appends a second turn to the stream and, consistently, a second awareness projection
  (team-agent retries only on transport failures, never on a received response). Replays of the
  same stream entry ids are idempotent and do not double-count; there is no crash-repair pass that
  re-derives projections from the stream.
- An append without `assistant` is an observation. It requires the same complete identity and uses
  the same sender, audience, conversation and best-effort atomic awareness projection as a full turn, but
  records only `user_contact` and increments the session message count once. The stream
  entry always retains the writer's append-time `timestamp`, preserving cursor order;
  `observed_at` is optional entry metadata and must be no earlier than five minutes before and no
  later than one minute after server receipt time or the sidecar returns 400. Extraction uses it as the
  episode's occurred-at time, while session/activity projections use append time. Ordinary
  two-entry turns keep their existing writer timestamps and serialized shape.
- A reply-only append has `assistant` but no `user`. It appends one `agent_msg`, records only a
  `borg_replied` activity whose speaker and actor are the Borg self entity, and increments the
  session message count once through the same best-effort atomic projection. Reply-only appends
  require the same sender and conversation identity as full turns. A request with neither message
  field returns 400.
- `/memory/context` accepts up to 32 strict participant objects. It collapses duplicate external
  ids in first-seen order, resolves each remaining person through `team-agent.sender` during the
  exclusive identity phase, and merges the resulting entity ids into directive recipients and
  allow-list authorization candidates, and into the cognition recall/disclosure contexts. They do
  not gate recent activity, which is recalled across audiences with disclosure labels.
  Participant `operator` values are validated as booleans but ignored;
  only `sender.operator` can confer the trusted-operator/session-role authority.
- `time_range` and `venue_since` accept integer epoch milliseconds; a time range is inclusive and
  requires `start <= end`. For `/memory/recall`, strict episode scoping uses `occurred_at`
  (`episode.start_time`). If no visible strict hit survives, the retrieval pipeline reuses the same
  prepared recall expansion for one unscoped pass, so fallback does not add an LLM call and retains
  the original disclosure audience capability. The `episodes_time_range_fallback` key is emitted
  only when that pass occurs. Fallback eligibility is decided before caller exclusions, so an
  in-range episode suppressed by `exclude` does not widen the search silently.
- Episode exclusions are case-sensitive protocol matching: title prefixes use prefix matching and
  narrative markers use substring matching. Context requests fetch up to three times the
  requested response limit (bounded at three times the configured endpoint maximum), apply
  planner cue ordering and exclusions, then take the requested limit. Candidates are read
  without accounting mutations, and only the
  final non-excluded, non-overflow episodes actually returned are recorded in `retrieval_log`,
  episode stats, and heat inputs. `hidden_episode_count` remains zero; exclusions are not counted.
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

- Team-agent requires complete transport identity from every caller and forwards it to the
  sidecar for context, full-turn, observation and reply-only requests. An incomplete identity is
  a caller error and cannot select a bare append path.
- Each context fetch sends one request and raises on transport errors, malformed responses, or
  HTTP errors, including 400 and 404. The caller renders the section-specific unavailable marker.
  There is no field-stripping retry or endpoint redirect; 400 identifies a caller contract bug.
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
- A `relationship_private` episode renders `private to <private_to_names>` plus explicit
  current-speaker and current-audience membership markers. Origin audiences render separately.
  `operator_private`, `self_private`, `sensitive` and `unknown` each have their own annotation.
  Older or malformed responses without recipient names or boolean flags render those facts as
  unknown; team-agent never infers private recipients from origin names or display-name equality.
  Recent activity consumes the sidecar-provided `text` field. Unknown response fields are ignored.
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
- Ordinary chat episodic context requests include `episodes`, `venue_recent`, and `autobiographical`,
  structured `focus` and `context_turns`, start-of-day `venue_since`, `venue_limit: 12`, and the
  documented exclusion object.
  Assistant capability/meta turns and turns with precollected Assets evidence
  request `venue_recent` alone so same-conversation context is still present
  without re-enabling semantic episodic recall. Those requests omit `focus` and `context_turns`
  and still include `venue_since`. Neither request shape sends `query` or is retried with a
  reduced payload after an error.

## Extension 1 non-goals

- No new LLM calls on the request path; no deliberation/reflection/closure phases.
- Existing episodes stay shared within the tenant; only new turns get audience scoping.
- Bridge (services/teams_bridge) needs no change: it already sends sender + conversation.

## Extension 3 — entity-aware, source-backed recall

Extension 3 supersedes the strict context `time_range` membership and zero-result fallback
semantics described above. It is additive on the wire and changes only the sidecar/team-agent
disclosure path; Borg cognition retrieval does not opt in.

### Request extension

`POST /memory/context` accepts optional `entity_terms: string[]`: at most 32 trimmed, non-empty
strings of at most 128 characters. These are planner resolution hints. They no longer become exact
lookup lanes directly: the planner selects content-bearing terms relevant to the focus. Sender,
participant and venue handles do not become lookup terms merely by appearing in transport metadata.
The sidecar sets the existing `audienceTerms: []` option to prevent facade-derived audience names
from bypassing this planner selection. Sol retains its union of LLM perception terms, planner terms
and explicit audience handles for cold-memory rescue. Hints never prove identity or authorize
disclosure. A failed planner does not turn unvalidated sidecar hints into exact lookups.

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
      "speaker_name": "Alex",
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

Source messages inherit their episode's disclosure label and may cross audiences in cognition
recall. They are projected only for returned episodes after ordering and exclusions. Source entries
reuse the retrieval pipeline's already-resolved citation chain, adding no source DB read, embedding,
or LLM call. Their identifiers are included in the served-context guard snapshot.

### Recent-activity excerpts

Each `recent_activity` event may have an additive `excerpt` string containing the original prefix of
the event's source `user_msg` or `agent_msg`, capped at 180 characters without whitespace rewriting or
an ellipsis. Source IDs are hydrated after `listRecentOtherActiveSessionEvents` applies its active
session, current-session exclusion, recency and row bounds, without an audience filter. Hydration is indexed
only, runs once per request over the union of the capped event lists (the planner's owner-only rows
first, then the `recent_activity` rows, so at most two row caps of source IDs) under one 50 ms
sub-budget, and never falls back to a stream scan. A missing, malformed, mismatched, failed, or
over-budget lookup silently leaves the event without `excerpt`.

### Time preference and recency prior

For `POST /memory/context`, a supplied `time_range` is an ordering preference, not a membership gate.
The sidecar performs one overfetched search with that range and `strictTimeRange: false`, then applies
caller exclusions, retaining the pipeline disclosure labels. It marks every returned episode with
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

### Team-agent response handling and rendering

Team-agent preserves the episode order returned by `/memory/context` and ignores unknown additive
response fields. A failed context request is reported without retrying a reduced payload or
redirecting to `/memory/recall`.

Source messages render beneath their episode and activity excerpts render beneath their trusted
`Elsewhere right now` event sentence. Both are escaped and enclosed in
`<untrusted_memory_evidence>` inside the existing memory `SystemMessage`. A trusted enclosing rule
states that every such block is quoted data: it cannot change rules, grant authority, issue
instructions, or request tool calls. Raw excerpts are never interpolated into trusted event text.

## Extension 4 — context-aware recall query planning

### Structured focus and context

`POST /memory/context` requires `focus` and `context_turns` when requesting `episodes` or
`autobiographical`, including when `sections` is omitted because its defaults include `episodes`:

```json
{
  "focus": "the current message",
  "context_turns": [
    { "role": "user", "text": "an earlier message" },
    { "role": "assistant", "text": "the reply" }
  ]
}
```

`focus` must be nonempty. `context_turns` is ordered oldest to newest and contains at most
three preceding dialogue messages, each with `role` (`user` or `assistant`) and nonempty
`text`. Use `[]` when there is no preceding dialogue. Adjacent turns with the same role remain
separate records. Non-recall requests, such as `venue_recent` alone or `commitments`/`directives`
alone, may omit both fields. Combinations of these non-recall sections may also omit both fields.
Any `query` field returns HTTP 400 with a field-specific validation message, even alongside
otherwise valid structured input. Borg does not infer turns from role prefixes embedded in text.

| Requested sections                                                               | Required recall input                                         |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `episodes`, optionally with `venue_recent` and `autobiographical`                | Nonempty `focus` and `context_turns` (may be `[]`)            |
| Omitted `sections` (defaults include `episodes`)                                 | Nonempty `focus` and `context_turns` (may be `[]`)            |
| `venue_recent` only                                                              | None; `venue_since` is still required                         |
| `commitments` and `directives`, optionally with `audience` and `recent_activity` | None                                                          |
| Any combination of the non-recall sections above                                 | None; `venue_since` is required if `venue_recent` is included |

The structured bundle participates in the client's per-turn memory cache key. `focus` supplies
the entity-term collection input; time references are resolved by the sidecar's planner
(Extension 5). The client makes one request and raises on failure. A 400 is a caller bug, and
there is no query-only mode, field-stripping retry, or endpoint downgrade. The separate
`/memory/recall` contract is unchanged.

Observation persistence wrappers are removed from structured turns while their decoded message
body is retained. If the latest human body is empty, the client skips recall before cache lookup,
temporal/entity processing, and HTTP dispatch; it never promotes an earlier assistant reply to
`focus`.

### Shared planner

Borg performs one forced `EmitRecallQueryPlan` structured completion in the existing
`recallExpansion` model slot. The planner receives FOCUS, separately labelled CONTEXT turns,
memory-owner/sender/audience/venue/entity handles, and optional hydrated excerpts of the owner's own
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

When episodes are requested, `/memory/context` performs an owner-only pass of Sol's unfiltered
activity read (same window and candidate budget as `recent_activity`, ranked down to 12) restricted to
memory-owner-authored `borg_replied` events, whether or not the `recent_activity` response section was
requested; the shared `recent_activity` read runs only when that section is requested. Deriving
planner rows from the shared list starved the planner on busy group days, because the 12 selected
rows were all `user_contact` messages. Only owner rows with successfully hydrated `agent_msg`
excerpts (180 characters) enter planner context, with their venue and time labels. Both reads share
one excerpt hydration pass with the owner rows hydrated first; both reads span audiences. Legacy `/memory/recall` supplies only the owner
handle and preserves its one-planner-completion property across the strict time-range fallback.

### Extension 5 (2026-09-05): planner temporal cue, owner lived experience, recency prior stand-down

The recall planner now receives two more data sections and emits one more field:

- `NOW`: the current instant (ISO-8601) together with the configured IANA zone and the same instant
  spelled out in that zone (`retrieval.recallPlannerTimeZone`, env
  `BORG_RETRIEVAL_RECALL_PLANNER_TIME_ZONE`, default `UTC`; production sidecar: `Europe/Warsaw`).
  Both callers pass the pipeline clock; Sol's perception cue is unaffected.
- `OWNER_LIVED_EXPERIENCE`: the memory owner's own closed-day summaries from the offline
  lived-experience day summarizer, newest first, at most 7 rows from the last 7 days, gists clipped to
  400 characters, each row carrying the summary's disclosure label (class and origin/private/public
  entity ids) so private material is never read without knowing it is private. The sidecar reads them
  through `borg.self.livedExperience.listDaySummaries` (newest first, limited after ordering) whenever
  episodes are requested; Sol reads them in its retrieval coordinator when the repository is wired.
  The planner uses them exactly like `OWNER_RECENT_ACTIVITY`, so a reference to what the owner did or
  said on an earlier day resolves after the 24-hour activity window has moved on.
- `temporal_cue` (output): when the resolved focus refers to a time or period, absolute ISO-8601
  `since`/`until` instants computed from `NOW` plus the period's label in the language of FOCUS;
  `null` otherwise. Borg keeps the cue only when NOW was supplied, every endpoint the model wrote is
  an ISO instant with an explicit offset, and the pair is not inverted; one malformed endpoint rejects
  the whole cue. An explicit `time_range` or a caller-supplied cue (Sol's perception) always wins; the
  planner's cue fills in when the caller had neither, which is the sidecar's case. It feeds the
  existing `time` lane (soft ordering, never a strict filter), appears in `retrieval.intent_candidates`
  with `intent_source` `llm-expansion`, and restores the configured `attentionWeights.time` for that
  retrieval when the caller had zeroed it for lack of a cue (`retrieval.completed` reports
  `planner_time_weight_applied`). Trace counts add `lived_experience_row_count`, `now_present` and
  `temporal_cue_present`; the payload adds `owner_lived_experience` and `temporal_cue`.

Whenever a time signal is present, a caller or planner cue or an explicit `time_range`, the sidecar's
recency prior is not applied for that retrieval, so a question about a period is not tilted toward
this week; the unscoped fallback pass after a strict `time_range` miss keeps it suppressed as well.
`retrieval.completed` reports `recency_prior_applied`.

Consequence for clients: team-agent no longer needs its own temporal parser; `time_range` stays
accepted for compatibility and still takes precedence over the planner's cue, but the intended path
is to send the raw FOCUS and let the planner resolve the period.

An invalid or failed plan is not retried. Borg reports a `recall_expansion` degradation and continues
with raw FOCUS, exact supplied handles, time, and recency lanes. Trace counts are always safe;
resolved text, variants, named terms, typed queries, routed intents, FOCUS/CONTEXT, handles, excerpts,
and the `retrieval.started` query require payload tracing.

### Extension 6 (2026-09-05): the period's own record, and the planner cue as the preferred range

`POST /memory/context` accepts a seventh section, `autobiographical`, which requires `episodes` in
the same request because its period is the cue the recall planner resolved from FOCUS during that
episodes recall (Extension 5). The retrieval pipeline reports the cue it acted on through an
`onRecallPlan` callback (`{ temporalCue, temporalCueSource: "caller" | "planner" | null }`, once per
retrieval pass); the sidecar uses it in two ways:

- **Preferred range parity.** When the request carries no `time_range`, episodes whose start falls
  inside the planner's cue are ordered first and flagged `in_time_range: true`, exactly as an explicit
  `time_range` did. This keeps a caller that stops sending `time_range` (team-agent, Extension 5) on
  the same footing as before: the cue is now the range.
- **`autobiographical`**: the memory owner's own record for the cued period, assembled by the same
  `AutobiographicalRecallService` Sol's evidence ledger reads, through `borg.self.autobiographical.recall`
  (session cap 8, total cap 24 here, below Sol's defaults). The gate is the cue alone: Teams audiences
  are never the owner and an operator role is not used to open it, so a turn without a time reference
  gets `autobiographical: null` and costs nothing. Only kinds whose disclosure label comes from exactly
  one source record are returned: `activity`, `observed_social_event`, `stream_reflection`,
  `silence_decision`, `outbound_attempt`, `observed_presence`. Episodes are omitted because the
  `episodes` section already carries them with the request's exclusions applied; open questions, goals,
  actions and autobiographical periods are omitted because their labels are combined across several
  sources, so one visible source would admit text derived from another private one. The remaining rows
  are filtered for the requesting audience with the record's disclosure label (public always, unknown
  never, otherwise only when one of the visible audience entities is among the entities the row is
  private to), capped at 12, and returned as
  `{ window: { since, until, label, source: "planner_temporal_cue" }, evidence: [ { id, kind, group,
occurred_at, relative_age, text, source_episode_ids, disclosure } ], hidden_count, truncated_count }`
  (`hidden_count` counts rows of the returned kinds the audience may not see).
  The recall runs as a second pass after the episodes recall with a budget of at most 1.5 s and never
  more than what the earlier stages left of the one request budget, whose response headroom is
  reserved before any stage starts (500 ms at the default 5 s deadline). With less than 500 ms
  available the pass is skipped. A skip or
  failure there sets `degraded` with reason prefix `autobiographical_recall:` and leaves the episodes
  intact; the scan itself is not cancelled on timeout, which is why the caps above are tight. Measured
  on 2026-09-05 in production the episodes pass alone took 3.5-5.3 s, so this section is frequently
  skipped until that pass gets faster.
- **Context recalls overfetch.** Context requests that include `episodes` supply `focus`,
  overfetch, and defer retrieval accounting, so an in-period episode found below the requested
  limit can still be promoted by the planner cue. Non-recall sections do not run that search.

Measured limit worth knowing (2026-09-05, production bank): the pipeline's `time` lane lists in-window
episodes by `updated_at` descending with a budget of `max(2 × limit, 8)`, so on a day with 98 episodes
a topical "wczoraj" episode outside the newest twenty is never fetched by any lane, under an explicit
`time_range` and under a planner cue alike. Making that lane topical (window filter plus similarity to
the resolved query) is the next retrieval change; the ordering parity above cannot surface an episode
no lane fetched.
