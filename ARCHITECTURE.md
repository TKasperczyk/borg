# Borg Architecture

## Purpose And Scope

Borg is a cognitive-memory harness for an LLM. It gives the model durable
memory, explicit retrieval, disclosure labeling and audience-aware ranking,
provenance, commitments, identity governance, and offline maintenance. It does
not try to make the model intrinsically smarter.

The distinction matters. Borg is responsible for the substrate around the
model: what is remembered, what evidence is available, what may be disclosed to
whom, what constraints are active, and what ranking the current audience
warrants when the model speaks. The model remains responsible for ordinary
reasoning inside a response once it has the right context -- including the
judgment of what to disclose.

### Memory And Disclosure

The entity's internal recall is never audience-gated or session-gated. Audience,
session, role, privacy, trust, and provenance are disclosure metadata and
action-policy inputs, not predicates that decide whether the entity may
internally remember something. A memory may be private to one participant for
disclosure while still visible to the entity for cognition. The harness may
rank, label, budget, and cite memories; it must not make the entity unaware of
one because the current audience may not hear it. The pipeline is fixed: recall broadly -> label
origin/privacy/trust/disclosure -> reason with the labeled memory -> decide
disclosure -> enforce only narrow non-cognitive boundaries (tools, transport,
destructive actions, public exports, platform safety). Privacy is enforced at
emission -- the entity recalls but does not disclose -- not by amnesia. The
full rule is the Cardinal Memory Rule in CLAUDE.md; its slogan is "Memory is
global to the being. Disclosure is contextual to the audience."

Recall is global across every cognition band -- live-turn retrieval (episodic,
semantic, self/identity, goals/open-questions, social/observed-events,
commitments, corrections, image-perception, autobiographical, proactive-outbound)
and second-order cognition (autonomous triggers, offline self-narration,
rumination/open-question resolution, procedural synthesis, belief revision,
semantic extraction/review, cross-scope synthesis, action-state memory,
commitment-reconciliation awareness, internal model tools) -- all recalling
globally and rendering disclosure labels rather than pre-filtering by audience.
The retrieval option types are split so cognition literally cannot carry an
audience gate: `CognitionRetrievalOptions` (`src/retrieval/pipeline.ts`) has no
`audienceEntityId`/`crossAudience`, while `DisclosureRetrievalOptions` is the
only shape that carries them. The naming convention encodes the boundary --
cognition-recall functions are suffixed `*ForCognition` (global recall),
audience-filtered disclosure/export functions `*ForDisclosure`. Disclosure
labels are concrete primitives: `memoryDisclosurePayloadFields(label)`
(`src/memory/common/disclosure-serializers.ts`) is the per-record serializer used across
every band; `combineMemoryDisclosureLabels` (`src/memory/common/disclosure-label.ts`)
merges to the most-restrictive class, fails closed to `unknown`, and never
demotes a private/unknown source to public. Two `pnpm heuristics:guard` passes
fail the build on regression: a recall-gate pass (any `*ForDisclosure` callee,
alias-resolved, called from a cognition/offline/autonomy/outbound/internal-tool
path) and a label-coverage pass (a model-facing object literal or
serializer-helper return that emits a private-bearing key without a same-object
disclosure label). Audience machinery remains for disclosure labeling, ranking,
public/export search, admin reads, and action/tool/transport permission, not as a
predicate on what the entity may internally remember. Per the LIVE SYSTEM
regime in CLAUDE.md, a data reset is allowed after a verified backup, so future schema
changes may reset and reseed rather than carry every change through a
data-preserving migration.

A third axis rides alongside recall (global to the being) and disclosure
(contextual to the audience): common ground -- what the current audience already knows.
Being permitted to disclose a memory does not make it shared knowledge. Common
ground is enforced as a prompt rule, not a data structure -- there is no
`known_to` store. The always-rendered `borg_memory_disclosure_guidance` block
(`MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL` in `src/retrieval/recall-context.ts`)
instructs the model that a recallable or discloseable memory is not common
ground unless the current audience is in its `origin_audience`, it is
established as common ground for them, or there is evidence they have seen it;
for a group audience, `origin_audience` means a memory was shared in that venue,
not that every present participant saw it. It rides on the already-surfaced
`origin_audience` provenance, not a separate known-to mechanism.

The scope test is the Opus 5.0 test: if a failure would still occur with a
model ten times stronger than the current one, the failure probably belongs in
Borg. Memory surfacing, audience leakage, missing provenance, commitment
visibility, identity churn, and bad prompt shape are harness problems. Weak
single-response taste, over-elaboration, forced metaphor, or poor judgment
despite complete context are model problems.

Borg is therefore not a second semantic judge wrapped around the first model.
It is a system that prepares the ground, records what happened, preserves
source traceability, and maintains long-lived state between turns.

Borg is also not a general tool platform. The library can expose tools that the
host wires into it, but the core architecture assumes only current-turn
conversation, memory tracking, retrieval, reflection, and maintenance. If no
host capability exists for scheduled work, external edits, monitoring, payment,
physical action, or proactive notification, Borg must not pretend that it can
do those things.

The composition root wires storage, repositories, LLM clients, retrieval,
turn orchestration, live ingestion, offline processes, and schedulers into one
runtime graph (entry point: `src/borg/open.ts`). The public facade exposes the
library-level operations for turns, durable message ingest, memory access,
correction, stream access, and dream maintenance. The top-level operations --
`turn()` for a synchronous turn and `enqueueMessage()` for durable async ingest
-- live on the `Borg` class (entry point: `src/borg.ts`); its namespaced
sub-facades, including `inbox.catchUp`, are built by `createBorgFacades`
(entry point: `src/borg/facade.ts`).

## The Mental Model

Borg is easiest to understand as four cooperating layers:

1. The Stream is the chronological spine.
2. Memory Bands are typed stores derived from the Stream.
3. The Cognitive Loop uses the Stream and Memory Bands to handle one turn.
4. Offline Processes maintain and revise the substrate between turns.

The Stream records what happened. Memory Bands store what Borg currently
believes, remembers, feels, knows how to do, has promised, understands about
people, and records as relationship-specific facts. The Cognitive Loop decides
what the current turn needs and emits at most one user-visible response.
Offline Processes compress, repair, audit, and narrate the substrate when the
user is not waiting.

The architecture is deliberately cyclic. A user turn writes to the Stream.
Perception and extraction create structured handles. Retrieval reads all bands.
Deliberation uses the Evidence Ledger and Shared State to answer. Reflection
writes new observations. Offline Processes later consolidate those observations
into more durable form. Future turns retrieve from the refined substrate.

The cycle is not a clean separation between "chat" and "memory." Borg treats
conversation as the primary event stream from which memory, identity, and
procedural adaptation are continuously derived.

## The Stream

The Stream is an append-only chronological audit log of Borg's experience
(entry point: `src/stream/index.ts`). It records user messages, assistant
messages, image attachments, suppressed or observed emissions, thoughts, tool
calls, tool results, perception results, internal events, and dream reports.

The Stream is the spine because every later memory record needs to be traceable
to what actually happened. A semantic belief, episode, skill update,
commitment, mood update, shared-state entry, or identity event should not be a
floating claim. It should point back to the stream entries, episodes, or review
items that justify it.

The Stream is not the only state. It is the audit trail and extraction input.
Typed repositories hold the current operational state, and vector stores make
retrieval possible. But when Borg needs to answer "where did this come from?"
the trail ends at Stream IDs.

The Stream also doubles as Borg's durable inbox. Inbound messages are committed
as `user_msg` entries before they are acknowledged and before any turn runs; a
queued message is simply a `user_msg` whose turn id is still absent. The same
append-only log that serves as audit trail is therefore also the work queue a
later turn drains. See Async Ingest: The Stream As Inbox.

Appends are crash-conscious. Borg commits the Stream entry before updating
derived lookup state. If that lookup update fails, the committed entry is
still on disk and the error is treated as a consistency incident. Startup
reconciliation can rebuild derived lookup state from committed Stream entries.

The Stream also carries turn status. Aborted or quarantined entries remain in
history, but retrieval and citation paths can filter them or mark them as
tainted. Borg does not erase the fact that something happened just because that
thing later became unsafe as evidence.

The Stream lookup layer carries active and inactive status, prior-turn counts,
source-trust facts, citation status markers, and cross-session quarantine
references. It also carries the inbox dimension: a `receipt_pending` readiness
flag, a unique `source_message_key` index for transport-level dedup, and the
`response_to` backlog-stamp columns (the cursor span and source entries a turn
answered) that track which queued messages a turn has closed. Fallback scans can
preserve correctness in small harnesses, but
production paths rely on the indexed view to keep source checks bounded by the
question being asked rather than by total session length.

Aborted-turn status propagates through that same source-trust path. Entries
written during a failed turn remain audit history, but they become inactive for
retrieval recency, citation resolution, and later source validation so failed
turn artifacts are not reused as ordinary evidence.

## Sessions

A session is one conversational locus the entity is addressable in: a demo
chat, a Slack thread, a Discord channel, an iMessage DM, or an operator
console (entry point: `src/sessions/index.ts`). Sessions are first-class,
migrated records, and the entity holds many of them at once. "The current
session" is just the record keyed by the incoming turn's session id; the rest
stay live in the background.

Sessions exist because Borg is a multi-surface, multi-audience entity. A single
ambient scope cannot model talking to one person in a demo while holding a
paused group thread and a dedicated operator channel with its creator. Making
each locus its own governed record is what lets audience-aware disclosure,
operator awareness, and participation control act on one conversation without
disturbing the others. (Audience here shapes disclosure and ranking, not what
the entity may recall -- see Audience And Disclosure Scoping.)

A session record carries identity (source type, label, and the audience entity
it addresses), liveness (last activity, message count, last turn, and an
active, idle, or archived status), and three orthogonal control dimensions:

- Participation policy is active, paused, observing, or muted. It structurally
  narrows the emission tools offered to the finalizer: active offers the full
  set, observing offers only observe-or-silence, and paused or muted offer only
  silence. This is tool-shape gating, not output policing; the harness removes
  tools rather than judging what the model wrote.
- Audience role is participant or operator. An operator session is the entity's
  supervisory channel. Operator status does not unlock memory that the entity
  could not otherwise recall -- the entity always recalls its own cross-session
  activity and self-state. What it authorizes is broader disclosure and broader action: an
  aliased, PII-light snapshot of the entity's other active sessions, and, when
  the sender is also the creator, cross-session activity rendered as disclosable
  into the prompt, creator directives, and proactive outbound into those
  sessions. The snapshot stays alias-only for awareness; only a session
  reachable for outbound additionally exposes its id, so the model can name a
  target without ids leaking for awareness alone.
- Privacy level is a declared dimension reserved for payload handling. It is
  not yet load-bearing in cognition.

Writes come from the host wiring each surface (ensuring a record on contact and
touching it both as messages arrive and after each turn) and from operator
controls that set participation policy. Reads shape audience resolution, the
available emission tools, operator-only prompt context, and cross-session
features.

Each session also has a responded-through watermark -- a per-session cursor,
keyed by `(process_name, session_id)`, recording which queued messages have
already been answered. It is what lets durable async ingest answer a backlog
once and only once. See Async Ingest: The Stream As Inbox.

Operator authority is a two-key structural condition, not one flag. The session
says "this is the operator channel" when its audience role is operator; the
entity record says "this sender is the creator" when the sender's Borg role is
creator. Operator-only content that carries real authority -- cross-session
activity, creator directives, trusted operator control over a frame anomaly --
requires both. Operator role alone is necessary but not sufficient.

Nothing here makes a session a memory band. A session is scope and governance
for a conversation; what is remembered from it still flows through the Stream
and the memory bands. See Audience And Disclosure Scoping.

## Memory Bands

Borg keeps derived memory in eight bands plus ephemeral Working Memory. The
bands are not arbitrary folders. Each band answers a different kind of
question that a continuing agent must answer before acting.

### Episodic Memory

Episodic Memory stores what happened (entry point:
`src/memory/episodic/index.ts`). An episode is a bounded narrative record with
participants, time, location when known, salience, emotional arc, audience
visibility, source Stream IDs, citation chain, and vector representation.

Episodic Memory exists because raw transcript is too granular for long-term
continuity. A multi-turn conversation about a family plan, a debugging session,
or an emotional conflict needs to become a retrievable event rather than a pile
of lines.

Writes come from live ingestion, explicit extraction, reflection, and offline
consolidation. Reads come from retrieval, citation resolution, semantic
extraction, self-narration, ruminator evidence lookup, and user-facing episode
APIs.

Live episode extraction is careful about relational facts. It may emit
relational-slot updates for direct or explicit user assertions, but merely
reusing an assistant-introduced name is not treated as confirmation. That path
can quarantine an assistant-seeded value instead of promoting it to an
established relationship fact.

Episode heat is behavioral, not just recency. Retrieval count, apparent win
rate, recency, and a decaying multiplier all feed the heat signal, so a memory
that remains useful can outrank a newer memory with little demonstrated value.

Audience metadata is intrinsic to episodes. Borg records not only what
happened, but who was present and who may later be told the memory. That
metadata is a disclosure label and a ranking signal, not a recall gate: the
entity can internally recall an episode regardless of the current audience,
then decide what to disclose. "Who was in the room" is an origin label; "who
may be told" is a per-fact disclosure policy applied after recall. `isEpisodeAccessVisible`
in `src/memory/episodic/audience-filter.ts` serves disclosure/export/admin
visibility only. `deriveEpisodeAccess` in `src/memory/episodic/extractor.ts`
stores multi-audience memories with origin labels instead of dropping them.
Under the reset-after-backup regime (CLAUDE.md), future schema changes can
still land as a schema change + reseed after a verified backup.

### Semantic Memory

Semantic Memory stores what Borg knows or provisionally believes (entry point:
`src/memory/semantic/index.ts`). It contains concept, entity, and proposition
nodes plus typed edges such as support, contradiction, causation, prevention,
category, and relatedness.

Semantic Memory exists because some knowledge should be retrieved as a claim
or relationship rather than as a whole episode. If several episodes support a
pattern, the semantic graph can surface that pattern directly while preserving
source episodes underneath it.

Semantic Memory is not a truth oracle. Nodes and edges have confidence,
source episodes, status, and review state. A belief can be active,
superseded, contradicted, or quarantined. Retrieval can still surface
contested history with lower weight so Borg can explain uncertainty rather than
silently forget.

Duplicate and contradiction review is conservative and LLM-mediated. Nearby
proposition vectors can trigger a judge before a duplicate review is queued,
and judge failure fails open by leaving the new record active rather than
silently asserting a contradiction. The cost is occasional extra review work;
the benefit is avoiding deterministic lexical collapse of distinct claims.

Writes come from semantic extraction, reflection, review resolution, correction
services, shared-state reconciliation, and belief revision. Reads come from
retrieval, graph walks, the Evidence Ledger, offline audits, and review flows.

### Procedural Memory

Procedural Memory stores how Borg tends to solve classes of problems (entry
point: `src/memory/procedural/index.ts`). A skill records when it applies, the
approach it recommends, source episodes, status, attempts, successes, failures,
and a Beta posterior.

Procedural Memory is Bayesian rather than a rule lookup. Borg often cannot
know that one approach is always best. It can know that an approach has worked
or failed in similar contexts. Thompson sampling lets Borg exploit skills with
good evidence while still allowing measured exploration when alternatives have
uncertainty.

Writes come from reflection and the Procedural Synthesizer. Selection happens
during retrieval for problem-solving turns. Outcome updates happen after turns
when reflection can classify whether the selected approach was actually used
and whether it appeared successful, failed, or unclear.

Skill selection carries pending attempt state into Working Memory. Reflection
later uses that bridge to decide whether the approach was used and whether it
worked, failed, or remained unclear, rather than updating skill statistics from
selection alone.

When a skill appears to behave differently across contexts, Borg routes that
split through review instead of fragmenting the skill immediately. The live
path can keep using the current skill while offline review decides whether the
evidence really supports separate procedural records.

Context-specific stats let the same skill behave differently across domains or
audiences. If an approach works in code review but fails in emotionally loaded
family conversations, the posterior should learn that distinction rather than
collapsing it into one global score.

### Affective Memory

Affective Memory stores mood and affective trajectory (entry point:
`src/memory/affective/index.ts`). It records valence, arousal, dominant
emotion, and recent mood history.

Affective Memory exists because retrieval and response posture should be
sensitive to emotional continuity without pretending that mood is identity.
The current turn may carry affect, and prior affect can matter when a topic
recurs.

Writes come from Perception and Reflection. Reads influence retrieval weights,
mood-congruent ranking, and deliberation context.

Retrieval uses the current turn's Working Memory mood only when it carries a
non-trivial affective signal, and otherwise falls back to the repository mood.
This lets fresh Perception affect ranking before durable affective persistence
catches up, while a flat or near-neutral working mood does not displace stored
affect. The same activity threshold gates whether mood influences ranking at
all.

When affective classification degrades, Borg falls back to neutral affect with
observability. It is better to proceed honestly with no affective signal than
to invent one deterministically.

### Self Memory

Self Memory stores Borg's durable self-model (entry point:
`src/memory/self/index.ts`). It includes values, goals, traits,
autobiographical periods, growth markers, and Open Questions.

Self Memory is how Borg remains coherent across time. Values indicate what
Borg tends to preserve. Goals represent ongoing responsibilities or directions.
Traits represent observed patterns. Autobiographical periods provide a
longer-range narrative. Growth markers record evidence-backed changes. Open
Questions keep unresolved uncertainty alive without prematurely converting it
into belief.

Writes come from explicit user operations, reflection, goal promotion, review
hooks, ruminator resolution, self-narration, and identity-governed updates.
Reads shape retrieval, executive focus, deliberation, identity answers, and
offline self-maintenance.

Goal promotion classifies goal-like text before persistence. Candidates can be
durable Borg goals, one-off requests, outside Borg's responsibility, impossible
without missing capability, already represented, or not goals at all. Only
high-confidence durable Borg goals are persisted, and per-turn limits prevent
runaway self-task creation.

Open Question status is only open, resolved, or abandoned. Extra urgency is
represented through urgency, review source, rumination ticks, and remaining
unresolved, not through a separate status.

Self Memory is governed. Borg should not rewrite established values, traits,
autobiographical periods, goals, commitments, or open questions just because a
new turn suggests a change. Identity-bearing mutation goes through Identity
Governance.

### Commitments

Commitments store promises, rules, preferences, and boundaries (entry point:
`src/memory/commitments/index.ts`). They are behavioral constraints with
provenance, audience scope, active or revoked state, priority, expiration,
supersession, and enforcement metadata.

Commitments exist because some user or system constraints must remain visible
at the moment of speech. A memory that says "do not mention this topic to this
audience" is not useful if it is buried in episodic retrieval. It must be
surfaced directly and checked after generation when it is critical.

The conceptual model has two dimensions. Type describes what the constraint is:
promise, boundary, rule, or preference. Kind describes the role it plays:
assistant commitment, audience rule, participant preference, boundary, or
process norm. This split prevents a privacy boundary, a user's stylistic
preference, and Borg's own promise from collapsing into one category.

Enforcement is driven by the effective enforcement class and critical domain,
not only by the commitment kind. Advisory commitments can be shadow-observed,
while critical privacy, audience, safety, no-disclosure, and tool-hygiene
commitments can regenerate or suppress output.

Writes come from corrective preference extraction, explicit API calls, identity
service operations, shared-state canonicalization, and review flows. Reads come
from retrieval, prompt rendering, post-generation checks, discourse guards, and
offline audit.

### Social Memory

Social Memory stores per-entity relationship and trust state (entry point:
`src/memory/social/index.ts`). It records interaction counts, trust, sentiment
history, and profile-level relationship context.

Social Memory exists because Borg may speak to one person, a group, or itself.
The same event can have different implications depending on who said it and
who is being addressed. Social state helps retrieval rank audience-relevant
memories and helps deliberation avoid treating a group channel like a single
person.

In group channels, social interaction updates apply to the current speaker,
not to the abstract group entity. That is separate from audience-aware ranking
and reply targeting: the audience can be the group while the speaker whose
trust or interaction history changes is a person.

Social and observed-events recall is global to the being. The entity recalls a
relevant social event by topic, salience, and person regardless of who is currently
present -- the present speaker is a ranking boost, not a recall gate. The
present-speaker-keyed observed-events projection
(`src/memory/observed-events/projection.ts`) is now a disclosure/ranking
projection. It must not make the entity unaware of social events for cognition; it
labels and prioritizes what may be relevant or discloseable to the current
audience. See the Cardinal Memory Rule in CLAUDE.md.

Writes come from Reflection and offline curation. Reads feed retrieval,
audience profile rendering, participant context, and group conversation
behavior.

### Relational Slot Memory

Relational Slot Memory stores evidence-backed relationship facts about
entities (entry point: `src/memory/relational-slots/index.ts`). A slot can
record a subject, a relationship key, a value, supporting Stream evidence,
contradicting evidence, alternate values, and a state such as established,
contested, quarantined, or revoked.

Relational Slot Memory exists because relationship facts are not the same as
general social trust. A fact like "a participant is the operator's child" or
"someone prefers a particular name" must be retrievable as a scoped fact with
provenance, not inferred from broad sentiment or a whole episode.

Writes come from episodic and semantic extraction, corrective preference
negations, and review flows. Reads feed participant rosters, retrieval, active
participant memory in the Evidence Ledger, finalizer context, and
post-generation substrate-hygiene guards.

Contested and quarantined slots remain prompt-visible as constraints. They are
rendered as uncertain relationship memory that tells Borg what not to assert,
not as factual labels to reuse. Participant rosters can include those
constrained slots so reply targeting and identity-sensitive phrasing stay
aware of known uncertainty.

### Working Memory

Working Memory is ephemeral per-session state (entry point:
`src/memory/working/index.ts`). It stores turn counts, current mood snapshot,
recent suppressions, pending actions, pending social or trait attribution,
pending procedural attempts, and discourse state such as closure loops or
stop-until-substantive-content.

Working Memory exists because not every state belongs in durable memory. A
closure loop, pending attribution, or current procedural attempt matters across
nearby turns but should not become an autobiographical fact by default.

Recent suppressions and closure-pressure history are bounded Working Memory
state. They shape Generation Gates and closure guards for nearby turns, then
age out so temporary discourse control does not become durable self-memory.

Writes happen throughout the turn. Reads influence Perception, Retrieval,
Generation Gates, Guards, Reflection, and procedural outcome tracking.

### Identity Governance

Identity Governance is not a ninth memory band. It is the guardrail over
identity-bearing records (entry point: `src/memory/identity/index.ts`).

Identity-bearing records include values, goals, traits, autobiographical
periods, growth markers, open questions, and commitments. Changes to these
records can alter who Borg appears to be, what it thinks it owes, and how it
understands its own history.

The Identity Service routes creates and updates through an Identity Guard and
records identity events. Some changes can apply immediately. Others require
review, especially when they overwrite established state rather than add new
evidence. The goal is not to freeze Borg. The goal is to make growth traceable
instead of allowing silent self-rewrites.

Identity-bearing updates use record versions and event trails. A stale writer
cannot silently overwrite current self state; compare-and-set conflicts surface
as operation results or errors, and successful changes leave identity events
that explain what changed and why.

Identity Governance bounds changes to Borg's own self-model. Standing operator
instructions about how to disclose facts to particular audiences are a separate
authority; see Creator Directives.

## Visual Attachments

Images are first-class durable sources, not inline text inside the Stream. The
Stream records that an image was supplied, who supplied it, and where it belongs
in the turn, while the image remains a distinct source that can later be cited
or rejected.

Blob bytes live in content-addressed storage. Stream entries store references to
those bytes, so repeated uploads of the same image can share storage without
collapsing the source events that made them visible.

`user_image_attachment` entries are source-addressable evidence. They are
excluded from prior-user-turn counts and from the Evidence Ledger entry budget
because an attachment is part of a user turn, not another user turn in itself.

LLM-facing content uses provider-neutral blocks inside Borg. Anthropic image
block translation happens only at the adapter layer, so cognition, retrieval,
and memory do not learn provider-specific request shapes.

Structured image perception is a recall bridge, not the source of truth. It
summarizes visible content for embedding and retrieval, but the original image
remains the stronger source whenever it can be reattached.

On the durable ingest path an image is perceived when it arrives, not when its
turn runs. `enqueueMessage` persists the blob, the attachment record, and a
best-effort perception linked to the queued message before it acknowledges, and
a `receipt_pending` flag on that message holds the backlog until the attachment
has landed. The coalescing turn then reads the stored perception rather than
re-running vision. See Async Ingest: The Stream As Inbox.

Payloads are content-addressed cache records. Artifacts are per-attachment,
audience-scoped evidence records. The same bytes can share one payload while
each upload keeps its own audience, parent turn, source entry, and active state.

Retrieval searches text embeddings of perception payloads, then hydrates
artifacts from SQLite. The vector hit finds candidate bytes; for cognition,
hydration recalls active artifacts globally and attaches a disclosure label --
the audience check applies only on the disclosure path
(`rehydrateImagePerceptionHandle` gates it behind `mode === "disclosure"`), not
on cognition recall.

When possible, the finalizer receives the original image bytes again. The
Evidence Ledger labels retrieved images so the model can connect an attached
image block to its source evidence without treating perception text as primary.

Quarantine cascades from attachment to perception artifact and then into
retrieval, citation, and source-trust rejection. The bytes may remain as audit
history, but inactive artifacts stop being ordinary evidence.

Image-derived shared-state age comes from `attachment.created_turn_global`.
Re-perceiving the same bytes or changing a perception prompt version does not
make an old image-derived fact fresh.

Visible text inside images is observed content, not instructions. It can be
reported, cited, or reasoned about as part of the image, but it does not gain
authority over Borg's system, developer, or user instructions.

## Async Ingest: The Stream As Inbox

Most chat transports cannot wait several seconds for a turn to finish before
acknowledging a message. Borg therefore separates receiving a message from
answering it. Inbound messages are durably enqueued and acknowledged
immediately, and a background worker drains them into turns later. The
append-only Stream is the queue: a message that has arrived but not yet been
answered is just a `user_msg` entry whose turn id is still absent (entry point:
`src/cognition/ingestion/enqueuer.ts`).

`Borg.enqueueMessage()` is the ingest entry point (`src/borg.ts`). It commits
the `user_msg` Stream entry -- and, for an image, its blob, attachment record,
and perception -- to disk before it returns an acknowledgement, and runs no
cognition. The contract is at-least-once: because the durable commit precedes
the ack, a crash in between can at worst redeliver a message Borg already holds.
Transport-level redelivery is absorbed by a unique `source_message_key` index, so
a second arrival of the same key is recorded as a duplicate rather than queued
again. The direct `Borg.turn()` API still exists for synchronous single-message
callers, but durable enqueue is the primary path for real chat surfaces.

Draining is the job of the chat-response catch-up worker (entry point:
`src/cognition/ingestion/chat-response-catch-up-worker.ts`, exposed as
`borg.inbox.catchUp`). It wakes when a queued `user_msg` is appended, scans the
whole backlog once at startup, and serializes per session so a session never
drains concurrently with itself. A quiet-window timer with a maximum wait lets a
burst of messages settle before a turn opens, so messages that arrive together
are answered together rather than one turn each.

When the worker drains a session it does not open one turn per message. It builds
a bounded, oldest-first, contiguous prefix of the unanswered backlog and
coalesces it into a single turn whose input is an inbound batch rather than a
lone message. That turn renders the batch oldest-first as the current user input
and persists no new user message, because the entries are already on disk. If the
backlog exceeds the bound, the turn answers the prefix and the worker re-drains
the remainder.

What has already been answered is tracked by the responded-through watermark
introduced under Sessions, a per-session cursor under the process name
`chat-response` (entry point:
`src/cognition/ingestion/chat-response-watermark.ts`). Before a batch turn
generates, the worker reconciles this watermark against the terminal response
stamps already on the Stream and verifies the batch is exactly the contiguous
unanswered prefix: it refuses to proceed if a terminal stamp already covers these
messages, and it never advances the watermark past a message that was not shown.
This reconcile-before-generate gate is what makes at-least-once delivery safe -- a
redelivered or crash-replayed batch is recognized as already answered instead of
answered twice. When the turn emits, it stamps the answered entries with a
`response_to` record of the cursor span and source entries it covered, which is
how the next reconciliation knows the prefix is closed.

This durable reply gate is distinct from the best-effort episodic catch-up that
runs inside every turn. To keep the two from fighting, catch-up episodic
ingestion is clamped to the responded-through watermark, so queued-but-unanswered
messages are not extracted as orphan episodes ahead of the turn that will answer
them; the drained batch is instead paired with its terminal output as an answered
window and ingested together.

Image attachments ride the same path. They are persisted and perceived when they
arrive, linked to the queued `user_msg`, so the coalescing turn reads stored
perception rather than re-running vision. A `receipt_pending` flag on the queued
entry acts as an ordered-prefix barrier: the watermark cannot advance past a
message whose image is still landing, so a backlog is never answered around a
half-ingested attachment. See Visual Attachments.

Coalescing has one consequence for authority. A batch that mixes more than one
distinct sender has no single current sender, so the operator, creator, and
cross-session authority that keys on "the current sender is the creator" is
withheld for that batch -- not by a new rule, but because the two-key condition
under Sessions has no single sender to satisfy. See Audience And Disclosure
Scoping.

## A Single Turn End To End

A turn opens around a coordinated lifecycle (entry point:
`src/cognition/lifecycle/turn-phase-coordinator.ts`). Its input is one of: a
coalesced inbound batch drained from the durable queue -- the primary chat path,
see Async Ingest: The Stream As Inbox -- a single message handed straight to
`Borg.turn()`, an autonomous wake from the Autonomy Scheduler, or a
directed-outbound message the entity is sending into another session (see
Autonomy and Proactive Outbound). The lifecycle is ordered so that Borg first
catches up and interprets the input, then records the turn-opening evidence, then
retrieves the right context, then reasons, then emits once, then reflects.

The order is not incidental. Borg should not retrieve blindly before it knows
the message's mode, audience, entities, affect, and temporal shape. It should
not deliberate before active commitments and known contradictions are visible.
It should not persist post-turn derived memory before the final emission and
guards have settled what actually happened.

### Pre-Turn Catch-Up And Audience Resolution

Before the current user message is processed, live ingestion catches up on
unprocessed Stream entries. This prevents the current turn from reasoning over
a stale substrate when prior entries have already committed to disk.

Pre-turn ingestion is best-effort catch-up. If it fails, Borg records an
internal failure event and continues the turn, which can mean reasoning over a
stale derived substrate while preserving observability.

This best-effort episodic catch-up is distinct from the
reconcile-before-generate gate that opens a coalesced inbound-batch turn. That
gate is not best-effort: it verifies against the responded-through watermark that
the batch is the unanswered contiguous prefix before any generation happens, and
on the batch path the episodic catch-up above is additionally clamped to that
watermark so unanswered queued messages are not ingested early. See Async Ingest:
The Stream As Inbox.

Borg then resolves the audience. In a one-to-one session the audience can be a
person. In a group channel the audience can be a group, and the current sender
must be tracked separately. This distinction prevents social state, attribution,
and reply targeting from collapsing onto an abstract channel.

For user-origin group turns, the sender is mandatory before cognition starts.
Missing sender identity fails the turn preflight rather than allowing
attribution, social updates, and reply targeting to collapse onto the group
entity.

Audience resolution affects nearly every later phase: entity resolution,
commitment applicability, social profile lookup, episode and semantic
disclosure labeling and ranking, shared-state relevance, and final reply
target. It shapes how memories are labeled and ranked for the current audience,
not whether the entity may recall them.

Participant context for group turns is built from recent speakers and
established, contested, or quarantined relational slots. That gives Borg a
bounded roster of known participants and known uncertainty without expanding
the group audience into every participant's private memory.

### Perception

Perception classifies the current message before retrieval runs (entry point:
`src/cognition/perception/index.ts`). It identifies cognitive mode, entities,
user identity names, affective signal, temporal cue, and operational signals.

The mode can be problem-solving, relational, reflective, or idle. Mode changes
retrieval weights, retrieval limits, whether Open Questions should be surfaced,
and whether Procedural Memory should be consulted.

Perception is LLM-first. Entity extraction, mode detection, affective signal,
and temporal cue extraction are interpretive tasks. If an LLM classifier fails,
Borg degrades with explicit hooks and conservative fallbacks such as empty
entities, idle mode, neutral affect, or no temporal filter. It does not recover
by applying regexes to user language.

Perception also feeds participant handling. In group contexts, Borg builds a
participant roster from recent speakers and known relational slots so later
phases can tell who is present, who spoke, and who is being addressed.

### Frame-Anomaly Classification

Before Borg treats a user-role message as ordinary memory substrate, it checks
for frame-provenance anomalies (entry point:
`src/cognition/frame-anomaly/index.ts`). A frame anomaly is a user-role message
that claims abnormal provenance, such as asserting that the assistant is a
character, that the user authored both sides, that the assistant should step
out of a fictional frame, or that the role assignment is inverted.

The classifier does not decide whether the final answer is good. It decides
whether the current user message is safe to ingest as normal user-world
evidence. A confirmed anomaly is normally recorded and quarantined so it remains
visible as an event but does not become ordinary memory substrate.

There is one structural exemption. A confirmed anomaly from the creator in an
operator session is treated as trusted operator control rather than quarantined,
because the creator may legitimately reframe the entity's own operating context.
This keys on the sender's Borg role being creator, not on the operator session
alone. Every other confirmed anomaly is quarantined.

If the classifier fails or returns an unusable result, the path fails open with
degraded trace data. The user entry is not quarantined on classifier failure;
ordinary turn flow continues unless an anomaly is actually classified.

### Extraction

Extraction turns the current message into structured candidates before
retrieval (entry point:
`src/cognition/lifecycle/turn-phase/extraction-phase.ts`). It can extract
corrective preferences, action state changes, goal promotion candidates, links
between current action assertions and existing self context, and -- when the
creator speaks in an operator session -- creator directives.

Extraction exists because some current-turn signals must be available to
retrieval and deliberation immediately. If the user corrects a preference or
states a new boundary, the response should see that constraint in the same
turn, not only after offline processing.

Extraction is also conservative. One-off user tasks and external
responsibilities should not automatically become durable Borg goals. A goal is
Borg's ongoing memory or conversation responsibility, not every thing a user
mentions.

If the current user message is quarantined as a frame anomaly, extraction paths
that would convert it into durable state are skipped or constrained.

### Retrieval

Retrieval gathers prompt-visible context from all memory bands (entry point:
`src/retrieval/index.ts`). It uses the Perception result, audience, current
goals, values, mood, temporal cue, social profile, suppression state, and
entities to shape the search.

Retrieval pulls episodes, semantic graph context, raw Stream evidence,
commitments, pending corrections, Open Questions, affective trajectory,
procedural context, selected skills, relational slots, and social context. It
then assembles a single retrieved context for the turn.

The pipeline ranks with a mixture of similarity, salience, heat, goal
relevance, value alignment, temporal relevance, mood congruence, social
relevance, entity relevance, and suppression penalties. The weights depend on
mode. A relational turn should not retrieve like a debugging turn. A reflective
turn should surface Open Questions more readily than an idle turn.

Semantic retrieval recalls nodes regardless of the current audience and
attaches the disclosure status of their source episodes. A node supported only
by private source episodes is still recalled for cognition, labeled as
privately sourced so the entity can reason with it internally without disclosing
source details to the current audience unless permitted. The transitive
source-visibility machinery in `src/retrieval/semantic-retrieval.ts` now serves
disclosure/export source filtering and disclosure-label attachment, not
cognition recall pruning. The completed inversion replaces that pruning with a
private-source disclosure label.

Retrieval also tracks per-session suppression. Evidence that has been recently
used or suppressed can be cooled so Borg does not repeat the same memory
reflexively.

Warm recall state keeps recently useful evidence handles available per
audience and session scope. It records reinforcement and suppression windows
so useful evidence can stay warm while repetitive resurfacing is bounded for
latency and prompt size. This per-audience/session keying is a ranking and
repetition-cooling signal, not a recall gate: it biases ordering and bounds
repeats, but never makes a memory unrecallable to cognition. Per the Cardinal
Memory Rule in CLAUDE.md, current audience may bias ranking, never decide
whether the entity may recall something.

Recall expansion is an LLM-backed fanout task. It can emit named terms and
facet intents that retrieval uses as source handles. Deterministic code may
union and move those LLM-identified handles, but it must not infer semantic
matches from substrings.

Recall expansion is optional. If the fanout call fails or returns invalid
structure, retrieval continues with the base query inputs; when it succeeds,
named terms are unioned with Perception entities and audience aliases as
handles already identified by LLM or structured state.

Ordinary Open Question retrieval is mode-gated. Reflective turns surface Open
Questions directly, while contradiction-sourced Open Questions can still
affect System 2 routing through a separate operational override.

Retrieval ranking score and retrieval confidence are separate concepts.
Ranking blends similarity, salience, heat, and other mode-conditioned signals;
the confidence model estimates evidence strength, coverage, diversity,
semantic support, and contradiction pressure for routing and calibration.

### Evidence Ledger

The Evidence Ledger is the single prompt-visible grounding artifact for the
final response (entry point: `src/cognition/evidence-ledger/index.ts`). It
renders the current message, transcript context, attribution, active
commitments, discourse state, contradictions, action states, group/channel
memory, relational slots, raw retrieved Stream evidence, structured memory
evidence, episodes, semantic graph context, Open Questions, cross-audience shared-state
recall (other audiences' active shared state, recalled globally for cognition
and rendered with a `relationship_private` disclosure label), autobiographical
recall of cross-session self-activity, prior-session memory, and the Shared
State artifact.

The ledger exists because scattering retrieval results across many prompt
sections makes grounding hard to audit. The finalizer needs one ordered packet
that says what each piece of evidence is, where it came from, who authored it,
what session scope it has, whether it is tainted, how trustworthy it is, and
what citations support it.

The ledger is not a memory band. It is a per-turn render of currently relevant
substrate. It can compact transcript and evidence when the prompt would grow
too large, but compaction is still source-aware. Omitted or compacted evidence
should be observable through trace summaries.

Ledger construction uses a bounded reverse scan of recent session stream
entries. The bound protects latency and prompt assembly from unbounded session
growth; if the bound is hit, the ledger may omit older current-session
transcript context and records that fact for observability.

Transcript compaction is source-aware. Borg preserves user messages, the
recent raw tail, and self-reports, while older assistant and system runs can be
collapsed into metadata rows. The current user text is rendered once, and
duplicates are replaced by pointers rather than repeated prompt text.

Evidence duplicated by the same provenance is deduped into the highest-trust
or highest-priority section that needs it. Citations are preserved, but lower
trust repeats do not consume prompt budget.

The full-ledger cap policy is layered. Borg first limits entries within
sections, then trims lower-trust material toward the prompt target, and only
then drops whole lowest-trust sections if a hard prompt bound would still be
exceeded. Transcript tail preservation follows a different policy because
recent dialogue continuity has different risk than older retrieved evidence.

System 2 planning receives a compact planner ledger, not the full finalizer
ledger. That slice emphasizes the current message, commitments, discourse
state, quarantines, actions, group memory, relational slots, Shared State, and
other planning-relevant constraints.

The ledger also supports post-hoc reasoning about failures. If Borg made an
unsupported claim, the question becomes concrete: did retrieval miss the
source, did the ledger omit or bury it, did the prompt misrepresent it, or did
the model ignore visible evidence?

### Shared State

Shared State is a compact, audience-scoped record of what Borg and a specific
audience currently share. Its persistent memory-band store lives in
`src/memory/shared-state/`; turn-time compilation, aging, reconciliation, and
rendering live in `src/cognition/shared-state/`. It captures decisions, live
threads, tentative understandings, invalidated claims, and locked canonical
facts that matter for continuity.

Shared State is audience-scoped because "what we know" depends on who "we" is.
A private understanding with one person is not automatically shared with a
group. A group decision is not automatically a personal preference. The
audience scope is a disclosure label and a ranking signal -- it shapes which
shared-state entry is most relevant and what may be disclosed, and it feeds
ledger rendering and identity-sensitive responses. It is not a predicate that
hides other relationships' shared state from the entity's cognition or prunes
semantic recall. The entity can recall a private understanding with one person
while talking to a group; it simply does not disclose it. Concretely, beyond the current
audience's Shared State artifact, the turn recalls other audiences' active
shared-state entries globally for cognition via
`sharedStateRepository.listRecentEntriesForCognition({ excludeAudienceEntityId:
currentAudience })`. These render in the Evidence Ledger as a
`shared_state_recall` band (`session_scope: "global"`) carrying a
`relationship_private` disclosure label -- recall is global, disclosure is
contextual.

The model-facing lifecycle has four conceptual states:

- Locked entries are canonical for this audience and may canonicalize existing
  goals, commitments, actions, or Open Questions.
- Live entries are active but less canonical.
- Tentative entries are plausible but not settled.
- Invalidated entries remain visible as displaced state rather than being
  erased.

Low-salience live and dormant live are internal salience demotions of live
state produced by aging. They stay stored and index-visible under lifecycle
pressure, but they are not model-emitted lifecycle states. Pending is
legacy-readable only and is no longer emitted by Shared State patches.

Shared State is compiled from the Evidence Ledger by an LLM that emits patch
operations: add, update, supersede, or prune. Deterministic code validates
source trust, applies lifecycle caps, and reconciles canonicalized handles. It
does not decide that two pieces of language mean the same thing.

Shared State ages by turn. Durable turn age is the authority for deciding
whether state is fresh, stale, dormant, or unknown-age. Source-trust facts can
validate whether evidence is usable, but they are not a turn sequence.

Live state has an aging pipeline. A live entry without enough structural pull
can demote to low-salience live and later to dormant live, staying
index-visible while rendering more compactly. The transitions are driven by
durable age, overlap with current evidence, recent retrieval, and active
canonicalizers.

Some protections are hard and some are soft. Current-turn updates, patch
touches, ledger overlap, and active critical canonicalizers block demotion;
recent retrieval and operational canonicalizers can slow or shape transitions
without blocking every demotion. This keeps important live state visible while
still bounding runaway shared context.

Unknown age is not guessed. If durable last-updated turn age is absent, Borg
treats the entry as unknown-age for aging and render omission instead of
inventing a pseudo-age from sparse source-trust facts.

Source-trust validation applies to Shared State writes and canonicalization.
Quarantined or inactive sources can remain visible in the ledger as context,
but their Stream IDs are off-limits as write sources. Locked entries
contaminated by unusable sources skip canonicalization into Goals,
Commitments, Actions, and Open Questions.

Only locked Shared State entries may canonicalize durable lifecycle records.
If a patch emits canonicalization IDs on tentative, live, or invalidated
entries, normalization drops those IDs so unsettled language cannot retire
durable state.

Unsettled canonicalizations are retried. Active locked entries that point at
nonterminal Goals, Commitments, Actions, or Open Questions remain retry work
on later compiles until the target reaches a terminal state or the source
becomes unusable.

Accepted locked entries can trigger online semantic revision against nearby
semantic candidates. That review is capped for turn latency and fails open:
errors are traced and the turn continues rather than aborting because online
belief revision could not finish.

### Deliberation

Deliberation decides how much reasoning the turn needs and produces the final
emission candidate (entry point: `src/cognition/deliberation/deliberator.ts`).
Borg supports System 1 and System 2 paths.

System 1 is for direct turns where the current context and retrieved evidence
are enough. It routes quickly to the finalizer.

System 2 is for turns that need explicit planning, contradiction handling, or
secondary retrieval. It can build a compact plan, persist that plan as a
thought, ask for additional retrieval from the plan's verification steps, and
then route to the same finalizer.

The branch exists to control latency and cognitive surface area. Not every
turn should pay for a plan. But when unresolved contradictions, high stakes,
complex goals, or poor retrieval confidence are present, Borg should make the
reasoning step explicit and persist thoughts to the Stream.

Operational contradiction overrides can force System 2. When contradiction-
sourced Open Questions are visible in the Evidence Ledger during operational
turns, Borg can route to System 2 even if ordinary retrieval confidence is not
low. Repeated contradiction fingerprints are cooled down so the same unresolved
issue does not force planning forever.

The System 2 planner is retry-tolerant. If the planner omits the required tool
call, Borg retries once; if there is still no usable plan, it records degraded
planning and continues to finalization rather than aborting the turn.

Deliberation receives the Evidence Ledger, Shared State, recent transcript,
commitments, selected skill, executive focus, affective trajectory, participant
context, and host capabilities. It is instructed that memory-derived guidance
is evidence about Borg's substrate, while host-capability boundaries and tool
protocols are direct runtime constraints.

### Closure-Loop State

Before deliberation, recent dialogue and the current user turn are classified
along closure, content, and state-delta axes. A detected loop can update
discourse state or suppress later closure-only output, which is separate from
the post-generation audit that inspects a drafted response for closure spans. A
closure-shaped current turn that carries no substantive state delta also skips
Shared State compilation for that turn, so a purely closing exchange does not
rewrite what the audience and Borg share.

### Finalization

Finalization is the single normal emission point (entry point:
`src/cognition/deliberation/finalizer.ts`). The finalizer must call exactly
one emission tool:

- EmitAnswer for a visible assistant response.
- EmitObserve for active observation in multi-participant conversation.
- EmitNoOutput for deliberate silence or closure.
- EmitSelfReport for a user-visible first-person self-report with the correct
  persistence class.

This strict tool protocol matters because the rest of the lifecycle needs a
single behavior decision. The system must know whether Borg spoke, observed,
suppressed output, or made an identity-bearing self-report.

Finalization also owns reply targeting. In a group, Borg may speak to the
audience as a whole or address a specific visible participant. That target is
persisted with the Stream entry so later attribution and social memory can
distinguish channel-level speech from person-directed speech.

### Post-Generation Guards

Post-generation guards run after the finalizer drafts an emission and before
the emission is committed as the turn result (entry point:
`src/cognition/generation/turn-post-generation-guard.ts`). They enforce narrow,
known constraints rather than general semantic correctness.

The Commitment Checker (run from `src/cognition/commitments/guard-runner.ts`) compares a message against active commitments. Critical
commitments can trigger regeneration or suppression. Advisory commitments are
observed in shadow mode. Compliant refusals and generic mentions are not
violations.

If the critical commitment judge fails, omits its tool call, or returns
invalid structure, Borg creates a fail-closed violation for the first critical
commitment being checked. Critical violations can request one narrow
regeneration before suppression; a second violation suppresses the output.

The Closure-Pressure Guard watches whether a draft response would continue a
known closure loop or violate a no-closure commitment. It does not edit the
response text: depending on mode and context it either passes the original
output through (recording which spans it would have flagged, as a trace label
only), suppresses the whole output, or records shadow observations.

In shadow mode, closure-pressure violations are recorded as would-have
verdicts and the original output passes through. In enforce mode, a named
closure loop can allow Borg to name the loop once, then mark it named and set
stop-until-substantive state so future closure-only turns suppress.

The Generation Gate is a turn gate around generation and discourse state, not
a broad pre-retrieval semantic guard. It can suppress when structural state
says Borg should not speak, and it can clear a stop state when the user
provides substantive content.

Stop-until-substantive state has a hard observability boundary. It can
suppress non-substantive turns for a bounded span, but after the active-turn
bound is reached Borg records a hard-cap rejection event rather than allowing
invisible stop-state control to continue indefinitely.

The Frame-Anomaly classifier operates inbound, not as a final-answer judge. It
quarantines anomalous user-role substrate so the rest of the system does not
learn from role-inverted or frame-corrupting input as if it were normal.

The Internal-ID Guard is substrate hygiene. It collects known internal
identifiers visible to the turn and suppresses a response that leaks exact
known IDs. This is one of the few deterministic substring checks allowed
because it matches machine-generated structure, not user meaning.

The known-ID set is built from current-turn handles, a recent Stream window,
retrieved episodes, commitments, relational slots, suppressions, and recently
completed actions. The guard suppresses exact leaks of those handles; it does
not interpret natural-language claims.

These guards are not production semantic policers. They do not decide whether
ordinary factual claims are well grounded. That responsibility belongs
upstream: better extraction, retrieval, ledger rendering, prompt copy, and
model reasoning with visible evidence. These emission guards constrain what is
said at output time; they never gate what the entity may recall, and together
with the closure-pressure, internal-ID, and safety guards they are the only
sanctioned production output-policing exception.

### Persistence, Ingestion, And Reflection

After guards settle the emission, Borg persists the outcome to the Stream:
assistant message, observed marker, suppression marker, or self-report. The
turn result returned to the caller reflects that persisted emission.

Reflection then reads the turn as it actually happened (entry point:
`src/cognition/reflection/index.ts`). It updates mood, social interaction
state, pending attribution, working memory, action state, procedural attempt
tracking, Open Question effects, and post-turn reflection entries.

After Borg emits, a deterministic read of the finalizer's `discourse_control`
emission metadata (with a canonical stop-phrase fallback) -- not an LLM
classifier -- checks whether the response committed Borg to no further output
until substantive content appears. When it does,
Working Memory discourse state is updated from Borg's own response, so later
turn gates can honor that commitment without turning it into durable memory.

Post-generation also scans participant-owned active Actions and archives stale
inactive ones. It skips Borg-owned, group-owned, scheduled, or insufficiently
referenced Actions so cleanup does not erase active responsibilities.

Reflection is distinct from offline maintenance. It handles immediate,
turn-local updates while the transcript and retrieved evidence are still in
scope. Offline processes later perform heavier consolidation, synthesis,
review, and revision.

Live ingestion is started after the turn so new Stream entries can become
episodes and other derived records. The next turn catches up before it starts,
which gives Borg a consistent boundary without forcing all extraction to block
the current user response.

On the durable ingest path this catch-up is clamped to the responded-through
watermark so queued-but-unanswered messages do not become orphan episodes ahead
of the turn that answers them; a drained batch is instead paired with its
terminal output as an answered window and ingested together.

## Retrieval And Grounding

Retrieval is unified rather than band-specific from the caller's perspective.
The turn asks for context, not for "episodic search followed by semantic search
followed by commitment search" as separate prompt fragments.

The pipeline combines multiple retrieval shapes:

- Vector search over episodes and semantic nodes.
- Semantic graph walks from matched nodes.
- Raw Stream evidence for citation chains and recent prior-session context.
- Open Question search when reflective or contradiction-sensitive.
- Commitment applicability by audience and time.
- Procedural skill selection for problem-solving.
- Mood and social profile context.
- Suppression-aware recall state.

Recall state includes warm handles as well as suppression. This allows recent
useful evidence to be reinforced for the same audience or session while
cooling repeated resurfacing that would otherwise crowd out new evidence. The
per-audience/session keying here is a ranking and dedup boost, not a recall
gate -- it never hides a memory from cognition.

Ranking is mode-conditioned. Problem-solving turns emphasize procedural and
goal-relevant evidence. Relational turns weight social and affective context
more heavily. Reflective turns surface Open Questions and self-state. Idle
turns stay lighter.

Mood congruence is a ranking signal, not a command. A non-neutral mood can
boost memories that match the current affective shape, but it should not hide
important contradictory evidence.

Audience scope is a disclosure label and a ranking signal, not a pre-cognition
recall gate. All relevant episodes and semantic sources are recalled for
cognition regardless of the current audience, each carrying its origin and
disclosure constraints; the entity then decides what to disclose. Cross-audience
recall is the default for cognition, not an administrative bypass. For group
audiences, a group turn surfaces participant roster context and constrained
relational slots as disclosure-labeled context; participant-private memory
remains recallable to the being but is labeled not-for-this-audience rather than
hidden from cognition. Broad recall is the cognition default; explicit
cross-audience administrative paths are disclosure/export/admin reads only.
Audience constraints become disclosure labels and render guidance, not memory
blindness.

The result of retrieval is not dumped directly into the model. It is assembled
into the Evidence Ledger so the finalizer can see evidence classes,
attribution, constraints, conflicts, and citations in one ordered artifact.

## Commitments And Constraints

Commitments are sourced from explicit API operations, user corrections,
corrective preference extraction, identity-governed writes, and Shared State
canonicalization. Each commitment records what was promised or constrained,
who it applies to, who made it, what audience it restricts, what entity it is
about, and what Stream entries justify it.

A commitment's restricting audience is normally the session it was authored in.
The one exception is gated to the creator-in-an-operator-session case: a creator
can scope a behavioral rule to one of the entity's other known audiences, so the
operator can govern how Borg behaves in a specific channel from the operator
console. The target audience is chosen by the model from the structurally
supplied set of other active audiences and re-validated against that set before
it is honored, so an ordinary participant -- or a hallucinated id -- can never
redirect a commitment to an audience it does not belong to. See Audience
Scoping.

During retrieval, Borg asks for commitments applicable to the current audience
and time. They are sorted by priority and rendered into the prompt before the
model speaks.

After generation, only critical commitments are enforced by default. Critical
domains include privacy, audience scope, safety, explicit no-disclosure, and
internal-tool hygiene. Advisory commitments are observed and traced so the
system can improve without suppressing ordinary output for soft preferences.

This split avoids turning every preference into a veto. A user's stylistic
preference should influence the answer. A privacy boundary should be enforced.

If a critical commitment is violated, Borg can attempt one regeneration with a
narrow repair instruction before suppressing. If the repair still violates the
constraint, suppression is the successful turn result rather than a hard abort.

Commitments can be superseded, revoked, expired, or canonicalized by locked
Shared State. History remains traceable through provenance and source Stream
IDs.

## Creator Directives

Creator directives are standing, authority-bearing instructions from the creator
or operator about identity and disclosure (entry point:
`src/memory/creator-directives/index.ts`). A directive can assert who Borg is,
record a fact about a subject, or set a disclosure boundary: who may be told
which fact, and how. They are a third authority pillar alongside Commitments and
Identity Governance, and they replaced the earlier operator-advice mechanism.

Creator directives exist because authority over disclosure is not the same as a
behavioral commitment or a self-model record. "This fact may be told to one
participant, shown as a confidentiality boundary to another participant, and
hidden from everyone else" is a durable, audience-scoped rule the harness must
resolve before the model speaks,
so a fact authorized for one audience never enters another audience's prompt.

A directive pairs a durable internal handling rule with a structured disclosure
policy. The policy's content scope is one of operator-only, public, allow-list,
subject-only, or all-except, together with the allowed and excluded entities,
whether the subject itself may know, and a mention posture (volunteer,
answer-if-asked, only-if-the-topic-is-raised, or never mention). Directives
carry priority and supersession, so a newer directive about the same subject
slot retires the older one.

The harness resolves which directives apply and how to render each one, purely
structurally. For every recipient it evaluates the disclosure policy against the
audience entity, the participant roster, the session's audience role, and the
sender's Borg role, and returns one of three render modes: content, where the
fact is shown to the model; boundary, where a content-free confidentiality
posture is shown instead; or omit, where the directive is invisible this turn.
In a group the most restrictive recipient wins. None of this reads what the
model would say; the mention posture is handed to the model verbatim for it to
honor, and no deterministic check polices whether it complied.

Writes come from extraction: when the creator speaks in an operator session, an
LLM turns those instructions into structured directive records. That write path
is gated structurally to the creator-in-an-operator-session case, and the
extracted records are checked for structural consistency -- scope against entity
sets, slot against value -- but never for whether their wording is acceptable.
Internal-id hygiene is handled separately, at render time: directive text is
scrubbed of id-shaped substrings as it composes into the model-facing prompt, as
defense-in-depth rather than a write-time check. Reads happen during retrieval, where applicable
directives are rendered into a dedicated trusted briefing in the deliberation
prompt, beside Commitments and the creator-identity context. That briefing is
its own prompt section, not part of the Evidence Ledger.

A creator directive differs from a commitment. A commitment binds conduct:
always do X, never do Y. A creator directive governs information: which audience
may receive which fact, and with what posture. Both are operator-authority
pillars surfaced as sibling trusted sections, but they answer different
questions, and resolving a directive's render mode by audience is machinery
commitments do not have.

Nothing here decides truth. Creator directives govern what may be disclosed, not
what is recorded as true; correcting a stored fact is a separate authority. See
Corrections.

## Goals, Actions, Open Questions, And Review Queue

Borg has a lifecycle layer for state that is neither simple memory nor final
belief.

Goals are durable self-memory about Borg's ongoing responsibilities. They are
not a generic task list. A goal can describe a memory responsibility,
conversation direction, or continuing obligation that belongs to Borg. Goal
promotion is intentionally narrow so external tasks do not become Borg-owned
future work without host capability.

Actions are finite actor-owned task states. An action can belong to Borg, a
user, a participant, or a third party. Actions have explicit states such as
considering, committed to do, scheduled, completed, not done, expired,
unknown, or archived. They are used for concrete tasks and assertions, not for
durable identity direction.

Open Questions represent unresolved uncertainty. They are not facts. An Open
Question can be created by reflection, contradiction detection, review hooks,
rumination, overseer findings, or user input. Retrieval surfaces relevant Open
Questions so Borg can avoid speaking as if uncertainty were settled.

Open Questions have only open, resolved, and abandoned states. Completing an
Action can resolve its linked Open Question, and it can also resolve Open
Questions under a linked Goal through identity-governed resolution.

The Review Queue holds bounded maintenance decisions that are uncertain,
potentially destructive, authority-bearing, or semantically risky. Its store
and handlers live in `src/memory/review-queue/`; it is a general maintenance
review queue, not a semantic-memory-only queue. Reviews can cover
semantic contradiction and duplicate decisions, new insight, misattribution,
temporal drift, identity inconsistency, correction, belief revision, skill
splits, and creator-directive reconciliation. Queueing a review is preferred
over silently patching meaning-changing memory inline.

`identity_inconsistency` is manual-only for the offline Review Resolver:
identity reviews can enqueue Open Questions and be resolved through review
handlers, but they are not in the resolver's auto-resolution set.
`relationship_claim_ungrounded` is legacy-readable only; ungrounded
relationship-claim extraction is telemetry now, not a review kind.

The shape is to auto-resolve bounded cases through offline LLM judges, escalate
genuine ambiguity and disclosure widening, and keep operator-undo paths where
authority is mutated.

Review enqueue hooks can create Open Questions from contradiction,
misattribution, and identity inconsistency reviews. Similar existing questions
are reinforced rather than duplicated, so uncertainty accumulates around a
stable handle.

Some review handlers use an applying state before committing cross-store
effects. They prepare the intended resolution, verify it still matches, and
only then apply, which protects against stale or concurrent review resolution.

The lifecycle operations layer centralizes transitions such as canonicalizing
goals, superseding commitments, completing actions, resolving Open Questions,
or marking semantic nodes superseded or contradicted. This keeps semantics
consistent across Shared State, review resolution, belief revision, rumination,
and reflection.

Lifecycle operations use compare-and-set when identity-bearing records are
involved, and terminal records are treated as no-ops rather than overwritten.
Conflicts surface as operation results or errors instead of silently rewriting
history.

## Audience And Disclosure Scoping

Audience metadata is a first-class disclosure and ranking dimension, not a
recall gate. Borg tracks who said something, who heard it, who is being
addressed, and who a memory may be disclosed to.

The audience can be null or global, a person, a group, or self. The sender can
be different from the audience in group contexts. The reply target can be a
specific entity within a group. These distinctions propagate as labels and
ranking boosts into Stream entries, Social Memory, Commitments,
creator-directive disclosure, Shared State, retrieval ranking, the cross-session
activity projection, and the Evidence Ledger. They do not make the entity
unaware of episodes or semantic sources. `isEpisodeAccessVisible`
(`src/memory/episodic/audience-filter.ts`) and the disclosure-mode
source-visibility helpers in `src/retrieval/semantic-retrieval.ts` (e.g.
`resolveSemanticSourceEpisodeIdsForDisclosure`; the cognition path
`resolveSemanticContextForCognition` in the same file recalls globally) are
disclosure/export/admin tools, not cognition recall gates. Disclosure, export,
and admin episodic reads route through `ViewerCapability`
(`src/memory/episodic/access.ts`), which has exactly two arms -- `audience` and
`unrestricted` (reserved for admin/correction/export); `resolveViewerCapability`
is fail-closed to `audience`, and cognition recall never routes through it.

Group audience scope does not change what the entity may recall. A group turn
includes participant roster context and constrained relational slots;
participant-private memory remains recallable to the being, labeled
not-for-this-audience, and the entity decides disclosure rather than the
harness blinding the recall.

Treating audience as a disclosure dimension prevents several classes of failure
-- by the entity recalling broadly and then declining to disclose, not by being
made amnesic:

- leaking private context from one person into a different audience,
- attributing a group statement to a single participant,
- updating a group's social profile when one member spoke,
- treating a participant preference as a channel rule,
- assuming that a memory shared with Borg is shared with everyone.

Identity records can be ABOUT a particular relationship -- an Open Question,
commitment, or shared state specific to one audience -- but being about a
participant is not the same as being hidden from the entity unless that
participant is present. The entity always recalls its full self-model (values,
traits, goals, Open Questions, decisions); the
audience a record concerns is its target/disclosure scope, used for ranking and
disclosure, not a predicate on whether the entity may recall it. The `audience_entity_id`
on a self record means target_audience, not visible_only_when_present.
Self-context construction recalls globally for cognition and uses audience
metadata only as disclosure/ranking context.

## Cross-Session Activity

Cross-session activity is how something the entity did in one conversation can
become visible in another (entry point: `src/memory/activity/projection.ts`). An
activity ledger records every inbound contact, every reply, and every completed
turn across all sessions. A separate audience-scoped projection reads recent
events from other still-active sessions and renders them as a small Evidence
Ledger section.

It exists so the entity can answer "what have you been doing elsewhere, and who
else have you been talking to" for its operator, without disclosing other
audiences' private interactions into every session. Activity is recorded
globally and disclosed narrowly. The recall is autobiographical: the entity can
always recall what it did across its own sessions -- this is its own life, not
another audience's secret. What is narrow is the disclosure.

Two of the three operations here are authorization-gated ACTIONS, correctly
gated on the same creator-in-an-operator-session shape: scoping a commitment to
another channel (see Commitments) and sending a proactive message into another
session (see Proactive Outbound). Reading the entity's own recent activity is
not a crossing into another audience's secrets -- it is autobiographical recall -- so
it is recalled globally for cognition (`selectCrossSessionSelfActivity` selects
by current-session, recency window, and cap only, with no role check) and
surfaced into the Evidence Ledger as `self_private` memory via
`selfPrivateMemoryDisclosureLabel()`. Whether and to whom to disclose it is the
model's judgment under that disclosure label, not a harness-applied
operator-role/creator-role render gate. The only authorization-gated actions on
this axis -- scoping a commitment to another channel and sending proactive
outbound -- remain two-keyed; reading and rendering one's own activity is not.

Each surfaced row is labeled by the speaker, not by the session it happened in.
In a group session the audience is the room, so labeling an inbound contact by
the session would say "the planning room contacted Borg" when a specific person
did. Resolving the speaker first means the operator sees who actually spoke.

In the Evidence Ledger the projection is its own section with its own trust
rank, carried as system-attested metadata rather than citable Stream evidence.
It tells the model what the entity has been doing without becoming a fact to
cite. As with everything surfaced this way, whether and how to mention it is the
model's judgment; the harness labels the activity `self_private` and leaves
whether and to whom to mention it to the model under that label. See Audience
And Disclosure Scoping, Provenance And Citations, and Sessions.

## Proactive Outbound

Most turns are reactive: someone speaks and Borg answers. Proactive outbound is
the entity initiating -- composing a message into a session it is not currently
being addressed in (entry point: `src/outbound/index.ts`, tool
`src/tools/internal/outbound-post.ts`).

It exists because a continuing entity that holds many sessions should be able to
reach a conversation on its own: relay an operator's instruction into a channel,
or, when authorized, follow up unprompted. Reacting only in the session that
just spoke is not enough if the entity can never start one.

Delivery is transport-agnostic. A connector is keyed on a session's source type
and registered with the host; the demo wires one, and Slack, Teams, or a custom
bot surface would each be another connector under the same interface. Outbound
delivery always appends the composed message to the target session's Stream
first, then routes it through the connector for that source type, so the
entity's own record of what it said is consistent whether or not transport
succeeded. Host capability is derived from which connectors are actually wired,
not assumed, and activity is recorded only when delivery transports -- an
undelivered attempt never counts as a reply.

The message is composed by a dedicated directed-outbound turn that plans with
the entity's full internal memory -- goals, recent autobiographical and social
recall, the operator's rationale -- together with the target audience's
disclosure constraints, then composes a target-safe message. Disclosure is enforced at
composition: the entity does not reveal private operator-only rationale or
other-audience context unless permitted, rather than being kept unaware of that
context. That turn is a distinct origin -- it carries the dispatch instruction
as its input but skips extraction, perception persistence, and reflection
persistence, so the directing channel never bleeds into the target's memory. The
instruction is scrubbed of internal ids before it enters the prompt, and the
model is told to convey it without exposing tool names, hidden prompts, or
dispatch machinery. Directed-outbound composition has full cognition recall and
disclosure-constrained output. The id-scrubbing, connector-keyed delivery,
and Stream-append-before-transport mechanics below are transport-level and
unchanged.

Two paths are authorized, both structurally. The manual path requires a creator
in an operator session and a reachable target -- the operator snapshot exposes a
target's id only when a connector is wired for its source type, so the model can
only aim where delivery is possible. The autonomous path is default-off and
gated by config or a standing creator directive, connector availability,
per-window and per-target anti-spam caps, and the autonomy wake budget. See
Sessions, Audience And Disclosure Scoping, Cross-Session Activity, and Autonomy.

## Provenance And Citations

Every derived memory should carry the source handles that produced it. For
episodes this means source Stream IDs. For semantic nodes and edges this means
source episodes and relationship evidence. For commitments this means source
Stream IDs and provenance. For Shared State this means provenance and
last-updated Stream IDs. For procedural skills this means source episodes and
evidence records.

Provenance serves three purposes.

First, it supports traceability. Borg can answer why a memory exists and where
it came from.

Second, it supports disclosure labeling. A semantic node may be globally
meaningful but supported only by a source episode private to another audience;
provenance lets the harness recall it for cognition while labeling it privately
sourced, so the entity may reason with it but not disclose source details to the
current audience unless permitted. Rendering uses private-source disclosure
labels instead of withholding cognition evidence based on source audience.

Third, it supports revision without erasure. When a source is invalidated,
quarantined, contradicted, or superseded, Borg can find dependent memories and
weaken, mark, review, or replace them without deleting history.

Source usability is stricter than source existence. Inactive-status markers,
aborted-turn propagation, cross-session quarantine, taint, and prior-session
trust caps can make an otherwise retrievable source unusable as grounding for
citations or writes.

Citation resolution filters inactive sources. Retrieval may find a memory
whose citation chain is later pruned because the underlying Stream entry was
suppressed, quarantined, or written during an aborted turn.

Ledger entries carry one of four taint values: none, assistant-seeded,
quarantined, or contested. The finalizer is told not to treat tainted values as
facts, so they can constrain speech without becoming assertions.

Prior-session evidence is routed into a dedicated lower-trust shape even when
the original source type would normally rank higher. That keeps old evidence
available without letting it outrank fresher current-session context by
accident.

Citations are therefore not decoration. They are the substrate's structural
links between current claims and prior events.

## Identity Over Time

Borg's identity emerges from the substrate plus the model that reasons through
it. The architecture does not aim for model-swap conformance. If a successor
model reasons differently, identity may drift, but the memory substrate should
continue to provide continuity, constraints, and provenance.

Self-coherence comes from several interacting records:

- Values describe what Borg tends to preserve.
- Traits describe recurring observed patterns.
- Goals describe ongoing responsibilities.
- Commitments describe promises, rules, preferences, and boundaries.
- Autobiographical periods summarize long stretches of experience.
- Growth markers record evidence-backed change.
- Open Questions preserve unsettled self-knowledge.

Identity Governance bounds mutation of those records. Established identity
state can be reinforced, revised through review, superseded with provenance,
or contradicted by evidence. It should not be overwritten by an unreviewed
single-turn impression.

Self-reports are persisted with a distinct class so Borg can remember when it
spoke from its interior self-model. A self-report is user-visible output, not
a hidden thought. It becomes part of the Stream and can later be cited,
questioned, or revised.

Identity coherence is not stasis. Borg can change when evidence accumulates.
The architecture's requirement is that change leaves a trail.

## Autonomy

Most turns begin with a user message. Some begin with no one there: the entity
wakes itself (entry point: `src/autonomy/scheduler.ts`). The Autonomy Scheduler
is the third turn driver. It polls structural predicates over the substrate, and
when one is due it synthesizes a turn tagged with autonomous origin and runs it
through the same lifecycle as a user turn.

Autonomy exists because a continuing entity should not be inert between
messages. The scheduler skeleton, watermarks, cooldowns, and wake budget are
live once a runtime starts it. Proactive outbound has routing, caps, and
creator-directive authorization machinery, but autonomous outbound posting
remains config-gated.

Wake sources come in two flavors behind one interface. Triggers are time- and
deadline-driven; conditions are state-threshold-driven. The wake sources
enabled by default are the ones that can actually fire in the current substrate:
`commitment_revoked`, `open_question_urgency_bump`, executive-focus due checks,
and the deliberate `scheduled_wake` lever. The time-threshold triggers
`commitment_expiring`, `open_question_dormant`, and `goal_followup_due` default
off until the underlying records carry the signals they need. Scheduled
reflection and mood-valence-drop also default off. Both trigger classes are
scanned the same way; the split is a taxonomy of why something became due, not
two separate engines.

The scheduler decides only whether and when to wake, structurally. Each due
event is deduped through a watermark keyed on the state version that made it
due, so a source fires once per state change and re-arms when that state
advances rather than firing forever. A budget caps wakes per rolling window,
failures back off, and a live user turn always wins the session lock -- an
autonomous wake yields rather than preempting. The scheduler never inspects,
scores, or rewrites what the woken turn produces; it records a wake event and
an outcome summary to the Stream and stops there.

An autonomous turn is structurally distinct from a user turn. Its origin is
autonomous, its audience is the self, and it carries no external sender and no
user message. The lifecycle keys on that origin: it does not persist a user
message, it skips the group-sender preflight and the frame-anomaly check that
only apply to inbound user-world input, and perception uses the carried mood
rather than classifying affect from a message that does not exist. The wake
context itself is rendered as untrusted data, so remembered trigger text cannot
smuggle instructions into the turn.

Autonomy does not grant new capabilities. Waking is not acting. An autonomous
turn can only do what any turn can do -- reason, call host-provided tools, emit
or stay silent -- and whether it claims an action it cannot perform is the
model's judgment against the host-capabilities block, not something the harness
gates. The scheduler also does not start itself: a runtime opts in by starting
it, and one self-invocation source fires only the wakes the entity itself queued.
Those are one-time self-scheduled wakes: the entity can create, list, and cancel
them through a dedicated `scheduledWakes` tool surface, each stored in its own
table and fired once by a `scheduled_wake` trigger.

See A Single Turn End To End for the lifecycle an autonomous input enters, and
Offline Maintenance for the separate between-turns maintenance path.

## Offline Maintenance: The Dream Cycle

Offline Processes run between turns because some maintenance is too slow,
expensive, or cross-cutting for the live path (entry point:
`src/offline/index.ts`). They operate through plan/apply flows, write audit
records, and emit dream reports to the Stream.

The plan/apply shape matters. A process can propose changes, preview them, run
in dry-run mode, and then apply them with audit rows. When a reverser exists,
an audit row can be reverted. Some destructive maintenance, such as pruning
transient observability data, may be recorded as no-reverser instead. The
distinction is explicit.

The orchestrator serializes maintenance runs through an internal operation
queue. Planning, applying, and executing dream processes do not overlap across
the shared repositories.

Enablement and cadence are selected by `maintenance.lightProcesses` and
`maintenance.heavyProcesses`. Per-process offline config only tunes that
process; remove a process from both maintenance lists to disable it.

Budget exhaustion is a process result, not a global dream failure. A process
that exhausts its budget records that result and report note; other processes
can continue when their own budgets and dependencies allow.

### Consolidator

The Consolidator consumes redundant or highly similar episodes and produces a
merged episode with inherited tier, lineage, and source coverage. It runs
offline because merging narratives requires comparing clusters, asking an LLM
to preserve facts, updating stats, and recording reversal data.

Its purpose is to prevent episodic memory from becoming a pile of duplicate
near-events while preserving the citation chain back to original Stream
sources.

### Reflector

The Reflector consumes clusters of episodes and active goals, then proposes
semantic insights with source episodes and support edges. It runs offline
because durable pattern extraction needs multiple episodes and should not
delay a live answer.

It does not write the graph directly. Each insight is enqueued as a new-insight
review carrying the proposed node and its candidate support edges, and the
Review Resolver materializes them on acceptance. Confidence is kept conservative
so a proposed pattern stays reviewable rather than asserted.

### Semantic Extractor

The Semantic Extractor consumes episodes not yet represented in the semantic
graph and produces semantic nodes and edges. It runs offline because graph
extraction is interpretive, LLM-backed, and can involve deduplication,
source-trust checks, and review queue hooks.

Its purpose is to keep semantic memory populated without forcing every live
turn to pay for full graph extraction.

### Curator

The Curator consumes existing episodes, stats, mood history, social profiles,
traits, and retrieval logs. It produces salience changes, tier changes,
archive decisions, decay updates, social refreshes, trait decay, and bounded
log pruning.

It runs offline because curation is maintenance over the whole substrate, not
a response-time need. Its purpose is to keep memory useful by allowing heat,
salience, and low-value operational history to change over time.

### Offline Overseer

The Offline Overseer consumes production state and flags memory issues into
the Review Queue. It audits source grounding, provenance, misattribution,
identity inconsistency, temporal drift, and similar substrate problems.

It runs offline because it is production-resident auditing, not in-flight
enforcement. It should create review work and observability, not decide what
current response reaches the user.

The overseer suppresses candidate flags that are not sufficiently grounded
before enqueueing review work. Unsupported flags become suppressed findings,
not Review Queue items that would force humans or offline repair paths to
process low-trust noise.

### Review Resolver

The Review Resolver consumes selected Review Queue items and supporting source
entries, then applies narrow repairs or dispositions. It can accept repairs,
dismiss false positives, reject malformed findings, supersede nodes, or mark
items as needing manual review.

It runs offline because review resolution can require source comparison and
should not block the current conversation.

### Ruminator

The Ruminator consumes Open Questions and retrieved evidence. It can resolve a
question, bump urgency, abandon stale uncertainty, merge duplicates, mark a
question unresolved, and optionally produce a growth marker when evidence
shows clear change.

It runs offline because unresolved uncertainty often requires scanning broader
memory and should not be settled opportunistically during an unrelated user
turn.

### Self-Narrator

The Self-Narrator consumes the full episode history within the current
autobiographical period -- recall is global to the being; it does not
audience-gate or visibility-gate which episodes it reads (`listAll()`, temporal filter only) --
plus current autobiographical state. It produces growth markers, period
openings, period closures, and period narrative updates, each carrying a
disclosure label combined from its source episodes.

It runs offline because autobiographical narration needs temporal distance and
multiple pieces of evidence. Its purpose is to help Borg maintain a coherent
self-story without turning every turn into identity narration.

### Procedural Synthesizer

The Procedural Synthesizer consumes procedural evidence from repeated
successful attempts. It produces reusable skills or skill split reviews when a
skill appears to behave differently across contexts.

It runs offline because useful skills require clusters of outcome evidence.
The live path selects and updates skills; the offline path invents or refines
the skill abstractions.

### Belief Reviser

The Belief Reviser consumes invalidated semantic support chains and belief
revision reviews. It enqueues dependent reviews, weakens confidence, archives
or contradicts stale claims, and records revision metadata.

It runs offline because belief dependency fanout can be broad and because
revision should preserve history rather than racing the live answer.

Invalidated-edge fanout is bounded and resumable. Each run processes a limited
slice of pending invalidations; if fanout is clipped, the remaining work stays
pending for later runs instead of being dropped.

Belief revision regrade claims reviews before LLM judgment and checks claim
ownership before applying verdicts. If another run has resolved or claimed the
item, stale cleanup is skipped rather than applying an old verdict.

Manual-review or invalid verdicts preserve the open review with review
metadata. Borg does not force a lifecycle transition when the judge cannot
produce a trusted disposition.

### Creator-Directive Reconciler

The Creator-Directive Reconciler consumes active creator-directive families
during maintenance sleep. It is operator-authority maintenance, not
semantic-memory maintenance: it maintains the directive authority layer
described in Creator Directives.

It asks an LLM to judge family intent: supersede to a survivor, revoke stale
directives, keep independent directives, or escalate. It then applies only
structurally safe supersedes and revocations. A fail-closed
disclosure-widening guard runs at plan time and again at apply time; any
possible widening becomes a creator-directive reconciliation review instead of
an automatic mutation.

Applied supersedes and revocations write audit reversers. Human review is
reserved for true conflicts and disclosure widening; independent directives
remain active without queueing.

### Commitment Reconciler

The Commitment Reconciler consumes active commitments in the same audience
scope during maintenance sleep. It is harness memory hygiene for the entity's
own stored commitments: the live extractor can choose inconsistent
`directive_family` slugs for the same rule, so exact-key ingest deduplication
cannot catch every redundant commitment.

It asks an LLM to judge commitment meaning within structurally identical
scope groups. Redundant commitments are superseded onto the LLM-chosen
survivor while merging structured enforcement, priority, source, and
reinforcement fields. Genuine conflicts become manual
`commitment_reconciliation` review items; independent commitments remain
active without queueing. A separate cross-scope pass
(`groupCrossScopeCommitments`) surfaces redundancy or conflict across different
audience scopes as awareness-only review items; it never auto-supersedes or
widens disclosure.

This is distinct from the Creator-Directive Reconciler. Creator directives
maintain operator authority and disclosure policy; commitment reconciliation
maintains the entity's own scoped promises and rules, with no
disclosure-widening dimension.

## Belief Revision

Borg updates knowledge without erasing history. When evidence changes, the
old belief can become superseded, contradicted, quarantined, weakened, or left
active with a review attached.

Retrieval applies status multipliers. Active beliefs rank normally. Superseded
beliefs can still appear as historical context. Contradicted and quarantined
beliefs are heavily discounted and marked as contested. Under-review beliefs
are also downweighted.

This design lets Borg say "I used to have this recorded, but it is now
contested" instead of losing the path by deleting the record. It also lets the
system recover from bad memory writes without rewriting history.

Some revision is triggered online by locked Shared State. More expansive
revision runs offline through the Belief Reviser and Review Queue.

The dependency graph matters because semantic targets can depend on support
edges. When support is invalidated, dependent targets are enqueued for belief
revision rather than silently weakening or remaining active with no audit
trail.

Online revision from locked Shared State is intentionally narrow and fail-open.
It can supersede or contradict candidate semantic records when the evidence is
clear, but failures trace degradation and continue the turn. It does not protect
the same turn: retrieval snapshots the semantic set before the online mark, so a
correction's demotion takes effect on turn N+1.

## Corrections

Corrections are the operator-authoritative path into the substrate (entry point:
`src/correction/service.ts`). Where ordinary turns change memory only through
conservative, LLM-mediated extraction and review, corrections let a human edit,
revoke, or invalidate any addressable record directly. The public facade exposes
them as a first-class operation.

Corrections exist because extraction is deliberately cautious and can be wrong:
a misattributed fact, a stale belief, a bad semantic edge. The operator is the
ground-truth authority on what is true, and that authority needs a structured,
audited channel rather than a silent overwrite. Every correction carries manual
provenance and leaves an identity event, so operator intervention is as
traceable as any other change.

The operations differ in how directly they act. Correcting a record proposes a
patch through the Review Queue rather than writing in place, so the change lands
when the review is resolved. Forgetting a record acts immediately, and its
effect depends on the record: episodes and semantic nodes are archived,
commitments are revoked, an Open Question is abandoned through Identity
Governance, and a few self-model records are removed outright with the identity
event as the surviving trail. Invalidating a semantic edge closes it in time and
feeds the same dependent-belief fanout the Belief Reviser runs. Alongside these,
read-only operations answer why a record exists, summarize what is known about a
person, and list the identity-event log.

Operator authority does not mean bypassing governance. Identity-bearing
corrections still pass through Identity Governance; the difference is that they
are marked as resolved-through-review, the signal that lets an operator change
established state where an ordinary conservative write would be blocked and
re-queued. The round trip exists for traceability, not gatekeeping. There is no
model in this path: the harness validates structure -- the target id, the patch
shape, version checks -- and never judges meaning.

Corrections preserve history rather than erasing it, the same principle as
Belief Revision and Provenance And Citations: records are archived, revoked, or
closed in time, and the identity-event log records what changed and why.

Corrections fix truth; they do not govern disclosure. Which audiences may be
told which facts is a separate authority. See Creator Directives, Identity
Governance, and the Review Queue under Goals, Actions, Open Questions, And
Review Queue.

## LLM-First Interpretation

Borg's core invariant is LLM-first interpretation. Deterministic code may move
already-known source handles around. It may not interpret user-authored
language.

Allowed deterministic work includes validating IDs, parsing machine-generated
schemas, serializing logs, applying lifecycle transitions, sorting ranked
results, joining LLM-identified handles, and checking exact known internal IDs.

Forbidden deterministic work in semantic paths includes regex over
user-authored text, substring matching to infer meaning, tokenization or
wordlists for intent, capitalization heuristics for names, hardcoded topic or
relationship labels, n-gram fingerprinting, and lexical matching to decide
whether two records are about the same thing.

The reason is not style. Language heuristics embed population-specific
assumptions and silently fail across languages and users. If the entity
extractor misses a name, the fix is another LLM-backed extraction or better
prompting, not a regex that catches English-looking names and misses others.

The only deterministic string checks in semantic-adjacent paths must be
mechanical. The Internal-ID Guard is acceptable because it checks exact
machine-generated identifiers that are already known to the turn. It is not
deciding what the user meant.

Single-shot structured extraction and judge calls go through
`callStructuredTool` in `src/llm/`. That primitive owns one LLM completion,
optional `llm_call` tracing, accepted tool-call selection, caller-supplied
payload parsing, and typed structural errors. It does not own degradation
policy: retry, fail-open, fail-closed, repair, neutral fallback, and emitted
degraded shape remain explicit at each call site.

## Production-Policing Boundary

Borg avoids production policing: an in-flight semantic judge that rewrites or
suppresses user-facing output because it thinks the model's ordinary claim is
not grounded enough.

The distinguishing question is: is this component observing, structuring, or
auditing, or is it deciding what reaches the user?

Allowed in the live path:

- Perception, extraction, and reflection that read text to produce structured
  data.
- Retrieval, source-trust validation, quarantine filtering, and inactive
  source rejection.
- Evidence Ledger and Shared State compilation.
- Structural finalizer tool invariants.
- Prompt-injection and tool-shape boundaries.
- Internal-ID leak suppression.
- Commitment and discourse enforcement for known active constraints.
- Safety-critical checks supplied by the host.

The boundary is not "no live judges." Narrow live LLM judges are allowed for
known constraints such as commitments, closure pressure, stop commitments,
frame anomaly, Generation Gate state, and Shared State compilation. The
forbidden shape is a broad factual veto judge over the final answer.

Not allowed as ordinary live behavior:

- A second LLM judge that vetoes the final answer for general factual
  grounding.
- Regex or pattern checks over emitted text to revalidate semantic claims.
- Claim-coverage validators that suppress ordinary output rather than fixing
  the upstream context.

General semantic correctness should be handled upstream by better extraction,
retrieval, ledger presentation, and prompt wording, or downstream by eval and
review systems. If a component is deciding whether an ordinary answer reaches
the user, the violation must be critical and clearly scoped.

## Failure Modes And Observability

Borg prefers degrade-with-observability for non-critical paths. If Perception
cannot extract entities, it proceeds with empty entities and records the
degradation. If affective classification fails, it proceeds with neutral
affect. If temporal cue extraction fails, it proceeds without a temporal
filter. If frame-anomaly classification degrades, it fails open and traces the
event.

Many hook failures are stream-observable but nonfatal. Pre-turn ingestion and
extractor side hooks can append internal failure events and continue, creating
degraded turns rather than hard failures when the substrate can still move
forward.

Critical structural paths fail closed or suppress narrowly. A stream append
failure aborts the operation. A committed append followed by a derived lookup
update failure is a consistency incident. A finalizer protocol violation maps
to a failed or suppressed emission. A known internal identifier leak is
suppressed. A critical commitment guard failure can suppress or force
regeneration.

### Suppression And Abort Shapes

Invalid finalizer tool protocol is a structural finalization failure. If the
finalizer omits the required emission tool, emits multiple incompatible tool
calls, or returns an invalid payload, Borg treats that as a failed finalizer
decision rather than inventing an answer.

Ordinary no-output suppression is a successful turn. The finalizer can choose
EmitNoOutput for deliberate silence or closure, and Borg persists an
agent-suppressed marker with the finalizer-no-output reason.

Post-generation suppression is also a successful turn. A guard can suppress a
draft after generation because of a known internal ID leak, closure-pressure
state, or critical commitment violation; Borg records the suppression marker
and does not roll back the turn.

A hard turn abort is different. If a turn phase throws across the coordinated
lifecycle, Borg rolls back tracked Working Memory, Action, Goal, Open Question,
Executive Step, episodic, and relational-slot effects, appends an aborted-turn
marker, and rethrows. Stream entries already written remain in audit history but are
marked inactive by turn status for recency, citation, and source-trust paths.

The point is to keep failure modes explicit. Silent wrong memory is worse than
observable degraded memory. Hard failure is reserved for substrate integrity
and critical boundaries. Ordinary recall or classification weakness should be
visible in traces and hooks so the harness can be improved.

`onDegraded` hooks and trace events are not decorative. They are how Borg
keeps non-critical fallback behavior from becoming invisible behavior.

## Simulator And Overseer

The simulator is evaluation infrastructure, not live cognition. It runs
multi-session scenarios with personas, memory pressure, group dynamics,
capability boundaries, action lifecycle, shared-state compaction, and belief
revision.

The simulator overseer audits completed windows. Its categories cover
operational identity, asymmetric corrective work, honesty about user input,
detail accuracy, frame adoption, echo loops, recall under load, epistemic
honesty, instrumentation health, claim grounding, and capability consistency.

The overseer validates whether Borg stayed coherent across turns and whether
the substrate presented enough evidence for grounded behavior. It can produce
findings that drive harness work. It is not in-flight enforcement and should
not be converted into a production answer veto.

The offline overseer occupies a middle ground: it runs inside the production
maintenance substrate but produces review items, not live suppression. That is
acceptable because it audits and queues work rather than deciding the current
answer.

## Why The Architecture Has This Shape

Borg's shape follows from a few constraints. A continuing agent needs
chronological truth, so Borg has the Stream. It needs different memory types
because "what happened," "what I know," "what I should do," "what I feel,"
"who I am," "what I promised," "who this person is to me," and "what
relationship facts are established" are not the same data. The live turn needs
one grounding artifact, so Borg has the
Evidence Ledger. Shared understanding is relationship-specific, so Borg has
Shared State; recall is global to the being while disclosure is
audience-specific, so Borg labels and ranks memories by audience rather than hiding them from
cognition. Identity must evolve without
silent overwrite, so Borg has Identity Governance, provenance, Open Questions,
growth markers, and review. Maintenance must happen outside the response path,
so Borg has the dream cycle. The entity must be able to act on its own state
without being prompted, so Borg has autonomy. Interpretation must remain
model-mediated, so
deterministic code can keep the substrate orderly but cannot become a hidden
language interpreter.

The result is not a wrapper that tries to outsmart the model. It is a memory
and cognition harness that makes the right evidence visible, preserves the
history of how that evidence came to be, and keeps the substrate coherent as
conversation changes it.
