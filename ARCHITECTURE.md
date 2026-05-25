# Borg Architecture

## Purpose And Scope

Borg is a cognitive-memory harness for an LLM. It gives the model durable
memory, explicit retrieval, audience scoping, provenance, commitments,
identity governance, and offline maintenance. It does not try to make the
model intrinsically smarter.

The distinction matters. Borg is responsible for the substrate around the
model: what is remembered, what is shown, what is scoped to whom, what
constraints are active, and what evidence is available when the model speaks.
The model remains responsible for ordinary reasoning inside a response once it
has the right context.

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
library-level operations for turns, memory access, correction, stream access,
and dream maintenance (entry point: `src/borg/facade.ts`).

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
messages, suppressed or observed emissions, thoughts, tool calls, tool results,
perception results, internal events, and dream reports.

The Stream is the spine because every later memory record needs to be traceable
to what actually happened. A semantic belief, episode, skill update,
commitment, mood update, shared-state entry, or identity event should not be a
floating claim. It should point back to the stream entries, episodes, or review
items that justify it.

The Stream is not the only state. It is the audit trail and extraction input.
Typed repositories hold the current operational state, and vector stores make
retrieval possible. But when Borg needs to answer "where did this come from?"
the trail ends at Stream IDs.

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
references. Fallback scans can preserve correctness in small harnesses, but
production paths rely on the indexed view to keep source checks bounded by the
question being asked rather than by total session length.

Aborted-turn status propagates through that same source-trust path. Entries
written during a failed turn remain audit history, but they become inactive for
retrieval recency, citation resolution, and later source validation so failed
turn artifacts are not reused as ordinary evidence.

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

Audience visibility is intrinsic to episodes. Borg must know not only what
happened, but who was present and who may later be shown the memory. An
episode from one audience is not automatically visible to another.

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

Retrieval uses the current turn's Working Memory mood when it is available,
then falls back to the repository mood. This lets fresh Perception affect
ranking before durable affective persistence catches up.

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
not to the abstract group entity. That is separate from audience scoping for
retrieval and reply targeting: the audience can be the group while the speaker
whose trust or interaction history changes is a person.

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
general social trust. "Maya is Tom's daughter" or "Sam prefers this name" must
be retrievable as a scoped fact with provenance, not inferred from broad
sentiment or a whole episode.

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

## A Single Turn End To End

A turn begins when the harness receives a user-origin or autonomous input and
opens a coordinated turn lifecycle (entry point:
`src/cognition/lifecycle/turn-phase-coordinator.ts`). The lifecycle is ordered
so that Borg first catches up and interprets the input, then records the
turn-opening evidence, then retrieves the right context, then reasons, then
emits once, then reflects.

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

Borg then resolves the audience. In a one-to-one session the audience can be a
person. In a group channel the audience can be a group, and the current sender
must be tracked separately. This distinction prevents social state, attribution,
and reply targeting from collapsing onto an abstract channel.

For user-origin group turns, the sender is mandatory before cognition starts.
Missing sender identity fails the turn preflight rather than allowing
attribution, social updates, and reply targeting to collapse onto the group
entity.

Audience resolution affects nearly every later phase: entity resolution,
commitment applicability, social profile lookup, episode visibility, semantic
source visibility, shared-state selection, and final reply target.

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
evidence. Confirmed anomalies are recorded and quarantined so they remain
visible as events but do not become ordinary memory substrate.

If the classifier fails or returns an unusable result, the path fails open with
degraded trace data. The user entry is not quarantined on classifier failure;
ordinary turn flow continues unless an anomaly is actually classified.

### Extraction

Extraction turns the current message into structured candidates before
retrieval (entry point:
`src/cognition/lifecycle/turn-phase/extraction-phase.ts`). It can extract
corrective preferences, action state changes, goal promotion candidates, and
links between current action assertions and existing self context.

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

Semantic retrieval respects audience-visible source episodes. A semantic node
can have partial source visibility when some supporting episodes are visible
to the current audience and others are not. Borg can surface the visible part
without pretending the hidden part does not exist.

Retrieval also tracks per-session suppression. Evidence that has been recently
used or suppressed can be cooled so Borg does not repeat the same memory
reflexively.

Warm recall state keeps recently useful evidence handles available per
audience and session scope. It records reinforcement and suppression windows
so useful evidence can stay warm while repetitive resurfacing is bounded for
latency and prompt size.

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
evidence, episodes, semantic graph context, Open Questions, prior-session
memory, and Shared State.

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
audience currently share (entry point: `src/cognition/shared-state/index.ts`).
It captures decisions, live threads, tentative understandings, invalidated
claims, pending state, and locked canonical facts that matter for continuity.

Shared State is audience-scoped because "what we know" depends on who "we" is.
A private understanding with one person is not automatically shared with a
group. A group decision is not automatically a personal preference. The
audience scope propagates through retrieval, ledger rendering, semantic
visibility, and identity-sensitive responses.

The lifecycle has several conceptual states:

- Locked entries are canonical for this audience and may canonicalize existing
  goals, commitments, actions, or Open Questions.
- Live entries are active but less canonical.
- Tentative entries are plausible but not settled.
- Invalidated entries remain visible as displaced state rather than being
  erased.
- Pending entries track unresolved or awaiting state.
- Low-salience and dormant live entries are demoted live state kept under
  lifecycle pressure.

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
If a patch emits canonicalization IDs on tentative, live, pending, or
invalidated entries, normalization drops those IDs so unsettled language
cannot retire durable state.

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
the post-generation audit that inspects a drafted response for closure spans.

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

The Commitment Checker compares a message against active commitments. Critical
commitments can trigger regeneration or suppression. Advisory commitments are
observed in shadow mode. Compliant refusals and generic mentions are not
violations.

If the critical commitment judge fails, omits its tool call, or returns
invalid structure, Borg creates a fail-closed violation for the first critical
commitment being checked. Critical violations can request one narrow
regeneration before suppression; a second violation suppresses the output.

The Closure-Pressure Guard watches whether a draft response would continue a
known closure loop or violate a no-closure commitment. It can delete closure
spans, suppress output, or record shadow observations depending on mode and
context.

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
model reasoning with visible evidence.

### Persistence, Ingestion, And Reflection

After guards settle the emission, Borg persists the outcome to the Stream:
assistant message, observed marker, suppression marker, or self-report. The
turn result returned to the caller reflects that persisted emission.

Reflection then reads the turn as it actually happened (entry point:
`src/cognition/reflection/index.ts`). It updates mood, social interaction
state, pending attribution, working memory, action state, procedural attempt
tracking, Open Question effects, and post-turn reflection entries.

After Borg emits, a self-stop classifier checks whether the response committed
Borg to no further output until substantive content appears. When it does,
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
cooling repeated resurfacing that would otherwise crowd out new evidence.

Ranking is mode-conditioned. Problem-solving turns emphasize procedural and
goal-relevant evidence. Relational turns weight social and affective context
more heavily. Reflective turns surface Open Questions and self-state. Idle
turns stay lighter.

Mood congruence is a ranking signal, not a command. A non-neutral mood can
boost memories that match the current affective shape, but it should not hide
important contradictory evidence.

Audience scope is enforced before evidence becomes prompt-visible. Episodes
and semantic sources must be visible to the current audience unless an
explicit cross-audience operation is requested through an administrative API.
For group audiences, visibility does not automatically expand to every
participant's private memory. Group turns see group-scoped or global state
unless another mechanism explicitly resolves participant-specific context.

The result of retrieval is not dumped directly into the model. It is assembled
into the Evidence Ledger so the finalizer can see evidence classes,
attribution, constraints, conflicts, and citations in one ordered artifact.

## Commitments And Constraints

Commitments are sourced from explicit API operations, user corrections,
corrective preference extraction, identity-governed writes, and Shared State
canonicalization. Each commitment records what was promised or constrained,
who it applies to, who made it, what audience it restricts, what entity it is
about, and what Stream entries justify it.

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
considering, committed to do, scheduled, completed, not done, unknown, or
archived. They are used for concrete tasks and assertions, not for durable
identity direction.

Open Questions represent unresolved uncertainty. They are not facts. An Open
Question can be created by reflection, contradiction detection, review hooks,
rumination, overseer findings, or user input. Retrieval surfaces relevant Open
Questions so Borg can avoid speaking as if uncertainty were settled.

Open Questions have only open, resolved, and abandoned states. Completing an
Action can resolve its linked Open Question, and it can also resolve Open
Questions under a linked Goal through identity-governed resolution.

The Review Queue holds memory changes that are uncertain, potentially
destructive, or semantically risky. Reviews can cover contradiction,
duplicate, new insight, misattribution, temporal drift, identity
inconsistency, correction, belief revision, and skill splits. Queueing a
review is preferred over silently patching meaning-changing memory inline.

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

## Audience Scoping

Audience scoping is a first-class invariant. Borg tracks who said something,
who heard it, who is being addressed, and who may later see the memory.

The audience can be null or global, a person, a group, or self. The sender can
be different from the audience in group contexts. The reply target can be a
specific entity within a group. These distinctions propagate into Stream
entries, Social Memory, Commitments, episodic visibility, semantic source
visibility, Shared State, retrieval, and the Evidence Ledger.

Group audience scope does not imply participant-private visibility. A group
turn may include participant roster context and constrained relational slots,
but it does not automatically retrieve each participant's private memories.

Audience scoping prevents several classes of failure:

- leaking private context from one person into a different audience,
- attributing a group statement to a single participant,
- updating a group's social profile when one member spoke,
- treating a participant preference as a channel rule,
- assuming that a memory shared with Borg is shared with everyone.

Identity also has audience scope. Borg can have global self-memory, but some
Open Questions, commitments, or shared states are specific to the relationship
with a particular audience. A coherent identity does not require the same
state to be visible to every audience.

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

Second, it supports audience filtering. A semantic node may be globally
meaningful, but if its only source episode is private to another audience, it
should not be rendered as evidence for the current one.

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

Ledger entries carry taint values such as none, assistant-seeded, quarantined,
and contested. The finalizer is told not to treat tainted values as facts, so
they can constrain speech without becoming assertions.

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

Its purpose is to convert repeated experience into semantic memory while
keeping confidence conservative and reviewable.

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

The Self-Narrator consumes identity-visible episodes and current
autobiographical state. It produces growth markers, period openings, period
closures, and period narrative updates.

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
clear, but failures trace degradation and continue the turn.

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
episodic, and relational-slot effects, appends an aborted-turn marker, and
rethrows. Stream entries already written remain in audit history but are
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
Evidence Ledger. Shared understanding is audience-specific, so Borg has
Shared State and audience-scoped retrieval. Identity must evolve without
silent overwrite, so Borg has Identity Governance, provenance, Open Questions,
growth markers, and review. Maintenance must happen outside the response path,
so Borg has the dream cycle. Interpretation must remain model-mediated, so
deterministic code can keep the substrate orderly but cannot become a hidden
language interpreter.

The result is not a wrapper that tries to outsmart the model. It is a memory
and cognition harness that makes the right evidence visible, preserves the
history of how that evidence came to be, and keeps the substrate coherent as
conversation changes it.
