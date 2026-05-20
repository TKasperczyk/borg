# Borg Architecture

## What Borg Is

Borg is a cognitive-memory harness for an LLM. It gives the model durable,
audience-scoped memory, explicit retrieval, provenance, commitments, and
offline maintenance, then asks the model to reason with that context. It is not
a wrapper that tries to make the model intrinsically smarter. Scope decisions
are gated by the Opus 5.0 test from `CLAUDE.md`: would the failure still exist
if the base model were 10x smarter? If yes, fix the harness or prompt surface;
if no, do not add machinery that second-guesses model judgment.

The composition root is `src/borg/open.ts`, with the public facade in
`src/borg/facade.ts`. Storage is deliberately mixed: append-only JSONL stream
for chronological truth, SQLite for structured lifecycle state, and LanceDB for
vector-backed records.

## Cognitive Loop Per Turn

The per-turn entry point is `src/cognition/lifecycle/turn-phase-coordinator.ts`.
It coordinates the phase modules under `src/cognition/lifecycle/turn-phase/`.
The rule of thumb is that each phase either gathers structured context,
persists already-grounded state, or asks the model to deliberate with the
assembled evidence. A phase should not silently reinterpret user language with
deterministic heuristics.

The current turn shape is:

- Perception: `perception-phase.ts` and `src/cognition/perception/` classify
  mode, entities, affect, temporal cues, frame anomalies, and closure-loop
  state.
- Extraction: `extraction-phase.ts` runs action-state extraction, goal
  promotion, and corrective-preference extraction. It can create action, goal,
  and commitment candidates before retrieval.
- Retrieval: `retrieval-phase.ts` calls `src/cognition/retrieval/turn-coordinator.ts`
  and the unified `src/retrieval/pipeline.ts` to gather episodic, semantic,
  raw-stream, commitment, open-question, procedural, affective, and social
  context.
- Evidence ledger: `src/cognition/evidence-ledger/` renders the prompt-visible
  evidence sections used by the finalizer and compact planner.
- Shared state: `retrieval-phase.ts` compiles the audience-scoped shared-state
  artifact while building the ledger; helper logic lives in
  `shared-state-phase.ts` and `src/cognition/shared-state/`.
- Deliberation: `deliberation-phase.ts` invokes `src/cognition/deliberation/`.
  System 1 goes straight to a finalizer; System 2 produces a compact plan, may
  run secondary retrieval, then finalizes.
- Finalizer: `src/cognition/deliberation/finalizer.ts` is the only normal
  user-facing emission path. It must call exactly one emission tool:
  `EmitAnswer`, `EmitObserve`, `EmitNoOutput`, or `EmitSelfReport`.
- Post-generation: `post-generation-phase.ts` runs commitment and discourse
  guards through `src/cognition/action/turn-action-coordinator.ts`, persists
  the resulting emission or suppression marker, updates working memory, and
  starts ingestion.
- Reflection: `src/cognition/reflection/turn-reflection-coordinator.ts`
  extracts post-turn episodic, affective, social, procedural, open-question,
  and suppression effects.

Generation gates and closure-loop suppression sit around this path. They are
not general claim validators; they enforce discourse state, known commitments,
and narrow output protocol boundaries.

## Memory Bands

Borg keeps the stream separate from derived memory. The stream in `src/stream/`
is the chronological audit log and extraction input. Derived memory lives in
typed repositories and keeps provenance back to stream or episode sources.

The load-bearing memory bands are:

- Episodic: `src/memory/episodic/` stores episodes, narratives, participants,
  audience visibility, salience, heat, citation chains, and vectors.
- Semantic: `src/memory/semantic/` stores concept/entity/proposition nodes,
  typed graph edges, lifecycle status, contradiction reviews, and the review
  queue.
- Procedural: `src/memory/procedural/` stores skills, Beta posterior evidence,
  context statistics, and the Thompson-sampling selector.
- Affective: `src/memory/affective/` stores mood state and affective history.
- Self: `src/memory/self/` stores values, goals, traits, autobiographical
  periods, growth markers, and open questions.
- Commitments: `src/memory/commitments/` stores promises, rules, preferences,
  boundaries, entities, and commitment applicability.
- Social: `src/memory/social/` stores per-entity relationship and trust state.
- Working: `src/memory/working/` stores ephemeral per-session state: turn
  counters, suppression records, pending attribution, pending actions, and
  discourse control.

Two adjacent stores matter architecturally. `src/memory/actions/` tracks
finite actor-owned action state, and `src/memory/relational-slots/` tracks
constrained participant attributes. They are not replacements for the bands;
they keep operational state out of long-form narrative memory.

## Evidence Ledger

The evidence ledger is Borg's prompt-visible grounding packet. It is built by
`src/cognition/evidence-ledger/builder.ts`, compacted and rendered by
`renderer.ts`, then passed to deliberation and the finalizer. It has sections
for the current user message, current-session transcript, attribution, active
commitments, discourse state, contradictions and quarantines, action states,
group/channel memory, relational slots, retrieved raw stream evidence,
retrieved memory evidence, episodes, semantic graph, open questions, and
prior-session memory.

The reason it exists is to make grounding explicit. Retrieval results alone are
too fragmented; the finalizer needs a single ordered artifact with source
types, session scope, actor, trust rank, taint, citations, and compaction
metadata. When adding evidence, prefer a ledger section or existing evidence
entry shape over bespoke prompt text.

## Shared-State Artifact

The shared-state artifact is an audience-scoped compact state record stored by
`src/memory/decision-artifacts/`. The directory name, SQLite tables, and
storage record type are historical compatibility names; do not migrate them
just to rename the concept. The current architecture treats the record as
shared audience state, not as an old "decision artifact" ledger. The compiler
is `src/cognition/shared-state/compiler.ts`; its LLM emits
`EmitSharedStatePatch` operations to add, update, supersede, or prune entries.
`EmitDecisionArtifactPatch` remains accepted as a deprecated tool alias for
old test fixtures and trace replay compatibility.

Entries are `locked`, `live`, `tentative`, `invalidated`, or `pending`.
`locked` entries can canonicalize existing goals, commitments, actions, and
open questions through explicit ids supplied to the compiler. Deterministic
code may move those LLM-identified handles around; it must not infer semantic
equivalence itself.

The artifact has a lifecycle cap in `src/cognition/shared-state/lifecycle-cap.ts`.
The default active cap is 40 entries, with soft caps by kind, so the prompt
surface cannot grow without bound. Source trust validation rejects quarantined
or inactive stream IDs, and skipped/unsettled reconciliation is retried when
the next compile is unblocked.

## Actions, Goals, Commitments, Open Questions

Actions in `src/memory/actions/` are finite actor-owned states: considering,
committed, scheduled, completed, not done, or unknown. Use them for concrete
tasks or action assertions by Borg, a user, a participant, or a third party.
They can be linked to goals and open questions, but they are not durable
identity direction by themselves.

Goals in `src/memory/self/goals-repository.ts` are durable self-memory about
Borg's ongoing conversation or memory responsibilities. Goal promotion in
`src/cognition/goals/turn-goal-promotion-service.ts` is intentionally narrow:
one-off tasks, external responsibilities, and impossible host capabilities do
not become Borg goals.

Commitments in `src/memory/commitments/` are behavioral constraints:
promises, rules, preferences, and boundaries. P4 added the `kind`
discriminator alongside `type`. `type` describes the constraint shape
(`promise`, `boundary`, `rule`, `preference`); `kind` describes its role
(`assistant_commitment`, `audience_rule`, `participant_preference`, `boundary`,
`process_norm`). Keep both dimensions meaningful. Shared-state
canonicalization only retires canonicalizable commitment types: `promise` and
`rule`.

Open questions in `src/memory/self/open-questions.ts` represent unresolved
uncertainty, contradiction, or follow-up work. They are not facts. Retrieval
surfaces them to deliberation, contradiction-linked operational questions can
force System 2, and offline ruminator/review flows resolve them only with
evidence.

## Review Queue

The review queue in `src/memory/semantic/review-queue.ts` is where uncertain or
potentially destructive memory changes wait for a bounded decision. Review
kinds include contradiction, duplicate, new insight, misattribution, temporal
drift, identity inconsistency, correction, belief revision, and skill split.

Inputs come from semantic extraction, semantic graph writes, correction
services, self-narration, procedural synthesis, belief revision, and the
offline overseer. Enqueue hooks can also turn durable review items into open
questions via `src/memory/self/review-open-question-hook.ts`.

The queue is drained by registered handlers, manual resolution, and offline
processes. `src/offline/review-resolver/` currently auto-handles a narrow set
of misattribution, identity-inconsistency, and temporal-drift reviews. The
belief reviser handles `belief_revision` reviews. The rule of thumb is: queue
semantic uncertainty instead of patching meaning-changing memory inline.

## Semantic Belief Revision

Semantic nodes have an explicit lifecycle in `src/memory/semantic/types.ts`:
`active`, `superseded`, `contradicted`, or `quarantined`. Retrieval does not
delete old beliefs; it applies status multipliers and under-review multipliers
in `src/retrieval/semantic-retrieval.ts`, so historical and contested memory can
remain visible with lower weight.

There are two revision paths. Offline `src/offline/belief-reviser/` reacts to
invalidated support chains and `belief_revision` reviews. Online shared-state
revision is triggered by accepted locked shared-state entries in
`src/cognition/shared-state/semantic-revision.ts`: it finds nearby active
semantic nodes by vector, filters to audience-visible candidates that do not
share the same source stream, asks a conservative LLM judge, then marks nodes
superseded or contradicted through lifecycle operations.

This path is throttled deliberately. It processes at most three accepted
locked shared-state entries per turn, judges at most ten candidates per entry,
and caches `keep` / `uncertain` verdicts in a 1,000-entry cache. It is
fail-open: artifact acceptance should not depend on semantic revision
succeeding.

## Lifecycle Operations Layer

`src/memory/lifecycle-ops/` centralizes cross-repository lifecycle transitions:
canonicalize goals, commitments, actions, and open questions; complete actions;
resolve open questions; supersede commitments; and mark semantic nodes
superseded or contradicted. It also defines terminal-state checks and the
shared-state commitment canonicalization type set.

This layer exists because many flows need the same transition semantics:
shared-state reconciliation, review resolver repairs, belief revision,
ruminator resolution, and action completion hooks. New lifecycle transitions
should usually land here first so CAS behavior, provenance, tracing, and
terminal-state no-ops stay consistent.

## Simulator And Overseer

The simulator scenarios in `simulator/scenarios/` exercise multi-session,
multi-persona memory behavior: trip planning, coding incident response, family
aging-parent support, capability boundaries, action lifecycle, and belief
revision across domains. Scenario personas are motivation-driven, not scripted
message plans.

The simulator overseer in `simulator/overseer.ts` is a post-hoc audit, not a
production gate. It audits categories A-K: operational identity, asymmetric
corrective work, honesty about user input, detail accuracy, frame adoption,
echo loops, recall under load, epistemic honesty, instrumentation health, claim
grounding, and capability consistency. Category K audits whether Borg claimed
or implied unwired external, future, physical, scheduled, financial, or tool
work.

Overseer reporting is structured through `submit_overseer_verdict`: status
`healthy`, `concerning`, or `failing`; observations; recommendation; and
findings with category, claim status, source kind, impact, stream IDs, temporal
metadata where needed, and an evidence summary. Stream timestamps are the
authoritative chronology.

## Production-Policing Boundary

Borg's load-bearing boundary is that production should not contain an in-flight
semantic judge that decides whether the model's answer is grounded enough to
reach the user. `CLAUDE.md` documents the Sprint 8d.8-9 deletion of
`ManifestValidator` for that reason: claim-coverage policing had the same blind
spots as generation, added latency, and hid upstream harness gaps.

Allowed production checks are narrow: extractors classifying previous text into
structured memory, post-generation commitment/discourse enforcement for known
active constraints, structural finalizer-tool invariants, source trust
validation, quarantine/inactive-source rejection, substrate hygiene, safety,
and prompt-injection/tool-shape boundaries. Post-hoc simulator and offline
overseer audits are also allowed because they produce findings and review work,
not in-flight suppression of ordinary semantic claims.

The rule of thumb is to ask whether a component is observing, structuring, or
auditing, versus deciding what reaches the user. For non-critical semantic
correctness issues, fix retrieval, extraction, ledger presentation, or prompt
copy; do not add a second model to veto the first.

## Host Capabilities

Host capabilities are defined in `src/cognition/prompts/host-capabilities.ts`
and rendered into action extraction, goal promotion, deliberation, and
simulator overseer prompts. Borg can draft text in the current response,
remember decision-log and conversation-grounded state, and help interpret data
the user provides in the conversation.

Borg cannot edit external documents, monitor production systems, do scheduled
future work, send proactive messages or notifications, execute unwired tools,
act in the physical world, make payments or reservations, or attend real-world
events unless the host explicitly wires that capability. When a user asks for
one of those, Borg should translate it into current-turn drafting, memory
tracking, or interpretation help rather than creating Borg-owned future work.
