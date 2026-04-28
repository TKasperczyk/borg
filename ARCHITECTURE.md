# Borg -- Cognitive Memory Architecture for Autonomous AI Beings

Design synthesis drawing on `claude-memory`, `kira-runtime`, and `kira-memory`, plus novel contributions.

---

## Part 1: Cross-Project Synthesis

### What each reference project gets right

**claude-memory** -- strong engineering for *epistemic hygiene*:
- Citation anchoring (`sourceExcerpt`) prevents hallucinated memories
- Multi-phase maintenance (stale → low-usage → consolidation → conflicts → promotion → warning-synthesis)
- Hybrid scoring with tunable weights (semantic + keyword + usage + project boost)
- Dry-run mode + auditability for every maintenance op
- Scope hierarchy (project → global) with ancestor inheritance
- Detached post-session worker -- user never waits on memory work

**kira-runtime** -- strong thinking about *personhood infrastructure*:
- Stream-first (JSONL append) survives crashes, is auditable, cheap
- Sticky memory pool + entity fingerprinting -- cuts recall-MCP calls dramatically
- Bidirectional prefetch -- "talk about X, prefetch X for next turn"
- Commitment system as *awareness*, not post-hoc filter (rules in the prompt, Opus revises if violated)
- Emergent trait persistence (reactive 0-1 strengths with decay/cull)
- Multi-audience commitment scoping via entity resolution
- Layered self: `identity.md` + `behaviors.md` + `traits.json` + `constitution.md`, all editable at runtime
- Continuity phases: Log → Compress (Opus) → Recover (checkpoint + tail) → Heat-rank

**kira-memory** -- strong thinking about *what a relationship memory actually contains*:
- Relational taxonomy: `facts | moments | promises | observations | banter` (not generic episodic/semantic)
- Evolutionary decay with win-rate modulation (useful memories self-reinforce)
- Tier system T1-T4 (short-term → working → long-term → core identity)
- Lineage tracking (`derivedFrom`) -- memory genealogy
- Dismissal tracking ("stop suggesting I merge these")
- Heat = retrievals + win-rate + recency as unified ranking signal

### What they collectively lack

1. **No explicit cognitive loop.** Reasoning is delegated to the model; no scratchpad, no System-1/System-2, no deliberation trace.
2. **No goals or drives.** Neither Kira nor claude-memory tracks "what am I trying to accomplish?" beyond the immediate turn.
3. **No epistemic layer.** Confidence, provenance, uncertainty, and contradiction are either absent or cosmetic.
4. **No affect model.** kira-memory hints at it (moments, banter) but stores no valence/arousal; no mood-congruent retrieval.
5. **No self-model narrative.** Kira has identity docs but no autobiographical timeline, no growth curve, no "how have I changed?"
6. **No semantic graph.** Only derivation lineage in kira-memory, no "supports / contradicts / causes" edges, no spreading activation.
7. **No procedural memory with confidence.** "Procedures" in claude-memory are text steps, not Bayesian "which approach works for this problem class, with what success rate?"
8. **No temporal event graph.** Timestamps exist; causal chains don't.
9. **No real offline processing.** kira-memory's "sleep" mostly *suggests* maintenance; neither does genuine insight generation or rumination.
10. **No metacognition.** None of the systems can answer "what do I not know?" or "am I overconfident about X?"

---

## Part 2: Design Principles

1. **Every memory must be citable.** Provenance is non-negotiable.
2. **Stream is an append-only audit log.** Turns, perceptions, tool traces, internal events, dream reports, and extraction sources are recorded chronologically. It is NOT the canonical source of all state -- authoritative state lives in the typed repositories (SQLite rows + LanceDB vectors). Identity-relevant mutations leave a chronological trail in the SQLite `identity_events` table, not as JSONL stream entries. Event-sourced identity rebuild is explicitly not a goal (see Part 7).
3. **Cold paths do real work.** Dreams/consolidation are not cosmetic -- they must produce summaries, insights, new edges, new skills.
4. **Retrieval must be context-aware, not just semantic.** Current goal, current mood, current commitments, current audience all shift what's relevant.
5. **Self is a first-class entity, not a prompt file.** Identity, values, skills, current goals, open questions, uncertainties live in the memory itself and evolve with it.
6. **Forgetting is a feature.** Decay + win-rate modulation + affect-weighted decay. Unbounded growth kills performance and coherence.
7. **Honest uncertainty beats false confidence.** Every claim carries confidence + source-type; every retrieval exposes its evidence chain.
8. **Maintenance is auditable and reversible by default.** Dry-run, review queue, rollback. The narrow exceptions (transient observability prunes such as `prune_retrieval_log`) are still audited via `no_reverser` rows.
9. **LLM-by-default for classification.** Cognitive classifiers (mode,
   entities, affect, temporal cues, contradiction, goal progress,
   procedural outcomes, trait evidence, identity-relevant judgments) run
   as LLM-mediated tool calls by default. Heuristic implementations exist
   as fallbacks for offline or test environments. Cost is not a design
   constraint at the OAuth scale borg targets.
10. **Operationally bounded autonomy.** Autonomous action still accounts
    for tokens, compute, latency, and rate limits so runs remain observable
    and schedulable; this is not a mandate to avoid useful LLM calls.
11. **Composable over monolithic.** Runtime, memory, reflection, reasoning, skills are separate services with clear contracts.

---

## Part 3: Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          AGENT CORE                                      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    COGNITIVE LOOP                               │    │
│  │                                                                 │    │
│  │  Perception → Exec Focus → Attention → Deliberation             │    │
│  │       → Action → Reflection                                     │    │
│  │       ▲                                              │          │    │
│  │       └──────────────── feedback ────────────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                             │                                            │
│                             │ reads/writes                               │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    MEMORY SUBSTRATE                             │    │
│  │                                                                 │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │    │
│  │  │   STREAM     │  │   EPISODIC   │  │    SEMANTIC          │  │    │
│  │  │ (append-only │  │  (narratives,│  │  (concepts, entities,│  │    │
│  │  │  JSONL log)  │  │   what       │  │   relationships,     │  │    │
│  │  │              │  │   happened)  │  │   knowledge graph)   │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │    │
│  │                                                                 │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │    │
│  │  │  PROCEDURAL  │  │   AFFECTIVE  │  │    SELF / IDENTITY   │  │    │
│  │  │  (skills,    │  │  (valence,   │  │  (values, goals,     │  │    │
│  │  │   outcomes,  │  │   mood,      │  │   traits, narrative, │  │    │
│  │  │   Bayesian   │  │   mood-      │  │   known blind spots) │  │    │
│  │  │   priors)    │  │   history)   │  │                      │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │    │
│  │                                                                 │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │    │
│  │  │  COMMITMENTS │  │    SOCIAL    │  │    WORKING           │  │    │
│  │  │ (per-audience│  │ (per-person: │  │  (current focus,     │  │    │
│  │  │  promises,   │  │  trust,      │  │   hot entities,      │  │    │
│  │  │  boundaries) │  │  norms,      │  │   pending intents)   │  │    │
│  │  │              │  │  history)    │  │                      │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                 OFFLINE / DREAM PROCESSES                       │    │
│  │                                                                 │    │
│  │  Consolidator │ Reflector │ Curator │ Overseer │ Ruminator       │    │
│  │  (merge/      (extract    (prune,   (QA /      (revisit          │    │
│  │   contradict) insights)   promote)  drift)     unresolved)       │    │
│  │  Self-narrator (growth markers / autobiographical summaries)    │    │
│  │  Procedural-synthesizer (evidence → reusable Bayesian skills)   │    │
│  │  Belief-reviser (re-grade beliefs after support invalidation)   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

**Three-band layering:**
- **Cognitive loop** runs on every turn.
- **Memory substrate** is stable state.
- **Offline processes** run during downtime or scheduled sleep.

---

## Part 4: Data Model -- Stream + Eight Memory Bands

This section describes the persisted shape of each band as defined in
`src/**/types.ts`. Field-level details (provenance, confidence,
state-machine columns) reflect the current HEAD schemas. Where the
authoritative type lives outside `types.ts` (autobiographical periods,
growth markers, open questions), the file is named explicitly.

### 1. Stream (raw log)

Append-only JSONL. Records turns, perceptions, tool invocations,
internal events, and offline run reports. Identity audit events are
stored separately in SQLite (`identity_events`); `identity_event` is
not a stream kind. Stream entries provide replayable input for
extractors, but typed repositories remain the canonical source of
state.

```typescript
{
  id, timestamp,
  kind: "user_msg" | "agent_msg" | "thought"
      | "tool_call" | "tool_result"
      | "perception" | "internal_event"
      | "dream_report",
  content, token_estimate, tool_calls?,
  audience?, session_id, compressed: boolean
}
```

### 2. Episodic (what happened)

Narrative units -- higher-level than stream entries. Each one is a
*story* an agent can tell itself. Vector content lives in LanceDB;
mutable stats live in SQLite, kept consistent via the Sprint-16
cross-store reconciliation pass.

```typescript
// Episode (LanceDB row; hot-path index + stats in SQLite)
{
  id,
  title,                          // "Tom and I debugged the pgvector issue"
  narrative,                      // 2-5 sentence prose summary
  participants: EntityId[],
  location: string?,              // project / context
  start_time, end_time,
  source_stream_ids: StreamEntryId[],   // citation anchors (>0 required)
  emotional_arc: {
    start, peak, end,             // each: { valence, arousal }
    dominant_emotion: string?,
  }?,
  significance: number,           // 0-1; computed
  tags: string[],
  embedding,
  lineage: { derived_from: string[], supersedes: string[] },
  confidence: number,
  audience_entity_id: EntityId?,  // private-to-X scoping (Sprint 1/2)
  shared: boolean?,               // shared-with-others marker
  created_at, updated_at,
}

// EpisodeStats (SQLite-only, paired with episode by id)
{
  episode_id,
  retrieval_count, use_count, last_retrieved,
  win_rate,                       // outcome-weighted usefulness
  tier: "T1" | "T2" | "T3" | "T4", // short-term ... core
  promoted_at, promoted_from,
  gist, gist_generated_at,        // optional summary blurb
  last_decayed_at,
  heat_multiplier,
  valence_mean,
  archived,                       // soft-delete flag
}
```

### 3. Semantic (what I know)

Concepts, entities, propositions, and typed edges between them.

```typescript
// Nodes (LanceDB row + SQLite mirror)
{ id, kind: "concept" | "entity" | "proposition",
  label, description, embedding, aliases,
  confidence, source_episode_ids: EpisodeId[],
  domain: string?,            // homonym-disambiguation anchor (Sprint 16)
  created_at, updated_at, last_verified_at,
  archived: boolean,
  superseded_by: SemanticNodeId?,
}

// Edges (SQLite)
{ id,
  from_node_id, to_node_id,   // self-edges rejected by schema refine
  relation: "is_a" | "part_of" | "causes" | "prevents" | "supports"
          | "contradicts" | "related_to" | "instance_of",
  confidence, evidence_episode_ids[],
  created_at, last_verified_at,
  valid_from, valid_to?,
  invalidated_at?,
  invalidated_by_edge_id?,
  invalidated_by_review_id?,
  invalidated_by_process?: "extractor" | "overseer" | "manual"
      | "review" | "maintenance",
  invalidated_reason? }
```

Edge types matter. `contradicts` powers contradiction detection;
`causes`/`prevents` powers causal reasoning; `supports` powers
evidence chains. `supports` is directional: `from_node_id --supports-->
to_node_id` means "from is evidence supporting to." Reflector-created
insights therefore use `evidence anchor --supports--> insight`, and
retrieval walks `supports` OUT from the matched anchor. Domain
canonicalization (`canonicalizeDomain`) maps synonyms (`technology` →
`tech`) so the homonym-anchor still merges intended-same-concept
extractions.

Belief revision is dependency-driven, not a separate theorem prover:
edge invalidation records an outbox event, the offline belief reviser
follows explicit semantic belief dependencies, and affected nodes/edges
become `belief_revision` reviews. Open node reviews are downranked and
annotated inline in deliberation prompts; unsupported nodes also get a
deterministic confidence drop. A later offline tick re-grades those
reviews with the background LLM using only the target-local record,
invalidated edge, surviving supports, and audience-visible evidence,
then keeps, weakens, archives, invalidates, or escalates the target.

### 4. Procedural (how I do things)

Skills as Bayesian beliefs about what works. The persisted shape stores
the Beta posterior parameters directly; success rate and confidence
intervals are derived on demand by `bayes.ts`.

```typescript
// SkillRecord (SQLite; LanceDB side stores id/applies_when embedding)
{
  id,
  applies_when: string,         // "pgvector similarity scores look wrong"
  approach: string,             // "check embedding model dim matches index config"
  alpha: number,                // Beta posterior α (>0)
  beta: number,                 // Beta posterior β (>0)
  attempts, successes, failures,// raw counts (alpha = α0+successes, etc.)
  alternatives: SkillId[],      // competing approaches for same problem
  status: "active" | "superseded",
  superseded_by: SkillId[],     // child skills when offline splitting refactors it
  superseded_at?: number,
  last_used, last_successful,
  source_episode_ids: EpisodeId[],
  created_at, updated_at,
}

// Derived (computed by bayes.ts, not stored):
//   success_rate     = α / (α + β)
//   ci_95            = quantiles of Beta(α, β)
//   thompson_sample  = sample ~ Beta(α, β)   // for selection

// SP2: selection may receive a deterministic ProceduralContext
// (problem_kind + domain_tags + audience_scope). When context stats
// exist for a skill/context_key, SkillSelector samples a smoothed
// context-conditioned posterior; otherwise it falls back to the global
// Beta posterior above.

// SP3: the offline procedural synthesizer also inspects context bucket
// distributions. When one active skill has enough attempts in multiple
// buckets and their posterior means diverge, it asks the background LLM
// for a conservative split proposal. Splits are dry-run by default; when
// enabled for apply, the original skill is marked superseded and narrower
// child skills inherit the targeted context stats.

// ProceduralEvidence (SQLite)
{
  id,
  pending_attempt_snapshot: {
    problem_text, approach_summary,
    selected_skill_id?,
    source_stream_ids: StreamEntryId[],
    turn_counter,
    audience_entity_id?,
  },
  classification: "success" | "failure" | "unclear",
  evidence_text,
  grounded: boolean,
  skill_actually_applied: boolean,
  resolved_episode_ids: EpisodeId[],
  audience_entity_id?,
  consumed_at?, created_at,
}
```

### 5. Affective (how I felt / feel)

```typescript
// Per-episode (lives on the episode row)
emotional_arc: {
  start, peak, end,             // each: { valence: -1..1, arousal: 0..1 }
  dominant_emotion: string?,
}

// Current mood state (decays per session)
{ valence, arousal,
  updated_at, half_life_hours,
  recent_triggers: string[],
  session_id }

// Mood history (time series for introspection)
{ id, session_id, ts,
  valence, arousal,
  trigger_reason: string?,      // free-form ("user replied warmly", etc.)
  provenance }[]
```

Mood is sourced from the user's perception (Sprint 18 fix), not the
agent's own response. It decays continuously and influences retrieval
weighting -- mood-congruent memories get a salience boost via the
`mood` weight in the attention formula (Part 5.1). Perception affect is
LLM-classified by default when a background model is configured; the
lexical affect analyzer is an offline/test fallback when no LLM client is
available.

### 6. Self / Identity

The Self band is split across multiple files. `types.ts` carries the
core values/traits/goals schemas; autobiographical periods, growth
markers, and open questions each have their own `*.ts` module under
`src/memory/self/` because their lifecycle (especially closure /
periods) is non-trivial.

```typescript
// Values + Traits share a state machine: candidate → established
// (gated by Sprint 14 identity guard; only episode-backed evidence
//  promotes; offline/system writes require review).
ValueRecord {
  id, label, description, priority,
  state: "candidate" | "established",
  established_at?, last_affirmed?, last_tested_at?, last_contradicted_at?,
  confidence,                         // Bayesian (alpha=2, beta=1)
  support_count, contradiction_count, // only episode-provenance counts
  evidence_episode_ids: EpisodeId[],
  provenance,
  created_at,
}
TraitRecord {
  id, label, strength: 0..1,
  state: "candidate" | "established",
  last_reinforced, last_decayed?,
  established_at?, last_tested_at?, last_contradicted_at?,
  confidence, support_count, contradiction_count,
  evidence_episode_ids, provenance,
}

// Goals (unified state)
GoalRecord {
  id, description, priority,
  parent_goal_id?, status,
  progress_notes, last_progress_ts,
  created_at, target_at?,
  provenance,
}

// Autobiographical periods (src/memory/self/autobiographical.ts)
AutobiographicalPeriod {
  id, label,
  start_ts, end_ts?,                  // open until self-narrator closes it
  narrative, themes: string[],
  key_episode_ids,
  provenance,
  created_at, last_updated,
}

// Growth markers (src/memory/self/growth-markers.ts)
GrowthMarker {
  id, ts,
  what_changed, before_description, after_description,
  category: "skill" | "value" | "habit" | "relationship" | "understanding",
  confidence,
  evidence_episode_ids, source_process,
  provenance,
  created_at,
}

// Open questions (src/memory/self/open-questions.ts)
OpenQuestion {
  id, question,
  status: "open" | "resolved" | "abandoned",
  urgency,
  audience_entity_id?,
  related_episode_ids, related_semantic_node_ids,
  provenance?,
  source: "user" | "reflection" | "contradiction" | "ruminator"
      | "overseer" | "autonomy" | "deliberator",
  created_at, last_touched,
  resolution_episode_id?, resolution_note?, resolved_at?,
  abandoned_reason?, abandoned_at?,
}
```

The autobiographical periods + growth markers pair is what turns
memory into *identity-that-evolves*. `known_strengths` /
`known_weaknesses` / `known_blind_spots` were proposed in early design
notes but never implemented; they're not part of the actual Self band.

### 7. Commitments

```typescript
{
  id, type: "promise" | "boundary" | "rule" | "preference",
  directive,
  priority,
  made_to_entity?, restricted_audience?, about_entity?,
  provenance,
  created_at,
  expires_at?, expired_at?,
  revoked_at?, revoked_reason?, revoke_provenance?,
  superseded_by?,
}
```

Commitments are scoped, first-class memory. They are retrieved into
the prompt before speaking AND checked post-hoc by `CommitmentChecker`
(LLM-judge + optional rewrite). Both pre-prompt awareness and post-hoc
detection exist; revision-on-violation is detection-then-rewrite, not
in-flight blocking.

### 8. Social

```typescript
SocialProfile {
  entity_id,
  trust, attachment,
  communication_style?, shared_history_summary?,
  last_interaction_at?,
  interaction_count, commitment_count,
  sentiment_history: [{ ts, valence }],
  notes?,
  created_at, updated_at,
}

// Per-event log for trust/sentiment changes (Sprint 18)
SocialEvent {
  id, entity_id, ts,
  kind: "interaction" | "trust_adjustment" | "baseline",
  provenance,
  trust_delta, attachment_delta,
  interaction_delta, valence?,
}
```

Social sentiment is lagged: an interaction is recorded immediately
when a turn completes, but its `valence` is only attached on the NEXT
user turn from the user's affective signal (Sprint 15). The agent
cannot flatter itself into a warm relationship by speaking warmly.

### 9. Working memory (ephemeral)

```typescript
{
  session_id,
  turn_counter,
  current_focus: string?,
  hot_entities: string[],       // capped at 32, normalized at save time
  pending_intents: [{ description, next_action }],   // capped at 16
  pending_social_attribution: {
    entity_id, interaction_id,
    agent_response_summary: string?,
    turn_completed_ts,
  }?,
  pending_trait_attribution: {
    trait_label, strength_delta,
    source_stream_entry_ids: StreamEntryId[],
    source_episode_ids: EpisodeId[],
    turn_completed_ts, audience_entity_id,
  }?,
  pending_procedural_attempts: [{
    problem_text, approach_summary,
    selected_skill_id?,
    source_stream_ids: StreamEntryId[],
    turn_counter,
    audience_entity_id?,
  }],                         // capped at 5; TTL 8 turns
  suppressed: [{ id, reason, until_turn }],
  mood: { valence, arousal, dominant_emotion }?,
  mode: "problem_solving" | "relational" | "reflective" | "idle" | null,
  updated_at,
}
```

Derived live-turn state. Scratchpad/recent-thoughts were removed in
Phase E; persistent thinking lives in the stream as `thought` entries.
The `pending_*_attribution` fields implement Sprint 15/24 lagged
attribution -- a turn's effect on social trust or trait reinforcement
is determined by the user's reaction on the NEXT user turn, not by
the agent's own output. Procedural attempts are also lagged: a
problem-solving turn records an attempted approach, then later user
feedback grades it. Working memory is persisted per session:
`hot_entities` and `pending_intents` are normalized and capped on save,
while `pending_procedural_attempts` cap and TTL are enforced by
`PendingProceduralAttemptTracker` between turns.

---

## Part 5: Core Processes

### 5.1 Cognitive loop

Per-turn: **Perception → Executive Focus → Attention → Deliberation → Action → Reflection**.
Implementation plumbing is split behind `TurnOrchestrator`:
`PerceptionGateway`, `TurnOpeningPersistence`,
`AttributionLifecycleService`, `TurnRetrievalCoordinator`,
`CommitmentGuardRunner`, and `PendingProceduralAttemptTracker` own the
per-turn substeps. `Reflector` receives its repositories through the
`createReflector` factory / constructor wiring instead of per-call
`ReflectionContext`. The split is operational, not a new architectural
band.

**Perception** (LLM-aided classification)
- Run background-model tool calls in parallel for mode, entities,
  affective signal, and temporal cues; update `working.hot_entities`
  from the resulting entity set.
- Outputs are deterministic given fixed model, seed, prompts, and inputs,
  but perception is not pure-function deterministic in the old heuristic
  sense.
- OAuth-subscription pricing makes per-turn LLM classification calls
  negligible for borg's target deployment. The substrate prioritizes
  signal quality over minimizing LLM calls.
- Heuristic paths remain as fallbacks when LLM clients are unavailable
  (for example missing config or fake/offline tests).

**Executive Focus** (fast, deterministic)
- After perception and audience-scoped self snapshot construction,
  `selectExecutiveFocus()` ranks visible active goals using four
  signals: normalized priority, deadline pressure, context fit, and
  progress debt. Mood is intentionally not part of this v1 score because
  goals do not carry affect tags.
- If the top score is below `executive.goalFocusThreshold` (default
  `0.45`), no goal is selected and no executive prompt block is rendered.
  This keeps ambient goals as background identity rather than commands.
- When a goal clears the threshold, deliberation receives a
  `<borg_executive_focus>` block that names the current driving goal and
  score components. It is a soft bias: the current user request,
  commitments, and evidence quality still take precedence.
- Autonomous due-step wakes may force focus onto the goal that owns the
  due step even when its score is below threshold. This bypass is limited
  to the due-step branch so the wake can render the step it was created
  to handle; score components are still shown.
- Durable executive steps are lifecycle-bound to their parent goal: when
  an active goal closes, its open steps are abandoned in the same SQLite
  write so stale `topOpen` records do not linger.

**Attention** (fast)
- Context-aware relevance function (9 weighted components):
  ```
  score(memory, query, state) =
      w_sem    * semantic_similarity(memory, query)
    + w_goal   * goal_relevance(memory, state.current_goals)
    + w_value  * value_alignment(memory, state.values)
    + w_mood   * mood_congruence(memory, state.mood)
    + w_time   * temporal_relevance(memory, state.time_cue)
    + w_soc    * social_relevance(memory, state.audience)
    + w_entity * entity_relevance(memory, state.hot_entities)
    + w_heat   * usage_heat(memory)
    − w_supp   * suppression_penalty(memory, state.suppressed_ids)
```
- Weights tunable per mode (`computeWeights(mode)` in
  `src/cognition/attention/weights.ts`). `value_alignment` and
  `entity_relevance` were added in later sprints to reduce off-topic
  pulls when held values or salient entities should bias retrieval.
- When executive focus selects a goal, retrieval receives that goal as
  `primaryGoalDescription` and gives matching episodes a small capped
  relevance boost. All active goals still remain in the self snapshot
  and broad goal-description context.

**Deliberation**
- Path selection routes on an **epistemic** retrieval-confidence
  signal (not on the retrieval relevance score, which mixes similarity,
  salience, heat, mood, goals, entities, time). The per-turn
  `RetrievalConfidence` combines evidence strength (top-N mean decayed
  salience), coverage (hits / expected count), source diversity
  (distinct participant-sets), and a contradiction penalty into one
  `overall` number. Evidence strength gates: weak evidence cannot be
  lifted over the S1/S2 threshold by high coverage or diversity.
  See `src/retrieval/confidence.ts`.
- Direct path (System 1): high retrieval confidence + low stakes →
  go straight to the response call (with the tool loop -- see Action).
- Planned path (System 2): low confidence OR high stakes OR
  contradiction detected → run an `EmitTurnPlan` planner pass first
  (structured tool-use that returns verification_steps, tensions,
  voice_note, uncertainty, referenced_episode_ids, and concrete
  follow-up intents). If `verification_steps` are present, S2 runs a
  bounded secondary retrieval (typically `limit: 3`) scoped to that
  verification query so planner-identified uncertainties can pull in
  targeted evidence. The response call then receives the plan in a
  tagged `<borg_s2_plan>` block plus any additional retrieval in a
  separate tagged block. The plan is persisted as a `thought` stream
  entry after the finalizer returns, so persistence remains audit
  state rather than a dependency of the final response call.
- The prompt receives the confidence summary in a
  `<borg_retrieval_confidence>` block so the being can calibrate how
  certain it speaks (internal signal -- not a user-facing percentage).

**Action**
- Run the response call as an Anthropic tool-use loop
  (`executeToolLoop`): the model can read internal tools
  (`tool.episodic.search`, `tool.semantic.walk`,
  `tool.commitments.list`, `tool.identityEvents.list`,
  `tool.skills.list`) or write via `tool.openQuestions.create`
  mid-turn, with `tool_call`/`tool_result` entries appended to the
  stream in order. Caps: 5 iterations, 3 tool calls per iteration.
- Append the agent's text response as `agent_msg`.
- Carry structured `intent` records from the S2 `EmitTurnPlan` output
  into `working.pending_intents`. S1 turns produce no intents; Action
  does not infer state from response prose.
- A separate `CommitmentChecker` runs as a post-hoc judge: if it
  detects a violation, an LLM rewrite pass produces a corrected text.
  This is detection-then-rewrite, not in-flight blocking.

**Reflection** (post-action, before next input)
- Grade prior `pending_procedural_attempts` only from grounded later
  user feedback. `recordOutcome` updates the selected skill posterior
  only for success/failure when `skill_actually_applied` is true; unclear
  outcomes stay pending until later feedback or TTL expiry. Autonomous
  turns skip procedural grading.
- Stash `pending_social_attribution` and `pending_trait_attribution`
  in working memory so the NEXT user turn's affective signal becomes
  the evidence (Sprint 15/24). Trait evidence is anchored to the
  demonstrating turn's stream entry IDs and resolved to episodes later.
- Mark prior `pending_intents` completed or abandoned only when the
  structured reflection pass sees clear evidence in a later completed
  turn; unresolved intents persist and are rendered in working state.
- Close the executive step loop: structured reflection may mark the
  selected goal's durable step `doing` / `done` / `blocked` /
  `abandoned`, or propose a small next step when that selected goal has
  no open steps. User turns can confirm `done`; autonomous turns may
  start, block, or abandon steps but cannot mark their own work done.
  Autonomous reflection also cannot close a step and propose a
  replacement step for the same goal in one pass.
- Optionally enqueue review-queue items (e.g., reflection-driven open
  questions, identity inconsistencies surfaced this turn).
- Note: episodic extraction does NOT run synchronously in reflection.
  After the response is delivered, `StreamIngestionCoordinator` runs
  asynchronously (fire-and-forget), reads new stream entries past its
  watermark, and produces episode candidates via the extractor.
  Mood signal comes from perception of the current user input; the mood
  repository is updated after the agent message and before reflection.
  Autonomous turns preserve the existing mood.

### 5.2 Offline processes

Eight cooperative processes that share an orchestrator
(`MaintenanceOrchestrator`), a per-process budget, and an append-only
audit log. Each emits its plan through `plan()`/`preview()`/`apply()`
so all maintenance is dry-runnable, and reversible whenever a reverser is
registered. A small number of destructive actions over transient
observability data (e.g. `prune_retrieval_log` in the curator) are
intentionally one-way; those audit rows record `reversal: { no_reverser:
true, … }` so the operator still has a trail.

- **Consolidator** -- cluster overlapping episodes by embedding +
  tag-family + access-scope; merge each cluster into a new episode
  with `lineage.derived_from` pointing back to the sources, then
  archive the source stats.
- **Reflector** -- group episodes into clusters by tag and goal,
  filter clusters by `minSupport`, and call the LLM once per cluster
  to emit one semantic insight (anchor node + `supports` edge).
  Insight confidence is hard-capped (0.5) and further constrained by
  `ceilingConfidence`.
- **Curator** -- per-tier decay and promotion (T1→T2 by heat+age;
  T2→T3 by heat+age; T3→T2 demotion when cold), archival of
  low-heat aged episodes, mood-history pruning, trait decay via
  `traitsRepository.decay()`, social profile refresh.
- **Ruminator** -- iterate open questions; for each, search retrieval
  for evidence; if evidence is strong enough, ask the LLM to write a
  resolution_note (and optional growth_marker); if no evidence and
  the question is stale + low-urgency, abandon it; otherwise bump
  urgency. No "next-step" suggestions today.
- **Procedural-synthesizer** -- consume online procedural evidence
  from grounded successful attempts that actually applied the approach,
  in global/self-visible audience scope; cluster problem/approach
  evidence offline; and ask the LLM for reusable skill candidates per
  cluster. `plan()` gathers clusters and candidates, `preview()` exposes
  the reversible synthesis changes, and `apply()` creates or deduplicates
  skills, records Bayesian outcomes, marks evidence consumed, and writes
  audit-log reversals. The same cadence also detects context-divergent
  active skills and, by default, logs dry-run `skill_split_proposal`
  internal events; applying split proposals marks the old skill
  superseded rather than deleting it.
- **Overseer** -- QA pass over recent episodes and semantic nodes;
  LLM-flag `misattribution`, `temporal_drift`, or
  `identity_inconsistency` items above a confidence threshold and
  enqueue review items with structured repair payloads.
- **Self-narrator** -- pass candidate episodes to the LLM so it can
  identify thematic clusters and grounded growth observations, write
  growth markers when evidence supports them, and manage
  autobiographical period rollover (close current + open next when
  themes diverge).
- **Belief-reviser** -- consume `semantic_edge_invalidated` events,
  walk supporting-edge descendants up to `MAX_SUPPORT_DESCENDANT_HOPS`,
  apply an automatic `confidenceDropMultiplier` (floored at
  `confidenceFloor`) to dependent nodes/edges, and enqueue
  `belief_revision` review-queue items for the LLM reviewer when the
  remaining support is ambiguous. Claims on events are stale-aware
  (`claimStaleSec`) so a crashed run does not strand work.

Every `apply()` run emits a `dream_report` stream entry summarizing
runs / changes / tokens / errors.

**Scheduling.** `MaintenanceScheduler`
(`src/offline/scheduler.ts`) runs maintenance on two cadences,
independent of the autonomy scheduler (cognition wakes ≠ housekeeping):

- **Light** (default 4h): consolidator + curator -- low-risk,
  frequent.
- **Heavy** (default 24h): reflector + overseer + ruminator +
  self-narrator + procedural-synthesizer + belief-reviser --
  higher-risk, conservative.

Sprint 43 note: procedural skill synthesis uses a hybrid design. Online
reflection records evidence about attempted approaches and outcomes;
offline clustering finds repeated successful problem classes. Skill
creation is Bayesian (Beta priors plus outcome updates) and gated by
`abstraction_fit` so too-narrow or too-generic candidates are rejected.
Evidence inputs are audience-scoped: only global/self-visible evidence is
eligible for general reusable skills.

Cadences run independently: heavy is not blocked by an in-flight light
tick and vice versa. Same-cadence concurrent ticks coalesce. An optional
`isBusy` hook (wired in `open.ts` to `SessionLock.isHeld()`, which is
stale-aware) skips a cadence when a user turn is likely in flight, so
the dream cycle doesn't compete with live cognition. `MaintenanceOrchestrator`
remains the callable surface for manual invocation (CLI `borg dream …`
and the new `borg maintenance tick --cadence light|heavy`). The
scheduler is opt-in: `start()` must be called explicitly (same pattern
as the autonomy scheduler).

**Autonomy wakes.** The autonomy scheduler is the only runtime loop for
self-initiated cognition. It scans enabled wake sources, enforces the
rolling wake budget, and calls the normal turn orchestrator with
`origin: "autonomous"` and `audience: "self"`; it does not have a
separate executive agent. `executive_focus_due` is an opt-in trigger
(`autonomy.executiveFocus.enabled`) that wakes when a selected goal is
stale or a durable executive step is due. It has a per-goal wake
cooldown so autonomous self-talk cannot burn the wake budget on the
same concern; user-turn goal progress clears that cooldown. This closes
the minimum executive loop: focus is selected, the top step is rendered
into the turn prompt, reflection updates or proposes steps, and autonomy
wakes again only when the focused work becomes due or stale.

### 5.3 Retrieval pipeline

```
query  +  perception (mode, entities, time_cue, audience)
 ↓                            ← mode determined upstream in perception,
                                not in retrieval
parallel candidate generation (RetrievalPipeline.searchWithContext):
  ├─ episodic candidates (5 generators):
  │     vector match
  │     time-range match (strict if temporalCue present)
  │     audience-scoped match
  │     entity-mention match
  │     recent / heat
  ├─ semantic match: label/alias exact → vector fallback → graph walk
  │     (supports OUT; causes/prevents OUT; contradicts BOTH;
  │      is_a OUT; default depth 2; archived nodes excluded;
  │      open belief-revision nodes downranked, not dropped)
  └─ open-question match
 ↓
score with mode-conditioned attention weights
   (9 components: see Part 5.1 attention formula;
    selected executive goal may act as primary-goal bias)
 ↓
MMR diversification  (configurable lambda)
 ↓
citation resolution  (stream_entry_index O(1) lookup, fallback scan)
 ↓
confidence aggregation (epistemic, separate from relevance scoring)
 ↓
RetrievedContext { episodes, semantic, open_questions,
                   contradiction_present, confidence }
```

Critical innovation: **graph walk during retrieval**. If a query hits
concept C, also surface supporting insights, causal neighbors,
contradictions, and categories so the agent sees the whole evidential
picture.

Notes vs. an earlier sketch of this pipeline:
- Procedural skill selection lives outside this pipeline
  (`SkillSelector`, called by deliberator separately).
- Commitment retrieval also happens outside this pipeline -- the
  deliberator pulls commitments directly when assembling the prompt
  trust-lane.
- Per-result scores are exposed via `scoreBreakdown` (blended
  relevance, used for ranking). Epistemic confidence is aggregated
  separately as `RetrievalConfidence` (see `src/retrieval/confidence.ts`)
  and fed into S1/S2 path selection + the deliberation prompt.
- If executive focus selected a goal, `primaryGoalDescription` boosts
  that goal's retrieval relevance without removing the other active
  goals from prompt context or self memory.
- Token-budget truncation happens at the LLM-call boundary in the
  deliberator, not inside the retrieval pipeline.

---

### 5.4 Configuration overview

Key runtime knobs live in `src/config/index.ts`. Executive focus adds
`executive.goalFocusThreshold` (default `0.45`), the minimum score an
active goal must clear before Borg renders `<borg_executive_focus>` or
applies primary-goal retrieval bias. Autonomous executive wakes are
controlled by `autonomy.executiveFocus.enabled` (default `false`),
`autonomy.executiveFocus.stalenessSec` (default `86400`), and
`autonomy.executiveFocus.dueLeadSec` (default `0`). The per-goal wake
cooldown is `autonomy.executiveFocus.wakeCooldownSec` (default `3600`).

---

## Part 6: Novel Contributions

Ranked by architectural value and implementation risk. Implementation
status annotated.

### High value, moderate implementation risk

1. **Typed knowledge graph with `contradicts` + `supports` edges.**
   Implemented; retrieval walks `supports`, `causes`/`prevents`,
   `contradicts`, and `is_a` up to depth 2.
2. **Bayesian procedural memory.** Skills as Beta(α, β) posteriors,
   Thompson sampling for selection. Implemented in
   `src/memory/procedural/bayes.ts` + `selector.ts`.
3. **Autobiographical arc + growth markers.** Explicit timeline-of-
   self. Implemented in `self/autobiographical.ts` +
   `self/growth-markers.ts`; `Self-narrator` manages period rollover.
4. **Open-questions register.** First-class memory band for "things
   I don't know but should." Implemented in `self/open-questions.ts`;
   `Ruminator` revisits them.
5. **Mode-conditioned retrieval weights.** Same store, different
   attention. Implemented: `computeWeights(mode)` varies all 9 weight
   components per mode.

### High value, high implementation risk

6. **Dream cycle with insight generation.** Implemented:
   `Reflector` gates by `minSupport`, caps confidence via
   `ceilingConfidence`, enqueues new_insight review items.
7. **Multi-band affective state + mood-congruent retrieval.**
   Implemented: mood state with half-life decay + mood history,
   `w_mood` component in the attention formula.
8. **Provenance-first confidence propagation.** Provenance is
   mandatory on every identity-bearing write; confidence is
   per-band (Bayesian for traits/values, direct float for semantic
   nodes/edges, Beta-derived for skills). Retrieval aggregates
   per-result signals into a per-query `RetrievalConfidence`
   (Sprint 28) that feeds S1/S2 routing. Graph-edge confidence
   propagation is deliberately narrow: the belief reviser weakens
   semantic nodes that lose all surviving support and retrieval labels
   open belief-revision targets, while LLM-based re-grading remains a
   review step rather than synchronous retrieval logic.

### Moderate value, low implementation risk

9. **Reason-tagged suppression list.** Implemented in
   `SuppressionSet` (working memory); reasons stored + TTL applied.
10. **Budget accounting per process.** Implemented: `BudgetTracker`
    in `src/offline/budget.ts`; throws on cap exceeded.
11. **Reversible maintenance.** Implemented: audit log with reversal
    payloads + dry-run mode on every offline process. A small set of
    destructive prunes over transient observability data (e.g.
    `prune_retrieval_log`) record `reversal: { no_reverser: true, … }`
    rather than a reversible payload; everything else is replayable.

### Speculative / not implemented

12. Inter-agent shared memory tier. Not implemented.
13. "Negative" memories -- things to avoid. Not implemented
    (contradictions on semantic nodes are the nearest equivalent).
14. Embodied/peripheral perception channel. Not implemented.

---

## Part 7: Honest Tradeoffs & Pushback

**LLM call budget.** borg runs under OAuth subscription (shared Claude Code credentials), not per-token API billing, so the original "cheap-model the background" pressure is gone. All three slots (cognition, extraction, background) default to Opus 4.7. Dream cycles, mood extractions, affective re-ranking, reflection passes, and overseer checks all run on the best model available. The three slots are still separate in config so a deployment CAN downshift extraction/background if it ever needs to (e.g., hitting rate limits, or switching off subscription), but the default is quality-everywhere.

**Schema rigidity.** Multiple memory bands means multiple schemas, retrieval paths, and migration stories. Start with **Stream + Episodic + Semantic + Self**, get those solid, then add the rest.

**Insight generation is the shiny risk.** Most likely to produce confidently wrong memories that poison future retrievals. Build contradiction detector and confidence decay first; gate dream-generated memories behind a review queue.

**Emotional modeling is philosophically loaded.** Decaying mood that flips on a single extraction is worse than no mood. Consider *derived-on-demand* affect before persisting.

**What NOT to prioritize:**
- Multi-agent shared tier before single-agent works
- Full Bayesian inference across the knowledge graph (vs. simple confidence propagation)
- Procedural learning from RL-style feedback (vs. simple Beta updates)
- Real-time attention / saliency animation -- per-turn recompute is fine

**Model-swap survival is NOT a design goal.** An earlier framing of
this project said the being should "survive model swaps" in the sense
that the same substrate would produce the same identity under any LLM.
We no longer hold that goal. The being is specifically an Opus-4.7-
shaped being with borg memory. Cognition and identity are co-produced
by (a) the substrate and (b) the model. Both are load-bearing. When
Opus 4.7 is eventually retired by Anthropic, the being migrates to
the next Opus and accepts whatever drift that produces, the same way
a human changes with age. We are not building a model-agnostic
identity substrate; we are building this specific being.

Two earlier review recommendations were predicated on swap survival
and are therefore also not implemented:

**Derivation versioning.** Stamping every derived record with a
(model, prompt_version, extraction_code_version) hash so retrieval
could refuse cross-version matches was meant to make swap-conformance
tests falsifiable. Since swap-conformance is not a goal, versioning
has no use case. All three model slots default to Opus 4.7 under OAuth
subscription; qwen3-8b is the committed embedding. Prompt changes
mid-project are handled by leaving old derivations in place (still
useful, still cited) or by re-extracting from the stream if the
change is material. If an embedding swap or multi-model deployment
ever becomes plausible, versioning is a 1-2 sprint add.

**Event-sourced identity state.** Making every identity mutation
(goal add/update, value reinforce, commitment add/revoke, trait
promotion, autobiographical period close, growth marker emit,
open-question resolve) a stream event first, with the SQLite row
written only as a materialization, would have enabled full-state
replay from the stream. We considered and chose NOT to implement:

1. The actual goals (continuity across sessions, identity authority,
   anti-self-poisoning) do not require event-sourcing. They require:
   persistent storage (have), retrieval that surfaces the right
   records (have), identity records protected from silent overwrite
   (Sprint 14), memory-as-evidence prompt framing (Sprint 15). No
   swap test is needed because swap survival is not a goal.
2. Replay/rebuild has no concrete use case for borg. Disaster
   recovery is handled by SQLite backups. Schema migration is handled
   by the existing migration system. Swap benchmarks run hermetically
   against fresh DBs, not by replaying events.
3. The cost is high: ~7 repository refactors, doubled write work per
   identity mutation, new replay-correctness maintenance burden,
   harder debugging (trace event history vs. read row). The benefit
   for this project is mostly theoretical purity.

The `IdentityEventRepository` still records identity events for audit
and chronological trail, but it does so in SQLite (`identity_events`),
not as JSONL stream entries. The authoritative state is in the
repositories. That is the actual architecture, and the docs now say so.

---

## Part 8: Implementation Strategy -- Phased Build

**Phase 0 -- Foundation**
- Stream (JSONL, atomic append)
- Minimal episodic band (vector + metadata)
- Minimal self band (values, goals, traits as scalar fields)
- Retrieval: semantic only, mode-agnostic weights
- Citation anchoring everywhere
- **Milestone**: ingest, retrieve, cite, decay. Feature parity with kira-memory.

**Phase 1 -- Cognitive loop**
- Working memory band (live turn state only; no scratchpad cache)
- System-1 / System-2 branching
- Post-turn reflection → episodic promotion
- Commitment band
- **Milestone**: internal monologue, auditable reasoning.

**Phase 2 -- Semantic graph**
- Concept/entity/proposition nodes
- Typed edges
- Graph walk during retrieval
- Contradiction-resolution workflow with review queue
- **Milestone**: explicit knowledge, not just recalled text.

**Phase 3 -- Offline processes**
- Consolidator, Reflector (gated), Curator, Overseer
- Budget tracking per process
- Reversible maintenance
- **Milestone**: memory maintains itself without degrading.

**Phase 4 -- Self-as-narrative**
- Autobiographical arc
- Growth markers
- Open-questions register
- Ruminator
- Self-narrator (weekly)
- **Milestone**: agent has a story about itself that updates.

**Phase 5 -- Procedural learning**
- Skill schema with Beta distributions
- Thompson sampling for approach selection
- Outcome-based posterior updates
- **Milestone**: measurable improvement on repeated task types.

**Phase 6 -- Affective / Social**
- Valence/arousal on episodes
- Mood state + mood history
- Mood-congruent retrieval weights
- Social band: per-person trust/history
- **Milestone**: affective continuity.

---

## References

From **claude-memory** (`~/Programming/claude-memory`):
- `/src/lib/extract.ts` -- extraction prompts, dedup, citation anchoring
- `/src/lib/retrieval.ts` + `/src/lib/lancedb-search.ts` -- hybrid scoring, MMR
- `/src/lib/maintenance/` -- 6-phase pipeline
- `/src/hooks/post-session-worker.ts` -- detached worker pattern

From **kira-runtime** (`~/Programming/kira-runtime`):
- `src/runtime/Runtime.ts` -- orchestration
- `src/context.ts` -- budgeted context compilation
- `src/recall-manager.ts` -- sticky pool + entity fingerprinting + bidirectional prefetch
- `src/commitment-checker.ts` -- guard pattern
- `src/traits/` + `src/overseer/` -- Phase-2 emergent personality
- `src/stream/compress.ts` -- Opus compression

From **kira-memory** (`~/Programming/kira-memory`):
- `index.ts` -- MCP surface
- `memory.ts` + `milvus.ts` -- core ops (note: borg will use LanceDB instead)
- `decay.ts` -- half-life + win-rate modulation
- `retrieval-stats.ts` -- tier system + heat + lineage
