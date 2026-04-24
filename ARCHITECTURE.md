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
2. **Stream is an append-only audit log.** Turns, perceptions, identity events, and extraction sources are recorded chronologically. It is NOT the canonical source of all state -- authoritative state lives in the typed repositories (SQLite rows + LanceDB vectors). The stream exists so extraction has replayable input and so identity-relevant mutations leave a chronological trail. Event-sourced identity rebuild is explicitly not a goal (see Part 7).
3. **Cold paths do real work.** Dreams/consolidation are not cosmetic -- they must produce summaries, insights, new edges, new skills.
4. **Retrieval must be context-aware, not just semantic.** Current goal, current mood, current commitments, current audience all shift what's relevant.
5. **Self is a first-class entity, not a prompt file.** Identity, values, skills, current goals, open questions, uncertainties live in the memory itself and evolve with it.
6. **Forgetting is a feature.** Decay + win-rate modulation + affect-weighted decay. Unbounded growth kills performance and coherence.
7. **Honest uncertainty beats false confidence.** Every claim carries confidence + source-type; every retrieval exposes its evidence chain.
8. **Maintenance is auditable and reversible.** Dry-run, review queue, rollback.
9. **Resource-bounded autonomy.** Every autonomous action has a cost accounting (LLM tokens, compute).
10. **Composable over monolithic.** Runtime, memory, reflection, reasoning, skills are separate services with clear contracts.

---

## Part 3: Proposed Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          AGENT CORE                                      │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    COGNITIVE LOOP                               │    │
│  │                                                                 │    │
│  │  Perception → Attention → Deliberation → Action → Reflection    │    │
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
│  │  │   heuristics,│  │   mood,      │  │   traits, narrative, │  │    │
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
state-machine columns) reflect the post-Sprint-24 schemas. Where the
authoritative type lives outside `types.ts` (autobiographical periods,
growth markers, open questions), the file is named explicitly.

### 1. Stream (raw log)

Append-only JSONL. Records turns, perceptions, identity events, tool
invocations, internal events, and offline run reports. Everything that
gets persisted into the typed repositories is derived from these
entries (or from manual API writes), but the repositories remain the
canonical source of state.

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
// Episode (LanceDB row + SQLite mirror)
{
  id, kind: "episode",
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
  domain: string?,                // canonicalized topic anchor (Sprint 16/24)
  audience_entity_id: EntityId?,  // private-to-X scoping (Sprint 1/2)
  shared: boolean?,               // shared-with-others marker
  created_at, updated_at,
  archived: boolean,              // soft-delete flag
  superseded_by: EpisodeId?,      // consolidator merge target
}

// EpisodeStats (SQLite-only, paired with episode by id)
{
  episode_id,
  retrieval_count, use_count, last_retrieved,
  win_rate,                       // outcome-weighted usefulness
  tier: 1 | 2 | 3 | 4,            // T1 short-term ... T4 core
  promoted_at, promoted_from,
  gist, gist_generated_at,        // optional summary blurb
  last_decayed_at,
  valence_mean,
  archived,
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
  created_at, last_verified_at }
```

Edge types matter. `contradicts` powers contradiction detection;
`causes`/`prevents` powers causal reasoning; `supports` powers
evidence chains. Domain canonicalization (`canonicalizeDomain`) maps
synonyms (`technology` → `tech`) so the homonym-anchor still merges
intended-same-concept extractions.

### 4. Procedural (how I do things)

Skills as Bayesian beliefs about what works. The persisted shape stores
the Beta posterior parameters directly; success rate and confidence
intervals are derived on demand by `bayes.ts`.

```typescript
// Skill (LanceDB row for applies_when embedding + SQLite stats)
{
  id, kind: "skill",
  applies_when: string,         // "pgvector similarity scores look wrong"
  approach: string,             // "check embedding model dim matches index config"
  alpha: number,                // Beta posterior α (>0)
  beta: number,                 // Beta posterior β (>0)
  attempts, successes, failures,// raw counts (alpha = α0+successes, etc.)
  alternatives: SkillId[],      // competing approaches for same problem
  last_used, last_successful,
  source_episode_ids: EpisodeId[],
  created_at, updated_at,
}

// Derived (computed by bayes.ts, not stored):
//   success_rate     = α / (α + β)
//   ci_95            = quantiles of Beta(α, β)
//   thompson_sample  = sample ~ Beta(α, β)   // for selection
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
`mood` weight in the attention formula (Part 5.1).

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
}

// Growth markers (src/memory/self/growth-markers.ts)
GrowthMarker {
  id, ts,
  what_changed, before_description, after_description,
  category, confidence,
  evidence_episode_ids, source_process,
  provenance,
}

// Open questions (src/memory/self/open-questions.ts)
OpenQuestion {
  id, question,
  status: "open" | "resolved" | "abandoned",
  urgency,
  related_episode_ids, related_semantic_node_ids,
  source: "user" | "reflection" | "autonomy" | "deliberator",
  resolution_note?, resolved_at?, resolved_by?,
  provenance?, created_at,
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
  trust_delta?, attachment_delta?,
  interaction_delta?, valence?,
}
```

Social sentiment is lagged: an interaction is recorded immediately
when a turn completes, but its `valence` is only attached on the NEXT
user turn from the user's affective signal (Sprint 15). The agent
cannot flatter itself into a warm relationship by speaking warmly.

### 9. Working memory (ephemeral)

```typescript
{
  current_focus: string,
  hot_entities: string[],       // capped at 32, normalized at save time
  pending_social_attribution: {
    entity_id, interaction_id,
    agent_response_summary: string?,
    turn_completed_ts,
  }?,
  pending_trait_attribution: {
    trait_label, source_episode_ids,
    turn_completed_ts, audience_entity_id,
  }?,
  suppressed: [{ id, reason, until_turn }],
  mood: { valence, arousal, dominant_emotion }?,
  last_selected_skill_id: string?,
  last_selected_skill_turn: number?,
  mode: "problem_solving" | "relational" | "reflective" | "idle" | null,
  pending_intents: [{ description, next_action }],   // capped at 16
  turn_counter,
  updated_at,
}
```

Derived live-turn state. Scratchpad/recent-thoughts were removed in
Phase E; persistent thinking lives in the stream as `thought` entries.
The `pending_*_attribution` fields implement Sprint 15/24 lagged
attribution -- a turn's effect on social trust or trait reinforcement
is determined by the user's reaction on the NEXT user turn, not by
the agent's own output. Working memory is persisted (per-session
file) but normalized + bounded on every save to stay live-turn-ish.

---

## Part 5: Core Processes

### 5.1 Cognitive loop

Per-turn: **Perception → Attention → Deliberation → Action → Reflection**.

**Perception** (fast, deterministic)
- Parse input, extract entities, detect affective signals, detect temporal cues
- Update `working.hot_entities`
- Detect "mode" (problem-solving, relational, reflective, idle)

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
- Cheap path (System 1): high retrieval confidence + low stakes →
  go straight to the response call (with the tool loop -- see Action).
- Expensive path (System 2): low confidence OR high stakes OR
  contradiction detected → run an `EmitTurnPlan` planner pass first
  (structured tool-use that returns verification_steps, tensions,
  voice_note, uncertainty), persist the plan as a `thought` stream
  entry, then make the response call with the plan rendered into a
  tagged `<borg_s2_plan>` block. The response call itself is one LLM
  invocation enriched with the plan, not a second retrieval.
- The prompt receives the confidence summary in a
  `<borg_retrieval_confidence>` block so the being can calibrate how
  certain it speaks (internal signal -- not a user-facing percentage).

**Action**
- Run the response call as an Anthropic tool-use loop
  (`executeToolLoop`): the model can read internal tools
  (`episodic.search`, `semantic.walk`, `commitments.list`,
  `identityEvents.list`) or write via `openQuestions.create` mid-turn,
  with `tool_call`/`tool_result` entries appended to the stream in
  order. Caps: 5 iterations, 3 tool calls per iteration.
- Append the agent's text response as `agent_msg`.
- Infer structured `intent` records from the response text for audit.
- A separate `CommitmentChecker` runs as a post-hoc judge: if it
  detects a violation, an LLM rewrite pass produces a corrected text.
  This is detection-then-rewrite, not in-flight blocking.

**Reflection** (post-action, before next input)
- Update procedural skill posteriors (Beta update via
  `recordOutcome` if an approach was tried).
- Stash `pending_social_attribution` and `pending_trait_attribution`
  in working memory so the NEXT user turn's affective signal becomes
  the evidence (Sprint 15/24).
- Optionally enqueue review-queue items (e.g., reflection-driven open
  questions, identity inconsistencies surfaced this turn).
- Note: episodic extraction does NOT run synchronously in reflection.
  After the response is delivered, `StreamIngestionCoordinator` runs
  asynchronously (fire-and-forget), reads new stream entries past its
  watermark, and produces episode candidates via the extractor.
  Mood updates also happen in the perception phase of the next turn,
  not in this reflection.

### 5.2 Offline processes

Six cooperative processes that share an orchestrator
(`MaintenanceOrchestrator`), a per-process budget, and an append-only
audit log. Each emits its plan through `plan()`/`preview()`/`apply()`
so all maintenance is dry-runnable and reversible.

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
- **Overseer** -- QA pass over recent episodes and semantic nodes;
  LLM-flag `misattribution`, `temporal_drift`, or
  `identity_inconsistency` items above a confidence threshold and
  enqueue review items with structured repair payloads.
- **Self-narrator** -- cluster episodes by tag, ask the LLM whether
  each cluster shows a growth observation, write growth markers when
  it does, manage autobiographical period rollover (close current +
  open next when themes diverge).

Every `apply()` run emits a `dream_report` stream entry summarizing
runs / changes / tokens / errors.

**Scheduling.** `MaintenanceScheduler`
(`src/offline/scheduler.ts`) runs maintenance on two cadences,
independent of the autonomy scheduler (cognition wakes ≠ housekeeping):

- **Light** (default 4h): consolidator + curator -- cheap,
  low-risk, frequent.
- **Heavy** (default 24h): reflector + overseer + ruminator +
  self-narrator -- expensive, higher-risk, conservative.

Cadences run independently: heavy is not blocked by an in-flight light
tick and vice versa. Same-cadence concurrent ticks coalesce. An optional
`isBusy` hook (wired in `open.ts` to `SessionLock.isHeld()`, which is
stale-aware) skips a cadence when a user turn is likely in flight, so
the dream cycle doesn't compete with live cognition. `MaintenanceOrchestrator`
remains the callable surface for manual invocation (CLI `borg dream …`
and the new `borg maintenance tick --cadence light|heavy`). The
scheduler is opt-in: `start()` must be called explicitly (same pattern
as the autonomy scheduler).

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
  │     (relations: supports, contradicts; default depth 2; archived
  │      nodes excluded)
  └─ open-question match
 ↓
score with mode-conditioned attention weights
   (9 components: see Part 5.1 attention formula)
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
concept C, also surface `supports(C)` and `contradicts(C)` so the
agent sees the whole evidential picture.

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
- Token-budget truncation happens at the LLM-call boundary in the
  deliberator, not inside the retrieval pipeline.

---

## Part 6: Novel Contributions

Ranked by value / cost. Implementation status annotated.

### High value, moderate cost

1. **Typed knowledge graph with `contradicts` + `supports` edges.**
   Implemented; retrieval walks both relations up to depth 2.
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

### High value, high cost

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
   propagation (weakening derived beliefs when supporting evidence
   is weak) is still not implemented -- each semantic node/edge
   carries its own confidence in isolation.

### Moderate value, low cost

9. **Reason-tagged suppression list.** Implemented in
   `SuppressionSet` (working memory); reasons stored + TTL applied.
10. **Budget accounting per process.** Implemented: `BudgetTracker`
    in `src/offline/budget.ts`; throws on cap exceeded.
11. **Reversible maintenance.** Implemented: audit log with reversal
    payloads + dry-run mode on every offline process.

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

The stream still logs identity events for audit and chronological
trail. The authoritative state is in the repositories. That is the
actual architecture, and the docs now say so.

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
