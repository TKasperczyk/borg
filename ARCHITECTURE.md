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
2. **Stream is the source of truth.** Append-only log first, everything else is derived.
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

### 1. Stream (raw log)

Append-only JSONL. Truth-of-record. Everything else is derived.

```typescript
{
  id, timestamp,
  kind: "user_msg" | "agent_msg" | "thought" | "tool_call" | "tool_result" | "perception" | "internal_event",
  content, token_estimate, tool_calls?, audience?, session_id, compressed: boolean
}
```

### 2. Episodic (what happened)

Narrative units -- higher-level than stream entries. Each one is a *story* an agent can tell itself.

```typescript
{
  id, kind: "episode",
  title,                  // "Tom and I debugged the pgvector issue"
  narrative,              // 2-5 sentence prose summary
  participants: string[], // entity IDs
  location: string?,      // project / context
  start_time, end_time,
  source_stream_ids: string[],   // citation anchors
  emotional_arc: { start, peak, end }?, // valence trajectory
  significance: number,   // 0-1; computed, not set
  tags: string[],
  embedding,
  lineage: { derived_from: string[], supersedes: string[] },
  confidence: number,     // how reliably remembered
}
```

### 3. Semantic (what I know)

Concepts, entities, propositions, and typed edges between them.

```typescript
// Nodes
{ id, kind: "concept" | "entity" | "proposition", label, description,
  embedding, aliases, confidence, source_episode_ids[] }

// Edges
{ id, from, to,
  relation: "is_a" | "part_of" | "causes" | "prevents" | "supports"
          | "contradicts" | "related_to" | "instance_of",
  confidence, evidence_episode_ids[], created_at, last_verified_at }
```

Edge types matter. `contradicts` powers contradiction detection; `causes`/`prevents` powers causal reasoning; `supports` powers evidence chains.

### 4. Procedural (how I do things)

Skills as Bayesian beliefs about what works.

```typescript
{
  id, kind: "skill",
  applies_when: string,      // "pgvector similarity scores look wrong"
  applies_when_embedding,
  approach: string,          // "check embedding model dimension matches index config"
  attempts: number,
  successes: number,
  success_rate: number,      // Beta-distributed; Thompson sampling for selection
  confidence_interval: [lo, hi],
  alternatives: string[],    // IDs of competing approaches for same problem
  last_used, last_successful,
  source_episode_ids: string[]
}
```

### 5. Affective (how I felt / feel)

```typescript
// Per-memory (on episodic)
emotional_arc: { valence: -1..1, arousal: 0..1, dominant_emotion: string? }

// Current mood state (decays)
{ valence, arousal, updated_at, half_life_hours, recent_triggers: string[] }

// Mood history (time series for introspection)
{ ts, valence, arousal, trigger_episode_id }[]
```

Mood influences retrieval weighting -- mood-congruent memories get a salience boost. It also decays.

### 6. Self / Identity

```typescript
// Stable
{ values: [{ label, description, priority, source_episode_ids, last_affirmed }] }
{ known_strengths: [...], known_weaknesses: [...], known_blind_spots: [...] }

// Evolving
{ traits: { label: { strength: 0..1, last_reinforced, last_decayed } } }
{ autobiographical_arc: [{ period, narrative, key_episode_ids }] }
{ growth_markers: [{ ts, what_changed, evidence_episode_ids }] }

// Dynamic
{ current_goals: [{ id, description, priority, parent_goal?, status,
    progress_notes, created_at, target_at? }] }
{ open_questions: [{ id, question, urgency, related_episode_ids }] }
{ active_commitments: [...]  }
```

The `autobiographical_arc` + `growth_markers` pair is what turns memory into *identity-that-evolves*.

### 7. Commitments

```typescript
{
  id, type: "promise" | "boundary",
  directive,
  priority,
  made_to_entity?,
  restricted_audience?,
  about_entity?,
  provenance,
  created_at,
  expires_at?,
  revoked_at?,
  superseded_by?
}
```

Commitments are scoped, first-class memory. They are retrieved into the prompt
before speaking rather than enforced only as a post-hoc filter.

### 8. Social

```typescript
{
  entity_id,
  trust, attachment,
  communication_style?,
  shared_history_summary?,
  last_interaction_at?,
  interaction_count,
  commitment_count,
  sentiment_history: [{ ts, valence }],
  notes?
}
```

Social memory tracks per-entity trust/history and stores user-turn sentiment on
interactions separately from the agent's own outgoing tone.

### 9. Working memory (ephemeral)

```typescript
{
  current_focus: string,        // what I'm attending to
  hot_entities: string[],       // currently salient
  pending_social_attribution: { entity_id, interaction_id, turn_completed_ts }?,
  suppressed: [{ id, reason, until_turn }],
  mood: { valence, arousal, dominant_emotion }?,
  last_selected_skill_id: string?,
  last_selected_skill_turn: number?,
  mode: "problem_solving" | "relational" | "reflective" | "idle" | null,
  pending_intents: [{ description, next_action }]
}
```

Derived live-turn state only. Scratchpad/recent-thoughts were removed; persistent
thinking lives in the stream as `thought` entries.

---

## Part 5: Core Processes

### 5.1 Cognitive loop

Per-turn: **Perception → Attention → Deliberation → Action → Reflection**.

**Perception** (fast, deterministic)
- Parse input, extract entities, detect affective signals, detect temporal cues
- Update `working.hot_entities`
- Detect "mode" (problem-solving, relational, reflective, idle)

**Attention** (fast)
- Context-aware relevance function:
  ```
  score(memory, query, state) =
      w_sem   * semantic_similarity(memory, query)
    + w_goal  * goal_relevance(memory, state.current_goals)
    + w_mood  * mood_congruence(memory, state.mood)
    + w_time  * temporal_relevance(memory, state.time_cue)
    + w_soc   * social_relevance(memory, state.audience)
    + w_heat  * usage_heat(memory)
    − w_supp  * suppression_penalty(memory, state.suppressed_ids)
  ```
- Weights tunable per mode.

**Deliberation**
- Cheap path (System 1): high retrieval confidence + low stakes → respond directly.
- Expensive path (System 2): low confidence OR high stakes OR contradiction detected → internal monologue persisted as `thought` stream entries, re-retrieved, possibly a second LLM call.

**Action**
- Generate response + tool calls
- Commitment guard -- revise if violation detected
- Emit structured `intent` records alongside user-visible output

**Reflection** (post-action, before next input)
- Promote worth-remembering material from the completed turn / stream to episodic
- Update procedural confidence (Beta update if an approach was tried)
- Update mood state
- Flag anything needing offline attention

### 5.2 Offline processes

Run on idle, on sleep command, or on cron. Each has a **budget** and produces an **audit log**.

- **Consolidator** -- find redundant episodes, merge with lineage.
- **Reflector** -- pick N recent episodes, generate 1-3 new semantic propositions + edges. Require ≥ K supporting episodes and attach confidence to guard against hallucinated patterns.
- **Curator** -- apply decay, promote/demote tiers, archive cold memories, prune affective state.
- **Ruminator** -- revisit `open_questions` and unresolved items; generate next-step suggestions; update priorities.
- **Overseer** -- QA pass: flag misattributions, temporal drift, identity-inconsistent memories.
- **Self-narrator** -- periodically generate autobiographical summaries and `growth_marker` entries.

Every offline run emits a `dream_report` entry to stream.

### 5.3 Retrieval pipeline

```
query
 ↓
intent classifier  →  (mode: problem-solve | relational | temporal | reflective)
 ↓
query expansion    →  (entities, related concepts, likely temporal scope)
 ↓
parallel retrieve
  ├─ episodic (vector + temporal filter)
  ├─ semantic (vector + graph walk up to depth 2)
  ├─ procedural (match applies_when embedding)
  └─ commitments (audience filter)
 ↓
re-rank with mode-specific weights
 ↓
MMR diversification
 ↓
evidence chain trace  →  follow `supports` / `derived_from` edges
 ↓
confidence aggregation  →  per-result + overall answer confidence
 ↓
budget fit  →  truncate by token cap, preserve citation anchors
```

Critical innovation: **graph walk during retrieval**. If a query hits concept C, also surface `supports(C)` and `contradicts(C)` so the agent sees the whole evidential picture.

---

## Part 6: Novel Contributions

Ranked by value / cost.

### High value, moderate cost

1. **Typed knowledge graph with `contradicts` + `supports` edges.**
2. **Bayesian procedural memory.** Skills as Beta(α, β) distributions, Thompson sampling for selection.
3. **Autobiographical arc + growth markers.** Explicit timeline-of-self.
4. **Open-questions register.** First-class memory band for "things I don't know but should."
5. **Mode-conditioned retrieval weights.** Same store, different attention.

### High value, high cost

6. **Dream cycle with insight generation.** Guard: require N supporting episodes, attach low initial confidence, decay unused insights fast.
7. **Multi-band affective state + mood-congruent retrieval.**
8. **Provenance-first confidence propagation.**

### Moderate value, low cost

9. **Reason-tagged suppression list.**
10. **Budget accounting per process.**
11. **Reversible maintenance.**

### Speculative

12. Inter-agent shared memory tier.
13. "Negative" memories -- things to avoid.
14. Embodied/peripheral perception channel.

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

**Derivation versioning (explicitly not implemented).** A prior review
suggested stamping every derived record (episodes, semantic nodes,
commitment rewrites, open-question resolutions, etc.) with a hash of
(model, prompt_template_version, extraction_code_version) so retrieval
could refuse cross-version matches and swap-conformance tests could
distinguish records produced by different pipelines. We considered
this and chose NOT to implement it:

1. All three model slots default to Opus 4.7 under OAuth subscription.
   There is no embedding swap planned (qwen3-8b is the committed
   embedding). There is no extraction/background model swap planned.
   The dimensions the versioning would protect against have collapsed
   to "prompt edits", which git history already tracks.
2. Prompt changes mid-project are handled either by leaving old
   derivations as-is (they're still useful, just produced under older
   rules) or by re-extracting from the stream if the change is
   material enough to care about.
3. Swap-conformance testing runs hermetically on fresh databases per
   model, so it doesn't need version stamps to distinguish records
   from different runs.

If a real embedding swap or a multi-model-extraction deployment ever
becomes plausible, versioning is a 1-2 sprint add later. Today it is
defensive engineering for a scenario we are not planning, and the
cost is a touch-every-LLM-derived-table migration.

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
