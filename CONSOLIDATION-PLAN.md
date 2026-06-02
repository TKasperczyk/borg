# Borg Consolidation Plan

Architecture-audit follow-up (2026-06-03). Source: the adversarial coherence
audit (38 agents: map -> prosecution/defense/hunts + blind first-principles ->
adversarial verify -> synthesis + completeness critic).

## Verdict (the part that matters)

**Coherent core, not a Frankenstein -- consolidate, do not rebuild.** A blind
first-principles agent that never saw the code re-derived ~85-90% of borg's
spine. The strongest "it's accreted scar tissue" prosecution claims were handed
to skeptics who read the actual code and **refuted** them. The authority model
is **confirmed** structural throughout (keys on roles/entity-IDs/enums, never
user wording). The four visibility/scoping systems are **confirmed genuinely
distinct**, not duplicates.

What's real:
- **Live-test failures (authority levels, cross-session awareness, directive
  visibility) are NOT missing capabilities.** The machinery exists and is
  verified in code (goals/commitments are audience-scoped; authority is a
  structural two-key; cross-session activity is projected). They are
  **surfacing / presentation gaps** inside an over-conservative, scattered
  disclosure layer. This is the in-scope, subtractive-fix category.
- **A bounded consolidation backlog** of genuine but mostly cosmetic
  duplication, plus a few speculative subsystems that the requirements don't
  strictly force.

What this plan is NOT: a rewrite, a re-scoping of the memory bands, or an
attack on the scoping layer (which the audit confirmed is load-bearing).

## Governing principle for every item here

**Re-trace before you cut.** The audit's own synthesis assembled its cut-list
partly by skimming names, and its completeness critic caught it red-handed --
it called the affective band "inert" (it is live, tested, and load-bearing) and
mis-signed a "cut" against an observability hook. So: the verdict is trustable
about the forest; **every specific item below must be confirmed by a real
call-graph trace before any code is touched.** No "behavior-preserving" claim
without proof. Each implementation goes through codex-workflow + full
`pnpm typecheck` + full `pnpm test` + a human diff review, and pure refactors
are NOT deployed (the live service only loads `src` at restart).

## Corrections to the audit (verified by hand)

- **Affective / mood band: KEEP.** It is live and load-bearing
  (`computeMoodBoost` feeds the retrieval weight vector; tested). The owner is
  also extending it -- future demo-UI mood animations driven by the live
  valence/arousal values. It is OFF the consolidation list entirely. (Its
  internal half-life decay math may share the common helper -- see T1 -- but the
  band's behavior and API are untouched.)
- **Closure-pressure guard does NOT rewrite output.** Verified by reading the
  full guard: every path emits either the original response unchanged or a
  whole-output suppression; `removed_spans` is a trace label only. The cardinal
  posture is cleaner than the audit's record claimed. (Doc drift fixed + dead
  `rewriteModel` knob removed -- already committed, 3ce960e.)
- **The two "high" procedural cardinal-rule flags are probably mis-scored** (the
  0.35 threshold gates the model's own emitted confidence like the blessed 0.85
  goal-promotion gate; `centered_proper_noun` is prompt copy telling the model
  to avoid overfitting, not a harness regex). Listed under "Decisions" for a
  definitive adjudication, not treated as confirmed violations.

---

## TIER 0 -- Presentation consolidation (highest leverage)

This is the single most valuable move and it resolves several live failures at
once. It is design-significant, so it is **speced here for greenlight, not
implemented unsupervised.** Full design proposal below.

### The problem

Authority/disclosure context is resolved correctly but rendered in **parallel
rails**: commitments (conduct), creator-directive disclosure (what may be told),
identity governance (self-write), and cross-session activity each resolve and
render in their own evidence-ledger section or a separate trusted briefing.
There is no single place that answers, for the current addressee:

> "Who are you to this person? What have you promised them? What may you
> disclose to them? What have you been doing elsewhere they might hear about?"

The model has to reassemble that picture itself every turn. The owner's
recurring live failures -- directive visibility, authority levels, cross-session
awareness -- all live at exactly this seam. (Today's operator-directive-in-self-
cognition bug was one instance: correct resolution, scattered/over-conservative
surfacing.)

### The design: a per-audience "Standing With <addressee>" briefing

A single consolidated block in the evidence ledger / shared-state compile,
assembled per turn for the current audience, collecting the **already-resolved**
outputs of the existing deciders into one coherent view:

1. **Who you are to them** -- relational slots + identity facts scoped to this
   audience.
2. **What conduct applies / what you've promised them** -- commitments scoped to
   this audience/family.
3. **What you may and may not disclose to them** -- the creator-directive
   render decisions already computed per-audience (content / boundary / omit).
4. **What you've been doing elsewhere they may hear about** -- cross-session
   activity projection, scoped to what is shareable with this audience.
5. **Active authority context** -- operator/creator presence and the governance
   in effect this turn.

### Hard constraints (what makes it safe and in-scope)

- It is a **render/presentation consolidation only.** It does NOT change the
  firewall, the disclosure evaluator, the two-key authority, or any resolution
  logic. Those remain the sole authoritative deciders; the briefing only
  *collects* their outputs. No widening: if the disclosure evaluator says
  "omit" for this audience, it stays omitted.
- **Keyed on the structural audience** (entity-id / self / group), never on
  surface wording. Multilingual-safe.
- **No schema change, no migration, no new resolution path.** It assembles from
  inputs already present at the `runRetrievalPhase` / evidence-ledger-finalizer
  seam (audienceEntityId, the resolved creator-directive briefing, applicable
  commitments, cross-session activity, relational slots).
- Opus 5.0 verdict: **in-scope.** The model cannot reliably reassemble scattered
  authority context every turn; presenting one coherent per-audience view is the
  harness's job, not the model's.

### Why it fixes the live failures

- *Directive visibility*: the disclosure decisions are surfaced in one expected
  place instead of a separate trusted section the model may not weight.
- *Cross-session awareness*: the activity projection is rendered where the model
  is already reading "about this person," not in a separate rail.
- *Authority levels*: the model gets an explicit "who am I to this addressee +
  what governance is active" view instead of inferring it.

### Risk + validation

Low structural risk (no new deciders, no data change), but it changes the
**prompt the model sees**, so it must be validated by simulator A/B before live
deploy -- confirm it improves grounding rather than adding noise. This is the
one Tier-0 item; everything else is mechanical or a judgment call.

**Status: SPECED -- needs greenlight + a design review pass, then a
codex-workflow implementation gated behind sim eval.**

---

## TIER 1 -- Safe mechanical cleanups (behavior-preserving)

| Item | Evidence | Status |
|---|---|---|
| Dead `rewriteModel` knob in closure-pressure guard | full chain traced: orchestrator -> postgen -> closure, never read | **DONE (3ce960e)** |
| ARCHITECTURE.md closure-guard doc drift ("delete spans") | guard never edits text; verified | **DONE (3ce960e)** |
| Half-life decay primitive x3 -> shared `util/math.halfLifeDecay` | curator/index.ts, episodic/decay.ts, affective/mood.ts -- identical `Math.pow(0.5, e/h)` | **IN PROGRESS (codex, behavior-preserving)** |
| Duplicate `cosineSimilarity` (mmr.ts local vs embedding-similarity.ts export) | impls differ (export throws on dim mismatch) -- unify ONLY if no mixed-dim caller | **IN PROGRESS (codex assessing; safety-gated)** |
| Vestigial `src/operator-advice/` (0 LOC) | scout flagged empty | **IN PROGRESS (codex, remove if unreferenced)** |

Note on cosineSimilarity: this is the canonical example of "looks like dedup,
is actually a behavior change." The exported version throws on dimension
mismatch; mmr's local copy tolerates it. Unifying is only safe if vectors are
provably uniform-dimension on the mmr path. Codex is gated to leave it alone if
it cannot prove that. Safety beats DRY.

---

## TIER 2 -- Medium refactors (need greenlight; ready-to-run codex tasks)

These are real and valuable but touch load-bearing or security-adjacent code,
so they are **not** run unsupervised. Each is scoped for a one-command codex
greenlight.

1. **Authority/origin scatter -> central resolver.** The two-key authority check
   (`senderBorgRole === "creator" && sessionAudienceRole === "operator"`) is
   composed at 6+ call sites; turn-origin gating is ~6 predicate helpers across
   ~38 sites. Centralizing into one resolver reduces the "feels Frankenstein"
   surface the audit named. **Risk: authority is security-adjacent -- the
   refactor must confirm every site uses the IDENTICAL predicate and leave any
   variant site alone (BOUNDARIES.md warns against flattening authority
   distinctions).** Behavior-preserving; full-suite + human-diff gated. Greenlight
   to run.

2. **Offline plan/apply boilerplate (x13 processes).** Every offline process
   hand-rolls the plan/preview/apply + zod + error + tool-def lifecycle. Extract
   a shared `OfflineProcess` base/helper. The PROCESSES stay (each solves a
   distinct consistency problem -- not cut-worthy); only the boilerplate
   collapses. Larger surface; do as its own sprint with careful review.

3. **Four `shared-state/reconcile-*.ts` files.** Near-identical skeletons
   (goals/commitments/actions/open-questions) differing in counter names + the
   canonicalize fn. **DEFERRED with a caveat, not recommended as-is:** they
   diverge enough (commitments has an extra `non_canonicalizable_commitment_type`
   branch + `nowMs`) that a generic factory risks the "wrong abstraction" -- less
   readable than four explicit files. Only do this if a clean shared-skeleton +
   per-channel-hook split actually reads better; otherwise leave it.

---

## TIER 3 -- Judgment calls (decisions needed, no code change yet)

These are the "does this speculative subsystem earn its weight" questions. Each
is individually defensible; collectively they are where "overengineered" has
teeth. None should be cut without an explicit decision.

1. **Procedural memory as a full Bayesian/Thompson-sampling subsystem** (~6-8K
   LOC across memory/procedural + cognition/procedural + offline/procedural-
   synthesizer). The synthesizer routes on only one of three context dimensions
   (`domain_tags`); `problem_kind` and `audience_scope` are extraction overhead;
   the epsilon-greedy explore path is reportedly never invoked. Question: keep
   the ML machinery, or fold "approaches that worked before in similar contexts"
   into retrieved facts the model ranks itself? **Decision needed.**

2. **Four independent aging/decay clocks** (episodic heat, shared-state salience
   tiers, warm-recall TTL, working-memory suppression TTL). Several exist for
   prompt-size control the context assembler arguably already handles. Question:
   can any be collapsed without losing real retrieval behavior? **Decision
   needed** (note: the decay *math* is being deduped in T1; this is about whether
   the *clocks themselves* are all necessary).

3. **Dual belief-revision** (online semantic-revision in shared-state, ~400 LOC,
   alongside the offline Belief Reviser -- both can supersede/contradict semantic
   nodes). The online path exists for same-turn latency but is fail-open and
   capped. Question: is the dual-maintenance + sync risk worth the latency win,
   or route all belief revision through the offline worker? **Decision needed.**

4. **Self-Narrator empty-output error** (self-narrator/index.ts ~437-443).
   **The agents disagreed**: one called it a cardinal-rule drift (deterministic
   code judging that the model "should" have produced a narrative -> cut); the
   critic called it a guarded observability hook ("had sufficient episodes,
   produced nothing" -- a real "silent failure" signal -> keep). It only fires
   when `sourceEpisodes.length >= minSupportEpisodes && errors.length === 0`.
   **Decision needed -- do NOT auto-cut.** My lean: keep as observability but
   reclassify it as a trace/warning event rather than an error, so it surfaces
   the signal without "legislating" that the model was wrong.

5. **Procedural cardinal-rule flags -- ADJUDICATED (both compliant, no change).**
   Two "high" flags from the map were investigated by call-graph read tonight and
   are **false positives**:
   - `MIN_PROCEDURAL_CONTEXT_CONFIDENCE` (0.35) gate, context-extractor.ts:151 --
     gates on the **model's own emitted `confidence` field** (the tool schema,
     line 25), for an **internal routing handle** (procedural context used in
     retrieval scoring, never user-facing), with an `onDegraded("low_confidence")`
     observability callback. Same accepted shape as the blessed 0.85 goal-
     promotion gate. **Compliant.** (Minor optional enhancement, not a fix:
     surface the confidence to the model instead of a hard 0.35 drop -- low
     priority.)
   - `centered_proper_noun` / `abstraction_fit`, procedural-synthesizer/index.ts
     -- these are a **schema enum the MODEL emits** (lines 51, 57) plus **prompt
     copy** instructing the model when to self-classify overfitting (263-265);
     the harness only routes on the model's own emitted `abstraction_fit` (line
     825). There is **no harness name/proper-noun matching anywhere.** This is the
     *sanctioned* anti-overfit pattern (ask the model to recognize overfit
     skills), the opposite of a violation. **Compliant.**

   Net: borg has **zero confirmed cardinal-rule violations** in the audited
   surface. The post-gen critical-domain guards (KEEP list) are the only
   production policing, and they are within the explicitly-allowed envelope.

---

## KEEP (confirmed load-bearing -- do not touch)

- **Affective / mood band** (live, tested, + future UI use).
- **The four visibility/scoping systems** (confirmed genuinely distinct).
- **The episode audience firewall** (~15-line auditable chokepoint; semantic
  visibility inherits transitively -- BOUNDARIES.md, confirmed).
- **The two-key operator/creator authority** (structural; the single dimension
  the live-test gaps cluster on).
- **Post-generation commitment + closure guards** -- live judges that can
  suppress output, BUT confirmed scoped to **critical domains only**
  (privacy/audience/safety/tool-hygiene); the non-critical path is shadow-mode.
  This is the explicitly-allowed critical-violation enforcement, not a cardinal
  violation. **Action: add a standing written justification** (see below) since
  it is the codebase's most expensive recurring temptation -- keep it honest, but
  keep it.

## Standing justification to record (post-gen guards)

Add to ARCHITECTURE.md / a BOUNDARIES note: the commitment/closure post-gen
guards are production policing, permitted ONLY because they are pinned to
`enforcement_class === "critical"` (the five critical domains) with everything
else in shadow mode. Any extension to non-critical/semantic judgment (claim
grounding, citation accuracy, attribution) is forbidden by the cardinal rules.
Recording this prevents silent scope-creep.

---

## The real cost signal (not incoherence -- watch it)

A single user turn fans out to ~12-15 LLM calls before one user-visible
sentence (partial-confirmed; 17 LLM call-sites in the cognition layer). That is
**latency and usage-window cost, not architectural incoherence.** Worth
profiling as live usage scales -- which extraction/audit calls can be merged,
made lazy, or moved offline -- but it is a performance backlog, separate from
this consolidation.

---

## Decisions I need from you

1. **Tier 0 (per-audience briefing): greenlight the design?** This is the
   high-leverage one. If yes, I'll do a design-review pass then implement via
   codex-workflow, gated behind a simulator A/B before any live deploy.
2. **Tier 2.1 (authority resolver centralization): run it?** Behavior-preserving
   but security-adjacent -- I want your explicit yes before refactoring auth.
3. **Tier 3 judgment calls (procedural ML, aging clocks, dual belief-revision,
   self-narrator):** these are yours. I can write a one-page decision brief on
   any of them with the concrete tradeoff if you want before deciding.
4. **Deploy cadence:** none of tonight's cleanups are deployed (pure refactors;
   live service unaffected until restart). Tell me when you want a restart to
   pick them up, or batch with the Tier-0 work.
