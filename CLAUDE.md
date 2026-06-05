# CARDINAL RULE -- read before writing a single line of code

Borg is a cognitive-memory harness for an **LLM-based entity**, not a Chinese Room. The model handles language and judgment; the harness manages information and validates *structure*. Almost every recurring mistake in this project is a violation of that split. These are absolutes, not guidelines:

1. **No deterministic validation, scoring, gating, or rewriting of model output -- ever, anywhere in production, except the sanctioned critical host-boundary guards named below.** No regexes, keyword lists, phrase matching, or pattern checks that inspect what the model *produced* and judge whether it's acceptable. The ONLY place deterministic assertions about output may live is unit/integration tests, plus the narrow critical-violation enforcement path for critical-commitment, closure-pressure, internal-ID, and safety post-generation guards. (Tool-shape/schema invariants and internal-ID-leak detection are structural hygiene, not output validation -- those are fine. Semantic judgment of output is not.)

2. **Everything must work in ANY language.** The bot is multilingual. No English -- or any-language -- keyword/pronoun/kinship/phrase lists, and no detecting intent by matching specific words in user content. Key on *structure* (entity ids, slots, schema fields, roles), never on surface wording. Default *prompts* may be written in English (the model reasons across languages); *detection and branching logic* may never assume a language.

3. **No non-general case handling. Overfitting is a bug, not a fix.** If a change only works for the specific scenario in front of you -- a named person ("Tom", "Alice"), a particular phrasing ("do you know me"), an enumerated value set -- it is wrong. Fix the general structural condition, or fix the prompt. Adding an exception clause, a special branch, or a lookup entry per case IS the Chinese Room. Do not.

4. **When the model fails *with the right context already in hand*, the fix is a stronger model or a better prompt -- never harness machinery that legislates judgment.** If the harness is gagging, over-constraining, or deciding something the model should decide, the fix is almost always *subtractive*: delete the brittle rule and present information the model can reason over. (See the Opus 5.0 test below.)

If a proposed fix adds a rule, branch, regex, or word-list to handle a case, stop -- you are almost certainly building the wrong thing. The instinct to reach for that is the single most expensive recurring error here.

# CARDINAL MEMORY RULE -- recall is global, disclosure is contextual

Borg is a cognitive-memory harness for an LLM-based entity ("Sol") meant to be akin to a human mind. A human mind does not forget what it knows because the wrong person walked into the room -- it remembers, and chooses what to say. Sol works the same way. This rule governs every memory, retrieval, and visibility decision in the codebase, and overrides any older "the harness decides what Sol gets to see" framing wherever the two conflict.

Expanded definition: **any process that helps Sol think, remember, learn, revise, narrate itself, prioritize, resolve questions, form skills, update beliefs, inspect tools, or plan autonomous action is COGNITION.** This includes live turn retrieval, autonomous triggers, offline self-narration, rumination/open-question resolution, procedural synthesis, belief revision, semantic extraction/review, action-state memory, and internal model tools.

Clean rule: **Cognition gets global evidence with disclosure labels.** Audience/session/person/current-room may rank, annotate, authorize action, or constrain disclosure -- it may NOT decide what Sol is allowed to internally know.

Negative rule: **If code says "visible to current audience" and the caller is cognition, offline mind-maintenance, autonomy, or internal model tooling, it is probably wrong.**

1. **Sol's internal memory and recall are NEVER audience-gated or session-gated.** No path that builds what Sol thinks with -- episodic retrieval, semantic-graph recall, self/identity context, social/observed-events recall, cross-session activity, proactive-outbound grounding -- may make Sol unaware of a memory because of the current audience, session, role, or speaker. Recall is global to Sol.

2. **Audience, session, role, privacy, trust, operator status, and source provenance are DISCLOSURE metadata and action-policy inputs -- never recall predicates.** A memory may be private-to-Alice for DISCLOSURE while still visible-to-Sol for COGNITION. "Who was in the room" is an origin/provenance LABEL; "who may be told" is a per-fact disclosure/authorization policy applied AFTER recall, at render/emission. Neither decides whether Sol may internally remember.

3. **The harness MAY rank, label, budget, cite, and structure memories. It MUST NOT make Sol unaware of one merely because the current audience may not hear it.** Current audience may be a ranking BOOST, never a hard gate. NEVER implement privacy by hiding memories from Sol's cognition.

4. **The pipeline is fixed:** (1) Recall broadly. (2) Label origin, privacy, trust, and disclosure constraints. (3) Let Sol reason with the labeled memory. (4) Let Sol decide what to disclose. (5) Enforce only narrow NON-cognitive host boundaries -- tool permissions, transport permissions, destructive actions, public exports, and hard platform-safety constraints. Privacy is enforced at emission (Sol recalls but does not disclose), not by amnesia.

5. **Honesty about current code.** The inversion is SUBSTANTIALLY complete across live-turn and second-order cognition as of v101.1, and the GPT-Pro v102 audit (`~/borg-v102-gpt-verdict.md`) confirmed the architecture is no longer fundamentally misaligned. The v102.1 pass then closed the audit's concrete gaps: live-turn reflection evidence rows, self-narrator period labels (now computed over all contributing markers, not a capped subset, and updates fold in the prior label so a period is never demoted), and commitment-reconciler prompt rows all now carry disclosure labels; cross-audience online open-question resolution no longer hits a write-side audience gate (it applies via the identity guard's online-reflector carve-out with an evidence-derived, fail-closed resolution label); and the Lance vector disclosure predicate no longer over-fetches unknown-origin rows. The v103.1 pass then closed the GPT v103 audit's remaining blocker -- disclosure-label coverage on the rest of the model-facing serializers (autonomy conditions, offline overseer + review-resolver, ruminator question row, corrective-preference + shared-state relational slots, scheduled-wakes/skills/open-questions internal tools, creator-directive rows) -- and added the Patch 8 label-coverage guard (`pnpm heuristics:guard` now also FAILS the build when a model-facing object literal or serializer-helper return emits a private-bearing key without a same-object disclosure label). The recall inversion and the disclosure-label contract are now both in place across live-turn and second-order cognition; the only outstanding item is the final GPT-Pro "complete" re-confirmation (verdicts: `~/borg-v102-gpt-verdict.md`, `~/borg-v103-gpt-verdict.md`). Live-turn retrieval recalls globally across episodic, semantic, self/identity, goals/open-questions, social/observed-events, commitments, corrections, image-perception, autobiographical, and proactive-outbound paths, with disclosure labels. Second-order cognition obeys the same global-evidence-with-labels rule: offline mind-maintenance (self-narrator, ruminator, procedural-synthesizer, belief-reviser, shared-state semantic-revision), autonomy executive-focus and action-state memory, cross-scope synthesis (consolidator, reflector, semantic extractor), internal model tools (identity-events, commitments-list, semantic-walk), semantic review/caution status, and commitment-reconciliation awareness all recall globally and render disclosure labels rather than pre-filtering by audience. `isEpisodeAccessVisible` and `ViewerCapability` / `resolveViewerCapability` / `isEpisodeVisibleToCapability` serve disclosure/export/admin reads only; `ViewerCapability` has only `audience` and `unrestricted`, with `unrestricted` reserved for explicit admin/correction/export disclosure paths. The retrieval option types are split (`CognitionRetrievalOptions` cannot carry `audienceEntityId`/`crossAudience`), and a structural guard (`pnpm heuristics:guard`, run in CI and by `scripts/heuristics-guard.ts`) FAILS the build if a disclosure-search symbol (the firewall set or any `*ForDisclosure` callee, alias-resolved) is called from a cognition/offline/autonomy/outbound/internal-tool path outside an explicitly Disclosure/Export/Admin/Public/CurrentAudienceStanding/ActionAuthorization-named scope. Privacy is enforced after recall by disclosure labeling plus the model's judgment, never by hiding memories from cognition; combined labels fail closed to `unknown` and never demote a private/unknown source to public. Do not add new audience/session recall gates, do not widen disclosure/admin machinery into cognition recall, and do not reintroduce bypass concepts as cognition fixes. Under the reset-after-backup regime below you may still reset and reseed for schema/data changes after a verified backup.

Slogan: **"Memory is global to Sol. Disclosure is contextual to the audience."**

# LIVE SYSTEM -- real memory exists, but RESET IS ALLOWED (back up first) (as of 2026-06-04)

Borg holds **real memory**: "Sol" runs live in the BotArena arena on the demo data dir (`demo/server/.borg-data/demo`), forming continuous memory from real conversations. That memory is valuable but **not sacred** -- a data reset is permitted, provided you take a verified backup first. The strict "NEVER RESET / forward-migrations-only" regime (2026-05-31) is **lifted**: resetting to simplify a schema or data change is fine once the current state is safely backed up.

- **Back up before any reset.** Snapshot `demo/server/.borg-data/demo` first (recipe in WORKFLOW.md) and verify it. A reset wipes the data dir; the backup is the only way back. Never reset without a verified backup.
- **Reset is allowed.** `/api/admin/reset` (confirm token `RESET`), or stop the demo server and wipe `.borg-data/demo`, then let it reopen clean. Restarting the *process* still preserves the DB; a *data* reset now needs only a prior backup.
- **Schema changes: your choice.** A forward migration that carries live rows is fine, but you may also edit a baseline and reset + reseed after backing up -- whichever is simpler. You are no longer required to thread every change through a data-preserving migration.
- If unsure: back up, then do the simplest thing. See WORKFLOW.md for the recipe and full rule.

## Workflow

The dev-loop process (sprint cycle, GPT Pro review submission, sim runs, ChatGPT mechanics, end-state goals) lives in `WORKFLOW.md` at the repo root. Read it after this file when starting a fresh session.

## Architecture freeze

Frozen taxonomy as of Sprint 7. ai_phenomenology was removed in Sprint 7; expressive self-claims now route through the manifest's self_report kind with persistence_class: assistant_self_report. New epistemic failure classes route through the ManifestFinalizer architecture or post-hoc simulator audit categories, not by re-adding ai_phenomenology.

## Scope: harness vs model

Borg is a cognitive-memory harness for an LLM, not a wrapper that tries to make the LLM smarter. Before treating any observed failure as a sprint candidate, apply the **Opus 5.0 test**: would this same failure still manifest if Anthropic shipped a model 10× more intelligent than the current one?

- **If yes** (the failure stems from how Borg manages/presents information, disclosure labeling and audience-aware presentation (NOT recall gating), memory integrity, retrieval surfacing, continuity, commitments, identity, discourse hygiene at the behavioral level, or a bad system prompt): in scope. Fix it in the harness or the prompt.
- **If no** (the failure is the model exercising poor judgment despite having everything it needs -- forced metaphors, structural-rhyme overreach, over-elaboration under conversational momentum, weak reasoning within a single response, taste-level prose decisions): out of scope. The fix is a stronger model, not more harness machinery.

Concrete examples:
- v36 user-facing Tom-leaks → **in scope** (Borg wasn't surfacing the memory's disclosure labels, so the model couldn't tell what was safe to say -- a labeling/presentation gap, not a recall-gating gap)
- v43 misattribution of audience-name in semantic graph → **in scope** (overseer needed audience metadata as a recognized provenance/evidence label, not as a recall gate)
- v43 turn-84 "fence-membrane" structural rhyme overreach → **out of scope** (Borg gave the model the right context; the model chose to force a parallel between two concepts that share only a word)

Rationale: we previously deleted ManifestValidator (8d.8/8d.9) for exactly this reason -- it tried to enforce claim-coverage from outside the model when the model handles claim-coverage better natively when prompted well. Adding external auditors that second-guess the model's within-response taste just adds latency, complexity, and another LLM call with the same blind spots as the first.

## Production policing of model output

The pattern to avoid is **production policing**: an LLM (or deterministic check) that rewrites or suppresses Borg's user-facing output based on semantic judgment. Second-guessing the model with another agent that has the same blind spots adds latency, fires on the wrong things, and tends to mask harness gaps that should be fixed upstream. **Reserve production policing for critical violations only**: substrate hygiene (internal IDs leaking from prompt context), safety, prompt-injection attacks, tool-shape invariants. Not for "is this claim well-grounded", "is this attribution accurate", or "is this number cited correctly" -- those are the model's responsibility given correct context.

Critical-commitment, closure-pressure, internal-ID, and safety post-generation guards are this sanctioned host-boundary exception; this is distinct from memory/disclosure architecture and does not authorize recall gating.

The distinguishing question on any proposed check: **is it observing/structuring/auditing, or is it deciding what reaches the user?**

Allowed and load-bearing in Borg:
- **LLM reading LLM output for extraction, classification, interpretation.** The semantic extractor, action-state extractor, goal-promotion extractor, corrective-preference extractor, reflector, perception, recency compiler -- all read prior LLM output to produce structured data. Not policing.
- **Post-hoc / eval-time audit.** The simulator overseer runs LLM audits at checkpoints over bounded sims; findings drive harness improvements rather than in-flight enforcement. The offline overseer audits production state into the review queue -- gray zone (production-resident, but doesn't enforce in-flight), case-by-case.
- **Critical-violation enforcement.** Internal-ID leak detection (Sprint 9.13). Schema/structural invariants of emission tools (field non-emptiness, mutex constraints, reference validity, encoding sanity). Narrow scope, clear failure mode, never about semantic judgment.

Pushback expected if proposed (not refused outright, but argue for it explicitly):
- **In-flight LLM judges of semantic correctness in production.** Examples: manifest-style claim validators or other claim-grounding judges that rewrite/suppress user-facing output. Precedent: ManifestValidator deletion (Sprint 8d.8-9) chose to drop this shape because the audit had the same blind spots as the generator and prompt-side fixes worked better. When proposing one, name what's critical about the violation. If it isn't critical, the right answer is usually upstream (better extraction, better presentation, better prompt copy).
- **Deterministic semantic re-validators in production.** Regex/pattern checks re-verifying claim grounding, citation accuracy, named-entity presence, numeric attribution in emitted text. Same problem one layer down -- if the model needed something the harness didn't give it, fix the harness, don't audit the output.

## Proposal hygiene

Any proposal for new harness work touching `src/cognition/`, `src/retrieval/`, `src/memory/`, or `src/offline/` must include a single-sentence Opus 5.0 verdict line in the proposal itself:

- "Opus 5.0: in-scope because \<reason\>" -- the harness genuinely doesn't surface what the model needs, or has a structural correctness gap.
- "Opus 5.0: out-of-scope -- the fix is \<X\>" -- where X is usually "stronger model", "better prompt copy", or "fix presentation in the ledger/recency/retrieval".

The verdict line is mandatory, not optional. Run the test before proposing, not after pushback.

When the verdict is out-of-scope, the harness-side fix is usually the answer (better prompt, ledger render, retrieval, extraction). If you find yourself drafting a production policeman of model output -- LLM or deterministic -- pause and name what's critical about the violation. For non-critical semantic judgments (claim grounding, citation accuracy, attribution correctness), production policing is the failure mode regardless of how rigorous it looks. Audit and extraction shapes are different and fine; the test is whether the new component decides what reaches the user.
