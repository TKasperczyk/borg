## Workflow

The dev-loop process (sprint cycle, GPT Pro review submission, sim runs, ChatGPT mechanics, end-state goals) lives in `WORKFLOW.md` at the repo root. Read it after this file when starting a fresh session.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

## Architecture freeze

Frozen taxonomy as of Sprint 7. ai_phenomenology was removed in Sprint 7; expressive self-claims now route through the manifest's self_report kind with persistence_class: assistant_self_report. New epistemic failure classes route through the ManifestFinalizer architecture or post-hoc simulator audit categories, not by re-adding ai_phenomenology.

## Scope: harness vs model

Borg is a cognitive-memory harness for an LLM, not a wrapper that tries to make the LLM smarter. Before treating any observed failure as a sprint candidate, apply the **Opus 5.0 test**: would this same failure still manifest if Anthropic shipped a model 10× more intelligent than the current one?

- **If yes** (the failure stems from how Borg manages/presents information, audience scoping, memory integrity, retrieval surfacing, continuity, commitments, identity, discourse hygiene at the behavioral level, or a bad system prompt): in scope. Fix it in the harness or the prompt.
- **If no** (the failure is the model exercising poor judgment despite having everything it needs -- forced metaphors, structural-rhyme overreach, over-elaboration under conversational momentum, weak reasoning within a single response, taste-level prose decisions): out of scope. The fix is a stronger model, not more harness machinery.

Concrete examples:
- v36 user-facing Tom-leaks → **in scope** (Borg's audience-scoping wasn't surfacing the right grounding to the model)
- v43 misattribution of audience-name in semantic graph → **in scope** (overseer needed audience metadata as a recognized evidence source)
- v43 turn-84 "fence-membrane" structural rhyme overreach → **out of scope** (Borg gave the model the right context; the model chose to force a parallel between two concepts that share only a word)

Rationale: we previously deleted ManifestValidator (8d.8/8d.9) for exactly this reason -- it tried to enforce claim-coverage from outside the model when the model handles claim-coverage better natively when prompted well. Adding external auditors that second-guess the model's within-response taste just adds latency, complexity, and another LLM call with the same blind spots as the first.

## Production policing of model output

The pattern to avoid is **production policing**: an LLM (or deterministic check) that rewrites or suppresses Borg's user-facing output based on semantic judgment. Second-guessing the model with another agent that has the same blind spots adds latency, fires on the wrong things, and tends to mask harness gaps that should be fixed upstream. **Reserve production policing for critical violations only**: substrate hygiene (internal IDs leaking from prompt context), safety, prompt-injection attacks, tool-shape invariants. Not for "is this claim well-grounded", "is this attribution accurate", or "is this number cited correctly" -- those are the model's responsibility given correct context.

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
