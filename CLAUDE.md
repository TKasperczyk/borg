## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

## Architecture freeze

Frozen taxonomy as of Sprint 7. ai_phenomenology was removed in Sprint 7; expressive self-claims now route through the manifest's self_report kind with persistence_class: assistant_self_report. New epistemic failure classes route through the ManifestFinalizer architecture, not by re-adding ai_phenomenology or expanding the relational claim taxonomy.

## Scope: harness vs model

Borg is a cognitive-memory harness for an LLM, not a wrapper that tries to make the LLM smarter. Before treating any observed failure as a sprint candidate, apply the **Opus 5.0 test**: would this same failure still manifest if Anthropic shipped a model 10× more intelligent than the current one?

- **If yes** (the failure stems from how Borg manages/presents information, audience scoping, memory integrity, retrieval surfacing, continuity, commitments, identity, discourse hygiene at the behavioral level, or a bad system prompt): in scope. Fix it in the harness or the prompt.
- **If no** (the failure is the model exercising poor judgment despite having everything it needs -- forced metaphors, structural-rhyme overreach, over-elaboration under conversational momentum, weak reasoning within a single response, taste-level prose decisions): out of scope. The fix is a stronger model, not more harness machinery.

Concrete examples:
- v36 user-facing Tom-leaks → **in scope** (Borg's audience-scoping wasn't surfacing the right grounding to the model)
- v43 misattribution of audience-name in semantic graph → **in scope** (overseer needed audience metadata as a recognized evidence source)
- v43 turn-84 "fence-membrane" structural rhyme overreach → **out of scope** (Borg gave the model the right context; the model chose to force a parallel between two concepts that share only a word)

Rationale: we previously deleted ManifestValidator (8d.8/8d.9) for exactly this reason -- it tried to enforce claim-coverage from outside the model when the model handles claim-coverage better natively when prompted well. Adding external auditors that second-guess the model's within-response taste just adds latency, complexity, and another LLM call with the same blind spots as the first.

## Solutions to refuse on sight

Before proposing any of these, stop. If the Opus 5.0 test verdict is "out of scope", refuse to propose, even when the user asks for options. Naming a pattern and refusing it is the right move; offering the pattern with caveats is not. The shapes below are seductive because they look like rigorous engineering -- testable, deterministic, predictable -- and they regress past lessons every time.

- **LLM-grades-LLM validators.** A second LLM call that audits the first for claim correctness, attribution, citation grounding, or semantic quality. Same blind spots as the generator. Precedent: ManifestValidator deletion (Sprint 8d.8-9).
- **Deterministic semantic re-validators.** Regex/pattern checks that re-verify claim grounding, citation accuracy, named-entity presence, numeric attribution, or other semantic correctness in model output. These are always symptoms of presentation gaps -- fix what the model sees in the prompt/ledger/recency, do not audit what it says afterward.
- **Post-generation rewriters with semantic judgment.** Code that edits model output to "improve" it based on inferred intent. The model owns its prose; we own its input.

In-scope deterministic checks (substrate hygiene only):
- Internal ID leak detection (Sprint 9.13): catches `strm_`, `ep_`, `oq_`, `semn_`, `sess_`, `cmt_` and similar in emitted text. These should never appear regardless of model intelligence.
- Tool-shape structural invariants: schema validation of emission tool fields, content non-emptiness, mutex constraints between fields, reference validity (e.g., reply_target.entity_id must be a known entity), encoding sanity.

The line: **substrate vs semantic.** Enforce substrate (artifacts that should never appear, structural invariants of our own tool shapes). Never enforce meaning (claim grounding, attribution, citation truth, naming correctness, numeric accuracy). Meaning is the model's job given the right context.

## Proposal hygiene

Any proposal for new harness work touching `src/cognition/`, `src/retrieval/`, `src/memory/`, or `src/offline/` must include a single-sentence Opus 5.0 verdict line in the proposal itself:

- "Opus 5.0: in-scope because \<reason\>" -- the harness genuinely doesn't surface what the model needs, or has a structural correctness gap.
- "Opus 5.0: out-of-scope -- the fix is \<X\>" -- where X is usually "stronger model", "better prompt copy", or "fix presentation in the ledger/recency/retrieval".

The verdict line is mandatory, not optional. Run the test before proposing, not after pushback. If you find yourself drafting an external auditor / validator / grader / claim checker / deterministic claim grounding, you almost certainly skipped this check -- the named patterns above exist precisely because that solution shape feels like good engineering and isn't.

When the verdict is out-of-scope, the right output is one sentence naming the harness-side fix, not a list of options that includes the refused pattern with caveats. Listing the refused pattern at all is the failure mode.
