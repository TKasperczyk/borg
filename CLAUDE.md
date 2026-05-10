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
