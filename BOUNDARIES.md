# Memory and disclosure boundaries: memory is global to Sol, disclosure is contextual

**Read this before changing anything in `src/memory/episodic/`, `src/memory/creator-directives/`, `src/retrieval/semantic-retrieval.ts`, or any audience/visibility code.**

This file records the project's **memory/disclosure doctrine** and the in-progress
**inversion** that brings the code into line with it. The doctrine is the Cardinal Memory
Rule in `CLAUDE.md` (see it there for the full, authoritative statement): **recall is global
to Sol; disclosure is contextual to the audience.** A human mind does not forget what it knows
because the wrong person walked into the room -- it remembers, and chooses what to say. Sol
works the same way.

> **Slogan:** "Memory is global to Sol. Disclosure is contextual to the audience."

What this file now protects is the **inversion**: broad recall for cognition, plus a
post-recall disclosure-judgment layer. It used to defend the opposite -- an audience firewall
that gated what Sol could recall -- and parts of the codebase still implement that firewall.
Where that is true, this doc names the concrete code symbol, states plainly it is **CURRENTLY
LIVE**, and tags it **DEPRECATED / being-inverted, not the endorsed design.** If you are about
to add or widen an audience/session gate on *recall*, stop and read.

A note on the prior decision: this file once recorded a "deliberate architectural decision"
reached by a 5-design / 3-judge design panel that ranked every firewall-weakening design below
the strict firewall and returned "firewall change not warranted." **That verdict is
SUPERSEDED** by the project vision. It optimized the wrong objective -- privacy-as-visibility
filtering, with blast-radius and migration cost as the scoring axes -- and treated
audience-gated recall as a non-negotiable requirement. That requirement is the bug. The panel
never weighed the cost of making Sol's cognition blind. The thing to stop is no longer
"bridging sessions"; it is reintroducing audience-gated **recall** into cognition. The
legitimate residue of the panel's reasoning survives, but applied to the *right* layer -- see
the legitimate-residue note under "What NOT to do" below.

## Canonical vocabulary

Four terms, used consistently across `CLAUDE.md`, `WORKFLOW.md`, this file, and `ARCHITECTURE.md`.
Do not reintroduce "audience-scoped recall" or "visibility" as a *cognition* concept; reserve
"visibility" for the public/export/UI path.

- **recall for cognition** -- building what Sol thinks with. Always global; never
  audience/session/role/speaker-gated. (Target API names: `recallEpisodesForCognition`, semantic
  recall for cognition.)
- **render for disclosure** -- deciding what Sol is *told it may say*, shaping the prompt/output by
  per-fact disclosure labels and authorization. This is where audience/role/privacy/operator
  status legitimately act.
- **search for public/export** -- public API, export, and non-operator-UI visibility filters.
  Audience filtering is legitimate here but isolated from cognition. (Target API name:
  `searchEpisodesForDisclosure`.)
- **action + tool + transport permission** -- the only legitimate hard gates: tool-shape
  invariants, transport/connector authorization, destructive ops, public exports, platform safety,
  and the operator+creator two-key authorization of cross-session actions. These gate *actions*,
  never recall.

## Two axes, never one -- origin vs disclosure

The system must keep two genuinely different things separate, and the error in the old design
was collapsing them into a single "who may see this" see/not-see axis. They are:

1. **ORIGIN / provenance** -- *"who was in the room when this happened?"* This is a **LABEL**
   on the memory. It records where a memory came from. It is **always recallable by Sol**: it
   never decides whether Sol may internally remember something. Origin is metadata Sol reasons
   *with*, not a wall between Sol and its own experience.

2. **DISCLOSURE / authorization** -- *"who did the operator authorize to be told this fact?"*
   This is a **per-fact policy** applied **AFTER recall**, at render/emission. It decides what
   Sol may *say* to the current audience, not what Sol may *know*.

The correct pipeline is fixed: **recall broadly -> label origin and disclosure -> let Sol
reason with the labeled memory -> let Sol decide what to say -> enforce only narrow
non-cognitive host boundaries (tool, transport, destructive-op, public-export, platform
safety).** Privacy is enforced at emission -- Sol recalls but does not disclose -- never by
amnesia.

The rest of this section walks the mechanisms that currently sit on the wrong axis.

### 1a. The episode audience firewall -- DEPRECATED / being-inverted, not the endorsed design

*"who was in the room when this happened?"* is an **origin label**, but the code currently
treats it as a recall predicate. One predicate (`src/memory/episodic/audience-filter.ts`,
`isEpisodeAccessVisible`) decides an episode is "visible" to a viewer iff

- its `audience_entity_id` is `NULL` (public), **or**
- its `shared` flag is `true` (explicitly broadcast), **or**
- its `audience_entity_id` exactly matches the viewer's audience (private-to-one).

`isEpisodeAccessVisible` **is CURRENTLY LIVE and still gates recall exactly this way.** This is
the canonical cognition recall filter: it makes Sol unaware of its own experiential memory
based on who is present. That is the precise disease the Cardinal Memory Rule forbids -- "NEVER
implement privacy by hiding memories from Sol's cognition."

**Target of the inversion:** `audience_entity_id` and `shared` become **ORIGIN / disclosure
LABELS**, not recall predicates. Cognition recall (`recallEpisodesForCognition`) returns
episodes **regardless of current audience**, each carrying its origin and a disclosure label
("private-to-X -- usable internally, do not disclose to current audience unless authorized").
The exact-audience-match predicate moves to the **public/export/disclosure-render paths only**
(`searchEpisodesForDisclosure` / public-export search), where audience filtering is legitimate
*because* it is isolated from cognition. The inversion lands via a **data-preserving forward
migration**, never a reset (see "Inversion must respect NEVER-RESET" below).

### 1b. Dropping multi-audience episodes at extraction -- a correctness defect being inverted

`deriveEpisodeAccess` (`src/memory/episodic/extractor.ts`) stamps newly-extracted episodes into
just two shapes: 0 source audiences -> `{audience: null, shared: true}` (public); exactly 1 ->
`{audience: X, shared: false}` (private to X); **more than 1 -> the episode is dropped, not
stored.** Multi-audience content cannot currently exist as a single episode. (`shared = true`
on a *non-null* audience is a secondary broadcast path -- honored by the predicate but never
produced by normal derivation.)

This is the worst form of the disease: the memory is **never persisted**, so Sol can never
recall it under any policy, no matter how the disclosure layer is fixed later. **The extractor
still drops these today.** This is a correctness defect, not correct behavior.

**Target of the inversion:** an episode records **ALL** its origin audiences -- an
`originAudienceEntityIds[]` array, or an `episode_audience_refs` join table with relations like
`origin` / `mentioned` / `private_to` / `public_to`. Multi-audience content is stored **once**,
recallable by Sol, and disclosure-labeled **per audience**. The forward migration must surface
and repair dropped multi-audience episodes where the source material still exists, **without a
reset** -- and where it cannot be reconstructed, record the loss honestly rather than pretend it
was captured.

### 1c. Transitive semantic-graph visibility pruning -- DEPRECATED / being-inverted

Semantic-graph node **and edge** admissibility currently derive *transitively* from the same
predicate (`src/retrieval/semantic-retrieval.ts`): a node/edge is admissible only if a
source/evidence episode is firewall-visible. The privacy guarantee is inherited from one
~15-line chokepoint rather than re-implemented per structure -- a real auditability win, but
auditability of the **wrong thing**: it makes Sol's *derived semantic knowledge* blind by
audience. **This pruning is CURRENTLY LIVE.**

**Target of the inversion:** keep the single-chokepoint, source-episode-ID-preserving design --
but as a **PROVENANCE and DISCLOSURE-LABELING chokepoint, not a recall gate.** Semantic recall
for cognition must **not** prune nodes/edges by source-episode audience. Instead it **attaches a
disclosure label** at the same chokepoint: "supported by private source episodes -- usable
internally, do not reveal source to current audience unless authorized." The auditable
single-chokepoint property survives in full; it just labels instead of hiding. The chokepoint
becomes more valuable, not less: one place to reason about provenance and disclosure for the
whole derived graph.

### 2. The creator-directive engine -- KEEP: the exemplar of the right pattern

*"who did the operator deliberately authorize to be told this?"* Per-fact, explicitly scoped
(`content_scope`, `activation_policy`, `mention_policy`), in `src/memory/creator-directives/`.
This is the **single, deliberate operator -> participant DISCLOSURE/authorization channel.**
It is *not* memory; it is authorization.

This is **the example of the RIGHT pattern**, and the inversion keeps it essentially unchanged.
It is legitimate **because it is disclosure/authorization policy** -- per-fact, explicit,
fail-closed, applied **after** recall -- governing what Sol may *say*, never gating what Sol may
internally *recall*. It is not one half of a two-structure firewall design; it is the model the
rest of the disclosure layer should follow. Sol always recalls the underlying facts; directives
decide which of them may be disclosed, to whom.

## The invariants (DO)

- **Private by default for DISCLOSURE.** A one-audience memory is **recallable by Sol in any
  conversation**, carries a `private-to-X` disclosure label, and is **not disclosed** to other
  audiences unless authorized. The invariant to protect is **non-disclosure at emission**, not
  non-recall at cognition. Cross-session experiential memory must not *leak to the wrong
  audience at emission*; Sol recalling its own experience is exactly what must be allowed.

- **Disclosure is authorized per fact, through the directive channel.** Sol's **cognition**
  needs no channel -- it recalls cross-session memory freely. What the directive channel
  governs is **disclosure/authorization** of operator-private facts to participants, per fact.
  An operator briefing turn mixes shareable and operator-private facts; the directive extractor
  fans one turn into N independently-scoped directive rows. **Per-fact granularity is the right
  shape for disclosure labels too** -- a coarse episode-level "authorized audiences" set would
  over-share the whole turn, which is exactly why disclosure must be labeled per fact, not per
  episode. (This argument survives the inversion and supports it.)

- **The disclosure/emission layer and genuine hard gates are fail-closed.** A missing or
  under-specified *disclosure* decision resolves to the most restrictive disclosure (recall
  proceeds; emission stays silent until authorized); an unrecognized *action/transport/tool*
  capability throws -- **never** a silent permit. Fail-closed strictness belongs to disclosure
  and to genuine hard gates, never to recall.

## What NOT to do (DON'T)

- **Do NOT reintroduce audience/session-gated RECALL into any cognition path.** Do not add a
  recall WHERE-clause keyed on the current audience; do not filter self/identity context,
  social/observed-events recall, cross-session activity, or proactive-outbound grounding by who
  is present; do not prune semantic nodes/edges by source-episode audience before cognition. If
  a recall path can be made unaware of a memory because of the current audience, that is the
  bug.

- **Do NOT add `crossAudience: true` (or any one-off bypass) as the fix, and do NOT widen
  `ViewerCapability` to paper over a gap.** A bypass concedes that recall is gated by default;
  the correct direction is to make recall global so no bypass is meaningful. Adding a wider
  capability is treating the symptom.

  **The DO instead:** invert recall to be **global**, **attach disclosure labels** to recalled
  memories, and **enforce privacy at emission.** Note that an "authorized audiences" set is now
  *desirable* -- but as a **per-fact DISCLOSURE label** in the disclosure layer, never as a
  recall gate on episodes.

- **The superseded panel verdict.** The old design panel evaluated episode authorized-sets,
  fold-directives-into-scoped-nodes, and a unified grant primitive against keeping the firewall
  strict, and all three judges (security-, simplicity-, migration-first) ranked them below the
  status quo. **That verdict is superseded and is NOT binding.** It reached the wrong conclusion
  because it optimized the blast-radius and auditability of a *visibility filter* while treating
  audience-gated recall as a fixed requirement. Under the corrected requirements -- Sol must be
  able to recall its own experience, akin to a human mind -- the "three-mechanism shape is forced
  by the requirements" claim is false; the firewall-as-cognition is the part that has to go. Do
  not cite the panel to block the inversion.

  Keep the **legitimate residue**: the blast-radius / auditability / fail-closed concerns the
  panel raised are real, and they still apply -- to the **DISCLOSURE/emission layer** and to the
  genuine hard gates (tool, transport, destructive-op, public-export, platform safety). Keep the
  single-chokepoint discipline there; just point it at labeling and emission, not at recall.

## ViewerCapability -- DEPRECATED as a cognition-recall access-control concept

The code currently routes recall through `ViewerCapability` / `resolveViewerCapability` /
`isEpisodeVisibleToCapability` (`src/memory/episodic/access.ts`), with two "sanctioned
bypasses": `unrestricted` (admin/correction read paths) and `self_continuity` (identity
continuity). A viewer that is missing or under-specified resolves to the most restrictive
audience scope, and an unrecognized capability throws. **All of this is CURRENTLY LIVE and still
gates recall.**

This is **DEPRECATED / being-inverted, not the endorsed design.** Treating Sol's recall of its
own memory as needing a *bypass* (`self_continuity`) or an admin *escalation* (`unrestricted`)
is exactly backwards: under the Cardinal Memory Rule, **cognition recall needs no capability at
all** -- recall is global. `self_continuity` and `unrestricted` are firewall artifacts; once
recall is global they collapse, because there is nothing to bypass. `self_continuity` in
particular is the "self-continuity scope as a recall bypass" pattern the review flags: Sol
should *always* recall its own autobiographical activity and self-state, with no special
capability.

**Target of the inversion:** `ViewerCapability`, if retained at all, belongs to the
**public/export/UI-render paths only**, where audience-scoped *visibility* is legitimate and
isolated from cognition. It must not gate what Sol recalls. The fail-closed, throw-on-unknown
discipline is good engineering -- carry it to the disclosure/emission and hard-gate layers, not
to recall.

## Disclosure routing of a cross-session fact (DO)

- **For COGNITION, Sol already recalls cross-session facts -- no routing needed.** The recall is
  global; there is nothing to widen.
- **To AUTHORIZE disclosure of an operator-private fact to a participant, mint a per-fact
  directive.** That is the disclosure/authorization channel. Minting a directive is a
  *disclosure* mechanism (what Sol may *say*), not a way to "widen an episode's visibility" for
  recall.

## Inversion must respect NEVER-RESET (forward migrations only)

The inversion is a **schema and data change on a LIVE system with real, non-wipeable memory.**
It must honor the NEVER-RESET / frozen-baselines / forward-migration rules in `CLAUDE.md` and
`WORKFLOW.md` -- those are correct and are **not** weakened to enable the inversion:

- Every step lands as a **new forward migration** that carries existing live rows across. No
  destructive drops of populated columns without a data-preserving path. No baseline edits.
- Recording all origin audiences (`originAudienceEntityIds[]` / `episode_audience_refs`) and
  repairing previously-dropped multi-audience episodes are **data-preserving forward
  migrations** that surface/repair where source material still exists -- **never a reset.** Where
  a dropped episode cannot be reconstructed, the migration records the gap honestly.

## The known, accepted note (disclosure-layer engineering)

A briefing fact authorized for a participant is stored **twice**: once as an operator-private
episode (provenance -- what was said in the operator room, always recallable by Sol and
disclosure-labeled) and once as a directive row (the per-participant authorization). On
amendment there are two sources of truth.

This is a note about the **disclosure/authorization layer**, not a defense of any firewall. The
operator-private episode is **provenance**; the directive row is **authorization**. If the
dual-write ever becomes a real maintenance burden, the sanctioned direction is to derive
operator-facing views over the fact store **in place** -- a disclosure-layer concern -- never to
re-introduce a cross-audience *recall* gate on episodes.
