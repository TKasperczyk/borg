# Memory and disclosure boundaries: memory is global to the being, disclosure is contextual

**Read this before changing anything in `src/memory/episodic/`, `src/memory/creator-directives/`, `src/retrieval/semantic-retrieval.ts`, or any audience/visibility code.**

This file records the project's **memory/disclosure doctrine**. The authoritative statement is
the Cardinal Memory Rule in `CLAUDE.md`: **recall is global to the being;
disclosure is contextual to the audience.** The same rule governs every cognition band -- live-turn retrieval, offline
mind-maintenance, autonomy and action-state memory, cross-scope synthesis, internal model tools,
semantic review status, and commitment-reconciliation awareness: each recalls globally and
attaches disclosure labels rather than pre-filtering by audience. Two `heuristics:guard` passes
keep this from being reintroduced under a new name -- one bans disclosure-search symbols from
cognition paths (recall gates), the other requires a disclosure label on every model-facing
memory row (disclosure coverage). A human mind does not forget what it knows because the wrong
person walked into the room -- it remembers, and chooses what to say. The
entity works the same way.

> **Slogan:** "Memory is global to the being. Disclosure is contextual to the audience."

What this file now protects is the architecture target: broad recall for cognition, plus a
post-recall disclosure-judgment layer. It used to defend the opposite -- an audience firewall
that gated what the entity could recall. That firewall-as-cognition design is
gone for both live-turn retrieval and second-order cognition. Audience machinery that
remains in the codebase is disclosure/export/admin plumbing, ranking metadata, or action
permission, not a predicate on what the entity may internally remember. If you are about to add or
widen an audience/session gate on *recall*, stop and read -- the `heuristics:guard` will also
fail the build if a disclosure-search symbol is called from a cognition path.

A note on the prior decision: this file once recorded a "deliberate architectural decision"
reached by a 5-design / 3-judge design panel that ranked every firewall-weakening design below
the strict firewall and returned "firewall change not warranted." **That verdict is
SUPERSEDED** by the project vision. It optimized the wrong objective -- privacy-as-visibility
filtering, with blast-radius and migration cost as the scoring axes -- and treated
audience-gated recall as a non-negotiable requirement. That requirement is the bug. The panel
never weighed the cost of making the entity's cognition blind. The thing to stop is no longer
"bridging sessions"; it is reintroducing audience-gated **recall** into cognition. The
legitimate residue of the panel's reasoning survives, but applied to the *right* layer -- see
the legitimate-residue note under "What NOT to do" below.

## Canonical vocabulary

Four terms, used consistently across `CLAUDE.md`, `WORKFLOW.md`, this file, and `ARCHITECTURE.md`.
Do not reintroduce "audience-scoped recall" or "visibility" as a *cognition* concept; reserve
"visibility" for the public/export/UI path.

- **recall for cognition** -- building what the entity thinks with. Always global; never
  audience/session/role/speaker-gated. Cognition includes live turn retrieval, autonomous
  triggers, offline self-narration, rumination/open-question resolution, procedural synthesis,
  belief revision, semantic extraction/review, action-state memory, and internal model tools.
  (Target API names: `recallEpisodesForCognition`, semantic recall for cognition.)
- **render for disclosure** -- deciding what the entity is *told it may say*,
  shaping the prompt/output by per-fact disclosure labels and authorization. This is where audience/role/privacy/operator
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
   on the memory. It records where a memory came from. It is **always recallable by the being**:
   it never decides whether the entity may internally remember something. Origin is metadata the
   entity reasons *with*, not a wall between the entity and its own experience.

2. **DISCLOSURE / authorization** -- *"who did the operator authorize to be told this fact?"*
   This is a **per-fact policy** applied **AFTER recall**, at render/emission. It decides what
   the entity may *say* to the current audience, not what the entity may *know*.

The correct pipeline is fixed: **recall broadly -> label origin and disclosure -> let the entity
reason with the labeled memory -> let the entity decide what to say -> enforce only narrow
non-cognitive host boundaries (tool, transport, destructive-op, public-export, platform
safety).** Privacy is enforced at emission -- the entity recalls but does not disclose -- never by
amnesia.

The rest of this section walks the mechanisms that used to sit on the wrong axis and records
their completed state.

### 1a. Episode origin and disclosure filtering -- completed inversion

*"who was in the room when this happened?"* is an **origin label**, not a recall predicate.
`isEpisodeAccessVisible` (`src/memory/episodic/audience-filter.ts`) now belongs to explicit
disclosure/export/admin visibility reads only. It decides whether an episode is discloseable or
export-visible to a viewer iff

- its `audience_entity_id` is `NULL` (public), **or**
- its `shared` flag is `true` (explicitly broadcast), **or**
- its `audience_entity_id` exactly matches the viewer's audience (private-to-one).

For cognition, `audience_entity_id`, `origin_audience_entity_ids`, and `shared` are **ORIGIN /
disclosure LABELS**, not recall predicates. Cognition recall (`recallEpisodesForCognition`)
returns episodes regardless of current audience, each carrying its origin and disclosure label
("private-to-X -- usable internally, do not disclose to current audience unless authorized").
The exact-audience-match predicate is legitimate only in **public/export/disclosure-render
paths** (`searchEpisodesForDisclosure` / public-export search), because it is isolated from
cognition.

### 1b. Multi-audience episodes -- completed inversion

`deriveEpisodeAccess` (`src/memory/episodic/extractor.ts`) stores multi-audience content as
memory instead of dropping it. An episode records all origin audiences via
`origin_audience_entity_ids` while preserving `audience_entity_id` as the single-origin legacy
projection when there is exactly one origin. Multi-audience content is stored once, recallable by
the being, and disclosure-labeled by origin/disclosure metadata. If a future schema change is
needed, the reset-after-backup regime below still allows a reset + reseed after a verified backup.

### 1c. Semantic source disclosure labeling -- completed inversion

Semantic-graph node **and edge** recall no longer derives admissibility from episode visibility
for cognition. Semantic recall for cognition recalls nodes/edges globally and attaches
disclosure labels from their source/evidence episodes. The single-chokepoint,
source-episode-ID-preserving design remains, but it is now a **PROVENANCE and
DISCLOSURE-LABELING chokepoint, not a recall gate**. Disclosure/export searches may still filter
or redact source details by audience; cognition receives the labeled semantic memory.

### 2. The creator-directive engine -- KEEP: the exemplar of the right pattern

*"who did the operator deliberately authorize to be told this?"* Per-fact, explicitly scoped
(`content_scope`, `activation_policy`, `mention_policy`), in `src/memory/creator-directives/`.
This is the **single, deliberate operator -> participant DISCLOSURE/authorization channel.**
It is *not* memory; it is authorization.

This is **the example of the RIGHT pattern**, and the inversion keeps it essentially unchanged.
It is legitimate **because it is disclosure/authorization policy** -- per-fact, explicit,
fail-closed, applied **after** recall -- governing what the entity may *say*, never gating what
the entity may internally *recall*. It is not one half of a two-structure firewall design; it is
the model the rest of the disclosure layer should follow. The entity always recalls the underlying
facts; directives decide which of them may be disclosed, to whom.

## The invariants (DO)

- **Private by default for DISCLOSURE.** A one-audience memory is **recallable by the being in any
  conversation**, carries a `private-to-X` disclosure label, and is **not disclosed** to other
  audiences unless authorized. The invariant to protect is **non-disclosure at emission**, not
  non-recall at cognition. Cross-session experiential memory must not *leak to the wrong
  audience at emission*; the entity recalling its own experience is exactly what must be allowed.

- **Disclosure is authorized per fact, through the directive channel.** The entity's **cognition**
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

- **Do NOT add `crossAudience: true` (or any one-off bypass) as a cognition fix, and do NOT widen
  `ViewerCapability` into cognition recall.** `crossAudience` is explicit disclosure/admin
  all-audiences plumbing. `ViewerCapability` is disclosure/export/admin visibility plumbing.
  Cognition recall is already global, so a cognition bypass is a category error.

  **The DO instead:** use the recall-for-cognition APIs, attach disclosure labels to recalled
  memories, and enforce privacy at emission. Note that an "authorized audiences" set is desirable
  as a **per-fact DISCLOSURE label** in the disclosure layer, never as a recall gate on episodes.

- **The superseded panel verdict.** The old design panel evaluated episode authorized-sets,
  fold-directives-into-scoped-nodes, and a unified grant primitive against keeping the firewall
  strict, and all three judges (security-, simplicity-, migration-first) ranked them below the
  status quo. **That verdict is superseded and is NOT binding.** It reached the wrong conclusion
  because it optimized the blast-radius and auditability of a *visibility filter* while treating
  audience-gated recall as a fixed requirement. Under the corrected requirements -- the entity
  must be able to recall its own experience, akin to a human mind -- the "three-mechanism shape is forced
  by the requirements" claim is false; the firewall-as-cognition is the part that has to go. Do
  not cite the panel to block the inversion.

  Keep the **legitimate residue**: the blast-radius / auditability / fail-closed concerns the
  panel raised are real, and they still apply -- to the **DISCLOSURE/emission layer** and to the
  genuine hard gates (tool, transport, destructive-op, public-export, platform safety). Keep the
  single-chokepoint discipline there; just point it at labeling and emission, not at recall.

## ViewerCapability -- disclosure/export/admin only

`ViewerCapability` / `resolveViewerCapability` / `isEpisodeVisibleToCapability`
(`src/memory/episodic/access.ts`) are retained only for explicit disclosure/export/admin
audience-filtered reads. The capability has two arms: `audience` (public/shared plus
exact-origin audience matches) and `unrestricted` (explicit admin/correction/export reads).
`self_continuity` and `operator_introspection` cognition-bypass arms are gone.

Cognition recall needs no capability at all -- recall is global. A missing or under-specified
viewer resolves to the most restrictive disclosure scope, and an unrecognized capability throws.
That fail-closed discipline belongs to disclosure/export/admin and genuine hard gates, not to
recall.

## Disclosure routing of a cross-session fact (DO)

- **For COGNITION, the entity already recalls cross-session facts -- no routing needed.** The
  recall is global; there is nothing to widen.
- **To AUTHORIZE disclosure of an operator-private fact to a participant, mint a per-fact
  directive.** That is the disclosure/authorization channel. Minting a directive is a
  *disclosure* mechanism (what the entity may *say*), not a way to "widen an episode's
  visibility" for recall.

## Future schema changes: back up, then reset + reseed (or forward-migrate)

The completed inversion used schema and data changes. As of 2026-06-04 a data reset is
**allowed**, gated only on a verified backup (see the LIVE SYSTEM regime in `CLAUDE.md` /
`WORKFLOW.md`):

- **Back up `demo/server/.borg-data/demo` first.** Then take the simplest path for future schema
  changes: edit the baseline and **reset + reseed**, or write a data-preserving backfill when that
  is simpler.
- A forward migration that preserves live rows is still fine if you want to keep the accumulated
  memory -- but you are no longer required to. "Edit the baseline + reset" is a legitimate path
  again, post-backup.
- The old Sprint-12 backfill burden is closed: multi-audience source memories are represented as
  stored origin labels, not repaired by recall filtering.

## The known, accepted note (disclosure-layer engineering)

A briefing fact authorized for a participant is stored **twice**: once as an operator-private
episode (provenance -- what was said in the operator room, always recallable by the being and
disclosure-labeled) and once as a directive row (the per-participant authorization). On
amendment there are two sources of truth.

This is a note about the **disclosure/authorization layer**, not a defense of any firewall. The
operator-private episode is **provenance**; the directive row is **authorization**. If the
dual-write ever becomes a real maintenance burden, the sanctioned direction is to derive
operator-facing views over the fact store **in place** -- a disclosure-layer concern -- never to
re-introduce a cross-audience *recall* gate on episodes.
