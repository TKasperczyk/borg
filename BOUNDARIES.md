# Architectural boundaries: cross-session visibility

**Read this before changing anything in `src/memory/episodic/`, `src/memory/creator-directives/`, `src/retrieval/semantic-retrieval.ts`, or any audience/visibility code.**

This file records a deliberate architectural decision, reached by a 5-design / 3-judge
design panel (unanimous) and confirmed against the code. It exists to stop a tempting,
catastrophic refactor. If you are about to "just make memory bridge across sessions",
stop and read.

## Two questions, two structures -- on purpose

The system answers two genuinely different questions about "who may see this", and they
are kept in two separate structures **by design, not by accident**:

1. **The episode audience firewall** -- *"who was in the room when this happened?"*
   Involuntary experiential memory, scoped at the episode level. Two states only
   (`src/memory/episodic/audience-filter.ts`, `isEpisodeAccessVisible`):
   - `shared` (`audience_entity_id === null`) -> visible to everyone, or
   - `private-to-one` (`audience = X`) -> visible **only** when the viewer's audience is
     exactly `X`.

   `deriveEpisodeAccess` (`src/memory/episodic/extractor.ts`) stamps this mechanically:
   0 audiences -> shared; exactly 1 -> private to that one; **more than 1 -> the episode
   is dropped, not stored.** Multi-audience content cannot exist as a single episode.

   Semantic-graph node **and edge** visibility derive *transitively* from this one
   predicate (`src/retrieval/semantic-retrieval.ts`): a node/edge is admissible only if a
   source/evidence episode is firewall-visible. The privacy guarantee is **inherited from
   one ~15-line chokepoint**, not re-implemented per structure. That is the whole point --
   it is auditable.

2. **The creator-directive engine** -- *"who did the operator deliberately authorize to be
   told this?"* Per-fact, explicitly scoped (`content_scope`, `activation_policy`,
   `mention_policy`), in `src/memory/creator-directives/`. This is the **single, deliberate
   operator -> participant bridge.** It is *not* memory; it is authorization.

## The invariants (DO)

- **Private by default.** A one-audience episode is visible only to that audience. Cross-
  session experiential memory does **not** leak across audiences.
- **Cross-session authorized information flows only through the directive channel**, and it
  is authorized **per fact**. An operator briefing turn mixes shareable and operator-private
  facts; the directive extractor fans one turn into N independently-scoped directive rows.
  Per-fact granularity lives there, *not* in episode storage. (This is why an episode-level
  "authorized audiences" set cannot work -- it would over-share the whole turn.)
- **The only sanctioned firewall bypasses are explicit and fail-closed:** `unrestricted`
  (admin/correction read paths only) and `self_continuity` (identity continuity). They are
  unified behind one resolver (`ViewerCapability` / `resolveViewerCapability` /
  `isEpisodeVisibleToCapability`, `src/memory/episodic/access.ts`). A viewer that is missing
  or under-specified resolves to the most restrictive audience scope, and an unrecognized
  capability throws -- **never** a silent see-all. Keep it that way.

## The catastrophic refactor to NOT do (DON'T)

- **Do NOT make the firewall bridge operator -> participant.** Do not add a third
  "authorized audiences" state to episodes; do not relax the exact-audience match; do not
  move the audience-isolation invariant out of the single episode predicate and into a
  per-node scope evaluator or a grant table consulted on every read.

  The design panel evaluated exactly these (episode authorized-set; fold-directives-into-
  scoped-nodes; a unified grant primitive) against keeping the firewall strict. **All three
  judges (security-, simplicity-, and migration-first) ranked them below the status quo and
  returned "firewall change not warranted."** Every firewall-weakening design traded the one
  immutable, auditable scalar invariant for a mutable / scattered / cache-consistency leak
  surface, and raised a permissive-bug's blast radius from a handful of admin call sites to
  *every* read. A from-first-principles clean-room redesign re-derived ~85-90% of the current
  architecture -- the three-mechanism shape is close to *forced* by the requirements, not
  accidental debt.

- **Do NOT route a cross-session fact by widening an episode's visibility.** Mint a directive.

## The known, accepted limitation

A briefing fact authorized for a participant is stored **twice**: once as an operator-private
episode (what was said in the operator room) and once as a directive row (the authorized
fact). On amendment there are two sources of truth. **This is a known, accepted trade**, not
a bug to "fix" by merging the two structures -- merging them is the catastrophic refactor
above. If the dual-write ever becomes a real maintenance problem, the sanctioned direction is
to derive the operator-facing cross-session views over the fact store *in place*, never to
give episodes a cross-audience state.
