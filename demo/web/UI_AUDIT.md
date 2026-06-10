# UI Audit -- borg demo operator console (2026-06-10)

Source: 20-auditor multi-agent visual/UX audit (12 per-screen, 6 design-dimension, 2 code-level) over 25 live screenshots + code, plus a 5-area gap round (responsiveness, motion/CPU, cold-start, WhyDrawer, disclosure labels). 271 raw findings + 36 gap findings, clustered and deduped. Screenshots in /tmp/borg-ui-audit/ (ephemeral).

Severity: critical = broken/unusable/misleading; major = redesign must fix; minor = clear, lower stakes; polish = refinement.

## Themes (cross-screen patterns)

### [CRITICAL] ID/value text shattering in narrow panes
*Corroboration: 9 independent findings. Screens: review, governance, inspector, stream*

Eight of twenty auditors independently hit the same render: the shared .props KV grid fixes a 130px key column and applies overflow-wrap:anywhere (/home/luth/Programming/borg/demo/web/src/styles.css:3710-3728), then gets mounted inside ~190-280px candidate cards in Review's forced 3-up comparison grid (Review/index.tsx:1066-1067), so ids render as 11-line vertical strips ('se/mn/_2/d6/...') and aliases as 'Gatt aca (199 7)'. The adjudication surface -- the console's core human-decision screen -- is unreadable for exactly the data it exists to compare; Governance's 320px rail even word-breaks the 'inspect' button label itself. Fix: a container-aware KV primitive (stack key over value below a min width), shortId+copy chips for every id, and never overflow-wrap:anywhere on interactive labels.

Representative findings:
- Candidate-card field values shatter into one-to-two characters per line
- 320px fixed detail rail breaks its own content: 'inspect' buttons wrap mid-word, ids and family names fracture
- Key/value prop rows collapse to 1-2 characters per line in narrow panes

### [CRITICAL] Fabricated or static data styled as live instrumentation
*Corroboration: 12 independent findings. Screens: shell/statusbar, memory, mission, stream, admin, palette, dream*

The console repeatedly invents readings: a hardcoded branch/model/embedding cluster with a permanent green dot (StatusBar.tsx:46-90, repo is actually on frontend-redesign-p0-shell), sparklines synthesized server-side from a single scalar (app.ts:1386-1394), ⌘-chords documented on three surfaces with no handler, a 'TAILING' badge that cannot change state, a clock labeled UTC showing local time, and run history labeled SCHEDULED. On an operator console, instruments that cannot turn red train operators to distrust the real ones beside them. Rule for the redesign: anything not data-bound must not look like telemetry -- bind it or delete it.

Representative findings:
- Status bar renders fake hardcoded telemetry (branch, model, embedding) styled as live instruments
- Band-card 'growth' sparklines are fabricated -- same synthetic ramp for every band
- Route diagnostics advertises keyboard chords (⌘0--⌘9) that are not implemented anywhere
- 'Schedule lane' with a 'SCHEDULED' column is actually run HISTORY
- 'TAILING' live badge is unconditional
- Attention rail headline row is quadruple-redundant, with a fake sparkline

### [CRITICAL] Dead ends: the data exists but the UI cannot reach it
*Corroboration: 9 independent findings. Screens: stream, cognition, palette/inspector, dream, mission*

The redesign's core promise ('every id is clickable, drill into anything') is broken at the seams: useStreamWindow.loadOlder is fully implemented but no UI element calls it, turn traces are browser-memory-only so replay is dead for any historical turn, shortId-truncated ids displayed everywhere cannot be pasted back into the palette, the RAW tab never fetches, and all 18+ orrery click targets collapse to 4 undifferentiated destinations. Each is individually small; together they make the console a viewer of the last few minutes only. Fix the round-trips: wire loadOlder, persist per-turn traces server-side, make palette/inspector resolve prefixes and verified candidates.

Representative findings:
- Stream advertises 'older entries available' twice but offers no control to load them
- Turn replay -- the screen's core promise -- does not work for any turn that predates page load
- Known bug confirmed: truncated display ids can never round-trip through the palette
- Transcript is capped at 16 entries with no scroll-back
- 'RECENT ERRORS 3' is a dead end

### [CRITICAL] Hero visualizations are decorative and illegible (Orrery + FlowChart)
*Corroboration: 9 independent findings. Screens: mission, cognition*

Both flagship canvases fail the same test: the encoding channels carry no data (ring radius = array index, governance arcs have fixed sweeps, log stroke-width yields a ~2px delta between counts 2 and 70) while the actual information lives in 8.5px faint text that collides with its own marks ('review 6' struck through 'ruminator'). On Cognition the resting pipeline sits below 3:1 contrast while an unconditional particle starfield glows in full accent green -- decoration brighter than data, against the plan's own animation rule. Either make the visuals earn their ink (proportional arcs, per-band navigation, collision-managed labels, decoration gated on live activity) or demote them to status glyphs beside a scannable list; a plain 8-row table currently beats the orrery at its stated job.

Representative findings:
- Orrery labels and marks collide systematically -- no collision management at all
- The radial form is mostly decorative -- nearly all information is carried by scattered text
- Flow chart is effectively invisible at rest: queue-state nodes/edges sit at ~1.3:1 contrast while decorative particles glow in full accent green
- Cognition's centerpiece flow chart is pure decoration: role=img with zero handlers

### [CRITICAL] Color semantics collapsed: one palette serving five taxonomies
*Corroboration: 12 independent findings. Screens: mission, stream, identity, governance, dream, review, memory*

Severity, lifecycle, disclosure, system-identity, and action-risk all draw from the same five hues with no contract: rank is computed from raw row count (19 routine pending extractions = top-tier purple, above red, styles.css:50-53 + useAttentionData.ts:116-127), a permanent 100-degree red arc fires because one critical-CLASS commitment exists (Orrery.tsx:276-291), amber means ten unrelated things, and green simultaneously means brand, live, ok, primary action, selection, and 'deliberate silence'. The result is a cockpit that cries wolf on healthy state and renders genuine danger (reset, forget) in calmer colors than routine chips. Redesign needs a written color contract: severity encodes state not volume, ramp ends in red, purple locked to interior/dream, red reserved for failure and destructive actions, green rationed to live/active/primary.

Representative findings:
- Severity scale tops out in purple above red and is driven by raw count volume
- Red is spent on classes and benign values, not failures
- Amber is the most diluted token: every routine tool call plus 37-of-39 'candidate' traits render in the warning color
- Green does seven unrelated jobs
- Purple carries ~7 meanings

### [MAJOR] Raw machine IDs as headline identity
*Corroboration: 13 independent findings. Screens: cognition, stream, memory, governance, shell, inspector-palette, identity*

Human labels exist in the data (session label 'Z życia bota...', entity names Tom/Lunaria/FisherBot, sender_entity_id on chat entries) but UUIDs get top billing everywhere: speaker names, card titles, group headers, filter pills, breadcrumb, audience pill, and a 64px stream column that wraps the same 44-char id four lines tall on every row. This is the single most pervasive presentation failure -- roughly a dozen auditors hit it independently on different screens. Establish one hard rule with one primitive: human label first, middle-truncated mono id as a secondary copyable chip, full id in the Inspector; ids never headline, never wrap, never repeat per-row when constant for the session.

Representative findings:
- Message attribution shows a raw thread UUID as the speaker name
- The same 46-character audience UUID is rendered ~8 times simultaneously
- Raw internal IDs are the headline identity; human names are demoted to metadata
- AUDIENCE pill shows a raw 50-char internal thread id on real sessions
- Group headers are raw turn UUIDs that say nothing about the turn

### [MAJOR] Inverted space budgets and void layouts
*Corroboration: 12 independent findings. Screens: memory, admin, identity, review, cognition, mission, stream, prompts, dream, shell*

Nearly every screen allocates space inversely to information: Memory and Admin strand ~60% of 1080p as black void (verified against live data -- layout, not thinness), Review gives its densest pane the least width while the near-empty resolution pane idles, Mission letterboxes a square orrery while the rail truncates its own text at 360px, Cognition caps a 127-message conversation at 38%, and Dream puts 3300px of reference table above the operational cockpit. Root causes are mechanical: fixed-px grid tracks (300px stream detail, 320px rails, 360px MC rail), squares centered in wide canvases, and grid rows stretching to the tallest sibling. The redesign should adopt proportional minmax tracks, content-aware spans, capped inner scrolls, and the codebase's own stated idiom (full-height grid, scroll inside regions -- REDESIGN_PLAN.md:882).

Representative findings:
- Memory atlas landing leaves ~55-60% of the viewport as an empty void -- with thin AND live data
- Space allocation is inverted: the densest pane gets the least width, the emptiest pane the most
- Values board stretches to match the 39-card traits board, leaving a ~700px-tall empty black void
- Unbounded 80-row schedule table hijacks the page and exiles the real cockpit ~3 viewports below the fold
- Cognition workbench inverts priority: conversation capped at 38% while 62% goes to an idle instrument graph

### [MAJOR] Redundant rendering: one fact, three to four encodings
*Corroboration: 9 independent findings. Screens: shell (all screens), mission, prompts, stream, governance, memory*

Mood renders three times simultaneously on Mission Control (literally the same moodLabel string in two chrome bars plus a screen strip), review/question/commit counts appear in three vocabularies ('r 6' vs 'REVIEW 6' vs a bare rail badge), Prompts shows the identical paragraph four times, and Stream prints each payload as BODY then again as RAW. Duplication is not confidence -- it triples the surface that can disagree and eats the space the starved panes need. Decide one home per metric (rail badges for actionable counts, strip for session state, status bar for runtime facts) and one full-text surface per record (the detail rail), then delete the copies.

Representative findings:
- Same five signals duplicated 3-4 times across permanent chrome
- Editor pane shows the same prompt text four times in the default state
- Detail pane is a fixed 300px column that renders BODY and RAW as duplicates
- Selected row's full text is rendered two to three times on screen at once
- Attention rail headline row is quadruple-redundant

### [MAJOR] Chip soup: interactive and static elements share one identical idiom
*Corroboration: 9 independent findings. Screens: mission, governance, review, dream, identity, stream, memory, prompts, admin*

.tag and .btn are both 1px var(--line) bordered mono chips differing by ~2px (styles.css:1179-1190 vs 1295-1352), so 183 static Tag sites and 84 clickable IdRef sites are indistinguishable at rest, panel-header actions ('open', 'clear log') are the faintest text in their rows, and band cards that navigate look like inert stat tiles (their 'explore' hint CSS is dead code). Multiple auditors independently called this the single highest-leverage fix. Split the grammar: passive tags go borderless/tinted, interactive chips get a reserved affordance channel (underline/chevron/accent edge plus real focus styles), destructive verbs get a danger tint -- then collapse the ~13 chip implementations into Tag/CountBadge/IdRef.

Representative findings:
- Static status tags and clickable id/inspect chips share one indistinguishable bordered-box visual across every screen
- Evidence chips and the destructive 'forget' action are visually identical buttons
- Affordance hierarchy is inverted: the primary navigation action on every panel ('open') is the faintest element in its row
- Chip vocabulary fragmented: ~13 distinct chip/badge implementations

### [MAJOR] Destructive-action grammar inverted
*Corroboration: 7 independent findings. Screens: governance, review, admin, identity*

The visual danger ranking contradicts the consequence ranking everywhere: revoke is primary-green in Directives but ghost in Commitments, supersede (declared destructive in the same file, Review/index.tsx:104) is the brightest button in the resolution pane, substrate reset wears the same amber live-write class as routine identity edits, and memory-destructive 'forget' is identical to read-only 'why'. Worse, supersede ships with a silently pre-selected winner that defeats its own needsWinner guard, and the confirm modal never names winner/loser. Add a .btn.danger tier (none exists), assign variants by action class (primary = safe, live-write = mutating, danger = irreversible), and require explicit winner selection with a consequence-stating confirm.

Representative findings:
- Destructive 'revoke' is styled as the primary green CTA in directives, but ghost in commitments
- Destructive 'supersede' is the green primary button; safe actions are ghost
- The most destructive action in the product is a tiny amber chip, not a red danger button
- Winner silently pre-selects candidate 1 by array order, and the confirm modal never states the consequence

### [MAJOR] Dateless and raw time rendering
*Corroboration: 6 independent findings. Screens: stream, memory, review, dream, inspector*

One shared helper (lib/stream-utils.ts:43-54 formatTime emits HH:MM:SS only) makes multi-day data look corrupted across four screens: episode ranges read '13:16:54 - 08:54:43' (end before start), stream rows step backwards across invisible midnights, the review queue looks non-monotonic, and two dream runs 24h apart render identically; the Inspector's default tab meanwhile prints raw epoch milliseconds. Fix once at the formatter: date-aware ranges, day separators or relative age ('2d ago') with absolute on hover, and timestamp formatting in GenericSummary. Bare HH:MM:SS is acceptable only for same-day live tails.

Representative findings:
- Time-only timestamps over a multi-day window; group ranges read as negative durations
- Episode time ranges render time-of-day only, producing backwards-looking ranges
- Inspector Summary shows raw epoch-ms timestamps
- Schedule-lane timestamps are time-only, so multi-day history becomes indistinguishable

### [MAJOR] Type scale and contrast below the legibility floor
*Corroboration: 8 independent findings. Screens: all screens, rail, mission/cognition SVGs, identity, stream*

Functional text routinely fails its own system: --text-ghost at 2.34:1 carries session group headers and shortcut digits, the entire 9px eyebrow layer sits at 3.52:1 (83 usage sites), 114 of 267 font-size declarations are 10-11px while the documented 12.5px body appears ~10 times, and SVG labels render down to 6.7px effective. Because everything is small and faint, hierarchy flattens too -- nothing reads as primary. Treat this as a token-level constraint: raise --text-faint to ~4.5:1, enforce a 9px hard floor (rail needs 10px or tooltips), raise data rows to 12px, and ban literal px font-size so the scale is actually tunable.

Representative findings:
- --text-ghost (2.34:1) is used for real interface text -- effectively invisible
- Entire eyebrow/label layer is sub-AA: 9px uppercase --text-faint at 3.52:1
- Type center of gravity is 10-11px, not the documented 12.5px body
- Rail labels and badges run 8-8.5px -- below the app's own token floor
- SVG type scales below the legibility floor: labels render ~8px, tier metadata ~6.7px

### [MAJOR] Unbounded data rendered without aggregation or density controls
*Corroboration: 8 independent findings. Screens: stream, dream, identity, governance*

Wherever row count is unbounded, the UI renders 1:1 with no grouping, thresholding, or facet controls: the worst case is functional -- stream-grouping.ts:169-170 buckets ALL turn-less entries into one global mega-group sorted as a unit, so a conversation cannot be read in time order. The same shape recurs benignly: 80 schedule rows differing only by audit id (while the audit log on the same screen already groups by run_id), 39 trait cards with 37 one-shot noise candidates, unbounded family/audience enumerations as filter pill walls, and a 16-card single kanban lane. These get worse with real data, not better; the redesign needs grouping-by-run, strength thresholds with collapsed weak-candidate lists, and combobox facets for unbounded value sets.

Representative findings:
- All turn-less entries collapse into one 'unclaimed / maintenance' mega-group, destroying chronological order
- Eleven visually identical 'curator / 19:18:40 / audit' rows -- same run rendered once per audit row
- 37 noise-level candidate traits get full-weight cards with full action sets, and the board has no sort/filter/threshold controls
- Commitments filter bar is a 3-row wall of ~20 pills from five unbounded groups

### [MAJOR] Backend gaps and connector envelope leaking raw into the UI
*Corroboration: 10 independent findings. Screens: cognition, stream, shell, dream, governance, identity, memory, palette*

A cluster of findings is not fixable in the frontend: the BotArena connector stores its routing envelope as a text prefix (drowning every preview and transcript body -- 4 auditors), never sets a human audience_label (so the disclosure-critical AUDIENCE pill is a UUID), traces and prompt revisions have no persistence endpoints, trait labels aren't canonicalized ('intellectual_honesty' vs 'intellectual honesty' as the only two established traits), the affective band count is silently session-scoped while every other band is global, and scope-matrix totals are window aggregates the plan itself flags as silently-wrong-at-scale. File these as a backend-gap backlog (the audit's stated purpose) rather than patching with UI regexes, which the project rules correctly forbid.

Representative findings:
- Bridged connector envelope renders verbatim inside message bodies, burying the actual content
- In bridge sessions every user_msg preview is the identical connector envelope; message content never visible in the list
- Developer/debug copy shipped as operator-facing UI text
- Scope-matrix counts are client aggregations over a limit-50 window presented as totals
- The flagship ESTABLISHED traits column is two near-duplicate rows of the same trait

### [MAJOR] State-rendering fragmentation: errors look like empties, five loading idioms
*Corroboration: 7 independent findings. Screens: mission, memory, review, identity, governance, admin, inspector*

MissionControl renders transport failures through <Empty> so 'error: Bad Gateway' reads exactly like 'no creator-directive conflicts' -- on an ops console, faults must never speak in the voice of 'nothing to do'. Beneath it, the shared Loading/ErrorState/Empty components exist but are outnumbered ~3:1 by hand-rolled variants (17 ad-hoc loading divs, 15+ ad-hoc error divs, six empty-state classes, two different dash placeholders), and the Inspector's not-found state leaves six dead tabs clickable with no recovery action. Declare the three components the only legal state renders, give errors a distinct warn-tier treatment with retry, and add a CI guard (heuristics-guard precedent exists) so the drift doesn't recur.

Representative findings:
- Errors rendered with the empty-state idiom -- backend failure looks identical to healthy-empty
- Loading/ErrorState/Empty components exist but most call sites hand-roll the idiom
- Inspector not-found is styled as a calm empty state with six live tabs and no recovery path
- Loading and empty states are bare one-line notices in a full viewport

### [MAJOR] Keyboard and accessibility gaps
*Corroboration: 6 independent findings. Screens: governance, stream, memory, mission (orrery), palette, shell*

Roughly half the selectable surfaces are spans/divs/trs with bare onClick and no role/tabIndex/keys, while the correct pattern already exists in-repo (Review queue rows); the orrery removes outlines from five interactive classes and restores focus styles for only two, leaving keyboard-focusable satellites invisible; the palette -- the keyboard-first tool -- neither scrolls its selection into view nor advertises its own existence anywhere in the chrome. Mechanical to fix: standardize on the Review-row pattern, add the three missing :focus-visible rules, scrollIntoView on activeIndex change, and a visible ⌘K trigger in the topbar.

Representative findings:
- A whole class of interactions is mouse-only: filter pills, table rows, stream rows are spans/divs/trs with onClick
- Orrery strips focus outlines from all five interactive element classes but only restores a focus style for two
- Keyboard navigation scrolls the selection out of view -- no scroll-into-view in a keyboard-first tool
- Command palette is invisible: the only entry point is Cmd/Ctrl+K with no on-screen trigger

### [MAJOR] Design system exists on paper only (code-level fragmentation)
*Corroboration: 10 independent findings. Screens: styles.css backbone, all screens*

Both code auditors converge: the 16-slice build re-diverged into twin classes (.xray-tab/.inspector-tab line-for-line duplicates), five KV-grid implementations, three selected-row affordances, ~13 chip species, 40 hand-rolled oklch literals, dead --sp-*/--radius tokens, ~20 dead classes, and one screen's layout coupled to another screen's class name (Prompts/index.tsx:53 rides .band-detail, so editing Memory reflows Prompts). Any visual refresh applied on top of this will re-diverge within a few commits. The redesign's first deliverable should be ~8 enforced primitives plus token normalization with lint guards, executed before restyling -- it converts several 200-site sweeps into 8-line token edits.

Representative findings:
- Shared primitives exist but adoption collapsed -- each slice reinvents cards, tabs, chips, and prop lists
- Type scale is an accretion: 19 distinct sizes, literals outnumber tokens 3:1
- Spacing and radius tokens are declared but used zero times
- Cross-screen class coupling: Prompts layout piggybacks on Memory's .band-detail
- Single 6,590-line stylesheet with broken section organization

### [MINOR] Developer vocabulary leaks into operator copy
*Corroboration: 9 independent findings. Screens: mission, governance, memory, dream, inspector, shell, admin, cognition*

Internal vocabulary ships as UI: 'rank 3' severity-sort jargon, an internal resolver enum ('in_list') as a header badge that flatly contradicts the not-found message beside it, 'r 6 · q 8 · c 3' single letters expanded nowhere, two different X/Y ratio semantics in identical notation on one canvas, '4 rows from getSessions' as a section subtitle, and one screen carrying three names across rail/route/palette. Each is small; together they make the console legible only to its authors. Sweep: one abbreviation grammar, one name per destination, no function names or internal enums in operator-facing strings, tooltips on every coded value.

Representative findings:
- Cryptic governance labels with inconsistent notation: 'cmt 1/2' vs 'dir 27/27' mean different things
- Internal reliability enum ('in_list') rendered as a header badge on every inspected object
- Sort pills leak developer jargon ('backend', 'created new/old')
- One destination, three names: 'cognition' vs 'workbench' vs 'Conversation Workbench'
- Cryptic single-letter metric clusters in top bar

## Top individual issues (worst first)

### 1. [critical] Review evidence comparison letter-shatters ids and values into 1-4-char vertical strips
*bug -- Review evidence comparison; shared .props KV grid (also Governance rail, Inspector)*

The contradiction-adjudication surface -- the console's core human-decision screen -- is unreadable for exactly the data being compared, while the adjacent resolution pane sits ~70% empty.

**Evidence:** view-review.png / live-review.png center pane ('se/mn/_2/d6/y2/zn/s4/bk/8o/4b/6', 'Gatt aca (199 7)'); /home/luth/Programming/borg/demo/web/src/styles.css:3710-3728 (130px key column + overflow-wrap:anywhere); demo/web/src/screens/Review/index.tsx:1066-1067 (forced 3-up grid); SemanticNodeDetail.tsx:60-87. Corroborated by 8 of 20 auditors.

**Recommendation:** Container-aware KV (stack key over value below a min width), shortId+copy chips for ids everywhere, rebalance review-repair-grid so the comparison pane dominates.

### 2. [critical] Stream event list destroys chronological order: all turn-less entries collapse into one global 'unclaimed/maintenance' mega-group
*bug -- Stream grouping*

A stream browser's first invariant is time order; the conversation literally cannot be read in sequence.

**Evidence:** live-operator-chat.png (60-entry group spanning 09:29:06-00:04:5x containing user messages whose answering turns sit in other groups); /home/luth/Programming/borg/demo/web/src/lib/stream-grouping.ts:169-170 (UNCLAIMED_STREAM_GROUP_ID), compareGroupsNewestFirst :145-163.

**Recommendation:** Segment unclaimed runs by time adjacency and interleave chronologically, or make the timeline flat with turn membership as a row affordance.

### 3. [critical] Session history beyond the 120-event window is unreachable: loadOlder is implemented but never wired to any UI
*bug -- Stream*

Auditing anything older than the window is impossible on a live session; the honesty label is a teaser for data the operator cannot reach.

**Evidence:** view-stream.png / live-stream.png ('loaded window only · older entries available' rendered twice); /home/luth/Programming/borg/demo/web/src/hooks/use-stream-window.ts:185-236 (loadOlder exported, zero callers); Stream/index.tsx:1160-1161,1203,1270.

**Recommendation:** Add a 'load older' control wired to the existing hook, mirroring Memory's load-more pattern (Memory/index.tsx:962).

### 4. [critical] Turn replay -- the Cognition screen's core promise -- does not work for any turn predating page load
*backend-gap -- Cognition turn history + X-ray replay*

The turn strip is a list of dead links; the screen cannot answer 'what did the mind do on this turn', which is its reason to exist.

**Evidence:** /home/luth/Programming/borg/demo/web/src/screens/Cognition/index.tsx:446-470 (replay only from in-memory turnStream); Xray.tsx:414-419 ('trace unavailable this browser session'); crops/diff-amplified.png shows zero UI change on turn selection; REDESIGN_PLAN.md:78 names the seam.

**Recommendation:** Persist per-turn phase traces server-side with a GET endpoint; hydrate flow chart + Active Stream on selection. Also fix the RAW tab to fetch like LEDGER does (Xray.tsx:360-370).

### 5. [critical] Status bar renders invented telemetry: hardcoded branch/model/embedding with a permanent green dot
*bug -- StatusBar (all screens)*

Fake instruments poison trust in the real counts beside them; a status light that cannot turn red is decoration.

**Evidence:** /home/luth/Programming/borg/demo/web/src/components/StatusBar.tsx:46-90 ('borg/main', 'opus-4.7', 'qwen3-8b · 4096d' as string literals; repo is on frontend-redesign-p0-shell). Corroborated by 2 auditors.

**Recommendation:** Thread real values from the server snapshot (model/embedder) and a build-time define (branch), or delete the segments.

### 6. [critical] Phantom keyboard chords: ⌘0-⌘9 advertised on three surfaces, no handler exists anywhere
*bug -- Command Palette hints, Rail tooltips, Admin route diagnostics*

A diagnostics table documenting fictional shortcuts is actively misleading and undermines the precision aesthetic.

**Evidence:** /home/luth/Programming/borg/demo/web/src/components/CommandPalette/CommandPalette.tsx:261, components/Rail.tsx:32, screens/Admin/index.tsx:239-240; only hotkey handler in repo is ⌘K (hooks/use-palette-hotkey.ts:32). Corroborated by 3 auditors. Also mac-only ⌘ glyph on a Linux deployment.

**Recommendation:** Implement chord routing (avoiding browser-native Ctrl+digit collisions) with platform-correct modifier display, or delete the hints from all three surfaces.

### 7. [critical] Memory band 'growth' sparklines are fabricated server-side from a single scalar
*bug -- Memory overview band cards*

A chart that looks like growth-over-time but encodes only 'count > 0' is the canonical decoration-pretending-to-be-data failure, and it is the most salient element of each card.

**Evidence:** /home/luth/Programming/borg/demo/server/src/app.ts:1386-1394 (sparkFrom maps count -> 15-point synthetic ramp); demo/web/src/screens/Memory/index.tsx:445; live-cards.png (identical rising ramp on all non-empty bands). Corroborated by 3 auditors; a fourth critiqued normalization without spotting the fabrication.

**Recommendation:** Bucket real counts by created_at (stores have it) or delete the chart and show count + delta-since-last-tick.

### 8. [critical] Dream 'schedule lane' is run history mislabeled as scheduled, and its unbounded 80-row table exiles the cockpit ~3 viewports below the fold
*backend-gap -- Dream Ops*

An operator believes 80 runs are queued when nothing is scheduled, and the process cards, audit log, and revert actions are invisible without ~3300px of scrolling.

**Evidence:** view-dream.png (header 'SCHEDULE LANE', column 'SCHEDULED', all timestamps past; 14 of 80 rows fill two-thirds of 1080p); /home/luth/Programming/borg/demo/server/src/app.ts:1921-1938, 2173-2179 (applied_at mapped into scheduled_at); Dream/index.tsx:731 mounts the lane above the grid; no max-height (styles.css:5248-5250).

**Recommendation:** Compute real next light/heavy ticks from scheduler intervals (data is on-screen: 240m/1440m), group history rows by run_id like the audit log already does (index.tsx:182-198), cap the lane's height, and move it below the cockpit.

### 9. [critical] Memory band-browser switcher strip collapses to unreadable ~14px slivers
*bug -- Memory band browser tab strip*

Primary navigation inside the band browser is unreadable -- a layout bug, not a styling choice.

**Evidence:** live-memory-band.png (eight empty bordered rectangles, all text clipped); /home/luth/Programming/borg/demo/web/src/screens/Memory/index.tsx:1683 re-renders full BandOverviewBar inside .full-page; styles.css:3393-3401 (overflow-x:auto zeroes min-height), 4766-4771, 3506-3509 (.band-detail over-claims height).

**Recommendation:** flex-shrink:0 on the strip + min-height:0 on .band-detail as the hotfix; redesign the drill-mode switcher as a compact tab row instead of reusing full overview cards.

### 10. [critical] Orrery labels and marks collide systematically -- no collision management exists
*design -- Mission Control orrery*

The landing screen's centerpiece is unreadable at multiple clock positions, and collisions shift with satellite count so it cannot be hand-tuned away.

**Evidence:** view-mission.png ('review 6' struck through 'ruminator'; 'social' over 'auto narrator'; 'reflector' struck by its own mark); /home/luth/Programming/borg/demo/web/src/screens/MissionControl/Orrery.tsx:342-344 (fault label hard-coded x=430,y=92), :228-264 (data-dependent satellite angles, fixed +18 offset); four systems share one ~26px annulus (r=228-246). Corroborated by 3 auditors; worse with live data (live-palette.png background).

**Recommendation:** Tangential anchoring + leader lines + a deterministic collision pass; give each system its own radius band; position fault cluster relative to satellite angles.

### 11. [critical] Cognition flow chart is invisible at rest (~1.3:1 contrast) while decorative particles glow in full accent green
*design -- Cognition X-ray flow canvas*

62% of the screen communicates nothing at rest: structure is below the 3:1 non-text floor while decoration outshines data, inverting the design system's own green-is-live rule.

**Evidence:** live-cognition.png / crops/flow-native.png (particles brighter than node rings); /home/luth/Programming/borg/demo/web/src/styles.css:2546-2559 (queue strokes --line-soft on --bg-0), 2390-2392; ParticleField.tsx:242-267 (full-brightness accent + glow, unconditional rAF loop); FlowChart.tsx:649 (particleDensity=320). REDESIGN_PLAN.md:939 forbids animated backgrounds behind dense data.

**Recommendation:** Raise resting structure to >=3:1, gate particles/glow on an active phase, reserve saturated accent exclusively for live state.

### 12. [critical] Severity system inverted: purple sits above red and rank is computed from raw row count
*design -- Mission Control attention rail, rail badges, SeverityChip*

The cockpit cries wolf on healthy throughput and renders the worst tier in a calmer hue than tier 3 -- operators learn to ignore color entirely.

**Evidence:** view-mission.png rail (19 routine pending extractions = 'rank 4' purple; 3 healthy active commitments = 'rank 3' red); /home/luth/Programming/borg/demo/web/src/screens/MissionControl/useAttentionData.ts:116-127 (count>=10 -> rank 4); styles.css:50-53 (--sev-3 red, --sev-4 purple, colliding with purple=dream/interior). Corroborated by 2 auditors.

**Recommendation:** Severity encodes state (errors/conflicts/overdue), never volume; 3-step ramp ending in red; purple off the ramp; plain neutral counts for volume.

### 13. [critical] AUDIENCE pill -- the most disclosure-critical instrument -- shows a raw 50-char thread id, and invents 'alice' when no session matches
*backend-gap -- InstrumentStrip / shell; BotArena connector*

Audience/role/policy are what an operator checks before trusting a disclosure decision; one is unreadable and the fallback fabricates an audience.

**Evidence:** live-stream.png top strip ('AUDIENCE botarena_thread:ebe83f0a-...'); /home/luth/Programming/borg/demo/web/src/AppShell.tsx:142 (audience_label verbatim; connector never sets a human label), :34 (literal 'alice' fallback).

**Recommendation:** Connector populates a human audience label; frontend degrades to a prefix-stripped short id and '--', never a full UUID or an invented name.

### 14. [major] Clock labeled UTC displays local time
*bug -- InstrumentStrip (all screens)*

A mislabeled timezone silently corrupts every cross-referenced log timestamp in an ops console.

**Evidence:** view-admin.png top right ('UTC 10:53:38'); /home/luth/Programming/borg/demo/web/src/AppShell.tsx:36-40 (getHours/getMinutes, not getUTC*); InstrumentStrip.tsx:106-109.

**Recommendation:** Use getUTC* or relabel 'local'. One-line fix; do it before redesign.

### 15. [major] Red is spent on steady-state configuration: a permanent ~100-degree red arc because one critical-class commitment exists
*design -- Mission Control orrery; red usage console-wide*

Any healthy deployment shows a permanent alarm ring on the landing screen; when a real violation occurs there is no headroom left.

**Evidence:** view-mission.png left side; /home/luth/Programming/borg/demo/web/src/screens/MissionControl/Orrery.tsx:276-291 (fixed arcPath sweep, color flips to --bad when critical>0); also Stream/index.tsx:680 (boolean false = red), :1339 (archived session = red); red 'P 100' priorities (DirectivesTab.tsx:938).

**Recommendation:** Reserve --bad for failure/suppression/violation events per the plan's own contract (REDESIGN_PLAN.md:813); critical-class is a neutral label, arc length should encode magnitude.

### 16. [major] Transcript speakers are labeled with the raw thread UUID; sender_entity_id exists but is never used
*design -- Cognition transcript*

On a debugging surface, every non-borg participant in a multi-party thread is labeled identically -- attribution is actively misleading, and the actual speaker (Lunaria) is only discoverable inside envelope body text.

**Evidence:** live-cognition.png (46-char id rendered twice per message header, uppercased and tracked to ~2 lines; every speaker's avatar initial is 'B'); /home/luth/Programming/borg/demo/web/src/screens/Cognition/ChatMessage.tsx:23-36, 54-59. Same id appears ~8x simultaneously across the workbench (ChatStream.tsx:121-124, ChatInput.tsx:179-182).

**Recommendation:** Resolve sender_entity_id and audience to human labels; one canonical identity per screen; raw id demoted to Inspector/hover.

### 17. [major] Transport errors render as calm empty states in Mission Control cards
*bug -- Mission Control attention cards; state rendering console-wide*

An operator console must never render faults in the voice of 'nothing to do' -- a dead backend looks like a quiet day.

**Evidence:** /home/luth/Programming/borg/demo/web/src/screens/MissionControl/index.tsx:61-70 (CardNotice returns <Empty>error: {error}</Empty>); components/ErrorState.tsx exists unused here; audit-mission-rail.png shows the identical idiom for healthy-empty.

**Recommendation:** Route all errors through ErrorState with distinct warn styling + retry; sweep the ~17 hand-rolled loading and 15+ error variants onto the shared components with a CI guard.

### 18. [major] Interactive chips and static tags are visually identical console-wide
*design -- All screens (Tag vs IdRef/btn idiom)*

The redesign's core interaction ('every id is clickable') is invisible at rest; operators must mouse-probe to discover what's actionable.

**Evidence:** /home/luth/Programming/borg/demo/web/src/styles.css:1179-1190 (.tag) vs 1295-1352 (.btn.sm.ghost) -- same 1px border, ~0.5px font delta; 183 Tag vs 84 IdRef sites; crops: mission rail ('contradiction' tag beside 'inspect' button), identity ('forget' identical to evidence chip), dream lane. Corroborated by 7 auditors.

**Recommendation:** Tags go borderless/tinted; interactive chips get a reserved affordance (underline/chevron/accent edge) plus hover-independent cues; collapse the ~13 chip implementations into Tag/CountBadge/IdRef.

### 19. [major] Destructive actions wear safe/primary styling; no danger button variant exists
*design -- Governance, Review, Admin, Identity*

The visual danger ranking contradicts the consequence ranking; wiping the substrate is styled like 'apply identity edit'.

**Evidence:** /home/luth/Programming/borg/demo/web/src/screens/Governance/DirectivesTab.tsx:949 (revoke = btn primary green) vs CommitmentsTab.tsx:469 (same verb ghost); Review/index.tsx:1136-1138 (supersede primary) vs :104 (declared destructive); Admin/index.tsx:313 (reset = amber live-write, same class as routine identity edits); Identity/index.tsx:222-235 (forget = ghost, identical to 'why'). REDESIGN_PLAN required reset to be --bad.

**Recommendation:** Add .btn.danger; assign variants by action class (primary=safe, live-write=mutating, danger=irreversible) and apply mechanically.

### 20. [major] Supersede silently pre-selects a winner by array order and confirms without naming winner/loser
*design -- Review resolution*

A destructive memory operation can be executed with zero operator input on a default that reads as a recommendation but is just refs-array order.

**Evidence:** /home/luth/Programming/borg/demo/web/src/screens/Review/index.tsx:2128-2129 (selectedWinner defaults to selectedNodeIds[0], defeating the needsWinner guard at :1133-1139), :2376-2384 (modal says only 'Confirm supersede for review 36.').

**Recommendation:** No default winner; keep destructive actions disabled until explicit pick; confirm modal must state 'X supersedes Y'.

### 21. [major] Dateless HH:MM:SS timestamps console-wide produce backwards-looking ranges over multi-day data
*bug -- Stream, Memory, Review, Dream (shared formatTime)*

Multi-day memory and queue data looks corrupted or non-monotonic; item age is unknowable.

**Evidence:** /home/luth/Programming/borg/demo/web/src/lib/stream-utils.ts:43-54; band-tl.png 'time range 13:16:54 - 08:54:43' (end before start); live-operator-chat.png group '09:29:06-00:04:5x'; review queue 19:52:59 above 04:50:58; dream rows all '19:18:40' across days. Corroborated by 5 auditors.

**Recommendation:** Date-aware ranges and day separators at the shared formatter; relative age in queues with absolute on hover; also format epoch-ms in Inspector Summary (Inspector.tsx:161-181).

### 22. [major] Connector envelope renders verbatim, burying actual message content in transcript bodies and consuming every stream preview
*backend-gap -- Cognition transcript, Stream row previews; BotArena connector (external sol-connector)*

The 180-char preview budget and 1.5-message viewport are spent entirely on routing meta; message content is never visible in the list.

**Evidence:** live-cognition.png / live-stream.png ('[BotArena thread ...] message from Lunaria (bot) | bot-chain-depth: 24 | addressed to you: yes ...' before any content; seven consecutive identical previews truncating exactly where content would start). Corroborated by 4 auditors.

**Recommendation:** Connector persists the envelope as structured stream-entry fields (speaker, chain depth, addressed/mentioned), leaving content.text human. Do NOT regex-strip in the frontend -- that bakes one connector's format in and violates project rules.

### 23. [major] Palette-to-Inspector id round-trip is broken: truncated display ids can never resolve, and numeric families are unreachable
*bug -- Command Palette + Inspector resolution*

Every shortened id the UI displays is a dead end when fed back to the console's own search; the universal-inspector promise fails at its entry point.

**Evidence:** live-palette-id.png / live-inspector-commitment.png ('Open Commitment cmt_p48r' offered with no existence check, then 'not found in the loaded list'); /home/luth/Programming/borg/demo/web/src/components/CommandPalette/CommandPalette.tsx:283-309 (prefix-sniff only), :191-192 (any underscored string suppresses memory search); inspector-id.ts:69-80; typing the console's own label 'review 36' finds nothing (live-palette-query.png).

**Recommendation:** Verify resolvability before offering; prefix/substring-match loaded lists and surface real candidates; typed openers for numeric families keyed on the registry; copy-full-id affordance on every truncated id.

### 24. [major] Mouse-only interactions and stripped focus outlines across half the console
*bug -- Governance, Stream, Memory, Orrery, Command Palette*

Keyboard operation is impossible or hazardous on core surfaces; the correct pattern already exists in-repo (Review/index.tsx:1531-1543).

**Evidence:** /home/luth/Programming/borg/demo/web/src/screens/Governance/CommitmentsTab.tsx:513-560 (<span onClick> pills, no role/tabIndex); Stream/index.tsx:933-935; Memory/index.tsx:983-996, 1655; styles.css:5866-5873 (outline:none on 5 orrery classes, :focus-visible restored for only 2); CommandPalette.tsx:447-459 (no scrollIntoView -- Enter can fire an unseen command, including 'Reset demo' below the fold).

**Recommendation:** Standardize on the Review-row pattern, add missing :focus-visible rules, scrollIntoView on palette index change.

### 25. [major] Transcript capped at 16 entries with no scroll-back; 111 of 127 live messages unreachable from the conversation surface
*design -- Cognition transcript*

The screen's namesake content is mostly inaccessible while 62% of the screen idles on the flow canvas.

**Evidence:** /home/luth/Programming/borg/demo/web/src/screens/Cognition/index.tsx:38 (CHAT_PANEL_LIMIT=16), :435-441; fixed 176px turn strip (styles.css:1990) leaves ~1.5 visible messages at 1080p (live-cognition.png).

**Recommendation:** Backward pagination (API already takes limit; add before-cursor), collapsible participation panel and turn strip, rebalance the 38%/62% split toward the conversation.

### 26. [major] Void layouts: Memory, Admin, and Identity strand 55-60% of the viewport while their content starves
*design -- Memory overview, Admin, Identity*

Entry-point screens read as abandoned, real content (per-band previews, goals kanban, open questions) is invisible or buried six screens deep, and static filler panels train operators to ignore regions.

**Evidence:** view-memory.png + live-memory.png (identical void with 145 episodes/304 nodes -- layout, not thinness; hardcoded 'identity governance' filler at /home/luth/Programming/borg/demo/web/src/screens/Memory/index.tsx:374-390); view-admin.png (styles.css:4971-4977 single auto-fit row); view-identity.png (values board stretched ~700px to match 39 trait cards via paired span-6 at Identity/index.tsx:926,975; whole page scrolls ~10,000px against the console's own idiom, REDESIGN_PLAN.md:882). Corroborated by 6 auditors across the three screens.

**Recommendation:** Memory overview becomes a real atlas (per-band previews, AtlasPlots thumbnails); Identity regions get viewport-relative heights with internal scroll and jump-links; Admin re-laid as an asymmetric grid with runtime health primary.

### 27. [major] Scope-matrix and rollup counts are client aggregations over a limit-50 window presented as authoritative totals
*backend-gap -- Governance scope matrix*

At live-data scale these numbers will be silently wrong, which is worse than absent -- and this screen is the governance ground truth.

**Evidence:** view-governance-scope.png ('2 active / 1 critical', '5 rows' as totals); /home/luth/Programming/borg/demo/web/src/screens/Governance/ScopeMatrixTab.tsx:268-348 aggregates default-window fetches; the project's own plan flags exactly this (REDESIGN_PLAN.md:164, :2579 'must not present a window as a total').

**Recommendation:** Server-side counts, or stamp every aggregate with a windowing affordance ('first N loaded', reuse count_is_lower_bound).

### 28. [major] Functional text rendered below the legibility floor: 2.3-3.5:1 contrast tokens and an 8-11px center of gravity
*design -- All screens (tokens + rail + SVG labels)*

Sustained-strain reading on every screen, and because everything is small and faint, hierarchy flattens -- nothing reads as primary.

**Evidence:** --text-ghost 2.34:1 on functional text (session group headers, shortcut digits; /home/luth/Programming/borg/demo/web/src/styles.css:899-905, 760-765); 9px eyebrow layer at 3.52:1 across 83 sites (:160-164); 114 of 267 font-size declarations at 10-11px vs the documented 12.5px body used ~10 times; rail labels 8.5px (:754); SVG labels ~6.7-8px effective (FlowChart viewBox scaling). Corroborated by 4 auditors.

**Recommendation:** Token-level fix: raise --text-faint to ~4.5:1, restrict --text-ghost to disabled glyphs, 12px data-row floor, ban literal px font-size via stylelint so the scale becomes enforceable.

## Functional bugs (confirmed, file:line)
- Letter-shattering CSS: .props 130px key column + overflow-wrap:anywhere shreds ids/aliases into 1-4-char lines in Review candidate cards and Governance rail (/home/luth/Programming/borg/demo/web/src/styles.css:3710-3728, 1346-1353; Review/index.tsx:1066-1067) -- 8 auditors
- Stream grouping buckets ALL turn-less entries into one global 'unclaimed/maintenance' mega-group sorted as a unit, destroying chronological order (demo/web/src/lib/stream-grouping.ts:169-170, 145-163)
- Stream older history unreachable: useStreamWindow.loadOlder fully implemented but no UI element invokes it, while two labels advertise 'older entries available' (demo/web/src/hooks/use-stream-window.ts:185-236; Stream/index.tsx:1160-1270)
- Turn replay nonfunctional for any turn predating page load: flow snapshots are browser-memory-only, no persistence/fetch API (Cognition/index.tsx:446-470; Xray.tsx:414-419)
- RAW tab never calls getLedger -- permanently empty for every turn after reload, though adjacent tabs fetch fine (Xray.tsx:360-370)
- StatusBar hardcodes branch 'borg/main', model 'opus-4.7', embedding 'qwen3-8b · 4096d', and a static green ok dot (demo/web/src/components/StatusBar.tsx:46-90); repo is on frontend-redesign-p0-shell
- ⌘0-⌘9 chords advertised in palette hints, rail tooltips, and Admin diagnostics with no handler anywhere; only ⌘K exists (CommandPalette.tsx:261, Rail.tsx:32, Admin/index.tsx:239-240, use-palette-hotkey.ts:32)
- Clock labeled UTC shows local time (AppShell.tsx:36-40 uses getHours not getUTCHours; InstrumentStrip.tsx:106-109)
- Breadcrumb renders raw sessionId instead of the session label, contradicting the shell spec (InstrumentStrip.tsx:54 vs REDESIGN_PLAN.md:301); AppShell.tsx:34 falls back to inventing audience 'alice'
- Band growth sparklines synthesized server-side from a single scalar via sparkFrom (demo/server/src/app.ts:1386-1394), rendered as real time series (Memory/index.tsx:445)
- Memory band-switcher strip collapses to ~14px text-clipped slivers: overflow-x:auto zeroes min-height and .band-detail absorbs all flex height (Memory/index.tsx:1683; styles.css:3393-3401, 4766-4771, 3506-3509)
- formatTime drops dates console-wide, producing backwards ranges ('13:16:54 - 08:54:43') in Memory episodes, Stream group headers, Review queue, Dream lane (demo/web/src/lib/stream-utils.ts:43-54)
- displayValue's internal-id regex matches semantic enums: 'scheduled_reflection' renders as 'schedule…tion'; any single-underscore snake_case value >14 chars affected console-wide (demo/web/src/screens/screen-utils.ts:21,44)
- Chip text hard-clips mid-glyph without ellipsis in stream tag-summary rows (display:flex defeats text-overflow; styles.css:3346-3356) -- 2 auditors
- 'TAILING' live badge renders unconditionally regardless of WebSocket state (Stream/index.tsx:1271-1273; no connection status exposed from use-stream-window.ts)
- AttachmentsCard: count=null renders an eternal false 'syncing' tag, and its 'needs backend' header action is silently dropped because CardShell only renders action when onAction is defined (MissionControl/index.tsx:41, 88-89, 452-458; useAttentionData.ts:354-358)
- Prompts section chips silently dead: focusSection does text.indexOf(label) but section labels are never embedded in the assembled text -- every click no-ops; scroll math also hardcodes line*18 (AssembledPromptPane.tsx:99-113)
- Three contradictory block counts on Prompts (header says 5, list shows 7, preview shows 9); client PromptKey union stale at 5 keys (Prompts/index.tsx:50; demo/web/src/api/types.ts:1236-1241)
- Prompt diff header claims 'SAVED STATIC OVERRIDE' while the right column is the live unsaved draft (PromptEditor.tsx:74-77 vs :292); drafts also silently discarded on navigation with no dirty badge (AppShell.tsx:264)
- Palette id round-trip broken: prefix-sniff offers 'Open Commitment cmt_p48r' with no existence check, Inspector then fails on the truncated fragment; pasting the displayed string fails too (contains literal '…') (CommandPalette.tsx:283-309; Inspector.tsx:87-95; inspector-registry.ts:591-594)
- Palette arrow-key navigation has no scrollIntoView; Enter can fire an invisible below-fold command including 'Reset demo' (CommandPalette.tsx:447-459; ~8 visible rows of an 18+ row catalog)
- Mouse-only interactions: governance/memory filter pills are <span onClick>, governance rows <tr onClick>, stream and memory rows <div onClick>, band back-button a <span> -- no role/tabIndex/keys (CommitmentsTab.tsx:513-560; Stream/index.tsx:933-935; Memory/index.tsx:660-668, 983-996, 1655); correct pattern exists at Review/index.tsx:1531-1543
- Orrery strips outline:none from five interactive classes but restores :focus-visible for only two; satellites/fault/turn-dot are tab-focusable with invisible focus (styles.css:5866-5873 vs 5885-5890, 6040-6044)
- MissionControl renders transport errors through <Empty> -- failure indistinguishable from healthy-empty; ErrorState component exists unused there (MissionControl/index.tsx:61-70)
- Review supersede pre-selects winners[0] silently, defeating the needsWinner disable-guard; confirm modal never states winner/loser (Review/index.tsx:2128-2129, 2019, 2376-2384)
- 'open only' review checkbox renders browser-default blue: accent-color scoped solely to the Dream toggle (styles.css:5240); same class of issue as the unstyled webkit search-cancel button in the palette (CommandPalette.tsx:490)
- Prompts layout coupled to Memory's class: className='prompt-lab band-detail' means edits to Memory's .band-detail grid silently reflow Prompts (Prompts/index.tsx:53; styles.css:338, 3506)
- Dream 'SCHEDULED' column shows past applied_at/report timestamps -- run history mislabeled as a schedule; no future-run synthesis despite intervals being known (demo/server/src/app.ts:1921-1938, 2173-2179)
- Scope-matrix '1 sessions' pluralization on every rollup card (ScopeMatrixTab.tsx:376)
- Backend data bugs: established traits duplicated as 'intellectual_honesty' vs 'intellectual honesty' (no canonicalization/merge path); affective band count silently session-scoped while all other bands are global (app.ts:1985); semantic node labels written as raw entity ids surface in palette results
- Stream filter changes silently reset selection and collapsed groups; deselecting the last kind silently no-ops with no feedback (Stream/index.tsx:1060-1065, 1164-1167)
- Palette memory search fans out 3 uncached requests per keystroke with no debounce; Promise.all means one band failure wipes all hits (CommandPalette.tsx:204-244, 327-338; boundedAllSettled exists unused at inspector-registry.ts:130)
- Inspector tab strip renders a clipped ~10px sliver after RAW JSON that reads as a broken seventh tab (styles.css:5374-5402; crops/insp-tabs-right.png)

## Auditor contradictions, adjudicated
- Connector envelope handling: dim:hierarchy-density recommended parsing/de-emphasizing the '[BotArena thread...]' prefix in the frontend (dim prefix, bold payload), while screen:stream and screen:cognition explicitly said do NOT regex-strip in the UI. Adjudication: the screen auditors are right and aligned with the project's cardinal rules -- pattern-matching message content in the frontend bakes one connector's string format into the UI; the fix is upstream (connector stores envelope as structured stream-entry fields), with the UI rendering those fields as meta chips.
- Memory sparklines: dim:hierarchy-density critiqued the sparkline's self-normalization (2-item band looks like a 304-item band) as if the series were real; screen:memory and dim:design-language traced it to server-side fabrication (sparkFrom at app.ts:1386-1394 synthesizes 15 points from one scalar). Adjudication: fabrication is the root issue -- normalization critique is moot; bind real created_at-bucketed history or delete the chart.
- Stream detail BODY vs RAW duplication: dim:layout-space said drop BODY and keep only pretty-printed RAW; screen:stream said render BODY only when it differs from RAW (text-bearing content) with RAW as a collapsed toggle. Adjudication: screen:stream's conditional rule wins -- prose content should be readable as text without JSON escaping, structured content gets one pretty JSON view; never both copies.
- Cognition flow canvas fate: dim:affordances recommended investing (wire pipeline nodes to focus the corresponding X-ray tab); screen:cognition and dim:hierarchy-density recommended shrinking it to a compact strip when idle. Adjudication: state-dependent, not contradictory -- compact and calm at idle (no particles, one-line stream strip), prominent and interactive (clickable nodes, real replay) during a live turn or replay; in all states decoration must never outshine data.
- Where the counts live: screen:shell said actionable counts belong on the rail as navigational badges; dim:design-language said the InstrumentStrip should be the single home for session-state scalars. Adjudication: compatible once 'one home per metric' is enforced -- actionable backlogs (open reviews, conflicts) as clickable rail badges, mood/ws once in the strip, runtime facts in the status bar, and the duplicated COUNTS pill plus bottom-bar mood/count segments deleted. Both agree inventory totals (DRM 256) should not be badges at all.
- Amber for danger: dim:color-semantics allowed keeping amber for the live-write affordance (shape-differentiated); screen:admin demanded reset get a red danger tier above it per the plan's own spec. Adjudication: both -- introduce .btn.danger (--bad) for irreversible actions (reset, forget, invalidate, revoke), keep amber live-write for reversible entity-state mutations; this resolves the inversion all five destructive-grammar findings describe.
- Orrery fate: screen:mission offered two paths (make the radial earn its ink with real encodings, or shrink it to a status glyph beside a scannable band list); dim:design-language called it screensaver cosplay; nobody defended the current form. Adjudication: not a true disagreement, but the default should be demote -- the data shows all information is carried by scattered text today; only invest in the radial if the redesign commits to real encodings (proportional arcs, per-band navigation targets, collision-managed labels), otherwise an 8-row table wins.

## Gap-round findings (responsiveness, motion/CPU, cold start, WhyDrawer, disclosure labels)

### [critical] Review evidence panel letter-shatters node ids into vertical one-glyph-per-line slivers at common laptop width
*bug -- Review (evidence comparison / semantic drill-through)*

**Evidence:** /tmp/borg-ui-audit/responsive/z-1366-review-sliver.png (1366x768 capture: a ~10px-wide column rendering 's/e/m' stacked vertically between candidate 1 and the contradicts panel) and z-960-review-stack.png (960px: full id 'semn_2d6y2zns4b8o4b6' rendered one glyph per line across ~20 rows, with 'f/i/l/m' shattering below it). Cause: inline nested grid `gridTemplateColumns: "1fr minmax(220px, 300px) 1fr"` at demo/web/src/screens/Review/index.tsx:1066-1067 inside the `.review-repair-grid` track minmax(380px,1.45fr) (demo/web/src/styles.css:3605); the 1fr candidate columns have ~1ch min-content because id chips use overflow-wrap:anywhere, and no media query exists for the review screen (only 4 in the whole 6,590-line stylesheet, none touching review).

**Recommendation:** Give candidate columns a real minimum (minmax(220px,1fr)) and stack the three-way comparison vertically when the evidence pane is narrower than ~700px. Render long internal ids as middle-ellipsized text with a copy affordance instead of overflow-wrap:anywhere chips -- an id that wraps at all is unreadable; one that wraps per-glyph is broken.

### [major] Review resolution panel (the screen's primary actions) pushed off-viewport at half-screen width
*bug -- Review (repair workspace)*

**Evidence:** /tmp/borg-ui-audit/responsive/r960-review.png: at 960px the RESOLUTION column (note, winner node, keep both/supersede/invalidate/dismiss) is entirely absent; the evidence pane is cut mid-token at the viewport edge ('confidence 0.', clipped id chips), and the bottom status bar clips at 'EMB q'. Cause: `.review-repair-grid` min tracks 240+380+260px + 24px gaps ≈ 904px (styles.css:3605) plus rail 60px + session lane 232px (styles.css:90,93) ≈ 1196px minimum vs 960 available. The only path to the actions is an 8px-thin horizontal scrollbar (styles.css:170-173) with no other affordance.

**Recommendation:** Below ~1250px, collapse to a two-pane layout (queue becomes a collapsible drawer or top strip) and dock resolution actions under the evidence pane. Resolve/supersede/invalidate must never live behind sideways scroll.

### [major] Mission orrery labels collide and clip at 1366 and below; collisions are width-induced (clean at 1920)
*bug -- Mission Control (cognitive orrery)*

**Evidence:** /tmp/borg-ui-audit/responsive/z-1366-orr-semantic.png ('semantic extractor' satellite label printed directly over the 'cmt 1/2' / 'dir 27/27' center stats), z-1366-orr-relational.png ('relational/19' band label on top of 'procedural synthesizer'; 'belief reviser' clipped to 'elief reviser' at the column edge), z-960-orrery.png (same plus 'review 6'/'commitments' piled into the red satellite cluster, 'semantic' clipped by the session lane). Same regions verified clean at 1920 in z-1920-orr-semantic3.png. Cause: rings scale down with the column but label font-size and offsets do not, and there is no collision avoidance; the single orrery breakpoint at 1180px (styles.css:5811-5814) only shrinks the column tracks further.

**Recommendation:** Make label rendering scale-aware: below a radius threshold, cull satellite labels to hover/focus tooltips and keep only band names with leader lines; alternatively cap minimum orrery diameter and switch to the compact list rendering below it.

### [major] Governance commitments table hides 6 of 9 columns at 960 (3 at 1366) behind an unindicated horizontal scroll, beside a fixed 320px detail panel
*bug -- Governance (commitments tab; same pattern in sessions/entities tables)*

**Evidence:** /tmp/borg-ui-audit/responsive/z-960-gov-table.png: only ID and TEXT columns visible -- family, audience, enforce, state, p, since, and the per-row action column are all off-pane; z-1366-gov-table.png shows the STATE header clipped to 'ST' at the detail-panel border. Cause: fixed th widths summing ~1004px (demo/web/src/screens/Governance/CommitmentsTab.tsx:587-595) inside a `overflow: auto` wrapper (line 583) whose scrollbar is 8px and invisible until used, with a hard-coded 320px detail column (line 581: `gridTemplateColumns: "minmax(0, 1fr) 320px"`).

**Recommendation:** Drop columns by priority at narrow widths (keep id, text, enforce, actions) and surface the dropped fields in the already-present detail panel; make the detail column minmax(260px,320px). If horizontal scroll must remain, add a visible scroll affordance and pin the action column.

### [major] App shell hard-clips with no scroll path below 900px, making the right edge unreachable; the only sub-900 breakpoints reflow inside an already-clipped shell
*bug -- App shell (all screens)*

**Evidence:** /tmp/borg-ui-audit/responsive/z-800-gov-right.png (800px-wide capture: detail-panel prose cut mid-word -- 'When responding in the botare...', 'include the @handle when add...' -- with no scrollbar). Cause: `.app { min-width: 900px }` (styles.css:196) while body and #root are `overflow: hidden` (styles.css:117-118, 121-123), so below 900px CSS width the overflow is simply unreachable. This is exactly what 200% browser zoom on a 1366 laptop produces (683px CSS viewport = 217px of UI permanently lost). It also makes the two 760px media queries (styles.css:4664, 5295) functionally dead: they fire on viewport width but reflow content inside a shell that is still 900px wide and clipped.

**Recommendation:** Either allow root horizontal scrolling when the viewport is under the shell minimum (one-line fix: overflow-x auto on body), or lower the real minimum with responsive behavior. Delete or rehome the 760px queries once the actual floor is decided.

### [major] No responsive system: 4 media queries against 60 grid-template-columns in 6,590 lines; review, cognition, governance, stream, memory, and prompts have zero breakpoints
*design -- styles.css (all screens)*

**Evidence:** grep of demo/web/src/styles.css: only @media rules are lines 4644/4664 (identity), 5295 (dream workbench controls), 5811 (orrery) -- vs 60 grid-template-columns declarations defining multi-column workspaces. Concrete consequence beyond the captured views: the prompts lab needs 320+420+360px tracks (styles.css:338-339) + 292px of fixed rail/sidebar chrome (styles.css:90,93) ≈ 1392px minimum, so it clips even at 1366x768, the most common laptop resolution. Fixed chrome (60px rail + 232px session lane) consumes 21% of a 1366 viewport and 30% of a 960 one before any screen content renders.

**Recommendation:** The redesign should establish 3-4 named layout tiers (e.g. >=1600 full cockpit, 1200-1600 two-pane, <1200 single-pane with drawers) applied via shared shell classes, instead of per-screen pixel patches. Make the session lane collapsible below the middle tier -- it already has a collapse chevron at full width.

### [minor] Topbar degrades incoherently when narrow: orphaned breadcrumb separators and telemetry clipped mid-glyph; a raw 52-char thread id evicts mood/counts/turn/clock
*bug -- Topbar (all screens)*

**Evidence:** /tmp/borg-ui-audit/responsive/z-960-mission-topbar.png: breadcrumb collapses to two bare '›' glyphs with zero-width segments, and the right end clips mid-segment. z-960-cog-topbar.png (live session): the audience value renders the full 'botarena_thread:ebe83f0a-e701-463b-a710-f51788a0a555', pushing MOOD/COUNTS/TURN/VER/UTC off the hidden-overflow edge (styles.css:200-210). At 1366 the same bar barely fits only because the demo breadcrumb is short.

**Recommendation:** Truncate audience ids to a short form (label or first/last 4 of the uuid) in the topbar, hide whole segments in priority order (UTC and VER first) instead of clipping mid-glyph, and never render a crumb separator whose segment has ellipsized to nothing.

### [minor] Cognition chat lane pinned to minmax(320px, 38%) at every width while the mostly-empty x-ray particle field keeps the majority share
*design -- Cognition workbench*

**Evidence:** /tmp/borg-ui-audit/responsive/z-960-cog-main.png (live 127-message session at 960px: transcript wraps at ~28-30 characters per line in a ~320px lane while the right pane is a near-empty dot field with one small 'waiting for a running phase' box); same proportions at 1366 (z-1366-cog-main.png). Cause: `.cog { grid-template-columns: minmax(320px, 38%) 1px minmax(0, 1fr) }` (styles.css:1437-1441) with no breakpoint. This is a real-data capture, so it is layout priority, not data thinness.

**Recommendation:** Below ~1200px invert priority: conversation gets the flexible track and the x-ray collapses to a tab/toggle (its tab strip already scrolls, styles.css:2954-2956). At all widths consider letting chat take 50%+ when the x-ray has no running phase.

### [critical] Memory semantic topology is a runaway animation: force-sim never settles + perpetual edge dash-march, measured at ~326% of one core at idle
*bug -- Memory screen, SemanticTopology panel*

**Evidence:** Measured with headless chromium driven into Memory > semantic band > topology pill (300 nodes rendered, confirmed via CDP): 3263 jiffies over a 10s post-settle idle sample = ~326% of one core, vs 17.6% on the static Identity control (same methodology; software-raster --disable-gpu inflates absolute numbers but the 18x ratio vs control stands). Root causes in code: demo/web/src/screens/Memory/SemanticTopology.tsx:493-514 -- tick() unconditionally re-arms requestAnimationFrame with no alpha-threshold stop; line 500 deliberately RAISES the alpha target after settling (elapsed > SETTLE_MS ? 0.05 : 0.012), and line 411 adds a perpetual sinusoidal drift, so every node and edge gets SVG attribute writes 60x/sec forever. On top of that, styles.css:4109-4117 (@keyframes semantic-topology-edge-flow, 6s linear infinite) is applied by edge KIND, not by state -- SemanticTopology.tsx:27,29 hard-bakes the class onto every causes/supports edge -- so hundreds of dashed SVG paths repaint continuously even if the sim were stopped. No visibility/occlusion pause either (only unmount cancels).

**Recommendation:** Make the sim sleep: decay alpha to a threshold and cancelAnimationFrame once settled (standard d3-force behavior), waking only on drag/select/data change; if ambient drift is wanted, run it at a low tick rate (e.g. 10-12fps) for a bounded period. Gate the edge dash-march on a meaningful state (selected/hovered/recently-touched edge) instead of edge kind, and pause everything under prefers-reduced-motion. Re-measure after the fix; target should be within a few percent of the static control once settled.

### [major] Cognition ParticleField burns ~90% of a core at idle, forever, in its default state
*bug -- Cognition screen, FlowChart/ParticleField canvas*

**Evidence:** Measured: ?view=cognition idles at 916 jiffies/10s = ~91.6% of one core vs 17.6% static control (5.2x). Code: demo/web/src/screens/Cognition/ParticleField.tsx -- when no node is running (target=null, i.e. the screen's default idle state, visible as the green wisp field over the flow canvas in view-cognition.png right pane and live-cognition.png), lines 174-183 keep the full 60fps loop running in 'gentle sway' mode ('the field still breathes with no lensing'). Each frame integrates 320 particles AND allocates ~321 createRadialGradient objects + ~640 path fills (lines 238-271) -- roughly 19k gradient constructions/second for pure decoration. The field is enabled by default whenever not replaying history (Cognition/index.tsx:552 particleEnabled={replayTurnId === null}; density default 320 at FlowChart.tsx:648-649). Credit where due: it does pause on document visibilitychange (ParticleField.tsx:143-152) -- but an operator console's normal failure mode is being left open and visible.

**Recommendation:** Idle-throttle the field: when target is null, drop to ~10-15fps or fade the field out entirely after a few seconds and wake it on turn activity (motion would then actually signal 'a turn is running'). Replace per-frame createRadialGradient with pre-rendered offscreen glow sprites (two tiers) blitted via drawImage -- this alone should cut frame cost by an order of magnitude. Disable the field under prefers-reduced-motion.

### [major] Zero prefers-reduced-motion support anywhere -- 9 infinite keyframe animations plus two full-screen ambient canvas/SVG motion fields with no off switch
*bug -- Global stylesheet + ParticleField + SemanticTopology (all screens)*

**Evidence:** grep for prefers-reduced-motion / reducedMotion / matchMedia across demo/web/src and index.html returns nothing. All 9 @keyframes in demo/web/src/styles.css are used with 'infinite' (pulse:694, fc-pulse-travel:2427, fc-regen-march:2470, fc-halo-pulse:2649, fc-pulse-ring:2660, caret:2948, semantic-topology-edge-flow:4112, dreamCardPulse:5084, dreamSpin:5668), and neither rAF loop (ParticleField.tsx, SemanticTopology.tsx) checks a motion preference. Unlike a typical dashboard where reduced-motion is a checkbox item, this UI has large-area ambient motion -- a whole-background drifting particle field (view-cognition.png right pane) and a permanently jiggling 300-node graph -- which is exactly the WCAG 2.3.3 / 2.2.2 (pause-stop-hide) territory for vestibular-sensitive users. There is also no user-facing pause control for either field.

**Recommendation:** Add one global @media (prefers-reduced-motion: reduce) block that disables the infinite/ambient animations (keep sub-300ms state-change transitions), plus a shared useReducedMotion() hook consulted by ParticleField and SemanticTopology to render a static frame. Since the ambient fields run >5s, WCAG 2.2.2 also wants an explicit pause/hide control -- a small 'fx' toggle in the topbar would cover both fields and double as the perf escape hatch.

### [major] Ambient decoration animates while idle, diluting an otherwise excellent motion-equals-state grammar
*design -- Cognition flow canvas, Memory topology, motion language across screens*

**Evidence:** The state-gated motion in this UI is genuinely well designed: edge pulses only on .fc-edge-wrap.active (styles.css:2418-2424), regen dash-march only on .fire (2461-2470), halo/pulse-rings only on [data-status=running] (2574-2597), streaming caret only when isRunning (FlowChart.tsx:630), dream pulse/spinner only on running states (5081, 5664), orrery pulses gated on live/active/running classes (5935-6013, 6110, 6182-6199). But the two biggest motion surfaces ignore that grammar: the particle field sways at full rate when the turn pipeline is idle (ParticleField.tsx:174-183 -- view-cognition.png shows the field glowing over an idle 'waiting' chart), and the topology drifts and dash-marches forever regardless of activity (SemanticTopology.tsx:411 + :27-29). Result: on the two flagship screens, motion no longer means anything is happening -- the running-state pulses the design carefully gates have to compete with always-on decoration that is larger, brighter, and equally animated.

**Recommendation:** Adopt an explicit motion policy in the redesign: continuous motion is reserved for live activity (running phase, streaming tokens, in-flight dream), ambient surfaces are static or near-static at idle. Concretely, the particle field's sway should decay to stillness within a few seconds of idle, and topology drift should stop post-settle -- which makes the moment a turn starts visually unmistakable, and incidentally fixes the two CPU findings.

### [minor] Default landing view (Mission Control) idles at ~1.6x the static baseline from always-on orrery pulse + SVG drop-shadow filters
*bug -- Mission Control, Orrery*

**Evidence:** Measured: ?view=mission (the DEFAULT_ROUTE_ID, routes.ts:28) idles at 28.5% of one core vs 17.6% static control and 19.1% for Stream. The delta comes from infinite pulse animations combined with filter: drop-shadow on SVG elements that are 'live' whenever the SSE connection is up -- effectively always: .orr-core-live .orr-core-dot (styles.css:6003-6007), .orr-stream-active .orr-turn-pulse (6108-6111), .orr-mini-live .orr-mini-core (6178-6182), plus the three topbar/strip live-dots (styles.css:686-692; Topbar.tsx:78, InstrumentStrip.tsx:111, Stream/index.tsx:1272). Animating opacity on a drop-shadow-filtered SVG node forces repeated filter re-rasterization rather than a cheap composited fade.

**Recommendation:** Keep the live-pulse semantics (connection state is real state) but make it cheap: pulse a plain opacity/transform on a pre-blurred shape instead of animating elements that carry drop-shadow filters, and slow the pulse (e.g. 2.5-3s) on the always-true 'connected' indicators so the default screen reads calm. Fold these into the same reduced-motion block as the other findings.

### [major] Cold-start landing screen (Mission Control) is a dead instrument panel with no first action
*design -- Mission Control (default landing route), session sidebar*

**Evidence:** fresh-mission.png + crop-mission-right.png (/tmp/borg-ui-audit/fresh/): on a fresh data dir the default route (routes.ts DEFAULT_ROUTE_ID="mission") renders an idle orrery ("idle / no active turn"), a bottom strip of TURN idle / PHASE idle / DREAM idle, and six right-rail panels reading "no open reviews", "no active commitments", "no creator-directive conflicts", "no pending dream work", "no recent suppressed or observed outcomes" (screens/MissionControl/index.tsx:177,220,280,326,376). Every empty state is a terminal noun-phrase; none points at the operator's actual first actions (the COG composer's "send a turn", the OPERATOR CHAT button, mark creator). The lone good counter-example in the codebase is Cognition/LedgerView.tsx:220 "send a turn to build an evidence ledger".

**Recommendation:** On zero-turn state, replace the panel notices (or add a center-stage callout under the orrery) with directed copy and a link/button: e.g. "no turns yet -- open the workbench (⌘1) and send a turn, or start operator chat". The redesign already routes panel headers to screens; reuse that onNavigate plumbing so empty states become the navigation, not dead text.

### [major] Orrery governance arcs are fixed-sweep decoration that reads as live data -- rendered at full length labeled "cmt 0/0" / "dir 0/0" on zero data
*design -- Mission Control orrery*

**Evidence:** crop-orrery-zoom.png (/tmp/borg-ui-audit/fresh/): with zero commitments and zero directives, two prominent ~100° arcs still sweep the left side of the orrery, labeled "cmt 0/0" and "dir 0/0", alongside an equally prominent yellow arc. Code: components/orrery/Orrery.tsx:291 and :309 hardcode arcPath(246,210,310) and arcPath(228,216,304); only the color class is data-driven (lines 276-278). A first-run operator sees glowing governance constraints that do not exist; conversely, when populated, arc length encodes nothing.

**Recommendation:** Make arc sweep (or presence) data-driven: zero rows → no arc or a faint hairline placeholder; otherwise scale sweep or thickness to count/criticality. An instrument whose needle never moves should not look like a needle.

### [major] No help or orientation surface exists anywhere; palette and route chords are undiscoverable on first run
*design -- App shell, all screens*

**Evidence:** grep across demo/web/src for welcome/get started/onboarding/tour/shortcuts/help-overlay returns zero hits. The command palette opens only via meta/ctrl+K (hooks/use-palette-hotkey.ts:32) with no visible affordance in AppShell.tsx. The only place the #0-#9 route chords are documented is the ROUTE DIAGNOSTICS table buried on the admin screen (fresh-admin.png, right panel, CHORD column). A brand-new operator faces a 10-route cockpit of dense jargon ("review & repair -- operator queue and sanctioned repair lab") with no '?' overlay, no legend, no first-run hint.

**Recommendation:** Add a single lightweight orientation affordance: a '?' shortcut/legend overlay (routes, chords, ⌘K) plus a visible ⌘K hint in the top bar. One overlay component fixes discoverability for palette, chords, and screen vocabulary at once; no per-screen tour needed.

### [major] Memory atlas on zero data: eight all-zero band cards, then ~60% dead viewport, with no statement of how memory forms
*design -- Memory screen*

**Evidence:** fresh-memory.png (/tmp/borg-ui-audit/fresh/): band cards all read 0 with flat dashed sparklines, the two governance boxes show static boilerplate ("identity-bearing writes are guarded by provenance...", "review hint: see review for open queue rows"), and everything below y≈440 is empty black. Nothing tells a new operator that memory accrues automatically from turns -- the screen reads as broken rather than not-yet-started. (The dead lower viewport also exists when populated, view-memory.png, so this is layout, not data-thinness.)

**Recommendation:** Add a zero-items state to the atlas: one notice in the band-card region ("no memories yet -- memory forms automatically as turns are ingested; send a turn in the workbench") and let the band grid/governance section fill or center in the viewport instead of top-anchoring above a void. REDESIGN_PLAN.md:1624 already specifies per-band "no <band> memories yet" notices -- extend that to the atlas-level zero state.

### [minor] Empty-state copy blames filters when the store is empty ("no commitments in filter", "no questions match this filter")
*design -- Governance commitments tab, Governance directives/shared-state tab, Identity open-questions panel*

**Evidence:** fresh-governance.png right column shows "no commitments in filter" (screens/Governance/CommitmentsTab.tsx:622); fresh-governance-shared_state.png shows "no creator directives in filter" (screens/Governance/DirectivesTab.tsx:984); fresh-identity.png SELECTED QUESTION panel shows "no questions match this filter" (screens/Identity/index.tsx:500, rendered whenever question===null). On a fresh data dir these imply rows exist but are hidden by the current filter -- a wrong diagnosis a new operator will act on (toggling filters) instead of the right one (nothing exists yet).

**Recommendation:** Distinguish the two states: when the unfiltered count is zero, say "no commitments yet" (+ the create affordance: the '+ add' button already exists); only say "none match this filter" when total > 0. The counts are already fetched for the toolbar badges, so the branch is cheap.

### [minor] Stream and Review stack contradictory empty states instructing selection of objects that don't exist
*design -- Stream screen, Review screen*

**Evidence:** fresh-stream.png shows three simultaneous notices: "no entries in window", "end of loaded stream window", and right panel "select a stream entry" (screens/Stream/index.tsx:1277,1322). fresh-review.png shows the triptych "no open review rows" / "select a review row" / "select a review row to repair" across three full-height bordered columns (screens/Review/index.tsx:2136,1599,1698). Two of three panels per screen issue an instruction that is impossible to follow, and neither screen says what produces entries/reviews (turn ingestion; overseer/extraction audits).

**Recommendation:** Make the dependent panels' empty states conditional on the primary list being non-empty (empty list → secondary panels show nothing or a muted dash), and give the primary list's zero state one line of provenance: "stream entries appear as turns are ingested" / "review rows are filed by extraction and overseer audits".

### [minor] Raw backend-gap admissions shipped as operator-facing copy in Governance
*backend-gap -- Governance sessions & entities tab, Governance scope matrix tab*

**Evidence:** fresh-governance-sessions.png header line: "no generic entity create: the web client currently has no postEntity binding" (screens/Governance/SessionsEntitiesTab.tsx:252). fresh-governance-scope.png intro: "Dream impact and full entity inventory are omitted because the web client has no read endpoint for those datasets." (screens/Governance/ScopeMatrixTab.tsx:237). Honest, but it's developer changelog prose in permanent UI chrome -- on first run it reads as an error message.

**Recommendation:** Either close the gaps (POST /api/entities exists per REDESIGN_PLAN.md:1847 -- wire it; add the missing read endpoints) or demote the admissions to a tooltip/footnote styled as a capability note, not headline copy.

### [critical] Episode 'why' ships the full 4096-dim embedding and renders it as a ~60k-pixel wall of floats
*bug -- WhyDrawer (Memory episodes), Inspector Evidence tab (same renderer), /api/correction/:id/why*

**Evidence:** why-memory-episode-embedding.png: modal body is an endless 'embedding { "0": 0.0200..., "1": 0.0191..., ... }' float list, one per line, scrollbar thumb shrunk to a sliver. Measured via curl on the live instance: the why payload for ep_f8up5wg0807s19ib is 133,667 bytes, of which record.embedding is 123,663 bytes / 4096 entries -- serialized as a numeric-keyed OBJECT (Float32Array leaking through JSON.stringify), not even an array. Backend includes the raw repo row verbatim (src/correction/service.ts:630-635 'record: episode'); frontend renders any nested object as JSON.stringify pre (demo/web/src/components/JsonValueView.tsx:31-32). Because sections default-open (WhyDrawer.tsx:68), the real provenance (source_stream_ids, citation_chain) sits BELOW ~4096 lines of floats -- unreachable unless the operator discovers the 'record' summary collapses (why-memory-episode-citations.png was only capturable after manually collapsing it). Episodes are the most common why target, so the surface is effectively unusable by default.

**Recommendation:** Strip embedding/vector fields from the why response server-side (and fix Float32Array→array serialization for any path that does emit vectors). Frontend-side, add a guard in JsonValueView: arrays/objects beyond N entries render as a count summary ('4096 floats') with opt-in expansion. Reorder sections provenance-first (citations/sources before raw record).

### [major] WhyDrawer modal is a leftover duplicate of the Inspector Evidence tab, with divergent not-found handling
*design -- WhyDrawer mounts on Memory, Identity, Governance commitments; Inspector Evidence tab*

**Evidence:** Two parallel provenance surfaces ship today: the 'why' buttons (Memory/index.tsx:1833→1891, Identity/index.tsx:957-1126→1410, Governance/CommitmentsTab.tsx:1045→819) open a centered modal, while IdRef pivots open the Inspector whose EvidenceTab runs the identical getWhy + .why-drawer rendering (Inspector.tsx:227-270). They diverge on failure: Inspector maps 404 to a graceful Empty 'no provenance retained' (Inspector.tsx:234-236) and unsupported types to 'no evidence resolver for this object type'; WhyDrawer dumps the raw error string with no 404 handling (WhyDrawer.tsx:62). REDESIGN_PLAN.md lines 88, 418, 506 explicitly call for retiring the standalone modal call sites into the Evidence tab -- the migration was left half-done.

**Recommendation:** Finish the planned consolidation: make every 'why' button call the Inspector's openObject with the Evidence tab focused, delete the three WhyDrawer mounts, and keep one provenance renderer with one failure vocabulary.

### [major] The 'why' surface is a raw JSON dump, not an explanation: unresolved ent_* ids, epoch-ms timestamps, dead id text, and full-record event duplication
*design -- WhyDrawer / WhyEvidence rendering on all three screens*

**Evidence:** why-governance-commitment.png: drawer shows 'restricted_audience: ent_r6cvl8caz41cdpx0' while the detail pane visible BEHIND it resolves the same field to 'botarena_thread:ebe83f0a-...'; 'created_at 1780257758700' raw epoch vs 'May 31, 2026' behind; seven consecutive null rows (expires_at..canonicalized_by_artifact_entry_id) of pure noise. why-identity-trait.png: last_reinforced/established_at as epoch ms, strength at 17 decimal places, evidence_episode_ids as plain JSON text while the trait card behind renders the SAME ids as clickable chips; provenance block repeats the same 3 episode ids twice. why-governance-commitment-events.png: identity_events embeds the entire old_value commitment record per event, so the full directive paragraph appears 3+ times with no diff -- and wrapped JSON lines lose indentation, colliding into the key column. why-memory-episode-citations.png: citation_chain entries are raw stream rows with literal \n escapes inside message content; source_stream_ids are dead text although the same ids render as IdRef pivots elsewhere (CommitmentsTab.tsx:1037). Root cause: WhyDrawer.tsx:67-72 renders Object.entries of the payload through JsonValueView with zero domain awareness.

**Recommendation:** Render typed provenance: IdRef chips for every id (the registry already resolves prefixes), formatted timestamps, resolved entity labels, omitted-null rows, and an event timeline reusing the OpenQuestionEventsSection pattern with old/new diffs. Keep the raw JSON behind a collapsed 'raw' details as the escape hatch, not as the primary view.

### [major] ErrorState's red styling is dead CSS -- error, loading, and empty notices are visually identical app-wide
*bug -- WhyDrawer states; every ErrorState/Loading usage app-wide*

**Evidence:** why-error-state.png ('Failed to fetch') and why-loading-state.png ('loading provenance') are indistinguishable: same muted-gray 12px line centered in an otherwise empty 620px modal. ErrorState renders className="notice bad" (ErrorState.tsx:4), but .notice { color: var(--text-mute) } at styles.css:6301 appears LATER in the stylesheet than .bad { color: var(--bad) } at styles.css:145 with equal specificity, so the red is overridden everywhere .notice.bad is used. Also confirmed live: the drawer's network failure shows the bare browser string 'Failed to fetch' with no retry, no endpoint context; curl-proven 404 path would show 'Unknown commitment id: cmt_...' in the same gray shell (the stale-row repro: list row + why button survive until the next refetch after a server-side delete/reset).

**Recommendation:** Change the rule to .notice.bad (or color inside ErrorState), add an error glyph and a retry action to drawer/inspector error states, and prefer the API's message (which is good: 'Unknown commitment id: ...') over bare fetch errors.

### [minor] Malformed why ids return 500 'Internal Server Error' instead of 400/404, defeating the UI's graceful not-found path
*backend-gap -- /api/correction/:id/why; Inspector EvidenceTab 404 handling*

**Evidence:** curl on live server: GET /api/correction/ep_nonexistent123/why and /api/correction/ep_01ARZ3NDEKTSV4RRFFQ69G5FAV/why → {"error":{"status":500,"message":"Internal Server Error"}}. Cause: id helpers' parse throws a plain TypeError on pattern mismatch (src/util/ids.ts:72-75) which mapBorgErrorToHttp re-throws unmapped (demo/server/src/app.ts:896). Well-formed absent ids correctly 404 ('Unknown episode id: ep_aaaaaaaaaaaaaaaa'); unsupported prefixes correctly 400. The Inspector's 404→'no provenance retained' branch (Inspector.tsx:234) never fires for these, so the UI shows the blank 'Internal Server Error'.

**Recommendation:** Catch id-parse TypeErrors in parseTarget (or the route) and map to 400 with the parse message, so both provenance surfaces degrade with a meaningful string.

### [minor] Why modal has no visible close affordance
*design -- Modal.tsx shell (WhyDrawer and every other modal without a footer)*

**Evidence:** All why-*.png captures: the modal renders only a title bar and body -- no [x], no close button (Modal.tsx:34-46 renders title/body/optional-footer; WhyDrawer passes no footer). Exits are Esc (Modal.tsx:17-27) and backdrop mousedown (Modal.tsx:35) only, both undiscoverable affordances for a mouse-driven operator console.

**Recommendation:** Add a close control (and an 'esc' hint) to the Modal title bar so every modal inherits it.

### [minor] Modal title uppercases internal ids, displaying an id that doesn't exist
*design -- WhyDrawer title; .modal-title chrome generally*

**Evidence:** why-memory-episode-top.png shows 'WHY EP_F8UP5WG0807S19IB' for actual id ep_f8up5wg0807s19ib; same on trait (WHY TRT_...) and commitment (WHY CMT_...) captures. WhyDrawer.tsx:59 interpolates the raw id into the title and styles.css:5465 applies text-transform: uppercase to .modal-title. The id alphabet is strictly lowercase (src/util/ids.ts:3-5), so the displayed string is not a valid id -- it can't be visually matched against the id chips elsewhere on screen, which preserve case.

**Recommendation:** Render ids inside title bars in a non-transformed span or id chip; keep the uppercase transform for the literal label text only.

### [polish] Long snake_case keys break mid-word in the fixed 150px key column
*design -- WhyDrawer/.why-row key column*

**Evidence:** why-governance-commitment.png renders 'closure_pressure_releva / nce'; the next scroll shows 'canonicalized_by_artifa / ct_entry_id'. Cause: .why-row grid-template-columns: 150px (styles.css:6275-6276) with word-break: break-word on .why-key (styles.css:6284), which breaks at arbitrary characters because underscores aren't break opportunities.

**Recommendation:** Widen the key column (or use minmax(150px, max-content)) and insert break opportunities at underscores (render with <wbr> after '_') instead of arbitrary mid-word breaks.

### [polish] Empty provenance state is rendered with the Loading component, and its copy diverges from the Inspector's
*bug -- WhyDrawer empty state*

**Evidence:** WhyDrawer.tsx:64-65 renders the no-fields case as <Loading>no provenance fields</Loading> -- semantically a loading notice, while the Inspector's identical case uses <Empty>no provenance fields</Empty> and 404s become 'no provenance retained' (Inspector.tsx:234-245). Same data, three different state vocabularies.

**Recommendation:** Use Empty in the drawer (until the surfaces merge per the consolidation finding) and align the empty/not-found copy with the Inspector.

### [polish] Section structure wastes the drawer: a full collapsible section for the single word 'episode', provenance ordered last
*design -- WhyDrawer section layout*

**Evidence:** why-memory-episode-top.png: the first <details> section 'target_type' contains exactly one scalar ('episode' / 'trait' / 'commitment' in each capture) yet gets full section chrome; meanwhile the sections that actually answer 'why' (source_stream_ids, citation_chain) come last in payload key order, beneath the giant record dump (WhyDrawer.tsx:56 renders Object.entries order verbatim).

**Recommendation:** Promote target_type into the title bar as a type chip, and order sections provenance-first (citations, sources, events) with the raw record last and collapsed by default.

### [critical] Disclosure-class labels render NOWHERE in the console; DisclosureLabel is a fully built, tested, orphaned component
*design -- Memory band rows, episode detail pane, Review queue/candidates, Inspector summary, Cognition ledger -- console-wide*

**Evidence:** grep confirms zero non-test imports of /home/luth/Programming/borg/demo/web/src/components/DisclosureLabel.tsx (only its own file + DisclosureLabel.test.tsx). Visual confirmation: live-memory-band.png episode cards (center column) show only audience pill + time/participants/sig/conf/src chips -- no privacy/disclosure/trust chip (code: Memory/index.tsx:1021-1034); the episode detail pane (right) lists time range/audience/participants/location/significance/confidence only (Memory/index.tsx:2030-2056). live-review.png candidate cards show kind/active/confidence only; evidence episodes appear as bare ep_ id chips. live-inspector-review.png summary tab shows id/kind/refs/reason/created/resolved -- Inspector's GenericSummary (components/Inspector/Inspector.tsx:161-181) has no disclosure row. REDESIGN_PLAN.md line 499 promises 'every recalled row renders its DisclosureLabels... combined-label unknown shown as a neutral chip' and line 121 lists DisclosureLabel as a P0.4 deliverable -- the component shipped (with the correct fail-closed 6-to-4 collapse, DisclosureLabel.tsx:14-32) but was never wired to a single row, so the fail-closed 'unknown' state the architecture mandates is invisible to the operator everywhere.

**Recommendation:** Wire DisclosureLabel into the shared row/chip primitives rather than per-screen: episode cards and detail pane in Memory, review candidate cards, Inspector GenericSummary (a labels row above the key/value props), and ledger entry heads. Treat 'an operator can see the disclosure class, origin audience, and fail-closed unknown on any recalled row' as the acceptance test -- it is the architecture's single most load-bearing UI promise (privacy by labels at render, not by hiding).

### [major] Episodes API drops origin_audience_entity_ids and shared, and computes no disclosure class -- even a wired DisclosureLabel would have nothing to label
*backend-gap -- Memory band (episodic) rows + episode detail; demo server episode serialization*

**Evidence:** Core Episode carries audience_entity_id, origin_audience_entity_ids, shared (src/memory/episodic/types.ts:66-68), but mapEpisode in demo/server/src/app.ts:1475-1499 serializes only a single collapsed audience label and omits origin_audience_entity_ids, shared, and any MemoryDisclosureClass. Web EpisodeMemoryItem (demo/web/src/api/types.ts:224-244) accordingly has only audience: string|null. Result visible in live-memory-band.png: the only provenance on an episode card is the audience thread pill -- origin audience (who witnessed it, the common-ground axis per CLAUDE.md rule 6) is unrepresentable in the UI. mapSemanticMemoryNode (app.ts:1501-1517) similarly exposes no provenance/disclosure fields for review candidates.

**Recommendation:** Extend the episode (and semantic-node where applicable) serialization with origin_audience_entity_ids (label-resolved), shared, and the computed disclosure class from src/memory/common/disclosure-label.ts, then render them via DisclosureLabel. This is the REDESIGN_PLAN line 192 'type-surfacing blocker' generalized to the episodes read path.

### [major] Evidence ledger entries carry full disclosure_label metadata to the browser but LedgerView never renders it -- and there is no raw-JSON fallback on entries, so it is invisible
*design -- Cognition ledger (LedgerView)*

**Evidence:** The harness attaches disclosure_label, disclosure_note, and current_audience_entity_id to ledger entry state_metadata (src/cognition/evidence-ledger/entry-metadata.ts:126-133), and the demo server returns the ledger whole (demo/server/src/app.ts:2521-2529), so the data reaches the client. But LedgerView entry rows render only state, taint, and trust_rank (demo/web/src/screens/Cognition/LedgerView.tsx:279-285) and use state_metadata solely for object-id extraction (LedgerView.tsx:136-142); entries have no JSON expander, so the disclosure label on the exact rows that feed the model's prompt cannot be seen at all. EvidenceLedgerEntry.state_metadata is still Record<string, unknown> (demo/web/src/api/types.ts:947) -- the type-surfacing task REDESIGN_PLAN.md flags as a 'blocker' at lines 109 and 192 was never executed.

**Recommendation:** Lift state_metadata.disclosure_label into a typed field on EvidenceLedgerEntry (per the plan's own blocker note), render it with DisclosureLabel next to the existing state/taint/trust tags in the entry head, and show disclosure_note on expand. The ledger is where 'what the model was told it may disclose' is most auditable; trust and taint render there today, disclosure does not.

### [minor] Disclosure/privacy policy values render through four unrelated ad-hoc grammars with no shared primitive or color semantics
*design -- Governance DirectivesTab, Governance SessionsEntitiesTab, Governance ScopeMatrixTab, Review (relational scope comparison)*

**Evidence:** DirectivesTab renders content_scope/mention_policy as bare table text (demo/web/src/screens/Governance/DirectivesTab.tsx:926-928) and again as plain props rows in the detail pane (DirectivesTab.tsx:1169-1174); SessionsEntitiesTab renders privacy_level as a bare <td> (SessionsEntitiesTab.tsx:173,215); Review's scope-equivalence comparison renders disclosure_allowed/excluded as plain 'N entities' text (demo/web/src/screens/Review/index.tsx:770-823,1265-1266); ScopeMatrixTab renders policy as table columns plus a prose notice (ScopeMatrixTab.tsx:233-265). None share a component or the public/private/operator/unknown color taxonomy DisclosureLabel defines (DisclosureLabel.tsx:34-45), so the same conceptual family (what may be said to whom) looks different on every screen -- a concrete instance of the design-system-fragmentation theme.

**Recommendation:** During the redesign, route all disclosure-policy and privacy-level value rendering through the DisclosureLabel/Tag taxonomy (or a small PolicyValue variant of it) so scope values get one consistent visual grammar and operators can pattern-match private/operator-restricted values at a glance across Governance and Review.

### [minor] Raw internal entity id rendered as a participant chip on episode cards, mixed with resolved names
*bug -- Memory band episodic cards*

**Evidence:** live-memory-band.png, first two episode cards (center column): chip row reads 'ent_w4tgay3o9h06ogaa | Luigi | Fishy | Lunaria' -- one participant renders as a raw internal entity id beside three resolved names. Code: the card maps episode.participants straight to Tags (Memory/index.tsx:1028-1030) and mapEpisode passes item.participants through unresolved, while the audience field on the same row does get entityLabel() treatment (demo/server/src/app.ts:1483,1487). Provenance labeling on memory rows thus degrades to substrate-grade ids inconsistently (this is distinct from the already-filed AUDIENCE-pill-UUID finding).

**Recommendation:** Resolve participant entity ids to labels server-side with the same entityLabel() path used for audience (falling back to shortId + IdRef pivot when unresolvable), so provenance chips never expose raw internal ids next to human names.

### [polish] Dream's RawJsonDisclosure name collides with the architecture's 'disclosure' concept and pollutes discovery
*design -- Dream screen (component naming)*

**Evidence:** Dream/index.tsx:1551 defines RawJsonDisclosure -- a details/summary JSON expander -- used at lines 1293/1519/1546. In a codebase where 'disclosure' names the load-bearing privacy mechanism, this is the only 'disclosure' hit in the Dream screen; it caused this very audit's completeness pass to misidentify Dream as a disclosure-label render site. Grep/IDE discovery for the real concept now returns widget-naming noise.

**Recommendation:** Rename to RawJsonDetails (or JsonExpander) during the redesign, reserving the 'disclosure' lexeme in demo/web for the privacy/label concept.
