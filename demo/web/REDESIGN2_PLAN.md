# Redesign-2 Sprint Plan (2026-06-10)

Executes the findings in `UI_AUDIT.md` (271+36 findings, 18 themes). Branch: `frontend-redesign-2`.
Each sprint is one full codex workflow: explore -> implement -> full verification -> fresh-session adversarial review -> fix rounds -> path-scoped commit.

## Standing constraints (every sprint)

- Cardinal rules apply: recall is global, disclosure is labels-at-render -- NO audience gating of any data fetch that feeds cognition-like surfaces; labels are presentation only. No language/keyword keying; key on ids/prefixes/enums/structure.
- Codex must NOT: start dev servers, run git commit/push, touch files outside the sprint's scope list.
- Verification gates (run by orchestrator, not codex): root `pnpm typecheck`; `pnpm --filter @borg/demo-web test`; root `pnpm test` + `pnpm heuristics:guard` when anything under `src/` or `demo/server/` changed; `pnpm --filter @borg/demo-web build`; prettier check on touched files; git-scope check (diff only inside sprint scope).
- The live console (vite dev :5173) and Sol (:7740) run off this working tree; web changes hot-reload. Avoid leaving the tree broken between commits; restart `borg-demo-sol.service` only after a server-touching sprint's commit.
- Truth rule for all sprints: nothing may look like telemetry unless it is data-bound.

## Sprint sequence

### S1 -- Foundation: legibility floor + component kit (styles.css + shared components)
Themes: "Type scale and contrast below the legibility floor", "ID/value text shattering", "Chip soup", "Destructive-action grammar", "State-rendering fragmentation", "Color semantics collapsed" (token layer), keyboard/focus bugs.
- Type scale: raise functional-text floor to 12px; defined size scale; contrast >= 4.5:1 for functional text (new/adjusted tokens).
- Written color contract as token comments + usage: red = failure/destructive only; severity encodes state not volume; purple = interior/dream only; green rationed (live/active/primary); amber = reversible live-write.
- KV grid primitive fix: container-aware .props (stack key-over-value below min width), never overflow-wrap:anywhere on ids/labels/buttons; id values render as shortId+copy chips (shared primitive).
- Chip taxonomy: visually distinct interactive chip vs static tag; one Button system incl. `.btn.danger` (wire to reset/forget/invalidate/revoke/supersede).
- One state idiom: fix dead `.notice.bad` CSS; ErrorState visually distinct (glyph + red + retry slot); Empty vs Loading vs Error distinct; WhyDrawer empty uses Empty.
- Focus/keyboard floor: restore :focus-visible on all interactive orrery classes; accent-color globally; palette scrollIntoView on arrow nav; interactive rows/pills get role/tabIndex/key handlers (pattern from Review/index.tsx:1531).
- Decouple Prompts from Memory's .band-detail class.

### S2 -- Truth: kill fabricated instrumentation
Theme: "Fabricated or static data styled as live instrumentation".
- StatusBar: delete or data-bind branch/model/embedding cluster (server can expose real model/embedding config; no fake green dot).
- UTC clock actually UTC (or relabel local).
- TAILING badge bound to real WS/connection state (expose from use-stream-window).
- Implement ⌘0-9 route chords for real (use-palette-hotkey pattern), keeping the three advertisement surfaces honest.
- Remove server-side `sparkFrom` fabrication; band cards get real created_at-bucketed series or no chart.
- Orrery governance arcs data-driven (zero rows -> no arc/hairline; sweep scales with count/criticality).
- AttachmentsCard: no eternal "syncing" on null count; restore dropped "needs backend" header action.
- Dream "schedule lane" relabeled run history; SCHEDULED column only for genuinely future entries (synthesize next-run from known intervals if cheap).

### S3 -- Identity & time primitives
Themes: "Raw machine IDs as headline identity", "Dateless and raw time rendering".
- One identity primitive: human label first, middle-truncated copyable mono id chip second, full id in Inspector. Apply: transcript speakers (use sender_entity_id), stream session column, group headers, breadcrumb session label, AUDIENCE pill (label not 50-char id; never invent "alice" fallback), Memory participants (server resolves via entityLabel path), palette result titles.
- Date-aware time: `formatTime` family renders day boundary (today/yesterday/date); fix backwards-looking multi-day ranges; epoch-ms never rendered raw (Why/props formatters).
- Fix displayValue regex truncating semantic enums (scheduled_reflection -> "schedule…tion").
- Modal titles stop uppercasing ids.

### S4 -- Error resilience + empty states
Themes: "State-rendering fragmentation" (behavior layer), cold-start gap findings.
- Transport errors auto-retry with backoff + stale-while-revalidate; Mission Control cards show ErrorState (with retry) not Empty; recovered data replaces error without reload (fixes the Bad Gateway wedge).
- Empty-state honesty: "no X yet" vs "none match filter" branched on unfiltered count; dependent panes go quiet when primary list empty (Stream/Review stacked contradictions); zero-data screens state how data forms + first action (Mission idle, Memory atlas).
- Help/orientation: "?" shortcut legend overlay (routes, chords, ⌘K) + visible ⌘K hint in top bar.
- Demote backend-gap admissions in Governance to capability-note styling.

### S5 -- Round-trips & dead ends
Theme: "Dead ends: the data exists but the UI cannot reach it".
- Wire `loadOlder` UI (button + infinite-scroll guard) in Stream; transcript scroll-back beyond 16 entries in Cognition.
- RAW tab fetches ledger.
- Palette id round-trip: resolve full ids behind truncated display, existence-check before offering "Open X", copyable full ids; debounce memory fan-out + boundedAllSettled (partial failure tolerant).
- Stream grouping: turn-less entries interleave chronologically (no global mega-group).
- Stream filter changes preserve selection/collapsed state where still visible; no silent no-ops.
- Inspector tab-strip sliver fix.

### S6 -- Layout recomposition (uses S1 primitives)
Themes: "Inverted space budgets and void layouts", "Unbounded data without aggregation", responsive gap findings, "Redundant rendering".
- Memory: atlas overview fills viewport (real plots/expanded bands), band-switcher sliver fixed; Admin + Identity recomposed (no 55-60% voids).
- Review: comparison grid container-aware (no 3-up forced at narrow), resolution panel stays in viewport.
- Governance: 320px rail -> flexible min/max; commitments table column priority at narrow widths (indicated scroll).
- Responsive floor: shell scrolls below 900px; key screens usable at 1366; cognition lane proportions favor content.
- One home per metric: kill duplicated COUNTS pill / bottom-bar mood+counts; rail badges = actionable backlogs only; status bar = runtime facts.
- Dream: cockpit above the fold; schedule table bounded with expand.

### S7 -- Motion & CPU
Gap findings: force-sim 326%/core, ParticleField ~90%/core, no reduced-motion.
- Semantic topology force-sim settles and stops (alpha decay), dash-march gated.
- ParticleField only during live turn activity; idle = static.
- Pulse animations cheap (no animated drop-shadow filters) and slower on always-true indicators.
- `prefers-reduced-motion` honored globally (single media block disabling ambient motion).

### S8 -- Disclosure labels everywhere (architectural)
Gap findings: orphaned DisclosureLabel; episodes API drops origin/shared; ledger label dropped.
Opus 5.0: in-scope -- the harness computes disclosure labels but the console never surfaces them; pure presentation/serialization gap.
- demo/server: episode serialization adds origin_audience_entity_ids (label-resolved), shared, computed disclosure class (src/memory/common/disclosure-label.ts); semantic-node provenance where applicable; participants resolved (S3 dependency done).
- demo/web: DisclosureLabel wired into Memory rows + detail, Review candidates, Inspector GenericSummary, LedgerView entry heads (lift state_metadata.disclosure_label into typed field per plan blocker note); PolicyValue variant unifies scope/privacy value rendering (Directives/Sessions/ScopeMatrix/Review).
- Labels are render-only: no fetch may become audience-conditional. heuristics:guard must pass.

### S9 -- Why/provenance surface
Gap findings: 4096-dim embedding wall, WhyDrawer duplicate, raw JSON dump, 500s on malformed ids.
Opus 5.0: in-scope -- serializer ships Float32Array embeddings into a UI payload; structural presentation bug.
- Server: strip embedding/vector fields from why payloads (src/correction/service.ts); fix Float32Array->array anywhere vectors legitimately serialize; id-parse TypeErrors -> 400.
- Web: consolidate WhyDrawer mounts into Inspector Evidence tab (delete modal mounts per REDESIGN_PLAN 88/418/506); typed provenance rendering (IdRef chips, formatted times, provenance-first order, raw collapsed, JsonValueView guard summarizing >N collections); Modal close affordance.
- Rename Dream's RawJsonDisclosure -> RawJsonDetails.

### S10 -- Heroes & cockpit polish
Themes: "Hero visualizations decorative/illegible", Prompts bugs, leftover copy.
- Orrery earns its ink: collision-managed labels (leader lines/offsets), proportional encodings (count -> ring weight/badge), per-band + per-satellite click targets routing to real destinations, calm idle (no glow > data), legible at 1366.
- FlowChart: resting contrast >= 3:1, idle = compact strip (no particles -- done S7), nodes click -> corresponding X-ray tab during replay.
- Prompts: dead section chips fixed (real anchors, no text.indexOf), three contradictory block counts reconciled (PromptKey union refreshed), diff header truth ("live draft" vs saved), dirty-draft badge + navigation guard.
- Jargon pass on operator-facing copy (cmt/dir/r·q·c expansions via title/legend, not removal).

## Done criteria

All 5 critical themes neutralized; the 13 critical top-issues each verifiably fixed; gates green every sprint; visual spot-check screenshots after S6 and S10.
