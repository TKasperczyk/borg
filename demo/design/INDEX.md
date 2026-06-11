# Borg console design handoff

Source: Claude Design (claude.ai/design) handoff bundle, fetched 2026-06-11 from
`https://api.anthropic.com/v1/design/h/Y_zn8Wtb6U2P9QLjawDeHA` (`Borg-handoff.tar.gz`).
A design model explored this repo (backend-only main, HEAD 85a7721) and produced
six linked HTML/CSS/JS page prototypes for the operator console. These files are
the **authoritative visual + interaction spec** for `demo/web`; the prototypes are
mocks (simulated data, no network), to be recreated for real against the
`demo/server` REST + WS contract.

## Files

- `Overview.dc.html` -- design-exploration cover page (design system + honesty rules); not an app screen
- `Chat.dc.html` -- 01/CHAT flagship: sessions rail, operator thread (silence / observed / suppressed as first-class artifacts), cognition panel with canvas mind core
- `Mind.dc.html` -- 02/MIND: identity, creator-directives vs commitments ledgers, 8 memory bands, belief graph; ATLAS + INSPECTOR variants
- `Reviews.dc.html` -- 03/REVIEWS: per-kind resolution flows; LEDGER / SPLIT / TABLE variants
- `Dream.dc.html` -- 04/DREAM: 13 processes, plan->apply, live run feed, audit + revert, reports
- `Settings.dc.html` -- 05/SETTINGS: prompt-block editor + assembled preview, runtime, scheduler, entities, token-armed reset
- `Activity.dc.html` -- 06/ACTIVITY: day-grouped journal of turns/wakes/dreams, wake sources, train-of-thought lane
- `support.js` -- the design tool's prototype runtime (needed only to open the .dc.html files in a browser)
- `CONTRACT-NOTES.md` -- the design model's notes on the exact REST + WS shapes (verify against `demo/server/src` -- the code is the source of truth)
- `chats/chat1.md` -- the design conversation; where the intent lives
- `shots/chat-idle.png` -- look reference

## Decisions made at implementation time (2026-06-11, by the operator)

- Cognition animation: **variant C SWARM only** -- drop A CORE / B RINGS and the switcher
- Reviews layout: **variant 2 SPLIT (master-detail) only**
- Mind layout: **ATLAS as the page, INSPECTOR as band drill-in** (composed, as the mock's band-card clicks already do)
- Mock-only affordances are NOT implemented: SIMULATE buttons, click-to-cycle mood, raw endpoint-string footers, the Activity "wiring notes" panel
- The phase ring/grid binds to the real demo-server turn phases (from WS `turn:phase:*` frames), not the mock's abbreviated 9
