# Workflow

How Tom and Claude actually run sprints on borg. Read this after CLAUDE.md / AGENTS.md when you (Claude) come back from a context compaction.

CLAUDE.md / AGENTS.md tell you what NOT to do in the codebase (scope, guardrails, taste). This file tells you what TO do in the dev loop.

---

## LIVE PRODUCTION SYSTEM -- real memory now exists (as of 2026-05-31)

**Borg now holds real, non-wipeable memory.** "Sol" runs live in the BotArena arena on the demo data dir (`demo/server/.borg-data/demo`), forming continuous memory from real conversations. That memory is not reproducible from a sim re-run. **The "no production users" regime is OVER** -- we crossed into live data the moment Sol went into the arena.

The rules that now apply, without exception:

- **NEVER reset.** No `.borg-data` wipe, no `/api/admin/reset`, no deleting or recreating the live DB. Resetting destroys Sol's memory permanently. Restarting the *process* is fine (it preserves the DB); a *data* reset is not. If a fix "needs a reset to test," it needs a different fix.
- **Baselines are FROZEN.** Never edit an applied baseline/migration in place. It's already recorded as applied, so an in-place edit never runs and the live schema silently drifts from code.
- **Every schema change is a NEW forward migration.** Add a new migration entry; never mutate an existing one. Write it to carry existing live rows across -- no destructive drops of populated columns, no "wipe and re-run" shortcuts.
- **On-disk formats and persisted shapes are now contracts.** Changing a stored shape needs a forward migration that preserves existing data, not a re-create.

If you are ever unsure which regime applies: it's this one. There is no path back to the wipe-and-reset regime without explicit instruction from Tom that the live memory is expendable.

### History (no longer in force)

Earlier in development borg had no real data, and the rule was "squash to one baseline per module, edit it in place, then reset." That ended when Sol went live. **Do not follow that pattern anymore.** It's noted here only so older commit messages and code comments that reference "edit the baseline + reset" still make sense -- they describe the dead regime, not the current one.

## Project goal (the only one that matters)

A cognitive memory architecture for an LLM that is:

1. **Generic** -- works across group chats, single-user dev, relationship stuff, and other similar use cases. Not overfit to one scenario. (True multi-audience genericity here depends on the pending recall inversion -- today cross-audience recall is still gated in current code, so Sol can be blind across audiences; see the Cardinal Memory Rule in CLAUDE.md.)
2. **Clean code** -- no overengineering, no duplication, no dead paths, well organized.
3. **Simple flow** -- minimal paths. Every additional check or branch must be worth the complexity it adds. If you can't defend it, delete it.

Behavioral correctness across the sim suite looked good as of v83.1, but that is no longer the headline. A foundational memory-architecture inversion is pending and **not yet implemented**: Sol's internal recall is still audience/session-gated in current code, and the target doctrine is the opposite -- Sol remembers broadly; the harness labels disclosure constraints; Sol decides what to say. Until that inversion lands, do NOT treat the project as nearly done or as merely needing polish. The audience firewall in cognition is being inverted, not preserved -- treat firewall-as-recall-gate machinery as DEPRECATED / being-inverted, not the endorsed design. The still-live mechanisms (`isEpisodeAccessVisible`, `deriveEpisodeAccess` dropping multi-audience episodes, transitive semantic-graph source-visibility pruning, `ViewerCapability` self_continuity/unrestricted bypasses, among others) are CURRENTLY LIVE and must be inverted via data-preserving forward migrations, never a reset. See the Cardinal Memory Rule in CLAUDE.md and the inversion roadmap in BOUNDARIES.md.

**Stop at diminishing returns.** These signs apply to incremental quality work, NOT to the pending memory-architecture inversion. While the audience-firewall-in-cognition inversion (see the Cardinal Memory Rule in CLAUDE.md) is unfinished, the project is **not** at diminishing returns regardless of how minor individual sprints feel -- the core structural correctness gap is still open. Once recall is ungated for cognition and disclosure is a label/judgment layer applied after recall, then watch for these signs that you've hit diminishing returns:

- Each sprint produces only minor refinements.
- New mechanisms gate on questionable hypotheticals or rare edge cases.
- GPT review surfaces "nice to have" not "broken".
- You're proposing observability layers on observability layers.
- Coverage pressure exceeds correctness pressure.

When you notice these, raise it with Tom and stop the cycle.

---

## The standard sprint cycle

Tom runs this with a GPT Pro Extended reviewer (ChatGPT in Chromium) as the architectural second opinion. Claude drives codex, sims, commits, and the submission.

```
GPT review arrives in chat
  ↓
1.  Cross-verify reviewer claims with codex (don't trust blindly)
  ↓
2.  Triage by severity: P0 / P1 / P2. Pause at P3 for Tom.
  ↓
3.  For each priority, separate committable sprint:
      a. codex exploration (find existing utilities, peer files)
      b. codex implementation (require search-before-create)
      c. codex review of the diff
      d. fix CRITICAL / IMPORTANT only -- filter MINOR / SUGGESTION
      e. run tests + tsc + lint
      f. commit (single sprint = single commit; co-author Claude)
  ↓
4.  Run validation sims in parallel:
      family-aging-parent + shared-state-compaction
  ↓
5.  Draft `~/borg-v[N]-review-questions.md`:
      - cross-verification table (CONFIRMED / PARTIAL / REFUTED)
      - commit summaries
      - sim headlines comparing v[N-1] → v[N]
      - 5-10 questions for the reviewer
      - production-policing boundary status
  ↓
6.  Zip the project, submit, poll, ingest response.
  ↓
Loop.
```

Stop and discuss with Tom when:
- The reviewer claims something you can't reproduce in codex cross-verification.
- A proposed sprint fails the Opus 5.0 test (see CLAUDE.md).
- You're tempted to add an in-flight LLM judge of semantic output (see CLAUDE.md production-policing section).
- The sprint plan crosses P3.

---

## Pushback principles

GPT Pro Extended is a strong reviewer but not infallible. Filter every recommendation:

- **Opus 5.0 test** -- mandatory before any harness work. See CLAUDE.md.
- **Production-policing boundary** -- never add in-flight LLM/regex judges of non-critical semantic output. See CLAUDE.md.
- **Code hygiene rules** -- grep before adding a helper; extract repeated values; mimic existing patterns; read a peer file first. See user CLAUDE.md.
- **Defend sound positions** -- if Tom or the reviewer push back on a recommendation that was actually right, explain the trade-off rather than folding.
- **Honest negative findings** -- if a sim shows nothing changed, say so. Don't dress it up.

When you decline a recommendation, say why and offer the counter.

---

## ChatGPT submission mechanics

(As of 2026-05-24. Chromium on luth's headless sway session at :5901. May drift if ChatGPT redesigns.)

**Start a fresh thread each cycle.** ChatGPT threads get unwieldy after many turns (the "Project Analysis and Review" thread was used from ~v70 through v83 and got very long). Click the "new chat" pencil icon at top-left of the sidebar, then **verify the model picker shows `Pro · Extended`** (bottom-right of the composer). The dropdown lists `Latest · 5.5`, `Instant`, `Thinking · Heavy`, `Pro · Extended`, `Configure...`. **`Heavy` is NOT Pro** -- it's GPT-5.5 with heavy thinking effort. The Pro Extended model is its own entry below. Each review cycle is self-contained -- the zip + questions doc give GPT everything it needs.

### 1. Zip the project

The canonical recipe builds the file list from `git ls-files` plus explicit additions, then zips from stdin:

```bash
cd /home/luth/Programming && rm -f ~/borgvN.zip && \
  { git -C borg ls-files | sed 's|^|borg/|'; \
    find borg/.git borg/.design-dump borg/simulator-runs -type f 2>/dev/null; \
  } | sort -u | zip ~/borgvN.zip -@
```

Why this shape: the simple `zip -r borg/ -x '*/node_modules/*'` recipe blows up on pnpm's symlink-heavy `node_modules/.pnpm/` layout once the demo workspace (with multiple nested `node_modules/`) is present -- zip consumes >9 GB of memory and effectively hangs. `git ls-files` excludes everything `.gitignore` already excludes (node_modules, dist, .borg-data, etc.) with no fnmatch ambiguity, and the `find` line explicitly adds back the three useful gitignored trees: `.git` for commit history, `.design-dump` for design references (if present), `simulator-runs` for sim artifacts.

**INCLUDE** `simulator-runs/`, `.git/`, sim artifacts. GPT runs a sandbox and uses git history + sim outputs. Do not over-exclude. Tom corrected this once already; don't repeat it.

Size will land in the 150-200M range when `simulator-runs/` is populated. Demo-only / code-review zips without sim artifacts land closer to 10-15M.

### 2. Attach via the sway MCP

- Focus chromium: `swaymsg [app_id=chromium] focus`
- Click "+" attach button at bottom-left of the input. Coords drift, so always take a screenshot first and verify with `mark_x`/`mark_y` before clicking.
- Menu opens → click "Add photos & files"
- File picker opens with `~` as cwd → `~/borgvN.zip` is usually at the top (most recent)
- Click "Open"
- Wait briefly for the upload (the chip becomes solid, no spinner)

### 3. Paste the questions

ChatGPT will auto-attach any large paste as a separate document. We want it as the message text. Two options:

- Paste, then click "Show in text field" on the auto-attachment chip. Easiest.
- Or load to clipboard from bash (`wl-copy < ~/borg-vN-review-questions.md`) then ctrl+v in the focused composer.

### 4. Submit

Send button is the up-arrow at bottom-right. Click it. Composer changes to "Follow up" + stop button appears. Scroll the chat down to confirm the user message is visible and "Pro thinking" indicator is up.

### 5. Poll

GPT Pro Extended usually finishes in 20-40 min, sometimes 2-3 hours. Use a backgrounded sleep + bash:

```bash
sleep 1500; date  # 25 min, run_in_background=true
```

The harness notifies you when the sleep ends. Take a screenshot, check for the stop button (still thinking) vs the response actually rendered. If still thinking, schedule another 15-20 min and repeat. If 3+ hours with no progress, flag Tom.

### 6. Ingest the response

When done, scroll to the response, find the action buttons at the very bottom (the two-squares "copy markdown" button is leftmost). Click it. Then:

```bash
wl-paste > ~/review.md
```

That's your input for the next cross-verification cycle.

---

## Sim run mechanics

Family + compaction in parallel is the standard validation pair:

```bash
cd /home/luth/Programming/borg
pnpm sim:family-aging-parent --prefix vN.1 &
pnpm sim:shared-state-compaction --prefix vN.1 &
wait
```

(Check `package.json` for exact script names if they've moved.)

Sims write to `simulator-runs/` with the prefix. The overseer audit jsonl and the report.md are the most-useful artifacts for the review draft.

### LM Studio outages

LM Studio at `localhost:1234` provides embeddings. If Tom restarts it mid-sim, you'll see one or more `borg_hard_aborted_turn` events with ECONNREFUSED in logs. The aborted_turn mechanism (Sprint 6d-7) absorbs single outages; the sim continues. If 5+ consecutive turns fail, compaction will abort itself. Stop the sim, wait for Tom's all-clear, relaunch.

### Host switch

If Tom has been working on a different machine, the repo on ivory may be behind. Always check `git log --oneline -5` at the start of a session against what you expect. If commits you wrote in a previous session are missing, do `git pull --ff-only` before doing anything else. Tom caught this once; saved a wasted sprint.

---

## File conventions

- `~/review.md` -- latest GPT Pro response (overwritten each cycle)
- `~/borg-v[N]-review-questions.md` -- questions document drafted for cycle N
- `~/borgv[N].zip` -- artifact uploaded to GPT
- `simulator-runs/v[N].M-<scenario>-*` -- sim artifacts, keep across cycles for comparison

---

## What "done" looks like

The codebase Tom wants to look at and find:

0. **The memory-architecture inversion is complete** -- Sol's internal recall is global (never audience/session-gated for cognition); audience, session, role, and privacy are disclosure metadata and action-policy inputs only; and privacy is enforced as a post-recall disclosure judgment, not by hiding memories from Sol. The human-mind invariant tests pass. This lands via data-preserving forward migrations, never a reset. **Until this holds, the project is NOT done no matter how favorable the GPT review tone.** See the Cardinal Memory Rule in CLAUDE.md and BOUNDARIES.md. The criteria below are subordinate to this gate.
1. Works well across diverse scenarios (group chat, single-user dev, relationships, etc.).
2. Clean, not overengineered, no duplication, well organized.
3. Each mechanism justified by something concrete it prevents or enables.

Subordinate to criterion #0: when you've done a cycle where the GPT review is mostly "looks good, here are some forward-looking ideas" rather than "fix this", you're close on the quality axis. Run one more cycle to be sure, then -- only if the inversion in #0 also holds -- propose stopping.
