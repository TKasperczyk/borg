# Workflow

How Tom and Claude actually run sprints on borg. Read this after CLAUDE.md / AGENTS.md when you (Claude) come back from a context compaction.

CLAUDE.md / AGENTS.md tell you what NOT to do in the codebase (scope, guardrails, taste). This file tells you what TO do in the dev loop.

---

## No production users yet

Borg is in active development. Nothing depends on it externally. **Schema migrations, on-disk formats, and persisted shapes can be changed freely.** No backward-compat patching, no defensive migration-N rewrites, no conservative NULL backfills "in case real data exists." If a column needs to change, just change it -- worst case is wiping `.borg-data` and re-running sims, which is cheap.

This unlocks cleaner fixes than the orchestrator might otherwise reach for. For example: if `composeMigrations` orders migrations weirdly, fix the function; don't pile defensive updates into every migration that touches a re-created table. If a previous migration's column set is now wrong, edit it in place; don't add another migration that ALTERs the result.

The corollary: **don't optimize for the "what if a real user upgrades from version X" case.** That case doesn't exist. The code that polishes that case is code that has to be maintained for no benefit.

## Project goal (the only one that matters)

A cognitive memory architecture for an LLM that is:

1. **Generic** -- works across group chats, single-user dev, relationship stuff, and other similar use cases. Not overfit to one scenario.
2. **Clean code** -- no overengineering, no duplication, no dead paths, well organized.
3. **Simple flow** -- minimal paths. Every additional check or branch must be worth the complexity it adds. If you can't defend it, delete it.

Behavioral correctness is already in good shape across the sim suite as of v83.1. The remaining work is mostly architecture quality, not behavioral wins. Don't chase ghosts.

**Stop at diminishing returns.** Signs you've hit them:

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

1. Works well across diverse scenarios (group chat, single-user dev, relationships, etc.).
2. Clean, not overengineered, no duplication, well organized.
3. Each mechanism justified by something concrete it prevents or enables.

When you've done a cycle where the GPT review is mostly "looks good, here are some forward-looking ideas" rather than "fix this", you're close. Run one more cycle to be sure, then propose stopping.
