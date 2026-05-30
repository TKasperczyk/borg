/**
 * Live async-ingest smoke: drives the REAL stream-as-inbox path against the
 * real Anthropic model (OAuth) and the real LM Studio embedding endpoint.
 *
 * Unlike scripts/async-ingest-scenario.ts (scripted model, ManualClock), this
 * opens a real Borg via Borg.open({config}) with no injected clients, a real
 * SystemClock, and a fresh temp data dir, then exercises:
 *
 *   A. Burst + coalesce via the AUTONOMOUS worker (start() + onAppend wake +
 *      quiet-window). 3 rapid messages from one sender -> one coalesced turn.
 *   B. Transport dedup (same source_message_key enqueued twice).
 *   C. Multi-sender batch (2 distinct senders -> one batch turn; authority
 *      suppression is structural and unit-tested -- here we confirm it drains
 *      to a single terminal with count=2 without crashing).
 *   D. Crash-replay via the STARTUP-SCAN path. Enqueue with the worker stopped,
 *      close (crash), reopen on the same data dir, start the worker -> startup
 *      scan reconciles + drains exactly once; a second tick is empty.
 *
 * Spend: real model turns on the Max OAuth subscription (NOT the separate API
 * key -- auth is forced to "oauth"). ~3 cognition turns + their extraction.
 *
 * Run: pnpm tsx scripts/async-ingest-live.ts
 */
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  Borg,
  createSessionId,
  loadConfig,
  type EntityId,
  type SessionId,
  type StreamCursor,
  type StreamEntry,
} from "../src/index.js";

const EMBED_MODEL = "text-embedding-qwen3-8b-text-embedding";
const EMBED_DIMS = 4096;
const TERMINAL_KINDS = new Set<StreamEntry["kind"]>([
  "agent_msg",
  "agent_suppressed",
  "agent_observed",
]);
// Real Opus turns + extraction can take a while; give each drain a wide ceiling.
const DRAIN_TIMEOUT_MS = 240_000;

type Internals = {
  deps: {
    chatResponseWatermarkCoordinator: {
      getWatermark(sessionId: SessionId): StreamCursor | null;
    };
    sqlite: {
      prepare(sql: string): { get(...values: unknown[]): unknown };
    };
  };
};

type StreamResponseToView = {
  kind: string;
  from_cursor_exclusive: StreamCursor | null;
  through_cursor_inclusive: StreamCursor;
  source_entry_ids: string[];
  count: number;
};

const internals = (borg: Borg): Internals => borg as unknown as Internals;
const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

const stats = { passed: 0, failed: 0 };

async function step(name: string, fn: () => Promise<void>): Promise<void> {
  process.stdout.write(`\n=== ${name} ===\n`);
  try {
    await fn();
    stats.passed += 1;
    process.stdout.write(`PASS: ${name}\n`);
  } catch (error) {
    stats.failed += 1;
    process.stdout.write(`FAIL: ${name}\n`);
    process.stdout.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  }
}

async function waitFor(
  predicate: () => boolean,
  timeoutMs: number,
  pollMs = 500,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) {
      return true;
    }
    await sleep(pollMs);
  }
  return predicate();
}

function tail(borg: Borg, sessionId: SessionId): StreamEntry[] {
  return borg.stream.tail(500, { session: sessionId });
}

function terminalEntries(borg: Borg, sessionId: SessionId): StreamEntry[] {
  return tail(borg, sessionId).filter((entry) => TERMINAL_KINDS.has(entry.kind));
}

function userEntries(borg: Borg, sessionId: SessionId): StreamEntry[] {
  return tail(borg, sessionId).filter((entry) => entry.kind === "user_msg");
}

function getWatermark(borg: Borg, sessionId: SessionId): StreamCursor | null {
  return internals(borg).deps.chatResponseWatermarkCoordinator.getWatermark(sessionId);
}

function responseTo(entry: StreamEntry): StreamResponseToView | null {
  const value = (entry as { response_to?: StreamResponseToView }).response_to;
  return value ?? null;
}

function countUserContacts(borg: Borg): number {
  const row = internals(borg)
    .deps.sqlite.prepare(
      "SELECT COUNT(*) AS count FROM activity_events WHERE kind = 'user_contact'",
    )
    .get() as { count?: unknown } | undefined;
  return Number(row?.count ?? 0);
}

function readPerception(
  borg: Borg,
  parentEntryId: string,
): { caption: string; visible_text: string; image_kind: string } | null {
  const row = internals(borg)
    .deps.sqlite.prepare(
      "SELECT p.caption AS caption, p.visible_text AS visible_text, p.image_kind AS image_kind " +
        "FROM image_perception_artifacts a JOIN image_perception_payloads p ON a.payload_id = p.payload_id " +
        "WHERE a.parent_entry_id = ? LIMIT 1",
    )
    .get(parentEntryId) as
    | { caption?: string; visible_text?: string; image_kind?: string }
    | undefined;
  if (row === undefined || typeof row.caption !== "string") {
    return null;
  }
  return {
    caption: row.caption,
    visible_text: row.visible_text ?? "",
    image_kind: row.image_kind ?? "",
  };
}

function makeSession(externalId: string, label: string) {
  return {
    session_id: createSessionId(),
    source_type: "demo" as const,
    source_external_id: externalId,
    label,
    audience_label: label,
    conversation_kind: "thread" as const,
  };
}

async function enqueue(input: {
  borg: Borg;
  session: ReturnType<typeof makeSession>;
  senderEntityId: EntityId;
  externalMessageId: string;
  text: string;
}): Promise<{ status: string; streamEntryId: string }> {
  const result = await input.borg.enqueueMessage({
    session: input.session,
    userMessage: input.text,
    senderEntityId: input.senderEntityId,
    sourceMessageKey: {
      source_type: "demo",
      source_external_id: input.session.source_external_id,
      external_message_id: input.externalMessageId,
    },
    arrivedAt: Date.now(),
    audience: input.session.audience_label,
  });
  return { status: result.status, streamEntryId: result.streamEntryId };
}

function describeTerminal(entry: StreamEntry): string {
  const stamp = responseTo(entry);
  const text =
    entry.kind === "agent_msg"
      ? String((entry as { content?: unknown }).content ?? "").slice(0, 400)
      : `(${entry.kind})`;
  return `kind=${entry.kind} count=${stamp?.count ?? "n/a"} reply="${text}"`;
}

function openConfig(dataDir: string) {
  mkdirSync(dataDir, { recursive: true });
  writeFileSync(
    join(dataDir, "config.json"),
    JSON.stringify(
      {
        // Force OAuth (Max subscription) -- never the separately-billed API key.
        anthropic: { auth: "oauth" },
        embedding: { model: EMBED_MODEL, dims: EMBED_DIMS },
      },
      null,
      2,
    ),
  );
  return loadConfig({ dataDir });
}

async function main(): Promise<void> {
  // Optional filter so a single scenario can be re-run cheaply, e.g.
  //   LIVE_ONLY=D pnpm tsx scripts/async-ingest-live.ts
  const only = (process.env.LIVE_ONLY ?? "").toUpperCase();
  const wants = (letter: string): boolean => only === "" || only.includes(letter);

  const dataDir = mkdtempSync(join(tmpdir(), "borg-live-"));
  process.stdout.write(`data_dir=${dataDir}\n`);
  if (only !== "") {
    process.stdout.write(`scenario_filter=${only}\n`);
  }

  let borg = await Borg.open({ config: openConfig(dataDir) });

  try {
    // ---- A. Burst + coalesce via the autonomous worker ----------------------
    if (wants("A"))
    await step("A. burst -> autonomous coalesced drain (1 sender, 3 msgs)", async () => {
      const session = makeSession("live-burst", "Live burst room");
      const sender = borg.entities.resolve("Ada");
      borg.inbox.catchUp.start();

      const ids: string[] = [];
      for (let i = 1; i <= 3; i += 1) {
        const r = await enqueue({
          borg,
          session,
          senderEntityId: sender,
          externalMessageId: `burst-${i}`,
          text:
            i === 1
              ? "Hey, quick question for you."
              : i === 2
                ? "Actually three things -- first, are you around today?"
                : "And second, can you summarize what I just asked in one line?",
        });
        assert.equal(r.status, "enqueued", `burst-${i} should enqueue`);
        ids.push(r.streamEntryId);
      }

      assert.equal(userEntries(borg, session.session_id).length, 3, "3 queued user_msgs");
      assert.equal(getWatermark(borg, session.session_id), null, "no watermark before drain");

      const drained = await waitFor(
        () => terminalEntries(borg, session.session_id).length >= 1,
        DRAIN_TIMEOUT_MS,
      );
      assert.ok(drained, "worker should autonomously produce a terminal entry");

      const terminals = terminalEntries(borg, session.session_id);
      assert.equal(terminals.length, 1, "exactly ONE coalesced terminal for the burst");
      const terminal = terminals[0]!;
      const stamp = responseTo(terminal);
      assert.ok(stamp, "terminal carries a response_to stamp");
      assert.equal(stamp!.kind, "stream_backlog", "stamp kind");
      assert.equal(stamp!.count, 3, "stamp covers all 3 messages (coalesced)");
      assert.deepEqual(stamp!.source_entry_ids, ids, "stamp source ids == the 3 enqueued ids, in order");

      const watermark = getWatermark(borg, session.session_id);
      assert.deepEqual(watermark, stamp!.through_cursor_inclusive, "watermark == terminal through-cursor");

      process.stdout.write(`coalesced ${describeTerminal(terminal)}\n`);
    });

    // ---- B. Transport dedup -------------------------------------------------
    if (wants("B"))
    await step("B. dedup (same source_message_key twice)", async () => {
      const session = makeSession("live-dedup", "Live dedup room");
      const sender = borg.entities.resolve("Ada");
      const beforeContacts = countUserContacts(borg);

      const first = await enqueue({
        borg,
        session,
        senderEntityId: sender,
        externalMessageId: "dup-1",
        text: "Same message, sent twice by a flaky transport.",
      });
      assert.equal(first.status, "enqueued", "first is enqueued");

      const dup = await enqueue({
        borg,
        session,
        senderEntityId: sender,
        externalMessageId: "dup-1",
        text: "Same message, sent twice by a flaky transport.",
      });
      assert.equal(dup.status, "duplicate", "second is deduped");
      assert.equal(dup.streamEntryId, first.streamEntryId, "duplicate returns the original entry id");

      const queued = userEntries(borg, session.session_id);
      assert.equal(queued.length, 1, "only one user_msg persisted");
      assert.equal(
        countUserContacts(borg),
        beforeContacts + 1,
        "duplicate does not record a second user_contact",
      );
      process.stdout.write(`dedup ok: one user_msg, one user_contact\n`);

      // drain it so the session is clean (and exercise a 1-message drain).
      await waitFor(() => terminalEntries(borg, session.session_id).length >= 1, DRAIN_TIMEOUT_MS);
    });

    // ---- C. Multi-sender batch ---------------------------------------------
    if (wants("C"))
    await step("C. multi-sender batch (2 senders -> 1 batch turn, count=2)", async () => {
      const session = makeSession("live-group", "Live group room");
      const ada = borg.entities.resolve("Ada");
      const bo = borg.entities.resolve("Bo");

      const a = await enqueue({
        borg,
        session,
        senderEntityId: ada,
        externalMessageId: "grp-ada",
        text: "Bo and I are arguing -- is borg around to settle it?",
      });
      const b = await enqueue({
        borg,
        session,
        senderEntityId: bo,
        externalMessageId: "grp-bo",
        text: "Yeah, tell Ada she's wrong. What do you think?",
      });
      assert.equal(a.status, "enqueued");
      assert.equal(b.status, "enqueued");

      const drained = await waitFor(
        () => terminalEntries(borg, session.session_id).length >= 1,
        DRAIN_TIMEOUT_MS,
      );
      assert.ok(drained, "multi-sender batch drains to a terminal");

      const terminals = terminalEntries(borg, session.session_id);
      assert.equal(terminals.length, 1, "one terminal for the 2-sender batch");
      const stamp = responseTo(terminals[0]!);
      assert.ok(stamp, "batch terminal carries a stamp");
      assert.equal(stamp!.count, 2, "stamp covers both senders' messages");
      assert.deepEqual(
        stamp!.source_entry_ids,
        [a.streamEntryId, b.streamEntryId],
        "stamp covers both source ids in order",
      );
      assert.deepEqual(
        getWatermark(borg, session.session_id),
        stamp!.through_cursor_inclusive,
        "watermark advanced to batch through-cursor",
      );
      process.stdout.write(`multi-sender ${describeTerminal(terminals[0]!)}\n`);
    });

    // ---- D. Crash-replay via startup-scan path ------------------------------
    if (wants("D"))
    await step("D. crash-replay (enqueue, crash before drain, reopen, drain once)", async () => {
      const session = makeSession("live-crash", "Live crash room");
      const sender = borg.entities.resolve("Ada");

      // Stop the worker so onAppend does NOT wake a drain; messages stay pending.
      await borg.inbox.catchUp.stop({ graceful: true });

      const a = await enqueue({
        borg,
        session,
        senderEntityId: sender,
        externalMessageId: "crash-1",
        text: "If borg crashes right now, this must not be lost.",
      });
      const b = await enqueue({
        borg,
        session,
        senderEntityId: sender,
        externalMessageId: "crash-2",
        text: "And this one must be answered exactly once on restart.",
      });
      assert.equal(a.status, "enqueued");
      assert.equal(b.status, "enqueued");
      assert.equal(userEntries(borg, session.session_id).length, 2, "2 pending before crash");
      assert.equal(terminalEntries(borg, session.session_id).length, 0, "no terminal before crash");
      assert.equal(getWatermark(borg, session.session_id), null, "no watermark before crash");

      // Simulate a crash: close the process's borg and reopen on the SAME dir.
      await borg.close();
      borg = await Borg.open({ config: openConfig(dataDir) });

      // Startup scan should reconcile + drain this session exactly once.
      borg.inbox.catchUp.start();
      const drained = await waitFor(
        () => terminalEntries(borg, session.session_id).length >= 1,
        DRAIN_TIMEOUT_MS,
      );
      assert.ok(drained, "startup scan drains the recovered backlog");

      const terminals = terminalEntries(borg, session.session_id);
      assert.equal(terminals.length, 1, "exactly one terminal after replay (no double reply)");
      const stamp = responseTo(terminals[0]!);
      assert.ok(stamp);
      assert.equal(stamp!.count, 2, "replay coalesced both recovered messages");
      assert.deepEqual(stamp!.source_entry_ids, [a.streamEntryId, b.streamEntryId]);
      assert.deepEqual(getWatermark(borg, session.session_id), stamp!.through_cursor_inclusive);

      // Settle the worker first: graceful stop awaits the in-flight startup
      // drain + scan. Without this, a fresh tick() coalesces with the still-
      // in-flight drain (runTrackedDrain dedup) and returns ITS result.
      await borg.inbox.catchUp.stop({ graceful: true });

      // Now a fresh, fully-settled tick must find nothing -> proves exactly-once.
      const second = await borg.inbox.catchUp.tick(session.session_id);
      assert.equal(second.status, "empty", "second (settled) drain is empty -- no regeneration");
      assert.equal(terminalEntries(borg, session.session_id).length, 1, "still exactly one terminal");

      process.stdout.write(`replay ${describeTerminal(terminals[0]!)}; second_tick=${second.status}\n`);
    });

    // ---- E. Concurrent multi-person chatter while borg is mid-turn ----------
    // The real chat dynamic: several people keep posting WITHOUT waiting for a
    // reply, including while borg is already generating one. Proves the lock +
    // watermark + re-drain path: nothing lost, nothing answered twice, backlog
    // coalesced (borg does NOT force anyone to wait one-reply-at-a-time).
    if (wants("E"))
    await step("E. concurrent multi-sender chatter during in-flight turns (nobody waits)", async () => {
      const session = makeSession("live-concurrent", "Live concurrent room");
      const alice = borg.entities.resolve("Alice");
      const bob = borg.entities.resolve("Bob");
      const carol = borg.entities.resolve("Carol");

      borg.inbox.catchUp.start();

      const arrivals: Array<{ id: string; who: string; at: number }> = [];
      let t0 = 0;
      const fire = async (sender: EntityId, who: string, ext: string, text: string): Promise<void> => {
        const r = await enqueue({ borg, session, senderEntityId: sender, externalMessageId: ext, text });
        assert.equal(r.status, "enqueued", `${ext} enqueued`);
        const now = Date.now();
        if (arrivals.length === 0) {
          t0 = now;
        }
        arrivals.push({ id: r.streamEntryId, who, at: now });
        process.stdout.write(`  arrived ${who}:${ext} at +${now - t0}ms\n`);
      };

      // How many distinct turns have begun generating (perception/thought are
      // written early in a turn, before its terminal reply).
      const startedTurns = (): number =>
        new Set(
          tail(borg, session.session_id)
            .filter((e) => e.kind === "perception" || e.kind === "thought")
            .map((e) => (e as { turn_id?: string | null }).turn_id)
            .filter((turnId): turnId is string => typeof turnId === "string"),
        ).size;

      // 1) Alice opens the thread.
      await fire(alice, "Alice", "c-a1", "Hey borg, you around? Starting a thread.");

      // 2) Wait until turn 1 is actually GENERATING, then dump a flurry mid-turn.
      assert.ok(await waitFor(() => startedTurns() >= 1, DRAIN_TIMEOUT_MS, 200), "turn 1 starts");
      await fire(bob, "Bob", "c-b1", "oh while you're answering that -- the deploy went out btw");
      await fire(alice, "Alice", "c-a2", "yeah and staging is green now");
      await fire(carol, "Carol", "c-c1", "wait who approved the deploy though?");

      // 3) Wait until a SECOND turn begins, then add a straggler mid-turn-2.
      const turn2Started = await waitFor(() => startedTurns() >= 2, DRAIN_TIMEOUT_MS, 200);
      await fire(carol, "Carol", "c-c2", "...and is anyone watching the error rate right now?");
      process.stdout.write(`  turn2_started_before_straggler=${turn2Started}\n`);

      const expectedIds = arrivals.map((a) => a.id); // arrival order == entry_index order

      const coveredIds = (): string[] =>
        terminalEntries(borg, session.session_id).flatMap((t) => responseTo(t)?.source_entry_ids ?? []);
      const allCovered = (): boolean => expectedIds.every((id) => coveredIds().includes(id));

      // 4) Drain to quiescence; settle; backstop any final wave with manual ticks.
      await waitFor(allCovered, DRAIN_TIMEOUT_MS, 500);
      await borg.inbox.catchUp.stop({ graceful: true });
      let guard = 0;
      while (!allCovered() && guard < 6) {
        await borg.inbox.catchUp.tick(session.session_id);
        guard += 1;
      }
      assert.ok(allCovered(), "every concurrently-arrived message is answered");

      // 5) Core invariants: exactly-once, in arrival order, coalesced, monotonic.
      const terminals = terminalEntries(borg, session.session_id);
      const answeredInOrder = terminals.flatMap((t) => responseTo(t)?.source_entry_ids ?? []);
      assert.deepEqual(
        answeredInOrder,
        expectedIds,
        "every message answered exactly once, in arrival order (none lost, none doubled)",
      );
      assert.equal(new Set(answeredInOrder).size, answeredInOrder.length, "no message answered twice");
      assert.ok(terminals.length >= 2, "coalesced into batches -- did NOT wait one-reply-per-message");
      for (const terminal of terminals) {
        const stamp = responseTo(terminal)!;
        assert.equal(stamp.count, stamp.source_entry_ids.length, "stamp count == ids length");
      }
      assert.deepEqual(
        getWatermark(borg, session.session_id),
        responseTo(terminals[terminals.length - 1]!)!.through_cursor_inclusive,
        "watermark advanced to the final batch through-cursor",
      );

      // 6) "Nobody waited" proof: turn 1 answered Alice's opener ALONE, and the
      // flurry (Bob/Alice/Carol) arrived before that reply was even written.
      const a1Id = expectedIds[0]!;
      const turn1 = terminals.find((t) => {
        const stamp = responseTo(t);
        return stamp != null && stamp.source_entry_ids.length === 1 && stamp.source_entry_ids[0] === a1Id;
      });
      assert.ok(turn1, "turn 1 answered Alice's opener alone (the flurry was not yet drained)");
      for (const flurry of arrivals.slice(1, 4)) {
        assert.ok(
          flurry.at < turn1!.timestamp,
          `${flurry.who} arrived DURING turn 1's generation (before its reply landed)`,
        );
      }

      const shape = terminals.map((t) => responseTo(t)!.count).join(",");
      process.stdout.write(
        `  ${expectedIds.length} msgs from 3 senders -> ${terminals.length} coalesced turns (counts=[${shape}])\n`,
      );
      terminals.forEach((terminal, index) => {
        process.stdout.write(`  turn ${index + 1}: ${describeTerminal(terminal)}\n`);
      });
    });

    // ---- F. Real image on the async path (perceive-on-receipt, live vision) --
    // Enqueue a message carrying an actual PNG; the real vision model perceives
    // it AT RECEIPT (durable, before ack); the coalescing catch-up turn renders
    // the stored perception and the cognition model answers from it.
    if (wants("F"))
    await step("F. real image enqueued async -> perceived on receipt -> rendered -> answered", async () => {
      const imageBytes = readFileSync("/tmp/borg-live-img.png");
      const session = makeSession("live-image", "Live image room");
      const sender = borg.entities.resolve("Ada");
      borg.inbox.catchUp.start();

      const enq = await borg.enqueueMessage({
        session,
        userMessage: "Read this image for me -- what's the big number, and what does the text say?",
        senderEntityId: sender,
        sourceMessageKey: {
          source_type: "demo",
          source_external_id: session.source_external_id,
          external_message_id: "img-1",
        },
        arrivedAt: Date.now(),
        audience: session.audience_label,
        attachments: [{ mediaType: "image/png", bytes: imageBytes }],
      });
      assert.equal(enq.status, "enqueued", "image message enqueued");

      // perceive-on-receipt: by the time enqueue returns, the real vision model
      // has run and the perception is durably stored.
      const perception = readPerception(borg, enq.streamEntryId);
      assert.ok(perception, "perception durably stored at receipt (perceive-on-receipt)");
      assert.ok(perception!.caption.length > 0, "non-empty caption from the real vision model");
      process.stdout.write(`  caption="${perception!.caption.slice(0, 200)}"\n`);
      process.stdout.write(`  visible_text="${perception!.visible_text.slice(0, 120)}"\n`);
      process.stdout.write(`  image_kind="${perception!.image_kind}"\n`);

      const drained = await waitFor(
        () => terminalEntries(borg, session.session_id).length >= 1,
        DRAIN_TIMEOUT_MS,
      );
      assert.ok(drained, "image message drains to a terminal");
      const terminals = terminalEntries(borg, session.session_id);
      assert.equal(terminals.length, 1, "one terminal for the image message");
      assert.equal(responseTo(terminals[0]!)!.count, 1, "stamp covers the one image message");

      const terminal = terminals[0]!;
      const reply =
        terminal.kind === "agent_msg"
          ? String((terminal as { content?: unknown }).content ?? "")
          : `(${terminal.kind})`;
      process.stdout.write(`  reply="${reply.slice(0, 300)}"\n`);

      // End-to-end live proof: the real perception captured the rendered image
      // content (the big "73"), available either in the perception or echoed by
      // the cognition model in its reply.
      const sawContent =
        perception!.visible_text.includes("73") ||
        perception!.caption.includes("73") ||
        reply.includes("73");
      assert.ok(sawContent, "the real vision perception captured the image content (73)");
    });
  } finally {
    await borg.inbox.catchUp.stop({ graceful: true }).catch(() => undefined);
    await borg.close().catch(() => undefined);

    process.stdout.write(`\nSUMMARY: ${stats.passed} passed / ${stats.failed} failed\n`);
    if (stats.failed === 0) {
      rmSync(dataDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 20 });
      process.stdout.write("temp_data_dir_cleaned=true\n");
    } else {
      process.stdout.write(`temp_data_dir_left_for_inspection=${dataDir}\n`);
    }
  }

  if (stats.failed > 0) {
    process.exitCode = 1;
  }
}

await main();
