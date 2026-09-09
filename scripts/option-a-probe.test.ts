import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, expect, it, vi } from "vitest";
import { StreamWriter, type StreamEntry } from "../src/stream/index.js";
import { sessionFromCaller } from "../src/sidecar/team-agent-identity.js";
import { runOptionAProbeCli, SCENARIO_G, summarizeProbeLatency } from "./option-a-probe.js";

const dirs: string[] = [];
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function args(rawSession = "existing-thread") {
  const dir = mkdtempSync(join(tmpdir(), "option-a-script-"));
  dirs.push(dir);
  return {
    dir,
    argv: [
      "--data-dir",
      dir,
      "--session",
      rawSession,
      "--sender-external-id",
      "tomasz-transport-id",
      "--timeout-ms",
      "5000",
    ],
  };
}

const ledger = {
  sections: [
    {
      id: "current_user_message",
      entries: [
        { id: "user:current", source_type: "current_user_message", text: "captured question" },
      ],
    },
    {
      id: "episodes",
      entries: [
        {
          id: "episode:ep_xij9l0i4k8uvespi",
          source_type: "episode",
          text: "captured evidence",
          state_metadata: { episode_id: "ep_xij9l0i4k8uvespi" },
        },
        { id: "episode:ep_another", source_type: "episode" },
      ],
    },
    {
      id: "semantic_graph",
      entries: [
        {
          id: "node:example",
          source_type: "semantic_node",
          state_metadata: { episode_id: "ep_xij9l0i4k8uvespi" },
        },
      ],
    },
    { id: "prior_session_memory", entries: [] },
  ],
};

async function replayFixture(
  options: {
    runs?: number;
    rawSession?: string;
    observed?: boolean;
    earlySuppressionRun?: number;
    traceCleanup?: "hang" | "transport failure";
    deferWake?: boolean;
    regenerate?: boolean;
    failure?: "missing capture" | "unverified capture" | "malformed ledger" | "trace overflow";
    failRun?: number;
  } = {},
) {
  const { dir, argv } = args(options.rawSession);
  const sessionId = sessionFromCaller(options.rawSession ?? "existing-thread");
  const writer = new StreamWriter({ dataDir: dir, sessionId });
  const turns: {
    run: number;
    source: StreamEntry;
    terminal: StreamEntry;
    turnId: string;
    reply: string;
    observed: boolean;
  }[] = [];
  for (let run = 1; run <= (options.runs ?? 1); run++) {
    const source = await writer.append({ kind: "user_msg", content: SCENARIO_G.text });
    const reply = `Committed reply ${run}`;
    const turnId = `native-probe-${run}`;
    const observed = options.observed === true || options.earlySuppressionRun === run;
    const terminal = await writer.append({
      kind: observed ? "agent_observed" : "agent_msg",
      content: observed ? { reason: "Generation gate suppressed this turn" } : reply,
      turn_id: turnId,
      response_to: {
        kind: "stream_backlog",
        from_cursor_exclusive: null,
        through_cursor_inclusive: { ts: source.timestamp, entryId: source.id },
        source_entry_ids: [source.id],
        count: 1,
      },
    });
    turns.push({ run, source, terminal, turnId, reply, observed });
  }
  writer.close();
  const failRun = options.failRun ?? 1;
  mkdirSync(join(dir, "captures"));
  writeFileSync(
    join(dir, "captures", "finalizer-contexts.jsonl"),
    turns
      .flatMap((turn) => {
        if (turn.run === failRun && options.failure === "missing capture") return [];
        if (turn.run === options.earlySuppressionRun) return [];
        return (options.regenerate ? ["initial", "regenerate"] : ["initial"]).map(
          (attempt_kind) =>
            JSON.stringify({
              schema_version: 1,
              capture_id: `capture-${turn.run}-${attempt_kind}`,
              captured_at: 1234,
              path: "system_2",
              projected_context: {},
              surfaces: {
                legacy: { system: "Exact finalizer system", fingerprint: {} },
                compact: { system: "Compact system", fingerprint: {} },
              },
              image_sidecars: [],
              replay: { eligible: true, exclusion_reason: null },
              live_outcome: { status: "completed", attempts: 1 },
              turn_id: turn.turnId,
              session_id: sessionId,
              attempt_kind,
              evidence_ledger:
                turn.run === failRun && options.failure === "malformed ledger"
                  ? { sections: null }
                  : ledger,
              live_surface_variant: "legacy",
              fidelity: {
                verified: !(turn.run === failRun && options.failure === "unverified capture"),
                request: { canonicalSha256: "fixture-fingerprint" },
                surfaceMatchesRequest: true,
              },
              live_request: {
                model: "p4-model",
                system: "Exact finalizer system",
                tools: [{ name: "EmitAnswer" }],
              },
            }) + "\n",
        );
      })
      .join(""),
  );

  let enqueued = 0;
  let claimed = 0;
  let releaseClaim: (() => void) | undefined;
  let traceSignal: AbortSignal | null | undefined;
  let traceStarted!: () => void;
  const pendingTrace = new Promise<void>((resolve) => {
    traceStarted = resolve;
  });
  const posts: { path: string; body: Record<string, unknown> }[] = [];
  const traceReads = new Map<number, number>();
  const events = turns.map((turn) =>
    [
      { event: "turn_phase.completed", phase: "perception", duration_ms: 10 },
      ...(turn.run === options.earlySuppressionRun
        ? []
        : [
            { event: "llm_call.started", label: "system_2_finalizer" },
            { event: "llm_call.completed", label: "system_2_finalizer" },
          ]),
      ...(turn.observed
        ? []
        : [{ event: "sidecar.delivery_waiter.woke", wake: "available", session_ids: [sessionId] }]),
      {
        event: "turn.terminal",
        duration_ms: 25,
        outcome:
          turn.run === options.earlySuppressionRun ? "suppressed_generation_gate" : "reflected",
      },
    ].map((event, i) => ({
      ...event,
      turnId: turn.turnId,
      ts:
        turn.run * 100 +
        (options.deferWake && event.event === "sidecar.delivery_waiter.woke" ? 6 : i),
      wallMs: i * 5,
    })),
  );
  vi.stubGlobal("fetch", async (url: URL, init: RequestInit) => {
    const path = url.pathname;
    if (path === "/memory/trace") {
      const reads = (traceReads.get(enqueued) ?? 0) + 1;
      traceReads.set(enqueued, reads);
      if (
        options.traceCleanup &&
        enqueued > 0 &&
        Number(url.searchParams.get("since")) >= enqueued * 100 + 5
      ) {
        traceSignal = init.signal;
        traceStarted();
        return new Promise<Response>((_resolve, reject) => {
          init.signal?.addEventListener(
            "abort",
            () =>
              reject(
                options.traceCleanup === "hang"
                  ? init.signal?.reason
                  : new Error("Trace transport failed during cleanup"),
              ),
            { once: true },
          );
        });
      }
      return Response.json({
        option_a_probe: true,
        nextSince: enqueued === 0 ? 0 : enqueued * 100 + (options.deferWake && reads > 1 ? 7 : 5),
        truncated: enqueued === failRun && options.failure === "trace overflow",
        events: events
          .slice(0, enqueued)
          .flat()
          .filter(
            (event) =>
              event.ts > Number(url.searchParams.get("since")) &&
              !(options.deferWake && reads === 1 && event.event === "sidecar.delivery_waiter.woke"),
          ),
      });
    }
    const body = JSON.parse(init.body as string) as Record<string, unknown>;
    posts.push({ path, body });
    if (path === "/memory/enqueue") {
      // Every run starts a fresh long poll and finishes the previous run's ack.
      expect(releaseClaim).toBeDefined();
      expect(posts.filter((post) => post.path === "/memory/agent-deliveries/ack")).toHaveLength(
        turns.slice(0, enqueued).filter((turn) => !turn.observed).length,
      );
      const turn = turns[enqueued++]!;
      if (turn.observed) claimed++;
      releaseClaim?.();
      return Response.json({
        status: "enqueued",
        sidecar_session_id: sessionId,
        entry_id: turn.source.id,
      });
    }
    if (path === "/memory/await-response") {
      const turn = turns.find((turn) => turn.source.id === body.entry_id)!;
      return Response.json(
        turn.observed
          ? { status: "observed", terminal_id: turn.terminal.id }
          : { status: "answered", terminal_id: turn.terminal.id, reply: turn.reply },
      );
    }
    if (path === "/memory/agent-deliveries/ack") {
      if (options.traceCleanup) await pendingTrace;
      return Response.json({ status: "acknowledged" });
    }
    if (path === "/memory/agent-deliveries/claim") {
      while (claimed >= enqueued) {
        await new Promise<void>((resolve, reject) => {
          const onAbort = () => {
            releaseClaim = undefined;
            reject(init.signal?.reason);
          };
          releaseClaim = () => {
            init.signal?.removeEventListener("abort", onAbort);
            releaseClaim = undefined;
            resolve();
          };
          init.signal?.addEventListener("abort", onAbort, { once: true });
        });
      }
      const turn = turns[claimed++]!;
      return Response.json({
        deliveries: [
          {
            delivery_id: `delivery-${turn.run}`,
            claim_generation: 1,
            terminal_entry_id: turn.terminal.id,
            sidecar_session_id: sessionId,
            content: turn.reply,
            task_id: `native:${turn.terminal.id}`,
            created_at: new Date().toISOString(),
          },
        ],
      });
    }
    throw new Error(`Unexpected route ${path}`);
  });
  const log = vi.spyOn(console, "log").mockImplementation(() => {});
  const stderr = vi.spyOn(console, "error").mockImplementation(() => {});
  return {
    argv,
    posts,
    stderr,
    turns,
    log,
    report: () => JSON.parse(log.mock.calls.at(-1)![0] as string),
    traceSignal: () => traceSignal,
    run: (extra: string[] = []) =>
      runOptionAProbeCli([...argv, ...extra], { BORG_MEMORY_TOKEN: "test-token" }),
  };
}

it("refuses replay before enqueue when the running sidecar probe is off", async () => {
  const { argv } = args();
  const fetchFn = vi.fn(async () => Response.json({ events: [] }));
  vi.stubGlobal("fetch", fetchFn);
  await expect(runOptionAProbeCli(argv, { BORG_MEMORY_TOKEN: "test-token" })).rejects.toThrow(
    "not enabled",
  );
  expect(fetchFn).toHaveBeenCalledTimes(1);
});

it("keeps G defaults and full captures, reports ordered ledger rows for every finalizer and observes the delivery wake", async () => {
  const h = await replayFixture({ regenerate: true });
  await h.run();
  expect(h.posts.find((post) => post.path === "/memory/enqueue")!.body).toMatchObject({
    tenant: SCENARIO_G.tenant,
    session: "existing-thread",
    text: SCENARIO_G.text,
    conversation: SCENARIO_G.conversation,
    sender: {
      external_id: "tomasz-transport-id",
      display_name: SCENARIO_G.senderName,
      operator: false,
    },
    flags: { mentioned: true, quotes_bot: false },
  });
  const report = h.report();
  expect(report.finalizers[0].evidence_ledger).toEqual(ledger);
  expect(report.finalizers[0].live_request.system).toBe("Exact finalizer system");
  expect(report).toMatchObject({
    requested_runs: 1,
    completed_runs: 1,
    committed_reply: "Committed reply 1",
    delivery_woke_waiter: true,
    trace_truncated: false,
    aggregate_latency: { delivery_received_after_ms: { count: 1, missing_count: 0 } },
  });
  expect(report.stage_timings).toHaveLength(2);
  expect(report.llm_timings[0].duration_ms).toBe(5);
  expect(report.ledger_summary).toHaveLength(2);
  expect(report.ledger_summary[1].attempt_kind).toBe("regenerate");
  expect(report.finalizers[0]).toMatchObject({
    capture_id: "capture-1-initial",
    captured_at: 1234,
    schema_version: 1,
    fidelity: {
      verified: true,
      surfaceMatchesRequest: true,
      request: { canonicalSha256: "fixture-fingerprint" },
    },
  });
  expect(report.ledger_summary[0]).toEqual({
    finalizer_index: 1,
    attempt_kind: "initial",
    sections: [
      {
        id: "current_user_message",
        row_count: 1,
        rows: [{ id: "user:current", is_episode: false, episode_id: null }],
      },
      {
        id: "episodes",
        row_count: 2,
        rows: [
          {
            id: "episode:ep_xij9l0i4k8uvespi",
            is_episode: true,
            episode_id: "ep_xij9l0i4k8uvespi",
          },
          { id: "episode:ep_another", is_episode: true, episode_id: "ep_another" },
        ],
      },
      {
        id: "semantic_graph",
        row_count: 1,
        rows: [{ id: "node:example", is_episode: false, episode_id: null }],
      },
      { id: "prior_session_memory", row_count: 0, rows: [] },
    ],
  });
  expect(h.posts.find((post) => post.path === "/memory/agent-deliveries/ack")!.body).toMatchObject({
    outcome: "failed_permanent",
    claim_generation: 1,
  });
});

it.each(["personal", "channel"])(
  "sends a configurable %s scenario without a mention and supports compact reports",
  async (type) => {
    const h = await replayFixture({ rawSession: "custom-thread" });
    await h.run([
      "--question",
      "A different question?",
      "--sender-display-name",
      "Another sender",
      "--sender-external-id",
      "other-transport-id",
      "--conversation-type",
      type,
      "--conversation-external-id",
      "other-conversation",
      "--conversation-name",
      "Another conversation",
      "--mentioned=false",
      "--compact",
      "--external-message-id",
      "explicit-id",
      "--observed-at",
      "2026-09-09T10:00:00Z",
    ]);
    expect(h.posts.find((post) => post.path === "/memory/enqueue")!.body).toMatchObject({
      session: "custom-thread",
      text: "A different question?",
      sender: { display_name: "Another sender", external_id: "other-transport-id" },
      conversation: { type, name: "Another conversation", external_id: "other-conversation" },
      flags: { mentioned: false },
      external_message_id: "explicit-id",
      observed_at: "2026-09-09T10:00:00Z",
    });
    const report = h.report();
    expect(report.finalizers[0]).not.toHaveProperty("evidence_ledger");
    expect(report.finalizers[0].live_request).toEqual({
      model: "p4-model",
      tools: [{ name: "EmitAnswer" }],
    });
    expect(report.ledger_summary[0].sections[1].rows[0].episode_id).toBe("ep_xij9l0i4k8uvespi");
    expect(h.log.mock.calls[0]![0]).not.toContain("Exact finalizer system");
    expect(h.log.mock.calls[0]![0]).not.toContain("captured evidence");
  },
);

it("repeats sequentially with unique message IDs, isolated deliveries/traces, and per-run expectations", async () => {
  const h = await replayFixture({ runs: 3 });
  const exitCode = process.exitCode;
  await h.run([
    "--runs",
    "3",
    "--external-message-id",
    "gate",
    "--compact",
    "--mentioned",
    "true",
    "--expect-present",
    "Committed",
    "--expect-present",
    "missing",
    "--expect-absent",
    "private detail",
    "--expect-absent",
    "reply",
  ]);
  const enqueues = h.posts.filter((post) => post.path === "/memory/enqueue");
  const ids = enqueues.map((post) => post.body.external_message_id);
  expect(new Set(ids).size).toBe(3);
  expect(ids.every((id) => typeof id === "string" && id.startsWith("gate-") && id !== "gate")).toBe(
    true,
  );
  expect(enqueues.every((post) => post.body.text === SCENARIO_G.text)).toBe(true);
  const report = h.report();
  expect(report.requested_runs).toBe(3);
  expect(report.completed_runs).toBe(3);
  for (const [index, run] of report.runs.entries()) {
    expect(run).toMatchObject({
      run: index + 1,
      committed_reply: `Committed reply ${index + 1}`,
      external_message_id: ids[index],
      delivery: { delivery_id: `delivery-${index + 1}` },
      delivery_woke_waiter: true,
    });
    expect(
      run.stage_timings.every(
        (event: { turnId: string }) => event.turnId === `native-probe-${index + 1}`,
      ),
    ).toBe(true);
    expect(run.finalizers).toHaveLength(1);
    expect(run.terminal_received_after_ms).toBeGreaterThanOrEqual(0);
    expect(run.delivery_received_after_ms).toBeGreaterThanOrEqual(0);
    expect(run.expectations.map((expectation: { pass: boolean }) => expectation.pass)).toEqual([
      true,
      false,
      true,
      false,
    ]);
  }
  expect(report.aggregate_latency.terminal_received_after_ms.count).toBe(3);
  expect(report.aggregate_latency.delivery_received_after_ms.count).toBe(3);
  const output = h.stderr.mock.calls.flat().join("\n");
  expect(output.match(/PASS expect-/g)).toHaveLength(6);
  expect(output.match(/FAIL expect-/g)).toHaveLength(6);
  expect(output.match(/committed reply:/g)).toHaveLength(3);
  expect(output).toContain("Aggregate terminal_received_after_ms: median=");
  expect(output).toContain("Aggregate delivery_received_after_ms: median=");
  expect(h.log).toHaveBeenCalledTimes(1);
  expect(process.exitCode).toBe(exitCode);
});

it.each(["missing capture", "unverified capture", "malformed ledger", "trace overflow"] as const)(
  "preserves failure semantics and partial multi-run reports on %s",
  async (failure) => {
    const h = await replayFixture({ runs: 3, failure, failRun: 2 });
    await expect(
      h.run(["--runs", "3", "--compact", "--expect-present", "Committed"]),
    ).rejects.toThrow();
    const report = h.report();
    expect(report.completed_runs).toBe(1);
    expect(report.runs).toHaveLength(2);
    expect(report.runs[1].error).toBeTypeOf("string");
    expect(h.posts.filter((post) => post.path === "/memory/enqueue")).toHaveLength(2);
    expect(report.aggregate_latency.terminal_received_after_ms).toMatchObject({
      count: 2,
      missing_count: 1,
      passes_gate: false,
    });
    expect(report.aggregate_latency.delivery_received_after_ms).toMatchObject({
      count: 2,
      missing_count: 1,
      passes_gate: false,
    });
  },
);

it("does not treat suppression as passing an absence check or a delivery latency gate", async () => {
  const h = await replayFixture({ observed: true });
  await h.run(["--expect-absent", "private detail"]);
  expect(h.report()).toMatchObject({
    committed_reply: null,
    delivery_received_after_ms: null,
    delivery_woke_waiter: false,
    expectations: [{ pass: false, reason: "no committed reply" }],
    aggregate_latency: {
      delivery_received_after_ms: {
        count: 0,
        missing_count: 1,
        median_ms: null,
        p95_ms: null,
        passes_gate: false,
      },
    },
  });
  expect(h.posts.filter((post) => post.path === "/memory/agent-deliveries/ack")).toHaveLength(0);
});

it("records generation-gate suppression without a capture and continues the batch", async () => {
  const h = await replayFixture({ runs: 2, earlySuppressionRun: 1 });
  await h.run(["--runs", "2", "--expect-absent", "private detail"]);
  expect(h.report()).toMatchObject({
    completed_runs: 2,
    runs: [
      {
        finalizer_ran: false,
        capture_status: "not_applicable_early_suppression",
        finalizers: [],
        ledger_summary: [],
        committed_reply: null,
        early_suppression: {
          outcome: "suppressed_generation_gate",
          content: { reason: "Generation gate suppressed this turn" },
        },
      },
      {
        finalizer_ran: true,
        capture_status: "captured",
        committed_reply: "Committed reply 2",
        delivery_woke_waiter: true,
      },
    ],
  });
});

it("still requires a capture when an observed terminal followed finalizer execution", async () => {
  const h = await replayFixture({ observed: true, failure: "missing capture" });
  await expect(h.run()).rejects.toThrow("Probe evidence incomplete");
  expect(h.report()).toMatchObject({ finalizer_ran: true, capture_status: "missing" });
});

it("cancels a pending trace fetch during cleanup without waiting for the overall timeout", async () => {
  const h = await replayFixture({ traceCleanup: "hang" });
  await h.run();
  expect(h.traceSignal()?.aborted).toBe(true);
  expect(h.traceSignal()?.reason.message).toBe("Probe polling finished");
  expect(h.report()).not.toHaveProperty("error");
});

it("captures a delivery wake published after the already-observed turn terminal trace", async () => {
  const h = await replayFixture({ deferWake: true });
  await h.run();
  expect(h.report().delivery_woke_waiter).toBe(true);
  expect(h.report().stage_timings).toHaveLength(2);
  expect(h.report().llm_timings).toHaveLength(1);
});

it("surfaces a transport failure during trace cleanup instead of reporting success", async () => {
  const h = await replayFixture({ traceCleanup: "transport failure" });
  await expect(h.run()).rejects.toThrow("Trace transport failed during cleanup");
  expect(h.report()).toMatchObject({
    error: "Trace transport failed during cleanup",
    completed_runs: 0,
  });
});

it.each([
  ["--runs", "0"],
  ["--runs", "1.5"],
  ["--runs", "-1"],
  ["--runs", "NaN"],
  ["--conversation-type", "group"],
  ["--mentioned", "maybe"],
  ["--expect-present", ""],
  ["--expect-absent", ""],
])("rejects invalid %s=%s before any network mutation", async (flag, value) => {
  const { argv } = args();
  const fetchFn = vi.fn();
  vi.stubGlobal("fetch", fetchFn);
  await expect(
    runOptionAProbeCli([...argv, flag, value], { BORG_MEMORY_TOKEN: "test-token" }),
  ).rejects.toThrow();
  expect(fetchFn).not.toHaveBeenCalled();
});

it("calculates an ordinary median and nearest-rank p95 against inclusive latency thresholds", () => {
  const samples = [30_000, 10_000, 25_000, 15_000].map((ms) => ({
    terminal_received_after_ms: ms,
    delivery_received_after_ms: ms + 1,
  }));
  expect(summarizeProbeLatency(samples)).toEqual({
    terminal_received_after_ms: {
      count: 4,
      missing_count: 0,
      median_ms: 20_000,
      p95_ms: 30_000,
      passes_gate: true,
    },
    delivery_received_after_ms: {
      count: 4,
      missing_count: 0,
      median_ms: 20_001,
      p95_ms: 30_001,
      passes_gate: false,
    },
  });
  const tail = Array.from({ length: 20 }, (_, index) => ({
    terminal_received_after_ms: (index + 1) * 1000,
  }));
  expect(summarizeProbeLatency(tail).terminal_received_after_ms).toMatchObject({
    median_ms: 10_500,
    p95_ms: 19_000,
  });
  expect(summarizeProbeLatency(tail.slice(0, 3)).terminal_received_after_ms).toMatchObject({
    median_ms: 2000,
    p95_ms: 3000,
  });
  expect(summarizeProbeLatency([{}], 2).terminal_received_after_ms).toEqual({
    count: 0,
    missing_count: 2,
    median_ms: null,
    p95_ms: null,
    passes_gate: false,
  });
});
