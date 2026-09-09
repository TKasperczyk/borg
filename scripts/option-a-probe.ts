/** Replays configurable scenarios through a sidecar's durable inbox and claim/ack transport. */
import { randomUUID } from "node:crypto";
import { existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { parseArgs } from "node:util";
import { z } from "zod";

import { StreamReader } from "../src/stream/index.js";
import { sessionFromCaller } from "../src/sidecar/team-agent-identity.js";
import { sessionIdSchema, streamEntryIdSchema } from "../src/util/id-schemas.js";
import { runAbCliEntrypoint } from "./ab-cli.js";
import { openCaptureSnapshot } from "./capture-snapshot.js";
import {
  parseFinalizerContextCaptureRecord,
  type FinalizerContextCaptureRecord,
} from "../src/cognition/deliberation/finalizer-context-capture.js";

export const SCENARIO_G = {
  tenant: "team-agent-ai",
  text: "Czy Marcin będzie w następnym tygodniu w pracy?",
  senderName: "Tomasz Kasperczyk",
  conversation: {
    type: "groupChat" as const,
    name: "AI Ninjas",
    external_id: "19:d93832bc19034403b15732c6441d3391@thread.v2",
  },
};

const eventSchema = z
  .object({
    event: z.string(),
    turnId: z.string(),
    ts: z.number(),
    wallMs: z.number(),
  })
  .passthrough();
const traceSchema = z.object({
  option_a_probe: z.boolean().optional(),
  disabled: z.boolean().optional(),
  nextSince: z.number().optional(),
  truncated: z.boolean().optional(),
  events: z.array(eventSchema),
});
const deliverySchema = z.object({
  delivery_id: z.string(),
  claim_generation: z.number(),
  terminal_entry_id: streamEntryIdSchema,
  sidecar_session_id: sessionIdSchema,
  content: z.string(),
  task_id: z.string(),
  created_at: z.string(),
});
const terminalSchema = z.discriminatedUnion("status", [
  z.object({ status: z.literal("pending") }),
  z.object({ status: z.literal("generating") }),
  z.object({ status: z.literal("answered"), terminal_id: streamEntryIdSchema, reply: z.string() }),
  z.object({ status: z.literal("observed"), terminal_id: streamEntryIdSchema }),
]);
const ledgerRowsSchema = z.object({
  sections: z.array(
    z.object({
      id: z.string(),
      entries: z.array(
        z.object({
          id: z.string(),
          source_type: z.string(),
          state_metadata: z.object({ episode_id: z.string().optional() }).optional(),
        }),
      ),
    }),
  ),
});

type ProbeReport = Record<string, unknown> & {
  run: number;
  terminal_received_after_ms?: number;
  delivery_received_after_ms?: number | null;
  committed_reply?: string | null;
};

export async function runOptionAProbeCli(
  argv: readonly string[],
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  const { values } = parseArgs({
    args: [...argv],
    options: {
      "base-url": { type: "string", default: env.BORG_MEMORY_URL ?? "http://127.0.0.1:8088" },
      tenant: { type: "string", default: SCENARIO_G.tenant },
      session: { type: "string" },
      question: { type: "string", default: SCENARIO_G.text },
      "sender-display-name": { type: "string", default: SCENARIO_G.senderName },
      "sender-external-id": { type: "string" },
      "conversation-type": { type: "string", default: SCENARIO_G.conversation.type },
      "conversation-external-id": { type: "string", default: SCENARIO_G.conversation.external_id },
      "conversation-name": { type: "string", default: SCENARIO_G.conversation.name },
      mentioned: { type: "string", default: "true" },
      "data-dir": { type: "string" },
      "external-message-id": { type: "string" },
      "observed-at": { type: "string" },
      operator: { type: "boolean", default: false },
      "timeout-ms": { type: "string", default: "900000" },
      runs: { type: "string", default: "1" },
      "expect-absent": { type: "string", multiple: true, default: [] },
      "expect-present": { type: "string", multiple: true, default: [] },
      compact: { type: "boolean", default: false },
      help: { type: "boolean", short: "h" },
    },
  });
  if (values.help) {
    console.log(
      [
        "Usage: BORG_MEMORY_TOKEN=... pnpm exec tsx scripts/option-a-probe.ts --session <existing raw team-agent thread id> --sender-external-id <transport sender id> --data-dir <sidecar tenant directory>",
        "Scenario (defaults to G): [--question TEXT] [--sender-display-name NAME] [--conversation-type groupChat|personal|channel] [--conversation-external-id ID] [--conversation-name NAME] [--mentioned true|false] [--operator] [--tenant TENANT]",
        "Replay: [--runs N] [--external-message-id ID] [--observed-at ISO] [--base-url URL] [--timeout-ms 900000]",
        "Report: [--expect-present TEXT]... [--expect-absent TEXT]... [--compact]",
        "JSON goes to stdout; per-run replies, expectation results and latency summaries go to stderr. See docs/option-a-probe.md.",
      ].join("\n"),
    );
    return;
  }
  const required = (value: string | undefined, name: string) => {
    if (!value?.trim()) throw new Error(`Missing ${name}; see --help and docs/option-a-probe.md`);
    return value;
  };
  const token = required(env.BORG_MEMORY_TOKEN, "BORG_MEMORY_TOKEN");
  const rawSession = required(values.session, "--session");
  const senderExternalId = required(values["sender-external-id"], "--sender-external-id");
  const dataDir = required(values["data-dir"], "--data-dir");
  if (!statSync(dataDir).isDirectory())
    throw new Error("--data-dir must be the sidecar's tenant directory");
  const timeoutMs = z.coerce.number().int().positive().parse(values["timeout-ms"]);
  const runCount = z.coerce.number().int().positive().parse(values.runs);
  const observedAtOverride = z.iso
    .datetime({ offset: true })
    .optional()
    .parse(values["observed-at"]);
  const question = required(values.question, "--question");
  const sender = {
    external_id: senderExternalId,
    display_name: required(values["sender-display-name"], "--sender-display-name"),
    bot: false,
    operator: values.operator,
  };
  const conversation = {
    type: z.enum(["groupChat", "personal", "channel"]).parse(values["conversation-type"]),
    external_id: required(values["conversation-external-id"], "--conversation-external-id"),
    name: required(values["conversation-name"], "--conversation-name"),
  };
  const mentioned = z.enum(["true", "false"]).parse(values.mentioned) === "true";
  const expectations = [
    ...values["expect-present"].map((substring) => ({ kind: "present" as const, substring })),
    ...values["expect-absent"].map((substring) => ({ kind: "absent" as const, substring })),
  ];
  for (const expectation of expectations) z.string().min(1).parse(expectation.substring);
  const messageId = values["external-message-id"];
  if (messageId !== undefined) required(messageId, "--external-message-id");
  const sessionId = sessionFromCaller(rawSession);
  const scenario = { question, sender, conversation, mentioned };
  const reports: ProbeReport[] = [];

  async function runScenario(run: number): Promise<void> {
    const externalMessageId =
      runCount === 1 && messageId !== undefined
        ? messageId
        : `${messageId ?? "option-a-g"}-${randomUUID()}`;
    const observedAt = observedAtOverride ?? new Date().toISOString();
    const signal = AbortSignal.timeout(timeoutMs);
    const startedAt = performance.now();
    const post = async (path: string, body: object, requestSignal = signal): Promise<unknown> => {
      const response = await fetch(new URL(path, values["base-url"]), {
        method: "POST",
        redirect: "error",
        headers: { "x-borg-token": token, "content-type": "application/json" },
        body: JSON.stringify({ tenant: values.tenant, ...body }),
        signal: requestSignal,
      });
      if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
      return response.json();
    };
    const trace = async (since: number, requestSignal = signal) => {
      const url = new URL("/memory/trace", values["base-url"]);
      url.searchParams.set("tenant", values.tenant!);
      url.searchParams.set("since", String(since));
      const response = await fetch(url, {
        headers: { "x-borg-token": token },
        redirect: "error",
        signal: requestSignal,
      });
      if (!response.ok) throw new Error(`/memory/trace: HTTP ${response.status}`);
      return traceSchema.parse(await response.json());
    };
    const preflight = await trace(0);
    if (preflight.option_a_probe !== true)
      throw new Error("The sidecar probe is not enabled for this tenant");
    let since = preflight.nextSince ?? 0;
    const events: z.infer<typeof eventSchema>[] = [];
    const deliveryWaiterWoke = (candidates = events) =>
      candidates.some(
        (event) =>
          event.event === "sidecar.delivery_waiter.woke" &&
          event.wake === "available" &&
          Array.isArray(event.session_ids) &&
          event.session_ids.includes(sessionId),
      );
    let truncated = false;
    let done = false;
    const deliveries: z.infer<typeof deliverySchema>[] = [];
    const deliveryReceivedAt = new Map<string, number>();
    const claimAbort = new AbortController();
    const claimSignal = AbortSignal.any([signal, claimAbort.signal]);
    const cleanupReason = new Error("Probe polling finished");
    // Start the actual delivery long poll before enqueue. A wake trace, rather than
    // a fast response or a terminal lookup, is the evidence that notify woke it.
    const claimTask = (async () => {
      while (!done) {
        const result = z
          .object({ deliveries: z.array(deliverySchema) })
          .parse(
            await post(
              "/memory/agent-deliveries/claim",
              { sidecar_session_ids: [sessionId], wait_ms: 30_000, lease_ms: timeoutMs + 60_000 },
              claimSignal,
            ),
          );
        deliveries.push(...result.deliveries);
        for (const delivery of result.deliveries)
          deliveryReceivedAt.set(delivery.delivery_id, performance.now() - startedAt);
      }
    })();
    const traceTask = (async () => {
      while (!done) {
        const result = await trace(since, claimSignal);
        events.push(...result.events);
        since = result.nextSince ?? since;
        truncated ||= result.truncated === true;
        await delay(500, undefined, { signal: claimSignal });
      }
    })();
    // Observe background failures immediately; propagate them in the foreground.
    let backgroundError: unknown;
    const observePollFailure = (error: unknown) => {
      const intentional =
        claimAbort.signal.aborted &&
        (error === cleanupReason || (error instanceof Error && error.cause === cleanupReason));
      if (!intentional) backgroundError ??= error;
    };
    void claimTask.catch(observePollFailure);
    void traceTask.catch(observePollFailure);
    const check = () => {
      signal.throwIfAborted();
      if (backgroundError !== undefined) throw backgroundError;
    };
    let report: ProbeReport = {
      run,
      scenario,
      tenant: values.tenant,
      sidecar_session_id: sessionId,
      external_message_id: externalMessageId,
      question,
      observed_at: observedAt,
    };
    try {
      const queued = z
        .object({
          status: z.enum(["enqueued", "duplicate"]),
          sidecar_session_id: sessionIdSchema,
          entry_id: streamEntryIdSchema,
        })
        .parse(
          await post("/memory/enqueue", {
            session: rawSession,
            conversation,
            sender,
            text: question,
            external_message_id: externalMessageId,
            observed_at: observedAt,
            flags: { mentioned, quotes_bot: false },
          }),
        );
      if (queued.sidecar_session_id !== sessionId)
        throw new Error("Sidecar session mapping differs from this checkout");
      report.enqueue = queued;
      let terminal: z.infer<typeof terminalSchema>;
      let seenGenerating = false;
      do {
        check();
        terminal = terminalSchema.parse(
          await post("/memory/await-response", {
            sidecar_session_id: sessionId,
            entry_id: queued.entry_id,
            timeout_ms: 30_000,
            seen_generating: seenGenerating,
          }),
        );
        if (terminal.status === "generating") seenGenerating = true;
      } while (terminal.status === "pending" || terminal.status === "generating");
      const terminalReceivedAtMs = performance.now() - startedAt;
      report = {
        ...report,
        terminal,
        terminal_received_after_ms: terminalReceivedAtMs,
        committed_reply: terminal.status === "answered" ? terminal.reply : null,
      };
      const entry = new StreamReader({ dataDir, sessionId }).scanReverse({
        maxEntries: 20_000,
        maxBytes: 64 * 1024 * 1024,
        filter: (candidate) => candidate.id === terminal.terminal_id,
        stop: (matches) => matches.length > 0,
      }).entries[0];
      if (entry?.turn_id === undefined)
        throw new Error("Committed native terminal not found in --data-dir");
      report.turn_id = entry.turn_id;
      let delivery: z.infer<typeof deliverySchema> | undefined;
      while (terminal.status === "answered" && delivery === undefined) {
        check();
        delivery = deliveries.find((row) => row.terminal_entry_id === terminal.terminal_id);
        if (delivery === undefined) await delay(100, undefined, { signal });
      }
      // Preserve receipt measurements even if later trace/capture processing fails.
      report.delivery = delivery ?? null;
      report.delivery_received_after_ms =
        delivery === undefined ? null : deliveryReceivedAt.get(delivery.delivery_id);
      if (
        delivery !== undefined &&
        (terminal.status !== "answered" || delivery.content !== terminal.reply)
      ) {
        throw new Error("Delivery content does not match the committed terminal");
      }
      // Post-response reflection can finish later than the delivery. Wait for its
      // terminal trace as well so per-stage timings cover the whole native turn.
      while (
        !events.some((event) => event.turnId === entry.turn_id && event.event === "turn.terminal")
      ) {
        check();
        await delay(100, undefined, { signal });
      }
      const capturePath = join(dataDir, "captures", "finalizer-contexts.jsonl");
      const finalizers: FinalizerContextCaptureRecord[] = [];
      if (existsSync(capturePath)) {
        const { lines } = openCaptureSnapshot(capturePath);
        for await (const line of lines) {
          if (!line.trim()) continue;
          const record = parseFinalizerContextCaptureRecord(JSON.parse(line));
          if (record.turn_id === entry.turn_id && record.session_id === sessionId)
            finalizers.push(record);
        }
      }
      // Publication now follows run()'s terminal trace. A poll may have seen that
      // trace just before the delivery wake. Read the remaining wake evidence;
      // do not merge this concurrent snapshot into phase/LLM timing events.
      let deliveryTraceEvents: z.infer<typeof eventSchema>[] = [];
      if (delivery !== undefined && !deliveryWaiterWoke()) {
        const latest = await trace(since);
        truncated ||= latest.truncated === true;
        deliveryTraceEvents = latest.events;
      }
      const turnEvents = events.filter((event) => event.turnId === entry.turn_id);
      const outcome = turnEvents.find((event) => event.event === "turn.terminal")?.outcome;
      const finalizerRan =
        finalizers.length > 0 ||
        turnEvents.some(
          (event) =>
            event.event === "llm_call.started" &&
            (event.label === "system_1_finalizer" || event.label === "system_2_finalizer"),
        );
      const earlySuppression =
        terminal.status === "observed" &&
        !finalizerRan &&
        (outcome === "suppressed_generation_gate" || outcome === "suppressed_closure");
      const waiterWoke = deliveryWaiterWoke() || deliveryWaiterWoke(deliveryTraceEvents);
      report = {
        ...report,
        finalizer_ran: finalizerRan,
        capture_status: earlySuppression
          ? "not_applicable_early_suppression"
          : finalizers.length > 0
            ? "captured"
            : "missing",
        early_suppression: earlySuppression ? { outcome, content: entry.content } : null,
        ledger_summary: finalizers.map((record, index) => ({
          finalizer_index: index + 1,
          attempt_kind: record.attempt_kind,
          sections: ledgerRowsSchema.parse(record.evidence_ledger).sections.map((section) => ({
            id: section.id,
            row_count: section.entries.length,
            rows: section.entries.map((row) => ({
              id: row.id,
              is_episode: row.source_type === "episode",
              episode_id:
                row.source_type === "episode"
                  ? (row.state_metadata?.episode_id ??
                    (row.id.startsWith("episode:") ? row.id.slice("episode:".length) : row.id))
                  : null,
            })),
          })),
        })),
        finalizers: finalizers.map((record) => ({
          schema_version: record.schema_version,
          capture_id: record.capture_id,
          captured_at: record.captured_at,
          turn_id: record.turn_id,
          session_id: record.session_id,
          path: record.path,
          attempt_kind: record.attempt_kind,
          configured_surface_variant: record.configured_surface_variant,
          live_surface_variant: record.live_surface_variant,
          fidelity: record.fidelity,
          live_outcome: record.live_outcome,
          ...(values.compact ? {} : { evidence_ledger: record.evidence_ledger }),
          live_request:
            record.live_request === null
              ? null
              : {
                  model: record.live_request.model,
                  tools: record.live_request.tools?.map(({ name }) => ({ name })),
                  ...(values.compact ? {} : { system: record.live_request.system }),
                },
        })),
        delivery_woke_waiter: waiterWoke,
        saw_generating: seenGenerating,
        stage_timings: turnEvents.filter(
          (event) =>
            event.event === "turn_phase.completed" ||
            event.event === "turn_phase.failed" ||
            event.event === "turn.terminal",
        ),
        llm_timings: llmTimings(turnEvents),
        degraded_events: turnEvents.filter(
          (event) => event.event.endsWith(".degraded") || event.event.endsWith(".failed"),
        ),
        trace_truncated: truncated,
        unacknowledged_other_delivery_ids: deliveries
          .filter((row) => row !== delivery)
          .map((row) => row.delivery_id),
      };
      if (delivery !== undefined) {
        // The script is the test consumer. It prints, never posts to Teams. Mark
        // the delivery permanently consumed without claiming a Teams send occurred.
        await post("/memory/agent-deliveries/ack", {
          delivery_id: delivery.delivery_id,
          claim_generation: delivery.claim_generation,
          outcome: "failed_permanent",
          error: "option-a-probe: consumed by local replay; no Teams send",
        });
        report.delivery_ack = "failed_permanent (probe consumed; no Teams send)";
      }
      if (
        (!earlySuppression && finalizers.length === 0) ||
        finalizers.some((record) => !record.fidelity.verified) ||
        truncated
      ) {
        throw new Error(
          "Probe evidence incomplete: inspect capture settings/rotation and trace capacity",
        );
      }
    } catch (error) {
      report.error = error instanceof Error ? error.message : String(error);
      throw error;
    } finally {
      done = true;
      claimAbort.abort(cleanupReason);
      await Promise.allSettled([claimTask, traceTask]);
      const cleanupFailure = backgroundError ?? (signal.aborted ? signal.reason : undefined);
      const propagateCleanupFailure = report.error === undefined && cleanupFailure !== undefined;
      if (propagateCleanupFailure)
        report.error =
          cleanupFailure instanceof Error ? cleanupFailure.message : String(cleanupFailure);
      report.trace_truncated = truncated;
      report.stage_timings ??= events.filter(
        (event) =>
          event.session_id === sessionId &&
          (event.event === "turn_phase.completed" ||
            event.event === "turn_phase.failed" ||
            event.event === "turn.terminal"),
      );
      report.delivery_woke_waiter ??= deliveryWaiterWoke();
      report.expectations = expectations.map(({ kind, substring }) => {
        // Literal evaluation assertions requested by the caller; never used by
        // Borg's interpretation, retrieval, consent or generation paths.
        const hasReply = typeof report.committed_reply === "string";
        const pass =
          hasReply &&
          (kind === "present"
            ? report.committed_reply!.includes(substring)
            : !report.committed_reply!.includes(substring));
        console.error(
          `Run ${run}/${runCount} ${pass ? "PASS" : "FAIL"} expect-${kind} ${JSON.stringify(substring)}${hasReply ? "" : " (no committed reply)"}`,
        );
        return { kind, substring, pass, ...(hasReply ? {} : { reason: "no committed reply" }) };
      });
      console.error(
        `Run ${run}/${runCount}: terminal_received_after_ms=${report.terminal_received_after_ms ?? "n/a"} delivery_received_after_ms=${report.delivery_received_after_ms ?? "n/a"} delivery_woke_waiter=${report.delivery_woke_waiter}`,
      );
      console.error(
        `Run ${run}/${runCount} committed reply: ${JSON.stringify(report.committed_reply ?? null)}`,
      );
      reports.push(report);
      if (propagateCleanupFailure) throw cleanupFailure;
    }
  }

  try {
    for (let run = 1; run <= runCount; run++) await runScenario(run);
  } finally {
    if (reports.length > 0) {
      const aggregate = summarizeProbeLatency(reports, runCount);
      for (const [metric, summary] of Object.entries(aggregate)) {
        console.error(
          `Aggregate ${metric}: median=${summary.median_ms ?? "n/a"} ms p95=${summary.p95_ms ?? "n/a"} ms samples=${summary.count}/${runCount} ${summary.passes_gate ? "PASS" : "FAIL"} (median <= 20000 ms, p95 <= 30000 ms)`,
        );
      }
      console.log(
        JSON.stringify(
          {
            ...(runCount === 1
              ? reports[0]
              : { scenario, tenant: values.tenant, sidecar_session_id: sessionId, runs: reports }),
            requested_runs: runCount,
            completed_runs: reports.filter((report) => report.error === undefined).length,
            aggregate_latency: aggregate,
          },
          null,
          2,
        ),
      );
    }
  }
}

export function summarizeProbeLatency(
  reports: readonly Pick<
    ProbeReport,
    "terminal_received_after_ms" | "delivery_received_after_ms"
  >[],
  requestedRuns = reports.length,
) {
  const summarize = (metric: "terminal_received_after_ms" | "delivery_received_after_ms") => {
    const sorted = reports
      .flatMap((report) => {
        const value = report[metric];
        return typeof value === "number" && Number.isFinite(value) ? [value] : [];
      })
      .sort((a, b) => a - b);
    const count = sorted.length;
    const median =
      count === 0
        ? null
        : (sorted[Math.floor((count - 1) / 2)]! + sorted[Math.floor(count / 2)]!) / 2;
    // Nearest-rank p95 retains the observed slow tail for small latency samples.
    // The similarity-distributions helper instead interpolates quantiles.
    const p95 = count === 0 ? null : sorted[Math.ceil(count * 0.95) - 1]!;
    return {
      count,
      missing_count: requestedRuns - count,
      median_ms: median,
      p95_ms: p95,
      passes_gate:
        count === requestedRuns &&
        median !== null &&
        median <= 20_000 &&
        p95 !== null &&
        p95 <= 30_000,
    };
  };
  return {
    terminal_received_after_ms: summarize("terminal_received_after_ms"),
    delivery_received_after_ms: summarize("delivery_received_after_ms"),
  };
}

function llmTimings(events: z.infer<typeof eventSchema>[]) {
  const pending = new Map<unknown, z.infer<typeof eventSchema>[]>();
  const rows: object[] = [];
  for (const event of events) {
    if (event.event === "llm_call.started") {
      const queue = pending.get(event.label) ?? [];
      queue.push(event);
      pending.set(event.label, queue);
    } else if (event.event === "llm_call.completed") {
      const start = pending.get(event.label)?.shift();
      rows.push({
        label: event.label,
        duration_ms: start === undefined ? null : event.wallMs - start.wallMs,
        usage: event.usage,
        stop_reason: event.stopReason,
      });
    }
  }
  return rows;
}

runAbCliEntrypoint(import.meta.url, runOptionAProbeCli);
