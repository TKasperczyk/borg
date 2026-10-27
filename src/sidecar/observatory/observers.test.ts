import { describe, expect, it, vi } from "vitest";

import type { StreamEntry } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { createMaintenanceRunId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import type { MemoryMaintenanceRunReport } from "../memory-maintenance.js";
import { ObservatoryEmitter, type ObservatoryEvent } from "./emitter.js";
import { ObservatoryInboxObserver, observeMaintenanceReport } from "./observers.js";

function fixture() {
  const clock = new ManualClock(1000);
  const events: ObservatoryEvent[] = [];
  const fetchFn = vi.fn<typeof fetch>(async (_url, init) => {
    events.push(...(JSON.parse(String(init?.body)) as { events: ObservatoryEvent[] }).events);
    return new Response(null, { status: 202 });
  });
  const emitter = new ObservatoryEmitter({
    config: {
      enabled: true, baseUrl: "http://observatory", token: "secret",
      flushMs: 250, maxBatch: 100, queueCap: 5000, timeoutMs: 2000,
    },
    clock, fetchFn,
  });
  return { clock, events, emitter, inbox: new ObservatoryInboxObserver({ emitter, clock }) };
}

describe("observatory domain observers", () => {
  it("joins batch ends and deduplicated terminal commits to the runner attempt", async () => {
    const { emitter, inbox, events } = fixture();
    const sessionId = createSessionId();
    const sourceId = createStreamEntryId();
    const observer = inbox.forTenant("acme");
    const end = observer.onBatch?.({
      sessionId, entryIds: [sourceId], includedCount: 1, hasMore: false, watermark: null,
    });
    inbox.onAttempt("acme", sessionId, [sourceId], "turn_attempt");
    const terminal: StreamEntry = {
      id: createStreamEntryId(), session_id: sessionId, timestamp: 1001, kind: "agent_observed",
      content: { reason: "policy decision" }, sender_entity_id: null, reply_target_entity_id: null,
      compressed: false,
      response_to: {
        kind: "stream_backlog", from_cursor_exclusive: null,
        through_cursor_inclusive: { entryId: sourceId, ts: 1000 }, source_entry_ids: [sourceId], count: 1,
      },
    };
    inbox.onTerminalCommitted("acme", terminal, { sealed: true });
    inbox.onTerminalCommitted("acme", terminal);
    end?.({ sessionId, status: "drained", drained: 1, hasMore: false }, "turn_attempt");
    await emitter.shutdown();
    expect(events.map((event) => event.kind)).toEqual(["inbox.batch", "inbox.terminal", "inbox.batch"]);
    expect(events[1]).toMatchObject({
      turn_id: "turn_attempt", refs: { entry_ids: [sourceId] },
      data: { kind: "agent_observed", reason: "policy decision", sealed: true },
    });
    expect(events[2]).toMatchObject({ phase: "end", span_id: events[0]?.span_id, turn_id: "turn_attempt", data: { result: "drained" } });
  });

  it("projects full maintenance reports, including aborts, with stable spans", async () => {
    const { emitter, events } = fixture();
    const report: MemoryMaintenanceRunReport = {
      evt: "memory_maintenance", tenant: "acme", mode: "heavy", run_id: createMaintenanceRunId(),
      dryRun: false, state: "running", started_at: 1000, finished_at: null,
      requested_processes: ["consolidator"], resolved_processes: [], budget: null, budget_remaining: null,
      processes: [{ name: "consolidator", status: "pending", changes: 0, errors: 0, tokens_used: 0, budget_exhausted: false, duration_ms: 0, error_details: [] }],
      storage_optimize: { status: "pending" }, total_duration_ms: 0, errors_total: 0,
    };
    observeMaintenanceReport(emitter, report);
    observeMaintenanceReport(emitter, {
      ...report, state: "aborted", finished_at: 2000, total_duration_ms: 1000, errors_total: 1,
      processes: [{ ...report.processes[0]!, status: "completed", duration_ms: 900, errors: 1 }],
    });
    await emitter.shutdown();
    expect(events[0]).toMatchObject({ phase: "start", ts: 1000, data: { mode: "heavy", processes: ["consolidator"] } });
    expect(events[1]).toMatchObject({
      phase: "end", span_id: events[0]?.span_id, duration_ms: 1000, status: "error",
      data: { state: "aborted", total_duration_ms: 1000, errors_total: 1,
        process_reports: [{ name: "consolidator", status: "completed", duration_ms: 900, changes: 0, errors: 1, tokens_used: 0 }] },
    });
  });
});
