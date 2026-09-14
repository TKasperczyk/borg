import type { ChatResponseCatchUpObserver } from "../../cognition/ingestion/index.js";
import type { StreamEntry } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { MemoryMaintenanceRunReport } from "../memory-maintenance.js";

import {
  emitObservatoryEvent,
  startObservatoryStage,
  endObservatoryStage,
  reportObservatoryObserverError,
  type ObservatoryObserver,
  type ObservatoryStatus,
} from "./emitter.js";

export function observeMaintenanceReport(
  emitter: ObservatoryObserver,
  report: MemoryMaintenanceRunReport,
): void {
  const running = report.state === "running";
  const status: ObservatoryStatus = report.state === "aborted"
    ? "error" : report.errors_total > 0 ? "degraded" : "ok";
  emitObservatoryEvent(emitter, {
    tenant: report.tenant,
    kind: "job.maintenance",
    phase: running ? "start" : "end",
    span_id: `sp_${report.run_id}`,
    ts: running
      ? report.started_at
      : report.finished_at ?? report.started_at + report.total_duration_ms,
    refs: { run_id: report.run_id },
    ...(running ? {} : {
      duration_ms: report.total_duration_ms,
      status,
    }),
    data: {
      mode: report.mode,
      run_id: report.run_id,
      processes: report.resolved_processes.length > 0
        ? report.resolved_processes : report.requested_processes,
      ...(running ? {} : {
        state: report.state,
        total_duration_ms: report.total_duration_ms,
        errors_total: report.errors_total,
        process_reports: report.processes.map(
          ({ name, status, duration_ms, changes, errors, tokens_used }) =>
            ({ name, status, duration_ms, changes, errors, tokens_used }),
        ),
      }),
    },
  });
}

export class ObservatoryInboxObserver {
  private readonly clock: Clock;
  // One bounded lifecycle map holds attempt correlations and terminal notification dedup.
  private readonly lifecycle = new Map<string, { expiresAt: number; turnId?: string }>();

  constructor(private readonly options: { emitter: ObservatoryObserver; clock?: Clock }) {
    this.clock = options.clock ?? new SystemClock();
  }

  forTenant(tenant: string): ChatResponseCatchUpObserver {
    return {
      onSettle: (event) => emitObservatoryEvent(this.options.emitter, {
        tenant,
        kind: "inbox.settle",
        phase: "point",
        refs: { sidecar_session_id: event.sessionId, entry_id: event.entryId },
        data: {
          reason: event.reason, delay_ms: event.delayMs, bound: event.bound,
          oldest_pending_at: event.oldestPendingAt,
          quiet_window_ms: event.quietWindowMs, max_wait_ms: event.maxWaitMs,
        },
      }),
      onBatch: (event) => {
        const stage = startObservatoryStage(this.options.emitter, {
          tenant,
          kind: "inbox.batch",
          refs: { sidecar_session_id: event.sessionId, entry_ids: event.entryIds },
          data: {
            included_count: event.includedCount, has_more: event.hasMore, watermark: event.watermark,
          },
        });
        return (result, turnId, errorName) => endObservatoryStage(stage, {
          turn_id: turnId,
          status: result.status === "error" ? "error" : result.status === "busy" ? "skipped" : "ok",
          data: { result: result.status, error_name: errorName },
        });
      },
    };
  }

  onAttempt(tenant: string, sessionId: string, entryIds: readonly string[], turnId: string): void {
    for (const entryId of entryIds) {
      this.remember(JSON.stringify([tenant, sessionId, entryId]), turnId);
    }
  }

  onTerminalCommitted(tenant: string, entry: StreamEntry, details?: { sealed: boolean }): void {
    try {
      if (
        entry.response_to?.kind !== "stream_backlog" ||
        (entry.kind !== "agent_msg" && entry.kind !== "agent_observed")
      ) return;
      this.prune();
      const key = JSON.stringify([tenant, "terminal", entry.id]);
      if (this.lifecycle.has(key)) return;
      this.remember(key);
      const sourceIds = entry.response_to.source_entry_ids;
      const turnId = sourceIds
        .map((id) => this.lifecycle.get(JSON.stringify([tenant, entry.session_id, id]))?.turnId)
        .find((id) => id !== undefined);
      const reason =
        typeof entry.content === "object" && entry.content !== null && "reason" in entry.content
          ? entry.content.reason : undefined;
      emitObservatoryEvent(this.options.emitter, {
        tenant,
        kind: "inbox.terminal",
        phase: "point",
        turn_id: turnId,
        refs: { sidecar_session_id: entry.session_id, entry_id: entry.id, entry_ids: sourceIds },
        data: {
          kind: entry.kind,
          reason: typeof reason === "string" ? reason : undefined,
          // Reconciled terminals predate this observer and have no durable seal flag.
          sealed: details?.sealed,
          content_preview: typeof entry.content === "string" ? entry.content : undefined,
        },
      });
    } catch (error) {
      reportObservatoryObserverError("terminal", error);
    }
  }

  private remember(key: string, turnId?: string): void {
    this.prune();
    this.lifecycle.delete(key);
    this.lifecycle.set(key, { turnId, expiresAt: this.clock.now() + 60 * 60_000 });
    while (this.lifecycle.size > 10_000) {
      const oldest = this.lifecycle.keys().next().value;
      if (oldest === undefined) break;
      this.lifecycle.delete(oldest);
    }
  }

  private prune(): void {
    const now = this.clock.now();
    for (const [key, value] of this.lifecycle) {
      if (value.expiresAt > now) break;
      this.lifecycle.delete(key);
    }
  }
}
