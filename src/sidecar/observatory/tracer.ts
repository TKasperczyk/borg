import { PAYLOAD_GATED_TRACE_KEYS, type TurnTracer } from "../../tracing/tracer.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { clampPositiveIntegerOrFallback } from "../../util/math.js";

import {
  reportObservatoryObserverError,
  type ObservatoryEmitter,
  type ObservatoryStatus,
} from "./emitter.js";

export type ObservatoryTracerOptions = {
  emitter: Pick<ObservatoryEmitter, "emitLazy">;
  includePayloads: boolean;
  clock?: Clock;
  maxMappings?: number;
  mappingTtlMs?: number;
};

export class ObservatoryTracerSink {
  private readonly clock: Clock;
  private readonly maxMappings: number;
  private readonly mappingTtlMs: number;
  private readonly mappings = new Map<string, { turnId: string; expiresAt: number }>();

  constructor(private readonly options: ObservatoryTracerOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.maxMappings = clampPositiveIntegerOrFallback(options.maxMappings, 10_000);
    this.mappingTtlMs = clampPositiveIntegerOrFallback(options.mappingTtlMs, 60 * 60_000);
  }

  register(tenant: string, traceTurnId: string, turnId: string | undefined): void {
    try {
      this.prune();
      const key = JSON.stringify([tenant, traceTurnId]);
      this.mappings.delete(key);
      if (turnId === undefined) return;
      this.mappings.set(key, { turnId, expiresAt: this.clock.now() + this.mappingTtlMs });
      while (this.mappings.size > this.maxMappings) {
        const oldest = this.mappings.keys().next().value;
        if (oldest === undefined) break;
        this.mappings.delete(oldest);
      }
    } catch (error) {
      reportObservatoryObserverError("trace.register", error);
    }
  }

  tracerFor(tenant: string): TurnTracer {
    return {
      enabled: true,
      includePayloads: this.options.includePayloads,
      emit: (event, data) => {
        try {
          // The emitter checks capacity before this factory reads or copies any
          // trace fields. Its clamp pass applies the observatory-only policy.
          this.options.emitter.emitLazy(() => {
            this.prune();
            const lane = event === "retrieval.lane_settled";
            const laneStatus: ObservatoryStatus = data.outcome === "rejected" ? "error" : "ok";
            // Correlation keys normally contain short generated IDs. Skip lookup
            // for oversized IDs instead of serializing unbounded strings here.
            const turnId = tenant.length <= 4096 && data.turnId.length <= 4096
              ? this.mappings.get(JSON.stringify([tenant, data.turnId]))?.turnId
              : undefined;
            return {
              event: {
                tenant,
                ts: this.clock.now(),
                kind: lane ? "recall.lane" : `trace.${event}`,
                phase: lane ? "end" : "point",
                ...(lane
                  ? {
                      duration_ms: typeof data.duration_ms === "number" ? data.duration_ms : 0,
                      status: laneStatus,
                    }
                  : {}),
                turn_id: turnId,
                refs: typeof data.session_id === "string"
                  ? { sidecar_session_id: data.session_id }
                  : undefined,
                data,
              },
              traceTurnId: data.turnId,
              filterDataField: (key: string, parent: Readonly<Record<string, unknown>>, depth: number): boolean => {
                if (depth === 0 && key === "turnId") return false;
                if (this.options.includePayloads) return true;
                // entityTerms is intentionally local: legacy ring/JSONL output
                // keeps it, while observatory also redacts nested occurrences.
                return key !== "entityTerms" && !PAYLOAD_GATED_TRACE_KEYS.has(key) &&
                  (key !== "spans" || parent.reason === "mixed_closure_observed");
              },
            };
          });
        } catch (error) {
          reportObservatoryObserverError("trace.emit", error);
        }
      },
    };
  }

  private prune(): void {
    const now = this.clock.now();
    for (const [key, value] of this.mappings) {
      if (value.expiresAt > now) break;
      this.mappings.delete(key);
    }
  }
}
