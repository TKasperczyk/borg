import type { StreamEntry, TurnTraceData, TurnTraceEventName, TurnTracer } from "borg";

type SocketLike = {
  send(data: string): void;
  close?: () => void;
};

type LoggerLike = Pick<Console, "error">;

export type LiveFrame = {
  type: string;
  ts: number;
  [key: string]: unknown;
};

export class LiveBroadcaster {
  private readonly clients = new Set<SocketLike>();

  constructor(private readonly logger: LoggerLike = console) {}

  add(client: SocketLike): void {
    this.clients.add(client);
  }

  remove(client: SocketLike): void {
    this.clients.delete(client);
  }

  broadcast(frame: LiveFrame): void {
    const payload = JSON.stringify(frame);

    for (const client of this.clients) {
      try {
        client.send(payload);
      } catch {
        this.clients.delete(client);
      }
    }
  }

  closeAll(): void {
    for (const client of this.clients) {
      try {
        client.close?.();
      } catch (error) {
        this.logger.error("Live WebSocket close failed", {
          cause: error instanceof Error ? error.message : String(error),
        });
      } finally {
        this.clients.delete(client);
      }
    }
  }

  streamAppend(entries: readonly StreamEntry[]): void {
    this.broadcast({
      type: "stream:append",
      ts: Date.now(),
      entries,
    });
  }
}

export class WsBridgeTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = true;

  constructor(
    private readonly broadcaster: LiveBroadcaster,
    private readonly ledgerCache: Map<string, unknown>,
  ) {}

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    const turnId =
      typeof data.turn_id === "string"
        ? data.turn_id
        : typeof data.turnId === "string"
          ? data.turnId
          : null;

    if (event === "turn_phase.started") {
      this.broadcaster.broadcast({
        type: "turn:phase:started",
        ts: Date.now(),
        event,
        data,
      });
      return;
    }

    if (event === "turn_phase.completed") {
      this.broadcaster.broadcast({
        type: "turn:phase:completed",
        ts: Date.now(),
        event,
        data,
      });
      return;
    }

    if (event === "turn_phase.failed") {
      this.broadcaster.broadcast({
        type: "turn:phase:failed",
        ts: Date.now(),
        event,
        data,
      });
      return;
    }

    if (event === "turn.terminal") {
      this.broadcaster.broadcast({
        type: "turn:terminal",
        ts: Date.now(),
        event,
        data,
      });
      return;
    }

    if (event === "evidence_ledger.built" && turnId !== null) {
      const ledger = data.ledger ?? null;
      this.ledgerCache.set(turnId, ledger);
      this.broadcaster.broadcast({
        type: "evidence_ledger:built",
        ts: Date.now(),
        turn_id: turnId,
        ledger,
      });
    }
  }
}

export type LiveBridge = {
  broadcaster: LiveBroadcaster;
  tracer: TurnTracer;
  ledgerCache: Map<string, unknown>;
  onStreamAppend(entries: readonly StreamEntry[]): void;
};

export function createLiveBridge(): LiveBridge {
  const broadcaster = new LiveBroadcaster();
  const ledgerCache = new Map<string, unknown>();

  return {
    broadcaster,
    tracer: new WsBridgeTracer(broadcaster, ledgerCache),
    ledgerCache,
    onStreamAppend: (entries) => broadcaster.streamAppend(entries),
  };
}
