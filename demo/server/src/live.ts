import {
  parseSessionId,
  type SessionId,
  type StreamEntry,
  type TurnTraceData,
  type TurnTraceEventName,
  type TurnTracer,
} from "borg";

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

type LiveClient = {
  socket: SocketLike;
  subscribedSessions: Set<SessionId>;
  subscribedGlobal: boolean;
};

type BufferedLiveFrame = {
  frame: LiveFrame;
  ts: number;
};

const RING_BUFFER_MAX = 64;
const RING_BUFFER_MAX_AGE_MS = 60_000;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseMaybeSessionId(value: unknown): SessionId | undefined {
  if (typeof value !== "string") {
    return undefined;
  }

  try {
    return parseSessionId(value);
  } catch {
    return undefined;
  }
}

function firstStreamEntrySessionId(frame: LiveFrame): SessionId | undefined {
  if (frame.type !== "stream:append" || !Array.isArray(frame.entries)) {
    return undefined;
  }

  return parseMaybeSessionId((frame.entries[0] as { session_id?: unknown } | undefined)?.session_id);
}

function frameSessionId(frame: LiveFrame): SessionId | undefined {
  const topLevelSessionId = parseMaybeSessionId(frame.session_id);
  if (topLevelSessionId !== undefined) {
    return topLevelSessionId;
  }

  const dataSessionId = isObject(frame.data) ? parseMaybeSessionId(frame.data.session_id) : undefined;
  if (dataSessionId !== undefined) {
    return dataSessionId;
  }

  return firstStreamEntrySessionId(frame);
}

function parseSubscriptionPayload(raw: unknown): Record<string, unknown> | null {
  if (typeof raw === "string") {
    try {
      const parsed = JSON.parse(raw) as unknown;
      return isObject(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }

  if (raw instanceof ArrayBuffer) {
    return parseSubscriptionPayload(Buffer.from(raw).toString("utf8"));
  }

  if (ArrayBuffer.isView(raw)) {
    return parseSubscriptionPayload(Buffer.from(raw.buffer).toString("utf8"));
  }

  return isObject(raw) ? raw : null;
}

export class LiveBroadcaster {
  private readonly clients = new Map<SocketLike, LiveClient>();
  private readonly sessionBuffers = new Map<SessionId, BufferedLiveFrame[]>();

  constructor(private readonly logger: LoggerLike = console) {}

  add(client: SocketLike): void {
    this.clients.set(client, {
      socket: client,
      subscribedSessions: new Set(),
      subscribedGlobal: true,
    });
  }

  remove(client: SocketLike): void {
    this.clients.delete(client);
  }

  handleSubscriptionMessage(client: SocketLike, raw: unknown): void {
    const state = this.clients.get(client);
    const message = parseSubscriptionPayload(raw);

    if (state === undefined || message === null || typeof message.type !== "string") {
      return;
    }

    if (message.type === "subscribe_global") {
      state.subscribedGlobal = true;
      return;
    }

    if (message.type === "unsubscribe_global") {
      state.subscribedGlobal = false;
      return;
    }

    if (message.type !== "subscribe" && message.type !== "unsubscribe") {
      return;
    }

    const sessionId = parseMaybeSessionId(message.session_id);
    if (sessionId === undefined) {
      return;
    }

    if (message.type === "unsubscribe") {
      state.subscribedSessions.delete(sessionId);
      return;
    }

    if (state.subscribedSessions.has(sessionId)) {
      return;
    }

    state.subscribedSessions.add(sessionId);
    this.flushSessionBuffer(state, sessionId);
  }

  broadcast(frame: LiveFrame): void {
    const deliverToAll = frame.type === "borg:reset";
    const sessionId = deliverToAll ? undefined : frameSessionId(frame);
    if (sessionId !== undefined) {
      this.bufferSessionFrame(sessionId, frame);
    }

    const payload = JSON.stringify(frame);

    for (const client of this.clients.values()) {
      if (!deliverToAll && !this.shouldDeliver(client, sessionId)) {
        continue;
      }

      try {
        client.socket.send(payload);
      } catch {
        this.clients.delete(client.socket);
      }
    }
  }

  clearAllSessionBuffers(): void {
    this.sessionBuffers.clear();
  }

  private shouldDeliver(client: LiveClient, sessionId: SessionId | undefined): boolean {
    return sessionId === undefined
      ? client.subscribedGlobal
      : client.subscribedSessions.has(sessionId);
  }

  private bufferSessionFrame(sessionId: SessionId, frame: LiveFrame): void {
    const now = Date.now();
    const frames = this.sessionBuffers.get(sessionId) ?? [];
    frames.push({ frame, ts: typeof frame.ts === "number" ? frame.ts : now });
    const cutoff = now - RING_BUFFER_MAX_AGE_MS;
    const retained = frames.filter((entry) => entry.ts >= cutoff).slice(-RING_BUFFER_MAX);
    this.sessionBuffers.set(sessionId, retained);
  }

  private flushSessionBuffer(client: LiveClient, sessionId: SessionId): void {
    const now = Date.now();
    const cutoff = now - RING_BUFFER_MAX_AGE_MS;
    const frames = (this.sessionBuffers.get(sessionId) ?? []).filter((entry) => entry.ts >= cutoff);
    this.sessionBuffers.set(sessionId, frames);

    for (const entry of frames) {
      try {
        client.socket.send(JSON.stringify(entry.frame));
      } catch {
        this.clients.delete(client.socket);
        return;
      }
    }
  }

  closeAll(): void {
    for (const client of this.clients.values()) {
      try {
        client.socket.close?.();
      } catch (error) {
        this.logger.error("Live WebSocket close failed", {
          cause: error instanceof Error ? error.message : String(error),
        });
      } finally {
        this.clients.delete(client.socket);
      }
    }
  }

  streamAppend(entries: readonly StreamEntry[]): void {
    const sessionId = entries[0]?.session_id;
    this.broadcast({
      type: "stream:append",
      ts: Date.now(),
      ...(sessionId === undefined ? {} : { session_id: sessionId }),
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

    if (
      event === "turn.token" &&
      turnId !== null &&
      typeof data.phase === "string" &&
      typeof data.chunk_text === "string" &&
      typeof data.sequence === "number"
    ) {
      this.broadcaster.broadcast({
        type: "turn:token",
        ts: Date.now(),
        turn_id: turnId,
        ...(typeof data.session_id === "string" ? { session_id: data.session_id } : {}),
        phase: data.phase,
        chunk_text: data.chunk_text,
        sequence: data.sequence,
      });
      return;
    }

    if (
      event === "turn.token.flush" &&
      turnId !== null &&
      typeof data.phase === "string" &&
      typeof data.full_text === "string"
    ) {
      this.broadcaster.broadcast({
        type: "turn:token:flush",
        ts: Date.now(),
        turn_id: turnId,
        ...(typeof data.session_id === "string" ? { session_id: data.session_id } : {}),
        phase: data.phase,
        full_text: data.full_text,
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
        ...(typeof data.session_id === "string" ? { session_id: data.session_id } : {}),
        ledger,
      });
      return;
    }

    if (
      event === "deliberation.path.completed" &&
      turnId !== null &&
      (data.path === "system_1" || data.path === "system_2")
    ) {
      this.broadcaster.broadcast({
        type: "turn:delib_path",
        ts: Date.now(),
        turn_id: turnId,
        ...(typeof data.session_id === "string" ? { session_id: data.session_id } : {}),
        path: data.path,
      });
      return;
    }

    if (event === "commitment_guard.regeneration_requested" && turnId !== null) {
      // Each regeneration_requested is a finalizer re-attempt. Increment in the
      // UI; first attempt is implicit (no event fires for the initial call).
      this.broadcaster.broadcast({
        type: "turn:final_attempt",
        ts: Date.now(),
        turn_id: turnId,
        ...(typeof data.session_id === "string" ? { session_id: data.session_id } : {}),
        attempt: 2,
      });
      return;
    }

    if (event === "offline_process.started" && typeof data.process_name === "string") {
      this.broadcaster.broadcast({
        type: "dream:process:started",
        ts: Date.now(),
        process: data.process_name,
        run_id: typeof data.turnId === "string" ? data.turnId : null,
        phase: data.phase === "plan" || data.phase === "apply" ? data.phase : "apply",
      });
      return;
    }

    if (event === "offline_process.completed" && typeof data.process_name === "string") {
      this.broadcaster.broadcast({
        type: "dream:process:completed",
        ts: Date.now(),
        process: data.process_name,
        run_id: typeof data.turnId === "string" ? data.turnId : null,
        phase: data.phase === "plan" || data.phase === "apply" ? data.phase : "apply",
        duration_ms: typeof data.duration_ms === "number" ? data.duration_ms : undefined,
        errors: typeof data.errors === "number" ? data.errors : 0,
        candidates_accepted:
          typeof data.candidates_accepted === "number" ? data.candidates_accepted : 0,
      });
      return;
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
