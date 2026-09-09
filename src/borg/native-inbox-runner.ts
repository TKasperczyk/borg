import {
  type ChatResponseCatchUpRunInput,
  type ChatResponseCatchUpRunner,
} from "../cognition/ingestion/chat-response-catch-up-worker.js";
import type { AgentDeliveryRepository } from "../cognition/ingestion/agent-deliveries.js";
import type { TurnOrchestrator } from "../cognition/turn-orchestrator.js";
import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { readStreamEntryAtOffset, type StreamEntry, type StreamWriter } from "../stream/index.js";
import type { SessionId, StreamEntryId } from "../util/ids.js";
import { CognitionError } from "../util/errors.js";
import type { BorgOpenOptions } from "./types.js";

type InboxOptions = NonNullable<BorgOpenOptions["inbox"]>;
const NATIVE_INBOX_COMMIT_EVENT = "native_inbox_turn_committed";

/** Transport projection only: the native orchestrator owns generation and terminal append. */
export class NativeInboxRunner implements ChatResponseCatchUpRunner {
  private readonly runningSessions = new Set<SessionId>();

  constructor(
    private readonly options: {
      turnOrchestrator: Pick<TurnOrchestrator, "run">;
      db: SqliteDatabase;
      dataDir: string;
      createStreamWriter: (sessionId: SessionId) => StreamWriter;
      deliveries: Pick<AgentDeliveryRepository, "create">;
      acceptsSession: (sessionId: SessionId) => boolean;
      onGenerating?: InboxOptions["onGenerating"];
      onTerminalCommitted?: InboxOptions["onTerminalCommitted"];
    },
  ) {}

  private publish(entries: readonly StreamEntry[]): void {
    for (const entry of entries) {
      // HTTP-runner terminals have no turn_id. Never enqueue those or task-event
      // terminals, and never deliver scratch prose or pre-guard token events.
      if (
        entry.turn_id === undefined ||
        entry.response_to?.kind !== "stream_backlog" ||
        !this.options.acceptsSession(entry.session_id) ||
        !this.options.db
          .prepare(
            "SELECT 1 FROM stream_entry_index WHERE entry_id = ? AND active = 1 AND receipt_pending = 0",
          )
          .get(entry.id)
      )
        continue;
      if (entry.kind === "agent_msg" && typeof entry.content === "string") {
        this.options.deliveries.create({
          sessionId: entry.session_id,
          terminalEntryId: entry.id,
          taskId: `native:${entry.id}`,
          content: entry.content,
          createdAt: entry.timestamp,
        });
      }
      this.options.onTerminalCommitted?.(entry);
    }
  }

  private hasCommitMarker(entry: StreamEntry): boolean {
    const markers = this.options.db
      .prepare(
        `
      SELECT byte_offset FROM stream_entry_index
      WHERE session_id = ? AND turn_id = ? AND kind = 'internal_event'
        AND active = 1 AND receipt_pending = 0
    `,
      )
      .all(entry.session_id, entry.turn_id!) as { byte_offset: number }[];
    return markers.some((marker) => {
      const content = readStreamEntryAtOffset({
        dataDir: this.options.dataDir,
        sessionId: entry.session_id,
        byteOffset: marker.byte_offset,
      })?.content;
      return (
        content !== null &&
        typeof content === "object" &&
        !Array.isArray(content) &&
        "event" in content &&
        "terminal_entry_id" in content &&
        content.event === NATIVE_INBOX_COMMIT_EVENT &&
        content.terminal_entry_id === entry.id
      );
    });
  }

  private activeTerminal(
    sessionId: SessionId,
    sourceEntryId: StreamEntryId,
    turnId?: string,
  ): StreamEntry | null {
    const rows = this.options.db
      .prepare(
        `
      SELECT e.byte_offset FROM stream_entry_index e
      WHERE e.session_id = ? AND e.active = 1 AND e.receipt_pending = 0
        AND e.turn_id IS NOT NULL AND e.response_to_kind = 'stream_backlog'
        AND e.kind IN ('agent_msg', 'agent_observed', 'agent_suppressed')
        AND EXISTS (SELECT 1 FROM json_each(e.response_to_source_entry_ids) WHERE value = ?)
        ${turnId === undefined ? "" : "AND e.turn_id = ?"}
      ORDER BY e.byte_offset DESC
    `,
      )
      .all(sessionId, sourceEntryId, ...(turnId === undefined ? [] : [turnId])) as {
      byte_offset: number;
    }[];
    for (const row of rows) {
      const entry = readStreamEntryAtOffset({
        dataDir: this.options.dataDir,
        sessionId,
        byteOffset: row.byte_offset,
      });
      if (entry !== null && (turnId !== undefined || this.hasCommitMarker(entry))) return entry;
    }
    return null;
  }

  // Called at tenant open after index backfill and before each session drain.
  // Only a marker written after successful run() completion authorizes recovery.
  // An active append alone can belong to an interrupted, rollback-capable turn.
  async reconcile(sessionId?: SessionId): Promise<void> {
    const missing = this.options.db
      .prepare(
        `
      SELECT e.session_id, e.byte_offset FROM stream_entry_index e
      LEFT JOIN agent_deliveries d
        ON d.sidecar_session_id = e.session_id AND d.terminal_entry_id = e.entry_id
      WHERE e.kind = 'agent_msg' AND e.active = 1 AND e.receipt_pending = 0
        AND e.turn_id IS NOT NULL AND e.response_to_kind = 'stream_backlog'
        AND d.delivery_id IS NULL
        ${sessionId === undefined ? "" : "AND e.session_id = ?"}
      ORDER BY e.timestamp, e.entry_id
    `,
      )
      .all(...(sessionId === undefined ? [] : [sessionId])) as {
      session_id: SessionId;
      byte_offset: number;
    }[];
    for (const row of missing) {
      if (!this.options.acceptsSession(row.session_id) || this.runningSessions.has(row.session_id))
        continue;
      const entry = readStreamEntryAtOffset({
        dataDir: this.options.dataDir,
        sessionId: row.session_id,
        byteOffset: row.byte_offset,
      });
      if (entry !== null && this.hasCommitMarker(entry)) this.publish([entry]);
    }
  }

  async run(input: ChatResponseCatchUpRunInput): Promise<void> {
    const sourceEntryId = input.inboundBatch.entryIds.at(-1)!;
    const existing = this.activeTerminal(input.sessionId, sourceEntryId);
    if (existing !== null) {
      this.publish([existing]);
      return;
    }
    try {
      this.options.onGenerating?.({
        sessionId: input.sessionId,
        entryIds: input.inboundBatch.entryIds,
      });
    } catch (error) {
      console.error("Native inbox generating observer failed", error);
    }
    this.runningSessions.add(input.sessionId);
    try {
      // Keep the native catch-up invocation, retaining its result to bind the
      // durable transport commit to this exact successful lifecycle.
      const result = await this.options.turnOrchestrator.run({
        sessionId: input.sessionId,
        origin: "user",
        lockMode: "try",
        inboundBatch: input.inboundBatch,
      });
      const committed = this.activeTerminal(input.sessionId, sourceEntryId, result.turn_id);
      if (committed === null)
        throw new CognitionError("Successful native inbox turn has no active terminal");
      const writer = this.options.createStreamWriter(input.sessionId);
      try {
        await writer.append({
          kind: "internal_event",
          turn_id: result.turn_id,
          content: {
            event: NATIVE_INBOX_COMMIT_EVENT,
            terminal_entry_id: committed.id,
          },
        });
      } finally {
        writer.close();
      }
      this.publish([committed]);
    } finally {
      this.runningSessions.delete(input.sessionId);
    }
  }
}
