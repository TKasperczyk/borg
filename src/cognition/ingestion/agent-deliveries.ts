import { z } from "zod";

import type { Migration, SqliteDatabase } from "../../storage/sqlite/index.js";
import type { Clock } from "../../util/clock.js";
import { createIdHelpers, type SessionId, type StreamEntryId } from "../../util/ids.js";

const deliveryIdHelpers = createIdHelpers<"AgentDeliveryId">("delivery");
export const agentDeliveryIdSchema = z.string().refine(deliveryIdHelpers.is);
export const agentDeliveryAckSchema = z.object({
  delivery_id: agentDeliveryIdSchema,
  claim_generation: z.number().int().positive(),
  outcome: z.enum(["sent", "failed_retryable", "failed_permanent"]),
  teams_message_id: z.string().min(1).optional(),
  error: z.string().optional(),
});
export type AgentDeliveryAck = z.infer<typeof agentDeliveryAckSchema>;
export type AgentDeliveryState = "pending" | "leased" | "sent" | "failed";
export type AgentDelivery = {
  delivery_id: AgentDeliveryAck["delivery_id"];
  claim_generation: number;
  sidecar_session_id: SessionId;
  terminal_entry_id: StreamEntryId;
  task_id: string;
  content: string;
  created_at: string;
};
type DeliveryRow = Omit<AgentDelivery, "created_at" | "claim_generation"> & {
  created_at: number;
  state: AgentDeliveryState;
  lease_until: number | null;
  attempts: number;
  teams_message_id: string | null;
  last_error: string | null;
};

// This table lives in the tenant's own borg.db. Never share a repository across tenants.
export const agentDeliveryMigrations: Migration[] = [
  {
    id: 1,
    name: "agent_deliveries",
    up: `CREATE TABLE agent_deliveries (
    delivery_id TEXT PRIMARY KEY,
    sidecar_session_id TEXT NOT NULL,
    terminal_entry_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'leased', 'sent', 'failed')),
    lease_until INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    teams_message_id TEXT,
    last_error TEXT,
    UNIQUE (sidecar_session_id, terminal_entry_id)
  );
  CREATE INDEX agent_deliveries_claim ON agent_deliveries (sidecar_session_id, state, created_at);
  CREATE INDEX agent_deliveries_lease ON agent_deliveries (state, lease_until);`,
  },
  {
    id: 2,
    name: "agent_delivery_ack_receipts",
    up: `CREATE TABLE agent_delivery_ack_receipts (
      delivery_id TEXT NOT NULL REFERENCES agent_deliveries(delivery_id),
      claim_generation INTEGER NOT NULL,
      outcome TEXT NOT NULL,
      teams_message_id TEXT,
      error TEXT,
      acknowledged_at INTEGER NOT NULL,
      applied INTEGER NOT NULL,
      PRIMARY KEY (delivery_id, claim_generation)
    );`,
  },
];

function delivery(row: DeliveryRow): AgentDelivery {
  return {
    delivery_id: row.delivery_id,
    claim_generation: row.attempts,
    sidecar_session_id: row.sidecar_session_id,
    terminal_entry_id: row.terminal_entry_id,
    task_id: row.task_id,
    content: row.content,
    created_at: new Date(row.created_at).toISOString(),
  };
}

export class AgentDeliveryRepository {
  constructor(
    private readonly options: {
      db: SqliteDatabase;
      clock: Clock;
      onAvailable?: (sessionId: SessionId) => void;
    },
  ) {}

  hasTerminal(sessionId: SessionId, terminalId: StreamEntryId): boolean {
    return (
      this.options.db
        .prepare(
          "SELECT 1 FROM agent_deliveries WHERE sidecar_session_id = ? AND terminal_entry_id = ?",
        )
        .get(sessionId, terminalId) !== undefined
    );
  }

  create(input: {
    sessionId: SessionId;
    terminalEntryId: StreamEntryId;
    taskId: string;
    content: string;
    createdAt: number;
  }): void {
    const result = this.options.db
      .prepare(
        `
      INSERT INTO agent_deliveries
        (delivery_id, sidecar_session_id, terminal_entry_id, task_id, content, created_at, state)
      VALUES (?, ?, ?, ?, ?, ?, 'pending')
      ON CONFLICT (sidecar_session_id, terminal_entry_id) DO NOTHING
    `,
      )
      .run(
        deliveryIdHelpers.create(),
        input.sessionId,
        input.terminalEntryId,
        input.taskId,
        input.content,
        input.createdAt,
      );
    if (result.changes > 0) this.notify(input.sessionId);
  }

  claim(input: { sessionIds: readonly SessionId[]; leaseMs: number }): {
    deliveries: AgentDelivery[];
    nextLeaseUntil: number | null;
  } {
    const sessionIds = [...new Set(input.sessionIds)];
    if (sessionIds.length === 0) return { deliveries: [], nextLeaseUntil: null };
    const db = this.options.db;
    const placeholders = sessionIds.map(() => "?").join(",");
    const now = this.options.clock.now();
    return db
      .transaction(() => {
        db.prepare(
          `UPDATE agent_deliveries SET state = 'pending', lease_until = NULL
        WHERE sidecar_session_id IN (${placeholders}) AND state = 'leased' AND lease_until <= ?`,
        ).run(...sessionIds, now);
        const rows = db
          .prepare(
            `SELECT * FROM agent_deliveries
        WHERE sidecar_session_id IN (${placeholders}) AND state = 'pending'
        ORDER BY created_at, rowid`,
          )
          .all(...sessionIds) as DeliveryRow[];
        const lease = db.prepare(`UPDATE agent_deliveries SET state = 'leased',
        lease_until = ?, attempts = attempts + 1 WHERE delivery_id = ? AND state = 'pending'`);
        for (const row of rows) {
          lease.run(now + input.leaseMs, row.delivery_id);
          row.attempts += 1;
        }
        const next = db
          .prepare(
            `SELECT MIN(lease_until) AS next FROM agent_deliveries
        WHERE sidecar_session_id IN (${placeholders}) AND state = 'leased'`,
          )
          .get(...sessionIds) as { next: number | null };
        return { deliveries: rows.map(delivery), nextLeaseUntil: next.next };
      })
      .immediate();
  }

  ack(input: AgentDeliveryAck): "acknowledged" | null {
    const result = this.options.db
      .transaction(() => {
        const row = this.options.db
          .prepare("SELECT * FROM agent_deliveries WHERE delivery_id = ?")
          .get(input.delivery_id) as DeliveryRow | undefined;
        if (row === undefined || input.claim_generation > row.attempts) return null;
        const receipt = this.options.db
          .prepare(
            `SELECT 1 FROM agent_delivery_ack_receipts
          WHERE delivery_id = ? AND claim_generation = ?`,
          )
          .get(input.delivery_id, input.claim_generation);
        if (receipt !== undefined) return { notify: false, sessionId: row.sidecar_session_id };
        const now = this.options.clock.now();
        const applied =
          row.state === "leased" &&
          row.attempts === input.claim_generation &&
          row.lease_until !== null &&
          row.lease_until > now;
        this.options.db
          .prepare(
            `INSERT INTO agent_delivery_ack_receipts
          (delivery_id, claim_generation, outcome, teams_message_id, error, acknowledged_at, applied)
          VALUES (?, ?, ?, ?, ?, ?, ?)`,
          )
          .run(
            input.delivery_id,
            input.claim_generation,
            input.outcome,
            input.teams_message_id ?? null,
            input.error ?? null,
            now,
            applied ? 1 : 0,
          );
        if (!applied) return { notify: false, sessionId: row.sidecar_session_id };
        const state: AgentDeliveryState =
          input.outcome === "sent"
            ? "sent"
            : input.outcome === "failed_permanent"
              ? "failed"
              : "pending";
        this.options.db
          .prepare(
            `UPDATE agent_deliveries SET state = ?, lease_until = NULL,
        teams_message_id = ?, last_error = ? WHERE delivery_id = ?`,
          )
          .run(
            state,
            input.teams_message_id ?? row.teams_message_id,
            input.error ?? null,
            input.delivery_id,
          );
        return { notify: state === "pending", sessionId: row.sidecar_session_id };
      })
      .immediate();
    if (result?.notify) this.notify(result.sessionId);
    return result === null ? null : "acknowledged";
  }

  private notify(sessionId: SessionId): void {
    try {
      this.options.onAvailable?.(sessionId);
    } catch (error) {
      console.error("Agent delivery availability observer failed", error);
    }
  }
}
