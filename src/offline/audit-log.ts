import { z } from "zod";

import { SqliteDatabase } from "../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { StorageError } from "../util/errors.js";
import { serializeJsonValue } from "../util/json-value.js";
import {
  maintenanceRunIdHelpers,
  parseAuditId,
  parseMaintenanceRunId,
  type AuditId,
  type MaintenanceRunId,
} from "../util/ids.js";

import { OFFLINE_PROCESS_NAMES, type OfflineProcessName } from "./types.js";

const offlineProcessNameSchema = z.enum(OFFLINE_PROCESS_NAMES);

export const maintenanceAuditSchema = z.object({
  id: z
    .number()
    .int()
    .positive()
    .transform((value) => parseAuditId(value)),
  run_id: z
    .string()
    .refine((value) => maintenanceRunIdHelpers.is(value), {
      message: "Invalid maintenance run id",
    })
    .transform((value) => parseMaintenanceRunId(value)),
  process: offlineProcessNameSchema,
  action: z.string().min(1),
  targets: z.record(z.string(), z.unknown()),
  reversal: z.record(z.string(), z.unknown()),
  applied_at: z.number().finite(),
  reverted_at: z.number().finite().nullable(),
  reverted_by: z.string().min(1).nullable(),
});

export type MaintenanceAuditRecord = z.infer<typeof maintenanceAuditSchema>;

export type MaintenanceAuditRecordInput = {
  run_id: MaintenanceRunId;
  process: OfflineProcessName;
  action: string;
  targets: Record<string, unknown>;
  reversal: Record<string, unknown>;
};

export type Reverser = (input: {
  audit: MaintenanceAuditRecord;
  targets: Record<string, unknown>;
  reversal: Record<string, unknown>;
}) => Promise<void> | void;

function parseJsonRecord(value: string, label: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value) as unknown;

    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new TypeError(`${label} must be an object`);
    }

    return parsed as Record<string, unknown>;
  } catch (error) {
    throw new StorageError(`Invalid maintenance audit ${label}`, {
      cause: error,
      code: "MAINTENANCE_AUDIT_INVALID",
    });
  }
}

function mapAuditRow(row: Record<string, unknown>): MaintenanceAuditRecord {
  const parsed = maintenanceAuditSchema.safeParse({
    id: Number(row.id),
    run_id: row.run_id,
    process: row.process,
    action: row.action,
    targets: parseJsonRecord(String(row.targets ?? "{}"), "targets"),
    reversal: parseJsonRecord(String(row.reversal ?? "{}"), "reversal"),
    applied_at: Number(row.applied_at),
    reverted_at:
      row.reverted_at === null || row.reverted_at === undefined ? null : Number(row.reverted_at),
    reverted_by:
      row.reverted_by === null || row.reverted_by === undefined ? null : String(row.reverted_by),
  });

  if (!parsed.success) {
    throw new StorageError("Maintenance audit row failed validation", {
      cause: parsed.error,
      code: "MAINTENANCE_AUDIT_INVALID",
    });
  }

  return parsed.data;
}

export class ReverserRegistry {
  private readonly reversers = new Map<string, Reverser>();

  register(process: OfflineProcessName, action: string, reverser: Reverser): void {
    this.reversers.set(`${process}:${action}`, reverser);
  }

  get(process: OfflineProcessName, action: string): Reverser | undefined {
    return this.reversers.get(`${process}:${action}`);
  }
}

export type AuditLogOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  registry?: ReverserRegistry;
};

export class AuditLog {
  private readonly clock: Clock;
  readonly registry: ReverserRegistry;

  constructor(private readonly options: AuditLogOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.registry = options.registry ?? new ReverserRegistry();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  record(input: MaintenanceAuditRecordInput): MaintenanceAuditRecord {
    const process = offlineProcessNameSchema.parse(input.process);
    const appliedAt = this.clock.now();
    const result = this.db
      .prepare(
        `
          INSERT INTO maintenance_audit (
            run_id, process, action, targets, reversal, applied_at, reverted_at, reverted_by
          ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
        `,
      )
      .run(
        input.run_id,
        process,
        input.action,
        serializeJsonValue(input.targets),
        serializeJsonValue(input.reversal),
        appliedAt,
      );

    const row = this.db
      .prepare("SELECT * FROM maintenance_audit WHERE id = ?")
      .get(result.lastInsertRowid) as Record<string, unknown> | undefined;

    if (row === undefined) {
      throw new StorageError("Failed to read back maintenance audit row", {
        code: "MAINTENANCE_AUDIT_INSERT_FAILED",
      });
    }

    return mapAuditRow(row);
  }

  list(
    options: {
      run_id?: MaintenanceRunId;
      process?: OfflineProcessName;
      reverted?: boolean;
    } = {},
  ): MaintenanceAuditRecord[] {
    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.run_id !== undefined) {
      filters.push("run_id = ?");
      values.push(options.run_id);
    }

    if (options.process !== undefined) {
      filters.push("process = ?");
      values.push(offlineProcessNameSchema.parse(options.process));
    }

    if (options.reverted === true) {
      filters.push("reverted_at IS NOT NULL");
    } else if (options.reverted === false) {
      filters.push("reverted_at IS NULL");
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM maintenance_audit
          ${whereClause}
          ORDER BY applied_at DESC, id DESC
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map((row) => mapAuditRow(row));
  }

  get(id: AuditId): MaintenanceAuditRecord | null {
    const row = this.db.prepare("SELECT * FROM maintenance_audit WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapAuditRow(row);
  }

  async revert(auditId: AuditId, revertedBy = "manual"): Promise<MaintenanceAuditRecord | null> {
    const audit = this.get(auditId);

    if (audit === null) {
      return null;
    }

    if (audit.reverted_at !== null) {
      return audit;
    }

    const reverser = this.registry.get(audit.process, audit.action);

    if (reverser === undefined) {
      throw new StorageError(`No reverser registered for ${audit.process}:${audit.action}`, {
        code: "MAINTENANCE_REVERSER_MISSING",
      });
    }

    const revertedAt = this.clock.now();

    this.db.exec("BEGIN IMMEDIATE");
    try {
      await reverser({
        audit,
        targets: audit.targets,
        reversal: audit.reversal,
      });

      this.db
        .prepare("UPDATE maintenance_audit SET reverted_at = ?, reverted_by = ? WHERE id = ?")
        .run(revertedAt, revertedBy, auditId);
      this.db.exec("COMMIT");
    } catch (error) {
      try {
        this.db.exec("ROLLBACK");
      } catch {
        // Preserve the original failure.
      }
      throw error;
    }

    return {
      ...audit,
      reverted_at: revertedAt,
      reverted_by: revertedBy,
    };
  }
}
