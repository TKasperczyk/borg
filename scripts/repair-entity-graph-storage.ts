import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

import { connect, type Connection } from "@lancedb/lancedb";
import { LanceDbTable } from "../src/storage/lancedb/index.js";
import { SqliteDatabase, SqliteRawDatabase } from "../src/storage/sqlite/index.js";
import { StorageError } from "../src/util/errors.js";
import {
  episodeParticipantEntityId,
  episodeParticipantEntityIdTerm,
} from "../src/memory/episodic/participant-terms.js";
import { parseEntityId } from "../src/util/ids.js";

export type Row = Record<string, unknown>;
export type SqlChange = {
  table: string;
  rowid: number;
  before: Row | null;
  after: Row | null;
  reason: string;
};
export type LanceChange = { table: string; id: string; before: Row; after: Row; reembed?: boolean };

export function fail(message: string): never {
  throw new StorageError(message, { code: "ENTITY_GRAPH_REPAIR_INVALID" });
}

export function quoteIdentifier(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

export function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export function openRepairDatabase(dataDir: string, readOnly = true): SqliteDatabase {
  const path = join(dataDir, "borg.db");
  if (!existsSync(path)) fail(`No database at ${path}`);
  const db = new SqliteDatabase(
    new SqliteRawDatabase(
      new DatabaseSync(path, {
        readOnly,
        enableDoubleQuotedStringLiterals: true,
      }),
    ),
  );
  db.pragma("busy_timeout = 5000");
  db.pragma("foreign_keys = ON");
  if (readOnly) db.pragma("query_only = ON");
  return db;
}

/** Only exact machine handles / JSON structure. Never substitute inside prose. */
export function remap(value: unknown, ids: ReadonlyMap<string, string>): unknown {
  if (ArrayBuffer.isView(value)) return value;
  if (typeof value === "string") {
    const direct = ids.get(value);
    if (direct !== undefined) return direct;
    for (const prefix of ["semantic_node:", "semantic_edge:"]) {
      if (value.startsWith(prefix) && ids.has(value.slice(prefix.length)))
        return prefix + ids.get(value.slice(prefix.length));
    }
    const participant = episodeParticipantEntityId(value);
    if (participant !== null && ids.has(participant)) {
      return episodeParticipantEntityIdTerm(parseEntityId(ids.get(participant)!));
    }
    // Stored JSON is often nested inside SQL text or stream metadata.
    if (
      (value.startsWith("{") || value.startsWith("[")) &&
      [...ids.keys()].some((id) => value.includes(id))
    ) {
      let parsed: unknown;
      try {
        parsed = JSON.parse(value);
      } catch {
        return value;
      }
      const next = remap(parsed, ids);
      return same(parsed, next) ? value : JSON.stringify(next);
    }
    return value;
  }
  if (Array.isArray(value)) return value.map((item) => remap(item, ids));
  if (value !== null && typeof value === "object") {
    const result: Row = {};
    for (const [key, item] of Object.entries(value)) {
      const nextKey = String(remap(key, ids));
      // Stream text remains source evidence even when an author pasted JSON.
      const next = key === "content" && typeof item === "string" ? item : remap(item, ids);
      if (Object.hasOwn(result, nextKey) && !same(result[nextKey], next)) {
        // Recall suppression expiries retain the later turn, as in open-question reparenting.
        if (key.startsWith("semantic_node:") || key.startsWith("semantic_edge:")) {
          if (typeof next === "number" && typeof result[nextKey] === "number") {
            result[nextKey] = Math.max(Number(result[nextKey]), next);
            continue;
          }
        }
        fail(`Conflicting JSON map keys after remapping ${key}`);
      }
      result[nextKey] = next;
    }
    return result;
  }
  return value;
}

export function sqlInventory(
  db: SqliteDatabase,
  onlyTables?: readonly string[],
): Map<string, Array<{ rowid: number; row: Row }>> {
  const tables = db
    .prepare(
      "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
    )
    .all() as Array<{ name: string }>;
  return new Map(
    tables
      .filter(({ name }) => onlyTables === undefined || onlyTables.includes(name))
      .map(({ name }) => [
        name,
        (
          db
            .prepare(`SELECT rowid AS __repair_rowid, * FROM ${quoteIdentifier(name)}`)
            .all() as Row[]
        ).map(({ __repair_rowid, ...row }) => ({ rowid: Number(__repair_rowid), row })),
      ]),
  );
}

export function changedFields(before: Row, after: Row): Row {
  return Object.fromEntries(
    Object.entries(after).filter(([key, value]) => !same(value, before[key])),
  );
}

/** Only explicitly planned leaf deletions may precede updates. Never delete a parent. */
export function assertLeafDeletion(db: SqliteDatabase, table: string): void {
  for (const { name } of db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
    .all()) {
    const keys = db.prepare(`PRAGMA foreign_key_list(${quoteIdentifier(String(name))})`).all();
    if (keys.some((key) => key.table === table))
      fail(`Parent collision/deletion in ${table}; dependent ${name} requires an explicit repair`);
  }
}

export function assertOne(result: { changes: number }, label: string): void {
  if (result.changes !== 1) fail(`Expected one affected row for ${label}; got ${result.changes}`);
}

export function applySqlChanges(db: SqliteDatabase, changes: SqlChange[]): void {
  for (const change of changes.filter((item) => item.after === null)) {
    assertLeafDeletion(db, change.table);
    assertOne(
      db.prepare(`DELETE FROM ${quoteIdentifier(change.table)} WHERE rowid = ?`).run(change.rowid),
      `${change.table}/${change.rowid}`,
    );
  }
  // Closed edges release the partial unique key before any endpoint moves.
  const updates = changes
    .filter((item) => item.after !== null)
    .sort(
      (a, b) =>
        Number(a.before === null) - Number(b.before === null) ||
        Number(b.table === "semantic_edges" && b.after?.valid_to !== null) -
          Number(a.table === "semantic_edges" && a.after?.valid_to !== null),
    );
  for (const change of updates) {
    if (change.before === null) {
      const current = db
        .prepare(`SELECT * FROM ${quoteIdentifier(change.table)} WHERE rowid = ?`)
        .get(change.rowid);
      // SQLite triggers are part of the projected plan, and already executed by the parent update.
      if (current !== undefined) {
        if (!same(current, change.after))
          fail(`Unexpected trigger row ${change.table}/${change.rowid}`);
        continue;
      }
      insertRow(db, change.table, change.rowid, change.after!);
      continue;
    }
    const fields = changedFields(change.before, change.after!);
    const keys = Object.keys(fields);
    if (keys.length > 0)
      assertOne(
        db
          .prepare(
            `UPDATE ${quoteIdentifier(change.table)} SET ${keys.map((key) => `${quoteIdentifier(key)} = ?`).join(", ")} WHERE rowid = ?`,
          )
          .run(...Object.values(fields), change.rowid),
        `${change.table}/${change.rowid}`,
      );
  }
}

export function insertRow(db: SqliteDatabase, table: string, rowid: number, row: Row): void {
  const keys = Object.keys(row);
  assertOne(
    db
      .prepare(
        `INSERT INTO ${quoteIdentifier(table)} (rowid, ${keys.map(quoteIdentifier).join(",")}) VALUES (?, ${keys.map(() => "?").join(",")})`,
      )
      .run(rowid, ...Object.values(row)),
    `${table}/${rowid}`,
  );
}

export function digest(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

/** A private in-memory projection exercises the real schema, constraints and triggers. */
export function cloneDatabase(source: SqliteDatabase): SqliteDatabase {
  const db = new SqliteDatabase(
    new SqliteRawDatabase(new DatabaseSync(":memory:", { enableDoubleQuotedStringLiterals: true })),
  );
  try {
    db.pragma("foreign_keys = OFF");
    const schema = source
      .prepare(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid",
      )
      .all();
    for (const row of schema.filter((row) => row.type === "table")) db.exec(String(row.sql));
    db.transaction(() => {
      for (const [table, rows] of sqlInventory(source))
        for (const { rowid, row } of rows) insertRow(db, table, rowid, row);
    })();
    if (
      source.prepare("SELECT 1 FROM sqlite_master WHERE name = 'sqlite_sequence'").get() !==
      undefined
    ) {
      db.exec("DELETE FROM sqlite_sequence");
      for (const row of source.prepare("SELECT name, seq FROM sqlite_sequence").all())
        assertOne(
          db
            .prepare("INSERT INTO sqlite_sequence (name, seq) VALUES (?, ?)")
            .run(row.name, row.seq),
          "sqlite_sequence projection",
        );
    }
    for (const row of schema.filter((row) => row.type !== "table")) db.exec(String(row.sql));
    db.pragma("foreign_keys = ON");
    return db;
  } catch (error) {
    db.close();
    throw error;
  }
}

export function diffSql(
  before: ReturnType<typeof sqlInventory>,
  after: ReturnType<typeof sqlInventory>,
): SqlChange[] {
  const changes: SqlChange[] = [];
  for (const table of new Set([...before.keys(), ...after.keys()])) {
    const oldRows = new Map((before.get(table) ?? []).map(({ rowid, row }) => [rowid, row]));
    const newRows = new Map((after.get(table) ?? []).map(({ rowid, row }) => [rowid, row]));
    for (const rowid of new Set([...oldRows.keys(), ...newRows.keys()])) {
      const oldRow = oldRows.get(rowid) ?? null,
        newRow = newRows.get(rowid) ?? null;
      if (!same(oldRow, newRow))
        changes.push({
          table,
          rowid,
          before: oldRow,
          after: newRow,
          reason:
            newRow === null
              ? "coalesce identical leaf reference"
              : oldRow === null
                ? "repository/trigger insert"
                : "repair structural references",
        });
    }
  }
  return changes;
}

export async function openLance(dataDir: string): Promise<Connection | null> {
  return existsSync(join(dataDir, "lancedb")) ? await connect(join(dataDir, "lancedb")) : null;
}

export async function planLance(
  connection: Connection | null,
  ids: ReadonlyMap<string, string>,
  nodes: Map<string, Row>,
  touchedNodes: Set<string>,
): Promise<{ changes: LanceChange[]; inventory: Record<string, string[]> }> {
  const changes: LanceChange[] = [];
  const inventory: Record<string, string[]> = {};
  if (connection === null) {
    if (touchedNodes.size > 0) fail("Semantic repair requires the Lance mirror");
    return { changes, inventory };
  }
  for (const name of (await connection.tableNames()).sort()) {
    const native = await connection.openTable(name);
    const table = new LanceDbTable(native);
    try {
      const columns = (await native.schema()).fields.map((field) => field.name);
      inventory[name] = columns;
      const rows = await table.list({
        columns: columns.filter((column) => column !== "embedding"),
      });
      for (const row of rows) {
        const next = remap(row, ids) as Row;
        next.id = row.id;
        const reembed = name === "semantic_nodes" && touchedNodes.has(String(row.id));
        if (reembed) {
          const sql = nodes.get(String(row.id)) ?? fail(`Missing SQL node ${row.id}`);
          for (const column of columns) {
            if (Object.hasOwn(sql, column))
              next[column] = column === "archived" ? Boolean(sql[column]) : sql[column];
          }
        }
        if (!same(row, next) || reembed)
          changes.push({ table: name, id: String(row.id), before: row, after: next, reembed });
      }
      if (name === "semantic_nodes") {
        for (const id of touchedNodes)
          if (!rows.some((row) => row.id === id)) fail(`Lance node is missing: ${id}`);
      }
    } finally {
      table.close();
    }
  }
  return { changes, inventory };
}
