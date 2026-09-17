import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
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
  before: Row;
  after: Row | null;
  reason: string;
};
export type LanceChange = { table: string; id: string; before: Row; after: Row; reembed?: boolean };
export type FileChange = {
  path: string;
  hash: string;
  afterHash: string;
  lines: Array<{ index: number; before: string; after: string }>;
};

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
    const participant = episodeParticipantEntityId(value);
    if (participant !== null && ids.has(participant)) {
      return episodeParticipantEntityIdTerm(parseEntityId(ids.get(participant)!));
    }
    // Stored JSON is often nested inside SQL text or stream metadata.
    if (value.startsWith("{") || value.startsWith("[")) {
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
  if (Array.isArray(value)) {
    const next = value.map((item) => remap(item, ids));
    if (same(value, next)) return value;
    // Handle arrays are sets; do not deduplicate unrelated arrays or object evidence.
    return next.filter(
      (item, index) =>
        typeof item !== "string" ||
        ![...ids.values()].some((id) => item === id || item === `entity_id:${id}`) ||
        next.indexOf(item) === index,
    );
  }
  if (value !== null && typeof value === "object") {
    const result: Row = {};
    for (const [key, item] of Object.entries(value)) {
      const nextKey = ids.get(key) ?? key;
      // Stream text remains source evidence even when an author pasted JSON.
      const next = key === "content" && typeof item === "string" ? item : remap(item, ids);
      if (Object.hasOwn(result, nextKey) && !same(result[nextKey], next)) {
        fail(`Conflicting JSON map keys after remapping ${key}`);
      }
      result[nextKey] = next;
    }
    return result;
  }
  return value;
}

export function sqlInventory(db: SqliteDatabase): Map<string, Array<{ rowid: number; row: Row }>> {
  const tables = db
    .prepare(
      "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
    )
    .all() as Array<{ name: string }>;
  return new Map(
    tables.map(({ name }) => [
      name,
      (
        db.prepare(`SELECT rowid AS __repair_rowid, * FROM ${quoteIdentifier(name)}`).all() as Row[]
      ).map(({ __repair_rowid, ...row }) => ({ rowid: Number(__repair_rowid), row })),
    ]),
  );
}

export function changedFields(before: Row, after: Row): Row {
  return Object.fromEntries(
    Object.entries(after).filter(([key, value]) => !same(value, before[key])),
  );
}

export function applySqlChanges(db: SqliteDatabase, changes: SqlChange[]): void {
  // Link-table collisions are removed before retargeting. Entity deletion comes last.
  for (const change of changes.filter((item) => item.after === null && item.table !== "entities")) {
    db.prepare(`DELETE FROM ${quoteIdentifier(change.table)} WHERE rowid = ?`).run(change.rowid);
  }
  for (const change of changes.filter((item) => item.after !== null)) {
    const fields = changedFields(change.before, change.after!);
    const keys = Object.keys(fields);
    if (keys.length > 0)
      db.prepare(
        `UPDATE ${quoteIdentifier(change.table)} SET ${keys.map((key) => `${quoteIdentifier(key)} = ?`).join(", ")} WHERE rowid = ?`,
      ).run(...Object.values(fields), change.rowid);
  }
  for (const change of changes.filter((item) => item.after === null && item.table === "entities")) {
    db.prepare("DELETE FROM entities WHERE rowid = ?").run(change.rowid);
  }
}

export function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

export function planFiles(
  dataDir: string,
  ids: ReadonlyMap<string, string>,
): { files: FileChange[]; offsets: Map<string, number> } {
  const files: FileChange[] = [];
  const offsets = new Map<string, number>();
  // These are the authoritative non-SQL state files. Traces/blobs/config backups
  // are historical evidence or opaque attachment bytes, not identity indexes.
  for (const directory of ["stream", "working"]) {
    const root = join(dataDir, directory);
    if (!existsSync(root)) continue;
    for (const file of readdirSync(root).sort()) {
      if (!(directory === "stream" ? file.endsWith(".jsonl") : file.endsWith(".json"))) continue;
      const path = join(directory, file);
      const raw = readFileSync(join(dataDir, path), "utf8");
      const lines = directory === "stream" ? raw.split("\n") : [raw];
      const changes: FileChange["lines"] = [];
      let offset = 0;
      const streamOffsets: Array<[string, number]> = [];
      for (let index = 0; index < lines.length; index++) {
        const line = lines[index]!;
        if (line.trim() === "") {
          offset += Buffer.byteLength(line) + 1;
          continue;
        }
        const entry = JSON.parse(line) as Row;
        const next = remap(entry, ids) as Row;
        const after = same(entry, next) ? line : JSON.stringify(next);
        if (after !== line) {
          changes.push({ index, before: line, after });
          lines[index] = after;
        }
        if (directory === "stream" && typeof entry.id === "string")
          streamOffsets.push([entry.id, offset]);
        offset += Buffer.byteLength(after) + 1;
      }
      if (changes.length > 0) {
        files.push({
          path,
          hash: digest(raw),
          afterHash: digest(lines.join(directory === "stream" ? "\n" : "")),
          lines: changes,
        });
        for (const [id, position] of streamOffsets) offsets.set(id, position);
      }
    }
  }
  return { files, offsets };
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
