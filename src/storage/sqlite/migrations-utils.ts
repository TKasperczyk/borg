import type { SqliteDatabase } from "./index.js";

export function tableExists(db: SqliteDatabase, tableName: string): boolean {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?")
    .get(tableName) as { name: string } | undefined;

  return row !== undefined;
}

export function tableHasColumn(db: SqliteDatabase, tableName: string, columnName: string): boolean {
  const quotedTableName = tableName.replaceAll('"', '""');
  const rows = db.prepare(`PRAGMA table_info("${quotedTableName}")`).all() as Array<{
    name: string;
  }>;

  return rows.some((row) => row.name === columnName);
}
