import type { SqliteDatabase } from "./index.js";

function quoteSqlIdentifier(value: string): string {
  return `"${value.replaceAll('"', '""')}"`;
}

export function tableHasColumn(db: SqliteDatabase, table: string, column: string): boolean {
  const columns = db.prepare(`PRAGMA table_info(${quoteSqlIdentifier(table)})`).all() as Array<{
    name: string;
  }>;
  return columns.some((entry) => entry.name === column);
}

export function tableExists(db: SqliteDatabase, table: string): boolean {
  return (
    db
      .prepare(
        `
          SELECT 1
          FROM sqlite_master
          WHERE type = 'table' AND name = ?
          LIMIT 1
        `,
      )
      .get(table) !== undefined
  );
}
