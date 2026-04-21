import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

import Database from "better-sqlite3";
import type BetterSqlite3 from "better-sqlite3";

import { StorageError } from "../../util/errors.js";

type BetterSqliteDatabase = BetterSqlite3.Database;
type PreparedStatement = BetterSqlite3.Statement;

export type Migration = {
  id: number;
  name: string;
  up: string | ((db: SqliteDatabase) => void);
};

export type OpenDatabaseOptions = {
  migrations?: readonly Migration[];
};

export type AppliedMigration = {
  id: number;
  name: string;
  applied_at: number;
};

export class SqliteDatabase {
  private readonly statementCache = new Map<string, PreparedStatement>();

  constructor(readonly raw: BetterSqliteDatabase) {}

  prepare(sql: string): PreparedStatement {
    const cached = this.statementCache.get(sql);

    if (cached !== undefined) {
      return cached;
    }

    const statement = this.raw.prepare(sql);
    this.statementCache.set(sql, statement);
    return statement;
  }

  exec(sql: string): this {
    this.raw.exec(sql);
    return this;
  }

  pragma(source: string, options?: Parameters<BetterSqliteDatabase["pragma"]>[1]): unknown {
    return this.raw.pragma(source, options);
  }

  transaction<T extends (...args: never[]) => unknown>(fn: T): BetterSqlite3.Transaction<T> {
    return this.raw.transaction(fn);
  }

  listAppliedMigrations(): AppliedMigration[] {
    return this.prepare(
      "SELECT id, name, applied_at FROM _migrations ORDER BY id ASC",
    ).all() as AppliedMigration[];
  }

  close(): void {
    this.statementCache.clear();
    this.raw.close();
  }
}

function ensureMigrationTable(db: SqliteDatabase): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS _migrations (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      applied_at INTEGER NOT NULL
    )
  `);
}

function validateMigrations(migrations: readonly Migration[]): void {
  const seenIds = new Set<number>();

  for (const migration of migrations) {
    if (!Number.isInteger(migration.id) || migration.id <= 0) {
      throw new StorageError(`Migration ids must be positive integers: ${migration.name}`);
    }

    if (seenIds.has(migration.id)) {
      throw new StorageError(`Duplicate migration id ${migration.id}`);
    }

    seenIds.add(migration.id);
  }
}

function runMigrations(db: SqliteDatabase, migrations: readonly Migration[]): void {
  validateMigrations(migrations);
  ensureMigrationTable(db);

  const appliedIds = new Set(db.listAppliedMigrations().map((migration) => migration.id));
  const insertMigration = db.prepare(
    "INSERT INTO _migrations (id, name, applied_at) VALUES (?, ?, ?)",
  );

  for (const migration of [...migrations].sort((left, right) => left.id - right.id)) {
    if (appliedIds.has(migration.id)) {
      continue;
    }

    const applyMigration = db.raw.transaction(() => {
      if (typeof migration.up === "string") {
        db.exec(migration.up);
      } else {
        migration.up(db);
      }

      insertMigration.run(migration.id, migration.name, Date.now());
    });

    applyMigration();
  }
}

export function openDatabase(path: string, options: OpenDatabaseOptions = {}): SqliteDatabase {
  let raw: BetterSqliteDatabase | undefined;

  try {
    mkdirSync(dirname(path), { recursive: true });

    raw = new Database(path);
    const db = new SqliteDatabase(raw);

    try {
      db.pragma("journal_mode = WAL");
      db.pragma("foreign_keys = ON");
      runMigrations(db, options.migrations ?? []);
      return db;
    } catch (error) {
      try {
        raw.close();
      } catch {
        // Best-effort cleanup after partial initialization.
      }

      throw error;
    }
  } catch (error) {
    throw new StorageError(`Failed to open SQLite database at ${path}`, {
      cause: error,
    });
  }
}
