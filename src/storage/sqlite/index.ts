import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { StatementSync } from "node:sqlite";

import { StorageError } from "../../util/errors.js";

export type SqliteRunResult = {
  changes: number;
  lastInsertRowid: number | bigint;
};

type SqliteRow = Record<string, unknown>;

type BoundStatement = {
  run: (...args: unknown[]) => SqliteRunResult;
  get: (...args: unknown[]) => SqliteRow | undefined;
  all: (...args: unknown[]) => SqliteRow[];
  iterate: (...args: unknown[]) => IterableIterator<SqliteRow>;
  columns: () => unknown[];
};

function bindArguments(args: unknown[]): unknown[] {
  return args.length === 1 && Array.isArray(args[0]) ? args[0] : args;
}

function normalizeRow(row: SqliteRow): SqliteRow {
  return { ...row };
}

function* normalizeRowIterator(rows: Iterable<SqliteRow>): IterableIterator<SqliteRow> {
  for (const row of rows) {
    yield normalizeRow(row);
  }
}

export class SqliteStatement {
  constructor(private readonly statement: StatementSync) {}

  run(...args: unknown[]): SqliteRunResult {
    return (this.statement as unknown as BoundStatement).run(...bindArguments(args));
  }

  get(...args: unknown[]): SqliteRow | undefined {
    const row = (this.statement as unknown as BoundStatement).get(...bindArguments(args));
    return row === undefined ? undefined : normalizeRow(row);
  }

  all(...args: unknown[]): SqliteRow[] {
    return (this.statement as unknown as BoundStatement)
      .all(...bindArguments(args))
      .map(normalizeRow);
  }

  iterate(...args: unknown[]): IterableIterator<SqliteRow> {
    return normalizeRowIterator(
      (this.statement as unknown as BoundStatement).iterate(...bindArguments(args)),
    );
  }

  columns(): unknown[] {
    return (this.statement as unknown as BoundStatement).columns();
  }
}

export type SqlitePragmaOptions = {
  simple?: boolean;
};

type TransactionMode = "deferred" | "immediate" | "exclusive";

export type SqliteTransaction<T extends (...args: never[]) => unknown> = ((
  ...args: Parameters<T>
) => ReturnType<T>) & {
  default: (...args: Parameters<T>) => ReturnType<T>;
  deferred: (...args: Parameters<T>) => ReturnType<T>;
  immediate: (...args: Parameters<T>) => ReturnType<T>;
  exclusive: (...args: Parameters<T>) => ReturnType<T>;
};

export class SqliteRawDatabase {
  private readonly statementCache = new Map<string, SqliteStatement>();
  private savepointCounter = 0;

  constructor(private readonly database: DatabaseSync) {}

  get inTransaction(): boolean {
    return this.database.isTransaction;
  }

  prepare(sql: string): SqliteStatement {
    const cached = this.statementCache.get(sql);

    if (cached !== undefined) {
      return cached;
    }

    const statement = new SqliteStatement(this.database.prepare(sql));
    this.statementCache.set(sql, statement);
    return statement;
  }

  exec(sql: string): void {
    this.database.exec(sql);
  }

  pragma(source: string, options: SqlitePragmaOptions = {}): unknown {
    const rows = this.prepare(`PRAGMA ${source}`).all();

    if (options.simple !== true) {
      return rows;
    }

    const first = rows[0];
    return first === undefined ? undefined : Object.values(first)[0];
  }

  transaction<T extends (...args: never[]) => unknown>(fn: T): SqliteTransaction<T> {
    const run = (mode: TransactionMode, args: Parameters<T>): ReturnType<T> =>
      this.runTransaction<ReturnType<T>>(mode, () => fn(...args) as ReturnType<T>);
    const transaction = ((...args: Parameters<T>): ReturnType<T> =>
      run("deferred", args)) as SqliteTransaction<T>;

    transaction.default = (...args: Parameters<T>): ReturnType<T> => run("deferred", args);
    transaction.deferred = (...args: Parameters<T>): ReturnType<T> => run("deferred", args);
    transaction.immediate = (...args: Parameters<T>): ReturnType<T> => run("immediate", args);
    transaction.exclusive = (...args: Parameters<T>): ReturnType<T> => run("exclusive", args);

    return transaction;
  }

  close(): void {
    this.statementCache.clear();
    this.database.close();
  }

  private runTransaction<T>(mode: TransactionMode, callback: () => T): T {
    if (this.inTransaction) {
      return this.runNestedTransaction(callback);
    }

    this.database.exec(beginSql(mode));

    try {
      const result = callback();
      this.database.exec("COMMIT");
      return result;
    } catch (error) {
      try {
        if (this.inTransaction) {
          this.database.exec("ROLLBACK");
        }
      } catch {
        // Preserve the original failure.
      }

      throw error;
    }
  }

  private runNestedTransaction<T>(callback: () => T): T {
    const savepoint = `borg_tx_${++this.savepointCounter}`;
    this.database.exec(`SAVEPOINT ${savepoint}`);

    try {
      const result = callback();
      this.database.exec(`RELEASE ${savepoint}`);
      return result;
    } catch (error) {
      try {
        this.database.exec(`ROLLBACK TO ${savepoint}`);
      } catch {
        // Preserve the original failure.
      }

      try {
        this.database.exec(`RELEASE ${savepoint}`);
      } catch {
        // Preserve the original failure.
      }

      throw error;
    }
  }
}

function beginSql(mode: TransactionMode): string {
  switch (mode) {
    case "deferred":
      return "BEGIN";
    case "immediate":
      return "BEGIN IMMEDIATE";
    case "exclusive":
      return "BEGIN EXCLUSIVE";
  }
}

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
  constructor(readonly raw: SqliteRawDatabase) {}

  prepare(sql: string): SqliteStatement {
    return this.raw.prepare(sql);
  }

  exec(sql: string): this {
    this.raw.exec(sql);
    return this;
  }

  pragma(source: string, options?: SqlitePragmaOptions): unknown {
    return this.raw.pragma(source, options);
  }

  transaction<T extends (...args: never[]) => unknown>(fn: T): SqliteTransaction<T> {
    return this.raw.transaction(fn);
  }

  listAppliedMigrations(): AppliedMigration[] {
    return this.prepare(
      "SELECT id, name, applied_at FROM _migrations ORDER BY id ASC",
    ).all() as AppliedMigration[];
  }

  close(): void {
    this.raw.close();
  }
}

const MIGRATION_BAND_SIZE = 1_000_000;

// Band migration arrays restart at id=1; shared SQLite DBs need stable global ids.
export function composeMigrations(...groups: readonly (readonly Migration[])[]): Migration[] {
  const seenIds = new Map<number, string>();

  return groups.flatMap((group, bandIndex) =>
    group.map((migration) => {
      const label = `${migration.name}:${migration.id}`;
      if (
        !Number.isInteger(migration.id) ||
        migration.id <= 0 ||
        migration.id >= MIGRATION_BAND_SIZE
      ) {
        throw new StorageError(
          `Migration source id ${migration.id} must be an integer in [1, ${
            MIGRATION_BAND_SIZE - 1
          }]: ${label}`,
        );
      }

      const id = bandIndex * MIGRATION_BAND_SIZE + migration.id;
      const existing = seenIds.get(id);

      if (existing !== undefined) {
        throw new StorageError(`Composed migration id collision between ${existing} and ${label}`);
      }

      seenIds.set(id, label);

      return {
        ...migration,
        id,
      };
    }),
  );
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
  let raw: SqliteRawDatabase | undefined;

  try {
    mkdirSync(dirname(path), { recursive: true });

    raw = new SqliteRawDatabase(
      new DatabaseSync(path, { enableDoubleQuotedStringLiterals: true }),
    );
    const db = new SqliteDatabase(raw);

    try {
      db.pragma("busy_timeout = 5000");
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
    if (error instanceof StorageError) {
      throw error;
    }

    throw new StorageError(`Failed to open SQLite database at ${path}`, {
      cause: error,
    });
  }
}
