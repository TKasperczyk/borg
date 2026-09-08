import { copyFileSync, mkdirSync, mkdtempSync, realpathSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { connect, type Connection, type Table } from "@lancedb/lancedb";
import { openReadOnlyDatabase, type SqliteDatabase } from "../../src/storage/sqlite/index.js";
import { ResidueReportError } from "./report.js";

const APP_DIRECTORY = realpathSync(fileURLToPath(new URL("../../", import.meta.url)));

function canonicalFuturePath(path: string): string {
  return statSync(path, { throwIfNoEntry: false }) === undefined
    ? join(canonicalFuturePath(dirname(path)), basename(path))
    : realpathSync(path);
}

function within(parent: string, path: string): boolean {
  const child = relative(parent, path);
  return child === "" || (!isAbsolute(child) && child !== ".." && !child.startsWith(`..${sep}`));
}

export function cacheDirectory(bank: string, cacheDir?: string): string {
  const root = canonicalFuturePath(
    resolve(cacheDir ?? process.env.XDG_CACHE_HOME ?? join(tmpdir(), "borg-cache")),
  );
  if (within(realpathSync(bank), root) || within(APP_DIRECTORY, root)) {
    throw new ResidueReportError(
      "Cache directory must be outside the bank and app, including symlinks",
    );
  }
  return root;
}

export function scratchDirectory(bank: string, cacheDir?: string): string {
  const root = cacheDirectory(bank, cacheDir);
  mkdirSync(root, { recursive: true, mode: 0o700 });
  return mkdtempSync(join(root, "borg-legacy-residue-"));
}

export function fileSignature(path: string): string | null {
  const stat = statSync(path, { bigint: true, throwIfNoEntry: false });
  if (stat === undefined) return null;
  if (!stat.isFile()) throw new ResidueReportError("Expected a regular input file");
  return [stat.dev, stat.ino, stat.size, stat.mtimeNs, stat.ctimeNs].join(":");
}

export type SqliteSnapshot = {
  directory: string;
  db: SqliteDatabase;
  close: () => void;
};

export function openSqliteSnapshot(bank: string, cacheDir?: string): SqliteSnapshot {
  const source = join(bank, "borg.db");
  const hasRollbackJournal = () =>
    (statSync(`${source}-journal`, { throwIfNoEntry: false })?.size ?? 0) > 0;
  if (hasRollbackJournal()) {
    throw new ResidueReportError(
      "SQLite rollback journal present; use a clean idle bank or snapshot",
    );
  }
  const before = [fileSignature(source), fileSignature(`${source}-wal`)];
  if (before[0] === null) throw new ResidueReportError("Missing borg.db; no database was created");
  const directory = scratchDirectory(bank, cacheDir);
  let db: SqliteDatabase | undefined;
  try {
    const snapshot = join(directory, "borg.db");
    // Never open SQLite on the bank: even readOnly can write its WAL shared memory.
    // Only the main file and WAL are copied. No bank lock, journal, or SHM is opened.
    copyFileSync(source, snapshot);
    if (before[1] !== null) copyFileSync(`${source}-wal`, `${snapshot}-wal`);
    const after = [fileSignature(source), fileSignature(`${source}-wal`)];
    if (hasRollbackJournal() || before.some((signature, index) => signature !== after[index])) {
      throw new ResidueReportError(
        "SQLite changed during snapshot copy; retry on an idle bank or snapshot",
      );
    }
    db = openReadOnlyDatabase(snapshot); // DatabaseSync({ readOnly: true }); no migrations.
    const opened = db;
    return {
      directory,
      db: opened,
      close() {
        try {
          opened.close();
        } finally {
          rmSync(directory, { recursive: true, force: true });
        }
      },
    };
  } catch (error) {
    try {
      db?.close();
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
    throw error;
  }
}

export type EpisodeTable = { connection: Connection; table: Table; version: number };

export async function openEpisodeTable(bank: string): Promise<EpisodeTable> {
  const path = join(bank, "lancedb");
  if (!statSync(join(path, "episodes.lance"), { throwIfNoEntry: false })?.isDirectory()) {
    throw new ResidueReportError(
      "Missing lancedb/episodes.lance; no connection or table was created",
    );
  }
  const connection = await connect(path);
  let table: Table | undefined;
  try {
    table = await connection.openTable("episodes");
    const version = await table.version();
    // checkout provides a read-only view, without a new version or lock file.
    // https://docs.lancedb.com/tables/versioning
    await table.checkout(version);
    return { connection, table, version };
  } catch (error) {
    table?.close();
    connection.close();
    throw error;
  }
}
