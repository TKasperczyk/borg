import { existsSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

import { connect, type Connection } from "@lancedb/lancedb";

import { loadConfig, type Config } from "../../src/config/index.js";
import { EntityRepository, type EntityRecord } from "../../src/memory/commitments/index.js";
import { EpisodicRepository } from "../../src/memory/episodic/index.js";
import { LanceDbTable } from "../../src/storage/lancedb/index.js";
import { SqliteDatabase, SqliteRawDatabase } from "../../src/storage/sqlite/index.js";
import { StreamEntryIndexRepository } from "../../src/stream/index.js";

import { loadActiveEpisodeBank, type LoadedEpisodeBank } from "../embedding-ab/bank.js";

export type OpenRecallPlannerBank = {
  metadata: LoadedEpisodeBank;
  config: Config;
  episodicRepository: EpisodicRepository;
  entityRepository: EntityRepository;
  entryIndex: StreamEntryIndexRepository;
  memoryOwner: EntityRecord | null;
  close(): void;
};

function sqliteTableExists(db: SqliteDatabase, name: string): boolean {
  return (
    db
      .prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1")
      .get(name) !== undefined
  );
}

function openReadOnlySqlite(path: string): SqliteDatabase {
  const raw = new SqliteRawDatabase(
    new DatabaseSync(path, {
      enableDoubleQuotedStringLiterals: true,
      readOnly: true,
    }),
  );
  const db = new SqliteDatabase(raw);
  db.pragma("busy_timeout = 5000");
  db.pragma("foreign_keys = ON");
  db.pragma("query_only = ON");
  return db;
}

function assertEpisodeIndexAlreadyBackfilled(db: SqliteDatabase): void {
  if (!sqliteTableExists(db, "episode_index_metadata")) {
    throw new Error(
      "The copied bank has no episode_index_metadata table; migrate/backfill the copy before evaluating it",
    );
  }

  const marker = db
    .prepare("SELECT value FROM episode_index_metadata WHERE key = ? LIMIT 1")
    .get("lance_backfilled_at");

  if (marker === undefined) {
    throw new Error(
      "The copied bank's episode index is not marked as backfilled. Open the copy once with the current Borg build before running this read-only evaluator.",
    );
  }
}

export async function openRecallPlannerBank(dataDirectory: string): Promise<OpenRecallPlannerBank> {
  // Reuse the embedding evaluator's authoritative active/effective-visibility loader and corpus
  // fingerprint. A second handle remains open below solely because RetrievalPipeline needs the
  // real repository for vector, indexed-term, recency, scoring, and MMR behavior.
  const metadata = await loadActiveEpisodeBank(dataDirectory);
  const config = loadConfig({ dataDir: metadata.dataDir, env: {} });
  const databasePath = join(metadata.dataDir, "borg.db");
  const lancePath = join(metadata.dataDir, "lancedb");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${metadata.dataDir}`);
  }

  const db = openReadOnlySqlite(databasePath);
  let connection: Connection | undefined;
  let episodesTable: LanceDbTable | undefined;

  try {
    assertEpisodeIndexAlreadyBackfilled(db);
    if (!sqliteTableExists(db, "entities")) {
      throw new Error(
        "The copied bank has no entities table; migrate the copy with the current Borg build before evaluating it",
      );
    }
    if (!sqliteTableExists(db, "stream_entry_index")) {
      throw new Error(
        "The copied bank has no stream_entry_index table; migrate/backfill the copy before evaluating it",
      );
    }
    connection = await connect(lancePath);
    episodesTable = new LanceDbTable(await connection.openTable("episodes"));
    const episodicRepository = new EpisodicRepository({ table: episodesTable, db });
    const entityRepository = new EntityRepository({ db });
    const entryIndex = new StreamEntryIndexRepository({ db, dataDir: metadata.dataDir });
    const memoryOwner = entityRepository.getSelf();
    let closed = false;

    return {
      metadata,
      config,
      episodicRepository,
      entityRepository,
      entryIndex,
      memoryOwner,
      close: () => {
        if (closed) {
          return;
        }
        closed = true;
        try {
          episodesTable?.close();
        } finally {
          try {
            connection?.close();
          } finally {
            db.close();
          }
        }
      },
    };
  } catch (error) {
    try {
      episodesTable?.close();
    } finally {
      try {
        connection?.close();
      } finally {
        db.close();
      }
    }
    throw error;
  }
}

/**
 * The comparison baseline is deliberately narrower than today's no-LLM degraded path: it is the
 * historical raw FOCUS-blob vector lane only. RetrievalPipeline always adds a recent intent, so a
 * bound proxy returns no candidates for those two repository calls while every other production
 * repository operation remains untouched.
 */
export function rawFocusOnlyRepository(repository: EpisodicRepository): EpisodicRepository {
  const disabled = new Set<PropertyKey>(["listRecentForDisclosure", "listHottestForDisclosure"]);

  return new Proxy(repository, {
    get(target, property) {
      if (disabled.has(property)) {
        return async () => [];
      }

      const value = Reflect.get(target, property, target) as unknown;
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}
