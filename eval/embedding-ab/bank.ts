import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";

import { connect } from "@lancedb/lancedb";

import { EpisodicRepository, type Episode } from "../../src/memory/episodic/index.js";
import { LanceDbTable } from "../../src/storage/lancedb/index.js";
import { SqliteDatabase, SqliteRawDatabase } from "../../src/storage/sqlite/index.js";

import type { EpisodeDocument } from "./types.js";

export type LoadedEpisodeBank = {
  dataDir: string;
  episodes: EpisodeDocument[];
  allEpisodeCount: number;
  sourceEmbeddingDimensions: number;
  activeCorpusSha256: string;
};

function openReadOnlyDatabase(path: string): SqliteDatabase {
  let raw: SqliteRawDatabase | undefined;

  try {
    raw = new SqliteRawDatabase(
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
  } catch (error) {
    try {
      raw?.close();
    } catch {
      // Preserve the original open failure.
    }
    throw error;
  }
}

function embeddingDimensionsFromSchema(
  schema: Awaited<ReturnType<LanceDbTable["schema"]>>,
): number {
  const embeddingField = schema.fields.find((field) => field.name === "embedding");
  const embeddingType = embeddingField?.type as { listSize?: unknown } | undefined;
  const dimensions = embeddingType?.listSize;

  if (typeof dimensions !== "number" || !Number.isInteger(dimensions) || dimensions <= 0) {
    throw new Error("Existing episodes LanceDB schema has no valid fixed-size embedding vector");
  }

  return dimensions;
}

export function episodeEmbeddingText(
  episode: Pick<Episode, "title" | "narrative" | "tags">,
): string {
  // Keep byte-for-byte parity with EpisodicExtractor's embedding call.
  return `${episode.title}\n${episode.narrative}\n${episode.tags.join(" ")}`;
}

function toEpisodeDocument(episode: Episode): EpisodeDocument {
  const embeddingText = episodeEmbeddingText(episode);
  return {
    id: episode.id,
    title: episode.title,
    narrative: episode.narrative,
    tags: [...episode.tags],
    embedding_text: embeddingText,
    embedding_text_sha256: createHash("sha256").update(embeddingText).digest("hex"),
  };
}

function corpusHash(episodes: readonly EpisodeDocument[]): string {
  const hash = createHash("sha256");

  for (const episode of [...episodes].sort((left, right) => left.id.localeCompare(right.id))) {
    hash.update(episode.id);
    hash.update("\0");
    hash.update(episode.embedding_text_sha256);
    hash.update("\n");
  }

  return hash.digest("hex");
}

export async function loadActiveEpisodeBank(dataDirectory: string): Promise<LoadedEpisodeBank> {
  const dataDir = resolve(dataDirectory);
  const databasePath = join(dataDir, "borg.db");
  const lancePath = join(dataDir, "lancedb");
  const episodesTablePath = join(lancePath, "episodes.lance");

  if (!existsSync(databasePath)) {
    throw new Error(`No borg.db found in data directory ${dataDir}`);
  }
  if (!existsSync(episodesTablePath)) {
    throw new Error(`No episodes LanceDB table found in data directory ${dataDir}`);
  }

  const db = openReadOnlyDatabase(databasePath);
  let connection: Awaited<ReturnType<typeof connect>> | undefined;
  let episodesTable: LanceDbTable | undefined;

  try {
    connection = await connect(lancePath);
    const tableNames = await connection.tableNames();
    if (!tableNames.includes("episodes")) {
      throw new Error(`LanceDB at ${lancePath} has no episodes table`);
    }

    // Open the already-existing table directly. LanceDbStore.openTable is deliberately
    // not used here because its production schema-evolution path may add columns.
    episodesTable = new LanceDbTable(await connection.openTable("episodes"));
    const sourceEmbeddingDimensions = embeddingDimensionsFromSchema(await episodesTable.schema());
    const repository = new EpisodicRepository({ table: episodesTable, db });
    const allEpisodes = await repository.listAll();

    // This is the repository's authoritative effective-visibility predicate. Calling it
    // per row avoids listEffectivelyVisible()'s optional index-backfill write path.
    const activeEpisodes = allEpisodes.filter((episode) =>
      repository.isEpisodeEffectivelyVisible(episode.id),
    );
    const episodes = activeEpisodes.map(toEpisodeDocument);

    if (episodes.length === 0) {
      throw new Error(
        `The bank contains ${allEpisodes.length} episode row(s), but none are effectively visible and active`,
      );
    }

    return {
      dataDir,
      episodes,
      allEpisodeCount: allEpisodes.length,
      sourceEmbeddingDimensions,
      activeCorpusSha256: corpusHash(episodes),
    };
  } finally {
    try {
      episodesTable?.close();
    } finally {
      try {
        connection?.close();
      } finally {
        db.close();
      }
    }
  }
}
