import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { connect, type Connection, type Table } from "@lancedb/lancedb";
import { Field, Schema } from "apache-arrow";
import { z } from "zod";
import { createEpisodesTableSchema } from "../../src/memory/episodic/repository.js";
import { createSemanticNodesTableSchema } from "../../src/memory/semantic/repository.js";
import { createSkillsTableSchema } from "../../src/memory/procedural/repository.js";
import { createOpenQuestionsTableSchema } from "../../src/memory/self/open-questions.js";
import { createActionRecordsTableSchema } from "../../src/memory/actions/repository.js";
import { createObservedEventsTableSchema } from "../../src/memory/observed-events/repository.js";
import { createImagePerceptionTableSchema } from "../../src/attachments/perception.js";
import {
  buildEpisodeEmbeddingText,
  consolidationEmbeddingInputSchema,
  EpisodeEmbeddingTextError,
  type ConsolidationEmbeddingInput,
} from "../../src/memory/episodic/protected-lines.js";
import { buildNodeEmbeddingText } from "../../src/memory/semantic/embedding-text.js";
import { semanticObservationMetadataSchema } from "../../src/memory/semantic/types.js";
import {
  embeddingDimensionsFromSchema,
  EmbeddingBankError,
} from "../../src/embeddings/bank-profile.js";
import { serializedEmbeddingPaths } from "../../src/embeddings/serialized.js";
import { openReadOnlyDatabase, type SqliteDatabase } from "../../src/storage/sqlite/index.js";
import { vectorField } from "../../src/storage/lancedb/index.js";
import { quoteSqlString, toFloat32Array } from "../../src/storage/codecs.js";
import {
  fingerprintCanonicalValue,
  sha256Bytes,
} from "../../src/cognition/deliberation/request-fingerprint.js";

export const VECTOR_TABLES = [
  {
    name: "episodes",
    key: "id",
    sql: "episode_stats",
    sqlKey: "episode_id",
    schema: createEpisodesTableSchema,
  },
  {
    name: "semantic_nodes",
    key: "id",
    sql: "semantic_nodes",
    sqlKey: "id",
    schema: createSemanticNodesTableSchema,
  },
  { name: "skills", key: "id", sql: "skills", sqlKey: "id", schema: createSkillsTableSchema },
  {
    name: "open_questions",
    key: "id",
    sql: "open_questions",
    sqlKey: "id",
    schema: createOpenQuestionsTableSchema,
  },
  {
    name: "action_records",
    key: "id",
    sql: "action_records",
    sqlKey: "id",
    schema: createActionRecordsTableSchema,
  },
  {
    name: "image_perception_embeddings",
    key: "payload_id",
    sql: "image_perception_payloads",
    sqlKey: "payload_id",
    schema: createImagePerceptionTableSchema,
  },
  {
    name: "observed_events",
    key: "id",
    sql: "observed_events",
    sqlKey: "id",
    schema: createObservedEventsTableSchema,
  },
] as const;
export type VectorTableName = (typeof VECTOR_TABLES)[number]["name"];
export type RowIdentity = { id: string; text_hash: string | null; fields_hash: string };
export const unrecoverableInputSchema = z.object({
  episode_id: z.string(),
  reason: z.string(),
  candidate_count: z.number().int().nonnegative(),
});
export type UnrecoverableInput = z.infer<typeof unrecoverableInputSchema>;
type MigrationRecord = {
  row: Record<string, unknown>;
  text: string | null;
  identity: RowIdentity;
  unrecoverable?: UnrecoverableInput;
  longestPrefix?: ConsolidationEmbeddingInput;
};
export type TableInventory = {
  name: VectorTableName;
  present: boolean;
  dimensions: number | null;
  schema_hash: string;
  rows: RowIdentity[];
  sql_only: string[];
  vector_only: string[];
  text_disagreements: string[];
};
export type BankInventory = {
  tables: TableInventory[];
  sqlite_counts: Record<string, number>;
  serialized_vectors: { location: string; paths: string[] }[];
  problems: string[];
  // Omit when empty to retain the fingerprints of existing journals/backups.
  embedding_text_unrecoverable?: UnrecoverableInput[];
};

export function migrationRowText(
  name: VectorTableName,
  row: Record<string, unknown>,
  legacyProtectedSourceTexts?: readonly string[],
): string {
  const strings = (value: unknown): string[] =>
    z.array(z.string()).parse(typeof value === "string" ? JSON.parse(value) : (value ?? []));
  if (name === "episodes") {
    return buildEpisodeEmbeddingText({
      title: z.string().parse(row.title),
      narrative: z.string().parse(row.narrative),
      tags: strings(row.tags),
      participants: strings(row.participants),
      episode_kind: z.string().nullable().optional().parse(row.episode_kind),
      consolidation_embedding_input: consolidationEmbeddingInputSchema
        .nullable()
        .optional()
        .parse(
          typeof row.consolidation_embedding_input === "string"
            ? JSON.parse(row.consolidation_embedding_input)
            : row.consolidation_embedding_input,
        ),
      legacyProtectedSourceTexts,
    });
  }
  if (name === "semantic_nodes") {
    return buildNodeEmbeddingText({
      label: z.string().parse(row.label),
      description: z.string().parse(row.description),
      aliases: strings(row.aliases),
      observationMetadata: semanticObservationMetadataSchema
        .nullable()
        .parse(
          typeof row.observation_metadata === "string"
            ? JSON.parse(row.observation_metadata)
            : (row.observation_metadata ?? null),
        ),
    });
  }
  const field = {
    skills: "applies_when",
    open_questions: "question",
    action_records: "description",
    image_perception_embeddings: "embedding_text",
    observed_events: "interaction_text",
  }[name];
  return z.string().min(1).parse(row[field]);
}

export function migrationRowIdentity(
  name: VectorTableName,
  row: Record<string, unknown>,
  text: string | null = migrationRowText(name, row),
): RowIdentity {
  const { embedding: _embedding, ...fields } = row;
  return {
    id: z
      .string()
      .min(1)
      .parse(row[name === "image_perception_embeddings" ? "payload_id" : "id"]),
    text_hash: text === null ? null : sha256Bytes(Buffer.from(text)),
    fields_hash: fingerprintCanonicalValue(fields).canonicalSha256,
  };
}

export function migrationSchema(source: Schema, dimensions: number): Schema {
  return new Schema(
    source.fields.map((field) =>
      field.name === "embedding"
        ? new Field(
            field.name,
            vectorField("embedding", dimensions, field.nullable).type,
            field.nullable,
            field.metadata,
          )
        : field,
    ),
    source.metadata,
  );
}

export function migrationSchemaHash(source: Schema): string {
  return fingerprintCanonicalValue({
    fields: source.fields.map((field) => ({
      name: field.name,
      type: field.type.toString(),
      nullable: field.nullable,
      metadata: [...field.metadata].sort(),
    })),
    metadata: [...source.metadata].sort(),
  }).canonicalSha256;
}

/** Offset scans are stable while the tenant access lease excludes every writer. */
export async function* migrationRows(
  table: Table,
  batchSize: number,
  includeVectors = false,
): AsyncGenerator<Record<string, unknown>[]> {
  const columns = (await table.schema()).fields
    .map((field) => field.name)
    .filter((name) => includeVectors || name !== "embedding");
  for (let offset = 0; ; offset += batchSize) {
    const rows = (
      await table.query().select(columns).offset(offset).limit(batchSize).toArray()
    ).map((row) => ({ ...row }));
    if (rows.length === 0) return;
    if (includeVectors)
      for (const row of rows)
        row.embedding = toFloat32Array(row.embedding, {
          arrayLikeErrorMessage: "Stored embedding is not an array",
          nonFiniteErrorMessage: "Stored embedding contains non-finite values",
          errorCode: "EMBEDDING_VECTOR_INVALID",
        });
    yield rows;
  }
}

/** Resolve archived raw lineage directly, without the production list/get visibility filters. */
export async function* migrationRecords(
  table: Table,
  name: VectorTableName,
  batchSize: number,
  includeVectors = false,
  sourceTable = table,
): AsyncGenerator<MigrationRecord[]> {
  for await (const rows of migrationRows(table, batchSize, includeVectors)) {
    const records: MigrationRecord[] = [];
    for (const row of rows) {
      try {
        let sources: string[] | undefined;
        if (
          name === "episodes" &&
          row.episode_kind === "consolidation_version" &&
          row.consolidation_embedding_input == null
        ) {
          const ids = z
            .array(z.string())
            .parse(JSON.parse(String(row.lineage_derived_from ?? "[]")));
          const rawRows =
            ids.length === 0
              ? []
              : await sourceTable
                  .query()
                  .where(`id IN (${ids.map(quoteSqlString).join(", ")})`)
                  .select(["id", "narrative"])
                  .limit(ids.length)
                  .toArray();
          const byId = new Map(
            rawRows.map((source) => [String(source.id), z.string().parse(source.narrative)]),
          );
          if (byId.size !== new Set(ids).size || rawRows.length !== byId.size)
            throw new EmbeddingBankError(`Cannot recover raw consolidation sources for ${row.id}`, {
              code: "EMBEDDING_TEXT_UNRECOVERABLE",
            });
          // Production records derived_from in the same oldest-first order used
          // for protected source lines. The persisted lineage retains that order.
          sources = ids.map((id) => byId.get(id)!);
        }
        const text = migrationRowText(name, row, sources);
        records.push({ row, text, identity: migrationRowIdentity(name, row, text) });
      } catch (error) {
        if (!(error instanceof EmbeddingBankError) || error.code !== "EMBEDDING_TEXT_UNRECOVERABLE")
          throw error;
        records.push({
          row,
          text: null,
          identity: migrationRowIdentity(name, row, null),
          unrecoverable: {
            episode_id: String(row.id),
            reason: error.message,
            candidate_count: error instanceof EpisodeEmbeddingTextError ? error.candidateCount : 0,
          },
          longestPrefix:
            error instanceof EpisodeEmbeddingTextError ? error.longestPrefix : undefined,
        });
      }
    }
    yield records;
  }
}

export function sqliteInventory(db: SqliteDatabase): Record<string, number> {
  if (db.pragma("integrity_check", { simple: true }) !== "ok")
    throw new EmbeddingBankError("SQLite integrity_check failed", {
      code: "EMBEDDING_MIGRATION_INCOMPLETE",
    });
  const result: Record<string, number> = {};
  for (const row of db
    .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    .all()) {
    const name = String(row.name);
    result[name] = Number(
      db.prepare(`SELECT count(*) AS n FROM "${name.replaceAll('"', '""')}"`).get()?.n,
    );
  }
  return result;
}

export async function inventoryBank(
  tenantDir: string,
  targetDims: number,
  batchSize = 256,
  lanceDir = join(tenantDir, "lancedb"),
): Promise<BankInventory> {
  const db = openReadOnlyDatabase(join(tenantDir, "borg.db"));
  let connection: Connection | undefined;
  try {
    const result: BankInventory = {
      tables: [],
      sqlite_counts: sqliteInventory(db),
      serialized_vectors: [],
      problems: [],
    };
    if (existsSync(lanceDir)) connection = await connect(lanceDir);
    const names = connection ? await connection.tableNames() : [];
    for (const name of names) {
      if (!VECTOR_TABLES.some((definition) => definition.name === name))
        result.problems.push(`Unrecognized LanceDB table ${name}; refusing to omit it`);
    }
    for (const definition of VECTOR_TABLES) {
      const table = names.includes(definition.name)
        ? await connection!.openTable(definition.name)
        : undefined;
      try {
        const sourceSchema = table ? await table.schema() : definition.schema(targetDims);
        const entry: TableInventory = {
          name: definition.name,
          present: table !== undefined,
          dimensions: table ? embeddingDimensionsFromSchema(sourceSchema) : null,
          schema_hash: migrationSchemaHash(migrationSchema(sourceSchema, targetDims)),
          rows: [],
          sql_only: [],
          vector_only: [],
          text_disagreements: [],
        };
        const sqlRows = new Map<string, Record<string, unknown>>();
        if (result.sqlite_counts[definition.sql] !== undefined) {
          for (const row of db.prepare(`SELECT * FROM ${definition.sql}`).iterate())
            sqlRows.set(String(row[definition.sqlKey]), row);
        }
        const seen = new Set<string>();
        if (table)
          for await (const records of migrationRecords(table, definition.name, batchSize))
            for (const { row, identity, unrecoverable } of records) {
              if (unrecoverable) (result.embedding_text_unrecoverable ??= []).push(unrecoverable);
              if (seen.has(identity.id))
                result.problems.push(`Duplicate id ${definition.name}/${identity.id}`);
              seen.add(identity.id);
              entry.rows.push(identity);
              const sql = sqlRows.get(identity.id);
              if (!sql) entry.vector_only.push(identity.id);
              else if (
                definition.name !== "episodes" &&
                migrationRowText(definition.name, sql) !== migrationRowText(definition.name, row)
              )
                entry.text_disagreements.push(identity.id);
            }
        entry.rows.sort((a, b) => a.id.localeCompare(b.id));
        entry.sql_only = [...sqlRows.keys()].filter((id) => !seen.has(id)).sort();
        entry.vector_only.sort();
        entry.text_disagreements.sort();
        if (entry.sql_only.length || entry.vector_only.length || entry.text_disagreements.length)
          result.problems.push(
            `SQL/vector discrepancy in ${definition.name}: SQL-only=${entry.sql_only.length}, vector-only=${entry.vector_only.length}, text disagreements=${entry.text_disagreements.length}; repair before migrating`,
          );
        result.tables.push(entry);
      } finally {
        table?.close();
      }
    }
    for (const [table, columns] of [
      ["review_queue", ["refs"]],
      ["maintenance_audit", ["reversal", "targets"]],
    ] as const) {
      if (result.sqlite_counts[table] === undefined) continue;
      for (const row of db.prepare(`SELECT id, ${columns.join(", ")} FROM ${table}`).iterate()) {
        for (const column of columns) {
          const paths = serializedEmbeddingPaths(JSON.parse(String(row[column])));
          if (paths.length)
            result.serialized_vectors.push({ location: `${table}/${row.id}/${column}`, paths });
        }
      }
    }
    function inspectPlans(directory: string): void {
      for (const entry of readdirSync(directory, { withFileTypes: true })) {
        if (
          entry.name === "lancedb" ||
          entry.name.startsWith("lancedb.") ||
          entry.name.startsWith(".embedding-migration") ||
          entry.name === "backups"
        )
          continue;
        const path = join(directory, entry.name);
        if (entry.isDirectory()) inspectPlans(path);
        else if (entry.isFile() && entry.name.endsWith(".json")) {
          const value: unknown = JSON.parse(readFileSync(path, "utf8"));
          const paths = serializedEmbeddingPaths(value);
          if (paths.length)
            result.serialized_vectors.push({ location: relative(tenantDir, path), paths });
        }
      }
    }
    inspectPlans(tenantDir);
    result.embedding_text_unrecoverable?.sort((a, b) => a.episode_id.localeCompare(b.episode_id));
    return result;
  } finally {
    connection?.close();
    db.close();
  }
}
