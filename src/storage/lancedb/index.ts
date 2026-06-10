import { performance } from "node:perf_hooks";

import {
  connect,
  makeArrowTable,
  type AddColumnsSql,
  type Connection,
  type IntoVector,
  type OptimizeStats,
  type SchemaLike,
  type Table,
  type TableStatistics,
  type VectorQuery,
} from "@lancedb/lancedb";
import {
  Bool,
  Field,
  FixedSizeList,
  Float32,
  Float64,
  Int32,
  Int64,
  Schema,
  TimestampMillisecond,
  Utf8,
} from "apache-arrow";

import { BorgError, StorageError } from "../../util/errors.js";

export type LanceDbRow = Record<string, unknown>;

// Keep recent versions available for in-flight same-process readers that began
// before compaction. Fragment compaction still runs; only version pruning waits.
export const LANCEDB_OPTIMIZE_CLEANUP_GRACE_MS = 15 * 60 * 1_000;

export type LanceDbOpenTableOptions = {
  name: string;
  schema: SchemaLike;
};

export type LanceDbUpsertOptions = {
  on: string | string[];
};

export type LanceDbSearchOptions = {
  limit?: number;
  where?: string;
  columns?: string[];
  vectorColumn?: string;
  distanceType?: "l2" | "cosine" | "dot";
};

export type LanceDbListOptions = {
  where?: string;
  limit?: number;
  columns?: string[];
};

export type LanceDbStoreOptions = {
  uri: string;
  connection?: Connection | Promise<Connection>;
};

export type LanceDbOptimizeStorageOptions = {
  now?: number | Date;
  cleanupGraceMs?: number;
};

export type LanceDbOptimizeErrorDetails = {
  message: string;
  code?: string;
};

export type LanceDbOptimizeTableSuccess = {
  table: string;
  status: "ok";
  fragmentsRemoved: number;
  fragmentsAdded: number;
  versionsPruned: number;
  bytesRemoved: number;
  durationMs: number;
};

export type LanceDbOptimizeTableError = {
  table: string;
  status: "error";
  durationMs: number;
  error: LanceDbOptimizeErrorDetails;
};

export type LanceDbOptimizeTableResult = LanceDbOptimizeTableSuccess | LanceDbOptimizeTableError;

export type LanceDbOptimizeStorageResult = {
  cleanupOlderThan?: number;
  durationMs: number;
  tables: LanceDbOptimizeTableResult[];
  error?: LanceDbOptimizeErrorDetails;
};

function normalizeRows(rows: unknown): LanceDbRow[] {
  if (!Array.isArray(rows)) {
    throw new StorageError("LanceDB returned a non-array result");
  }

  return rows.map((row) => {
    if (row === null || typeof row !== "object" || Array.isArray(row)) {
      throw new StorageError("LanceDB returned a non-object row");
    }

    return row as LanceDbRow;
  });
}

function normalizeSchemaLike(schemaLike: SchemaLike): Schema {
  if (schemaLike instanceof Schema) {
    return schemaLike;
  }

  if ("fields" in schemaLike && Array.isArray(schemaLike.fields)) {
    return new Schema(schemaLike.fields as Field[], schemaLike.metadata);
  }

  throw new StorageError("Unsupported LanceDB schema shape", {
    code: "LANCEDB_SCHEMA_INVALID",
  });
}

function dataTypeSignature(type: Field["type"]): unknown {
  const signature: Record<string, unknown> = {
    typeId: type.typeId,
  };

  if ("precision" in type && typeof type.precision === "number") {
    signature.precision = type.precision;
  }

  if ("scale" in type && typeof type.scale === "number") {
    signature.scale = type.scale;
  }

  if ("unit" in type && typeof type.unit === "number") {
    signature.unit = type.unit;
  }

  if ("listSize" in type && typeof type.listSize === "number") {
    signature.listSize = type.listSize;
  }

  if ("children" in type && Array.isArray(type.children) && type.children.length > 0) {
    signature.children = type.children.map((child: Field) => ({
      name: child.name,
      type: dataTypeSignature(child.type),
    }));
  }

  return signature;
}

function fieldTypeSignature(field: Field): string {
  return JSON.stringify({
    nullable: field.nullable,
    type: dataTypeSignature(field.type),
  });
}

function defaultValueSqlForField(field: Field): string {
  const type = field.type;

  if (type instanceof Utf8) {
    return field.nullable ? "CAST(NULL AS STRING)" : "''";
  }

  if (type instanceof Bool) {
    return field.nullable ? "CAST(NULL AS BOOLEAN)" : "false";
  }

  if (type instanceof Int32 || type instanceof Int64 || type instanceof TimestampMillisecond) {
    return field.nullable ? "CAST(NULL AS BIGINT)" : "0";
  }

  if (type instanceof Float32 || type instanceof Float64) {
    return field.nullable ? "CAST(NULL AS DOUBLE)" : "0";
  }

  throw new StorageError(`Cannot add non-nullable LanceDB column ${field.name} automatically`, {
    code: "LANCEDB_SCHEMA_EVOLUTION_UNSUPPORTED",
  });
}

async function ensureSchemaCompatibility(
  table: Table,
  requestedSchemaLike: SchemaLike,
  tableName: string,
): Promise<void> {
  const requestedSchema = normalizeSchemaLike(requestedSchemaLike);
  const existingSchema = await table.schema();
  const existingByName = new Map(existingSchema.fields.map((field) => [field.name, field]));
  const missingColumns: AddColumnsSql[] = [];

  for (const requestedField of requestedSchema.fields) {
    const existingField = existingByName.get(requestedField.name);

    if (existingField === undefined) {
      missingColumns.push({
        name: requestedField.name,
        valueSql: defaultValueSqlForField(requestedField),
      });
      continue;
    }

    if (fieldTypeSignature(existingField) !== fieldTypeSignature(requestedField)) {
      throw new StorageError(
        `Existing LanceDB field ${requestedField.name} in ${tableName} does not match the requested schema`,
        {
          code: "LANCEDB_SCHEMA_MISMATCH",
        },
      );
    }
  }

  if (missingColumns.length > 0) {
    await table.addColumns(missingColumns);
  }
}

export function normalizeOptimizeError(error: unknown): LanceDbOptimizeErrorDetails {
  return {
    message: error instanceof Error ? error.message : String(error),
    ...(error instanceof BorgError ? { code: error.code } : {}),
  };
}

function durationSince(startedAt: number): number {
  return Math.round(performance.now() - startedAt);
}

function resolveOptimizeNowMs(now: number | Date | undefined): number {
  return now instanceof Date ? now.getTime() : (now ?? Date.now());
}

export class LanceDbTable {
  constructor(
    private readonly table: Table,
    private readonly onClose?: () => void,
  ) {}

  get name(): string {
    return this.table.name;
  }

  isOpen(): boolean {
    const table = this.table as Table & { isOpen?: () => boolean };
    return table.isOpen?.() ?? true;
  }

  async checkoutLatest(): Promise<void> {
    try {
      await this.table.checkoutLatest();
    } catch (error) {
      throw new StorageError(`Failed to refresh LanceDB table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  async schema(): Promise<Schema> {
    try {
      return await this.table.schema();
    } catch (error) {
      throw new StorageError(`Failed to read LanceDB schema for table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  async addColumns(columns: AddColumnsSql[] | Field | Field[] | Schema): Promise<void> {
    try {
      await this.table.addColumns(columns);
    } catch (error) {
      throw new StorageError(`Failed to evolve LanceDB table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  async upsert(rows: readonly LanceDbRow[], options: LanceDbUpsertOptions): Promise<void> {
    if (rows.length === 0) {
      return;
    }

    try {
      await this.table.checkoutLatest();
      const arrowTable = makeArrowTable([...rows], {
        schema: await this.table.schema(),
      });
      await this.table
        .mergeInsert(options.on)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute(arrowTable);
    } catch (error) {
      throw new StorageError(`Failed to upsert rows into LanceDB table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  async search(vector: IntoVector, options: LanceDbSearchOptions = {}): Promise<LanceDbRow[]> {
    try {
      let query = this.table.search(vector) as VectorQuery;

      if (options.vectorColumn !== undefined) {
        query = query.column(options.vectorColumn);
      }

      if (options.distanceType !== undefined) {
        const vectorQuery = query as VectorQuery & {
          distanceType?: (distanceType: "l2" | "cosine" | "dot") => VectorQuery;
        };

        query = vectorQuery.distanceType?.(options.distanceType) ?? query;
      }

      if (options.where !== undefined) {
        query = query.where(options.where);
      }

      if (options.columns !== undefined) {
        query = query.select(options.columns);
      }

      if (options.limit !== undefined) {
        query = query.limit(options.limit);
      }

      return normalizeRows((await query.toArray()) as unknown);
    } catch (error) {
      throw new StorageError(`Failed to search LanceDB table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  async remove(where: string): Promise<void> {
    try {
      await this.table.delete(where);
    } catch (error) {
      throw new StorageError(`Failed to delete rows from LanceDB table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  async list(options: LanceDbListOptions = {}): Promise<LanceDbRow[]> {
    try {
      let query = this.table.query();

      if (options.where !== undefined) {
        query = query.where(options.where);
      }

      if (options.columns !== undefined) {
        query = query.select(options.columns);
      }

      if (options.limit !== undefined) {
        query = query.limit(options.limit);
      }

      return normalizeRows((await query.toArray()) as unknown);
    } catch (error) {
      throw new StorageError(`Failed to list rows from LanceDB table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  async stats(): Promise<TableStatistics> {
    try {
      return await this.table.stats();
    } catch (error) {
      throw new StorageError(`Failed to read LanceDB stats for table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  async optimize(options: { cleanupOlderThan: Date }): Promise<OptimizeStats> {
    try {
      await this.table.checkoutLatest();
      const stats = await this.table.optimize({
        cleanupOlderThan: options.cleanupOlderThan,
      });
      await this.table.checkoutLatest();
      return stats;
    } catch (error) {
      throw new StorageError(`Failed to optimize LanceDB table ${this.table.name}`, {
        cause: error,
      });
    }
  }

  close(): void {
    try {
      this.table.close();
    } finally {
      this.onClose?.();
    }
  }
}

export class LanceDbStore {
  private readonly connectionPromise: Promise<Connection>;
  private readonly openTablesByName = new Map<string, Set<LanceDbTable>>();

  constructor(options: LanceDbStoreOptions) {
    this.connectionPromise =
      options.connection !== undefined ? Promise.resolve(options.connection) : connect(options.uri);
  }

  private async getConnection(): Promise<Connection> {
    try {
      return await this.connectionPromise;
    } catch (error) {
      throw new StorageError("Failed to open LanceDB connection", {
        cause: error,
      });
    }
  }

  private registerTable(table: LanceDbTable): void {
    const tables = this.openTablesByName.get(table.name) ?? new Set<LanceDbTable>();
    tables.add(table);
    this.openTablesByName.set(table.name, tables);
  }

  private unregisterTable(table: LanceDbTable): void {
    const tables = this.openTablesByName.get(table.name);

    if (tables === undefined) {
      return;
    }

    tables.delete(table);

    if (tables.size === 0) {
      this.openTablesByName.delete(table.name);
    }
  }

  private wrapTable(table: Table): LanceDbTable {
    let wrapped: LanceDbTable;
    wrapped = new LanceDbTable(table, () => this.unregisterTable(wrapped));
    this.registerTable(wrapped);
    return wrapped;
  }

  private openTablesFor(name: string): LanceDbTable[] {
    const tables = this.openTablesByName.get(name);

    if (tables === undefined) {
      return [];
    }

    const openTables = [...tables].filter((table) => table.isOpen());

    if (openTables.length !== tables.size) {
      this.openTablesByName.set(name, new Set(openTables));
    }

    if (openTables.length === 0) {
      this.openTablesByName.delete(name);
    }

    return openTables;
  }

  private async optimizeTable(
    connection: Connection,
    tableName: string,
    cleanupOlderThan: Date,
  ): Promise<LanceDbOptimizeTableResult> {
    const startedAt = performance.now();
    const openTables = this.openTablesFor(tableName);
    let temporaryTable: LanceDbTable | null = null;

    try {
      const table = openTables[0] ?? new LanceDbTable(await connection.openTable(tableName));

      if (openTables.length === 0) {
        temporaryTable = table;
      }

      const stats = await table.optimize({ cleanupOlderThan });

      for (const openTable of openTables) {
        if (openTable !== table && openTable.isOpen()) {
          await openTable.checkoutLatest();
        }
      }

      return {
        table: tableName,
        status: "ok",
        fragmentsRemoved: stats.compaction.fragmentsRemoved,
        fragmentsAdded: stats.compaction.fragmentsAdded,
        versionsPruned: stats.prune.oldVersionsRemoved,
        bytesRemoved: stats.prune.bytesRemoved,
        durationMs: durationSince(startedAt),
      };
    } catch (error) {
      return {
        table: tableName,
        status: "error",
        durationMs: durationSince(startedAt),
        error: normalizeOptimizeError(error),
      };
    } finally {
      temporaryTable?.close();
    }
  }

  async openTable(options: LanceDbOpenTableOptions): Promise<LanceDbTable> {
    const connection = await this.getConnection();

    const verifyAndReopenTable = async (table: Table): Promise<LanceDbTable> => {
      try {
        await ensureSchemaCompatibility(table, options.schema, options.name);
        await table.checkoutLatest();
        table.close();
        const reopenedTable = await connection.openTable(options.name);
        await reopenedTable.checkoutLatest();
        return this.wrapTable(reopenedTable);
      } catch (error) {
        if (error instanceof StorageError) {
          throw error;
        }

        throw new StorageError(`Failed to open LanceDB table ${options.name}`, {
          cause: error,
        });
      }
    };

    const openCompatibleTable = async (): Promise<LanceDbTable> =>
      verifyAndReopenTable(await connection.openTable(options.name));

    try {
      const tableNames = await connection.tableNames();

      if (tableNames.includes(options.name)) {
        return await openCompatibleTable();
      }

      return await verifyAndReopenTable(
        await connection.createEmptyTable(options.name, options.schema, {
          mode: "create",
          existOk: false,
        }),
      );
    } catch (error) {
      if (error instanceof StorageError) {
        throw error;
      }

      try {
        return await openCompatibleTable();
      } catch (fallbackError) {
        if (fallbackError instanceof StorageError) {
          throw fallbackError;
        }

        throw new StorageError(`Failed to open LanceDB table ${options.name}`, {
          cause: fallbackError,
        });
      }
    }
  }

  async optimizeStorage(
    options: LanceDbOptimizeStorageOptions = {},
  ): Promise<LanceDbOptimizeStorageResult> {
    const startedAt = performance.now();
    const cleanupGraceMs = options.cleanupGraceMs ?? LANCEDB_OPTIMIZE_CLEANUP_GRACE_MS;
    const cleanupOlderThan = new Date(resolveOptimizeNowMs(options.now) - cleanupGraceMs);
    const connection = await this.getConnection();
    let tableNames: string[];

    try {
      tableNames = await connection.tableNames();
    } catch (error) {
      throw new StorageError("Failed to list LanceDB tables for storage optimization", {
        cause: error,
      });
    }

    const tables: LanceDbOptimizeTableResult[] = [];

    for (const tableName of tableNames) {
      tables.push(await this.optimizeTable(connection, tableName, cleanupOlderThan));
    }

    return {
      cleanupOlderThan: cleanupOlderThan.getTime(),
      durationMs: durationSince(startedAt),
      tables,
    };
  }

  async close(): Promise<void> {
    const connection = await this.getConnection();
    connection.close();
  }
}

export function utf8Field(name: string, nullable = false): Field<Utf8> {
  return new Field(name, new Utf8(), nullable);
}

export function booleanField(name: string, nullable = false): Field<Bool> {
  return new Field(name, new Bool(), nullable);
}

export function int32Field(name: string, nullable = false): Field<Int32> {
  return new Field(name, new Int32(), nullable);
}

export function int64Field(name: string, nullable = false): Field<Int64> {
  return new Field(name, new Int64(), nullable);
}

export function float64Field(name: string, nullable = false): Field<Float64> {
  return new Field(name, new Float64(), nullable);
}

export function timestampMsField(name: string, nullable = false): Field<TimestampMillisecond> {
  return new Field(name, new TimestampMillisecond(), nullable);
}

export function vectorField(name: string, dimensions: number, nullable = false): Field {
  return new Field(
    name,
    new FixedSizeList(dimensions, new Field("item", new Float32(), false)),
    nullable,
  );
}

export function schema(fields: Field[]): Schema {
  return new Schema(fields);
}
