import {
  connect,
  makeArrowTable,
  type AddColumnsSql,
  type Connection,
  type IntoVector,
  type SchemaLike,
  type Table,
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

import { StorageError } from "../../util/errors.js";

export type LanceDbRow = Record<string, unknown>;

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

export class LanceDbTable {
  constructor(private readonly table: Table) {}

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

  close(): void {
    this.table.close();
  }
}

export class LanceDbStore {
  private readonly connectionPromise: Promise<Connection>;

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

  async openTable(options: LanceDbOpenTableOptions): Promise<LanceDbTable> {
    const connection = await this.getConnection();

    try {
      const tableNames = await connection.tableNames();

      if (tableNames.includes(options.name)) {
        const table = await connection.openTable(options.name);
        await ensureSchemaCompatibility(table, options.schema, options.name);
        await table.checkoutLatest();
        table.close();
        const reopenedTable = await connection.openTable(options.name);
        await reopenedTable.checkoutLatest();
        return new LanceDbTable(reopenedTable);
      }

      return new LanceDbTable(
        await connection.createEmptyTable(options.name, options.schema, {
          mode: "create",
          existOk: true,
        }),
      );
    } catch (error) {
      if (error instanceof StorageError) {
        throw error;
      }

      try {
        return new LanceDbTable(await connection.openTable(options.name));
      } catch {
        throw new StorageError(`Failed to open LanceDB table ${options.name}`, {
          cause: error,
        });
      }
    }
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
