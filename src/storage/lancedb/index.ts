import {
  connect,
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

export class LanceDbTable {
  constructor(private readonly table: Table) {}

  async upsert(rows: readonly LanceDbRow[], options: LanceDbUpsertOptions): Promise<void> {
    if (rows.length === 0) {
      return;
    }

    try {
      await this.table
        .mergeInsert(options.on)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute([...rows]);
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
        return new LanceDbTable(await connection.openTable(options.name));
      }

      return new LanceDbTable(
        await connection.createEmptyTable(options.name, options.schema, {
          mode: "create",
          existOk: true,
        }),
      );
    } catch (error) {
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
