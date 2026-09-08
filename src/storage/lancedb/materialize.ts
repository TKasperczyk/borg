import { DataType, type RecordBatch, type Vector } from "apache-arrow";

function isArrowVector(value: object): value is Vector {
  // The SDK loads Arrow via CJS while Borg uses ESM; instanceof Vector fails
  // across those two class identities even at the same package version.
  return (
    "get" in value &&
    typeof value.get === "function" &&
    "toArray" in value &&
    typeof value.toArray === "function" &&
    "length" in value &&
    typeof value.length === "number" &&
    Symbol.iterator in value
  );
}

// Arrow StructRow retains its parent batch; a list cell is itself an Arrow
// Vector. A spread/toJSON alone is shallow and still pins the backing buffers.
function detachValue(value: unknown, type?: DataType): unknown {
  if (value === null || typeof value !== "object") return value;
  if (value instanceof Date) return new Date(value.getTime());
  if (value instanceof DataView) {
    return new DataView(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
  }
  if (isArrowVector(value)) {
    if (
      DataType.isFixedSizeList(type) &&
      DataType.isFloat(type.valueType) &&
      value.nullCount === 0
    ) {
      // Arrow's toArray() can alias its batch. Copy the numeric view directly
      // into the Float32Array that repository hydration takes ownership of,
      // avoiding a boxed number[] and a second vector allocation on full scans.
      // Lance query schemas mark elements nullable even for dense vectors, so
      // use the actual null count rather than the schema's nullable flag.
      const values = value.toArray();
      if (values instanceof Float32Array || values instanceof Float64Array) {
        return new Float32Array(values);
      }
    }
    // Lists containing nulls or nested lists must retain their structure. In
    // particular, a TypedArray copy would silently turn null cells into zero.
    return Array.from(value, (cell) => detachValue(cell));
  }
  if (Array.isArray(value) || ArrayBuffer.isView(value)) {
    return Array.from(value as Iterable<unknown>, (cell) => detachValue(cell));
  }
  return Object.fromEntries(Object.entries(value).map(([key, cell]) => [key, detachValue(cell)]));
}

export async function materializeRows(
  batches: AsyncIterable<RecordBatch>,
): Promise<Record<string, unknown>[]> {
  const rows: Record<string, unknown>[] = [];
  // Consume to completion so native execution finishes, and release each batch
  // before requesting the next. No Arrow objects escape this boundary.
  for await (const batch of batches) {
    for (const row of batch) {
      rows.push(
        Object.fromEntries(
          batch.schema.fields.map(({ name, type }) => [name, detachValue(row[name], type)]),
        ),
      );
    }
  }
  return rows;
}
