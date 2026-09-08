import type { RecordBatch } from "apache-arrow";

function isArrowVector(value: object): value is Iterable<unknown> {
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
function detachValue(value: unknown): unknown {
  if (value === null || typeof value !== "object") return value;
  if (value instanceof Date) return new Date(value.getTime());
  if (value instanceof DataView) {
    return new DataView(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength));
  }
  if (isArrowVector(value) || Array.isArray(value) || ArrayBuffer.isView(value)) {
    return Array.from(value as Iterable<unknown>, detachValue);
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
      rows.push(detachValue(row) as Record<string, unknown>);
    }
  }
  return rows;
}
