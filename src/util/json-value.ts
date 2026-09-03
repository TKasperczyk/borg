import { z } from "zod";

export type JsonPrimitive = boolean | number | null | string;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

function isPlainObject(value: object): boolean {
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

export function assertJsonValue(value: unknown, path = "$"): asserts value is JsonValue {
  if (value === null) {
    return;
  }

  switch (typeof value) {
    case "boolean":
    case "string":
      return;
    case "number":
      if (!Number.isFinite(value)) {
        throw new TypeError(`${path} contains a non-finite number`);
      }

      return;
    case "undefined":
      throw new TypeError(`${path} contains undefined`);
    case "bigint":
      throw new TypeError(`${path} contains a bigint`);
    case "function":
      throw new TypeError(`${path} contains a function`);
    case "symbol":
      throw new TypeError(`${path} contains a symbol`);
    case "object":
      if (Array.isArray(value)) {
        for (const [index, item] of value.entries()) {
          assertJsonValue(item, `${path}[${index}]`);
        }

        return;
      }

      if (!isPlainObject(value)) {
        throw new TypeError(`${path} contains a non-plain object`);
      }

      for (const [key, nestedValue] of Object.entries(value)) {
        assertJsonValue(nestedValue, `${path}.${key}`);
      }

      return;
    default:
      throw new TypeError(`${path} contains an unsupported value`);
  }
}

export function serializeJsonValue(value: unknown): string {
  assertJsonValue(value);

  return JSON.stringify(value);
}

export const jsonValueSchema = z.custom<JsonValue>((value) => {
  try {
    assertJsonValue(value);
    return true;
  } catch {
    return false;
  }
});
