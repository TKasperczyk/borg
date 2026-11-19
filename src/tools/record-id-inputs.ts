import type { ToolDefinition } from "./dispatcher.js";
import { isPlainRecord } from "../util/guards.js";
import type { RecordIdKind, RecordIdResolver } from "./record-id-resolver.js";

export type RecordIdInput = {
  /** Structural field path; '*' addresses array elements. */
  path: string;
  kinds: readonly RecordIdKind[];
  /** Reference-only create inputs need existence checks for full IDs as well. */
  requireExisting?: boolean;
  /** Missing prefixes use the tool's normal absent-record result. */
  notFound: { error: (id: string) => Error } | { output: (id: string) => unknown } | "invoke";
};

export type ToolIdResolution = { field: string; input: string; resolved_id: string };
export type MissingToolId = { field: string; input: string; spec: RecordIdInput };

export function resolveToolRecordIds(
  tool: ToolDefinition,
  input: unknown,
  resolver: RecordIdResolver | undefined,
  resolutions: ToolIdResolution[],
): { input: unknown; missing: MissingToolId[] } {
  if (resolver === undefined || tool.recordIdInputs === undefined) return { input, missing: [] };
  const missing: MissingToolId[] = [];
  let resolved = input;
  for (const spec of tool.recordIdInputs) {
    const visit = (value: unknown, path: string[], field: string[]): unknown => {
      if (path.length === 0) {
        if (typeof value !== "string") return value;
        const result = resolver.resolve(value, spec.kinds, spec.requireExisting);
        if (result.status === "not_found")
          missing.push({ field: field.join("."), input: value, spec });
        if (result.status === "resolved")
          resolutions.push({ field: field.join("."), input: value, resolved_id: result.id });
        return result.id;
      }
      const [key, ...rest] = path;
      if (key === "*" && Array.isArray(value))
        return value.map((item, index) => visit(item, rest, [...field, String(index)]));
      if (key !== undefined && isPlainRecord(value) && Object.hasOwn(value, key)) {
        return { ...value, [key]: visit(value[key], rest, [...field, key]) };
      }
      return value;
    };
    resolved = visit(resolved, spec.path.split("."), []);
  }
  return { input: resolved, missing };
}

export function missingToolIdOutput(missing: MissingToolId): unknown {
  const outcome = missing.spec.notFound;
  if (outcome === "invoke") throw new Error("Missing record must be handled by the tool");
  if ("output" in outcome) return outcome.output(missing.input);
  throw outcome.error(missing.input);
}
