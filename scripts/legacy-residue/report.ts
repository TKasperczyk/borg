import { createHash } from "node:crypto";
import { StorageError } from "../../src/util/errors.js";

export class ResidueReportError extends StorageError {}

export type ResidueCheck = {
  id: string;
  description: string;
  query: string;
  count: number | null;
  total?: number;
  sample?: string[];
};

export type ResidueReport = {
  tenant: string;
  generated_at: string;
  checks: ResidueCheck[];
};

export type CheckCount = Pick<ResidueCheck, "count" | "total" | "sample">;

export function redactId(id: unknown): string {
  return `sha256:${createHash("sha256").update(String(id)).digest("hex").slice(0, 12)}`;
}

export function counter() {
  const result = { count: 0, sample: [] as string[] };
  return {
    result,
    add(id: unknown) {
      result.count += 1;
      if (result.sample.length < 3) result.sample.push(redactId(id));
    },
  };
}

export function objectValue(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function parseStoredJson(value: unknown): unknown {
  if (typeof value !== "string") throw new ResidueReportError("Expected stored JSON text");
  try {
    return JSON.parse(value) as unknown;
  } catch {
    throw new ResidueReportError("Malformed stored JSON; count unavailable");
  }
}

// Native errors and Zod errors can contain paths, IDs, or entire stored values.
export function safeFailure(error: unknown): string {
  return error instanceof ResidueReportError
    ? error.message
    : "Read or validation failed; stored values and native error details suppressed";
}

export class ReportChecks {
  readonly checks: ResidueCheck[] = [];

  constructor(private readonly diagnostic: (message: string) => void) {}

  async run(
    id: string,
    description: string,
    query: string,
    read: () => CheckCount | Promise<CheckCount>,
  ): Promise<void> {
    try {
      this.checks.push({ id, description, query, ...(await read()) });
    } catch (error) {
      const reason = safeFailure(error);
      this.checks.push({ id, description, query: `${query}\nUNAVAILABLE: ${reason}`, count: null });
      this.diagnostic(`${id}: ${reason}`);
    }
  }

  skip(id: string, description: string, reason: string): void {
    this.checks.push({ id, description, query: `SKIPPED: ${reason}`, count: null });
    this.diagnostic(`${id}: ${reason}`);
  }
}
