import type BetterSqlite3 from "better-sqlite3";

import { IdentityCasMismatchError } from "../../util/errors.js";

export type IdentityCasRecord = {
  record_version?: number;
};

export type IdentityCasOptions = {
  expectedVersion?: number;
};

export function expectedRecordVersion(
  record: IdentityCasRecord,
  options: IdentityCasOptions = {},
): number {
  return options.expectedVersion ?? record.record_version ?? 1;
}

export function nextRecordVersion(expectedVersion: number): number {
  return expectedVersion + 1;
}

export function assertIdentityCasUpdated(input: {
  result: Pick<BetterSqlite3.RunResult, "changes">;
  recordType: string;
  recordId: string;
  expectedVersion: number;
}): void {
  if (input.result.changes > 0) {
    return;
  }

  throw new IdentityCasMismatchError({
    recordType: input.recordType,
    recordId: input.recordId,
    expectedVersion: input.expectedVersion,
  });
}
