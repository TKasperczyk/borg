import { IdentityCasMismatchError } from "../../util/errors.js";

export type IdentityCasRecord = {
  record_version?: number;
};

export type IdentityCasOptions = {
  expectedVersion?: number;
};

type SqliteChangeResult = {
  changes: number | bigint;
};

export function expectedRecordVersion(
  record: IdentityCasRecord,
  options: IdentityCasOptions = {},
): number {
  // Missing versions mean a caller is applying a stale/unversioned snapshot.
  // Use a sentinel that cannot match persisted positive versions.
  return options.expectedVersion ?? record.record_version ?? -1;
}

export function nextRecordVersion(expectedVersion: number): number {
  return expectedVersion + 1;
}

export function assertIdentityCasUpdated(input: {
  result: SqliteChangeResult;
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
