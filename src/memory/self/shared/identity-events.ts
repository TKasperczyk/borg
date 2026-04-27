import { type IdentityEventRepository } from "../../identity/repository.js";

type IdentityEventInput = Parameters<IdentityEventRepository["record"]>[0];

export function recordIdentityEvent(
  repository: IdentityEventRepository | undefined,
  input: IdentityEventInput,
): void {
  repository?.record(input);
}

// Sprint 56: identity-bearing record writes (values/goals/traits/commitments)
// must commit the row mutation and the audit event together. Wrap both in
// a single SQLite transaction when the audit repository is available; fall
// back to a no-op wrapper for code paths without an audit repo (tests and
// some offline harnesses) so the behavior is unchanged for them.
export function runIdentityWrite<T>(
  repository: IdentityEventRepository | undefined,
  callback: () => T,
): T {
  if (repository === undefined) {
    return callback();
  }
  return repository.runInTransaction(callback);
}
