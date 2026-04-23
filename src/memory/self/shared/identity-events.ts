import { type IdentityEventRepository } from "../../identity/repository.js";

type IdentityEventInput = Parameters<IdentityEventRepository["record"]>[0];

export function recordIdentityEvent(
  repository: IdentityEventRepository | undefined,
  input: IdentityEventInput,
): void {
  repository?.record(input);
}
