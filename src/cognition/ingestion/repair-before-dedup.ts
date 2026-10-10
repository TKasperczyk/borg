import { StreamError } from "../../util/errors.js";
import type { SessionId } from "../../util/ids.js";

/** Repair committed stream facts before deciding whether an enqueue is a duplicate. */
export async function repairPoisonedSessionBeforeDedup(
  sessionId: SessionId,
  repairSession: (sessionId: SessionId) => Promise<unknown>,
): Promise<void> {
  try {
    await repairSession(sessionId);
  } catch (error) {
    throw new StreamError(`Stream entry index is poisoned for committed session ${sessionId}`, {
      cause: error,
      code: "STREAM_INDEX_POISONED",
    });
  }
}
