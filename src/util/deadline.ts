export type DeadlineSettlement =
  | { status: "completed" }
  | { status: "timed_out" }
  | { status: "error"; error: unknown };

/** Observe work until a caller-owned deadline; late rejections remain handled. */
export async function settleBeforeDeadline(
  work: () => Promise<void>,
  deadlineReached: Promise<void>,
): Promise<DeadlineSettlement> {
  const completed = (async (): Promise<DeadlineSettlement> => {
    try {
      await work();
      return { status: "completed" };
    } catch (error) {
      return { status: "error", error };
    }
  })();
  const deadline = (async (): Promise<DeadlineSettlement> => {
    await deadlineReached;
    return { status: "timed_out" };
  })();
  return await Promise.race([completed, deadline]);
}
