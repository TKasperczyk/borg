/** Reporting must not reject live work, but a failed report is still a failed drain. */
export class FailureReportTracker {
  private readonly pending = new Set<Promise<void>>();
  private readonly failures: unknown[] = [];

  constructor(private readonly label: string) {}

  async report(write: () => unknown | Promise<unknown>): Promise<void> {
    const task = (async () => {
      try {
        await write();
      } catch (error) {
        this.failures.push(error);
        console.error(`Failed to report ${this.label}`, error);
      }
    })();
    this.pending.add(task);
    try {
      await task;
    } finally {
      this.pending.delete(task);
    }
  }

  async flush(): Promise<void> {
    while (this.pending.size > 0) {
      await Promise.all([...this.pending]);
    }
    // Keep failures after a maintenance flush: a later shutdown must not claim
    // a successful drain merely because an earlier caller already saw the error.
    if (this.failures.length > 0) {
      throw new AggregateError([...this.failures], `Failed to report ${this.label}`);
    }
  }
}
