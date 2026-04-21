export type SuppressionEntry = {
  id: string;
  reason: string;
  until_turn: number;
};

export class SuppressionSet {
  static fromEntries(entries: readonly SuppressionEntry[], currentTurn: number): SuppressionSet {
    return new SuppressionSet(currentTurn, entries);
  }

  private readonly entries = new Map<string, SuppressionEntry>();
  private currentTurn: number;

  constructor(currentTurn = 0, entries: readonly SuppressionEntry[] = []) {
    this.currentTurn = Math.max(0, Math.floor(currentTurn));

    for (const entry of entries) {
      if (entry.until_turn < this.currentTurn) {
        continue;
      }

      this.entries.set(entry.id, {
        id: entry.id,
        reason: entry.reason,
        until_turn: entry.until_turn,
      });
    }
  }

  setCurrentTurn(currentTurn: number): void {
    this.currentTurn = Math.max(0, Math.floor(currentTurn));
    this.pruneExpired();
  }

  suppress(id: string, reason: string, ttlTurns: number): void {
    const turns = Math.max(1, Math.floor(ttlTurns));
    this.entries.set(id, {
      id,
      reason,
      until_turn: this.currentTurn + turns,
    });
  }

  isSuppressed(id: string): boolean {
    const entry = this.entries.get(id);

    return entry !== undefined && entry.until_turn >= this.currentTurn;
  }

  reasonFor(id: string): string | undefined {
    return this.entries.get(id)?.reason;
  }

  tickTurn(): void {
    this.currentTurn += 1;
    this.pruneExpired();
  }

  size(): number {
    return this.entries.size;
  }

  snapshot(): SuppressionEntry[] {
    this.pruneExpired();
    return [...this.entries.values()].map((entry) => ({
      id: entry.id,
      reason: entry.reason,
      until_turn: entry.until_turn,
    }));
  }

  private pruneExpired(): void {
    for (const [id, entry] of this.entries) {
      if (entry.until_turn < this.currentTurn) {
        this.entries.delete(id);
      }
    }
  }
}
