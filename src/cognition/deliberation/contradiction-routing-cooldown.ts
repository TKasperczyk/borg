export type ContradictionRoutingCooldownHit = {
  fingerprint: string;
  lastForcedTurn: number;
  currentTurn: number;
  cooldownTurns: number;
};

export class ContradictionRoutingCooldown {
  private readonly lastForcedTurnByKey = new Map<string, number>();

  getCoolingFingerprints(input: {
    audience: string;
    fingerprints: readonly string[];
    currentTurn: number;
    cooldownTurns: number;
  }): ContradictionRoutingCooldownHit[] {
    const cooldownTurns = Math.max(0, Math.floor(input.cooldownTurns));

    if (cooldownTurns === 0) {
      return [];
    }

    return [...new Set(input.fingerprints)]
      .map((fingerprint) => {
        const lastForcedTurn = this.lastForcedTurnByKey.get(this.key(input.audience, fingerprint));

        if (
          lastForcedTurn === undefined ||
          input.currentTurn - lastForcedTurn > cooldownTurns
        ) {
          return null;
        }

        return {
          fingerprint,
          lastForcedTurn,
          currentTurn: input.currentTurn,
          cooldownTurns,
        };
      })
      .filter((hit): hit is ContradictionRoutingCooldownHit => hit !== null);
  }

  recordForced(input: {
    audience: string;
    fingerprints: readonly string[];
    currentTurn: number;
  }): void {
    for (const fingerprint of new Set(input.fingerprints)) {
      this.lastForcedTurnByKey.set(this.key(input.audience, fingerprint), input.currentTurn);
    }
  }

  private key(audience: string, fingerprint: string): string {
    return `${audience}\0${fingerprint}`;
  }
}
