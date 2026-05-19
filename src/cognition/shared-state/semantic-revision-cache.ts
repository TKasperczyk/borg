import { createHash } from "node:crypto";

import type { SemanticNode } from "../../memory/semantic/index.js";

export const SEMANTIC_REVISION_VERDICT_CACHE_MAX_ENTRIES = 1_000;

export type SemanticRevisionCachedVerdict = {
  verdict: "keep" | "uncertain";
  entry_text_hash: string;
  candidate_status_at_review: SemanticNode["status"];
  candidate_updated_at_at_review: number;
  last_reviewed_at_turn: number;
};

function semanticRevisionVerdictCacheKey(input: {
  artifactEntryId: string;
  candidateNodeId: string;
}): string {
  return `${input.artifactEntryId}:${input.candidateNodeId}`;
}

export function semanticRevisionEntryTextHash(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

export class SemanticRevisionVerdictCache {
  private readonly records = new Map<string, SemanticRevisionCachedVerdict>();

  constructor(private readonly maxEntries = SEMANTIC_REVISION_VERDICT_CACHE_MAX_ENTRIES) {}

  get size(): number {
    return this.records.size;
  }

  get(input: {
    artifactEntryId: string;
    candidateNodeId: string;
  }): SemanticRevisionCachedVerdict | null {
    const key = semanticRevisionVerdictCacheKey(input);
    const cached = this.records.get(key);

    if (cached === undefined) {
      return null;
    }

    this.records.delete(key);
    this.records.set(key, cached);
    return cached;
  }

  set(input: {
    artifactEntryId: string;
    candidateNodeId: string;
    value: SemanticRevisionCachedVerdict;
  }): void {
    const key = semanticRevisionVerdictCacheKey(input);

    this.records.delete(key);
    this.records.set(key, input.value);

    while (this.records.size > this.maxEntries) {
      const oldestKey = this.records.keys().next().value as string | undefined;

      if (oldestKey === undefined) {
        break;
      }

      this.records.delete(oldestKey);
    }
  }

  clear(): void {
    this.records.clear();
  }
}

export const semanticRevisionVerdictCache = new SemanticRevisionVerdictCache();
let semanticRevisionFallbackTurnCounter = 0;

export function semanticRevisionReviewTurn(inputTurnCounter: number | undefined): number {
  if (
    inputTurnCounter !== undefined &&
    Number.isFinite(inputTurnCounter) &&
    inputTurnCounter >= 0
  ) {
    return Math.floor(inputTurnCounter);
  }

  semanticRevisionFallbackTurnCounter += 1;
  return semanticRevisionFallbackTurnCounter;
}

export function sharedStateSemanticRevisionVerdictCacheSize(): number {
  return semanticRevisionVerdictCache.size;
}

export function clearSharedStateSemanticRevisionVerdictCache(): void {
  semanticRevisionVerdictCache.clear();
  semanticRevisionFallbackTurnCounter = 0;
}
