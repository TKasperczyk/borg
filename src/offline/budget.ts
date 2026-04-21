import type { LLMClient, TokenUsageEvent, TokenUsageSink } from "../llm/index.js";
import { BudgetExceededError } from "../util/errors.js";

import type { OfflineProcessName } from "./types.js";

type BudgetTotals = Partial<Record<OfflineProcessName, number>>;

type BudgetBoundError = Error & {
  tokens_used?: number;
};

export class BudgetTracker {
  private readonly caps: BudgetTotals;
  private readonly totals: BudgetTotals = {};

  constructor(caps: BudgetTotals = {}) {
    this.caps = { ...caps };
  }

  createSink(processName: OfflineProcessName, cap = this.caps[processName]): TokenUsageSink {
    if (cap !== undefined) {
      this.caps[processName] = cap;
    }

    return async (event: TokenUsageEvent) => {
      const nextTotal = this.getTokensUsed(processName) + event.input_tokens + event.output_tokens;
      this.totals[processName] = nextTotal;
      const effectiveCap = this.caps[processName];

      if (effectiveCap !== undefined && nextTotal > effectiveCap) {
        throw new BudgetExceededError(
          `${processName} exceeded its token budget (${nextTotal}/${effectiveCap})`,
          {
            code: "OFFLINE_BUDGET_EXCEEDED",
          },
        );
      }
    };
  }

  getTokensUsed(processName: OfflineProcessName): number {
    return this.totals[processName] ?? 0;
  }
}

export function wrapLlmClientWithSink(client: LLMClient, sink: TokenUsageSink): LLMClient {
  return {
    async complete(options) {
      const result = await client.complete(options);
      await sink({
        budget: options.budget,
        model: options.model,
        input_tokens: result.input_tokens,
        output_tokens: result.output_tokens,
      });
      return result;
    },
  };
}

export async function withBudget<T>(
  processName: OfflineProcessName,
  cap: number,
  fn: (tools: {
    sink: TokenUsageSink;
    tracker: BudgetTracker;
    wrapClient: (client: LLMClient) => LLMClient;
  }) => Promise<T>,
): Promise<{
  result: T;
  tokens_used: number;
}> {
  const tracker = new BudgetTracker({
    [processName]: cap,
  });
  const sink = tracker.createSink(processName, cap);

  try {
    const result = await fn({
      sink,
      tracker,
      wrapClient: (client) => wrapLlmClientWithSink(client, sink),
    });

    return {
      result,
      tokens_used: tracker.getTokensUsed(processName),
    };
  } catch (error) {
    if (error instanceof Error) {
      (error as BudgetBoundError).tokens_used = tracker.getTokensUsed(processName);
    }

    throw error;
  }
}

export function getBudgetErrorTokens(error: unknown): number {
  return error instanceof Error && "tokens_used" in error && typeof error.tokens_used === "number"
    ? error.tokens_used
    : 0;
}
