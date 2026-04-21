import type { GoalRecord, TraitRecord, ValueRecord } from "../../memory/self/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { LLMClient, LLMToolCall } from "../../llm/index.js";
import type { RetrievedEpisode, RetrievalSearchOptions } from "../../retrieval/index.js";
import { StreamWriter } from "../../stream/index.js";
import { type CognitiveMode, type PerceptionResult } from "../types.js";
import type { SessionId } from "../../util/ids.js";
import { tokenizeText } from "../../util/text/tokenize.js";

export type TurnStakes = "low" | "medium" | "high";

export type SelfSnapshot = {
  values: ValueRecord[];
  goals: GoalRecord[];
  traits: TraitRecord[];
};

export type DeliberationContext = {
  sessionId: SessionId;
  audience?: string;
  userMessage: string;
  perception: PerceptionResult;
  retrievalResult: RetrievedEpisode[];
  workingMemory: WorkingMemory;
  selfSnapshot: SelfSnapshot;
  options?: {
    stakes?: TurnStakes;
    maxThinkingTokens?: number;
  };
  reRetrieve?: (query: string, options?: RetrievalSearchOptions) => Promise<RetrievedEpisode[]>;
};

export type DeliberationUsage = {
  input_tokens: number;
  output_tokens: number;
  stop_reason: string | null;
};

export type DeliberationResult = {
  path: "system_1" | "system_2";
  response: string;
  thoughts: string[];
  tool_calls: LLMToolCall[];
  usage: DeliberationUsage;
  decision_reason: string;
  retrievedEpisodes: RetrievedEpisode[];
  thoughtsPersisted: boolean;
};

export type DeliberatorOptions = {
  llmClient: LLMClient;
  cognitionModel: string;
  backgroundModel: string;
};

function aggregateUsage(
  current: DeliberationUsage,
  next: {
    input_tokens: number;
    output_tokens: number;
    stop_reason: string | null;
  },
): DeliberationUsage {
  return {
    input_tokens: current.input_tokens + next.input_tokens,
    output_tokens: current.output_tokens + next.output_tokens,
    stop_reason: next.stop_reason,
  };
}

function averageConfidence(results: readonly RetrievedEpisode[]): number {
  if (results.length === 0) {
    return 0;
  }

  const total = results.reduce((sum, result) => sum + result.score, 0);
  return total / results.length;
}

function hasContradictionSignal(
  userMessage: string,
  retrievedEpisodes: readonly RetrievedEpisode[],
): boolean {
  if (/\b(but|however|actually|though|yet|instead)\b/i.test(userMessage)) {
    return true;
  }

  const warnings = retrievedEpisodes.filter((result) => result.episode.tags.includes("warning"));
  const recommendations = retrievedEpisodes.filter((result) =>
    result.episode.tags.includes("recommended"),
  );

  for (const warning of warnings) {
    const warningTokens = tokenizeText(
      `${warning.episode.title} ${warning.episode.tags.join(" ")}`,
    );

    for (const recommendation of recommendations) {
      const recommendationTokens = tokenizeText(
        `${recommendation.episode.title} ${recommendation.episode.tags.join(" ")}`,
      );
      const overlap = [...warningTokens].some((token) => recommendationTokens.has(token));

      if (overlap) {
        return true;
      }
    }
  }

  return false;
}

function chooseDeliberationPath(
  mode: CognitiveMode,
  stakes: TurnStakes,
  retrievedEpisodes: readonly RetrievedEpisode[],
  userMessage: string,
): {
  path: "system_1" | "system_2";
  reason: string;
} {
  const confidence = averageConfidence(retrievedEpisodes);

  if (mode === "idle") {
    return {
      path: "system_1",
      reason: "Idle mode keeps the response on the cheap path.",
    };
  }

  if (mode === "reflective") {
    return {
      path: "system_2",
      reason: "Reflective mode always takes the deeper reasoning path.",
    };
  }

  if (hasContradictionSignal(userMessage, retrievedEpisodes)) {
    return {
      path: "system_2",
      reason: "Contradiction heuristic triggered deeper reasoning.",
    };
  }

  if (stakes === "high") {
    return {
      path: "system_2",
      reason: "High-stakes request requires explicit planning.",
    };
  }

  if (confidence < 0.45) {
    return {
      path: "system_2",
      reason: "Low retrieval confidence triggered deeper reasoning.",
    };
  }

  return {
    path: "system_1",
    reason: "Retrieval confidence is strong enough for a direct response.",
  };
}

function summarizeIdentity(selfSnapshot: SelfSnapshot): string {
  const values = selfSnapshot.values.map((value) => value.label).join(", ") || "none";
  const goals = selfSnapshot.goals.map((goal) => goal.description).join(" | ") || "none";
  const traits =
    selfSnapshot.traits.map((trait) => `${trait.label}:${trait.strength.toFixed(2)}`).join(", ") ||
    "none";

  return `Values: ${values}\nGoals: ${goals}\nTraits: ${traits}`;
}

function summarizeWorkingMemory(workingMemory: WorkingMemory): string {
  return [
    `Current focus: ${workingMemory.current_focus ?? "none"}`,
    `Recent thoughts: ${workingMemory.recent_thoughts.slice(-3).join(" | ") || "none"}`,
    `Hot entities: ${workingMemory.hot_entities.join(", ") || "none"}`,
  ].join("\n");
}

function summarizeRetrievedEpisodes(
  label: string,
  retrievedEpisodes: readonly RetrievedEpisode[],
): string {
  if (retrievedEpisodes.length === 0) {
    return `${label}: none`;
  }

  return [
    `${label}:`,
    ...retrievedEpisodes.map(
      (result) =>
        `- ${result.episode.title} [score=${result.score.toFixed(2)}] tags=${result.episode.tags.join(",")}`,
    ),
  ].join("\n");
}

export class Deliberator {
  constructor(private readonly options: DeliberatorOptions) {}

  private async persistThoughts(
    streamWriter: StreamWriter | undefined,
    thoughts: readonly string[],
  ): Promise<boolean> {
    if (streamWriter === undefined || thoughts.length === 0) {
      return false;
    }

    await streamWriter.appendMany(
      thoughts.map((thought) => ({
        kind: "thought",
        content: thought,
      })),
    );
    return true;
  }

  async run(
    context: DeliberationContext,
    streamWriter?: StreamWriter,
  ): Promise<DeliberationResult> {
    const stakes = context.options?.stakes ?? "low";
    const decision = chooseDeliberationPath(
      context.perception.mode,
      stakes,
      context.retrievalResult,
      context.userMessage,
    );
    const baseSystemPrompt = [
      "You are Borg, an agent with explicit memory and identity.",
      summarizeIdentity(context.selfSnapshot),
      summarizeWorkingMemory(context.workingMemory),
      summarizeRetrievedEpisodes("Retrieved context", context.retrievalResult),
    ].join("\n\n");

    if (decision.path === "system_1") {
      const response = await this.options.llmClient.complete({
        model: this.options.cognitionModel,
        system: baseSystemPrompt,
        messages: [
          {
            role: "user",
            content: context.userMessage,
          },
        ],
        max_tokens: context.options?.maxThinkingTokens ?? 600,
        budget: "cognition-system-1",
      });

      return {
        path: "system_1",
        response: response.text,
        thoughts: [],
        tool_calls: response.tool_calls,
        usage: {
          input_tokens: response.input_tokens,
          output_tokens: response.output_tokens,
          stop_reason: response.stop_reason,
        },
        decision_reason: decision.reason,
        retrievedEpisodes: [...context.retrievalResult],
        thoughtsPersisted: false,
      };
    }

    const planning = await this.options.llmClient.complete({
      model: this.options.backgroundModel,
      system:
        "Think briefly about what the assistant should verify, clarify, or compare before answering. Return plain text.",
      messages: [
        {
          role: "user",
          content: [
            `User message: ${context.userMessage}`,
            summarizeRetrievedEpisodes("Initial retrieval", context.retrievalResult),
            `Mode: ${context.perception.mode}`,
          ].join("\n\n"),
        },
      ],
      max_tokens: Math.min(context.options?.maxThinkingTokens ?? 256, 256),
      budget: "cognition-plan",
    });
    const scratchpad = planning.text.trim();
    const thoughts = scratchpad.length === 0 ? [] : [scratchpad];
    const secondaryRetrieval =
      scratchpad.length > 0 && context.reRetrieve !== undefined
        ? await context.reRetrieve(scratchpad, {
            limit: 3,
          })
        : [];
    const finalResponse = await this.options.llmClient.complete({
      model: this.options.cognitionModel,
      system: [
        baseSystemPrompt,
        summarizeRetrievedEpisodes("Additional retrieval", secondaryRetrieval),
        `Scratchpad:\n${scratchpad || "none"}`,
      ].join("\n\n"),
      messages: [
        {
          role: "user",
          content: context.userMessage,
        },
      ],
      max_tokens: context.options?.maxThinkingTokens ?? 700,
      budget: "cognition-system-2",
    });
    const thoughtsPersisted = await this.persistThoughts(streamWriter, thoughts);
    const usage = aggregateUsage(
      {
        input_tokens: planning.input_tokens,
        output_tokens: planning.output_tokens,
        stop_reason: planning.stop_reason,
      },
      finalResponse,
    );

    return {
      path: "system_2",
      response: finalResponse.text,
      thoughts,
      tool_calls: finalResponse.tool_calls,
      usage,
      decision_reason: decision.reason,
      retrievedEpisodes: dedupeRetrievedEpisodes([
        ...context.retrievalResult,
        ...secondaryRetrieval,
      ]),
      thoughtsPersisted,
    };
  }
}

function dedupeRetrievedEpisodes(results: readonly RetrievedEpisode[]): RetrievedEpisode[] {
  const seen = new Set<string>();
  const deduped: RetrievedEpisode[] = [];

  for (const result of results) {
    if (seen.has(result.episode.id)) {
      continue;
    }

    seen.add(result.episode.id);
    deduped.push(result);
  }

  return deduped;
}
