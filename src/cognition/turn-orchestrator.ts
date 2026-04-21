import type { Config } from "../config/index.js";
import { SuppressionSet, computeRetrievalLimit, computeWeights } from "./attention/index.js";
import { performAction } from "./action/index.js";
import { Deliberator, type SelfSnapshot, type TurnStakes } from "./deliberation/deliberator.js";
import { Perceiver } from "./perception/index.js";
import { Reflector } from "./reflection/index.js";
import type { RetrievalPipeline, RetrievalSearchOptions } from "../retrieval/index.js";
import type { LLMClient } from "../llm/index.js";
import {
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
  type GoalRecord,
} from "../memory/self/index.js";
import { WorkingMemoryStore, type WorkingMemory } from "../memory/working/index.js";
import { EpisodicRepository } from "../memory/episodic/index.js";
import { StreamWriter } from "../stream/index.js";
import { ConfigError } from "../util/errors.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../util/ids.js";
import type { CognitiveMode, IntentRecord } from "./types.js";
import type { LLMToolCall } from "../llm/index.js";

function flattenGoals(goals: ReadonlyArray<GoalRecord & { children?: unknown }>): GoalRecord[] {
  const flattened: GoalRecord[] = [];
  const stack = [...goals];

  while (stack.length > 0) {
    const next = stack.shift();

    if (next === undefined) {
      continue;
    }

    flattened.push(next);

    if ("children" in next && Array.isArray(next.children)) {
      stack.push(...(next.children as GoalRecord[]));
    }
  }

  return flattened;
}

export type TurnInput = {
  userMessage: string;
  audience?: string;
  stakes?: TurnStakes;
  sessionId?: SessionId;
};

export type TurnResult = {
  mode: CognitiveMode;
  path: "system_1" | "system_2";
  response: string;
  thoughts: string[];
  usage: {
    input_tokens: number;
    output_tokens: number;
    stop_reason: string | null;
  };
  retrievedEpisodeIds: string[];
  intents: IntentRecord[];
  toolCalls: LLMToolCall[];
};

export type TurnOrchestratorOptions = {
  config: Config;
  retrievalPipeline: RetrievalPipeline;
  episodicRepository: EpisodicRepository;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  workingMemoryStore: WorkingMemoryStore;
  llmFactory: () => LLMClient;
  clock?: Clock;
  createStreamWriter: (sessionId: SessionId) => StreamWriter;
};

export class TurnOrchestrator {
  private readonly clock: Clock;

  constructor(private readonly options: TurnOrchestratorOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  loadWorkingMemory(sessionId: SessionId): WorkingMemory {
    return this.options.workingMemoryStore.load(sessionId);
  }

  clearWorkingMemory(sessionId: SessionId): void {
    this.options.workingMemoryStore.clear(sessionId);
  }

  private buildSelfSnapshot(): SelfSnapshot {
    const values = this.options.valuesRepository.list();
    const goals = flattenGoals(this.options.goalsRepository.list({ status: "active" }));
    const traits = this.options.traitsRepository.list();

    return {
      values,
      goals,
      traits,
    };
  }

  private getOptionalLlmClient(): LLMClient | undefined {
    try {
      return this.options.llmFactory();
    } catch (error) {
      if (error instanceof ConfigError) {
        return undefined;
      }

      throw error;
    }
  }

  private async appendFailureEvent(
    streamWriter: StreamWriter,
    error: unknown,
    sessionId: SessionId,
  ): Promise<void> {
    const message =
      error instanceof Error
        ? `${error.name}: ${error.message}`
        : `Unknown error: ${String(error)}`;

    try {
      await streamWriter.append({
        kind: "internal_event",
        content: `Turn failed for ${sessionId}: ${message}`,
      });
    } catch {
      // Best-effort logging only.
    }
  }

  async run(input: TurnInput): Promise<TurnResult> {
    const sessionId = input.sessionId ?? DEFAULT_SESSION_ID;
    const streamWriter = this.options.createStreamWriter(sessionId);

    try {
      try {
        let workingMemory = this.options.workingMemoryStore.load(sessionId);
        const optionalPerceptionLlm =
          this.options.config.perception.useLlmFallback === true
            ? this.getOptionalLlmClient()
            : undefined;
        const perceiver = new Perceiver({
          llmClient: optionalPerceptionLlm,
          model: this.options.config.anthropic.models.background,
          useLlmFallback: this.options.config.perception.useLlmFallback,
          clock: this.clock,
        });
        const llmClient = this.options.llmFactory();
        const selfSnapshot = this.buildSelfSnapshot();
        const perception = await perceiver.perceive(
          input.userMessage,
          workingMemory.recent_thoughts.slice(-3),
        );

        workingMemory = {
          ...workingMemory,
          turn_counter: workingMemory.turn_counter + 1,
          scratchpad: "",
          current_focus: perception.entities[0] ?? (input.userMessage.slice(0, 80) || null),
          hot_entities: perception.entities,
          mode: perception.mode,
          updated_at: this.clock.now(),
        };

        const suppressionSet = SuppressionSet.fromEntries(
          workingMemory.suppressed,
          workingMemory.turn_counter,
        );
        workingMemory = this.options.workingMemoryStore.save({
          ...workingMemory,
          suppressed: suppressionSet.snapshot(),
          updated_at: this.clock.now(),
        });

        const userEntry = {
          kind: "user_msg",
          content: input.userMessage,
          ...(input.audience === undefined ? {} : { audience: input.audience }),
        } satisfies Parameters<StreamWriter["append"]>[0];

        await streamWriter.append(userEntry);

        const attentionWeights = computeWeights(perception.mode, {
          currentGoals: selfSnapshot.goals,
          hasTemporalCue: perception.temporalCue !== null,
        });
        const retrievalOptions: RetrievalSearchOptions = {
          limit: computeRetrievalLimit(perception.mode),
          attentionWeights,
          goalDescriptions: selfSnapshot.goals.map((goal) => goal.description),
          temporalCue: perception.temporalCue,
          suppressionSet,
          timeRange:
            perception.temporalCue === null
              ? undefined
              : {
                  start: perception.temporalCue.sinceTs ?? Number.NEGATIVE_INFINITY,
                  end: perception.temporalCue.untilTs ?? this.clock.now(),
                },
        };
        const retrievedEpisodes = await this.options.retrievalPipeline.search(
          input.userMessage,
          retrievalOptions,
        );
        const deliberator = new Deliberator({
          llmClient,
          cognitionModel: this.options.config.anthropic.models.cognition,
          backgroundModel: this.options.config.anthropic.models.background,
        });
        const deliberation = await deliberator.run(
          {
            sessionId,
            audience: input.audience,
            userMessage: input.userMessage,
            perception,
            retrievalResult: retrievedEpisodes,
            workingMemory,
            selfSnapshot,
            options: {
              stakes: input.stakes,
            },
            reRetrieve: (query, overrides = {}) =>
              this.options.retrievalPipeline.search(query, {
                ...retrievalOptions,
                ...overrides,
              }),
          },
          streamWriter,
        );

        workingMemory = {
          ...workingMemory,
          scratchpad: deliberation.thoughts.join("\n"),
          updated_at: this.clock.now(),
        };

        const actionResult = await performAction(
          {
            response: deliberation.response,
            toolCalls: deliberation.tool_calls,
            audience: input.audience,
            perception,
            workingMemory,
          },
          streamWriter,
        );
        const reflector = new Reflector({
          clock: this.clock,
        });
        const reflectedWorkingMemory = await reflector.reflect(
          {
            userMessage: input.userMessage,
            workingMemory: actionResult.workingMemory,
            selfSnapshot,
            deliberationResult: deliberation,
            actionResult,
            retrievedEpisodes: deliberation.retrievedEpisodes,
            episodicRepository: this.options.episodicRepository,
            goalsRepository: this.options.goalsRepository,
            traitsRepository: this.options.traitsRepository,
            suppressionSet,
          },
          streamWriter,
        );

        this.options.workingMemoryStore.save({
          ...reflectedWorkingMemory,
          suppressed: suppressionSet.snapshot(),
          updated_at: this.clock.now(),
        });

        return {
          mode: perception.mode,
          path: deliberation.path,
          response: actionResult.response,
          thoughts: deliberation.thoughts,
          usage: deliberation.usage,
          retrievedEpisodeIds: deliberation.retrievedEpisodes.map((result) => result.episode.id),
          intents: actionResult.intents,
          toolCalls: [...actionResult.tool_calls],
        };
      } catch (error) {
        await this.appendFailureEvent(streamWriter, error, sessionId);
        throw error;
      }
    } finally {
      streamWriter.close();
    }
  }
}
