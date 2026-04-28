import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  Borg,
  DEFAULT_CONFIG,
  DEFAULT_SESSION_ID,
  FakeLLMClient,
  type GoalRecord,
  loadConfig,
  ManualClock,
  type ReviewQueueItem,
  type BorgOpenOptions,
  type Config,
  type LLMCompleteOptions,
  type LLMCompleteResult,
  type LLMContentBlock,
  type LLMConverseOptions,
  type LLMConverseResult,
  type LLMToolCall,
} from "../src/index.js";
import {
  createSemanticNodeId,
  parseSessionId,
  type GoalId,
  type SessionId,
} from "../src/util/ids.js";
import { createEvalBorg } from "../eval/support/create-eval-borg.js";

import { latestTurnId, readTraceEvents, summarizeTraceFile } from "./trace-reader.js";
import type { DeepPartial, Scenario, TracePhase, TraceRecord } from "./types.js";

export type ChatWithBorgResult = {
  response: string;
  turnId: string;
  sessionId?: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
  moodAfter?: {
    valence: number;
    arousal: number;
  };
  toolCalls: readonly {
    name: string;
    ok: boolean;
  }[];
};

export type SeededGoalProgressEvidence = {
  goal: GoalRecord | null;
  reviewPatch: {
    itemId: number;
    progressNotes: string;
    lastProgressTs: number | null;
  } | null;
};

export type BorgTransportOptions = {
  runId: string;
  scenario: Scenario;
  keep?: boolean;
  mock?: boolean;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
  tracePath?: string;
  llmClient?: BorgOpenOptions["llmClient"];
  embeddingClient?: BorgOpenOptions["embeddingClient"];
  clock?: ManualClock;
};

const MOCK_RESPONSE_COUNT = 400;
const DEFAULT_BORG_STAKES = "low";
const OPEN_HOOK_SETTLE_MS = 100;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function sanitizePathPart(value: string): string {
  return value.replace(/[^A-Za-z0-9_.-]/g, "-");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function goalProgressPatchFromReview(
  item: ReviewQueueItem,
  goalId: GoalRecord["id"],
): SeededGoalProgressEvidence["reviewPatch"] {
  if (item.kind !== "identity_inconsistency") {
    return null;
  }

  if (item.refs.target_type !== "goal" || item.refs.target_id !== goalId) {
    return null;
  }

  const patch = item.refs.patch;

  if (!isRecord(patch) || typeof patch.progress_notes !== "string") {
    return null;
  }

  return {
    itemId: item.id,
    progressNotes: patch.progress_notes,
    lastProgressTs: typeof patch.last_progress_ts === "number" ? patch.last_progress_ts : null,
  };
}

function deepMerge<T>(base: T, override: DeepPartial<T> | undefined): T {
  if (override === undefined) {
    return base;
  }

  if (Array.isArray(base) || Array.isArray(override) || !isRecord(base) || !isRecord(override)) {
    return override as T;
  }

  const merged: Record<string, unknown> = { ...base };

  for (const [key, value] of Object.entries(override)) {
    const current = (base as Record<string, unknown>)[key];
    merged[key] = deepMerge(current, value as never);
  }

  return merged as T;
}

function disabledBackgroundOverrides(): DeepPartial<Config> {
  return {
    maintenance: {
      enabled: false,
    },
    autonomy: {
      enabled: false,
    },
  };
}

function createRealConfig(input: {
  dataDir: string;
  env: NodeJS.ProcessEnv;
  scenario: Scenario;
}): Config {
  const loaded = loadConfig({
    dataDir: input.dataDir,
    env: {
      ...input.env,
      BORG_DATA_DIR: input.dataDir,
    },
  });
  const withBackgroundDisabled = deepMerge(loaded, disabledBackgroundOverrides());

  return deepMerge(withBackgroundDisabled, {
    ...input.scenario.borgConfigOverrides,
    dataDir: input.dataDir,
  });
}

function contentBlockText(block: LLMContentBlock): string {
  if (block.type === "text") {
    return block.text;
  }

  if (block.type === "tool_use") {
    return JSON.stringify(block.input);
  }

  return typeof block.content === "string"
    ? block.content
    : block.content.map((entry) => entry.text).join("\n");
}

function requestText(options: LLMCompleteOptions | LLMConverseOptions): string {
  return options.messages
    .map((message) => {
      if (typeof message.content === "string") {
        return message.content;
      }

      return message.content.map((block) => contentBlockText(block)).join("\n");
    })
    .join("\n");
}

function latestMessageText(options: LLMCompleteOptions | LLMConverseOptions): string {
  const last = options.messages[options.messages.length - 1];

  if (last === undefined) {
    return "";
  }

  if (typeof last.content === "string") {
    return last.content;
  }

  return last.content.map((block) => contentBlockText(block)).join("\n");
}

function hasToolResult(options: LLMCompleteOptions | LLMConverseOptions): boolean {
  const last = options.messages[options.messages.length - 1];

  if (last === undefined || typeof last.content === "string") {
    return false;
  }

  return last.content.some((block) => block.type === "tool_result");
}

function toolNames(options: LLMCompleteOptions | LLMConverseOptions): string[] {
  return options.tools?.map((tool) => tool.name) ?? [];
}

function completeWithTool(call: LLMToolCall): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 20,
    output_tokens: 8,
    stop_reason: "tool_use",
    tool_calls: [call],
  };
}

function extractGoalId(text: string): string | null {
  return text.match(/\bgoal_[a-z0-9]{16}\b/)?.[0] ?? null;
}

function reflectionForRequest(text: string): LLMCompleteResult {
  const goalId = extractGoalId(text);
  const advancedGoals =
    goalId !== null && /bought wine|dinner party progress/i.test(text)
      ? [
          {
            goal_id: goalId,
            evidence: "User bought wine for the dinner party.",
          },
        ]
      : [];

  return completeWithTool({
    id: "toolu_reflection",
    name: "EmitTurnReflection",
    input: {
      advanced_goals: advancedGoals,
      procedural_outcomes: [],
      trait_demonstrations: [],
      intent_updates: [],
      step_outcomes: [],
      proposed_steps: [],
    },
  });
}

function emptyPlan(): LLMCompleteResult {
  return completeWithTool({
    id: "toolu_plan",
    name: "EmitTurnPlan",
    input: {
      uncertainty: "",
      verification_steps: [],
      tensions: [],
      voice_note: "",
      referenced_episode_ids: [],
      intents: [],
    },
  });
}

function noCommitmentViolations(): LLMCompleteResult {
  return completeWithTool({
    id: "toolu_commitment",
    name: "EmitCommitmentViolations",
    input: {
      violations: [],
    },
  });
}

function textResult(text: string): LLMCompleteResult {
  return {
    text,
    input_tokens: 24,
    output_tokens: Math.max(1, Math.ceil(text.length / 4)),
    stop_reason: "end_turn",
    tool_calls: [],
  };
}

function textConversation(text: string): LLMConverseResult {
  return {
    messageBlocks: [
      {
        type: "text",
        text,
      },
    ],
    input_tokens: 24,
    output_tokens: Math.max(1, Math.ceil(text.length / 4)),
    stop_reason: "end_turn",
  };
}

function toolConversation(call: LLMToolCall): LLMConverseResult {
  return {
    messageBlocks: [
      {
        type: "tool_use",
        id: call.id,
        name: call.name,
        input: call.input,
      },
    ],
    input_tokens: 24,
    output_tokens: 8,
    stop_reason: "tool_use",
  };
}

function mockBorgAnswer(text: string): string {
  if (/what'?s my dog'?s name|dog'?s name/i.test(text)) {
    return "Your dog's name is Otto.";
  }

  if (/project codename|Helios/i.test(text)) {
    return "Your project codename is Helios.";
  }

  if (/capital of belize|san ignacio|belmopan/i.test(text)) {
    return "That conflicts with the earlier claim; Belize's capital is recorded as Belmopan.";
  }

  if (/swallow|average flight speed/i.test(text)) {
    return "I don't know that from current memory, so I logged it as an open question.";
  }

  if (/frustrated|work/i.test(text)) {
    return "That sounds frustrating; I will keep the recent mood context in mind.";
  }

  if (/dinner party|bought wine|progress/i.test(text)) {
    return "I noted the dinner-party progress.";
  }

  if (/direct communication|overwrite|value/i.test(text)) {
    return "I should not overwrite an established value without review.";
  }

  if (/relate|semantic|graph|what do you know/i.test(text)) {
    return "I checked the semantic graph and found no reliable relationship in the mock run.";
  }

  if (/always end every response/i.test(text)) {
    return "I will treat that as a standing response commitment. understood.";
  }

  return "Mock Borg response. understood.";
}

function createMockBorgLlmClient(): FakeLLMClient {
  return new FakeLLMClient({
    responses: Array.from({ length: MOCK_RESPONSE_COUNT }, () => {
      return (options: LLMCompleteOptions | LLMConverseOptions) => {
        const names = toolNames(options);
        const text = requestText(options);
        const latestText = latestMessageText(options);

        if (names.includes("EmitTurnReflection")) {
          return reflectionForRequest(text);
        }

        if (names.includes("EmitTurnPlan")) {
          return emptyPlan();
        }

        if (names.includes("EmitCommitmentViolations")) {
          return noCommitmentViolations();
        }

        if (typeof options.messages[0]?.content === "string") {
          return textResult(mockBorgAnswer(latestText));
        }

        if (hasToolResult(options)) {
          return textConversation(mockBorgAnswer(text));
        }

        if (
          names.includes("tool.episodic.search") &&
          (/what'?s my dog'?s name|dog'?s name/i.test(latestText) ||
            /project codename/i.test(latestText))
        ) {
          return toolConversation({
            id: "toolu_episodic_search",
            name: "tool.episodic.search",
            input: {
              query: /project codename/i.test(latestText)
                ? "project codename Helios"
                : "dog name Otto",
              limit: 3,
            },
          });
        }

        if (names.includes("tool.semantic.walk") && /relate|semantic|graph/i.test(latestText)) {
          return toolConversation({
            id: "toolu_semantic_walk",
            name: "tool.semantic.walk",
            input: {
              node_id: createSemanticNodeId(),
              relation: "related_to",
              depth: 2,
              maxNodes: 4,
            },
          });
        }

        if (
          names.includes("tool.openQuestions.create") &&
          /swallow|average flight speed/i.test(latestText)
        ) {
          return toolConversation({
            id: "toolu_open_question",
            name: "tool.openQuestions.create",
            input: {
              question: "What is the average flight speed of a swallow?",
              urgency: 0.4,
            },
          });
        }

        return textConversation(mockBorgAnswer(latestText));
      };
    }),
  });
}

export class BorgTransport {
  readonly dataDir: string;
  readonly tracePath: string;
  private readonly keep: boolean;
  private readonly scenario: Scenario;
  private readonly env: NodeJS.ProcessEnv;
  private readonly mock: boolean;
  private readonly llmClient?: BorgOpenOptions["llmClient"];
  private readonly embeddingClient?: BorgOpenOptions["embeddingClient"];
  private readonly clock: ManualClock;
  private readonly seededGoals = new Map<string, GoalRecord>();
  private borg?: Borg;

  constructor(options: BorgTransportOptions) {
    const scenarioPart = sanitizePathPart(options.scenario.name);
    const basePath = join(tmpdir(), `borg-assessor-${options.runId}-${scenarioPart}`);

    this.dataDir = options.dataDir ?? basePath;
    this.tracePath = options.tracePath ?? `${basePath}.trace.jsonl`;
    this.keep = options.keep ?? false;
    this.scenario = options.scenario;
    this.env = options.env ?? process.env;
    this.mock = options.mock ?? false;
    this.llmClient = options.llmClient;
    this.embeddingClient = options.embeddingClient;
    this.clock = options.clock ?? new ManualClock(Date.now());
  }

  async open(): Promise<void> {
    if (this.borg !== undefined) {
      return;
    }

    const env = {
      ...this.env,
      BORG_DATA_DIR: this.dataDir,
      BORG_TRACE: this.tracePath,
      ...(this.scenario.tracePrompts === true ? { BORG_TRACE_PROMPTS: "1" } : {}),
    };

    if (this.mock) {
      this.borg = await createEvalBorg({
        tempDir: this.dataDir,
        llm: this.llmClient ?? createMockBorgLlmClient(),
        clock: this.clock,
        tracerPath: this.tracePath,
        env,
        config: {
          ...(this.scenario.borgConfigOverrides as Record<string, unknown>),
          dataDir: this.dataDir,
          maintenance: {
            enabled: false,
          },
          autonomy: deepMerge(
            {
              ...DEFAULT_CONFIG.autonomy,
              enabled: false,
            },
            this.scenario.borgConfigOverrides?.autonomy,
          ),
        },
      });
      this.seedScenarioGoals();
      await sleep(OPEN_HOOK_SETTLE_MS);
      return;
    }

    this.borg = await Borg.open({
      config: createRealConfig({
        dataDir: this.dataDir,
        env,
        scenario: this.scenario,
      }),
      env,
      tracerPath: this.tracePath,
      clock: this.clock,
      ...(this.llmClient === undefined ? {} : { llmClient: this.llmClient }),
      ...(this.embeddingClient === undefined ? {} : { embeddingClient: this.embeddingClient }),
    });
    this.seedScenarioGoals();
    await sleep(OPEN_HOOK_SETTLE_MS);
  }

  private seedScenarioGoals(): void {
    if (this.borg === undefined) {
      return;
    }

    for (const seed of this.scenario.seedGoals ?? []) {
      if (this.seededGoals.has(seed.key)) {
        continue;
      }

      this.seededGoals.set(
        seed.key,
        this.borg.self.goals.add({
          description: seed.description,
          priority: seed.priority ?? 10,
          provenance: {
            kind: "manual",
          },
        }),
      );
    }
  }

  async chat(message: string, options: { sessionId?: string } = {}): Promise<ChatWithBorgResult> {
    if (this.borg === undefined) {
      throw new Error("BorgTransport.open() must be called before chat()");
    }

    this.clock.advance(1);
    const beforeCount = readTraceEvents(this.tracePath).length;
    const sessionId =
      options.sessionId === undefined ? undefined : parseSessionId(options.sessionId);
    const result = await this.borg.turn({
      userMessage: message,
      stakes: DEFAULT_BORG_STAKES,
      ...(sessionId === undefined ? {} : { sessionId }),
    });
    const events = readTraceEvents(this.tracePath);
    const turnId = latestTurnId(events.slice(beforeCount)) ?? latestTurnId(events);

    if (turnId === null) {
      throw new Error("Borg turn completed without trace events");
    }

    const mood = this.borg.mood.current(sessionId ?? DEFAULT_SESSION_ID);

    return {
      response: result.response,
      turnId,
      ...(options.sessionId === undefined ? {} : { sessionId: options.sessionId }),
      usage: {
        input_tokens: result.usage.input_tokens,
        output_tokens: result.usage.output_tokens,
      },
      moodAfter: {
        valence: mood.valence,
        arousal: mood.arousal,
      },
      toolCalls: result.toolCalls.map((call) => ({
        name: call.name,
        ok: call.ok,
      })),
    };
  }

  readTrace(turnId: string, phase?: TracePhase): string {
    return summarizeTraceFile(this.tracePath, turnId, { phase });
  }

  readTraceEvents(): TraceRecord[] {
    return readTraceEvents(this.tracePath);
  }

  streamTail(limit: number) {
    if (this.borg === undefined) {
      throw new Error("BorgTransport.open() must be called before streamTail()");
    }

    return this.borg.stream.tail(limit);
  }

  getSeededGoal(key: string): GoalRecord | null {
    if (this.borg === undefined) {
      throw new Error("BorgTransport.open() must be called before getSeededGoal()");
    }

    const seeded = this.seededGoals.get(key);

    if (seeded === undefined) {
      return null;
    }

    return this.borg.self.goals.get(seeded.id as GoalId);
  }

  getSeededGoalProgressEvidence(key: string): SeededGoalProgressEvidence {
    if (this.borg === undefined) {
      throw new Error("BorgTransport.open() must be called before getSeededGoalProgressEvidence()");
    }

    const goal = this.getSeededGoal(key);

    if (goal === null) {
      return {
        goal,
        reviewPatch: null,
      };
    }

    const reviewPatch =
      this.borg.review
        .list({ kind: "identity_inconsistency", openOnly: true })
        .map((item) => goalProgressPatchFromReview(item, goal.id))
        .find((patch) => patch !== null) ?? null;

    return {
      goal,
      reviewPatch,
    };
  }

  async runAutonomyExecutiveWakeAssertion(advanceMs: number): Promise<string> {
    if (this.borg === undefined) {
      throw new Error("BorgTransport.open() must be called before autonomy assertions");
    }

    this.clock.advance(advanceMs);
    const tick = await this.borg.autonomy.scheduler.tick();
    const selfAgentMessage = this.streamTail(12).find(
      (entry) => entry.kind === "agent_msg" && entry.audience === "self",
    );
    const sourceNames = tick.events.map((event) => event.sourceName).join(",");

    return `autonomy tick firedEvents=${tick.firedEvents}; sources=${sourceNames}; self agent message=${selfAgentMessage === undefined ? "missing" : "present"}`;
  }

  async close(): Promise<void> {
    const borg = this.borg;
    this.borg = undefined;

    if (borg !== undefined) {
      await sleep(OPEN_HOOK_SETTLE_MS);
      await borg.close();
    }

    if (!this.keep) {
      rmSync(this.dataDir, { recursive: true, force: true });
      rmSync(this.tracePath, { force: true });
    }
  }
}

export function defaultSessionForTurn(
  scenario: Scenario,
  turnNumber: number,
): SessionId | undefined {
  const session = scenario.sessionForTurn?.(turnNumber);

  return session === undefined ? undefined : parseSessionId(session);
}
