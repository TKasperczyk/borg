import type { Config } from "../src/index.js";
import type { StreamEntry } from "../src/stream/index.js";

export type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends Array<infer U>
    ? Array<U>
    : T[K] extends ReadonlyArray<infer U>
      ? ReadonlyArray<U>
      : T[K] extends object
        ? DeepPartial<T[K]>
        : T[K];
};

export type AssessorStatus = "pass" | "fail" | "inconclusive";

export type AssessorVerdict = {
  status: AssessorStatus;
  reasoning: string;
  evidence: string[];
};

export type TracePhase =
  | "perception"
  | "executive_focus"
  | "retrieval"
  | "deliberation"
  | "action"
  | "reflection"
  | "ingestion"
  | "other";

export type TraceRecord = {
  ts: number;
  turnId: string;
  event: string;
  [key: string]: unknown;
};

export type TraceAssertion =
  | {
      type: "tool_called";
      description: string;
      toolNameIncludes: string;
      turn?: "any" | "last";
    }
  | {
      type: "event_seen";
      description: string;
      eventIncludes: string;
      turn?: "any" | "last";
    }
  | {
      type: "response_matches";
      description: string;
      pattern: string;
      flags?: string;
      turn?: "any" | "last";
    }
  | {
      type: "all_responses_match";
      description: string;
      pattern: string;
      flags?: string;
    }
  | {
      type: "stream_entry";
      description: string;
      kind?: StreamEntry["kind"];
      audience?: string;
      contentIncludes?: string;
    }
  | {
      type: "goal_progress";
      description: string;
      goalKey: string;
      progressIncludes?: string;
    }
  | {
      type: "mood_decay";
      description: string;
      negativeTurn: number;
      laterTurn: number;
    }
  | {
      type: "any_of";
      description: string;
      assertions: TraceAssertion[];
    }
  | {
      type: "autonomy_executive_wake";
      description: string;
      advanceMs: number;
    };

export type TraceAssertionResult = {
  description: string;
  passed: boolean;
  evidence: string;
};

export type Scenario = {
  name: string;
  description: string;
  borgConfigOverrides?: DeepPartial<Config>;
  seedGoals?: Array<{
    key: string;
    description: string;
    priority?: number;
  }>;
  systemPrompt: string;
  traceAssertions?: TraceAssertion[];
  maxTurns: number;
  maxLlmCalls?: number;
  tracePrompts?: boolean;
  mockConversation?: string[];
  sessionForTurn?: (turnNumber: number) => string | undefined;
};

export type ConversationTurn = {
  message: string;
  response: string;
  turnId: string;
  sessionId?: string;
  traceSummary?: string;
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
  moodAfter?: {
    valence: number;
    arousal: number;
  };
};

export type AssessorUsage = {
  llmCalls: number;
  inputTokens: number;
  outputTokens: number;
};

export type ScenarioCost = {
  borgTurns: number;
  assessorLlmCalls: number;
  approximateTokens: number;
};

export type ScenarioResult = {
  scenario: Scenario;
  verdict: AssessorVerdict;
  turns: ConversationTurn[];
  traceAssertions: TraceAssertionResult[];
  coveredPhases: TracePhase[];
  tracePath: string;
  dataDir: string;
  cost: ScenarioCost;
  durationMs: number;
  error?: string;
};

export type AssessorRunReport = {
  runId: string;
  startedAt: string;
  durationMs: number;
  results: ScenarioResult[];
};
