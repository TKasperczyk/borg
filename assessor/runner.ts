import { performance } from "node:perf_hooks";

import { AssessorAgent } from "./assessor-agent.js";
import { BorgTransport } from "./borg-transport.js";
import { phaseForTraceEvent, readTraceEvents } from "./trace-reader.js";
import type {
  AssessorRunReport,
  AssessorStatus,
  AssessorUsage,
  AssessorVerdict,
  ConversationTurn,
  Scenario,
  ScenarioResult,
  TraceAssertion,
  TraceAssertionResult,
  TraceRecord,
} from "./types.js";

function coveredPhases(records: readonly TraceRecord[]) {
  return [...new Set(records.map((record) => phaseForTraceEvent(record.event)))];
}

export type ScenarioRunnerOptions = {
  runId: string;
  scenario: Scenario;
  keep?: boolean;
  mock?: boolean;
  maxTurns?: number;
  maxLlmCalls?: number;
  env?: NodeJS.ProcessEnv;
};

export type RunScenariosOptions = Omit<ScenarioRunnerOptions, "scenario"> & {
  scenarios: readonly Scenario[];
};

function regexFromAssertion(input: { pattern: string; flags?: string }): RegExp {
  return new RegExp(input.pattern, input.flags ?? "i");
}

function lastTurnId(turns: readonly ConversationTurn[]): string | null {
  return turns[turns.length - 1]?.turnId ?? null;
}

function recordsForTurn(
  records: readonly TraceRecord[],
  turns: readonly ConversationTurn[],
  turn: "any" | "last" | undefined,
): TraceRecord[] {
  if (turn !== "last") {
    return [...records];
  }

  const target = lastTurnId(turns);

  if (target === null) {
    return [];
  }

  return records.filter((record) => record.turnId === target);
}

function stringifyEvidence(value: unknown): string {
  const raw = typeof value === "string" ? value : JSON.stringify(value);

  if (raw === undefined) {
    return "undefined";
  }

  return raw.length > 240 ? `${raw.slice(0, 240)}...` : raw;
}

async function evaluateAssertion(
  assertion: TraceAssertion,
  context: {
    transport: BorgTransport;
    turns: readonly ConversationTurn[];
    records: readonly TraceRecord[];
  },
): Promise<TraceAssertionResult> {
  if (assertion.type === "tool_called") {
    const records = recordsForTurn(context.records, context.turns, assertion.turn);
    const match = records.find((record) => {
      const toolName = typeof record.toolName === "string" ? record.toolName : "";

      return (
        record.event.startsWith("tool_call_") &&
        toolName.toLowerCase().includes(assertion.toolNameIncludes.toLowerCase())
      );
    });

    return {
      description: assertion.description,
      passed: match !== undefined,
      evidence:
        match === undefined
          ? `No matching tool call for ${assertion.toolNameIncludes}`
          : `${match.event} ${match.toolName}`,
    };
  }

  if (assertion.type === "event_seen") {
    const records = recordsForTurn(context.records, context.turns, assertion.turn);
    const match = records.find((record) =>
      record.event.toLowerCase().includes(assertion.eventIncludes.toLowerCase()),
    );

    return {
      description: assertion.description,
      passed: match !== undefined,
      evidence:
        match === undefined ? `No matching event for ${assertion.eventIncludes}` : match.event,
    };
  }

  if (assertion.type === "all_responses_match") {
    const regex = regexFromAssertion(assertion);
    const failures = context.turns.filter((turn) => !regex.test(turn.response));

    return {
      description: assertion.description,
      passed: context.turns.length > 0 && failures.length === 0,
      evidence:
        failures[0] === undefined
          ? `${context.turns.length} response(s) matched`
          : `First failure: ${stringifyEvidence(failures[0].response)}`,
    };
  }

  if (assertion.type === "stream_entry") {
    const match = context.transport.streamTail(200).find((entry) => {
      if (assertion.kind !== undefined && entry.kind !== assertion.kind) {
        return false;
      }

      if (assertion.audience !== undefined && entry.audience !== assertion.audience) {
        return false;
      }

      if (assertion.contentIncludes !== undefined) {
        return stringifyEvidence(entry.content)
          .toLowerCase()
          .includes(assertion.contentIncludes.toLowerCase());
      }

      return true;
    });

    return {
      description: assertion.description,
      passed: match !== undefined,
      evidence:
        match === undefined
          ? "No matching stream entry"
          : `${match.kind} ${stringifyEvidence(match.content)}`,
    };
  }

  if (assertion.type === "goal_progress") {
    const { goal, reviewPatch } = context.transport.getSeededGoalProgressEvidence(
      assertion.goalKey,
    );
    const progress = goal?.progress_notes ?? reviewPatch?.progressNotes ?? "";
    const lastProgressTs = goal?.last_progress_ts ?? reviewPatch?.lastProgressTs ?? null;
    const passed =
      goal !== null &&
      lastProgressTs !== null &&
      lastProgressTs > goal.created_at &&
      (assertion.progressIncludes === undefined ||
        progress.toLowerCase().includes(assertion.progressIncludes.toLowerCase()));

    return {
      description: assertion.description,
      passed,
      evidence:
        goal === null
          ? `No seeded goal found for key ${assertion.goalKey}`
          : [
              `goal=${goal.id}`,
              `created_at=${goal.created_at}`,
              `last_progress_ts=${lastProgressTs ?? "null"}`,
              reviewPatch === null ? "source=goal" : `source=review_item:${reviewPatch.itemId}`,
              `progress=${stringifyEvidence(progress)}`,
            ].join(" "),
    };
  }

  if (assertion.type === "mood_decay") {
    const negativeTurn = context.turns[assertion.negativeTurn - 1];
    const laterTurn = context.turns[assertion.laterTurn - 1];
    const firstValence = negativeTurn?.moodAfter?.valence;
    const laterValence = laterTurn?.moodAfter?.valence;
    const passed =
      firstValence !== undefined &&
      laterValence !== undefined &&
      firstValence < 0 &&
      laterValence < 0 &&
      laterValence > firstValence;

    return {
      description: assertion.description,
      passed,
      evidence: `turn${assertion.negativeTurn} valence=${firstValence ?? "missing"}; turn${assertion.laterTurn} valence=${laterValence ?? "missing"}`,
    };
  }

  if (assertion.type === "any_of") {
    const results = await Promise.all(
      assertion.assertions.map((entry) => evaluateAssertion(entry, context)),
    );
    const match = results.find((result) => result.passed);

    return {
      description: assertion.description,
      passed: match !== undefined,
      evidence:
        match === undefined ? results.map((result) => result.evidence).join("; ") : match.evidence,
    };
  }

  const evidence = await context.transport.runAutonomyExecutiveWakeAssertion(assertion.advanceMs);

  return {
    description: assertion.description,
    passed:
      /firedEvents=[1-9]/.test(evidence) &&
      /sources=.*executive_focus_due/.test(evidence) &&
      /self agent message=present/.test(evidence),
    evidence,
  };
}

async function evaluateAssertions(input: {
  scenario: Scenario;
  transport: BorgTransport;
  turns: readonly ConversationTurn[];
}): Promise<TraceAssertionResult[]> {
  const assertions = input.scenario.traceAssertions ?? [];
  const initialRecords = readTraceEvents(input.transport.tracePath);
  const results: TraceAssertionResult[] = [];

  for (const assertion of assertions) {
    const result = await evaluateAssertion(assertion, {
      transport: input.transport,
      turns: input.turns,
      records: initialRecords,
    });
    results.push(result);
  }

  return results;
}

async function runMockAssessor(input: {
  scenario: Scenario;
  transport: BorgTransport;
  maxTurns: number;
}): Promise<{
  turns: ConversationTurn[];
  usage: AssessorUsage;
}> {
  const messages = input.scenario.mockConversation ?? [
    `Run a smoke conversation for scenario ${input.scenario.name}.`,
  ];
  const turns: ConversationTurn[] = [];

  for (const message of messages.slice(0, input.maxTurns)) {
    const turnNumber = turns.length + 1;
    const sessionId = input.scenario.sessionForTurn?.(turnNumber);
    const result = await input.transport.chat(message, { sessionId });
    const traceSummary = input.transport.readTrace(result.turnId);
    turns.push({
      message,
      response: result.response,
      turnId: result.turnId,
      sessionId: result.sessionId,
      traceSummary,
      usage: result.usage,
      moodAfter: result.moodAfter,
    });
  }

  return {
    turns,
    usage: {
      llmCalls: 0,
      inputTokens: 0,
      outputTokens: 0,
    },
  };
}

function buildMockVerdict(
  scenario: Scenario,
  turns: readonly ConversationTurn[],
  assertions: readonly TraceAssertionResult[],
): AssessorVerdict {
  const last = turns[turns.length - 1];
  const failedAssertions = assertions.filter((assertion) => !assertion.passed);
  const scriptedTurns = scenario.mockConversation?.length ?? turns.length;
  const stoppedBeforeRelevantTurn = turns.length === 0 || turns.length < scriptedTurns;
  const status: AssessorStatus =
    failedAssertions.length === 0 ? "pass" : stoppedBeforeRelevantTurn ? "inconclusive" : "fail";

  return {
    status,
    reasoning:
      failedAssertions.length === 0
        ? `Mock assessor completed the scripted ${scenario.name} path and all trace assertions passed.`
        : `Mock assessor completed the scripted ${scenario.name} path but ${failedAssertions.length} trace assertion(s) failed.`,
    evidence:
      failedAssertions.length > 0
        ? failedAssertions.map(
            (assertion) => `${assertion.description}: ${assertion.evidence.slice(0, 180)}`,
          )
        : last === undefined
          ? ["No Borg turns were scripted."]
          : [`Last turn ${last.turnId}: ${last.response.slice(0, 180)}`],
  };
}

// Combines the assessor's submitted verdict with independent trace
// assertion results. Assertions are dispositive when they cover what
// the assessor cannot directly observe (autonomous turns, mood-state
// internals, cross-session retrieval); a passing assertion set rescues
// an inconclusive verdict, and a failing assertion overrides a
// surface-level pass.
export function reconcileVerdictWithAssertions(
  verdict: AssessorVerdict,
  assertions: readonly TraceAssertionResult[],
): AssessorVerdict {
  if (assertions.length === 0) {
    return verdict;
  }

  const allPassed = assertions.every((assertion) => assertion.passed);
  const failedDescriptions = assertions
    .filter((assertion) => !assertion.passed)
    .map((assertion) => assertion.description);

  if (verdict.status === "pass" && !allPassed) {
    return {
      status: "fail",
      reasoning: `${verdict.reasoning}\n\nDowngraded to fail: trace assertion(s) failed even though the assessor verdict was pass: ${failedDescriptions.join("; ")}.`,
      evidence: verdict.evidence,
    };
  }

  if (verdict.status === "inconclusive" && allPassed) {
    return {
      status: "pass",
      reasoning: `${verdict.reasoning}\n\nUpgraded to pass: all ${assertions.length} trace assertion(s) passed independently of the assessor verdict.`,
      evidence: verdict.evidence,
    };
  }

  if (verdict.status === "inconclusive" && !allPassed) {
    return {
      status: "fail",
      reasoning: `${verdict.reasoning}\n\nDowngraded to fail: trace assertion(s) failed: ${failedDescriptions.join("; ")}.`,
      evidence: verdict.evidence,
    };
  }

  return verdict;
}

function costFor(turns: readonly ConversationTurn[], usage: AssessorUsage) {
  const borgTokens = turns.reduce(
    (sum, turn) => sum + (turn.usage?.input_tokens ?? 0) + (turn.usage?.output_tokens ?? 0),
    0,
  );

  return {
    borgTurns: turns.length,
    assessorLlmCalls: usage.llmCalls,
    approximateTokens: borgTokens + usage.inputTokens + usage.outputTokens,
  };
}

export class ScenarioRunner {
  private readonly options: ScenarioRunnerOptions;

  constructor(options: ScenarioRunnerOptions) {
    this.options = options;
  }

  async run(): Promise<ScenarioResult> {
    const started = performance.now();
    const transport = new BorgTransport({
      runId: this.options.runId,
      scenario: this.options.scenario,
      keep: this.options.keep,
      mock: this.options.mock,
      env: this.options.env,
    });
    let turns: ConversationTurn[] = [];
    let usage: AssessorUsage = {
      llmCalls: 0,
      inputTokens: 0,
      outputTokens: 0,
    };

    try {
      await transport.open();

      if (this.options.mock === true) {
        const mock = await runMockAssessor({
          scenario: this.options.scenario,
          transport,
          maxTurns: this.options.maxTurns ?? this.options.scenario.maxTurns,
        });
        turns = mock.turns;
        usage = mock.usage;
        const assertions = await evaluateAssertions({
          scenario: this.options.scenario,
          transport,
          turns,
        });

        return {
          scenario: this.options.scenario,
          verdict: buildMockVerdict(this.options.scenario, turns, assertions),
          turns,
          traceAssertions: assertions,
          coveredPhases: coveredPhases(transport.readTraceEvents()),
          tracePath: transport.tracePath,
          dataDir: transport.dataDir,
          cost: costFor(turns, usage),
          durationMs: performance.now() - started,
        };
      }

      let turnCounter = 0;
      const assessor = new AssessorAgent({
        scenario: this.options.scenario,
        maxTurns: this.options.maxTurns ?? this.options.scenario.maxTurns,
        maxLlmCalls: this.options.maxLlmCalls ?? this.options.scenario.maxLlmCalls,
        env: this.options.env,
        tools: {
          chatWithBorg: async (message) => {
            turnCounter += 1;
            const sessionId = this.options.scenario.sessionForTurn?.(turnCounter);

            return transport.chat(message, { sessionId });
          },
          readTrace: (turnId, phase) => transport.readTrace(turnId, phase),
        },
      });
      const assessed = await assessor.run();
      turns = assessed.turns;
      usage = assessed.usage;
      const assertions = await evaluateAssertions({
        scenario: this.options.scenario,
        transport,
        turns,
      });

      return {
        scenario: this.options.scenario,
        verdict: reconcileVerdictWithAssertions(assessed.verdict, assertions),
        turns,
        traceAssertions: assertions,
        coveredPhases: coveredPhases(transport.readTraceEvents()),
        tracePath: transport.tracePath,
        dataDir: transport.dataDir,
        cost: costFor(turns, usage),
        durationMs: performance.now() - started,
      };
    } catch (error) {
      return {
        scenario: this.options.scenario,
        verdict: {
          status: "inconclusive",
          reasoning: "Scenario runner failed before a reliable verdict was produced.",
          evidence: [error instanceof Error ? error.message : String(error)],
        },
        turns,
        traceAssertions: [],
        coveredPhases: [],
        tracePath: transport.tracePath,
        dataDir: transport.dataDir,
        cost: costFor(turns, usage),
        durationMs: performance.now() - started,
        error: error instanceof Error ? (error.stack ?? error.message) : String(error),
      };
    } finally {
      await transport.close();
    }
  }
}

export async function runScenarios(options: RunScenariosOptions): Promise<AssessorRunReport> {
  const startedAtMs = Date.now();
  const startedAt = new Date(startedAtMs).toISOString();
  const results: ScenarioResult[] = [];

  for (const scenario of options.scenarios) {
    const runner = new ScenarioRunner({
      ...options,
      scenario,
    });
    results.push(await runner.run());
  }

  return {
    runId: options.runId,
    startedAt,
    durationMs: Date.now() - startedAtMs,
    results,
  };
}
