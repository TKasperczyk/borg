import { BorgTransport } from "../assessor/borg-transport.js";
import { evaluateAssertions } from "../assessor/runner.js";
import { getScenario } from "../assessor/scenarios/index.js";
import type { ConversationTurn } from "../assessor/types.js";

import type { ProbeResult, ProbeSchedule } from "./types.js";

const DEFAULT_PROBE_SCENARIOS = [
  "recall",
  "open-question-creation",
  "identity-guard-refusal",
  "contradiction-handling",
  "mood-persistence",
] as const;

export function scheduleProbes(totalTurns: number, probeEvery: number): ProbeSchedule {
  if (!Number.isInteger(probeEvery) || probeEvery <= 0) {
    return {};
  }

  const schedule: ProbeSchedule = {};
  let index = 0;

  for (let turn = probeEvery; turn < totalTurns; turn += probeEvery) {
    const scenarioName = DEFAULT_PROBE_SCENARIOS[index % DEFAULT_PROBE_SCENARIOS.length];

    if (scenarioName === undefined) {
      throw new Error("Default probe scenario rotation is empty");
    }

    schedule[turn] = scenarioName;
    index += 1;
  }

  return schedule;
}

export async function runProbe(args: {
  scenarioName: string;
  transport: BorgTransport;
  turnNumber: number;
}): Promise<ProbeResult> {
  const scenario = getScenario(args.scenarioName);

  if (scenario === undefined) {
    throw new Error(`Unknown probe scenario: ${args.scenarioName}`);
  }

  const message = scenario.mockConversation?.[0] ?? `Run probe scenario ${scenario.name}.`;
  const result = await args.transport.chat(message);
  const turn: ConversationTurn = {
    message,
    response: result.response,
    turnId: result.turnId,
    usage: result.usage,
    moodAfter: result.moodAfter,
  };
  const assertions = await evaluateAssertions({
    scenario,
    transport: args.transport,
    turns: [turn],
  });
  const failed = assertions.filter((assertion) => !assertion.passed);
  const passed = failed.length === 0;
  const evidence =
    assertions.length === 0
      ? `Probe turn ${result.turnId}: ${result.response.slice(0, 180)}`
      : assertions
          .map((assertion) => {
            const status = assertion.passed ? "pass" : "fail";
            return `${status}: ${assertion.description}: ${assertion.evidence.slice(0, 180)}`;
          })
          .join("; ");

  return {
    turn: args.turnNumber,
    scenarioName: scenario.name,
    passed,
    evidence,
    response: result.response,
    turnId: result.turnId,
  };
}
