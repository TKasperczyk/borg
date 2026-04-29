import { performance } from "node:perf_hooks";

import { BorgTransport } from "../assessor/borg-transport.js";
import type { Scenario } from "../assessor/types.js";

import { MetricsCapture } from "./metrics.js";
import { PersonaSession } from "./persona.js";
import { runOverseer, type RunOverseerOptions } from "./overseer.js";
import { runProbe, scheduleProbes } from "./probes.js";
import type {
  MetricsRow,
  OverseerVerdict,
  Persona,
  ProbeResult,
  ProbeSchedule,
  SimulatorRunReport,
} from "./types.js";

export type SimulatorRunnerOptions = {
  runId: string;
  persona: Persona;
  totalTurns: number;
  metricsPath: string;
  probeEvery: number;
  checkEvery: number;
  keep?: boolean;
  mock?: boolean;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
  tracePath?: string;
  probeSchedule?: ProbeSchedule;
  personaSession?: PersonaSession;
  probeRunner?: typeof runProbe;
  overseerRunner?: (options: RunOverseerOptions) => Promise<OverseerVerdict>;
};

function simulatorScenario(persona: Persona, totalTurns: number): Scenario {
  return {
    name: `simulator-${persona.key}`,
    description: `Long-horizon simulator run for ${persona.displayName}.`,
    systemPrompt: persona.systemPrompt,
    maxTurns: totalTurns,
  };
}

function stripProbe(result: ProbeResult): SimulatorRunReport["probes"][number] {
  return {
    turn: result.turn,
    scenarioName: result.scenarioName,
    passed: result.passed,
    evidence: result.evidence,
  };
}

// Walks the Error.cause chain to surface diagnostics that LLMError and
// other wrappers normally hide. Without this, every transient
// failure shows up in the simulator log as 'Failed to complete
// Anthropic request' with no signal about what actually failed.
function formatErrorChain(error: unknown): string {
  const parts: string[] = [];
  let current: unknown = error;
  let depth = 0;

  while (current !== null && current !== undefined && depth < 5) {
    if (current instanceof Error) {
      const name = current.name === "Error" ? "" : `${current.name}: `;
      parts.push(`${name}${current.message}`);
      current = (current as Error & { cause?: unknown }).cause;
    } else {
      parts.push(String(current));
      break;
    }
    depth += 1;
  }

  return parts.length === 0 ? String(error) : parts.join(" -> ");
}

export class SimulatorRunner {
  private readonly options: SimulatorRunnerOptions;
  private turnFailures: Array<{ turn: number; error: string }> = [];

  constructor(options: SimulatorRunnerOptions) {
    this.options = options;
  }

  async run(): Promise<SimulatorRunReport> {
    if (!Number.isInteger(this.options.totalTurns) || this.options.totalTurns <= 0) {
      throw new Error("totalTurns must be a positive integer");
    }

    const started = performance.now();
    const transport = new BorgTransport({
      runId: this.options.runId,
      scenario: simulatorScenario(this.options.persona, this.options.totalTurns),
      keep: this.options.keep,
      mock: this.options.mock,
      env: this.options.env,
      dataDir: this.options.dataDir,
      tracePath: this.options.tracePath,
    });
    const metrics = new MetricsCapture(this.options.metricsPath, {
      tracePath: transport.tracePath,
    });
    const persona =
      this.options.personaSession ??
      new PersonaSession({
        persona: this.options.persona,
        mock: this.options.mock,
        env: this.options.env,
      });
    const probes =
      this.options.probeSchedule ??
      scheduleProbes(this.options.totalTurns, this.options.probeEvery);
    const probeRunner = this.options.probeRunner ?? runProbe;
    const overseerRunner = this.options.overseerRunner ?? runOverseer;
    const probeResults: SimulatorRunReport["probes"] = [];
    const overseerCheckpoints: SimulatorRunReport["overseerCheckpoints"] = [];
    let lastBorgResponse: string | null = null;
    let finalMetrics: MetricsRow | undefined;

    try {
      await transport.open();

      // Long-horizon runs amortize cost across hours, so a single failing
      // turn (LLM rate-limit, transient API error, schema validation crash
      // in some Borg phase) shouldn't abort the whole run -- it should
      // be logged and the loop continues. We do bail if too many
      // consecutive turns fail, since that indicates the harness itself
      // is broken rather than an isolated turn-level fault.
      const MAX_CONSECUTIVE_FAILURES = 5;
      const TRANSIENT_RETRY_ATTEMPTS = 2;
      const TRANSIENT_RETRY_DELAY_MS = 2_000;
      let consecutiveFailures = 0;
      const turnFailures: Array<{ turn: number; error: string }> = [];

      const attemptTurn = async (
        turn: number,
        scheduledProbe: string | undefined,
      ): Promise<{ turnId: string; response: string }> => {
        if (scheduledProbe !== undefined) {
          const probe = await probeRunner({
            scenarioName: scheduledProbe,
            transport,
            turnNumber: turn,
          });
          probeResults.push(stripProbe(probe));
          return { turnId: probe.turnId, response: probe.response };
        }
        const message = await persona.nextTurn(lastBorgResponse);
        const result = await transport.chat(message);
        return { turnId: result.turnId, response: result.response };
      };

      for (let turn = 1; turn <= this.options.totalTurns; turn += 1) {
        const scheduledProbe = probes[turn];
        let success: { turnId: string; response: string } | null = null;
        let attemptError: unknown = null;

        for (let attempt = 0; attempt <= TRANSIENT_RETRY_ATTEMPTS; attempt += 1) {
          try {
            success = await attemptTurn(turn, scheduledProbe);
            attemptError = null;
            break;
          } catch (error) {
            attemptError = error;
            if (attempt < TRANSIENT_RETRY_ATTEMPTS) {
              await new Promise((resolve) =>
                setTimeout(resolve, TRANSIENT_RETRY_DELAY_MS * (attempt + 1)),
              );
            }
          }
        }

        if (success === null) {
          const detail = formatErrorChain(attemptError);
          turnFailures.push({ turn, error: detail });
          consecutiveFailures += 1;
          // eslint-disable-next-line no-console
          console.warn(`[simulator] turn ${turn} failed after retries: ${detail}`);

          if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            throw new Error(
              `Simulator aborting: ${consecutiveFailures} consecutive turn failures (last: ${detail})`,
            );
          }
          continue;
        }

        lastBorgResponse = success.response;
        consecutiveFailures = 0;

        finalMetrics = await metrics.capture(transport.getBorg(), success.turnId, turn);

        if (
          Number.isInteger(this.options.checkEvery) &&
          this.options.checkEvery > 0 &&
          turn % this.options.checkEvery === 0
        ) {
          overseerCheckpoints.push(
            await overseerRunner({
              transport,
              metricsPath: this.options.metricsPath,
              turnCounter: turn,
              totalTurns: this.options.totalTurns,
              mock: this.options.mock,
              env: this.options.env,
            }),
          );
        }
      }

      this.turnFailures = turnFailures;

      if (finalMetrics === undefined) {
        throw new Error("Simulator completed without metrics");
      }

      return {
        runId: this.options.runId,
        persona: this.options.persona.key,
        totalTurns: this.options.totalTurns,
        probes: probeResults,
        overseerCheckpoints,
        turnFailures: this.turnFailures,
        finalMetrics,
        durationMs: performance.now() - started,
      };
    } finally {
      metrics.close();
      await transport.close();
    }
  }
}

export async function runSimulation(options: SimulatorRunnerOptions): Promise<SimulatorRunReport> {
  return new SimulatorRunner(options).run();
}

export function formatSimulatorReport(report: SimulatorRunReport): string {
  const lines = [
    `# Borg Simulator Run ${report.runId}`,
    "",
    `Persona: ${report.persona}`,
    `Turns: ${report.totalTurns}`,
    `Duration: ${Math.round(report.durationMs)}ms`,
    "",
    "## Final Metrics",
    "",
    `- Episodes: ${report.finalMetrics.episode_count}`,
    `- Semantic nodes: ${report.finalMetrics.semantic_node_count}`,
    `- Semantic edges: ${report.finalMetrics.semantic_edge_count}`,
    `- Open questions: ${report.finalMetrics.open_question_count}`,
    `- Active goals: ${report.finalMetrics.active_goal_count}`,
    `- Mood: valence ${report.finalMetrics.mood_valence}, arousal ${report.finalMetrics.mood_arousal}`,
    "",
    "## Probes",
    "",
  ];

  if (report.probes.length === 0) {
    lines.push("No probes scheduled.", "");
  } else {
    for (const probe of report.probes) {
      lines.push(
        `- Turn ${probe.turn}: ${probe.scenarioName} ${probe.passed ? "passed" : "failed"} -- ${probe.evidence}`,
      );
    }
    lines.push("");
  }

  lines.push("## Overseer Checkpoints", "");

  if (report.overseerCheckpoints.length === 0) {
    lines.push("No overseer checkpoints scheduled.", "");
  } else {
    for (const checkpoint of report.overseerCheckpoints) {
      lines.push(
        `- Turn ${checkpoint.turn_counter}: ${checkpoint.status} -- ${checkpoint.recommendation}`,
      );
      for (const observation of checkpoint.observations) {
        lines.push(`  - ${observation}`);
      }
    }
    lines.push("");
  }

  if (report.turnFailures.length > 0) {
    lines.push("## Turn Failures", "");
    for (const failure of report.turnFailures) {
      lines.push(`- Turn ${failure.turn}: ${failure.error}`);
    }
    lines.push("");
  }

  return `${lines.join("\n")}\n`;
}
