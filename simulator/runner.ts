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

export class SimulatorRunner {
  private readonly options: SimulatorRunnerOptions;

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

      for (let turn = 1; turn <= this.options.totalTurns; turn += 1) {
        const scheduledProbe = probes[turn];
        let turnId: string;

        if (scheduledProbe !== undefined) {
          const probe = await probeRunner({
            scenarioName: scheduledProbe,
            transport,
            turnNumber: turn,
          });
          probeResults.push(stripProbe(probe));
          lastBorgResponse = probe.response;
          turnId = probe.turnId;
        } else {
          const message = await persona.nextTurn(lastBorgResponse);
          const result = await transport.chat(message);
          lastBorgResponse = result.response;
          turnId = result.turnId;
        }

        finalMetrics = await metrics.capture(transport.getBorg(), turnId, turn);

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

      if (finalMetrics === undefined) {
        throw new Error("Simulator completed without metrics");
      }

      return {
        runId: this.options.runId,
        persona: this.options.persona.key,
        totalTurns: this.options.totalTurns,
        probes: probeResults,
        overseerCheckpoints,
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

  return `${lines.join("\n")}\n`;
}
