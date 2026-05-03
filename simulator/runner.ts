import { performance } from "node:perf_hooks";

import { BorgTransport } from "../assessor/borg-transport.js";
import type { Scenario } from "../assessor/types.js";
import {
  createSessionId,
  type BorgOpenOptions,
  type MaintenanceCadence,
  type ReviewQueueItem,
  type SessionId,
} from "../src/index.js";

import { MetricsCapture } from "./metrics.js";
import { PersonaSession } from "./persona.js";
import { runOverseer, type RunOverseerOptions } from "./overseer.js";
import type {
  MetricsRow,
  OverseerVerdict,
  Persona,
  SimulatorRunReport,
  SimulatorSessionRecord,
} from "./types.js";

const SESSION_GAP_DESCRIPTIONS: readonly string[] = [
  "It's the next morning. You're at your desk with coffee.",
  "It's the next evening. You're back on the couch after dinner.",
  "Two days have passed; it's a Saturday afternoon.",
  "It's late at night a few days later; you can't sleep.",
  "It's the following weekend; the kitchen still smells like breakfast.",
  "A week has gone by. It's a quiet weekday lunch break.",
];

const MAX_SESSIONS_DEFAULT = 12;

export type SimulatorRunnerOptions = {
  runId: string;
  persona: Persona;
  totalTurns: number;
  metricsPath: string;
  checkEvery: number;
  maintenanceEvery?: number;
  maxSessions?: number;
  keep?: boolean;
  mock?: boolean;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
  tracePath?: string;
  llmClient?: BorgOpenOptions["llmClient"];
  embeddingClient?: BorgOpenOptions["embeddingClient"];
  personaSession?: PersonaSession;
  overseerRunner?: (options: RunOverseerOptions) => Promise<OverseerVerdict>;
};

const DEFAULT_MAINTENANCE_EVERY = 10;

function simulatorScenario(persona: Persona, totalTurns: number): Scenario {
  return {
    name: `simulator-${persona.key}`,
    description: `Long-horizon simulator run for ${persona.displayName}.`,
    systemPrompt: persona.systemPrompt,
    maxTurns: totalTurns,
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

async function autoAcceptNewInsightReviews(transport: BorgTransport, turn: number): Promise<void> {
  const borg = transport.getBorg();
  let reviews: ReviewQueueItem[];

  try {
    reviews = borg.review.list({ kind: "new_insight", openOnly: true });
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn(
      `[simulator] failed to list new_insight reviews after maintenance at turn ${turn}: ${formatErrorChain(error)}`,
    );
    return;
  }

  for (const review of reviews) {
    try {
      await borg.review.resolve(review.id, {
        decision: "accept",
        reason: "auto-accept (long-horizon harness)",
      });
    } catch (error) {
      // eslint-disable-next-line no-console
      console.warn(
        `[simulator] failed to auto-accept new_insight review ${review.id} at turn ${turn}: ${formatErrorChain(error)}`,
      );
    }
  }
}

async function runMaintenanceTick(
  transport: BorgTransport,
  turn: number,
  cadence: MaintenanceCadence,
): Promise<void> {
  try {
    await transport.getBorg().maintenance.scheduler.tick(cadence);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn(
      `[simulator] ${cadence} maintenance tick at turn ${turn} failed: ${formatErrorChain(error)}`,
    );
    return;
  }

  await autoAcceptNewInsightReviews(transport, turn);
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

    const maintenanceEvery = this.options.maintenanceEvery ?? DEFAULT_MAINTENANCE_EVERY;

    if (!Number.isInteger(maintenanceEvery) || maintenanceEvery <= 0) {
      throw new Error("maintenanceEvery must be a positive integer");
    }

    const started = performance.now();
    const transport = new BorgTransport({
      runId: this.options.runId,
      scenario: simulatorScenario(this.options.persona, this.options.totalTurns),
      keep: this.options.keep,
      mock: this.options.mock,
      maintenance: true,
      env: this.options.env,
      dataDir: this.options.dataDir,
      tracePath: this.options.tracePath,
      llmClient: this.options.llmClient,
      embeddingClient: this.options.embeddingClient,
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
    const overseerRunner = this.options.overseerRunner ?? runOverseer;
    const overseerCheckpoints: SimulatorRunReport["overseerCheckpoints"] = [];
    let lastBorgResponse: string | null = null;
    let finalMetrics: MetricsRow | undefined;
    let resultState: SimulatorRunReport["resultState"] = "completed";
    const sessions: SimulatorSessionRecord[] = [];
    let currentSessionStartTurn = 1;
    let currentSessionId: SessionId = createSessionId();
    const sessionIds: SessionId[] = [currentSessionId];
    const maxSessions = this.options.maxSessions ?? MAX_SESSIONS_DEFAULT;

    if (!Number.isInteger(maxSessions) || maxSessions <= 0) {
      throw new Error("maxSessions must be a positive integer");
    }

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

      const attemptTurn = async (): Promise<{
        turnId: string;
        response: string;
        emitted: boolean;
      }> => {
        const message = await persona.nextTurn(lastBorgResponse);
        const result = await transport.chat(message, {
          audience: this.options.persona.displayName,
          sessionId: currentSessionId,
        });
        return { turnId: result.turnId, response: result.response, emitted: result.emitted };
      };

      for (let turn = 1; turn <= this.options.totalTurns; turn += 1) {
        let success: { turnId: string; response: string; emitted: boolean } | null = null;
        let attemptError: unknown = null;

        for (let attempt = 0; attempt <= TRANSIENT_RETRY_ATTEMPTS; attempt += 1) {
          try {
            success = await attemptTurn();
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

        consecutiveFailures = 0;

        finalMetrics = await metrics.capture(transport.getBorg(), success.turnId, turn, {
          sessionId: currentSessionId,
          sessionIds,
        });

        if (!success.emitted) {
          // Borg suppressed -- in a real product this means the user
          // walked away. Treat it the same way: close out this session,
          // run a heavy maintenance pass (the "time gap" is when offline
          // work would actually fire in production), and rotate the
          // persona to a fresh session so the run keeps going.
          sessions.push({
            sessionIndex: sessions.length,
            sessionId: currentSessionId,
            startedAtTurn: currentSessionStartTurn,
            endedAtTurn: turn,
            endReason: "suppression",
          });

          if (sessions.length >= maxSessions) {
            resultState = "max_sessions_reached";
            break;
          }

          await runMaintenanceTick(transport, turn, "heavy");
          const gap =
            SESSION_GAP_DESCRIPTIONS[sessions.length % SESSION_GAP_DESCRIPTIONS.length] ??
            SESSION_GAP_DESCRIPTIONS[0]!;
          persona.startNewSession(gap);
          lastBorgResponse = null;
          currentSessionStartTurn = turn + 1;
          currentSessionId = createSessionId();
          sessionIds.push(currentSessionId);
          continue;
        }

        lastBorgResponse = success.response;

        const overseerDue =
          Number.isInteger(this.options.checkEvery) &&
          this.options.checkEvery > 0 &&
          turn % this.options.checkEvery === 0;

        if (turn % maintenanceEvery === 0) {
          await runMaintenanceTick(transport, turn, "light");
        }

        if (overseerDue) {
          await runMaintenanceTick(transport, turn, "heavy");
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

      if (resultState === "completed" && finalMetrics.turn_counter >= currentSessionStartTurn) {
        sessions.push({
          sessionIndex: sessions.length,
          sessionId: currentSessionId,
          startedAtTurn: currentSessionStartTurn,
          endedAtTurn: finalMetrics.turn_counter,
          endReason: "run_complete",
        });
      }

      return {
        runId: this.options.runId,
        persona: this.options.persona.key,
        totalTurns: this.options.totalTurns,
        resultState,
        sessions,
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
    `Result: ${report.resultState}`,
    `Sessions: ${report.sessions.length}`,
    `Duration: ${Math.round(report.durationMs)}ms`,
    "",
    "## Final Metrics",
    "",
    `- Episodes: ${report.finalMetrics.episode_count}`,
    `- Semantic nodes: ${report.finalMetrics.semantic_node_count}`,
    `- Semantic edges: ${report.finalMetrics.semantic_edge_count}`,
    `- Semantic added since previous check: ${report.finalMetrics.semantic_nodes_added_since_last_check} nodes, ${report.finalMetrics.semantic_edges_added_since_last_check} edges`,
    `- Open questions: ${report.finalMetrics.open_question_count}`,
    `- Active goals: ${report.finalMetrics.active_goal_count}`,
    `- Mood: valence ${report.finalMetrics.mood_valence}, arousal ${report.finalMetrics.mood_arousal}`,
    "",
  ];

  if (report.sessions.length > 0) {
    lines.push("## Sessions", "");
    for (const session of report.sessions) {
      lines.push(
        `- Session ${session.sessionIndex} (turns ${session.startedAtTurn}-${session.endedAtTurn}): ended via ${session.endReason}`,
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
