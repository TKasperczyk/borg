import { performance } from "node:perf_hooks";

import { BorgTransport } from "../assessor/borg-transport.js";
import { latestTurnId, readTraceEvents } from "../assessor/trace-reader.js";
import type { Scenario } from "../assessor/types.js";
import {
  AnthropicLLMClient,
  createSessionId,
  DEFAULT_CONFIG,
  type GenerationSuppressionReason,
  type BorgOpenOptions,
  type EntityId,
  type LLMClient,
  type MaintenanceCadence,
  type ReviewQueueItem,
  type SessionId,
} from "../src/index.js";
import { isNaturalSilenceSuppressionReason } from "../src/cognition/generation/types.js";

import { MetricsCapture } from "./metrics.js";
import { appendJsonlLine } from "./jsonl.js";
import {
  classifyPersonaRoleBleed,
  PersonaSession,
  type PersonaChannelTranscriptEntry,
  type PersonaTurnDraft,
  type PersonaRoleBleedDetection,
  type PriorBorgTurn,
} from "./persona.js";
import { runOverseer, type RunOverseerOptions } from "./overseer.js";
import type {
  MetricsRow,
  OverseerVerdict,
  Persona,
  SimulatorRunReport,
  SimulatorSessionRecord,
  SimulatorSuppressionRecord,
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
  personas?: readonly Persona[];
  channelName?: string;
  personaScheduler?: PersonaScheduler;
  totalTurns: number;
  metricsPath: string;
  checkEvery: number;
  maintenanceEvery?: number;
  maxSessions?: number;
  keep?: boolean;
  mock?: boolean;
  includePayloads?: boolean;
  shadowPostGenGuards?: boolean;
  pipelineCDoublePrime?: boolean;
  env?: NodeJS.ProcessEnv;
  dataDir?: string;
  tracePath?: string;
  llmClient?: BorgOpenOptions["llmClient"];
  embeddingClient?: BorgOpenOptions["embeddingClient"];
  personaRoleBleedLlmClient?: LLMClient;
  personaSession?: PersonaSession;
  personaSessions?: readonly PersonaSession[];
  overseerRunner?: (options: RunOverseerOptions) => Promise<OverseerVerdict>;
};

export type PersonaSchedulerInput = {
  turn: number;
  personas: readonly Persona[];
};

export type PersonaScheduler = {
  selectSpeakerIndex(input: PersonaSchedulerInput): number;
};

const DEFAULT_MAINTENANCE_EVERY = 10;
const MAX_PERSONAS = 4;
const PERSONA_ROLE_BLEED_EVENT = "persona_role_bleed";
const PERSONA_ROLE_BLEED_MAX_ATTEMPTS = 2;
const PERSONA_ROLE_BLEED_REJECTED_PREVIEW_CHARS = 500;
const PERSONA_CHANNEL_TRANSCRIPT_LIMIT = 10;
const BORG_OBSERVATION_MARKER_PREFIX = "[borg observation:";
export const PIPELINE_C_DOUBLE_PRIME_INCOMPATIBLE_SHADOW_MESSAGE =
  "--pipeline-c-double-prime sets per-guard modes explicitly; --shadow-post-gen-guards is incompatible";

type ChannelTranscriptLogEntry = PersonaChannelTranscriptEntry & {
  speakerIndex: number | null;
};

export const PIPELINE_C_DOUBLE_PRIME_BORG_CONFIG_OVERRIDES = {
  generation: {
    evidenceLedger: { enabled: true },
    postGenerationGuards: {
      commitment: { mode: "enforce" },
      closurePressure: { mode: "enforce" },
      relationalClaim: {
        mode: "shadow",
      },
    },
  },
} satisfies NonNullable<Scenario["borgConfigOverrides"]>;

export const ROUND_ROBIN_PERSONA_SCHEDULER: PersonaScheduler = {
  selectSpeakerIndex(input) {
    return (input.turn - 1) % input.personas.length;
  },
};

const SHADOW_POST_GEN_GUARDS_BORG_CONFIG_OVERRIDES = {
  generation: {
    postGenerationGuards: {
      commitment: {
        mode: "shadow",
      },
      relationalClaim: {
        mode: "shadow",
      },
      closurePressure: {
        mode: "shadow",
      },
    },
  },
} satisfies NonNullable<Scenario["borgConfigOverrides"]>;

function withMultiPersonaEvidenceLedger(
  overrides: NonNullable<Scenario["borgConfigOverrides"]> | undefined,
  personas: readonly Persona[],
): NonNullable<Scenario["borgConfigOverrides"]> | undefined {
  if (personas.length <= 1) {
    return overrides;
  }

  return {
    ...overrides,
    generation: {
      ...overrides?.generation,
      evidenceLedger: {
        ...overrides?.generation?.evidenceLedger,
        enabled: true,
      },
    },
  };
}

function isSessionEndingSuppression(reason: GenerationSuppressionReason | undefined): boolean {
  if (reason === undefined) return true;

  return isNaturalSilenceSuppressionReason(reason);
}

export function createSimulatorScenario(
  personaOrPersonas: Persona | readonly Persona[],
  totalTurns: number,
  options: Pick<
    SimulatorRunnerOptions,
    "shadowPostGenGuards" | "pipelineCDoublePrime" | "channelName"
  > = {},
): Scenario {
  if (options.pipelineCDoublePrime === true && options.shadowPostGenGuards === true) {
    throw new Error(PIPELINE_C_DOUBLE_PRIME_INCOMPATIBLE_SHADOW_MESSAGE);
  }

  const personas = Array.isArray(personaOrPersonas) ? personaOrPersonas : [personaOrPersonas];
  const baseBorgConfigOverrides =
    options.pipelineCDoublePrime === true
      ? PIPELINE_C_DOUBLE_PRIME_BORG_CONFIG_OVERRIDES
      : options.shadowPostGenGuards === true
        ? SHADOW_POST_GEN_GUARDS_BORG_CONFIG_OVERRIDES
        : undefined;
  const borgConfigOverrides = withMultiPersonaEvidenceLedger(baseBorgConfigOverrides, personas);

  const personaKeys = personas.map((persona) => persona.key).join("-");
  const personaNames = personas.map((persona) => persona.displayName).join(", ");
  const channelSuffix =
    personas.length === 1 ? "" : ` in ${options.channelName ?? "a group channel"}`;

  return {
    name: `simulator-${personaKeys}`,
    description: `Long-horizon simulator run for ${personaNames}${channelSuffix}.`,
    systemPrompt: personas.map((persona) => persona.systemPrompt).join("\n\n"),
    maxTurns: totalTurns,
    ...(borgConfigOverrides === undefined ? {} : { borgConfigOverrides }),
  };
}

function resolveActivePersonas(options: SimulatorRunnerOptions): readonly Persona[] {
  const personas = options.personas ?? [options.persona];

  if (personas.length === 0) {
    throw new Error("at least one persona is required");
  }

  if (personas.length > MAX_PERSONAS) {
    throw new Error(`simulator supports at most ${MAX_PERSONAS} personas`);
  }

  if (options.personas !== undefined && personas.length < 2) {
    throw new Error("--personas requires at least two personas");
  }

  if (personas.length > 1 && options.personaSession !== undefined) {
    throw new Error("personaSession is only valid for single-persona runs; use personaSessions");
  }

  return personas;
}

function resolveAudienceName(
  options: SimulatorRunnerOptions,
  personas: readonly Persona[],
): string {
  if (personas.length === 1) {
    return personas[0]?.displayName ?? options.persona.displayName;
  }

  return (
    options.channelName ?? `${personas.map((persona) => persona.displayName).join(", ")} Channel`
  );
}

function createPersonaSessions(
  options: SimulatorRunnerOptions,
  personas: readonly Persona[],
): PersonaSession[] {
  return personas.map(
    (persona, index) =>
      options.personaSessions?.[index] ??
      (personas.length === 1 && index === 0 && options.personaSession !== undefined
        ? options.personaSession
        : new PersonaSession({
            persona,
            mock: options.mock,
            env: options.env,
          })),
  );
}

function normalizeSpeakerIndex(index: number, personas: readonly Persona[]): number {
  if (!Number.isInteger(index) || index < 0 || index >= personas.length) {
    throw new Error(`persona scheduler returned invalid speaker index ${index}`);
  }

  return index;
}

function priorBorgTurnRetry(priorTurn: PriorBorgTurn): PriorBorgTurn {
  return { ...priorTurn, retry: PERSONA_ROLE_BLEED_EVENT };
}

function appendChannelTranscriptEntry(
  transcript: ChannelTranscriptLogEntry[],
  entry: ChannelTranscriptLogEntry,
): void {
  transcript.push(entry);

  if (transcript.length > PERSONA_CHANNEL_TRANSCRIPT_LIMIT) {
    transcript.splice(0, transcript.length - PERSONA_CHANNEL_TRANSCRIPT_LIMIT);
  }
}

function channelTranscriptForSpeaker(
  transcript: readonly ChannelTranscriptLogEntry[],
  speakerIndex: number,
): PersonaChannelTranscriptEntry[] {
  const lastOwnIndex = transcript.findLastIndex((entry) => entry.speakerIndex === speakerIndex);
  const sinceLastOwn = lastOwnIndex >= 0 ? transcript.slice(lastOwnIndex + 1) : transcript;

  return sinceLastOwn
    .filter((entry) => entry.speakerIndex !== speakerIndex)
    .filter((entry) => !isBorgObservationMarker(entry))
    .slice(-PERSONA_CHANNEL_TRANSCRIPT_LIMIT)
    .map((entry) => ({
      speaker_display_name: entry.speaker_display_name,
      text: entry.text,
    }));
}

function isBorgObservationMarker(entry: ChannelTranscriptLogEntry): boolean {
  return (
    entry.speaker_display_name === "Borg" &&
    entry.text.trim().startsWith(BORG_OBSERVATION_MARKER_PREFIX)
  );
}

function priorTurnForSpeaker(input: {
  priorTurn: PriorBorgTurn;
  transcript: readonly ChannelTranscriptLogEntry[];
  speakerIndex: number;
  personas: readonly Persona[];
}): PriorBorgTurn {
  if (input.personas.length <= 1) {
    return input.priorTurn;
  }

  const channelTranscript = channelTranscriptForSpeaker(input.transcript, input.speakerIndex);

  if (channelTranscript.length === 0) {
    return input.priorTurn;
  }

  return { ...input.priorTurn, channelTranscript };
}

function rejectedPreview(message: string): string {
  return message.length <= PERSONA_ROLE_BLEED_REJECTED_PREVIEW_CHARS
    ? message
    : message.slice(0, PERSONA_ROLE_BLEED_REJECTED_PREVIEW_CHARS);
}

function recordPersonaRoleBleed(input: {
  tracePath: string;
  turn: number;
  sessionId: SessionId;
  priorTurn: PriorBorgTurn;
  detection: PersonaRoleBleedDetection;
  rejectedMessage: string;
  attempt: number;
  action: "regenerated" | "aborted";
}): void {
  appendJsonlLine(
    input.tracePath,
    `${JSON.stringify({
      ts: Date.now(),
      wallMs: performance.now(),
      turnId: `simulator_turn_${input.turn}`,
      event: PERSONA_ROLE_BLEED_EVENT,
      artifact: "simulator",
      turn_counter: input.turn,
      sessionId: input.sessionId,
      prior_kind: input.priorTurn.kind,
      matched_patterns: input.detection.matched,
      category: input.detection.category,
      confidence: input.detection.confidence,
      classifier_source: input.detection.source,
      rationale: input.detection.rationale,
      rejected_preview: rejectedPreview(input.rejectedMessage),
      attempt: input.attempt,
      action: input.action,
    })}\n`,
  );
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
      await borg.review.resolve(
        review.id,
        {
          decision: "accept",
          reason: "auto-accept (long-horizon harness)",
        },
        {
          source: "auto",
          sourceProcess: "reflector",
          traceTurnId: `simulator_maintenance_${turn}_review`,
        },
      );
    } catch (error) {
      // eslint-disable-next-line no-console
      console.warn(
        `[simulator] failed to auto-accept new_insight review ${review.id} at turn ${turn}: ${formatErrorChain(error)}`,
      );
    }
  }
}

type MaintenanceBandSnapshot = {
  episode_count: number;
  semantic_node_count: number;
  semantic_edge_count: number;
  open_question_count: number;
  active_goal_count: number;
};

function flattenGoalCount(nodes: ReadonlyArray<{ children?: ReadonlyArray<unknown> }>): number {
  let count = 0;
  for (const node of nodes) {
    count += 1;
    const children = node.children;
    if (children !== undefined && children.length > 0) {
      count += flattenGoalCount(children as ReadonlyArray<{ children?: ReadonlyArray<unknown> }>);
    }
  }
  return count;
}

async function captureMaintenanceBandSnapshot(
  transport: BorgTransport,
): Promise<MaintenanceBandSnapshot> {
  const borg = transport.getBorg();
  const episodes = await borg.episodic.list({ limit: 9_999 });
  const semanticNodes = await borg.semantic.nodes.list({ limit: 9_999 });
  const semanticEdges = borg.semantic.edges.list({ includeInvalid: true });
  const openQuestions = borg.self.openQuestions.list({ status: "open", limit: 9_999 });
  const goals = borg.self.goals.list({ status: "active" });
  return {
    episode_count: episodes.items.length,
    semantic_node_count: semanticNodes.length,
    semantic_edge_count: semanticEdges.length,
    open_question_count: openQuestions.length,
    active_goal_count: flattenGoalCount(goals),
  };
}

async function runMaintenanceTick(
  transport: BorgTransport,
  turn: number,
  cadence: MaintenanceCadence,
  options: { final?: boolean } = {},
): Promise<void> {
  // Sprint 8d.4: capture pre/post snapshots so per-tick deltas surface in
  // the trace. Per-turn metrics already reflect post-maintenance state
  // by the next turn boundary, but they don't isolate the delta and
  // they hide ticks that fail or no-op. Snapshotting here lets v* runs
  // attribute semantic-graph movement (or stagnation) directly to
  // maintenance work.
  let before: MaintenanceBandSnapshot | null = null;
  try {
    before = await captureMaintenanceBandSnapshot(transport);
  } catch (error) {
    // eslint-disable-next-line no-console
    console.warn(
      `[simulator] ${cadence} pre-maintenance snapshot at turn ${turn} failed: ${formatErrorChain(error)}`,
    );
  }

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

  if (before !== null) {
    try {
      const after = await captureMaintenanceBandSnapshot(transport);
      const snapshot = {
        ts: Date.now(),
        wallMs: performance.now(),
        turnId: `simulator_maintenance_${turn}_${cadence}`,
        event: "maintenance_snapshot",
        artifact: "simulator",
        turn_counter: turn,
        cadence,
        before,
        after,
        delta: {
          episode_count: after.episode_count - before.episode_count,
          semantic_node_count: after.semantic_node_count - before.semantic_node_count,
          semantic_edge_count: after.semantic_edge_count - before.semantic_edge_count,
          open_question_count: after.open_question_count - before.open_question_count,
          active_goal_count: after.active_goal_count - before.active_goal_count,
        },
      };
      appendJsonlLine(transport.tracePath, `${JSON.stringify(snapshot)}\n`);

      if (options.final === true) {
        appendJsonlLine(
          transport.tracePath,
          `${JSON.stringify({
            ...snapshot,
            event: "maintenance_snapshot_finalized",
          })}\n`,
        );
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.warn(
        `[simulator] ${cadence} post-maintenance snapshot at turn ${turn} failed: ${formatErrorChain(error)}`,
      );
    }
  }
}

export class SimulatorRunner {
  private readonly options: SimulatorRunnerOptions;
  private turnFailures: Array<{ turn: number; error: string; attempts: number }> = [];

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

    const personas = resolveActivePersonas(this.options);
    const audienceName = resolveAudienceName(this.options, personas);
    const primaryPersona = personas[0] ?? this.options.persona;
    const scheduler = this.options.personaScheduler ?? ROUND_ROBIN_PERSONA_SCHEDULER;
    const scenario = createSimulatorScenario(personas, this.options.totalTurns, {
      shadowPostGenGuards: this.options.shadowPostGenGuards,
      pipelineCDoublePrime: this.options.pipelineCDoublePrime,
      channelName: audienceName,
    });

    if (this.options.pipelineCDoublePrime === true) {
      // eslint-disable-next-line no-console
      console.warn(
        "[simulator] Pipeline C″ active: emission-tool finalizer on; commitment and closure-pressure enforce; relational guard shadow.",
      );
    }

    const started = performance.now();
    const transport = new BorgTransport({
      runId: this.options.runId,
      scenario,
      keep: this.options.keep,
      mock: this.options.mock,
      maintenance: true,
      includeTracePayloads: this.options.includePayloads ?? true,
      env: this.options.env,
      dataDir: this.options.dataDir,
      tracePath: this.options.tracePath,
      llmClient: this.options.llmClient,
      embeddingClient: this.options.embeddingClient,
      defaultUser: primaryPersona.displayName,
    });
    const metrics = new MetricsCapture(this.options.metricsPath, {
      tracePath: transport.tracePath,
    });
    const personaSessions = createPersonaSessions(this.options, personas);
    const personaRoleBleedLlmClient =
      this.options.personaRoleBleedLlmClient ??
      (this.options.mock === true ? undefined : new AnthropicLLMClient({ env: this.options.env }));
    const overseerRunner = this.options.overseerRunner ?? runOverseer;
    const overseerCheckpoints: SimulatorRunReport["overseerCheckpoints"] = [];
    let priorBorgTurn: PriorBorgTurn = { kind: "new_session" };
    let finalMetrics: MetricsRow | undefined;
    let resultState: SimulatorRunReport["resultState"] = "completed";
    const sessions: SimulatorSessionRecord[] = [];
    const suppressionEvents: SimulatorSuppressionRecord[] = [];
    let currentSessionStartTurn = 1;
    let currentSessionId: SessionId = createSessionId();
    const sessionIds: SessionId[] = [currentSessionId];
    const maxSessions = this.options.maxSessions ?? MAX_SESSIONS_DEFAULT;
    const channelTranscript: ChannelTranscriptLogEntry[] = [];

    if (!Number.isInteger(maxSessions) || maxSessions <= 0) {
      throw new Error("maxSessions must be a positive integer");
    }

    try {
      await transport.open();
      const audienceEntityId = transport.resolveEntity(audienceName, {
        kind: personas.length === 1 ? "person" : "group",
        provenance: "transport_audience_label",
      });
      const personaEntityIds = personas.map((persona) =>
        transport.resolveEntity(persona.displayName, {
          kind: "person",
          provenance: "transport_audience_label",
        }),
      );

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
      const turnFailures: Array<{ turn: number; error: string; attempts: number }> = [];

      const attemptTurn = async (
        draft: PersonaTurnDraft,
        speakerEntityId: EntityId,
      ): Promise<{
        turnId: string;
        response: string;
        emitted: boolean;
        emissionKind: "message" | "observed" | "suppressed";
        suppressionReason?: GenerationSuppressionReason;
      }> => {
        const result = await transport.chat(draft.message, {
          audience: audienceName,
          sessionId: currentSessionId,
          senderEntityId: speakerEntityId,
        });
        const suppressionReason =
          result.emission?.kind === "suppressed" ? result.emission.reason : undefined;
        const emissionKind = result.emission?.kind ?? (result.emitted ? "message" : "suppressed");
        const response = emissionKind === "message" && result.emitted ? result.response : "";
        return {
          turnId: result.turnId,
          response,
          emitted: result.emitted,
          emissionKind,
          suppressionReason,
        };
      };

      for (let turn = 1; turn <= this.options.totalTurns; turn += 1) {
        let success: {
          turnId: string;
          response: string;
          emitted: boolean;
          emissionKind: "message" | "observed" | "suppressed";
          suppressionReason?: GenerationSuppressionReason;
          transportChatAttempts: number;
        } | null = null;
        let attemptError: unknown = null;
        let attemptsMade = 0;
        const speakerIndex = normalizeSpeakerIndex(
          scheduler.selectSpeakerIndex({ turn, personas }),
          personas,
        );
        const speaker = personas[speakerIndex]!;
        const personaSession = personaSessions[speakerIndex]!;
        const speakerEntityId = personaEntityIds[speakerIndex] ?? audienceEntityId;
        const speakerPriorTurn = priorTurnForSpeaker({
          priorTurn: priorBorgTurn,
          transcript: channelTranscript,
          speakerIndex,
          personas,
        });
        let draft = await personaSession.prepareNextTurn(speakerPriorTurn);
        let roleBleedAborted = false;

        for (
          let bleedAttempt = 1;
          bleedAttempt <= PERSONA_ROLE_BLEED_MAX_ATTEMPTS;
          bleedAttempt += 1
        ) {
          const bleedDetection = await classifyPersonaRoleBleed({
            message: draft.message,
            llmClient: personaRoleBleedLlmClient,
            model: DEFAULT_CONFIG.anthropic.models.recallExpansion,
            personaName: speaker.displayName,
          });

          if (!bleedDetection.flagged) {
            break;
          }

          const finalBleedAttempt = bleedAttempt === PERSONA_ROLE_BLEED_MAX_ATTEMPTS;
          recordPersonaRoleBleed({
            tracePath: transport.tracePath,
            turn,
            sessionId: currentSessionId,
            priorTurn: speakerPriorTurn,
            detection: bleedDetection,
            rejectedMessage: draft.message,
            attempt: bleedAttempt,
            action: finalBleedAttempt ? "aborted" : "regenerated",
          });
          personaSession.rollback(draft);

          if (finalBleedAttempt) {
            const detail = `${PERSONA_ROLE_BLEED_EVENT}: ${
              bleedDetection.matched.length > 0
                ? bleedDetection.matched.join(", ")
                : bleedDetection.category
            }`;
            turnFailures.push({ turn, error: detail, attempts: 0 });
            await metrics.captureAborted(transport.getBorg(), turn, {
              sessionId: currentSessionId,
              sessionIds,
              transportChatAttempts: 0,
              failureReason: detail,
              turnId: `${PERSONA_ROLE_BLEED_EVENT}_${turn}`,
            });
            consecutiveFailures += 1;
            // eslint-disable-next-line no-console
            console.warn(`[simulator] turn ${turn} failed before Borg chat: ${detail}`);

            if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
              throw new Error(
                `Simulator aborting: ${consecutiveFailures} consecutive turn failures (last: ${detail})`,
              );
            }
            roleBleedAborted = true;
            break;
          }

          draft = await personaSession.prepareNextTurn(priorBorgTurnRetry(speakerPriorTurn));
        }

        if (roleBleedAborted) {
          continue;
        }

        for (let attempt = 0; attempt <= TRANSIENT_RETRY_ATTEMPTS; attempt += 1) {
          attemptsMade = attempt + 1;
          const traceBeforeCount = readTraceEvents(transport.tracePath).length;
          try {
            const result = await attemptTurn(draft, speakerEntityId);
            success = {
              ...result,
              transportChatAttempts: attemptsMade,
            };
            attemptError = null;
            break;
          } catch (error) {
            attemptError = error;
            const failedTurnId = latestTurnId(
              readTraceEvents(transport.tracePath).slice(traceBeforeCount),
            );

            if (failedTurnId !== null) {
              await metrics.captureAborted(transport.getBorg(), turn, {
                event: "aborted_attempt",
                sessionId: currentSessionId,
                sessionIds,
                transportChatAttempts: attemptsMade,
                failureReason: formatErrorChain(error),
                turnId: failedTurnId,
              });
            }

            if (attempt < TRANSIENT_RETRY_ATTEMPTS) {
              await new Promise((resolve) =>
                setTimeout(resolve, TRANSIENT_RETRY_DELAY_MS * (attempt + 1)),
              );
            }
          }
        }

        if (success === null) {
          const detail = formatErrorChain(attemptError);
          personaSession.rollback(draft);
          turnFailures.push({ turn, error: detail, attempts: attemptsMade });
          await metrics.captureAborted(transport.getBorg(), turn, {
            sessionId: currentSessionId,
            sessionIds,
            transportChatAttempts: attemptsMade,
            failureReason: detail,
          });
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

        personaSession.commit(draft, success.response);
        appendChannelTranscriptEntry(channelTranscript, {
          speakerIndex,
          speaker_display_name: speaker.displayName,
          text: draft.message,
        });

        if (success.emissionKind === "message" && success.response.trim().length > 0) {
          appendChannelTranscriptEntry(channelTranscript, {
            speakerIndex: null,
            speaker_display_name: "Borg",
            text: success.response,
          });
        }
        consecutiveFailures = 0;

        const overseerDue =
          Number.isInteger(this.options.checkEvery) &&
          this.options.checkEvery > 0 &&
          turn % this.options.checkEvery === 0;
        const suppressionReason = success.emitted ? undefined : success.suppressionReason;
        const isObserveTurn = !success.emitted && success.emissionKind === "observed";
        const continuesSuppressedSession =
          !success.emitted &&
          !isObserveTurn &&
          suppressionReason !== undefined &&
          !isSessionEndingSuppression(suppressionReason);

        let heavyMaintenanceRan = false;

        if (turn % maintenanceEvery === 0) {
          await runMaintenanceTick(transport, turn, "light", {
            final: turn === this.options.totalTurns,
          });
        }

        if (overseerDue) {
          await runMaintenanceTick(transport, turn, "heavy", {
            final: turn === this.options.totalTurns,
          });
          heavyMaintenanceRan = true;
        }

        if (
          !success.emitted &&
          !isObserveTurn &&
          !continuesSuppressedSession &&
          !heavyMaintenanceRan
        ) {
          await runMaintenanceTick(transport, turn, "heavy", {
            final: turn === this.options.totalTurns,
          });
          heavyMaintenanceRan = true;
        }

        finalMetrics = await metrics.capture(transport.getBorg(), success.turnId, turn, {
          sessionId: currentSessionId,
          sessionIds,
          transportChatAttempts: success.transportChatAttempts,
          overseerDueOnSuppressedTurn: !success.emitted && !isObserveTurn && overseerDue,
        });

        if (overseerDue) {
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

        if (!success.emitted) {
          if (isObserveTurn) {
            continue;
          }

          if (continuesSuppressedSession) {
            suppressionEvents.push({
              sessionIndex: sessions.length,
              sessionId: currentSessionId,
              turn,
              reason: suppressionReason,
            });
            priorBorgTurn = { kind: "continued_suppression", reason: suppressionReason };
            continue;
          }

          sessions.push({
            sessionIndex: sessions.length,
            sessionId: currentSessionId,
            startedAtTurn: currentSessionStartTurn,
            endedAtTurn: turn,
            endReason: "suppression",
            ...(suppressionReason === undefined ? {} : { suppressionReason }),
          });

          if (sessions.length >= maxSessions) {
            resultState = "max_sessions_reached";
            break;
          }

          const gap =
            SESSION_GAP_DESCRIPTIONS[sessions.length % SESSION_GAP_DESCRIPTIONS.length] ??
            SESSION_GAP_DESCRIPTIONS[0]!;
          for (const session of personaSessions) {
            session.startNewSession();
          }
          channelTranscript.length = 0;
          priorBorgTurn = { kind: "new_session", gapContext: gap };
          currentSessionStartTurn = turn + 1;
          currentSessionId = createSessionId();
          sessionIds.push(currentSessionId);
          continue;
        }

        priorBorgTurn = { kind: "normal", text: success.response };
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
        persona: primaryPersona.key,
        personas: personas.map((persona) => persona.key),
        audience: audienceName,
        totalTurns: this.options.totalTurns,
        resultState,
        sessions,
        suppressionEvents,
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
  const participantLine =
    report.personas.length <= 1
      ? `Persona: ${report.persona}`
      : `Personas: ${report.personas.join(", ")}`;
  const lines = [
    `# Borg Simulator Run ${report.runId}`,
    "",
    participantLine,
    `Audience: ${report.audience}`,
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
      const reason =
        session.suppressionReason === undefined ? "" : ` (${session.suppressionReason})`;
      lines.push(
        `- Session ${session.sessionIndex} (turns ${session.startedAtTurn}-${session.endedAtTurn}): ended via ${session.endReason}${reason}`,
      );
    }
    lines.push("");
  }

  if (report.suppressionEvents.length > 0) {
    lines.push("## Continued Suppressions", "");
    for (const event of report.suppressionEvents) {
      lines.push(
        `- Turn ${event.turn} in session ${event.sessionIndex}: ${event.reason}; session continued`,
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
      lines.push(`- Turn ${failure.turn} after ${failure.attempts} attempts: ${failure.error}`);
    }
    lines.push("");
  }

  return `${lines.join("\n")}\n`;
}
