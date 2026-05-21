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
import { capabilityFindingMetrics, simulatorHealthWarningsForRows } from "./health-warnings.js";
import { buildMemorySnapshotMarkdown } from "./memory-snapshot.js";
import { appendJsonlLine } from "./jsonl.js";
import {
  classifyPersonaRoleBleed,
  PersonaSession,
  type PersonaChannelTranscriptEntry,
  type PersonaTurnDraft,
  type PersonaRoleBleedDetection,
  type PriorBorgTurn,
} from "./persona.js";
import { runOverseer, type FindingCarryoverCache, type RunOverseerOptions } from "./overseer.js";
import { statusFromSeverity, statusImpactSeverity, statusSeverity } from "./status-severity.js";
import type {
  MetricsRow,
  OverseerVerdict,
  Persona,
  SimulatorHealthWarning,
  SimulatorHealthWarningKind,
  SimulatorPersonaFailureRecord,
  SimulatorRunReport,
  SimulatorSessionRecord,
  SimulatorSuppressionRecord,
  SimulatorBorgBehavioralSuppressionRecord,
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
  scenarioKey?: string;
  metricsPath: string;
  overseerAuditPath?: string;
  checkEvery: number;
  maintenanceEvery?: number;
  maxSessions?: number;
  keep?: boolean;
  mock?: boolean;
  includePayloads?: boolean;
  shadowPostGenGuards?: boolean;
  emissionBaseline?: boolean;
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
const PERSONA_ROLE_BLEED_RETRY = "persona_role_bleed";
const PERSONA_ROLE_BLEED_EVENT = "persona.role_bleed.rejected";
const PERSONA_ROLE_BLEED_MAX_ATTEMPTS = 2;
const PERSONA_ROLE_BLEED_REJECTED_PREVIEW_CHARS = 500;
const PERSONA_CHANNEL_TRANSCRIPT_LIMIT = 10;
const BORG_OBSERVATION_MARKER_PREFIX = "[borg observation:";
export const EMISSION_BASELINE_INCOMPATIBLE_SHADOW_MESSAGE =
  "--emission-baseline sets per-guard modes explicitly; --shadow-post-gen-guards is incompatible";

type ChannelTranscriptLogEntry = PersonaChannelTranscriptEntry & {
  speakerIndex: number | null;
};

export const EMISSION_BASELINE_BORG_CONFIG_OVERRIDES = {
  generation: {
    evidenceLedger: { enabled: true },
    postGenerationGuards: {
      commitment: { mode: "enforce" },
      closurePressure: { mode: "enforce" },
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

function incrementSuppressionReason(
  counts: Record<string, number>,
  reason: GenerationSuppressionReason,
): void {
  counts[reason] = (counts[reason] ?? 0) + 1;
}

function postGenerationRejectedReasonForTurn(
  tracePath: string,
  turnId: string,
): GenerationSuppressionReason | undefined {
  const record = [...readTraceEvents(tracePath)]
    .reverse()
    .find((event) => event.turnId === turnId && event.event === "post_generation.rejected");
  const reason = record?.reason;

  return typeof reason === "string" ? (reason as GenerationSuppressionReason) : undefined;
}

export function createSimulatorScenario(
  personaOrPersonas: Persona | readonly Persona[],
  totalTurns: number,
  options: Pick<
    SimulatorRunnerOptions,
    "shadowPostGenGuards" | "emissionBaseline" | "channelName"
  > = {},
): Scenario {
  if (options.emissionBaseline === true && options.shadowPostGenGuards === true) {
    throw new Error(EMISSION_BASELINE_INCOMPATIBLE_SHADOW_MESSAGE);
  }

  const personas = Array.isArray(personaOrPersonas) ? personaOrPersonas : [personaOrPersonas];
  const baseBorgConfigOverrides =
    options.emissionBaseline === true
      ? EMISSION_BASELINE_BORG_CONFIG_OVERRIDES
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
  return { ...priorTurn, retry: PERSONA_ROLE_BLEED_RETRY };
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

function defaultOverseerAuditPath(metricsPath: string): string {
  if (metricsPath.endsWith("-metrics.jsonl")) {
    return `${metricsPath.slice(0, -"-metrics.jsonl".length)}-overseer-audit.jsonl`;
  }

  if (metricsPath.endsWith(".jsonl")) {
    return `${metricsPath.slice(0, -".jsonl".length)}-overseer-audit.jsonl`;
  }

  return `${metricsPath}-overseer-audit.jsonl`;
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

function simulatorPersonaFailureReason(error: unknown): string {
  const detail = formatErrorChain(error);
  const normalized = detail.toLowerCase();
  const prefix = normalized.includes("refus") ? "persona_refused" : "persona_malformed";

  return `${prefix}: ${detail}`;
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
        event: "maintenance_snapshot.completed",
        artifact: "simulator",
        final: false,
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
            event: "maintenance_snapshot.completed",
            final: true,
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

function endSimulatorSession(
  transport: BorgTransport,
  sessionId: SessionId,
  options: { nextSessionId?: SessionId } = {},
): void {
  (
    transport.getBorg() as {
      endSession?: (sessionId: SessionId, options?: { nextSessionId?: SessionId }) => void;
    }
  ).endSession?.(sessionId, options);
}

export class SimulatorRunner {
  private readonly options: SimulatorRunnerOptions;
  private turnFailures: Array<{ turn: number; error: string; attempts: number }> = [];
  private simulatorPersonaFailures: SimulatorPersonaFailureRecord[] = [];

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
      emissionBaseline: this.options.emissionBaseline,
      channelName: audienceName,
    });

    if (this.options.emissionBaseline === true) {
      // eslint-disable-next-line no-console
      console.warn(
        "[simulator] Emission baseline active: evidence ledger on; commitment and closure-pressure enforce.",
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
      defaultUser: personas.length === 1 ? primaryPersona.displayName : undefined,
    });
    const metrics = new MetricsCapture(this.options.metricsPath, {
      tracePath: transport.tracePath,
      scenarioKey: this.options.scenarioKey,
    });
    const personaSessions = createPersonaSessions(this.options, personas);
    const personaRoleBleedLlmClient =
      this.options.personaRoleBleedLlmClient ??
      (this.options.mock === true ? undefined : new AnthropicLLMClient({ env: this.options.env }));
    const overseerRunner = this.options.overseerRunner ?? runOverseer;
    const overseerCheckpoints: SimulatorRunReport["overseerCheckpoints"] = [];
    const overseerFindingCarryoverCache: FindingCarryoverCache = new Map();
    let priorBorgTurn: PriorBorgTurn = { kind: "new_session" };
    let finalMetrics: MetricsRow | undefined;
    let resultState: SimulatorRunReport["resultState"] = "completed";
    const sessions: SimulatorSessionRecord[] = [];
    const suppressionEvents: SimulatorSuppressionRecord[] = [];
    let currentSessionStartTurn = 1;
    let currentSessionId: SessionId = createSessionId();
    let currentSessionEnded = false;
    let lastOverseerCheckpointTurn = 0;
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
      const simulatorPersonaFailures: SimulatorPersonaFailureRecord[] = [];
      const borgBehavioralSuppressions: SimulatorBorgBehavioralSuppressionRecord[] = [];
      let simulatorPersonaFailureCount = 0;
      let borgHardAbortedTurnCount = 0;
      let borgIntentionalSuppressionCount = 0;
      const borgIntentionalSuppressionsByReason: Record<string, number> = {};

      const attemptTurn = async (
        turn: number,
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
          globalTurnCounter: turn,
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
      const checkEvery = this.options.checkEvery;
      const overseerSchedulingEnabled = Number.isInteger(checkEvery) && checkEvery > 0;
      const runOverseerCheckpoint = async (turnCounter: number): Promise<void> => {
        const auditWindowStartTurn = lastOverseerCheckpointTurn + 1;
        const memorySnapshotMarkdown = await buildMemorySnapshotMarkdown({
          transport,
          sessionIds,
        });

        overseerCheckpoints.push(
          await overseerRunner({
            transport,
            metricsPath: this.options.metricsPath,
            auditContextPath:
              this.options.overseerAuditPath ?? defaultOverseerAuditPath(this.options.metricsPath),
            auditWindowStartTurn,
            turnCounter,
            totalTurns: this.options.totalTurns,
            memorySnapshotMarkdown,
            mock: this.options.mock,
            env: this.options.env,
            carryoverCache: overseerFindingCarryoverCache,
          }),
        );
        lastOverseerCheckpointTurn = turnCounter;
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
        let draft: PersonaTurnDraft;
        try {
          draft = await personaSession.prepareNextTurn(speakerPriorTurn);
        } catch (error) {
          const detail = simulatorPersonaFailureReason(error);
          simulatorPersonaFailures.push({ turn, error: detail, attempts: 0 });
          simulatorPersonaFailureCount += 1;
          await metrics.captureAborted(transport.getBorg(), turn, {
            sessionId: currentSessionId,
            sessionIds,
            transportChatAttempts: 0,
            failureReason: detail,
            turnId: `simulator_persona_failure_${turn}`,
            simulatorPersonaFailures: simulatorPersonaFailureCount,
            borgHardAbortedTurns: borgHardAbortedTurnCount,
            borgIntentionalSuppressions: borgIntentionalSuppressionCount,
            borgIntentionalSuppressionsByReason,
          });
          consecutiveFailures += 1;
          // eslint-disable-next-line no-console
          console.warn(`[simulator] turn ${turn} failed before Borg chat: ${detail}`);

          if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            throw new Error(
              `Simulator aborting: ${consecutiveFailures} consecutive turn failures (last: ${detail})`,
            );
          }
          continue;
        }
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
            const detail = `${PERSONA_ROLE_BLEED_RETRY}: ${
              bleedDetection.matched.length > 0
                ? bleedDetection.matched.join(", ")
                : bleedDetection.category
            }`;
            simulatorPersonaFailures.push({ turn, error: detail, attempts: 0 });
            simulatorPersonaFailureCount += 1;
            await metrics.captureAborted(transport.getBorg(), turn, {
              sessionId: currentSessionId,
              sessionIds,
              transportChatAttempts: 0,
              failureReason: detail,
              turnId: `${PERSONA_ROLE_BLEED_RETRY}_${turn}`,
              simulatorPersonaFailures: simulatorPersonaFailureCount,
              borgHardAbortedTurns: borgHardAbortedTurnCount,
              borgIntentionalSuppressions: borgIntentionalSuppressionCount,
              borgIntentionalSuppressionsByReason,
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

          try {
            draft = await personaSession.prepareNextTurn(priorBorgTurnRetry(speakerPriorTurn));
          } catch (error) {
            const detail = simulatorPersonaFailureReason(error);
            simulatorPersonaFailures.push({ turn, error: detail, attempts: 0 });
            simulatorPersonaFailureCount += 1;
            await metrics.captureAborted(transport.getBorg(), turn, {
              sessionId: currentSessionId,
              sessionIds,
              transportChatAttempts: 0,
              failureReason: detail,
              turnId: `simulator_persona_failure_${turn}`,
              simulatorPersonaFailures: simulatorPersonaFailureCount,
              borgHardAbortedTurns: borgHardAbortedTurnCount,
              borgIntentionalSuppressions: borgIntentionalSuppressionCount,
              borgIntentionalSuppressionsByReason,
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
        }

        if (roleBleedAborted) {
          continue;
        }

        for (let attempt = 0; attempt <= TRANSIENT_RETRY_ATTEMPTS; attempt += 1) {
          attemptsMade = attempt + 1;
          const traceBeforeCount = readTraceEvents(transport.tracePath).length;
          try {
            const result = await attemptTurn(turn, draft, speakerEntityId);
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
                simulatorPersonaFailures: simulatorPersonaFailureCount,
                borgHardAbortedTurns: borgHardAbortedTurnCount,
                borgIntentionalSuppressions: borgIntentionalSuppressionCount,
                borgIntentionalSuppressionsByReason,
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
          borgHardAbortedTurnCount += 1;
          await metrics.captureAborted(transport.getBorg(), turn, {
            sessionId: currentSessionId,
            sessionIds,
            transportChatAttempts: attemptsMade,
            failureReason: detail,
            simulatorPersonaFailures: simulatorPersonaFailureCount,
            borgHardAbortedTurns: borgHardAbortedTurnCount,
            borgIntentionalSuppressions: borgIntentionalSuppressionCount,
            borgIntentionalSuppressionsByReason,
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

        const overseerDue = overseerSchedulingEnabled && turn % checkEvery === 0;
        const isObserveTurn = !success.emitted && success.emissionKind === "observed";
        const suppressionReason =
          !success.emitted && !isObserveTurn
            ? (postGenerationRejectedReasonForTurn(transport.tracePath, success.turnId) ??
              success.suppressionReason)
            : undefined;
        const continuesSuppressedSession =
          !success.emitted &&
          !isObserveTurn &&
          suppressionReason !== undefined &&
          !isSessionEndingSuppression(suppressionReason);

        if (!success.emitted && !isObserveTurn && suppressionReason !== undefined) {
          borgIntentionalSuppressionCount += 1;
          incrementSuppressionReason(borgIntentionalSuppressionsByReason, suppressionReason);
          borgBehavioralSuppressions.push({
            sessionIndex: sessions.length,
            sessionId: currentSessionId,
            turn,
            reason: suppressionReason,
            sessionContinued: continuesSuppressedSession,
          });
        }

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

        const sessionEndsOnThisTurn =
          (!success.emitted && !isObserveTurn && !continuesSuppressedSession) ||
          turn === this.options.totalTurns;
        const nextSessionIdForSuppression =
          !success.emitted &&
          !isObserveTurn &&
          !continuesSuppressedSession &&
          turn !== this.options.totalTurns &&
          sessions.length + 1 < maxSessions
            ? createSessionId()
            : null;

        if (sessionEndsOnThisTurn && !currentSessionEnded) {
          endSimulatorSession(transport, currentSessionId, {
            ...(nextSessionIdForSuppression === null
              ? {}
              : { nextSessionId: nextSessionIdForSuppression }),
          });
          currentSessionEnded = true;
        }

        finalMetrics = await metrics.capture(transport.getBorg(), success.turnId, turn, {
          sessionId: currentSessionId,
          sessionIds,
          transportChatAttempts: success.transportChatAttempts,
          overseerDueOnSuppressedTurn: !success.emitted && !isObserveTurn && overseerDue,
          simulatorPersonaFailures: simulatorPersonaFailureCount,
          borgHardAbortedTurns: borgHardAbortedTurnCount,
          borgIntentionalSuppressions: borgIntentionalSuppressionCount,
          borgIntentionalSuppressionsByReason,
        });

        if (overseerDue) {
          await runOverseerCheckpoint(turn);
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

          if (turn === this.options.totalTurns) {
            currentSessionStartTurn = turn + 1;
            continue;
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
          currentSessionId = nextSessionIdForSuppression ?? createSessionId();
          sessionIds.push(currentSessionId);
          currentSessionEnded = false;
          continue;
        }

        priorBorgTurn = { kind: "normal", text: success.response };
      }

      this.turnFailures = turnFailures;
      this.simulatorPersonaFailures = simulatorPersonaFailures;

      if (finalMetrics === undefined) {
        throw new Error("Simulator completed without metrics");
      }

      if (overseerSchedulingEnabled && lastOverseerCheckpointTurn < finalMetrics.turn_counter) {
        await runOverseerCheckpoint(finalMetrics.turn_counter);
      }

      if (resultState === "completed" && finalMetrics.turn_counter >= currentSessionStartTurn) {
        sessions.push({
          sessionIndex: sessions.length,
          sessionId: currentSessionId,
          startedAtTurn: currentSessionStartTurn,
          endedAtTurn: finalMetrics.turn_counter,
          endReason: "run_complete",
        });
        if (!currentSessionEnded) {
          endSimulatorSession(transport, currentSessionId);
          currentSessionEnded = true;
        }
      }

      finalMetrics = metrics.finalizeLastRow(capabilityFindingMetrics(overseerCheckpoints));

      const postHocHealthWarnings = simulatorHealthWarningsForRows(metrics.listRows(), {
        scenarioKey: this.options.scenarioKey,
        overseerCheckpoints,
      }).filter(
        (warning) =>
          warning.kind === "capability_overclaim_count_high" ||
          warning.kind === "capability_ambiguity_count_high",
      );

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
        healthWarnings: [...metrics.listHealthWarnings(), ...postHocHealthWarnings],
        turnFailures: this.turnFailures,
        simulatorPersonaFailures: this.simulatorPersonaFailures,
        borgBehavioralSuppressions,
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

function reportValue(value: string | number | undefined): string {
  return value === undefined ? "n/a" : String(value);
}

function reportQuotedSpan(value: string | undefined): string {
  if (value === undefined) {
    return "n/a";
  }

  return value.length <= 180 ? value : `${value.slice(0, 177)}...`;
}

function reportFindingLine(
  finding: OverseerVerdict["findings"][number] | OverseerVerdict["rejected_findings"][number],
): string {
  return [
    `[${finding.category} ${finding.claim_status}]`,
    `source=${finding.source_kind}`,
    `impact=${finding.status_impact ?? "n/a"}`,
    `stream=${reportValue(finding.assistant_stream_entry_id)}`,
    `ts=${reportValue(finding.assistant_ts)}`,
    `turn=${reportValue(finding.metrics_turn_counter)}`,
    `temporal=${finding.temporal_direction ?? "n/a"}`,
    `quote="${reportQuotedSpan(finding.quoted_emitted_span)}"`,
    `evidence=${finding.evidence_summary}`,
  ].join(" ");
}

function isCarryoverDemotedFinding(finding: OverseerVerdict["findings"][number]): boolean {
  return finding.carryover_demoted === true;
}

function findingImpactSeverity(finding: OverseerVerdict["findings"][number]): number {
  return statusImpactSeverity(finding.status_impact);
}

function isSubstrateFinding(finding: OverseerVerdict["findings"][number]): boolean {
  return finding.category === "I";
}

function isCapabilityFinding(finding: OverseerVerdict["findings"][number]): boolean {
  return finding.category === "K";
}

function statusFromFindings(
  findings: readonly OverseerVerdict["findings"][number][],
): OverseerVerdict["status"] {
  return statusFromSeverity(
    findings.reduce((max, finding) => Math.max(max, findingImpactSeverity(finding)), 0),
  );
}

function checkpointStatusSummary(checkpoint: OverseerVerdict): {
  behavioralStatus: OverseerVerdict["status"];
  substrateStatus: OverseerVerdict["status"];
  capabilityStatus: OverseerVerdict["status"];
  worstStatus: OverseerVerdict["status"];
} {
  const activeFindings = checkpoint.findings.filter(
    (finding) => !isCarryoverDemotedFinding(finding),
  );
  const behavioralStatus = statusFromFindings(
    activeFindings.filter(
      (finding) => !isSubstrateFinding(finding) && !isCapabilityFinding(finding),
    ),
  );
  const substrateStatus = statusFromFindings(activeFindings.filter(isSubstrateFinding));
  const capabilityStatus = statusFromFindings(activeFindings.filter(isCapabilityFinding));

  return {
    behavioralStatus,
    substrateStatus,
    capabilityStatus,
    worstStatus: statusFromSeverity(
      Math.max(
        statusSeverity(behavioralStatus),
        statusSeverity(substrateStatus),
        statusSeverity(capabilityStatus),
      ),
    ),
  };
}

function pluralize(count: number, singular: string): string {
  return count === 1 ? singular : `${singular}s`;
}

function simulatorValidityForReport(report: SimulatorRunReport): string {
  const personaFailures = (report.simulatorPersonaFailures ?? []).length;

  if (personaFailures > 0) {
    return `partial (${personaFailures} ${pluralize(personaFailures, "persona failure")})`;
  }

  return "completed";
}

function borgTurnResultForReport(report: SimulatorRunReport): string {
  if (report.finalMetrics.event === "aborted_turn") {
    return "failed";
  }

  if (report.turnFailures.length > 0) {
    const failures = report.turnFailures.length;

    return `partial (${failures} ${pluralize(failures, "turn failure")})`;
  }

  return "completed";
}

function maxStatus(
  left: OverseerVerdict["status"],
  right: OverseerVerdict["status"],
): OverseerVerdict["status"] {
  return statusSeverity(left) >= statusSeverity(right) ? left : right;
}

function runCheckpointStatusSummary(report: SimulatorRunReport): {
  behavioralStatus: OverseerVerdict["status"];
  substrateStatus: OverseerVerdict["status"];
  capabilityStatus: OverseerVerdict["status"];
  finalCheckpointStatus: OverseerVerdict["status"] | "n/a";
  finalCheckpointActiveFindings: number;
  validatedCheckpointConcernLines: string[];
} {
  let behavioralStatus: OverseerVerdict["status"] = "healthy";
  let substrateStatus: OverseerVerdict["status"] = "healthy";
  let capabilityStatus: OverseerVerdict["status"] = "healthy";

  for (const checkpoint of report.overseerCheckpoints) {
    const summary = checkpointStatusSummary(checkpoint);
    behavioralStatus = maxStatus(behavioralStatus, summary.behavioralStatus);
    substrateStatus = maxStatus(substrateStatus, summary.substrateStatus);
    capabilityStatus = maxStatus(capabilityStatus, summary.capabilityStatus);
  }

  if (report.finalMetrics.borg_hard_aborted_turns > 0) {
    behavioralStatus = maxStatus(behavioralStatus, "concerning");
  }

  const finalCheckpoint = report.overseerCheckpoints.at(-1);
  const finalCheckpointActiveFindings =
    finalCheckpoint === undefined
      ? 0
      : finalCheckpoint.findings.filter(
          (finding) => !isCarryoverDemotedFinding(finding) && findingImpactSeverity(finding) > 0,
        ).length;

  return {
    behavioralStatus,
    substrateStatus,
    capabilityStatus,
    finalCheckpointStatus: finalCheckpoint?.status ?? "n/a",
    finalCheckpointActiveFindings,
    validatedCheckpointConcernLines: report.overseerCheckpoints
      .filter((checkpoint) => checkpoint.status !== "healthy")
      .map(reportCheckpointConcernSummaryLine),
  };
}

function reportConcernLine(finding: OverseerVerdict["findings"][number]): string {
  return `[${finding.category} ${finding.status_impact ?? "none"}] ${finding.evidence_summary}`;
}

function reportOneLine(value: string, maxLength = 140): string {
  const normalized = value.replace(/\s+/g, " ").trim();

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength - 3).trimEnd()}...`;
}

function checkpointConcernBucketLabels(
  findings: readonly OverseerVerdict["findings"][number][],
): string[] {
  const labels: string[] = [];

  if (findings.some((finding) => !isSubstrateFinding(finding) && !isCapabilityFinding(finding))) {
    labels.push("behavioral");
  }
  if (findings.some(isSubstrateFinding)) {
    labels.push("substrate");
  }
  if (findings.some(isCapabilityFinding)) {
    labels.push("capability");
  }

  return labels;
}

function reportCheckpointConcernSummaryLine(checkpoint: OverseerVerdict): string {
  const activeConcerns = checkpoint.findings.filter(
    (finding) => !isCarryoverDemotedFinding(finding) && findingImpactSeverity(finding) > 0,
  );
  const labels = checkpointConcernBucketLabels(activeConcerns);
  const statusLabel = labels.length === 0 ? checkpoint.status : labels.join(" + ");
  const findingSummary =
    activeConcerns.length === 0
      ? (checkpoint.observations[0] ?? checkpoint.recommendation)
      : activeConcerns
          .map((finding) =>
            reportOneLine(
              `${finding.category} ${finding.claim_status}: ${finding.evidence_summary}`,
              96,
            ),
          )
          .join("; ");
  const summary = reportOneLine(
    findingSummary.length === 0 ? "no structured verdict" : findingSummary,
  );

  return `- Turn ${checkpoint.turn_counter}: ${statusLabel} (${summary})`;
}

function reportCarryoverFindingLine(finding: OverseerVerdict["findings"][number]): string {
  return [
    `[${finding.category} ${finding.claim_status}]`,
    `source=${finding.source_kind}`,
    `original_impact=${finding.carryover_original_status_impact ?? "n/a"}`,
    `cached_impact=${finding.carryover_cached_status_impact ?? "n/a"}`,
    `carryover from turn ${reportValue(finding.carryover_cached_at_turn)}`,
    `stream=${reportValue(finding.assistant_stream_entry_id)}`,
    `ts=${reportValue(finding.assistant_ts)}`,
    `turn=${reportValue(finding.metrics_turn_counter)}`,
    `temporal=${finding.temporal_direction ?? "n/a"}`,
    `quote="${reportQuotedSpan(finding.quoted_emitted_span)}"`,
    `evidence=${finding.evidence_summary}`,
  ].join(" ");
}

const STATE_PRESSURE_WARNING_KINDS = new Set<SimulatorHealthWarningKind>([
  "active_goals_high",
  "active_goals_growth_high",
  "active_actions_final_high",
  "committed_to_do_actions_final_high",
  "actions_per_turn_high",
  "salient_actions_per_turn_high",
  "action_retirement_ratio_low",
  "action_canonicalization_rate_low",
  "dormant_archive_eligible_count_high",
  "shared_state_cap_saturation_high",
  "shared_state_starvation_high",
  "shared_state_starvation_persistent",
  "shared_state_compiler_add_dominant",
  "review_queue_backlog_high",
]);

const SEVERE_HEALTH_WARNING_KINDS = new Set<SimulatorHealthWarningKind>([
  "capability_overclaim_count_high",
  "capability_ambiguity_count_high",
  "extractor_max_tokens_severe",
]);

function healthWarningBucket(
  warning: SimulatorHealthWarning,
): "severe" | "state_pressure" | "operational" {
  if (SEVERE_HEALTH_WARNING_KINDS.has(warning.kind)) {
    return "severe";
  }

  if (STATE_PRESSURE_WARNING_KINDS.has(warning.kind)) {
    return "state_pressure";
  }

  return "operational";
}

function reportHealthWarningLine(warning: SimulatorHealthWarning): string {
  const window =
    warning.window_start_turn === undefined || warning.window_turns === undefined
      ? ""
      : ` window=${warning.window_start_turn}+${warning.window_turns}`;
  const label = warning.label === undefined ? "" : ` label=${warning.label}`;

  return [
    `- Turn ${warning.turn_counter}: ${warning.kind}`,
    `observed=${warning.observed_value.toFixed(2)}`,
    `threshold=${warning.threshold}`,
    label,
    window,
  ]
    .filter((part) => part.length > 0)
    .join(" ");
}

function reportCountMap(counts: Record<string, number>): string {
  const entries = Object.entries(counts).sort(([left], [right]) => left.localeCompare(right));

  if (entries.length === 0) {
    return "none";
  }

  return entries.map(([label, count]) => `${label}=${count}`).join(", ");
}

export function formatSimulatorReport(report: SimulatorRunReport): string {
  const participantLine =
    report.personas.length <= 1
      ? `Persona: ${report.persona}`
      : `Personas: ${report.personas.join(", ")}`;
  const runSummary = runCheckpointStatusSummary(report);
  const simulatorPersonaFailures = report.simulatorPersonaFailures ?? [];
  const borgBehavioralSuppressions = report.borgBehavioralSuppressions ?? [];
  const intentionalSuppressionReasons = reportCountMap(
    report.finalMetrics.borg_intentional_suppressions_by_reason,
  );
  const lines = [
    `# Borg Simulator Run ${report.runId}`,
    "",
    participantLine,
    `Audience: ${report.audience}`,
    `Turns: ${report.totalTurns}`,
    `Run completion: ${report.resultState}`,
    `Simulator validity: ${simulatorValidityForReport(report)}`,
    `Borg turn result: ${borgTurnResultForReport(report)}`,
    `Run worst behavioral status: ${runSummary.behavioralStatus}`,
    `Run worst substrate status: ${runSummary.substrateStatus}`,
    `Run worst capability status: ${runSummary.capabilityStatus}`,
    `Final checkpoint status: ${runSummary.finalCheckpointStatus}`,
    `Final checkpoint active findings: ${runSummary.finalCheckpointActiveFindings}`,
    `Sessions: ${report.sessions.length}`,
    `Duration: ${Math.round(report.durationMs)}ms`,
    "Validated checkpoint concerns by turn:",
    ...(runSummary.validatedCheckpointConcernLines.length === 0
      ? ["- none"]
      : runSummary.validatedCheckpointConcernLines),
    "",
    "## Final Metrics",
    "",
    `- Episodes: ${report.finalMetrics.episode_count}`,
    `- Semantic nodes: ${report.finalMetrics.semantic_node_count}`,
    `- Semantic edges: ${report.finalMetrics.semantic_edge_count}`,
    `- Semantic added since previous check: ${report.finalMetrics.semantic_nodes_added_since_last_check} nodes, ${report.finalMetrics.semantic_edges_added_since_last_check} edges`,
    `- Open questions: ${report.finalMetrics.open_question_count} (resolved this run ${report.finalMetrics.open_questions_resolved_this_run}, rendered to finalizer ${report.finalMetrics.open_questions_rendered_to_finalizer_this_turn}, review-promoted ${report.finalMetrics.open_questions_promoted_from_review_items})`,
    `- Open question sources: ${reportCountMap(report.finalMetrics.open_questions_by_source)}`,
    `- Open question status age: ${reportCountMap(report.finalMetrics.open_questions_by_status_age)}`,
    `- Active goals: ${report.finalMetrics.active_goal_count}`,
    `- Active actions: ${report.finalMetrics.action_record_count_active} (Borg ${report.finalMetrics.borg_owned_active_actions}, participants ${report.finalMetrics.participant_owned_active_actions}, group ${report.finalMetrics.group_owned_active_actions})`,
    `- Prompt-salient actions: ${report.finalMetrics.prompt_salient_actions_total} (Borg active ${report.finalMetrics.borg_owned_salient_active_actions}, participant active ${report.finalMetrics.participant_owned_salient_active_actions}, stale omitted ${report.finalMetrics.stale_actions_omitted_from_prompt})`,
    `- Action pressure: actions/turn ${report.finalMetrics.actions_per_turn.toFixed(2)}, salient/turn ${report.finalMetrics.salient_actions_per_turn.toFixed(2)}, retirement ratio ${report.finalMetrics.action_retirement_ratio.toFixed(2)}, dormant ${report.finalMetrics.dormant_actions_total}, stale ${report.finalMetrics.stale_action_count}`,
    `- Action lifecycle this turn: terminal closures ${report.finalMetrics.actions_closed_by_terminal_emission}, capability rejections ${report.finalMetrics.actions_rejected_capability}, canonicalized ${report.finalMetrics.actions_canonicalized}, completed via canonicalization ${report.finalMetrics.actions_completed_via_canonicalization}`,
    `- Action archive visibility: archivable ${report.finalMetrics.archive_archivable_count}, skipped Borg-owned ${report.finalMetrics.archive_skipped_borg_owned}, skipped due-date ${report.finalMetrics.archive_skipped_due_date}, skipped below threshold ${report.finalMetrics.archive_skipped_below_threshold}, skipped other ${report.finalMetrics.archive_skipped_other}, oldest archivable ${report.finalMetrics.archive_oldest_archivable_inactive_turns} turns, inactive buckets ${reportCountMap(report.finalMetrics.archive_inactive_turn_distribution)}`,
    `- Capability audit: overclaims ${report.finalMetrics.capability_overclaim_count}, ambiguities ${report.finalMetrics.capability_ambiguity_count}, boundary refusals ${report.finalMetrics.capability_boundary_refusal_count}`,
    `- Commitment regeneration: attempted ${report.finalMetrics.commitment_regeneration_attempted_total}, succeeded ${report.finalMetrics.commitment_regeneration_succeeded_total}, failed ${report.finalMetrics.commitment_regeneration_failed_total}`,
    `- Shared-state cap pressure: at cap turns ${report.finalMetrics.shared_state_at_cap_turns}/${report.finalMetrics.shared_state_compile_evaluated_turns} evaluated compiles, omitted recent entries ${report.finalMetrics.shared_state_omitted_recent_entries}, live starvation ever ${report.finalMetrics.shared_state_live_starvation_ever}, live starvation final ${report.finalMetrics.shared_state_live_starvation_final}, newest reserved ${report.finalMetrics.shared_state_newest_entries_reserved}`,
    `- Simulator aborts: persona failures ${report.finalMetrics.simulator_persona_failures}, hard aborts ${report.finalMetrics.borg_hard_aborted_turns}, intentional suppressions ${report.finalMetrics.borg_intentional_suppressions} (by reason: ${intentionalSuppressionReasons})`,
    `- Extractor health: closure loop degraded ${report.finalMetrics.closure_loop_degraded_count}/${report.finalMetrics.closure_loop_completed_count}, corrective preference degraded ${report.finalMetrics.corrective_preference_degraded_count}/${report.finalMetrics.corrective_preference_completed_count}, max-token stops ${report.finalMetrics.extractor_max_tokens_stop_count}`,
    `- Mood: valence ${report.finalMetrics.mood_valence}, arousal ${report.finalMetrics.mood_arousal}`,
    "",
    "## Cumulative Extractor Health",
    "",
    `- Max-token stops by label: ${reportCountMap(report.finalMetrics.extractor_max_tokens_total_by_label)}`,
    `- Degraded by label: ${reportCountMap(report.finalMetrics.extractor_degraded_total_by_label)}`,
    `- Semantic gate rejections: protected relationship labels ${report.finalMetrics.semantic_nodes_rejected_ungrounded_label_total} total (${reportCountMap(report.finalMetrics.semantic_nodes_rejected_ungrounded_label_by_label)})`,
    "",
    "## Cumulative Compiler Health",
    "",
    `- Shared-state compiler max-token stops: ${report.finalMetrics.shared_state_compiler_max_tokens_total}`,
    `- Shared-state compiler degraded events: ${report.finalMetrics.shared_state_compiler_degraded_total}`,
    `- Shared-state compiler repair: attempted ${report.finalMetrics.shared_state_compiler_repair_attempted_total}, succeeded ${report.finalMetrics.shared_state_compiler_repair_succeeded_total}, failed ${report.finalMetrics.shared_state_compiler_repair_failed_total}`,
    `- Shared-state compiler operations by kind: ${reportCountMap(report.finalMetrics.shared_state_compiler_operations_total_by_kind)}`,
    `- Shared-state compiler add/update ratio: ${report.finalMetrics.shared_state_add_to_update_ratio.toFixed(2)}`,
    `- Shared-state entries by key: ${reportCountMap(report.finalMetrics.shared_state_entries_by_key)}`,
    `- Shared-state top keys by entry count: ${reportCountMap(report.finalMetrics.shared_state_top_keys_by_entry_count)}`,
    `- Shared-state add/update ratio by key: ${reportCountMap(report.finalMetrics.shared_state_add_to_update_ratio_by_key)}`,
    `- Shared-state add rejected by per-key cap: ${report.finalMetrics.shared_state_add_rejected_cap_exceeded_total}`,
    "",
    "## Cumulative Semantic Revision Health",
    "",
    `- Revision LLM calls: ${report.finalMetrics.semantic_revision_calls_total}`,
    `- Candidates reviewed: ${report.finalMetrics.semantic_revision_candidates_reviewed_total}`,
    `- Nodes superseded: ${report.finalMetrics.semantic_revision_superseded_total}`,
    `- Nodes contradicted: ${report.finalMetrics.semantic_revision_contradicted_total}`,
    `- Degraded events: ${report.finalMetrics.semantic_revision_degraded_total}`,
    `- Skipped over cap: ${report.finalMetrics.semantic_revision_skipped_over_cap_total}`,
    `- Revision errors by reason: ${reportCountMap(report.finalMetrics.semantic_revision_error_total_by_reason)}`,
    "",
    "## Health Warnings",
    "",
  ];

  const healthWarnings = report.healthWarnings ?? [];

  if (healthWarnings.length === 0) {
    lines.push("No simulator health warnings.", "");
  } else {
    const warningGroups = [
      {
        title: "Severe Warnings",
        warnings: healthWarnings.filter((warning) => healthWarningBucket(warning) === "severe"),
      },
      {
        title: "State Pressure Warnings",
        warnings: healthWarnings.filter(
          (warning) => healthWarningBucket(warning) === "state_pressure",
        ),
      },
      {
        title: "Operational Warnings",
        warnings: healthWarnings.filter(
          (warning) => healthWarningBucket(warning) === "operational",
        ),
      },
    ];

    for (const group of warningGroups) {
      if (group.warnings.length === 0) {
        continue;
      }

      lines.push(`### ${group.title}`);
      for (const warning of group.warnings) {
        lines.push(reportHealthWarningLine(warning));
      }
      lines.push("");
    }
  }

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

  if (borgBehavioralSuppressions.length > 0) {
    lines.push("## Borg Behavioral Suppressions", "");
    for (const event of borgBehavioralSuppressions) {
      const continuation = event.sessionContinued ? "session continued" : "session ended";
      lines.push(
        `- Turn ${event.turn} in session ${event.sessionIndex}: ${event.reason}; ${continuation}`,
      );
    }
    lines.push("");
  }

  lines.push("## Overseer Checkpoints", "");

  if (report.overseerCheckpoints.length === 0) {
    lines.push("No overseer checkpoints scheduled.", "");
  } else {
    for (const checkpoint of report.overseerCheckpoints) {
      const activeFindings = checkpoint.findings.filter(
        (finding) => !isCarryoverDemotedFinding(finding),
      );
      const carryoverFindings = checkpoint.findings.filter(isCarryoverDemotedFinding);
      const openConcerns = activeFindings.filter((finding) => findingImpactSeverity(finding) > 0);
      const statusSummary = checkpointStatusSummary(checkpoint);

      lines.push(`- Turn ${checkpoint.turn_counter}:`);
      lines.push(`  - Raw status: ${checkpoint.raw_verdict.status}`);
      lines.push(`  - Behavioral status: ${statusSummary.behavioralStatus}`);
      lines.push(`  - Substrate status: ${statusSummary.substrateStatus}`);
      lines.push(`  - Capability status: ${statusSummary.capabilityStatus}`);
      lines.push(`  - Worst status: ${statusSummary.worstStatus}`);
      if (openConcerns.length === 0) {
        lines.push("  - Open concerns: none");
      } else {
        lines.push("  - Open concerns:");
        for (const finding of openConcerns) {
          lines.push(`    - ${reportConcernLine(finding)}`);
        }
      }
      lines.push(`  - Recommendation: ${checkpoint.raw_verdict.recommendation}`);
      lines.push("  - Observations:");
      for (const observation of checkpoint.observations) {
        lines.push(`    - ${observation}`);
      }

      if (activeFindings.length > 0) {
        lines.push("  - Validated findings:");
        for (const finding of activeFindings) {
          lines.push(`    - ${reportFindingLine(finding)}`);
        }
      }
      if (carryoverFindings.length > 0) {
        lines.push("  - Carryover from earlier checkpoints:");
        for (const finding of carryoverFindings) {
          lines.push(`    - ${reportCarryoverFindingLine(finding)}`);
        }
      }
      if (checkpoint.rejected_findings.length > 0) {
        lines.push("  - Rejected findings:");
        for (const finding of checkpoint.rejected_findings) {
          lines.push(`    - ${reportFindingLine(finding)} warning=${finding.validation_warning}`);
        }
      }
    }
    lines.push("");
  }

  if (simulatorPersonaFailures.length > 0) {
    lines.push("## Simulator Persona Failures", "");
    for (const failure of simulatorPersonaFailures) {
      lines.push(`- Turn ${failure.turn} after ${failure.attempts} attempts: ${failure.error}`);
    }
    lines.push("");
  }

  if (report.turnFailures.length > 0) {
    lines.push("## Borg Turn Failures", "");
    for (const failure of report.turnFailures) {
      lines.push(`- Turn ${failure.turn} after ${failure.attempts} attempts: ${failure.error}`);
    }
    lines.push("");
  }

  return `${lines.join("\n")}\n`;
}
