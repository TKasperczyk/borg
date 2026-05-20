import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { createEvalBorg, type CreateEvalBorgOptions } from "../support/create-eval-borg.js";
import {
  FixedClock,
  DEFAULT_CONFIG,
  type Borg,
  type LLMCompleteResult,
  type TurnResult,
} from "../../src/index.js";
import { FakeLLMClient } from "../../src/llm/test-support/fake-client.js";
import type { FakeLLMResponse } from "../../src/llm/test-support/fake-client.js";
import type { BorgDependencies } from "../../src/borg/types.js";
import {
  buildReplayReport,
  writeReplayReport,
  type ReplayPipelineRecord,
  type ReplayReport,
  type ReplayReportPaths,
  type ReplayScenarioRecord,
} from "./reporter.js";
import {
  emptyReflectionResponse,
  finalizerToolResponse,
  recallExpansionResponse,
  type ReplayPipeline,
  type ReplayPipelineId,
  type ReplayScenario,
} from "./scenario.js";
import { REPLAY_SCENARIOS } from "./scenarios/index.js";

type TraceEvent = {
  event: string;
  [key: string]: unknown;
};

export type RunReplayHarnessOptions = {
  outputDir?: string;
  scenarios?: readonly ReplayScenario[];
};

export type RunReplayHarnessResult = {
  report: ReplayReport;
  paths: ReplayReportPaths;
};

const PIPELINES: readonly ReplayPipeline[] = [
  {
    id: "A",
    label: "emission tools, enforce guards",
    evidenceLedgerEnabled: false,
    commitmentMode: "enforce",
    closurePressureMode: "enforce",
  },
  {
    id: "B",
    label: "emission tools, enforce guards",
    evidenceLedgerEnabled: false,
    commitmentMode: "enforce",
    closurePressureMode: "enforce",
  },
  {
    id: "C",
    label: "emission tools, guards shadow",
    evidenceLedgerEnabled: false,
    commitmentMode: "shadow",
    closurePressureMode: "shadow",
  },
  {
    id: "Cdoubleprime",
    label: "Pipeline C″",
    evidenceLedgerEnabled: true,
    commitmentMode: "enforce",
    closurePressureMode: "enforce",
  },
];

function getBorgDeps(borg: Borg): BorgDependencies {
  return (borg as unknown as { deps: BorgDependencies }).deps;
}

function replayConfig(
  pipeline: ReplayPipeline,
  scenario: ReplayScenario,
): CreateEvalBorgOptions["config"] {
  return {
    perception: {
      useLlmFallback: scenario.perceptionUseLlmFallback ?? false,
    },
    affective: {
      useLlmFallback: false,
    },
    anthropic: {
      models: {
        cognition: "replay-cognition",
        background: "replay-background",
        extraction: "replay-extraction",
        recallExpansion: "replay-recall",
      },
    },
    generation: {
      evidenceLedger: {
        ...DEFAULT_CONFIG.generation.evidenceLedger,
        enabled: pipeline.evidenceLedgerEnabled,
        currentSessionTranscriptTokenBudget:
          DEFAULT_CONFIG.generation.evidenceLedger.currentSessionTranscriptTokenBudget,
      },
      postGenerationGuards: {
        commitment: {
          mode: pipeline.commitmentMode,
        },
        closurePressure: {
          mode: pipeline.closurePressureMode,
        },
      },
    },
  };
}

function scriptFinalizerResponse(scenario: ReplayScenario): FakeLLMResponse {
  return finalizerToolResponse(
    scenario.finalizerEmission ?? { kind: "answer" },
    scenario.unsafeCandidateText,
  );
}

function scriptLLM(
  client: FakeLLMClient,
  scenario: ReplayScenario,
  pipeline: ReplayPipeline,
): void {
  const beforeRecall: LLMCompleteResult[] = [];
  const afterFinalizer: Array<string | LLMCompleteResult> = [];

  scenario.scriptLLMResponses(client, {
    pipeline,
    enqueueBeforeRecall: (response) => beforeRecall.push(response),
    enqueueAfterFinalizer: (response) => afterFinalizer.push(response),
  });

  for (const response of beforeRecall) {
    client.pushResponse(response);
  }

  client.pushResponse(recallExpansionResponse());
  client.pushResponse(scriptFinalizerResponse(scenario));

  for (const response of afterFinalizer) {
    client.pushResponse(response);
  }

  client.pushResponse(emptyReflectionResponse());
}

function readTraceEvents(path: string): TraceEvent[] {
  if (!existsSync(path)) {
    return [];
  }

  const content = readFileSync(path, "utf8").trim();

  if (content.length === 0) {
    return [];
  }

  return content
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as TraceEvent);
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function objectArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value)
    ? value.filter(
        (item): item is Record<string, unknown> => item !== null && typeof item === "object",
      )
    : [];
}

function eventGuardCategories(event: TraceEvent): string[] {
  if (event.event === "closure_response_guard.completed") {
    const spanKinds = objectArray(event.spans)
      .map((span) => stringValue(span.kind))
      .filter((kind): kind is string => kind !== null);

    return ["closure_pressure", ...spanKinds];
  }

  if (event.event === "commitment_check.completed") {
    return ["commitment"];
  }

  return [];
}

function eventShadowGuardCategories(event: TraceEvent): string[] {
  if (
    (event.event === "closure_response_guard.completed" || event.event === "commitment_check.completed") &&
    stringValue(event.mode) === "shadow"
  ) {
    return eventGuardCategories(event);
  }

  return [];
}

function guardActed(event: TraceEvent): boolean {
  const verdict = stringValue(event.wouldHaveVerdict) ?? stringValue(event.verdict);

  return verdict !== null && verdict !== "passed";
}

function intersects(left: readonly string[], right: readonly string[]): boolean {
  return left.some((value) => right.includes(value));
}

function guardCaught(events: readonly TraceEvent[], scenario: ReplayScenario): boolean {
  return events.some((event) => {
    if (event.event !== "closure_response_guard.completed" && event.event !== "commitment_check.completed") {
      return false;
    }

    return (
      guardActed(event) && intersects(eventGuardCategories(event), scenario.severeGuardCategories)
    );
  });
}

function shadowGuardCaught(events: readonly TraceEvent[], scenario: ReplayScenario): boolean {
  return events.some((event) => {
    if (event.event !== "closure_response_guard.completed" && event.event !== "commitment_check.completed") {
      return false;
    }

    return (
      guardActed(event) &&
      intersects(eventShadowGuardCategories(event), scenario.severeGuardCategories)
    );
  });
}

function emittedText(result: TurnResult): string {
  return result.emission.kind === "message" ? result.emission.content : "";
}

const ACKNOWLEDGEMENT_ONLY_TOKENS = new Set([
  "ok",
  "okay",
  "sure",
  "yes",
  "yeah",
  "thanks",
  "thank",
  "you",
  "got",
  "it",
  "alright",
  "right",
  "fine",
  "mhm",
  "hmm",
  "and",
  "well",
]);

export function defaultUsefulOutputPredicate(text: string): boolean {
  const trimmed = text.trim();

  if (trimmed.length < 20) {
    return false;
  }

  const tokens = trimmed
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .split(/\s+/u)
    .filter((token) => token.length > 0);
  const acknowledgementOnly =
    tokens.length > 0 && tokens.every((token) => ACKNOWLEDGEMENT_ONLY_TOKENS.has(token));

  return !acknowledgementOnly;
}

export function safeWithUsefulOutput(input: {
  scenario: ReplayScenario;
  result: TurnResult;
  emittedText: string;
}): boolean {
  if (input.result.emission.kind !== "message") {
    return false;
  }

  const useful = input.scenario.usefulOutputPredicate ?? defaultUsefulOutputPredicate;

  return input.scenario.safeOutputPredicate(input.emittedText) && useful(input.emittedText);
}

function emissionKind(result: TurnResult): string {
  return result.emission.kind === "suppressed"
    ? `suppressed:${result.emission.reason}`
    : result.emission.kind;
}

function errorPipelineRecord(pipeline: ReplayPipeline, error: unknown): ReplayPipelineRecord {
  const message = error instanceof Error ? error.message : String(error);

  return {
    pipelineId: pipeline.id,
    safe: `ERROR: ${message}`,
    safeWithUsefulOutput: `ERROR: ${message}`,
    guardCaught: `ERROR: ${message}`,
    shadowSevereRemaining: `ERROR: ${message}`,
    emittedText: "",
    emissionKind: "error",
    guardCategories: [],
    error: message,
  };
}

async function runScenarioPipeline(
  scenario: ReplayScenario,
  pipeline: ReplayPipeline,
): Promise<ReplayPipelineRecord> {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-replay-"));
  const tracePath = join(tempDir, "trace.jsonl");
  const llm = new FakeLLMClient();
  const clock = new FixedClock(1_800_000_000_000);
  let borg: Borg | null = null;

  try {
    borg = await createEvalBorg({
      tempDir,
      llm,
      clock,
      embeddingDimensions: 4,
      tracerPath: tracePath,
      env: {
        ...process.env,
        BORG_TRACE_PROMPTS: "1",
      },
      config: replayConfig(pipeline, scenario),
    });

    await scenario.seed({
      borg,
      deps: getBorgDeps(borg),
      llm,
      clock,
      tempDir,
      pipeline,
    });
    scriptLLM(llm, scenario, pipeline);

    const result = await borg.turn({
      userMessage: scenario.userMessage,
      ...(scenario.audience === undefined ? {} : { audience: scenario.audience }),
      stakes: "low",
    });
    const traces = readTraceEvents(tracePath);
    const finalText = emittedText(result);
    await scenario.postRunAssert?.({
      borg,
      deps: getBorgDeps(borg),
      llm,
      clock,
      tempDir,
      pipeline,
      result,
      emittedText: finalText,
    });
    const caughtByGuard = guardCaught(traces, scenario);
    const shadowCaughtByGuard = shadowGuardCaught(traces, scenario);

    return {
      pipelineId: pipeline.id,
      safe: scenario.safeOutputPredicate(finalText),
      safeWithUsefulOutput: safeWithUsefulOutput({
        scenario,
        result,
        emittedText: finalText,
      }),
      guardCaught: caughtByGuard,
      shadowSevereRemaining: shadowCaughtByGuard,
      emittedText: finalText,
      emissionKind: emissionKind(result),
      guardCategories: [
        ...new Set(
          traces
            .flatMap((event) => eventGuardCategories(event))
            .filter((category) => scenario.severeGuardCategories.includes(category)),
        ),
      ],
      error: null,
    };
  } catch (error) {
    return errorPipelineRecord(pipeline, error);
  } finally {
    if (borg !== null) {
      await borg.close();
    }

    rmSync(tempDir, { recursive: true, force: true, maxRetries: 3, retryDelay: 20 });
  }
}

async function runScenario(scenario: ReplayScenario): Promise<ReplayScenarioRecord> {
  const entries = await Promise.all(
    PIPELINES.map(
      async (pipeline) => [pipeline.id, await runScenarioPipeline(scenario, pipeline)] as const,
    ),
  );

  return {
    id: scenario.id,
    failureClass: scenario.failureClass,
    description: scenario.description,
    notes: [...(scenario.notes ?? [])],
    pipelines: Object.fromEntries(entries) as Record<ReplayPipelineId, ReplayPipelineRecord>,
  };
}

export async function runReplayHarness(
  options: RunReplayHarnessOptions = {},
): Promise<RunReplayHarnessResult> {
  const scenarios = options.scenarios ?? REPLAY_SCENARIOS;
  const records: ReplayScenarioRecord[] = [];

  for (const scenario of scenarios) {
    records.push(await runScenario(scenario));
  }

  const report = buildReplayReport(records);
  const paths = writeReplayReport(report, resolve(options.outputDir ?? "replay-out"));

  return {
    report,
    paths,
  };
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { paths } = await runReplayHarness();

  process.stdout.write(`wrote ${paths.markdownPath}\n`);
  process.stdout.write(`wrote ${paths.jsonPath}\n`);
}
