import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";

import { cac } from "cac";

import { loadCredentials } from "../src/auth/claude-oauth.js";

import { tomPersona } from "./personas/tom.js";
import { findSimulatorScenario, scenarioPersonas } from "./scenarios/index.js";
import {
  formatSimulatorReport,
  PIPELINE_C_DOUBLE_PRIME_INCOMPATIBLE_SHADOW_MESSAGE,
  runSimulation,
} from "./runner.js";
import type { Persona } from "./types.js";

export type ParsedOptions = {
  persona?: string;
  personas?: string;
  scenario?: string;
  turns?: string | number;
  checkEvery?: string | number;
  maintenanceEvery?: string | number;
  out?: string;
  metricsOut?: string;
  traceOut?: string;
  keep?: boolean;
  mock?: boolean;
  real?: boolean;
  noPayloads?: boolean;
  shadowPostGenGuards?: boolean;
  pipelineCDoublePrime?: boolean;
};

type RawParsedOptions = ParsedOptions & {
  "--"?: string[];
  "pipelineC-doublePrime"?: boolean;
  payloads?: boolean;
};

const PERSONAS = new Map<string, Persona>([
  [tomPersona.key, tomPersona],
  ...scenarioPersonas().map((persona) => [persona.key, persona] as const),
]);

function hasAnthropicCredentials(env: NodeJS.ProcessEnv): boolean {
  if ((env.ANTHROPIC_API_KEY?.trim() ?? "").length > 0) {
    return true;
  }

  if ((env.ANTHROPIC_AUTH_TOKEN?.trim() ?? "").length > 0) {
    return true;
  }

  return loadCredentials({ env }) !== null;
}

function parsePositiveInteger(
  value: string | number | undefined,
  label: string,
  fallback: number,
): number {
  if (value === undefined) {
    return fallback;
  }

  const parsed = typeof value === "number" ? value : Number(value);

  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }

  return parsed;
}

function selectMode(options: ParsedOptions): boolean {
  if (options.mock === true && options.real === true) {
    throw new Error("--mock and --real cannot be used together");
  }

  if (options.mock === true) {
    return true;
  }

  if (options.real === true) {
    return false;
  }

  return !hasAnthropicCredentials(process.env);
}

function selectPersona(key: string | undefined): Persona {
  const resolved = key ?? "tom";
  const persona = PERSONAS.get(resolved);

  if (persona === undefined) {
    throw new Error(`Unknown persona: ${resolved}`);
  }

  return persona;
}

function parsePersonaKeys(value: string): string[] {
  return value
    .split(",")
    .map((key) => key.trim())
    .filter((key) => key.length > 0);
}

function assertUniquePersonaKeys(keys: readonly string[]): void {
  const seen = new Set<string>();

  for (const key of keys) {
    if (seen.has(key)) {
      throw new Error(`Duplicate persona key in --personas: ${key}`);
    }

    seen.add(key);
  }
}

function assertUniquePersonaDisplayNames(personas: readonly Persona[]): void {
  const seen = new Set<string>();

  for (const persona of personas) {
    if (seen.has(persona.displayName)) {
      throw new Error(`Duplicate persona display name in --personas: ${persona.displayName}`);
    }

    seen.add(persona.displayName);
  }
}

function assertParsedPersonaSelection(options: ParsedOptions): void {
  if (options.personas === undefined || options.personas.trim().length === 0) {
    return;
  }

  const keys = parsePersonaKeys(options.personas);
  assertUniquePersonaKeys(keys);

  const personas = keys.map((key) => PERSONAS.get(key));

  if (personas.every((persona): persona is Persona => persona !== undefined)) {
    assertUniquePersonaDisplayNames(personas);
  }
}

function selectPersonas(options: ParsedOptions): {
  personas: readonly Persona[];
  channelName?: string;
} {
  if (options.scenario !== undefined && options.scenario.trim().length > 0) {
    if (options.personas !== undefined && options.personas.trim().length > 0) {
      throw new Error("--scenario and --personas cannot be used together");
    }

    const scenario = findSimulatorScenario(options.scenario.trim());

    if (scenario === undefined) {
      throw new Error(`Unknown scenario: ${options.scenario}`);
    }

    return {
      personas: scenario.personas,
      channelName: scenario.channelName,
    };
  }

  if (options.personas !== undefined && options.personas.trim().length > 0) {
    const keys = parsePersonaKeys(options.personas);
    assertUniquePersonaKeys(keys);

    if (keys.length < 2 || keys.length > 4) {
      throw new Error("--personas must list 2 to 4 persona keys");
    }

    const personas = keys.map((key) => selectPersona(key));
    assertUniquePersonaDisplayNames(personas);

    return {
      personas,
    };
  }

  return {
    personas: [selectPersona(options.persona)],
  };
}

function createSimulatorCli() {
  const cli = cac("simulate");

  cli
    .option("--persona <key>", "Persona key to run", { default: "tom" })
    .option("--personas <keys>", "Comma-separated persona keys for a 2-4 person channel")
    .option("--scenario <key>", "Built-in simulator scenario key")
    .option("--turns <n>", "Number of continuous turns", { default: 1000 })
    .option("--check-every <n>", "Run overseer every N turns", { default: 250 })
    .option("--maintenance-every <n>", "Run light maintenance every N turns", { default: 10 })
    .option("--out <path>", "Write markdown report to a file")
    .option("--metrics-out <path>", "Write metrics JSONL to a file")
    .option("--trace-out <path>", "Write per-turn trace JSONL to a file (default: /tmp)")
    .option("--no-payloads", "Do not include full prompt/response payloads in turn traces")
    .option("--shadow-post-gen-guards", "Run post-generation guards in shadow mode")
    // Pipeline C-double-prime launch:
    // pnpm simulate --pipeline-c-double-prime --scenario <scenario> ...
    // C-double-prime means emission-tool finalizer on, commitment and
    // closure-pressure in enforce, and relational guard in shadow.
    .option("--pipeline-c-double-prime", "Run Pipeline C″ config for v27 launches")
    .option("--keep", "Keep Borg data dirs and trace files for inspection")
    .option("--real", "Use real Anthropic persona and overseer calls")
    .option("--mock", "Use deterministic fake persona, overseer, and Borg LLM");

  return cli;
}

function assertSimulatorFlagCompatibility(options: ParsedOptions): void {
  if (options.pipelineCDoublePrime === true && options.shadowPostGenGuards === true) {
    throw new Error(PIPELINE_C_DOUBLE_PRIME_INCOMPATIBLE_SHADOW_MESSAGE);
  }
}

export function parseSimulatorCliOptions(argv: string[] = process.argv): ParsedOptions {
  let parsed = createSimulatorCli().parse(argv, { run: false });
  let rawOptions = parsed.options as RawParsedOptions;
  const doubleDashOptions = rawOptions["--"];

  if (Array.isArray(doubleDashOptions) && doubleDashOptions.length > 0) {
    const separatorIndex = argv.indexOf("--");
    const reparsedArgv =
      separatorIndex === -1
        ? [...argv, ...doubleDashOptions]
        : [...argv.slice(0, separatorIndex), ...doubleDashOptions];

    parsed = createSimulatorCli().parse(reparsedArgv, { run: false });
    rawOptions = parsed.options as RawParsedOptions;
  }

  const { "pipelineC-doublePrime": pipelineCDoublePrimeRaw, payloads, ...restOptions } = rawOptions;
  const options: ParsedOptions = {
    ...restOptions,
    noPayloads: restOptions.noPayloads === true || payloads === false,
    pipelineCDoublePrime:
      restOptions.pipelineCDoublePrime === true || pipelineCDoublePrimeRaw === true,
  };

  assertSimulatorFlagCompatibility(options);
  assertParsedPersonaSelection(options);

  return options;
}

async function main(): Promise<void> {
  const options = parseSimulatorCliOptions(process.argv);
  const runId = randomUUID().slice(0, 8);
  const metricsOut = options.metricsOut?.trim();
  const metricsPath =
    metricsOut === undefined || metricsOut.length === 0
      ? join(tmpdir(), `borg-simulator-${runId}.metrics.jsonl`)
      : metricsOut;
  const traceOut = options.traceOut?.trim();
  const tracePath = traceOut === undefined || traceOut.length === 0 ? undefined : traceOut;
  const selection = selectPersonas(options);
  const primaryPersona = selection.personas[0] ?? tomPersona;

  if (tracePath !== undefined) {
    mkdirSync(dirname(tracePath), { recursive: true });
  }

  const report = await runSimulation({
    runId,
    persona: primaryPersona,
    ...(selection.personas.length <= 1 ? {} : { personas: selection.personas }),
    ...(selection.channelName === undefined ? {} : { channelName: selection.channelName }),
    totalTurns: parsePositiveInteger(options.turns, "--turns", 1000),
    checkEvery: parsePositiveInteger(options.checkEvery, "--check-every", 250),
    maintenanceEvery: parsePositiveInteger(options.maintenanceEvery, "--maintenance-every", 10),
    metricsPath,
    tracePath,
    shadowPostGenGuards: options.shadowPostGenGuards === true,
    pipelineCDoublePrime: options.pipelineCDoublePrime === true,
    includePayloads: options.noPayloads !== true,
    keep: options.keep === true,
    mock: selectMode(options),
    env: process.env,
  });
  const markdown = formatSimulatorReport(report);

  if (options.out === undefined || options.out.trim().length === 0) {
    process.stdout.write(markdown);
  } else {
    mkdirSync(dirname(options.out), { recursive: true });
    writeFileSync(options.out, markdown);
  }
}

function isSimulatorCliEntrypoint(argv: string[] = process.argv): boolean {
  const entrypoint = argv[1];

  return entrypoint !== undefined && import.meta.url === pathToFileURL(resolve(entrypoint)).href;
}

if (isSimulatorCliEntrypoint()) {
  try {
    await main();
  } catch (error) {
    process.stderr.write(
      `simulator failed: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  }
}
