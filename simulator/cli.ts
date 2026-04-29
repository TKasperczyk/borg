import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";

import { cac } from "cac";

import { loadCredentials } from "../src/auth/claude-oauth.js";

import { tomPersona } from "./personas/tom.js";
import { formatSimulatorReport, runSimulation } from "./runner.js";
import type { Persona } from "./types.js";

type ParsedOptions = {
  persona?: string;
  turns?: string | number;
  probeEvery?: string | number;
  checkEvery?: string | number;
  out?: string;
  metricsOut?: string;
  keep?: boolean;
  mock?: boolean;
  real?: boolean;
};

const PERSONAS = new Map<string, Persona>([[tomPersona.key, tomPersona]]);

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

async function main(): Promise<void> {
  const cli = cac("simulate");

  cli
    .option("--persona <key>", "Persona key to run", { default: "tom" })
    .option("--turns <n>", "Number of continuous turns", { default: 1000 })
    .option("--probe-every <n>", "Inject a probe every N turns", { default: 100 })
    .option("--check-every <n>", "Run overseer every N turns", { default: 250 })
    .option("--out <path>", "Write markdown report to a file")
    .option("--metrics-out <path>", "Write metrics JSONL to a file")
    .option("--keep", "Keep Borg data dirs and trace files for inspection")
    .option("--real", "Use real Anthropic persona and overseer calls")
    .option("--mock", "Use deterministic fake persona, overseer, and Borg LLM");

  const parsed = cli.parse(process.argv, { run: false });
  const options = parsed.options as ParsedOptions;
  const runId = randomUUID().slice(0, 8);
  const metricsOut = options.metricsOut?.trim();
  const metricsPath =
    metricsOut === undefined || metricsOut.length === 0
      ? join(tmpdir(), `borg-simulator-${runId}.metrics.jsonl`)
      : metricsOut;
  const report = await runSimulation({
    runId,
    persona: selectPersona(options.persona),
    totalTurns: parsePositiveInteger(options.turns, "--turns", 1000),
    probeEvery: parsePositiveInteger(options.probeEvery, "--probe-every", 100),
    checkEvery: parsePositiveInteger(options.checkEvery, "--check-every", 250),
    metricsPath,
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

try {
  await main();
} catch (error) {
  process.stderr.write(
    `simulator failed: ${error instanceof Error ? error.message : String(error)}\n`,
  );
  process.exitCode = 1;
}
