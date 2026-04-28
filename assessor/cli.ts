import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";

import { cac } from "cac";

import { loadCredentials } from "../src/auth/claude-oauth.js";

import { formatAssessorReport, hasAssessorRunFailure } from "./report.js";
import { runScenarios } from "./runner.js";
import { getScenario, SCENARIOS } from "./scenarios/index.js";

type ParsedOptions = {
  scenario?: string;
  out?: string;
  keep?: boolean;
  mock?: boolean;
  real?: boolean;
  fail?: boolean;
  list?: boolean;
  maxTurns?: string | number;
  maxLlmCalls?: string | number;
};

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
): number | undefined {
  if (value === undefined) {
    return undefined;
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

function selectScenarios(name: string | undefined) {
  if (name === undefined || name.trim().length === 0) {
    return [...SCENARIOS];
  }

  const scenario = getScenario(name);

  if (scenario === undefined) {
    throw new Error(`Unknown scenario: ${name}`);
  }

  return [scenario];
}

async function main(): Promise<void> {
  const cli = cac("assess");

  cli
    .option("--scenario <name>", "Run a single scenario")
    .option("--out <path>", "Write markdown report to a file")
    .option("--keep", "Keep data dirs and trace files for inspection")
    .option("--real", "Use real Anthropic assessor calls")
    .option("--mock", "Use deterministic fake assessor and fake Borg LLM")
    .option("--fail", "Set a non-zero exit code for failed scenario verdicts", {
      default: true,
    })
    .option("--list", "List available scenarios")
    .option("--max-turns <n>", "Override per-scenario Borg turn cap")
    .option("--max-llm-calls <n>", "Override per-scenario assessor LLM call cap");

  const parsed = cli.parse(process.argv, { run: false });
  const options = parsed.options as ParsedOptions;

  if (options.list === true) {
    for (const scenario of SCENARIOS) {
      process.stdout.write(`${scenario.name}\t${scenario.description}\n`);
    }
    return;
  }

  const mock = selectMode(options);
  const report = await runScenarios({
    runId: randomUUID().slice(0, 8),
    scenarios: selectScenarios(options.scenario),
    keep: options.keep === true,
    mock,
    maxTurns: parsePositiveInteger(options.maxTurns, "--max-turns"),
    maxLlmCalls: parsePositiveInteger(options.maxLlmCalls, "--max-llm-calls"),
    env: process.env,
  });
  const markdown = formatAssessorReport(report);
  const failed = hasAssessorRunFailure(report);

  if (options.out === undefined || options.out.trim().length === 0) {
    process.stdout.write(markdown);
  } else {
    mkdirSync(dirname(options.out), { recursive: true });
    writeFileSync(options.out, markdown);
  }

  if (failed && options.fail !== false) {
    process.exitCode = 1;
  }
}

try {
  await main();
} catch (error) {
  process.stderr.write(
    `assessor failed: ${error instanceof Error ? error.message : String(error)}\n`,
  );
  process.exitCode = 1;
}
