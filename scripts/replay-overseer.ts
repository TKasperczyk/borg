// Replay the overseer against a COPY of a tenant bank and capture what the model
// actually emits per target: finish reason, output tokens, raw tool-argument
// length, parse success, flag count, and a head/tail sample. Dry run only.
//
// Usage (inside the memory sidecar image, cwd = app root):
//   ./node_modules/.bin/tsx scripts/replay-overseer.ts <root> <tenant> [maxChecks] [lookbackHours] [budget] [maxTokens] [process=overseer]
//
// The bank under <root>/<tenant> must be a scratch copy: Borg.open takes the
// bank lease and this process must never share a live bank with the sidecar.
import { appendFileSync, writeFileSync } from "node:fs";

import OpenAI from "openai";

import { createCachingEmbeddingClient } from "../src/embeddings/cache.js";
import { StallGuardEmbeddingClient } from "../src/embeddings/stall-guard.js";
import {
  BorgPool,
  OpenAICompatibleEmbeddingClient,
  OpenAICompatibleLLMClient,
} from "../src/index.js";
import type { OpenAIChatCompletionsClient } from "../src/llm/openai-compatible.js";
import { SystemClock } from "../src/util/clock.js";

const [root, tenant, maxChecksArg, lookbackArg, budgetArg, maxTokensArg, processArg] =
  process.argv.slice(2);
const offlineProcess = (processArg ?? "overseer") as "overseer" | "self-narrator" | "reflector";
if (!root || !tenant) {
  console.error(
    "usage: replay-overseer.ts <root> <tenant> [maxChecks] [lookbackHours] [budget] [maxTokens]",
  );
  process.exit(2);
}

const maxChecks = Number(maxChecksArg ?? 40);
const lookbackHours = Number(lookbackArg ?? 24 * 14);
const budget = Number(budgetArg ?? 1_000_000);
const maxTokensOverride = maxTokensArg === undefined ? undefined : Number(maxTokensArg);

process.env.BORG_OFFLINE_OVERSEER_MAX_CHECKS_PER_RUN = String(maxChecks);
process.env.BORG_OFFLINE_OVERSEER_LOOKBACK_HOURS = String(lookbackHours);
process.env.BORG_OFFLINE_OVERSEER_BUDGET = String(budget);

const apiKey = process.env.LLM_API_KEY ?? "";
const baseURL = process.env.KRATOS_BASE_URL ?? "https://inference.kratos.hdp.it.p4/v1";
const embeddingModel = process.env.EMBEDDING_MODEL ?? "scw/bge-m3";
const embeddingDims = Number(process.env.EMBEDDING_DIMS ?? 1024);
const requestTimeoutMs = Number(process.env.BORG_MEMORY_LLM_TIMEOUT_MS ?? 300_000);

const captureFile = `${root}/replay-overseer-${tenant}.jsonl`;
const rawFile = `${root}/replay-overseer-${tenant}.raw.jsonl`;
writeFileSync(captureFile, "");
writeFileSync(rawFile, "");

type Capture = {
  call: number;
  tool: string | null;
  max_tokens: unknown;
  prompt_chars: number;
  finish_reason: string | null;
  prompt_tokens: number | undefined;
  completion_tokens: number | undefined;
  tool_calls: number;
  args_chars: number;
  parse_ok: boolean;
  flags: number | null;
  flag_kinds: string[];
  patch_chars: number[];
  repeated_tail: boolean;
  head: string;
  tail: string;
  duration_ms: number;
};

let callIndex = 0;
const openai = new OpenAI({ apiKey, baseURL, timeout: requestTimeoutMs, maxRetries: 1 });
const inner = openai as unknown as OpenAIChatCompletionsClient;

function promptChars(params: Record<string, unknown>): number {
  const messages = params.messages as Array<{ content?: unknown }> | undefined;
  return (messages ?? []).reduce((sum, m) => {
    const c = m.content;
    return sum + (typeof c === "string" ? c.length : JSON.stringify(c ?? "").length);
  }, 0);
}

// A crude degenerate-loop indicator: does the last 400 chars of the arguments
// also appear earlier in the text?
function repeatedTail(text: string): boolean {
  if (text.length < 1_200) return false;
  const tail = text.slice(-400);
  return text.slice(0, -400).includes(tail);
}

const capturing: OpenAIChatCompletionsClient = {
  chat: {
    completions: {
      async create(params, options) {
        const index = (callIndex += 1);
        const started = Date.now();
        if (maxTokensOverride !== undefined) {
          params = { ...params, max_tokens: maxTokensOverride };
        }
        const response = await inner.chat.completions.create(params, options);
        const choice = (response as unknown as { choices?: Array<Record<string, unknown>> })
          .choices?.[0];
        const message = choice?.message as
          | { tool_calls?: Array<{ function?: { name?: string; arguments?: string } }> }
          | undefined;
        const toolCalls = message?.tool_calls ?? [];
        const args = toolCalls[0]?.function?.arguments ?? "";
        let parsed: unknown;
        let parseOk = false;
        try {
          parsed = JSON.parse(args);
          parseOk = true;
        } catch {
          parseOk = false;
        }
        const flags =
          parseOk &&
          typeof parsed === "object" &&
          parsed !== null &&
          Array.isArray((parsed as { flags?: unknown }).flags)
            ? ((parsed as { flags: unknown[] }).flags as Array<Record<string, unknown>>)
            : null;
        const usage = (
          response as unknown as { usage?: { prompt_tokens?: number; completion_tokens?: number } }
        ).usage;
        const tools = params.tools as Array<{ function?: { name?: string } }> | undefined;
        const capture: Capture = {
          call: index,
          tool: tools?.[0]?.function?.name ?? null,
          max_tokens: params.max_tokens ?? params.max_completion_tokens,
          prompt_chars: promptChars(params),
          finish_reason: (choice?.finish_reason as string | null) ?? null,
          prompt_tokens: usage?.prompt_tokens,
          completion_tokens: usage?.completion_tokens,
          tool_calls: toolCalls.length,
          args_chars: args.length,
          parse_ok: parseOk,
          flags: flags === null ? null : flags.length,
          flag_kinds: flags === null ? [] : flags.map((flag) => String(flag.kind)),
          patch_chars:
            flags === null ? [] : flags.map((flag) => JSON.stringify(flag.patch ?? null).length),
          repeated_tail: repeatedTail(args),
          head: args.slice(0, 240),
          tail: args.slice(-320),
          duration_ms: Date.now() - started,
        };
        appendFileSync(captureFile, `${JSON.stringify(capture)}\n`);
        appendFileSync(
          rawFile,
          `${JSON.stringify({ call: index, params: { ...params, tools: undefined }, arguments: args })}\n`,
        );
        console.log(
          `call#${index} finish=${capture.finish_reason} in=${capture.prompt_tokens} out=${capture.completion_tokens} args=${capture.args_chars}ch parse=${parseOk} flags=${capture.flags} kinds=${capture.flag_kinds.join(",")} patch=${capture.patch_chars.join(",")} loop=${capture.repeated_tail} ${capture.duration_ms}ms`,
        );
        return response;
      },
    },
  },
};

const llmClient = new OpenAICompatibleLLMClient({ client: capturing });
const embeddingClient = createCachingEmbeddingClient(
  new StallGuardEmbeddingClient(
    new OpenAICompatibleEmbeddingClient({
      client: openai,
      model: embeddingModel,
      dims: embeddingDims,
    }),
    { timeoutMs: 1_000, batchTimeoutMs: 20_000, maxRetries: 1 },
  ),
  { model: embeddingModel, dims: embeddingDims },
);

const pool = new BorgPool({
  root,
  maxOpen: 1,
  openOptions: {
    embeddingDimensions: embeddingDims,
    embeddingProfile: { model: embeddingModel, dimensions: embeddingDims },
    embeddingClient,
    llmClient,
    clock: new SystemClock(),
    liveExtraction: false,
    liveCommitmentExtraction: false,
  },
});

try {
  console.log(
    `replay overseer: root=${root} tenant=${tenant} maxChecks=${maxChecks} lookbackHours=${lookbackHours} budget=${budget} maxTokens=${maxTokensOverride ?? "default"}`,
  );
  const result = await pool.withTenant(
    tenant,
    (borg) => borg.dream({ processes: [offlineProcess], dryRun: true, budget }),
    { exclusive: true },
  );
  const overseer = result.results.find((entry) => entry.process === offlineProcess);
  console.log(
    JSON.stringify(
      {
        calls: callIndex,
        changes: overseer?.changes.length,
        tokens_used: overseer?.tokens_used,
        budget_exhausted: overseer?.budget_exhausted,
        candidate_stats: overseer?.candidate_stats,
        errors: overseer?.errors.map((error) => ({
          code: error.code,
          message: error.message.slice(0, 300),
        })),
      },
      null,
      1,
    ),
  );
  console.log(`captures: ${captureFile}\nraw: ${rawFile}`);
} finally {
  await pool.shutdown();
}
