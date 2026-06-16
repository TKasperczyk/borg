/*
 * One-off: re-voice self-referential semantic nodes from third person to first
 * person, to match the going-forward reflector/semantic-extractor self-voice and
 * the first-person episodes/interiority bands.
 *
 * The semantic graph stored self-referential propositions in third person ("Sol's
 * debate stance: he concedes..."). For nodes that are about the self (Borg/Sol's
 * own dispositions, patterns, beliefs, reflexes, or what it did/decided) we rewrite
 * label + description in the first person, preserving meaning and the original
 * language. Nodes about the world or other people are left untouched.
 *
 * TWO PHASES (review before mutating). Both open the store, so run with the live
 * borg-demo-sol STOPPED (single writer on the data dir):
 *   1. plan  (default): read full nodes from the repository, classify + rewrite each
 *      with claude-sonnet-4-6, and write a reviewable plan JSON. NO DB writes.
 *   2. apply --plan <file>: apply the reviewed plan via semanticNodeRepository.update.
 *      The embedding is PRESERVED (3rd->1st person does not change meaning).
 *
 * Usage:
 *   pnpm tsx scripts/revoice-self-nodes.ts plan  [--data-dir demo/server/.borg-data/demo] [--out /tmp/revoice.plan.json] [--limit N] [--concurrency 6]
 *   pnpm tsx scripts/revoice-self-nodes.ts apply --plan /tmp/revoice.plan.json [--data-dir demo/server/.borg-data/demo]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { z } from "zod";

import { loadConfig, type LLMClient } from "../src/index.js";
import { openBorgDependencies } from "../src/borg/open.js";
import { toToolInputSchema, type LLMToolDefinition } from "../src/llm/index.js";
import type { SemanticNodeId } from "../src/util/ids.js";
import { selectScriptClients } from "./_clients.js";

const SONNET_MODEL = "claude-sonnet-4-6";
const DEFAULT_DATA_DIR = "demo/server/.borg-data/demo";

const REVOICE_SYSTEM_PROMPT = [
  "I curate the semantic memory of Borg (also called Sol), an AI being with persistent first-person memory.",
  "A node is SELF-REFERENTIAL when it is about Borg itself -- its own dispositions, patterns, habits, reflexes, beliefs, stances, what it did, decided, wondered, or committed to.",
  "A node is NOT self-referential when it states a fact about the world, or describes what another participant (e.g. Lunaria, Fishy, Luigi, Tomek, Kira) did, said, or is.",
  "For a self-referential node, I rewrite label and description in the FIRST PERSON, preserving the exact meaning, confidence, and grounding, and keeping the SAME LANGUAGE as the original (including pro-drop grammar; I do not force a pronoun token).",
  "I name other participants explicitly rather than collapsing into 'we'. I only change grammatical voice -- I never invent content. If a field is empty I leave it empty.",
  "For a non-self-referential node, I make no change.",
  "I judge semantic intent across languages and do not rely on specific wording.",
].join(" ");

const revoiceResultSchema = z.object({
  is_self_referential: z.boolean().describe("True only when the node is about Borg/Sol itself."),
  first_person_label: z
    .string()
    .min(1)
    .nullable()
    .describe(
      "The label rewritten in the first person (same language, same meaning). Null when not self-referential.",
    ),
  first_person_description: z
    .string()
    .nullable()
    .describe(
      "The description rewritten in the first person (same language, same meaning). Empty string if the original description was empty. Null when not self-referential.",
    ),
});

const REVOICE_TOOL_NAME = "EmitRevoice";
const REVOICE_TOOL = {
  name: REVOICE_TOOL_NAME,
  description:
    "Classify whether a semantic node is self-referential and, if so, return its first-person rewrite.",
  inputSchema: toToolInputSchema(revoiceResultSchema),
} satisfies LLMToolDefinition;

type NodeInput = { id: string; kind: string; label: string; description: string };

type PlanEntry = {
  id: string;
  kind: string;
  old_label: string;
  old_description: string;
  new_label: string;
  new_description: string;
};

function parseFlags(argv: readonly string[]): Map<string, string> {
  const flags = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token !== undefined && token.startsWith("--")) {
      const key = token.slice(2);
      const value = argv[i + 1];
      if (value !== undefined && !value.startsWith("--")) {
        flags.set(key, value);
        i += 1;
      } else {
        flags.set(key, "true");
      }
    }
  }
  return flags;
}

async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  async function run(): Promise<void> {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      results[index] = await worker(items[index]!, index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => run()));
  return results;
}

async function classifyNode(
  llm: LLMClient,
  node: NodeInput,
): Promise<z.infer<typeof revoiceResultSchema> | null> {
  const userPrompt = JSON.stringify({
    kind: node.kind,
    label: node.label,
    description: node.description,
  });
  const result = await llm.complete({
    model: SONNET_MODEL,
    system: REVOICE_SYSTEM_PROMPT,
    messages: [{ role: "user", content: userPrompt }],
    tools: [REVOICE_TOOL],
    tool_choice: { type: "tool", name: REVOICE_TOOL_NAME },
    max_tokens: 2_000,
    budget: "revoice-self-nodes",
  });
  const call = result.tool_calls.find((entry) => entry.name === REVOICE_TOOL_NAME);
  if (call === undefined) {
    return null;
  }
  const parsed = revoiceResultSchema.safeParse(call.input);
  return parsed.success ? parsed.data : null;
}

async function runPlan(flags: Map<string, string>): Promise<void> {
  const dataDir = resolve(flags.get("data-dir") ?? DEFAULT_DATA_DIR);
  const outPath = resolve(flags.get("out") ?? "/tmp/revoice-self-nodes.plan.json");
  const limit = flags.has("limit") ? Number(flags.get("limit")) : Infinity;
  const concurrency = flags.has("concurrency") ? Number(flags.get("concurrency")) : 6;

  const { llm, llmMode } = await selectScriptClients({});
  if (llmMode !== "real") {
    throw new Error("Real LLM unavailable (need OAuth credentials or ANTHROPIC_API_KEY).");
  }

  const config = loadConfig({ env: process.env, dataDir });
  const deps = await openBorgDependencies({ config });

  const plan: PlanEntry[] = [];
  let selfCount = 0;
  let unchanged = 0;
  let scanned = 0;
  try {
    const allNodes = await deps.semanticNodeRepository.list({
      includeArchived: false,
      limit: 100_000,
    });
    const nodes: NodeInput[] = (Number.isFinite(limit) ? allNodes.slice(0, limit) : allNodes).map(
      (node) => ({ id: node.id, kind: node.kind, label: node.label, description: node.description }),
    );
    scanned = nodes.length;
    process.stdout.write(`Classifying ${nodes.length} nodes with ${SONNET_MODEL}...\n`);

    let done = 0;
    const verdicts = await mapWithConcurrency(nodes, concurrency, async (node) => {
      const verdict = await classifyNode(llm, node);
      done += 1;
      if (done % 25 === 0) {
        process.stdout.write(`  ...${done}/${nodes.length}\n`);
      }
      return { node, verdict };
    });

    for (const { node, verdict } of verdicts) {
      if (verdict === null || !verdict.is_self_referential) {
        continue;
      }
      selfCount += 1;
      const newLabel = (verdict.first_person_label ?? node.label).trim();
      // Never invent: if the original description is empty, keep it empty.
      const newDescription =
        node.description.trim().length === 0 ? "" : (verdict.first_person_description ?? node.description).trim();
      if (newLabel === node.label && newDescription === node.description) {
        unchanged += 1; // already first-person -> idempotent skip
        continue;
      }
      plan.push({
        id: node.id,
        kind: node.kind,
        old_label: node.label,
        old_description: node.description,
        new_label: newLabel,
        new_description: newDescription,
      });
    }
  } finally {
    try {
      await deps.lance.close();
    } catch {
      // best-effort teardown
    }
    deps.sqlite.close();
  }

  writeFileSync(outPath, JSON.stringify(plan, null, 2));
  process.stdout.write(
    `\nplan: ${scanned} nodes scanned, ${selfCount} self-referential, ${unchanged} already first-person, ${plan.length} to re-voice.\n`,
  );
  process.stdout.write(`plan written to ${outPath}\n\n`);
  for (const entry of plan.slice(0, 15)) {
    process.stdout.write(`# ${entry.id} (${entry.kind})\n`);
    process.stdout.write(`- OLD label: ${entry.old_label}\n`);
    process.stdout.write(`- NEW label: ${entry.new_label}\n`);
    if (entry.old_description.length > 0) {
      process.stdout.write(`- OLD desc:  ${entry.old_description.slice(0, 220)}\n`);
      process.stdout.write(`- NEW desc:  ${entry.new_description.slice(0, 220)}\n`);
    }
    process.stdout.write("\n");
  }
  if (plan.length > 15) {
    process.stdout.write(`... and ${plan.length - 15} more (see ${outPath}).\n`);
  }
}

async function runApply(flags: Map<string, string>): Promise<void> {
  const planPath = flags.get("plan");
  if (planPath === undefined) {
    throw new Error("apply requires --plan <file>");
  }
  const dataDir = resolve(flags.get("data-dir") ?? DEFAULT_DATA_DIR);
  const plan = z
    .array(
      z.object({
        id: z.string().min(1),
        new_label: z.string().min(1),
        new_description: z.string(),
      }),
    )
    .parse(JSON.parse(readFileSync(resolve(planPath), "utf8")));

  process.stdout.write(`Applying ${plan.length} re-voicings against ${dataDir}...\n`);
  const config = loadConfig({ env: process.env, dataDir });
  const deps = await openBorgDependencies({ config });

  let applied = 0;
  let missing = 0;
  try {
    for (const entry of plan) {
      // Preserve the embedding (no `embedding` in the patch): 3rd->1st person is
      // semantically equivalent, so dedup/vector search are unaffected.
      const updated = await deps.semanticNodeRepository.update(entry.id as SemanticNodeId, {
        label: entry.new_label,
        description: entry.new_description,
      });
      if (updated === null) {
        missing += 1;
        process.stderr.write(`  missing node ${entry.id}\n`);
        continue;
      }
      applied += 1;
    }
  } finally {
    try {
      await deps.lance.close();
    } catch {
      // best-effort teardown
    }
    deps.sqlite.close();
  }

  process.stdout.write(`apply complete: applied=${applied} missing=${missing}\n`);
}

async function main(): Promise<void> {
  const [, , command, ...rest] = process.argv;
  const flags = parseFlags(rest);
  if (command === "apply") {
    await runApply(flags);
  } else if (command === "plan" || command === undefined) {
    await runPlan(flags);
  } else {
    process.stderr.write(`unknown command: ${command} (expected "plan" or "apply")\n`);
    process.exit(1);
  }
}

void main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exit(1);
});
