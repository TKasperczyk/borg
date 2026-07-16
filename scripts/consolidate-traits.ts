/*
 * One-off: consolidate near-duplicate rows in the self traits table.
 *
 * Run with the live Borg/demo writer STOPPED (single writer on the data dir) and
 * take a verified backup first using the WORKFLOW.md recipe. Dry-run/propose is
 * the default and performs no DB writes; only apply mutates.
 *
 * TWO PHASES (review before mutating):
 *   1. propose (default): read traits plus reinforcement/contradiction event
 *      summaries, ask claude-sonnet-4-6 to cluster labels that denote the same
 *      trait, and write a reviewable plan JSON. NO DB writes.
 *   2. apply --plan <file>: revalidate row versions and apply the reviewed plan
 *      via TraitsRepository.mergeInto.
 *
 * Usage:
 *   pnpm tsx scripts/consolidate-traits.ts propose [--data-dir demo/server/.borg-data/demo] [--out /tmp/consolidate-traits.plan.json]
 *   pnpm tsx scripts/consolidate-traits.ts apply --plan /tmp/consolidate-traits.plan.json [--data-dir demo/server/.borg-data/demo]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import { z } from "zod";

import { loadConfig, type LLMClient, type TraitRecord } from "../src/index.js";
import { openBorgDependencies } from "../src/borg/open.js";
import { toToolInputSchema, type LLMToolDefinition } from "../src/llm/index.js";
import { type TraitsRepository } from "../src/memory/self/repository.js";
import { traitIdHelpers, type EpisodeId, type TraitId } from "../src/util/ids.js";
import { selectScriptClients } from "./_clients.js";

const SONNET_MODEL = "claude-sonnet-4-6";
const DEFAULT_DATA_DIR = "demo/server/.borg-data/demo";
const DEFAULT_PLAN_PATH = "/tmp/consolidate-traits.plan.json";
const PLAN_KIND = "trait_consolidation_plan";
const PLAN_VERSION = 1;
const APPLY_PROVENANCE = { kind: "offline", process: "trait-consolidation" } as const;

const TRAIT_CONSOLIDATION_SYSTEM_PROMPT = [
  "I maintain Borg/Sol's self-trait vocabulary.",
  "I cluster only existing trait rows whose labels denote the SAME trait or disposition.",
  "I do not cluster labels that are merely related, broader/narrower, causal, antonymic, or frequently co-occurring.",
  "I choose an existing member as the canonical row; I do not invent a new label.",
  "If a cluster contains an established trait, I choose an established member as canonical. If several are established, I choose one established member and keep the others in member_ids.",
  "I return only clusters with two or more members. Rows that should remain independent are omitted.",
].join(" ");

const llmClusterSchema = z.object({
  canonical_id: z.string().min(1),
  canonical_label: z.string().min(1),
  member_ids: z.array(z.string().min(1)).min(2),
  llm_rationale: z.string().min(1),
});

const llmResultSchema = z.object({
  clusters: z.array(llmClusterSchema),
});

const TOOL_NAME = "EmitTraitConsolidationPlan";
const TOOL = {
  name: TOOL_NAME,
  description: "Emit duplicate trait clusters with an existing canonical trait per cluster.",
  inputSchema: toToolInputSchema(llmResultSchema),
} satisfies LLMToolDefinition;

const planMemberSchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  state: z.enum(["candidate", "established"]),
  record_version: z.number().int().positive(),
});

const distinctEpisodeCountSchema = z.object({
  id: z.string().min(1),
  count: z.number().int().min(0),
});

const planClusterSchema = z.object({
  canonicalId: z.string().min(1),
  canonicalLabel: z.string().min(1),
  members: z.array(planMemberSchema).min(2),
  sourceIds: z.array(z.string().min(1)).min(1),
  llm_rationale: z.string().min(1),
  beforeDistinctEpisodeCounts: z.array(distinctEpisodeCountSchema).min(2),
  afterDistinctEpisodeCount: z.number().int().min(0),
});

const planSchema = z.object({
  kind: z.literal(PLAN_KIND),
  version: z.literal(PLAN_VERSION),
  model: z.string().min(1),
  dataDir: z.string().min(1),
  generatedAt: z.string().min(1),
  clusters: z.array(planClusterSchema),
});

type LlmCluster = z.infer<typeof llmClusterSchema>;
type Plan = z.infer<typeof planSchema>;
type PlanCluster = z.infer<typeof planClusterSchema>;

type TraitEventSummary = {
  reinforcementEvents: number;
  episodeBackedReinforcementEvents: number;
  distinctEpisodeIds: EpisodeId[];
  latestReinforcementAt: number | null;
  contradictionEvents: number;
  latestContradictionAt: number | null;
};

type TraitForPrompt = {
  id: TraitId;
  label: string;
  state: TraitRecord["state"];
  record_version: number;
  strength: number;
  support_count: number;
  contradiction_count: number;
  evidence_episode_ids: EpisodeId[];
  reinforcement_summary: {
    total_events: number;
    episode_backed_events: number;
    distinct_episode_count: number;
    latest_ts: number | null;
  };
  contradiction_summary: {
    total_events: number;
    latest_ts: number | null;
  };
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

function parseTraitId(value: string, label: string): TraitId {
  try {
    return traitIdHelpers.parse(value);
  } catch (error) {
    throw new Error(`Invalid ${label}: ${value}`, { cause: error });
  }
}

function requireRecordVersion(trait: TraitRecord): number {
  if (trait.record_version === undefined) {
    throw new Error(`Trait ${trait.id} has no record_version`);
  }
  return trait.record_version;
}

function ensureUniqueIds(ids: readonly string[], label: string): void {
  const seen = new Set<string>();
  for (const id of ids) {
    if (seen.has(id)) {
      throw new Error(`${label} contains duplicate id ${id}`);
    }
    seen.add(id);
  }
}

function sameIdSet(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  const remaining = new Set(left);
  for (const id of right) {
    if (!remaining.delete(id)) {
      return false;
    }
  }
  return remaining.size === 0;
}

function eventSummary(
  trait: TraitRecord,
  reinforcementEvents: ReturnType<TraitsRepository["listReinforcementEvents"]>,
  contradictionEvents: ReturnType<TraitsRepository["listContradictionEvents"]>,
): TraitEventSummary {
  const distinctEpisodeIds = new Set<EpisodeId>();
  let episodeBackedReinforcementEvents = 0;

  for (const event of reinforcementEvents) {
    if (event.provenance.kind !== "episodes") {
      continue;
    }
    episodeBackedReinforcementEvents += 1;
    for (const episodeId of event.provenance.episode_ids) {
      distinctEpisodeIds.add(episodeId);
    }
  }

  return {
    reinforcementEvents: reinforcementEvents.length,
    episodeBackedReinforcementEvents,
    distinctEpisodeIds: [...distinctEpisodeIds],
    latestReinforcementAt: trait.last_tested_at,
    contradictionEvents: contradictionEvents.length,
    latestContradictionAt: trait.last_contradicted_at,
  };
}

type ReadOnlyTraitRepositoryLike = Pick<
  TraitsRepository,
  "get" | "list" | "listReinforcementEvents" | "listContradictionEvents"
>;

type MutatingTraitRepositoryLike = ReadOnlyTraitRepositoryLike &
  Pick<TraitsRepository, "mergeInto">;

function buildTraitInputs(repository: ReadOnlyTraitRepositoryLike): {
  promptTraits: TraitForPrompt[];
  traitsById: Map<TraitId, TraitRecord>;
  summariesById: Map<TraitId, TraitEventSummary>;
} {
  const traits = repository.list();
  const traitsById = new Map<TraitId, TraitRecord>();
  const summariesById = new Map<TraitId, TraitEventSummary>();
  const promptTraits: TraitForPrompt[] = [];

  for (const trait of traits) {
    const recordVersion = requireRecordVersion(trait);
    const reinforcementEvents = repository.listReinforcementEvents(trait.id);
    const contradictionEvents = repository.listContradictionEvents(trait.id);
    const summary = eventSummary(trait, reinforcementEvents, contradictionEvents);
    traitsById.set(trait.id, trait);
    summariesById.set(trait.id, summary);
    promptTraits.push({
      id: trait.id,
      label: trait.label,
      state: trait.state,
      record_version: recordVersion,
      strength: trait.strength,
      support_count: trait.support_count,
      contradiction_count: trait.contradiction_count,
      evidence_episode_ids: trait.evidence_episode_ids,
      reinforcement_summary: {
        total_events: summary.reinforcementEvents,
        episode_backed_events: summary.episodeBackedReinforcementEvents,
        distinct_episode_count: summary.distinctEpisodeIds.length,
        latest_ts: summary.latestReinforcementAt,
      },
      contradiction_summary: {
        total_events: summary.contradictionEvents,
        latest_ts: summary.latestContradictionAt,
      },
    });
  }

  return { promptTraits, traitsById, summariesById };
}

async function proposeClusters(
  llm: LLMClient,
  traits: readonly TraitForPrompt[],
): Promise<LlmCluster[]> {
  const result = await llm.complete({
    model: SONNET_MODEL,
    system: TRAIT_CONSOLIDATION_SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: JSON.stringify({
          instruction:
            "Cluster existing trait ids whose labels denote the same trait. Use only existing ids. Omit independent traits.",
          traits,
        }),
      },
    ],
    tools: [TOOL],
    tool_choice: { type: "tool", name: TOOL_NAME },
    max_tokens: 4_000,
    budget: "trait-consolidation",
  });

  const call = result.tool_calls.find((entry) => entry.name === TOOL_NAME);
  if (call === undefined) {
    throw new Error("Trait consolidation LLM did not call the required tool");
  }

  return llmResultSchema.parse(call.input).clusters;
}

function buildPlanCluster(
  cluster: LlmCluster,
  traitsById: Map<TraitId, TraitRecord>,
  summariesById: Map<TraitId, TraitEventSummary>,
): PlanCluster {
  const rawMemberIds = cluster.member_ids.map((id) => parseTraitId(id, "member id"));
  ensureUniqueIds(rawMemberIds, "LLM cluster member_ids");

  const llmCanonicalId = parseTraitId(cluster.canonical_id, "canonical_id");
  if (!rawMemberIds.includes(llmCanonicalId)) {
    throw new Error(`LLM canonical_id ${llmCanonicalId} is not present in member_ids`);
  }

  const members = rawMemberIds.map((id) => {
    const trait = traitsById.get(id);
    if (trait === undefined) {
      throw new Error(`LLM cluster references unknown trait id ${id}`);
    }
    return trait;
  });

  let canonicalId = llmCanonicalId;
  const llmCanonical = traitsById.get(llmCanonicalId);
  if (llmCanonical === undefined) {
    throw new Error(`LLM canonical_id references unknown trait id ${llmCanonicalId}`);
  }

  const establishedMembers = members.filter((member) => member.state === "established");
  if (establishedMembers.length > 0 && llmCanonical.state !== "established") {
    canonicalId = establishedMembers[0]!.id;
  }

  const canonical = traitsById.get(canonicalId);
  if (canonical === undefined) {
    throw new Error(`Canonical trait id disappeared while building plan: ${canonicalId}`);
  }

  const sourceIds = rawMemberIds.filter((id) => id !== canonicalId);
  const mergedEpisodeIds = new Set<EpisodeId>();
  const beforeDistinctEpisodeCounts = rawMemberIds.map((id) => {
    const summary = summariesById.get(id);
    if (summary === undefined) {
      throw new Error(`Missing event summary for trait id ${id}`);
    }
    for (const episodeId of summary.distinctEpisodeIds) {
      mergedEpisodeIds.add(episodeId);
    }
    return {
      id,
      count: summary.distinctEpisodeIds.length,
    };
  });

  return {
    canonicalId,
    canonicalLabel: canonical.label,
    members: members.map((member) => ({
      id: member.id,
      label: member.label,
      state: member.state,
      record_version: requireRecordVersion(member),
    })),
    sourceIds,
    llm_rationale: cluster.llm_rationale,
    beforeDistinctEpisodeCounts,
    afterDistinctEpisodeCount: mergedEpisodeIds.size,
  };
}

function buildPlan(input: {
  clusters: readonly LlmCluster[];
  traitsById: Map<TraitId, TraitRecord>;
  summariesById: Map<TraitId, TraitEventSummary>;
  dataDir: string;
  generatedAt: string;
}): Plan {
  const seenMemberIds = new Set<string>();
  const planClusters = input.clusters.map((cluster) =>
    buildPlanCluster(cluster, input.traitsById, input.summariesById),
  );

  for (const cluster of planClusters) {
    for (const member of cluster.members) {
      if (seenMemberIds.has(member.id)) {
        throw new Error(`Trait id ${member.id} appears in more than one cluster`);
      }
      seenMemberIds.add(member.id);
    }
  }

  return planSchema.parse({
    kind: PLAN_KIND,
    version: PLAN_VERSION,
    model: SONNET_MODEL,
    dataDir: input.dataDir,
    generatedAt: input.generatedAt,
    clusters: planClusters,
  });
}

function validatePlanStructure(plan: Plan): void {
  const seenMembers = new Set<string>();

  for (const cluster of plan.clusters) {
    const memberIds = cluster.members.map((member) => member.id);
    ensureUniqueIds(memberIds, "plan cluster members");
    ensureUniqueIds(cluster.sourceIds, "plan cluster sourceIds");

    if (!memberIds.includes(cluster.canonicalId)) {
      throw new Error(`Plan canonicalId ${cluster.canonicalId} is not present in members`);
    }

    const expectedSourceIds = memberIds.filter((id) => id !== cluster.canonicalId);
    if (!sameIdSet(cluster.sourceIds, expectedSourceIds)) {
      throw new Error(
        `Plan sourceIds do not match members minus canonical for ${cluster.canonicalId}`,
      );
    }

    for (const memberId of memberIds) {
      parseTraitId(memberId, "plan member id");
      if (seenMembers.has(memberId)) {
        throw new Error(`Plan trait id ${memberId} appears in more than one cluster`);
      }
      seenMembers.add(memberId);
    }
    parseTraitId(cluster.canonicalId, "plan canonicalId");
    for (const sourceId of cluster.sourceIds) {
      parseTraitId(sourceId, "plan sourceId");
    }
  }
}

function validatePlanAgainstDb(plan: Plan, repository: ReadOnlyTraitRepositoryLike): void {
  validatePlanStructure(plan);
  const currentById = new Map(repository.list().map((trait) => [trait.id, trait]));

  for (const cluster of plan.clusters) {
    const canonical = currentById.get(parseTraitId(cluster.canonicalId, "plan canonicalId"));
    if (canonical === undefined) {
      throw new Error(`Canonical trait ${cluster.canonicalId} no longer exists`);
    }

    for (const member of cluster.members) {
      const memberId = parseTraitId(member.id, "plan member id");
      const current = currentById.get(memberId);
      if (current === undefined) {
        throw new Error(`Trait ${member.id} no longer exists`);
      }
      if (requireRecordVersion(current) !== member.record_version) {
        throw new Error(
          `Trait ${member.id} record_version changed since propose: expected ${member.record_version}, got ${requireRecordVersion(current)}`,
        );
      }
      if (current.state !== member.state) {
        throw new Error(
          `Trait ${member.id} state changed since propose: expected ${member.state}, got ${current.state}`,
        );
      }
    }

    for (const sourceIdRaw of cluster.sourceIds) {
      const source = currentById.get(parseTraitId(sourceIdRaw, "plan sourceId"));
      if (source === undefined) {
        throw new Error(`Source trait ${sourceIdRaw} no longer exists`);
      }
      if (source.state === "established" && canonical.state !== "established") {
        throw new Error(
          `Plan would merge established source ${source.id} into candidate canonical ${canonical.id}`,
        );
      }
    }
  }
}

async function runPropose(flags: Map<string, string>, generatedAt: string): Promise<void> {
  const dataDir = resolve(flags.get("data-dir") ?? DEFAULT_DATA_DIR);
  const outPath = resolve(flags.get("out") ?? DEFAULT_PLAN_PATH);

  const { llm, llmMode } = await selectScriptClients({});
  if (llmMode !== "real") {
    throw new Error("Real LLM unavailable (need OAuth credentials or ANTHROPIC_API_KEY).");
  }

  const config = loadConfig({ env: process.env, dataDir });
  const deps = await openBorgDependencies({ config });

  let plan: Plan | null = null;
  let traitCount = 0;
  try {
    const { promptTraits, traitsById, summariesById } = buildTraitInputs(deps.traitsRepository);
    traitCount = promptTraits.length;
    process.stdout.write(`Clustering ${traitCount} traits with ${SONNET_MODEL}...\n`);
    const clusters = await proposeClusters(llm, promptTraits);
    plan = buildPlan({
      clusters,
      traitsById,
      summariesById,
      dataDir,
      generatedAt,
    });
  } finally {
    try {
      await deps.lance.close();
    } catch {
      // best-effort teardown
    }
    deps.sqlite.close();
  }

  if (plan === null) {
    throw new Error("Trait consolidation proposal did not produce a plan");
  }

  writeFileSync(outPath, JSON.stringify(plan, null, 2));
  const sourceCount = plan.clusters.reduce((sum, cluster) => sum + cluster.sourceIds.length, 0);
  process.stdout.write(
    `\npropose: ${traitCount} traits scanned, ${plan.clusters.length} cluster(s), ${sourceCount} source trait(s) to merge.\n`,
  );
  process.stdout.write(`plan written to ${outPath}\n\n`);

  for (const cluster of plan.clusters.slice(0, 12)) {
    process.stdout.write(`# canonical ${cluster.canonicalId}: ${cluster.canonicalLabel}\n`);
    process.stdout.write(`- sources: ${cluster.sourceIds.join(", ")}\n`);
    process.stdout.write(
      `- distinct episodes: ${cluster.beforeDistinctEpisodeCounts.map((entry) => `${entry.id}=${entry.count}`).join(", ")} -> ${cluster.afterDistinctEpisodeCount}\n`,
    );
    process.stdout.write(`- rationale: ${cluster.llm_rationale}\n\n`);
  }
  if (plan.clusters.length > 12) {
    process.stdout.write(`... and ${plan.clusters.length - 12} more (see ${outPath}).\n`);
  }
}

function readPlan(flags: Map<string, string>): { plan: Plan; dataDir: string } {
  const planPath = flags.get("plan");
  if (planPath === undefined) {
    throw new Error("apply requires --plan <file>");
  }

  const plan = planSchema.parse(JSON.parse(readFileSync(resolve(planPath), "utf8")));
  const dataDir = resolve(flags.get("data-dir") ?? plan.dataDir);

  if (flags.has("data-dir") && resolve(plan.dataDir) !== dataDir) {
    throw new Error(`--data-dir ${dataDir} does not match plan dataDir ${resolve(plan.dataDir)}`);
  }

  return { plan, dataDir };
}

async function runApply(flags: Map<string, string>): Promise<void> {
  const { plan, dataDir } = readPlan(flags);

  process.stdout.write(
    `Applying ${plan.clusters.length} trait consolidation cluster(s) against ${dataDir}...\n`,
  );
  const config = loadConfig({ env: process.env, dataDir });
  const deps = await openBorgDependencies({ config });

  let clustersMerged = 0;
  let traitsRemoved = 0;
  let traitsPromoted = 0;
  const promotedCanonicalIds = new Set<string>();

  try {
    const traitsRepository: MutatingTraitRepositoryLike = deps.traitsRepository;

    deps.sqlite.transaction(() => {
      validatePlanAgainstDb(plan, traitsRepository);

      for (const cluster of plan.clusters) {
        const canonicalId = parseTraitId(cluster.canonicalId, "plan canonicalId");
        clustersMerged += 1;

        for (const sourceIdRaw of cluster.sourceIds) {
          const sourceId = parseTraitId(sourceIdRaw, "plan sourceId");
          const sourceMember = cluster.members.find((member) => member.id === sourceIdRaw);
          if (sourceMember === undefined) {
            throw new Error(`Source ${sourceIdRaw} missing from cluster members`);
          }

          const canonical = traitsRepository.get(canonicalId);
          const source = traitsRepository.get(sourceId);
          if (canonical === null) {
            throw new Error(`Canonical trait ${canonicalId} no longer exists`);
          }
          if (source === null) {
            throw new Error(`Source trait ${sourceId} no longer exists`);
          }
          if (requireRecordVersion(source) !== sourceMember.record_version) {
            throw new Error(
              `Source trait ${sourceId} record_version changed during apply: expected ${sourceMember.record_version}, got ${requireRecordVersion(source)}`,
            );
          }
          if (source.state === "established" && canonical.state !== "established") {
            throw new Error(
              `Refusing established source ${sourceId} into candidate canonical ${canonicalId}`,
            );
          }

          process.stdout.write(
            `  merge ${source.id} (${source.label}) -> ${canonical.id} (${canonical.label})\n`,
          );
          const merged = traitsRepository.mergeInto({
            sourceId: source.id,
            canonicalId: canonical.id,
            expectedSourceVersion: sourceMember.record_version,
            expectedCanonicalVersion: requireRecordVersion(canonical),
            provenance: APPLY_PROVENANCE,
          });

          traitsRemoved += 1;
          if (
            canonical.state !== "established" &&
            merged.state === "established" &&
            !promotedCanonicalIds.has(canonical.id)
          ) {
            promotedCanonicalIds.add(canonical.id);
            traitsPromoted += 1;
          }
        }
      }
    })();
  } finally {
    try {
      await deps.lance.close();
    } catch {
      // best-effort teardown
    }
    deps.sqlite.close();
  }

  process.stdout.write(
    `apply complete: clusters_merged=${clustersMerged} traits_removed=${traitsRemoved} traits_promoted=${traitsPromoted}\n`,
  );
}

async function main(generatedAt: string): Promise<void> {
  const [, , maybeCommand, ...tail] = process.argv;
  const command =
    maybeCommand === undefined || maybeCommand.startsWith("--") ? "propose" : maybeCommand;
  const rest =
    maybeCommand === undefined || !maybeCommand.startsWith("--") ? tail : [maybeCommand, ...tail];
  const flags = parseFlags(rest);

  if (command === "apply") {
    await runApply(flags);
  } else if (command === "propose") {
    await runPropose(flags, generatedAt);
  } else {
    process.stderr.write(`unknown command: ${command} (expected "propose" or "apply")\n`);
    process.exit(1);
  }
}

void main(new Date().toISOString()).catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exit(1);
});
