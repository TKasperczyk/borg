import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { emotionalArcSchema } from "../../memory/affective/index.js";
import {
  buildConsolidationCoverageHash,
  consolidationFamilyIdSchema,
  normalizeEpisodeAccess,
  episodeIdSchema,
  episodeKindSchema,
  episodeLineageSchema,
  episodeTierSchema,
  type ConsolidationFamilyRecord,
  type ConsolidationMemberInput,
  type ConsolidationMemberRecord,
  type Episode,
  type EpisodeStats,
  type EpisodeTier,
} from "../../memory/episodic/index.js";
import type { EntityRecord } from "../../memory/commitments/index.js";
import { episodeAudienceEntityIdSchema, streamEntryIdSchema } from "../../memory/episodic/types.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
} from "../../retrieval/index.js";
import {
  createConsolidationFamilyId,
  createEpisodeId,
  type ConsolidationFamilyId,
  type EpisodeId,
} from "../../util/ids.js";
import { BudgetExceededError, StorageError } from "../../util/errors.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";

import type { ReverserRegistry } from "../audit-log.js";
import { getBudgetErrorTokens, withBudget } from "../budget.js";
import { episodeEvidencePromptRow } from "../evidence-labels.js";
import { offlineProcessError } from "../process-errors.js";
import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessError,
  OfflineResult,
} from "../types.js";

const mergeResponseSchema = z.object({
  title: z.string().min(1),
  narrative: z.string().min(1),
});
const MERGE_TOOL_NAME = "EmitConsolidation";
export const MERGE_TOOL = {
  name: MERGE_TOOL_NAME,
  description: "Emit the merged episode title and narrative for a redundant cluster.",
  inputSchema: toToolInputSchema(mergeResponseSchema),
} satisfies LLMToolDefinition;

const serializableEpisodeSchema = z.object({
  id: episodeIdSchema,
  title: z.string().min(1),
  narrative: z.string().min(1),
  participants: z.array(z.string().min(1)),
  location: z.string().min(1).nullable(),
  start_time: z.number().finite(),
  end_time: z.number().finite(),
  source_stream_ids: z.array(streamEntryIdSchema).min(1),
  significance: z.number().min(0).max(1),
  tags: z.array(z.string().min(1)),
  confidence: z.number().min(0).max(1),
  lineage: episodeLineageSchema,
  emotional_arc: emotionalArcSchema.nullable(),
  audience_entity_id: episodeAudienceEntityIdSchema.nullable().optional(),
  origin_audience_entity_ids: z.array(episodeAudienceEntityIdSchema).optional(),
  shared: z.boolean().optional(),
  episode_kind: episodeKindSchema.optional(),
  consolidation_family_id: consolidationFamilyIdSchema.nullable().optional(),
  consolidation_coverage_hash: z.string().min(1).nullable().optional(),
  embedding: z.array(z.number().finite()),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
});

const consolidatorPlanItemSchema = z.object({
  family_id: consolidationFamilyIdSchema,
  previous_current_version_episode_id: episodeIdSchema.nullable(),
  source_episode_ids: z.array(episodeIdSchema).min(1),
  new_raw_episode_ids: z.array(episodeIdSchema).min(1),
  coverage_hash: z.string().min(1),
  merged_episode: serializableEpisodeSchema,
  inherited_tier: episodeTierSchema,
});

export const consolidatorPlanSchema = z.object({
  process: z.literal("consolidator"),
  items: z.array(consolidatorPlanItemSchema),
  errors: z
    .array(
      z.object({
        process: z.literal("consolidator"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type ConsolidatorPlan = z.infer<typeof consolidatorPlanSchema>;

const consolidationReversalSchema = z.object({
  familyId: consolidationFamilyIdSchema,
  versionEpisodeId: episodeIdSchema,
  previousCurrentVersionEpisodeId: episodeIdSchema.nullable(),
  previousCoverageHash: z.string().min(1).nullable(),
  previousPolicyVersion: z.number().int().positive().nullable(),
});

const CONSOLIDATION_POLICY_VERSION = 1;
const TIER_ORDER: Record<EpisodeTier, number> = {
  T1: 1,
  T2: 2,
  T3: 3,
  T4: 4,
};

type MergeSelfEntity = Pick<EntityRecord, "id" | "canonical_name">;
type ConsolidatorConfig = OfflineContext["config"]["offline"]["consolidator"];

type RawEpisodeWithStats = {
  episode: Episode;
  stats: EpisodeStats;
};

type ActiveFamilyAnchor = {
  family: ConsolidationFamilyRecord;
  currentVersion: Episode;
  members: ConsolidationMemberRecord[];
  rawEpisodes: Episode[];
  rawEpisodeIds: EpisodeId[];
  rawEpisodeIdSet: ReadonlySet<EpisodeId>;
};

type ConsolidationCandidate = {
  familyId: ConsolidationFamilyId;
  previousCurrentVersionEpisodeId: EpisodeId | null;
  previousCurrentVersion: Episode | null;
  rawEpisodes: Episode[];
  newRawEpisodes: Episode[];
  stats: EpisodeStats[];
  coverageHash: string;
};

function compareTier(left: EpisodeTier, right: EpisodeTier): number {
  return TIER_ORDER[left] - TIER_ORDER[right];
}

function compareEpisodesOldestFirst(left: Episode, right: Episode): number {
  return left.created_at - right.created_at || left.id.localeCompare(right.id);
}

function compareRawRowsNewestFirst(left: RawEpisodeWithStats, right: RawEpisodeWithStats): number {
  return (
    right.episode.updated_at - left.episode.updated_at ||
    right.episode.created_at - left.episode.created_at ||
    left.episode.id.localeCompare(right.episode.id)
  );
}

function temporalGapMs(left: Episode, right: Episode): number {
  const leftStart = Math.min(left.start_time, left.end_time);
  const leftEnd = Math.max(left.start_time, left.end_time);
  const rightStart = Math.min(right.start_time, right.end_time);
  const rightEnd = Math.max(right.start_time, right.end_time);

  if (leftStart <= rightEnd && rightStart <= leftEnd) {
    return 0;
  }

  return Math.min(Math.abs(leftStart - rightEnd), Math.abs(rightStart - leftEnd));
}

function passesTemporalSoftGuard(
  left: Episode,
  right: Episode,
  similarity: number,
  config: ConsolidatorConfig,
): boolean {
  const gapMs = temporalGapMs(left, right);

  return (
    gapMs <= config.temporalProximityMs ||
    (similarity >= config.highSimilarityTemporalBypassThreshold &&
      gapMs <= config.highSimilarityTemporalBypassMaxGapMs)
  );
}

function pairCohesion(
  left: Episode,
  right: Episode,
  config: ConsolidatorConfig,
): {
  eligible: boolean;
  similarity: number;
} {
  const similarity = cosineSimilarity(left.embedding, right.embedding);
  const diameter = Math.max(0, 1 - similarity);

  return {
    eligible:
      similarity >= config.similarityThreshold &&
      diameter <= config.maxClusterDiameter &&
      passesTemporalSoftGuard(left, right, similarity, config),
    similarity,
  };
}

function maxTier(stats: readonly EpisodeStats[]): EpisodeTier {
  return stats.reduce<EpisodeTier>(
    (best, current) => (compareTier(current.tier, best) > 0 ? current.tier : best),
    "T1",
  );
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}

function uniqueEpisodeIds(ids: readonly EpisodeId[]): EpisodeId[] {
  return [...new Set(ids)];
}

function uniqueEpisodesById(episodes: readonly Episode[]): Episode[] {
  const byId = new Map<EpisodeId, Episode>();

  for (const episode of episodes) {
    byId.set(episode.id, episode);
  }

  return [...byId.values()].sort(compareEpisodesOldestFirst);
}

function sourceStreamIdsForEpisodes(episodes: readonly Episode[]): Episode["source_stream_ids"] {
  return uniqueStrings(
    episodes.flatMap((episode) => episode.source_stream_ids),
  ) as Episode["source_stream_ids"];
}

function coverageHashForSourceStreamIds(sourceStreamIds: readonly string[]): string {
  return buildConsolidationCoverageHash([
    ...sourceStreamIds,
    `consolidation_policy_version:${CONSOLIDATION_POLICY_VERSION}`,
  ]);
}

function coverageHashForEpisodes(episodes: readonly Episode[]): string {
  return coverageHashForSourceStreamIds(sourceStreamIdsForEpisodes(episodes));
}

function sameEpisodeIdSet(left: readonly EpisodeId[], right: readonly EpisodeId[]): boolean {
  const leftSet = new Set(left);
  const rightSet = new Set(right);

  return leftSet.size === rightSet.size && [...leftSet].every((id) => rightSet.has(id));
}

function episodeAccessFromCombinedDisclosureLabel(
  label: ReturnType<typeof combineMemoryDisclosureLabels>,
): Pick<Episode, "audience_entity_id" | "origin_audience_entity_ids" | "shared"> {
  return normalizeEpisodeAccess({
    origin_audience_entity_ids: [...label.originAudienceEntityIds],
    shared: label.disclosureClass === "public",
  });
}

function parseMergeResponse(result: LLMCompleteResult) {
  const call = result.tool_calls.find((toolCall) => toolCall.name === MERGE_TOOL_NAME);

  if (call === undefined) {
    throw new StorageError(`Consolidator did not emit tool ${MERGE_TOOL_NAME}`, {
      code: "CONSOLIDATOR_INVALID",
    });
  }

  return mergeResponseSchema.parse(call.input);
}

function buildMergePrompt(
  candidate: ConsolidationCandidate,
  selfEntity: MergeSelfEntity | null,
): string {
  const selfEntityGuidance =
    selfEntity === null
      ? null
      : `You are entity ${selfEntity.id} (${selfEntity.canonical_name}); content grounded in your own agent-authored source messages is self-owned. Use first person only for your own actions, statements, and decisions; keep every other participant named and world facts in third person.`;
  const previousContext =
    candidate.previousCurrentVersion === null
      ? []
      : [
          "Previous current consolidation context:",
          JSON.stringify(
            episodeEvidencePromptRow(candidate.previousCurrentVersion, {
              participants: candidate.previousCurrentVersion.participants,
              location: candidate.previousCurrentVersion.location,
              start_time: candidate.previousCurrentVersion.start_time,
              end_time: candidate.previousCurrentVersion.end_time,
              tags: candidate.previousCurrentVersion.tags,
              source_stream_ids: candidate.previousCurrentVersion.source_stream_ids,
            }),
          ),
          "Use the previous consolidation as context only. The new version lineage and coverage must remain grounded in raw leaf episodes.",
        ];

  return [
    "Merge the redundant raw episodes into one grounded consolidation version.",
    `Emit your result by calling the ${MERGE_TOOL_NAME} tool exactly once.`,
    "Preserve facts from all raw inputs. Keep the narrative to 2-5 sentences.",
    ...(selfEntityGuidance === null ? [] : [selfEntityGuidance]),
    `${SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE} Apply this to the merged narrative. Keep the title topic-neutral and scannable rather than first-person narration.`,
    ...previousContext,
    "New raw evidence:",
    ...candidate.newRawEpisodes.map((episode) =>
      JSON.stringify(
        episodeEvidencePromptRow(episode, {
          participants: episode.participants,
          location: episode.location,
          start_time: episode.start_time,
          end_time: episode.end_time,
          tags: episode.tags,
          source_stream_ids: episode.source_stream_ids,
        }),
      ),
    ),
  ].join("\n");
}

function buildCompleteLinkClusters(
  rows: readonly RawEpisodeWithStats[],
  config: ConsolidatorConfig,
): Array<{
  rows: RawEpisodeWithStats[];
}> {
  const sorted = [...rows].sort(compareRawRowsNewestFirst);
  const consumed = new Set<EpisodeId>();
  const clusters: Array<{ rows: RawEpisodeWithStats[] }> = [];

  for (const seed of sorted) {
    if (consumed.has(seed.episode.id)) {
      continue;
    }

    const candidates = sorted
      .filter(
        (candidate) =>
          candidate.episode.id !== seed.episode.id && !consumed.has(candidate.episode.id),
      )
      .map((candidate) => ({
        candidate,
        cohesion: pairCohesion(seed.episode, candidate.episode, config),
      }))
      .filter((candidate) => candidate.cohesion.eligible)
      .sort((left, right) => right.cohesion.similarity - left.cohesion.similarity);
    const cluster = [seed];

    for (const { candidate } of candidates) {
      if (cluster.length >= config.maxFamilyRawMembers) {
        break;
      }

      if (
        cluster.every((member) => pairCohesion(member.episode, candidate.episode, config).eligible)
      ) {
        cluster.push(candidate);
      }
    }

    if (cluster.length < config.minClusterSize) {
      continue;
    }

    for (const member of cluster) {
      consumed.add(member.episode.id);
    }

    clusters.push({
      rows: [...cluster].sort((left, right) =>
        compareEpisodesOldestFirst(left.episode, right.episode),
      ),
    });
  }

  return clusters.sort(
    (left, right) =>
      right.rows.length - left.rows.length ||
      (right.rows[0]?.episode.updated_at ?? 0) - (left.rows[0]?.episode.updated_at ?? 0),
  );
}

function familyCoversRawEpisodeIds(
  family: ActiveFamilyAnchor,
  rawEpisodeIds: readonly EpisodeId[],
): boolean {
  return (
    family.family.policy_version === CONSOLIDATION_POLICY_VERSION &&
    rawEpisodeIds.every((episodeId) => family.rawEpisodeIdSet.has(episodeId))
  );
}

function findCoveringFamily(
  families: readonly ActiveFamilyAnchor[],
  rawEpisodeIds: readonly EpisodeId[],
): ActiveFamilyAnchor | undefined {
  const uniqueRawEpisodeIds = uniqueEpisodeIds(rawEpisodeIds);
  return families.find((family) => familyCoversRawEpisodeIds(family, uniqueRawEpisodeIds));
}

function attachmentCohesionAgainstRawMembers(
  episode: Episode,
  family: ActiveFamilyAnchor,
  config: ConsolidatorConfig,
): number | null {
  if (
    family.rawEpisodeIdSet.has(episode.id) ||
    family.rawEpisodes.length === 0 ||
    family.rawEpisodes.length >= config.maxFamilyRawMembers
  ) {
    return null;
  }

  let weakestSimilarity = Number.POSITIVE_INFINITY;

  for (const rawMember of family.rawEpisodes) {
    const cohesion = pairCohesion(episode, rawMember, config);

    if (!cohesion.eligible) {
      return null;
    }

    weakestSimilarity = Math.min(weakestSimilarity, cohesion.similarity);
  }

  return Number.isFinite(weakestSimilarity) ? weakestSimilarity : null;
}

function attachmentAnchorsForRow(
  row: RawEpisodeWithStats,
  families: readonly ActiveFamilyAnchor[],
  config: ConsolidatorConfig,
): ActiveFamilyAnchor[] {
  return families
    .map((family) => ({
      family,
      similarity: attachmentCohesionAgainstRawMembers(row.episode, family, config),
    }))
    .filter(
      (candidate): candidate is { family: ActiveFamilyAnchor; similarity: number } =>
        candidate.similarity !== null,
    )
    .sort(
      (left, right) =>
        right.similarity - left.similarity ||
        right.family.family.updated_at - left.family.family.updated_at,
    )
    .map((candidate) => candidate.family);
}

function attachmentGroupAcceptsRow(
  group: {
    anchor: ActiveFamilyAnchor;
    rows: RawEpisodeWithStats[];
  },
  row: RawEpisodeWithStats,
  config: ConsolidatorConfig,
): boolean {
  if (group.anchor.rawEpisodes.length + group.rows.length >= config.maxFamilyRawMembers) {
    return false;
  }

  return group.rows.every((member) => pairCohesion(member.episode, row.episode, config).eligible);
}

async function loadActiveFamilyAnchors(ctx: OfflineContext): Promise<ActiveFamilyAnchor[]> {
  const families = ctx.episodicRepository.listConsolidationFamilies();

  if (families.length === 0) {
    return [];
  }

  const members = ctx.episodicRepository.listConsolidationMembers();
  const membersByFamilyId = new Map<ConsolidationFamilyId, ConsolidationMemberRecord[]>();

  for (const member of members) {
    const list = membersByFamilyId.get(member.family_id) ?? [];
    list.push(member);
    membersByFamilyId.set(member.family_id, list);
  }

  const currentVersions = await ctx.episodicRepository.getMany(
    families.map((family) => family.current_version_episode_id),
  );
  const currentVersionById = new Map(currentVersions.map((episode) => [episode.id, episode]));
  const rawEpisodeIds = uniqueEpisodeIds(members.map((member) => member.raw_episode_id));
  const rawEpisodes = await ctx.episodicRepository.getMany(rawEpisodeIds);
  const rawEpisodeById = new Map(rawEpisodes.map((episode) => [episode.id, episode]));
  const anchors: ActiveFamilyAnchor[] = [];

  for (const family of families) {
    const currentVersion = currentVersionById.get(family.current_version_episode_id);

    if (
      currentVersion === undefined ||
      currentVersion.episode_kind !== "consolidation_version" ||
      currentVersion.consolidation_family_id !== family.family_id ||
      !ctx.episodicRepository.isEpisodeEffectivelyVisible(currentVersion.id)
    ) {
      continue;
    }

    const familyMembers = membersByFamilyId.get(family.family_id) ?? [];
    const familyRawEpisodeIds = uniqueEpisodeIds(
      familyMembers.map((member) => member.raw_episode_id),
    );
    const familyRawEpisodes = familyRawEpisodeIds
      .map((episodeId) => rawEpisodeById.get(episodeId))
      .filter((episode): episode is Episode => episode !== undefined)
      .sort(compareEpisodesOldestFirst);

    if (familyRawEpisodes.length !== familyRawEpisodeIds.length) {
      continue;
    }

    anchors.push({
      family,
      currentVersion,
      members: familyMembers,
      rawEpisodes: familyRawEpisodes,
      rawEpisodeIds: familyRawEpisodeIds,
      rawEpisodeIdSet: new Set(familyRawEpisodeIds),
    });
  }

  return anchors;
}

async function collectConsolidationCandidates(
  ctx: OfflineContext,
  statsById: ReadonlyMap<EpisodeId, EpisodeStats>,
): Promise<ConsolidationCandidate[]> {
  const config = ctx.config.offline.consolidator;
  const activeFamilies = await loadActiveFamilyAnchors(ctx);
  const visibleRawRows = (await ctx.episodicRepository.listEffectivelyVisible())
    .filter((episode) => (episode.episode_kind ?? "raw") === "raw")
    .map((episode) => {
      const stats = statsById.get(episode.id);
      return stats === undefined ? null : { episode, stats };
    })
    .filter((row): row is RawEpisodeWithStats => row !== null)
    .sort(compareRawRowsNewestFirst);
  const attachedIds = new Set<EpisodeId>();
  const attachmentsByFamilyId = new Map<
    ConsolidationFamilyId,
    {
      anchor: ActiveFamilyAnchor;
      rows: RawEpisodeWithStats[];
    }
  >();

  for (const row of visibleRawRows) {
    for (const anchor of attachmentAnchorsForRow(row, activeFamilies, config)) {
      const group = attachmentsByFamilyId.get(anchor.family.family_id) ?? {
        anchor,
        rows: [],
      };

      if (!attachmentGroupAcceptsRow(group, row, config)) {
        continue;
      }

      group.rows.push(row);
      attachmentsByFamilyId.set(anchor.family.family_id, group);
      attachedIds.add(row.episode.id);
      break;
    }
  }

  const candidates: ConsolidationCandidate[] = [];

  for (const { anchor, rows } of attachmentsByFamilyId.values()) {
    const newRawEpisodes = rows.map((row) => row.episode).sort(compareEpisodesOldestFirst);
    const rawEpisodes = uniqueEpisodesById([...anchor.rawEpisodes, ...newRawEpisodes]);
    const stats = rawEpisodes
      .map((episode) => statsById.get(episode.id))
      .filter((statsItem): statsItem is EpisodeStats => statsItem !== undefined);

    if (stats.length !== rawEpisodes.length) {
      continue;
    }

    candidates.push({
      familyId: anchor.family.family_id,
      previousCurrentVersionEpisodeId: anchor.currentVersion.id,
      previousCurrentVersion: anchor.currentVersion,
      rawEpisodes,
      newRawEpisodes,
      stats,
      coverageHash: coverageHashForEpisodes(rawEpisodes),
    });
  }

  const clusterRows = visibleRawRows.filter((row) => !attachedIds.has(row.episode.id));

  for (const cluster of buildCompleteLinkClusters(clusterRows, config)) {
    const rawEpisodes = cluster.rows.map((row) => row.episode).sort(compareEpisodesOldestFirst);

    candidates.push({
      familyId: createConsolidationFamilyId(),
      previousCurrentVersionEpisodeId: null,
      previousCurrentVersion: null,
      rawEpisodes,
      newRawEpisodes: rawEpisodes,
      stats: cluster.rows.map((row) => row.stats),
      coverageHash: coverageHashForEpisodes(rawEpisodes),
    });
  }

  return candidates
    .filter(
      (candidate) =>
        findCoveringFamily(
          activeFamilies,
          candidate.rawEpisodes.map((episode) => episode.id),
        ) === undefined,
    )
    .sort(
      (left, right) =>
        Number(left.previousCurrentVersionEpisodeId === null) -
          Number(right.previousCurrentVersionEpisodeId === null) ||
        right.newRawEpisodes.length - left.newRawEpisodes.length ||
        (right.newRawEpisodes[0]?.updated_at ?? 0) - (left.newRawEpisodes[0]?.updated_at ?? 0),
    )
    .slice(0, config.maxClustersPerRun);
}

async function buildMergedEpisode(
  ctx: OfflineContext,
  llmClient: LLMClient,
  candidate: ConsolidationCandidate,
): Promise<{ episode: Episode; inheritedTier: EpisodeTier }> {
  const selfEntity = ctx.entityRepository.getSelf();
  const merged = parseMergeResponse(
    await llmClient.complete({
      model: ctx.config.anthropic.models.background,
      system:
        "You merge overlapping autobiographical episodes. Keep only grounded facts from the raw inputs.",
      messages: [
        {
          role: "user",
          content: buildMergePrompt(candidate, selfEntity),
        },
      ],
      tools: [MERGE_TOOL],
      tool_choice: { type: "tool", name: MERGE_TOOL_NAME },
      max_tokens: 6_000,
      budget: "offline-consolidator",
    }),
  );
  const rawEpisodes = [...candidate.rawEpisodes].sort(compareEpisodesOldestFirst);
  const participants = uniqueStrings(rawEpisodes.flatMap((episode) => episode.participants));
  const sourceStreamIds = sourceStreamIdsForEpisodes(rawEpisodes);
  const tags = uniqueStrings(rawEpisodes.flatMap((episode) => episode.tags));
  const startTime = Math.min(...rawEpisodes.map((episode) => episode.start_time));
  const endTime = Math.max(...rawEpisodes.map((episode) => episode.end_time));
  const significance = Math.max(...rawEpisodes.map((episode) => episode.significance));
  const confidence = Math.min(...rawEpisodes.map((episode) => episode.confidence));
  const locationValues = uniqueStrings(
    rawEpisodes.flatMap((episode) => (episode.location === null ? [] : [episode.location])),
  );
  const nowMs = ctx.clock.now();
  const embedding = await ctx.embeddingClient.embed(
    `${merged.title}\n${merged.narrative}\n${tags.join(" ")}\n${participants.join(" ")}`,
  );
  const access = episodeAccessFromCombinedDisclosureLabel(
    combineMemoryDisclosureLabels(rawEpisodes.map(memoryDisclosureLabelFromEpisodeAccess)),
  );
  const rawEpisodeIds = rawEpisodes.map((episode) => episode.id);

  return {
    episode: normalizeEpisodeAccess({
      id: createEpisodeId(),
      title: merged.title.trim(),
      narrative: merged.narrative.trim(),
      participants,
      location: locationValues.length === 1 ? (locationValues[0] ?? null) : null,
      start_time: startTime,
      end_time: endTime,
      source_stream_ids: sourceStreamIds,
      significance,
      tags,
      confidence,
      lineage: {
        derived_from: rawEpisodeIds,
        supersedes: rawEpisodeIds,
      },
      emotional_arc:
        rawEpisodes.find((episode) => episode.emotional_arc !== null)?.emotional_arc ?? null,
      audience_entity_id: access.audience_entity_id,
      origin_audience_entity_ids: access.origin_audience_entity_ids,
      shared: access.shared,
      episode_kind: "consolidation_version",
      consolidation_family_id: candidate.familyId,
      consolidation_coverage_hash: candidate.coverageHash,
      embedding,
      created_at: nowMs,
      updated_at: nowMs,
    }),
    inheritedTier: maxTier(candidate.stats),
  };
}

function serializeEpisode(episode: Episode) {
  return serializableEpisodeSchema.parse({
    ...episode,
    embedding: Array.from(episode.embedding),
  });
}

function deserializeEpisode(episode: z.infer<typeof serializableEpisodeSchema>): Episode {
  return {
    ...episode,
    embedding: Float32Array.from(episode.embedding),
  };
}

function buildChange(item: ConsolidatorPlan["items"][number]): OfflineChange {
  return {
    process: "consolidator",
    action: "consolidate",
    targets: {
      family_id: item.family_id,
      new_version_episode_id: item.merged_episode.id,
      previous_current_version_episode_id: item.previous_current_version_episode_id,
      source_ids: item.source_episode_ids,
      new_raw_episode_ids: item.new_raw_episode_ids,
      coverage_hash: item.coverage_hash,
    },
    preview: {
      title: item.merged_episode.title,
      narrative: item.merged_episode.narrative,
      family_id: item.family_id,
      source_ids: item.source_episode_ids,
      new_raw_episode_ids: item.new_raw_episode_ids,
    },
  };
}

function assertPlanItemMatchesSources(
  item: ConsolidatorPlan["items"][number],
  rawEpisodes: readonly Episode[],
): void {
  const rawEpisodeIds = rawEpisodes.map((episode) => episode.id);

  if (!sameEpisodeIdSet(item.source_episode_ids, rawEpisodeIds)) {
    throw new StorageError("Consolidator plan source ids no longer match raw episode rows", {
      code: "CONSOLIDATOR_PLAN_INVALID",
    });
  }

  if (item.coverage_hash !== coverageHashForEpisodes(rawEpisodes)) {
    throw new StorageError("Consolidator plan coverage hash is stale", {
      code: "CONSOLIDATOR_PLAN_INVALID",
    });
  }

  if (!sameEpisodeIdSet(item.source_episode_ids, item.merged_episode.lineage.derived_from)) {
    throw new StorageError("Consolidator version lineage must reference raw leaf episodes", {
      code: "CONSOLIDATOR_PLAN_INVALID",
    });
  }

  if (
    item.merged_episode.episode_kind !== "consolidation_version" ||
    item.merged_episode.consolidation_family_id !== item.family_id ||
    item.merged_episode.consolidation_coverage_hash !== item.coverage_hash
  ) {
    throw new StorageError("Consolidator plan version metadata is invalid", {
      code: "CONSOLIDATOR_PLAN_INVALID",
    });
  }
}

export type ConsolidatorProcessOptions = {
  episodicRepository: OfflineContext["episodicRepository"];
  registry: ReverserRegistry;
};

export class ConsolidatorProcess implements OfflineProcess {
  readonly name = "consolidator" as const;

  constructor(private readonly options: ConsolidatorProcessOptions) {
    this.options.registry.register(this.name, "consolidate", async ({ reversal }) => {
      const parsed = consolidationReversalSchema.safeParse(reversal);

      if (parsed.success) {
        await this.options.episodicRepository.revertConsolidationVersion({
          familyId: parsed.data.familyId,
          versionEpisodeId: parsed.data.versionEpisodeId,
          previousCurrentVersionEpisodeId: parsed.data.previousCurrentVersionEpisodeId,
          previousCoverageHash: parsed.data.previousCoverageHash,
          previousPolicyVersion: parsed.data.previousPolicyVersion,
        });
        return;
      }

      const legacy = reversal as Partial<{ newEpisodeId: string }>;

      if (typeof legacy.newEpisodeId === "string") {
        await this.options.episodicRepository.delete(legacy.newEpisodeId as EpisodeId);
      }
    });
  }

  async plan(ctx: OfflineContext, opts: { budget?: number } = {}): Promise<ConsolidatorPlan> {
    const errors: OfflineProcessError[] = [];
    const items: ConsolidatorPlan["items"] = [];
    const budget = opts.budget ?? ctx.config.offline.consolidator.budget;
    const statsById = new Map(
      ctx.episodicRepository.listStats().map((stats) => [stats.episode_id, stats] as const),
    );
    const candidates = await collectConsolidationCandidates(ctx, statsById);
    let tokensUsed = 0;
    let budgetExhausted = false;

    try {
      const budgeted = await withBudget(this.name, budget, async ({ wrapClient }) => {
        const llmClient = wrapClient(ctx.llm.background);

        for (const candidate of candidates) {
          try {
            const merged = await buildMergedEpisode(ctx, llmClient, candidate);
            items.push({
              family_id: candidate.familyId,
              previous_current_version_episode_id: candidate.previousCurrentVersionEpisodeId,
              source_episode_ids: candidate.rawEpisodes.map((episode) => episode.id),
              new_raw_episode_ids: candidate.newRawEpisodes.map((episode) => episode.id),
              coverage_hash: candidate.coverageHash,
              merged_episode: serializeEpisode(merged.episode),
              inherited_tier: merged.inheritedTier,
            });
          } catch (error) {
            if (error instanceof BudgetExceededError) {
              throw error;
            }

            errors.push(
              offlineProcessError(this.name, error, {
                code: error instanceof StorageError ? error.code : undefined,
                includeErrorCode: false,
              }),
            );
          }
        }
      });

      tokensUsed = budgeted.tokens_used;
    } catch (error) {
      tokensUsed = getBudgetErrorTokens(error);
      budgetExhausted = error instanceof BudgetExceededError;
      errors.push(offlineProcessError(this.name, error));
    }

    return consolidatorPlanSchema.parse({
      process: this.name,
      items,
      errors,
      tokens_used: tokensUsed,
      budget_exhausted: budgetExhausted,
    });
  }

  preview(plan: ConsolidatorPlan): OfflineResult {
    const parsed = consolidatorPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: parsed.items.map((item) => buildChange(item)),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: ConsolidatorPlan): Promise<OfflineResult> {
    const plan = consolidatorPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];

    for (const item of plan.items) {
      const rawEpisodes = await ctx.episodicRepository.getMany(item.source_episode_ids);

      if (rawEpisodes.length !== uniqueEpisodeIds(item.source_episode_ids).length) {
        throw new StorageError("Consolidator plan references missing raw episodes", {
          code: "CONSOLIDATOR_PLAN_INVALID",
        });
      }

      if (rawEpisodes.some((episode) => (episode.episode_kind ?? "raw") !== "raw")) {
        throw new StorageError("Consolidator plan source ids must be raw episodes", {
          code: "CONSOLIDATOR_PLAN_INVALID",
        });
      }

      assertPlanItemMatchesSources(item, rawEpisodes);

      const newRawEpisodeIds = uniqueEpisodeIds(item.new_raw_episode_ids);
      const newRawEpisodes = rawEpisodes.filter((episode) => newRawEpisodeIds.includes(episode.id));

      if (newRawEpisodes.length !== newRawEpisodeIds.length) {
        throw new StorageError("Consolidator plan references missing new raw episodes", {
          code: "CONSOLIDATOR_PLAN_INVALID",
        });
      }

      const existingMemberRawIds =
        item.previous_current_version_episode_id === null
          ? []
          : ctx.episodicRepository
              .listConsolidationMembers(item.family_id)
              .map((member) => member.raw_episode_id);
      const expectedNewRawEpisodeIds = uniqueEpisodeIds(item.source_episode_ids).filter(
        (episodeId) => !existingMemberRawIds.includes(episodeId),
      );

      if (!sameEpisodeIdSet(newRawEpisodeIds, expectedNewRawEpisodeIds)) {
        throw new StorageError(
          "Consolidator plan new raw episode ids must equal source ids minus existing family members",
          {
            code: "CONSOLIDATOR_PLAN_INVALID",
          },
        );
      }

      if (
        newRawEpisodeIds.some(
          (episodeId) => !ctx.episodicRepository.isEpisodeEffectivelyVisible(episodeId),
        )
      ) {
        continue;
      }

      const activeFamilies = await loadActiveFamilyAnchors(ctx);

      if (findCoveringFamily(activeFamilies, item.source_episode_ids) !== undefined) {
        continue;
      }

      const family = ctx.episodicRepository.getConsolidationFamily(item.family_id);

      if (item.previous_current_version_episode_id === null) {
        if (family !== null) {
          continue;
        }
      } else if (family?.current_version_episode_id !== item.previous_current_version_episode_id) {
        continue;
      }

      const previousCoverageHash = family?.coverage_hash ?? null;
      const previousPolicyVersion = family?.policy_version ?? null;

      const mergedEpisode = deserializeEpisode(item.merged_episode);
      const members: ConsolidationMemberInput[] = newRawEpisodes.map((episode) => ({
        raw_episode_id: episode.id,
        source_stream_ids: episode.source_stream_ids,
        added_by_version_episode_id: mergedEpisode.id,
      }));
      let createdVersion = false;

      try {
        await ctx.episodicRepository.createEpisode(mergedEpisode);
        createdVersion = true;
        ctx.episodicRepository.updateStats(mergedEpisode.id, {
          tier: item.inherited_tier,
          promoted_at: ctx.clock.now(),
          promoted_from: "consolidator",
        });

        if (item.previous_current_version_episode_id === null) {
          ctx.episodicRepository.createConsolidationFamily({
            familyId: item.family_id,
            currentVersionEpisodeId: mergedEpisode.id,
            coverageHash: item.coverage_hash,
            policyVersion: CONSOLIDATION_POLICY_VERSION,
            members,
          });
        } else {
          ctx.episodicRepository.extendConsolidationFamily({
            familyId: item.family_id,
            expectedCurrentVersionEpisodeId: item.previous_current_version_episode_id,
            nextVersionEpisodeId: mergedEpisode.id,
            coverageHash: item.coverage_hash,
            policyVersion: CONSOLIDATION_POLICY_VERSION,
            members,
          });
        }
      } catch (error) {
        if (createdVersion) {
          await ctx.episodicRepository.delete(mergedEpisode.id);
        }

        if (error instanceof StorageError && error.code === "CONSOLIDATION_FAMILY_STALE") {
          continue;
        }

        throw error;
      }

      ctx.auditLog.record({
        run_id: ctx.runId,
        process: this.name,
        action: "consolidate",
        targets: {
          familyId: item.family_id,
          versionEpisodeId: mergedEpisode.id,
          previousCurrentVersionEpisodeId: item.previous_current_version_episode_id,
          sourceIds: item.source_episode_ids,
          newRawEpisodeIds: item.new_raw_episode_ids,
          coverageHash: item.coverage_hash,
          previousCoverageHash,
          previousPolicyVersion,
        },
        reversal: {
          familyId: item.family_id,
          versionEpisodeId: mergedEpisode.id,
          previousCurrentVersionEpisodeId: item.previous_current_version_episode_id,
          previousCoverageHash,
          previousPolicyVersion,
        },
      });
      changes.push(buildChange(item));
    }

    return {
      process: this.name,
      dryRun: false,
      changes,
      tokens_used: plan.tokens_used,
      errors: plan.errors,
      budget_exhausted: plan.budget_exhausted,
    };
  }

  async run(
    ctx: OfflineContext,
    opts: { dryRun?: boolean; budget?: number },
  ): Promise<OfflineResult> {
    const plan = await this.plan(ctx, opts);
    return opts.dryRun === true ? this.preview(plan) : this.apply(ctx, plan);
  }
}
