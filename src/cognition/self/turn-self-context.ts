import { computeExecutiveContextFits, selectExecutiveFocus } from "../../executive/index.js";
import type { ExecutiveFocus, ExecutiveStepsRepository } from "../../executive/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import { resolveViewerCapability } from "../../memory/episodic/access.js";
import { isEpisodeVisibleToAudience } from "../../memory/episodic/index.js";
import type { EpisodicRepository } from "../../memory/episodic/index.js";
import type {
  AutobiographicalRepository,
  GoalRecord,
  GoalsRepository,
  GrowthMarkersRepository,
  TraitsRepository,
  ValuesRepository,
} from "../../memory/self/index.js";
import {
  buildSelfScoringFeatureSet,
  selectActiveScoringValues,
  toRetrievalScoringFeatures,
  type RetrievalScoringFeatures,
  type SelfScoringFeatureSet,
} from "../../retrieval/scoring-features.js";
import type { Clock } from "../../util/clock.js";
import {
  episodeIdHelpers,
  goalIdHelpers,
  type EntityId,
  type EpisodeId,
  type GoalId,
  type SessionId,
} from "../../util/ids.js";
import type { AutonomyTriggerContext } from "../autonomy-trigger.js";
import type { SelfSnapshot } from "../deliberation/deliberator.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { PerceptionResult } from "../types.js";

type ProvenanceScopedSelfRecord = {
  id: string;
  provenance?: {
    kind: string;
    episode_ids?: readonly EpisodeId[];
  } | null;
  evidence_episode_ids?: readonly string[] | null;
  key_episode_ids?: readonly EpisodeId[] | null;
};

type GoalTreeNode = GoalRecord & { children?: unknown };

export type TurnSelfContextOptions = {
  embeddingClient: EmbeddingClient;
  episodicRepository: Pick<EpisodicRepository, "getMany">;
  entityRepository: Pick<EntityRepository, "getSelf">;
  valuesRepository: Pick<ValuesRepository, "list">;
  goalsRepository: Pick<GoalsRepository, "list">;
  traitsRepository: Pick<TraitsRepository, "list">;
  autobiographicalRepository?: Pick<AutobiographicalRepository, "currentPeriod">;
  growthMarkersRepository?: Pick<GrowthMarkersRepository, "list">;
  executiveStepsRepository: Pick<ExecutiveStepsRepository, "topOpen">;
  clock: Clock;
  tracer: TurnTracer;
  goalFocusThreshold: number;
  goalFollowupLookaheadMs: number;
  goalFollowupStaleMs: number;
};

export type TurnSelfContextInput = {
  turnId: string;
  sessionId?: SessionId;
  cognitionInput: string;
  perception: PerceptionResult;
  autonomyTrigger?: AutonomyTriggerContext | null;
  audienceEntityId: EntityId | null;
};

export type TurnSelfContext = {
  selfSnapshot: SelfSnapshot;
  activeScoringValues: ReturnType<typeof selectActiveScoringValues>;
  selfScoringFeatures: SelfScoringFeatureSet;
  retrievalScoringFeatures: RetrievalScoringFeatures;
  executiveFocus: ExecutiveFocus;
};

function flattenGoals(goals: ReadonlyArray<GoalTreeNode>): GoalRecord[] {
  const flattened: GoalRecord[] = [];
  const stack = [...goals];

  while (stack.length > 0) {
    const next = stack.shift();

    if (next === undefined) {
      continue;
    }

    flattened.push(next);

    if ("children" in next && Array.isArray(next.children)) {
      stack.push(...(next.children as GoalRecord[]));
    }
  }

  return flattened;
}

function getForcedExecutiveFocusGoalId(
  autonomyTrigger: AutonomyTriggerContext | null | undefined,
): GoalId | null {
  if (
    autonomyTrigger?.source_name !== "executive_focus_due" ||
    autonomyTrigger.payload.reason !== "step_due"
  ) {
    return null;
  }

  const candidate = autonomyTrigger.payload.force_executive_focus_goal_id;

  return typeof candidate === "string" && goalIdHelpers.is(candidate) ? candidate : null;
}

function applyForcedExecutiveFocus(
  focus: ExecutiveFocus,
  forcedGoalId: GoalId | null,
): ExecutiveFocus {
  if (forcedGoalId === null) {
    return focus;
  }

  const forcedScore = focus.candidates.find((candidate) => candidate.goal_id === forcedGoalId);

  if (forcedScore === undefined) {
    return focus;
  }

  return {
    ...focus,
    selected_goal: forcedScore.goal,
    selected_score: forcedScore,
  };
}

function getSelfRecordEvidenceEpisodeIds(record: ProvenanceScopedSelfRecord): EpisodeId[] {
  if (record.provenance?.kind !== "episodes") {
    return [];
  }

  const hasExplicitEvidence =
    record.evidence_episode_ids !== undefined || record.key_episode_ids !== undefined;
  const explicitEpisodeIds = [
    ...(record.evidence_episode_ids ?? []),
    ...(record.key_episode_ids ?? []),
  ].filter((id): id is EpisodeId => episodeIdHelpers.is(id));

  if (hasExplicitEvidence && explicitEpisodeIds.length === 0) {
    return [];
  }

  return [...new Set([...(record.provenance.episode_ids ?? []), ...explicitEpisodeIds])];
}

function isSelfRecordVisible(
  record: ProvenanceScopedSelfRecord,
  visibleEpisodeIds: ReadonlySet<EpisodeId>,
): boolean {
  const episodeIds = getSelfRecordEvidenceEpisodeIds(record);

  if (episodeIds.length === 0) {
    return true;
  }

  return episodeIds.some((episodeId) => visibleEpisodeIds.has(episodeId));
}

function visibleEpisodeIdsForRecord(
  episodeIds: readonly EpisodeId[] | readonly string[] | null | undefined,
  visibleEpisodeIds: ReadonlySet<EpisodeId>,
): EpisodeId[] {
  return [...new Set(episodeIds ?? [])].filter(
    (episodeId): episodeId is EpisodeId =>
      episodeIdHelpers.is(episodeId) && visibleEpisodeIds.has(episodeId),
  );
}

function redactSelfRecordEvidence<T extends ProvenanceScopedSelfRecord>(
  record: T,
  visibleEpisodeIds: ReadonlySet<EpisodeId>,
): T {
  let next: T = record;

  if (record.evidence_episode_ids !== undefined) {
    next = {
      ...next,
      evidence_episode_ids: visibleEpisodeIdsForRecord(
        record.evidence_episode_ids,
        visibleEpisodeIds,
      ),
    };
  }

  if (record.key_episode_ids !== undefined) {
    next = {
      ...next,
      key_episode_ids: visibleEpisodeIdsForRecord(record.key_episode_ids, visibleEpisodeIds),
    };
  }

  if (record.provenance?.kind === "episodes") {
    const visibleProvenanceEpisodeIds = visibleEpisodeIdsForRecord(
      record.provenance.episode_ids,
      visibleEpisodeIds,
    );

    next = {
      ...next,
      provenance: {
        ...record.provenance,
        episode_ids: visibleProvenanceEpisodeIds,
      },
    };
  }

  return next;
}

export class TurnSelfContextBuilder {
  constructor(private readonly options: TurnSelfContextOptions) {}

  async buildSelfSnapshot(audienceEntityId: EntityId | null): Promise<SelfSnapshot> {
    const goals = flattenGoals(
      this.options.goalsRepository.list({
        status: "active",
        visibleToAudienceEntityId: audienceEntityId,
      }),
    );
    const values = this.options.valuesRepository.list();
    const traits = this.options.traitsRepository.list();
    const currentPeriod = this.options.autobiographicalRepository?.currentPeriod() ?? null;
    const recentGrowthMarkers = this.options.growthMarkersRepository?.list({ limit: 3 }) ?? [];
    const identityRecords = [
      ...values,
      ...traits,
      ...(currentPeriod === null ? [] : [currentPeriod]),
      ...recentGrowthMarkers,
    ];
    const visibleEvidenceEpisodeIds = await this.visibleEvidenceEpisodeIdsForAudience(
      [...goals, ...identityRecords],
      audienceEntityId,
    );
    const visibleGoals = this.filterSelfRecordsByVisibleEvidence(goals, visibleEvidenceEpisodeIds);
    const visibleIdentityRecords = this.resolveIdentityRecordsForSelfContinuity(
      identityRecords,
      visibleEvidenceEpisodeIds,
    );
    const visibleIdentityById = new Map(visibleIdentityRecords.map((record) => [record.id, record]));

    return {
      values: values.map(
        (value) => (visibleIdentityById.get(value.id) as typeof value | undefined) ?? value,
      ),
      goals: visibleGoals,
      traits: traits.map(
        (trait) => (visibleIdentityById.get(trait.id) as typeof trait | undefined) ?? trait,
      ),
      currentPeriod:
        currentPeriod === null
          ? null
          : ((visibleIdentityById.get(currentPeriod.id) ?? currentPeriod) as typeof currentPeriod),
      recentGrowthMarkers: recentGrowthMarkers.map(
        (marker) => (visibleIdentityById.get(marker.id) as typeof marker | undefined) ?? marker,
      ),
    };
  }

  async listActiveGoalsVisibleToAudience(audienceEntityId: EntityId | null): Promise<GoalRecord[]> {
    const goals = flattenGoals(
      this.options.goalsRepository.list({
        status: "active",
        visibleToAudienceEntityId: audienceEntityId,
      }),
    );
    const visibleEvidenceEpisodeIds = await this.visibleEvidenceEpisodeIdsForAudience(
      goals,
      audienceEntityId,
    );

    return this.filterSelfRecordsByVisibleEvidence(goals, visibleEvidenceEpisodeIds);
  }

  private resolveIdentityRecordsForSelfContinuity<T extends ProvenanceScopedSelfRecord>(
    records: readonly T[],
    visibleEvidenceEpisodeIds: ReadonlySet<EpisodeId>,
  ): T[] {
    const selfContinuityCapability = resolveViewerCapability({
      globalIdentitySelfAudienceEntityId: this.options.entityRepository.getSelf()?.id ?? null,
    });

    if (selfContinuityCapability.kind !== "self_continuity") {
      return [];
    }

    return records.map((record) => redactSelfRecordEvidence(record, visibleEvidenceEpisodeIds));
  }

  private async visibleEvidenceEpisodeIdsForAudience(
    records: readonly ProvenanceScopedSelfRecord[],
    audienceEntityId: EntityId | null,
  ): Promise<Set<EpisodeId>> {
    const evidenceEpisodeIds = [
      ...new Set(records.flatMap((record) => getSelfRecordEvidenceEpisodeIds(record))),
    ];
    const evidenceEpisodes = await this.options.episodicRepository.getMany(evidenceEpisodeIds);

    return new Set(
      evidenceEpisodes
        .filter((episode) => isEpisodeVisibleToAudience(episode, audienceEntityId))
        .map((episode) => episode.id),
    );
  }

  async build(input: TurnSelfContextInput): Promise<TurnSelfContext> {
    const selfSnapshot = await this.buildSelfSnapshot(input.audienceEntityId);
    const executiveContextText = [
      input.cognitionInput,
      ...input.perception.entities,
      input.autonomyTrigger === null || input.autonomyTrigger === undefined
        ? ""
        : JSON.stringify(input.autonomyTrigger.payload),
    ]
      .join(" ")
      .trim();
    const activeScoringValues = selectActiveScoringValues(selfSnapshot.values);
    let selfScoringFeatures: SelfScoringFeatureSet = {
      goalVectors: [],
      valueVectors: [],
    };
    let contextFitByGoalId: Awaited<ReturnType<typeof computeExecutiveContextFits>> = new Map();

    try {
      selfScoringFeatures = await buildSelfScoringFeatureSet({
        embeddingClient: this.options.embeddingClient,
        goals: selfSnapshot.goals,
        activeValues: activeScoringValues,
      });
    } catch (error) {
      if (this.options.tracer.enabled) {
        this.options.tracer.emit("retrieval.degraded", {
          turnId: input.turnId,
          ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
          subsystem: "scoring_features",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    try {
      contextFitByGoalId = await computeExecutiveContextFits({
        embeddingClient: this.options.embeddingClient,
        goalVectors: selfScoringFeatures.goalVectors,
        contextText: executiveContextText,
      });
    } catch (error) {
      if (this.options.tracer.enabled) {
        this.options.tracer.emit("retrieval.degraded", {
          turnId: input.turnId,
          ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
          subsystem: "executive_context_fit",
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    }

    const executiveFocus = applyForcedExecutiveFocus(
      selectExecutiveFocus({
        goals: selfSnapshot.goals,
        cognitionInput: input.cognitionInput,
        perceptionEntities: input.perception.entities,
        autonomyPayload: input.autonomyTrigger?.payload ?? null,
        nowMs: this.options.clock.now(),
        threshold: this.options.goalFocusThreshold,
        deadlineLookaheadMs: this.options.goalFollowupLookaheadMs,
        staleMs: this.options.goalFollowupStaleMs,
        contextFitByGoalId,
      }),
      getForcedExecutiveFocusGoalId(input.autonomyTrigger),
    );
    const retrievalScoringFeatures = toRetrievalScoringFeatures({
      selfFeatures: selfScoringFeatures,
      primaryGoalId: executiveFocus.selected_goal?.id ?? null,
    });
    const executiveFocusWithStep =
      executiveFocus.selected_goal === null
        ? executiveFocus
        : {
            ...executiveFocus,
            next_step: this.options.executiveStepsRepository.topOpen(
              executiveFocus.selected_goal.id,
            ),
          };

    return {
      selfSnapshot,
      activeScoringValues,
      selfScoringFeatures,
      retrievalScoringFeatures,
      executiveFocus: executiveFocusWithStep,
    };
  }

  private filterSelfRecordsByVisibleEvidence<T extends ProvenanceScopedSelfRecord>(
    records: readonly T[],
    visibleEpisodeIds: ReadonlySet<EpisodeId>,
  ): T[] {
    return records
      .filter((record) => isSelfRecordVisible(record, visibleEpisodeIds))
      .map((record) => redactSelfRecordEvidence(record, visibleEpisodeIds));
  }
}
