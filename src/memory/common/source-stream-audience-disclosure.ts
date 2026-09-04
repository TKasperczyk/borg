import type { CommitmentRecord, EntityRepository } from "../commitments/index.js";
import { flattenGoalTree, type GoalRecord, type GoalTreeNode } from "../self/index.js";
import type { SessionsRepository } from "../../sessions/index.js";
import {
  readStreamEntryAtOffset,
  type StreamEntry,
  type StreamEntryIndexRecord,
  type StreamEntryIndexRepository,
} from "../../stream/index.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import type { CommitmentId, EntityId, GoalId, SessionId, StreamEntryId } from "../../util/ids.js";
import {
  memoryDisclosureLabelMetadata,
  type MemoryDisclosureLabel,
  type MemoryDisclosureLabelMetadata,
} from "./disclosure-label.js";
import {
  commitmentScopeMemoryDisclosureLabel,
  goalScopeMemoryDisclosureLabel,
} from "./disclosure-serializers.js";

export type DisclosureResolvedCommitment = CommitmentRecord & {
  disclosure_label: MemoryDisclosureLabelMetadata;
};

export type DisclosureResolvedGoal = GoalRecord & {
  disclosure_label: MemoryDisclosureLabelMetadata;
};

export type DisclosureResolvedGoalTreeNode = DisclosureResolvedGoal & {
  children: DisclosureResolvedGoalTreeNode[];
};

export type SourceStreamAudienceDisclosureResolverOptions = {
  dataDir: string;
  entryIndex: Pick<StreamEntryIndexRepository, "lookupMany">;
  sessionsRepository: Pick<SessionsRepository, "getMany">;
  entityRepository: Pick<EntityRepository, "findByNames">;
};

export type CommitmentDisclosureSource = Pick<
  CommitmentRecord,
  "id" | "made_to_entity" | "restricted_audience" | "source_stream_entry_ids"
>;

export type GoalDisclosureSource = Pick<
  GoalRecord,
  "id" | "audience_entity_id" | "owner_entity_id" | "source_stream_entry_ids"
>;

export type ResolveCommitmentGoalDisclosureLabelsInput = {
  commitments?: readonly CommitmentDisclosureSource[];
  goals?: readonly GoalDisclosureSource[];
};

export type ResolvedCommitmentGoalDisclosureLabels = {
  commitmentLabels: readonly MemoryDisclosureLabel[];
  goalLabels: readonly MemoryDisclosureLabel[];
  commitmentLabelsById: ReadonlyMap<CommitmentId, MemoryDisclosureLabel>;
  goalLabelsById: ReadonlyMap<GoalId, MemoryDisclosureLabel>;
};

export type ResolveCommitmentGoalDisclosureInput = {
  commitments?: readonly CommitmentRecord[];
  goals?: readonly GoalRecord[];
  goalTrees?: readonly GoalTreeNode[];
};

export type ResolvedCommitmentGoalDisclosure = {
  commitments: DisclosureResolvedCommitment[];
  goals: DisclosureResolvedGoal[];
  goalTrees: DisclosureResolvedGoalTreeNode[];
  commitmentLabelsById: ReadonlyMap<CommitmentId, MemoryDisclosureLabel>;
  goalLabelsById: ReadonlyMap<GoalId, MemoryDisclosureLabel>;
};

type ResolvedSource = {
  entry: StreamEntry;
  indexRecord: StreamEntryIndexRecord;
};

type SourceResolutionContext = {
  sourcesById: ReadonlyMap<string, ResolvedSource>;
  entityIdsByAudienceLabel: ReadonlyMap<string, EntityId | null>;
  audienceEntityIdsBySessionId: ReadonlyMap<SessionId, EntityId | null>;
};

function compareMachineIds(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function compareSourcesChronologically(left: ResolvedSource, right: ResolvedSource): number {
  const timestampDifference = left.indexRecord.timestamp - right.indexRecord.timestamp;
  if (timestampDifference !== 0) {
    return timestampDifference;
  }

  const sessionDifference = compareMachineIds(
    left.indexRecord.session_id,
    right.indexRecord.session_id,
  );
  if (sessionDifference !== 0) {
    return sessionDifference;
  }

  const leftEntryIndex = left.indexRecord.entry_index;
  const rightEntryIndex = right.indexRecord.entry_index;
  if (leftEntryIndex !== rightEntryIndex) {
    if (leftEntryIndex === null) {
      return 1;
    }
    if (rightEntryIndex === null) {
      return -1;
    }
    return leftEntryIndex - rightEntryIndex;
  }

  return compareMachineIds(left.indexRecord.entry_id, right.indexRecord.entry_id);
}

function labelWithResolvedOrigin(
  sourceStreamEntryIds: readonly StreamEntryId[] | undefined,
  fallback: MemoryDisclosureLabel,
  context: SourceResolutionContext,
): MemoryDisclosureLabel {
  if (sourceStreamEntryIds === undefined || sourceStreamEntryIds.length === 0) {
    return fallback;
  }

  const sources: ResolvedSource[] = [];

  for (const sourceStreamEntryId of dedupePreservingOrder(sourceStreamEntryIds)) {
    const source = context.sourcesById.get(sourceStreamEntryId);
    if (source === undefined) {
      return fallback;
    }
    sources.push(source);
  }

  sources.sort(compareSourcesChronologically);
  const originAudienceEntityIds: EntityId[] = [];

  for (const source of sources) {
    const audienceEntityId =
      source.entry.audience === undefined
        ? context.audienceEntityIdsBySessionId.get(source.entry.session_id)
        : context.entityIdsByAudienceLabel.get(source.entry.audience);

    if (audienceEntityId === null || audienceEntityId === undefined) {
      return fallback;
    }

    originAudienceEntityIds.push(audienceEntityId);
  }

  return {
    ...fallback,
    originAudienceEntityIds: dedupePreservingOrder(originAudienceEntityIds),
  };
}

export class SourceStreamAudienceDisclosureResolver {
  constructor(private readonly options: SourceStreamAudienceDisclosureResolverOptions) {}

  resolveLabels(
    input: ResolveCommitmentGoalDisclosureLabelsInput,
  ): ResolvedCommitmentGoalDisclosureLabels {
    const commitments = input.commitments ?? [];
    const goals = input.goals ?? [];
    const sourceStreamEntryIds = dedupePreservingOrder([
      ...commitments.flatMap((commitment) => commitment.source_stream_entry_ids ?? []),
      ...goals.flatMap((goal) => goal.source_stream_entry_ids ?? []),
    ]);
    const indexedSources = this.options.entryIndex.lookupMany(sourceStreamEntryIds);
    const sourcesById = new Map<string, ResolvedSource>();

    for (const sourceStreamEntryId of sourceStreamEntryIds) {
      const indexRecord = indexedSources.get(sourceStreamEntryId);
      if (indexRecord === undefined) {
        continue;
      }

      const entry = readStreamEntryAtOffset({
        dataDir: this.options.dataDir,
        sessionId: indexRecord.session_id,
        byteOffset: indexRecord.byte_offset,
      });
      if (entry?.id !== sourceStreamEntryId || entry.session_id !== indexRecord.session_id) {
        continue;
      }

      sourcesById.set(sourceStreamEntryId, { entry, indexRecord });
    }

    const audienceLabels = dedupePreservingOrder(
      [...sourcesById.values()].flatMap(({ entry }) =>
        entry.audience === undefined ? [] : [entry.audience],
      ),
    );
    const entityIdsByAudienceLabel =
      audienceLabels.length === 0
        ? new Map<string, EntityId | null>()
        : this.options.entityRepository.findByNames(audienceLabels);
    const sessionIds = dedupePreservingOrder(
      [...sourcesById.values()].flatMap(({ entry }) =>
        entry.audience === undefined ? [entry.session_id] : [],
      ),
    );
    const sessionsById = new Map(
      (sessionIds.length === 0 ? [] : this.options.sessionsRepository.getMany(sessionIds)).map(
        (session) => [session.session_id, session.audience_entity_id],
      ),
    );
    const context: SourceResolutionContext = {
      sourcesById,
      entityIdsByAudienceLabel,
      audienceEntityIdsBySessionId: sessionsById,
    };
    const commitmentLabelsById = new Map<CommitmentId, MemoryDisclosureLabel>();
    const goalLabelsById = new Map<GoalId, MemoryDisclosureLabel>();
    const commitmentLabels: MemoryDisclosureLabel[] = [];
    const goalLabels: MemoryDisclosureLabel[] = [];

    for (const commitment of commitments) {
      const label = labelWithResolvedOrigin(
        commitment.source_stream_entry_ids,
        commitmentScopeMemoryDisclosureLabel(commitment),
        context,
      );
      commitmentLabels.push(label);
      commitmentLabelsById.set(commitment.id, label);
    }

    for (const goal of goals) {
      const label = labelWithResolvedOrigin(
        goal.source_stream_entry_ids,
        goalScopeMemoryDisclosureLabel(goal),
        context,
      );
      goalLabels.push(label);
      goalLabelsById.set(goal.id, label);
    }

    return { commitmentLabels, goalLabels, commitmentLabelsById, goalLabelsById };
  }

  resolve(input: ResolveCommitmentGoalDisclosureInput): ResolvedCommitmentGoalDisclosure {
    const commitments = [...(input.commitments ?? [])];
    const goalTrees = [...(input.goalTrees ?? [])];
    const goalsById = new Map<GoalId, GoalRecord>();

    for (const goal of [...(input.goals ?? []), ...flattenGoalTree(goalTrees)]) {
      if (!goalsById.has(goal.id)) {
        goalsById.set(goal.id, goal);
      }
    }
    const goals = [...goalsById.values()];
    const { commitmentLabelsById, goalLabelsById } = this.resolveLabels({ commitments, goals });

    const decorateCommitment = (commitment: CommitmentRecord): DisclosureResolvedCommitment => ({
      ...commitment,
      disclosure_label: memoryDisclosureLabelMetadata(
        commitmentLabelsById.get(commitment.id) ?? commitmentScopeMemoryDisclosureLabel(commitment),
      ),
    });
    const decorateGoal = (goal: GoalRecord): DisclosureResolvedGoal => ({
      ...goal,
      disclosure_label: memoryDisclosureLabelMetadata(
        goalLabelsById.get(goal.id) ?? goalScopeMemoryDisclosureLabel(goal),
      ),
    });
    const decorateGoalTree = (goal: GoalTreeNode): DisclosureResolvedGoalTreeNode => ({
      ...decorateGoal(goal),
      children: goal.children.map(decorateGoalTree),
    });

    return {
      commitments: commitments.map(decorateCommitment),
      goals: (input.goals ?? []).map(decorateGoal),
      goalTrees: goalTrees.map(decorateGoalTree),
      commitmentLabelsById,
      goalLabelsById,
    };
  }
}
