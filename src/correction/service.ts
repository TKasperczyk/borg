import { ConfigError, StorageError } from "../util/errors.js";
import type { JsonValue } from "../util/json-value.js";
import {
  parseCommitmentId,
  parseEpisodeId,
  parseGoalId,
  parseOpenQuestionId,
  parseSemanticEdgeId,
  parseSemanticNodeId,
  parseTraitId,
  parseValueId,
  type EntityId,
} from "../util/ids.js";
import { SystemClock, type Clock } from "../util/clock.js";
import type { Config } from "../config/index.js";
import {
  parseReviewProvenance,
  provenanceSchema,
  type Provenance,
} from "../memory/common/provenance.js";
import { type EpisodePatch, type EpisodicRepository } from "../memory/episodic/index.js";
import {
  type IdentityEventRepository,
  type IdentityRecordType,
  type IdentityService,
} from "../memory/identity/index.js";
import type {
  CommitmentRecord,
  CommitmentRepository,
  EntityRecord,
  EntityRepository,
} from "../memory/commitments/index.js";
import {
  type ReviewQueueItem,
  type ReviewQueueRepository,
  semanticNodePatchSchema,
  type SemanticEdge,
  type SemanticEdgeRepository,
  type SemanticGraph,
  type SemanticNodeRepository,
} from "../memory/semantic/index.js";
import type { RetrievalPipeline } from "../retrieval/index.js";
import type { SocialRepository } from "../memory/social/index.js";
import {
  GoalsRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
} from "../memory/self/index.js";
import { episodePatchSchema } from "../memory/episodic/types.js";
import { openQuestionPatchSchema, type OpenQuestionPatch } from "../memory/self/open-questions.js";

const MANUAL_PROVENANCE = {
  kind: "manual" as const,
};

type CorrectionTargetType =
  | "episode"
  | "semantic_node"
  | "semantic_edge"
  | "value"
  | "goal"
  | "trait"
  | "commitment"
  | "open_question";

type ParsedCorrectionTarget =
  | { type: "episode"; id: ReturnType<typeof parseEpisodeId> }
  | { type: "semantic_node"; id: ReturnType<typeof parseSemanticNodeId> }
  | { type: "semantic_edge"; id: ReturnType<typeof parseSemanticEdgeId> }
  | { type: "value"; id: ReturnType<typeof parseValueId> }
  | { type: "goal"; id: ReturnType<typeof parseGoalId> }
  | { type: "trait"; id: ReturnType<typeof parseTraitId> }
  | { type: "commitment"; id: ReturnType<typeof parseCommitmentId> }
  | { type: "open_question"; id: ReturnType<typeof parseOpenQuestionId> };

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseTarget(id: string): ParsedCorrectionTarget {
  if (id.startsWith("ep_")) {
    return { type: "episode", id: parseEpisodeId(id) };
  }
  if (id.startsWith("semn_")) {
    return { type: "semantic_node", id: parseSemanticNodeId(id) };
  }
  if (id.startsWith("seme_")) {
    return { type: "semantic_edge", id: parseSemanticEdgeId(id) };
  }
  if (id.startsWith("val_")) {
    return { type: "value", id: parseValueId(id) };
  }
  if (id.startsWith("goal_")) {
    return { type: "goal", id: parseGoalId(id) };
  }
  if (id.startsWith("trt_")) {
    return { type: "trait", id: parseTraitId(id) };
  }
  if (id.startsWith("cmt_")) {
    return { type: "commitment", id: parseCommitmentId(id) };
  }
  if (id.startsWith("oq_")) {
    return { type: "open_question", id: parseOpenQuestionId(id) };
  }

  throw new StorageError(`Unsupported correction target id: ${id}`, {
    code: "CORRECTION_TARGET_UNSUPPORTED",
  });
}

function truncate(text: string, max = 120): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  return normalized.length <= max ? normalized : `${normalized.slice(0, max - 3).trimEnd()}...`;
}

function toIdentityJsonValue(value: unknown): JsonValue {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }

  if (ArrayBuffer.isView(value)) {
    return Array.from(value as unknown as ArrayLike<number>).map((entry) => Number(entry));
  }

  if (Array.isArray(value)) {
    return value.map((entry) => toIdentityJsonValue(entry));
  }

  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, toIdentityJsonValue(entry)]),
    );
  }

  return String(value);
}

function jsonValuesEqual(left: JsonValue, right: JsonValue): boolean {
  if (left === right) {
    return true;
  }

  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
      return false;
    }

    return left.every((entry, index) => jsonValuesEqual(entry, right[index] as JsonValue));
  }

  if (isRecord(left) || isRecord(right)) {
    if (!isRecord(left) || !isRecord(right)) {
      return false;
    }

    const leftKeys = Object.keys(left).sort();
    const rightKeys = Object.keys(right).sort();

    return (
      leftKeys.length === rightKeys.length &&
      leftKeys.every(
        (key, index) =>
          key === rightKeys[index] &&
          jsonValuesEqual(left[key] as JsonValue, right[key] as JsonValue),
      )
    );
  }

  return false;
}

function patchValueAlreadyApplied(currentValue: unknown, patchValue: unknown): boolean {
  if (patchValue === undefined) {
    return true;
  }

  if (Array.isArray(patchValue)) {
    if (!Array.isArray(currentValue)) {
      return false;
    }

    const currentEntries = currentValue.map((entry) => toIdentityJsonValue(entry));

    return patchValue.every((entry) =>
      currentEntries.some((currentEntry) =>
        jsonValuesEqual(currentEntry, toIdentityJsonValue(entry)),
      ),
    );
  }

  if (isRecord(patchValue)) {
    if (!isRecord(currentValue)) {
      return false;
    }

    return Object.entries(patchValue).every(([key, value]) =>
      patchValueAlreadyApplied(currentValue[key], value),
    );
  }

  return jsonValuesEqual(toIdentityJsonValue(currentValue), toIdentityJsonValue(patchValue));
}

function recordAlreadyMatchesPatch(
  current: Record<string, unknown>,
  patch: Record<string, unknown>,
): boolean {
  return Object.entries(patch).every(([key, value]) =>
    patchValueAlreadyApplied(current[key], value),
  );
}

function summarizePatch(patch: Record<string, unknown>): string {
  const entries = Object.entries(patch).slice(0, 3);

  if (entries.length === 0) {
    return "no changes";
  }

  const summary = entries
    .map(([key, value]) => `${key}=${truncate(JSON.stringify(value))}`)
    .join(", ");

  return Object.keys(patch).length > entries.length ? `${summary}, ...` : summary;
}

function reviewPromptSummary(
  targetType: CorrectionTargetType,
  targetLabel: string,
  patch: Record<string, unknown>,
): string {
  return `user proposed changing ${targetType} ${targetLabel} to ${summarizePatch(patch)} (review pending)`;
}

export type CorrectionServiceOptions = {
  config: Config;
  clock?: Clock;
  retrievalPipeline: RetrievalPipeline;
  episodicRepository: EpisodicRepository;
  semanticNodeRepository: SemanticNodeRepository;
  semanticEdgeRepository: SemanticEdgeRepository;
  semanticGraph: SemanticGraph;
  valuesRepository: ValuesRepository;
  goalsRepository: GoalsRepository;
  traitsRepository: TraitsRepository;
  openQuestionsRepository: OpenQuestionsRepository;
  socialRepository: SocialRepository;
  entityRepository: EntityRepository;
  commitmentRepository: CommitmentRepository;
  reviewQueueRepository: ReviewQueueRepository;
  identityService: IdentityService;
  identityEventRepository: IdentityEventRepository;
};

export class CorrectionService {
  private readonly clock: Clock;

  constructor(private readonly options: CorrectionServiceOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private async resolveTargetMetadata(target: ParsedCorrectionTarget): Promise<{
    targetLabel: string;
    audienceEntityId: EntityId | null;
  }> {
    switch (target.type) {
      case "episode": {
        const episode = await this.options.episodicRepository.get(target.id);

        if (episode === null) {
          throw new StorageError(`Unknown episode id: ${target.id}`, {
            code: "EPISODE_NOT_FOUND",
          });
        }

        return {
          targetLabel: episode.title,
          audienceEntityId: episode.audience_entity_id ?? null,
        };
      }
      case "semantic_node": {
        const node = await this.options.semanticNodeRepository.get(target.id);

        if (node === null) {
          throw new StorageError(`Unknown semantic node id: ${target.id}`, {
            code: "SEMANTIC_NODE_NOT_FOUND",
          });
        }

        return {
          targetLabel: node.label,
          audienceEntityId: null,
        };
      }
      case "semantic_edge": {
        const edge = this.options.semanticEdgeRepository.getEdge(target.id);

        if (edge === null) {
          throw new StorageError(`Unknown semantic edge id: ${target.id}`, {
            code: "SEMANTIC_EDGE_NOT_FOUND",
          });
        }

        return {
          targetLabel: `${edge.relation} ${edge.from_node_id} -> ${edge.to_node_id}`,
          audienceEntityId: null,
        };
      }
      case "value": {
        const record = this.options.valuesRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown value id: ${target.id}`, {
            code: "VALUE_NOT_FOUND",
          });
        }

        return {
          targetLabel: record.label,
          audienceEntityId: null,
        };
      }
      case "goal": {
        const record = this.options.goalsRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown goal id: ${target.id}`, {
            code: "GOAL_NOT_FOUND",
          });
        }

        return {
          targetLabel: truncate(record.description),
          audienceEntityId: null,
        };
      }
      case "trait": {
        const record = this.options.traitsRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown trait id: ${target.id}`, {
            code: "TRAIT_NOT_FOUND",
          });
        }

        return {
          targetLabel: record.label,
          audienceEntityId: null,
        };
      }
      case "commitment": {
        const record = this.options.commitmentRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown commitment id: ${target.id}`, {
            code: "COMMITMENT_NOT_FOUND",
          });
        }

        return {
          targetLabel: truncate(record.directive),
          audienceEntityId: record.restricted_audience,
        };
      }
      case "open_question": {
        const record = this.options.openQuestionsRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown open question id: ${target.id}`, {
            code: "OPEN_QUESTION_NOT_FOUND",
          });
        }

        return {
          targetLabel: truncate(record.question),
          audienceEntityId: null,
        };
      }
    }
  }

  async forget(id: string): Promise<{
    id: string;
    target_type: CorrectionTargetType;
    archived: true;
    provenance: typeof MANUAL_PROVENANCE;
  }> {
    const target = parseTarget(id);
    const reason = "forgotten manually";

    switch (target.type) {
      case "episode": {
        const episode = await this.options.episodicRepository.get(target.id);

        if (episode === null) {
          throw new StorageError(`Unknown episode id: ${target.id}`, {
            code: "EPISODE_NOT_FOUND",
          });
        }

        const previousStats = this.options.episodicRepository.getStats(target.id);
        const nextStats = this.options.episodicRepository.updateStats(target.id, {
          archived: true,
        });
        this.options.identityEventRepository.record({
          record_type: "episode",
          record_id: target.id,
          action: "forget",
          old_value: toIdentityJsonValue({
            episode,
            stats: previousStats,
          }),
          new_value: toIdentityJsonValue({
            episode,
            stats: nextStats,
          }),
          reason,
          provenance: MANUAL_PROVENANCE,
        });
        break;
      }
      case "semantic_node": {
        const current = await this.options.semanticNodeRepository.get(target.id);

        if (current === null) {
          throw new StorageError(`Unknown semantic node id: ${target.id}`, {
            code: "SEMANTIC_NODE_NOT_FOUND",
          });
        }

        const next = await this.options.semanticNodeRepository.update(target.id, {
          archived: true,
        });

        this.options.identityEventRepository.record({
          record_type: "semantic_node",
          record_id: target.id,
          action: "forget",
          old_value: toIdentityJsonValue(current),
          new_value: next === null ? null : toIdentityJsonValue(next),
          reason,
          provenance: MANUAL_PROVENANCE,
        });
        break;
      }
      case "semantic_edge": {
        throw new StorageError(
          `Semantic edges are revoked with semantic edge invalidate: ${target.id}`,
          {
            code: "SEMANTIC_EDGE_FORGET_UNSUPPORTED",
          },
        );
      }
      case "value": {
        const current = this.options.valuesRepository.get(target.id);

        if (current === null) {
          throw new StorageError(`Unknown value id: ${target.id}`, {
            code: "VALUE_NOT_FOUND",
          });
        }

        this.options.valuesRepository.remove(target.id);
        this.options.identityEventRepository.record({
          record_type: "value",
          record_id: target.id,
          action: "forget",
          old_value: current,
          new_value: null,
          reason,
          provenance: MANUAL_PROVENANCE,
        });
        break;
      }
      case "goal": {
        const current = this.options.goalsRepository.get(target.id);

        if (current === null) {
          throw new StorageError(`Unknown goal id: ${target.id}`, {
            code: "GOAL_NOT_FOUND",
          });
        }

        this.options.goalsRepository.remove(target.id);
        this.options.identityEventRepository.record({
          record_type: "goal",
          record_id: target.id,
          action: "forget",
          old_value: current,
          new_value: null,
          reason,
          provenance: MANUAL_PROVENANCE,
        });
        break;
      }
      case "trait": {
        const current = this.options.traitsRepository.get(target.id);

        if (current === null) {
          throw new StorageError(`Unknown trait id: ${target.id}`, {
            code: "TRAIT_NOT_FOUND",
          });
        }

        this.options.traitsRepository.remove(target.id);
        this.options.identityEventRepository.record({
          record_type: "trait",
          record_id: target.id,
          action: "forget",
          old_value: current,
          new_value: null,
          reason,
          provenance: MANUAL_PROVENANCE,
        });
        break;
      }
      case "commitment": {
        const current = this.options.commitmentRepository.get(target.id);

        if (current === null) {
          throw new StorageError(`Unknown commitment id: ${target.id}`, {
            code: "COMMITMENT_NOT_FOUND",
          });
        }

        this.options.commitmentRepository.revoke(target.id, reason, MANUAL_PROVENANCE);
        break;
      }
      case "open_question": {
        const current = this.options.openQuestionsRepository.get(target.id);

        if (current === null) {
          throw new StorageError(`Unknown open question id: ${target.id}`, {
            code: "OPEN_QUESTION_NOT_FOUND",
          });
        }

        const next =
          current.status === "open"
            ? this.options.openQuestionsRepository.abandon(target.id, reason)
            : current;
        this.options.identityEventRepository.record({
          record_type: "open_question",
          record_id: target.id,
          action: "forget",
          old_value: current,
          new_value: next,
          reason,
          provenance: MANUAL_PROVENANCE,
        });
        break;
      }
    }

    return {
      id,
      target_type: target.type,
      archived: true,
      provenance: MANUAL_PROVENANCE,
    };
  }

  async why(id: string): Promise<Record<string, unknown>> {
    const target = parseTarget(id);

    switch (target.type) {
      case "episode": {
        const result = await this.options.retrievalPipeline.getEpisode(target.id, {
          crossAudience: true,
        });
        const episode = result?.episode ?? (await this.options.episodicRepository.get(target.id));

        if (episode === null) {
          throw new StorageError(`Unknown episode id: ${target.id}`, {
            code: "EPISODE_NOT_FOUND",
          });
        }

        return {
          target_type: "episode",
          record: episode,
          source_stream_ids: episode.source_stream_ids,
          citation_chain: result?.citationChain ?? [],
        };
      }
      case "semantic_node": {
        const node = await this.options.semanticNodeRepository.get(target.id);

        if (node === null) {
          throw new StorageError(`Unknown semantic node id: ${target.id}`, {
            code: "SEMANTIC_NODE_NOT_FOUND",
          });
        }

        return {
          target_type: "semantic_node",
          record: node,
          direct_edges: [
            ...this.options.semanticEdgeRepository.listEdges({ fromId: target.id }),
            ...this.options.semanticEdgeRepository.listEdges({ toId: target.id }),
          ],
          walked_edges: await this.options.semanticGraph.walk(target.id, {
            depth: 2,
          }),
        };
      }
      case "semantic_edge": {
        const edge = this.options.semanticEdgeRepository.getEdge(target.id);

        if (edge === null) {
          throw new StorageError(`Unknown semantic edge id: ${target.id}`, {
            code: "SEMANTIC_EDGE_NOT_FOUND",
          });
        }

        return {
          target_type: "semantic_edge",
          record: edge,
          from_node: await this.options.semanticNodeRepository.get(edge.from_node_id),
          to_node: await this.options.semanticNodeRepository.get(edge.to_node_id),
        };
      }
      case "value": {
        const record = this.options.valuesRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown value id: ${target.id}`, {
            code: "VALUE_NOT_FOUND",
          });
        }

        return {
          target_type: "value",
          record,
          reinforcement_events: this.options.valuesRepository.listReinforcementEvents(target.id),
          identity_events: this.options.identityService.listEvents({
            recordType: "value",
            recordId: target.id,
          }),
        };
      }
      case "goal": {
        const record = this.options.goalsRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown goal id: ${target.id}`, {
            code: "GOAL_NOT_FOUND",
          });
        }

        return {
          target_type: "goal",
          record,
          identity_events: this.options.identityService.listEvents({
            recordType: "goal",
            recordId: target.id,
          }),
        };
      }
      case "trait": {
        const record = this.options.traitsRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown trait id: ${target.id}`, {
            code: "TRAIT_NOT_FOUND",
          });
        }

        return {
          target_type: "trait",
          record,
          reinforcement_events: this.options.traitsRepository.listReinforcementEvents(target.id),
          identity_events: this.options.identityService.listEvents({
            recordType: "trait",
            recordId: target.id,
          }),
        };
      }
      case "commitment": {
        const record = this.options.commitmentRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown commitment id: ${target.id}`, {
            code: "COMMITMENT_NOT_FOUND",
          });
        }

        return {
          target_type: "commitment",
          record,
          identity_events: this.options.identityService.listEvents({
            recordType: "commitment",
            recordId: target.id,
          }),
        };
      }
      case "open_question": {
        const record = this.options.openQuestionsRepository.get(target.id);

        if (record === null) {
          throw new StorageError(`Unknown open question id: ${target.id}`, {
            code: "OPEN_QUESTION_NOT_FOUND",
          });
        }

        return {
          target_type: "open_question",
          record,
          identity_events: this.options.identityService.listEvents({
            recordType: "open_question",
            recordId: target.id,
          }),
        };
      }
    }
  }

  invalidateSemanticEdge(
    id: string,
    options: {
      at?: number;
      reason?: string;
    } = {},
  ): SemanticEdge {
    const target = parseTarget(id);

    if (target.type !== "semantic_edge") {
      throw new StorageError(`Expected semantic edge id: ${id}`, {
        code: "SEMANTIC_EDGE_ID_REQUIRED",
      });
    }

    const current = this.options.semanticEdgeRepository.getEdge(target.id);

    if (current === null) {
      throw new StorageError(`Unknown semantic edge id: ${target.id}`, {
        code: "SEMANTIC_EDGE_NOT_FOUND",
      });
    }

    const at = options.at ?? this.clock.now();

    if (!Number.isFinite(at)) {
      throw new StorageError("Semantic edge invalidation time must be finite", {
        code: "SEMANTIC_EDGE_INVALIDATION_TIME_INVALID",
      });
    }

    const reason =
      typeof options.reason === "string" && options.reason.trim().length > 0
        ? options.reason.trim()
        : undefined;
    const next = this.options.semanticEdgeRepository.invalidateEdge(target.id, {
      at,
      by_process: "manual",
      reason,
    });

    if (next === null) {
      throw new StorageError(`Unknown semantic edge id: ${target.id}`, {
        code: "SEMANTIC_EDGE_NOT_FOUND",
      });
    }

    if (current.valid_to === null) {
      this.options.identityEventRepository.record({
        record_type: "semantic_edge",
        record_id: target.id,
        action: "edge_invalidate",
        old_value: toIdentityJsonValue({
          edge_id: current.id,
          prior_valid_to: current.valid_to,
        }),
        new_value: toIdentityJsonValue({
          edge_id: next.id,
          prior_valid_to: current.valid_to,
          new_valid_to: next.valid_to,
          by_process: next.invalidated_by_process,
          by_review_id: next.invalidated_by_review_id,
          reason: next.invalidated_reason,
          by_edge_id: next.invalidated_by_edge_id,
        }),
        reason: next.invalidated_reason,
        provenance: MANUAL_PROVENANCE,
      });
    }

    return next;
  }

  async correct(
    id: string,
    patch: Record<string, unknown>,
    provenance: Provenance = MANUAL_PROVENANCE,
  ): Promise<ReviewQueueItem> {
    if (!isRecord(patch)) {
      throw new StorageError("Correction patch must be a JSON object", {
        code: "CORRECTION_PATCH_INVALID",
      });
    }

    const target = parseTarget(id);

    if (target.type === "semantic_edge") {
      throw new StorageError(
        `Semantic edge corrections are applied with semantic edge invalidate: ${id}`,
        {
          code: "SEMANTIC_EDGE_CORRECTION_UNSUPPORTED",
        },
      );
    }

    const metadata = await this.resolveTargetMetadata(target);
    const createdAtIso = new Date().toISOString();

    return this.options.reviewQueueRepository.enqueue({
      kind: "correction",
      refs: {
        target_id: id,
        target_type: target.type,
        patch,
        proposed_provenance: provenanceSchema.parse(provenance),
        audience_entity_id: metadata.audienceEntityId,
        prompt_summary: reviewPromptSummary(target.type, metadata.targetLabel, patch),
      },
      reason: `user corrected ${id} at ${createdAtIso}`,
    });
  }

  private resolveRememberEntity(entity: string | undefined): {
    entityId: EntityId;
    entityName: string;
  } {
    const entityName = entity?.trim() || this.options.config.defaultUser?.trim();

    if (entityName === undefined || entityName.length === 0) {
      throw new ConfigError("defaultUser is not configured; specify --entity explicitly", {
        code: "DEFAULT_USER_REQUIRED",
      });
    }

    return {
      entityId: this.options.entityRepository.resolve(entityName),
      entityName,
    };
  }

  async rememberAboutMe(options: { entity?: string } = {}): Promise<{
    entity: EntityRecord | null;
    social_profile: ReturnType<SocialRepository["getProfile"]>;
    active_commitments: CommitmentRecord[];
    scoped_episodes: Awaited<ReturnType<EpisodicRepository["listByAudience"]>>;
    related_episodes: Awaited<ReturnType<EpisodicRepository["searchByParticipantsOrTags"]>>;
  }> {
    const { entityId, entityName } = this.resolveRememberEntity(options.entity);
    const entityRecord = this.options.entityRepository.get(entityId);
    const resolvedEntity = entityRecord ?? {
      id: entityId,
      canonical_name: entityName,
      aliases: [],
      created_at: Date.now(),
    };
    const names = [resolvedEntity.canonical_name, ...resolvedEntity.aliases];
    const socialProfile = this.options.socialRepository.recomputeCommitmentCount(
      entityId,
      this.options.commitmentRepository,
    );
    const activeCommitments = this.options.commitmentRepository
      .list({
        activeOnly: true,
      })
      .filter(
        (commitment) =>
          commitment.made_to_entity === entityId ||
          commitment.restricted_audience === entityId ||
          commitment.about_entity === entityId,
      );

    return {
      entity: entityRecord,
      social_profile: socialProfile,
      active_commitments: activeCommitments,
      scoped_episodes: await this.options.episodicRepository.listByAudience(entityId, {
        orderBy: "recent",
        limit: 10,
      }),
      related_episodes: await this.options.episodicRepository.searchByParticipantsOrTags(names, {
        crossAudience: true,
        limit: 10,
      }),
    };
  }

  listIdentityEvents(
    options?: Parameters<IdentityEventRepository["list"]>[0],
  ): ReturnType<IdentityEventRepository["list"]> {
    return this.options.identityEventRepository.list(options);
  }

  async applyCorrectionReview(item: ReviewQueueItem): Promise<void> {
    const targetId = typeof item.refs.target_id === "string" ? item.refs.target_id : null;
    const patch = item.refs.patch;
    const proposedProvenance = parseReviewProvenance(item.refs);

    if (targetId === null) {
      throw new StorageError("Correction review item is missing target_id", {
        code: "REVIEW_QUEUE_INVALID",
      });
    }

    if (!isRecord(patch)) {
      throw new StorageError("Correction review item is missing an object patch", {
        code: "REVIEW_QUEUE_INVALID",
      });
    }

    const target = parseTarget(targetId);

    switch (target.type) {
      case "episode": {
        const current = await this.options.episodicRepository.get(target.id);

        if (current === null) {
          throw new StorageError(`Unknown episode id: ${target.id}`, {
            code: "EPISODE_NOT_FOUND",
          });
        }

        const parsedPatch = episodePatchSchema.parse(patch) as EpisodePatch;

        if (recordAlreadyMatchesPatch(current, parsedPatch as Record<string, unknown>)) {
          return;
        }

        const next = await this.options.episodicRepository.update(target.id, parsedPatch);
        const existingEvent = this.options.identityEventRepository.findByReviewKey({
          reviewItemId: item.id,
          recordType: "episode",
          recordId: target.id,
          action: "correction_apply",
        });

        if (existingEvent === null) {
          this.options.identityEventRepository.record({
            record_type: "episode",
            record_id: target.id,
            action: "correction_apply",
            old_value: toIdentityJsonValue(current),
            new_value: next === null ? null : toIdentityJsonValue(next),
            reason: item.reason,
            provenance: proposedProvenance,
            review_item_id: item.id,
          });
        }
        return;
      }
      case "semantic_node": {
        const current = await this.options.semanticNodeRepository.get(target.id);

        if (current === null) {
          throw new StorageError(`Unknown semantic node id: ${target.id}`, {
            code: "SEMANTIC_NODE_NOT_FOUND",
          });
        }

        const parsedPatch = semanticNodePatchSchema.parse(patch);

        if (recordAlreadyMatchesPatch(current, parsedPatch as Record<string, unknown>)) {
          return;
        }

        const next = await this.options.semanticNodeRepository.update(target.id, parsedPatch);
        const existingEvent = this.options.identityEventRepository.findByReviewKey({
          reviewItemId: item.id,
          recordType: "semantic_node",
          recordId: target.id,
          action: "correction_apply",
        });

        if (existingEvent === null) {
          this.options.identityEventRepository.record({
            record_type: "semantic_node",
            record_id: target.id,
            action: "correction_apply",
            old_value: toIdentityJsonValue(current),
            new_value: next === null ? null : toIdentityJsonValue(next),
            reason: item.reason,
            provenance: proposedProvenance,
            review_item_id: item.id,
          });
        }
        return;
      }
      case "semantic_edge": {
        throw new StorageError(
          `Semantic edge corrections are applied with semantic edge invalidate: ${target.id}`,
          {
            code: "SEMANTIC_EDGE_CORRECTION_UNSUPPORTED",
          },
        );
      }
      case "value": {
        const result = this.options.identityService.updateValue(
          target.id,
          patch,
          proposedProvenance,
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );

        if (result.status !== "applied") {
          throw new StorageError(`Correction for value ${target.id} still requires review`, {
            code: "IDENTITY_REVIEW_REQUIRED",
          });
        }

        return;
      }
      case "goal": {
        const next = this.options.identityService.updateGoal(target.id, patch, proposedProvenance, {
          throughReview: true,
          reason: item.reason,
          reviewItemId: item.id,
        });

        if (next.status !== "applied") {
          throw new StorageError(`Correction for goal ${target.id} still requires review`, {
            code: "IDENTITY_REVIEW_REQUIRED",
          });
        }

        return;
      }
      case "trait": {
        const result = this.options.identityService.updateTrait(
          target.id,
          patch,
          proposedProvenance,
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );

        if (result.status !== "applied") {
          throw new StorageError(`Correction for trait ${target.id} still requires review`, {
            code: "IDENTITY_REVIEW_REQUIRED",
          });
        }

        return;
      }
      case "commitment": {
        const result = this.options.identityService.updateCommitment(
          target.id,
          patch,
          proposedProvenance,
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );

        if (result.status !== "applied") {
          throw new StorageError(`Correction for commitment ${target.id} still requires review`, {
            code: "IDENTITY_REVIEW_REQUIRED",
          });
        }

        return;
      }
      case "open_question": {
        const next = this.options.identityService.updateOpenQuestion(
          target.id,
          openQuestionPatchSchema.parse(patch) as OpenQuestionPatch,
          proposedProvenance,
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );

        if (next.status !== "applied") {
          throw new StorageError(
            `Correction for open question ${target.id} still requires review`,
            {
              code: "IDENTITY_REVIEW_REQUIRED",
            },
          );
        }
      }
    }
  }
}
