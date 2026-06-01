import { z } from "zod";

import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createCreatorDirectiveId,
  parseCreatorDirectiveId,
  type CreatorDirectiveId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  activationPolicySchema,
  creatorDirectiveApplicableOptionsSchema,
  creatorDirectiveIdSchema,
  creatorDirectiveListFilterSchema,
  creatorDirectiveQueueInputSchema,
  creatorDirectiveSchema,
  disclosurePolicySchema,
  type ActivationPolicy,
  type CreatorDirective,
  type CreatorDirectiveActivationReason,
  type CreatorDirectiveApplicable,
  type CreatorDirectiveApplicableOptions,
  type CreatorDirectiveListFilter,
  type CreatorDirectiveQueueInput,
  type CreatorDirectiveRenderReason,
  type CreatorDirectiveRenderMode,
  type DisclosurePolicy,
} from "./types.js";

const CREATOR_DIRECTIVE_JSON_ARRAY_CODEC = {
  errorCode: "CREATOR_DIRECTIVE_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse creator directive ${label}`,
} satisfies JsonArrayCodecOptions;

function uniqueIds<T extends string>(values: readonly T[]): T[] {
  return dedupePreservingOrder(values);
}

function parseStoredArray<T extends string>(value: unknown, label: string): T[] {
  return parseJsonArray<T>(String(value ?? "[]"), label, CREATOR_DIRECTIVE_JSON_ARRAY_CODEC);
}

function fromStoredBoolean(value: unknown): boolean | null {
  if (value === null || value === undefined) {
    return null;
  }

  return Number(value) === 1;
}

function toStoredBoolean(value: boolean | null): number | null {
  return value === null ? null : value ? 1 : 0;
}

function normalizeDisclosurePolicy(policy: DisclosurePolicy): DisclosurePolicy {
  return disclosurePolicySchema.parse({
    ...policy,
    allowed_entity_ids: uniqueIds(policy.allowed_entity_ids),
    excluded_entity_ids: uniqueIds(policy.excluded_entity_ids),
    topic_tags: uniqueIds(policy.topic_tags),
  });
}

function normalizeActivationPolicy(policy: ActivationPolicy): ActivationPolicy {
  return activationPolicySchema.parse({
    ...policy,
    allowed_entity_ids: uniqueIds(policy.allowed_entity_ids),
    excluded_entity_ids: uniqueIds(policy.excluded_entity_ids),
  });
}

function mapCreatorDirectiveRow(row: Record<string, unknown>): CreatorDirective {
  const disclosurePolicy = disclosurePolicySchema.parse({
    content_scope: row.content_scope,
    allowed_entity_ids: parseStoredArray<EntityId>(row.allowed_entity_ids, "allowed_entity_ids"),
    excluded_entity_ids: parseStoredArray<EntityId>(row.excluded_entity_ids, "excluded_entity_ids"),
    subject_may_know: fromStoredBoolean(row.subject_may_know),
    mention_policy: row.mention_policy,
    denied_audience_behavior: row.denied_audience_behavior,
    boundary_prompt:
      row.boundary_prompt === null || row.boundary_prompt === undefined
        ? null
        : String(row.boundary_prompt),
    topic_tags: parseStoredArray<string>(row.topic_tags, "topic_tags"),
  });
  const activationPolicy = activationPolicySchema.parse({
    scope: row.activation_scope,
    allowed_entity_ids: parseStoredArray<EntityId>(
      row.activation_entity_ids,
      "activation_entity_ids",
    ),
    excluded_entity_ids: parseStoredArray<EntityId>(
      row.activation_excluded_entity_ids,
      "activation_excluded_entity_ids",
    ),
  });
  const parsed = creatorDirectiveSchema.safeParse({
    id: row.id,
    record_version: Number(row.record_version ?? 1),
    status: row.status,
    kind: row.kind,
    created_by_entity_id: row.created_by_entity_id,
    source_session_id: row.source_session_id,
    authorization_stream_entry_ids: parseStoredArray<StreamEntryId>(
      row.authorization_stream_entry_ids,
      "authorization_stream_entry_ids",
    ),
    content_source_stream_entry_ids: parseStoredArray<StreamEntryId>(
      row.content_source_stream_entry_ids,
      "content_source_stream_entry_ids",
    ),
    subject_kind: row.subject_kind,
    subject_entity_id:
      row.subject_entity_id === null || row.subject_entity_id === undefined
        ? null
        : String(row.subject_entity_id),
    semantic_slot:
      row.semantic_slot === null || row.semantic_slot === undefined
        ? null
        : String(row.semantic_slot),
    canonical_fact:
      row.canonical_fact === null || row.canonical_fact === undefined
        ? null
        : String(row.canonical_fact),
    operational_directive: row.operational_directive,
    disclosure_policy: disclosurePolicy,
    activation_policy: activationPolicy,
    priority: Number(row.priority),
    superseded_by:
      row.superseded_by === null || row.superseded_by === undefined
        ? null
        : parseCreatorDirectiveId(String(row.superseded_by)),
    revoked_reason:
      row.revoked_reason === null || row.revoked_reason === undefined
        ? null
        : String(row.revoked_reason),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Creator directive row failed validation", {
      cause: parsed.error,
      code: "CREATOR_DIRECTIVE_ROW_INVALID",
    });
  }

  return parsed.data;
}

function hasEntity(values: readonly EntityId[], entityId: EntityId | null): boolean {
  return entityId !== null && values.includes(entityId);
}

function excludedRecipientReason(
  excludedEntityIds: readonly EntityId[],
  recipientEntityIds: readonly EntityId[],
): Extract<
  CreatorDirectiveActivationReason,
  "explicit_exclude" | "group_contains_excluded_entity"
> | null {
  const excluded = recipientEntityIds.find((recipientEntityId) =>
    hasEntity(excludedEntityIds, recipientEntityId),
  );

  if (excluded === undefined) {
    return null;
  }

  return recipientEntityIds.length > 1 ? "group_contains_excluded_entity" : "explicit_exclude";
}

function disclosurePolicyBlocksPrivateOperation(input: {
  directive: CreatorDirective;
  recipientEntityIds: readonly EntityId[];
}): boolean {
  const policy = input.directive.disclosure_policy;

  if (
    input.recipientEntityIds.some((recipientEntityId) =>
      hasEntity(policy.excluded_entity_ids, recipientEntityId),
    )
  ) {
    return true;
  }

  return (
    policy.subject_may_know === false &&
    input.directive.subject_entity_id !== null &&
    input.recipientEntityIds.some(
      (recipientEntityId) => recipientEntityId === input.directive.subject_entity_id,
    )
  );
}

type CreatorDirectiveDisclosureEvaluation = {
  render_mode: CreatorDirectiveRenderMode;
  reason: CreatorDirectiveRenderReason;
};

type CreatorDirectiveActivationEvaluation = {
  active: boolean;
  reason: CreatorDirectiveActivationReason;
};

function boundaryOrOmit(
  directive: CreatorDirective,
  boundaryReason: CreatorDirectiveRenderReason,
  omitReason: CreatorDirectiveRenderReason = "unauthorized_omit",
): CreatorDirectiveDisclosureEvaluation {
  if (directive.disclosure_policy.denied_audience_behavior !== "render_boundary_when_relevant") {
    return { render_mode: "omit", reason: omitReason };
  }

  return { render_mode: "boundary", reason: boundaryReason };
}

function isCreatorOperator(
  directive: CreatorDirective,
  options: CreatorDirectiveApplicableOptions,
): boolean {
  return (
    options.sessionRole === "operator" &&
    (options.currentSenderBorgRole === "creator" ||
      options.currentAudienceEntityId === directive.created_by_entity_id)
  );
}

function evaluateDisclosureRenderMode(
  directive: CreatorDirective,
  options: CreatorDirectiveApplicableOptions,
  audienceId: EntityId | null = options.currentAudienceEntityId,
): CreatorDirectiveDisclosureEvaluation {
  const policy = directive.disclosure_policy;

  if (hasEntity(policy.excluded_entity_ids, audienceId)) {
    return boundaryOrOmit(directive, "explicit_exclude_boundary");
  }

  if (
    audienceId !== null &&
    directive.subject_entity_id !== null &&
    policy.subject_may_know === false &&
    audienceId === directive.subject_entity_id
  ) {
    return boundaryOrOmit(directive, "subject_may_not_know", "subject_may_not_know");
  }

  if (policy.content_scope === "public") {
    return { render_mode: "content", reason: "public" };
  }

  if (policy.content_scope === "allow_list") {
    if (hasEntity(policy.allowed_entity_ids, audienceId)) {
      return { render_mode: "content", reason: "explicit_allow" };
    }

    return { render_mode: "omit", reason: "unauthorized_omit" };
  }

  if (
    policy.content_scope === "subject_only" &&
    audienceId !== null &&
    audienceId === directive.subject_entity_id
  ) {
    return { render_mode: "content", reason: "subject_allowed" };
  }

  if (policy.content_scope === "operator_only") {
    return isCreatorOperator(directive, options)
      ? { render_mode: "content", reason: "operator_only" }
      : { render_mode: "omit", reason: "operator_only_omitted" };
  }

  if (policy.content_scope === "all_except") {
    return { render_mode: "content", reason: "public" };
  }

  return { render_mode: "omit", reason: "unauthorized_omit" };
}

function evaluateRenderMode(
  directive: CreatorDirective,
  options: CreatorDirectiveApplicableOptions,
  audienceId: EntityId | null = options.currentAudienceEntityId,
): CreatorDirectiveRenderMode {
  return evaluateDisclosureRenderMode(directive, options, audienceId).render_mode;
}

function effectiveRecipientEntityIds(
  options: CreatorDirectiveApplicableOptions,
): readonly EntityId[] {
  const participantEntityIds = options.participantEntityIds ?? [];

  if (participantEntityIds.length === 0) {
    return options.currentAudienceEntityId === null ? [] : [options.currentAudienceEntityId];
  }

  return uniqueIds(participantEntityIds);
}

function evaluateApplicableDisclosureRenderMode(
  directive: CreatorDirective,
  options: CreatorDirectiveApplicableOptions,
): CreatorDirectiveDisclosureEvaluation {
  const recipientEntityIds = effectiveRecipientEntityIds(options);

  if (recipientEntityIds.length <= 1) {
    return evaluateDisclosureRenderMode(directive, options, recipientEntityIds[0] ?? null);
  }

  const recipientEvaluations = recipientEntityIds.map((recipientEntityId) => ({
    isExcluded: hasEntity(directive.disclosure_policy.excluded_entity_ids, recipientEntityId),
    evaluation: evaluateDisclosureRenderMode(directive, options, recipientEntityId),
  }));

  const firstBoundary = recipientEvaluations.find(
    (evaluation) => evaluation.evaluation.render_mode === "boundary",
  );

  if (firstBoundary !== undefined) {
    return {
      render_mode: "boundary",
      reason: firstBoundary.isExcluded
        ? "group_contains_excluded_entity"
        : firstBoundary.evaluation.reason,
    };
  }

  const firstNonContent = recipientEvaluations.find(
    (evaluation) => evaluation.evaluation.render_mode !== "content",
  );

  if (firstNonContent !== undefined) {
    return {
      render_mode: "omit",
      reason: firstNonContent.isExcluded
        ? "group_contains_excluded_entity"
        : firstNonContent.evaluation.reason,
    };
  }

  return recipientEvaluations[0]?.evaluation ?? { render_mode: "content", reason: "public" };
}

function evaluateActivationMode(
  directive: CreatorDirective,
  options: CreatorDirectiveApplicableOptions,
  disclosure: CreatorDirectiveDisclosureEvaluation,
): CreatorDirectiveActivationEvaluation {
  const policy = directive.activation_policy;
  const recipientEntityIds = effectiveRecipientEntityIds(options);

  if (policy.scope === "same_as_disclosure") {
    return disclosure.render_mode === "omit"
      ? { active: false, reason: "same_as_disclosure_omitted" }
      : { active: true, reason: "same_as_disclosure" };
  }

  if (policy.scope === "operator_only") {
    return isCreatorOperator(directive, options)
      ? { active: true, reason: "operator_only" }
      : { active: false, reason: "operator_only_omitted" };
  }

  if (policy.scope === "public") {
    return { active: true, reason: "public" };
  }

  if (policy.scope === "allow_list") {
    const exclusionReason = excludedRecipientReason(policy.excluded_entity_ids, recipientEntityIds);

    if (exclusionReason !== null) {
      return {
        active: false,
        reason: exclusionReason,
      };
    }

    return recipientEntityIds.some((recipientEntityId) =>
      hasEntity(policy.allowed_entity_ids, recipientEntityId),
    )
      ? { active: true, reason: "explicit_allow" }
      : { active: false, reason: "unauthorized_omit" };
  }

  if (policy.scope === "subject_only") {
    return directive.subject_entity_id !== null &&
      recipientEntityIds.some(
        (recipientEntityId) => recipientEntityId === directive.subject_entity_id,
      )
      ? { active: true, reason: "subject_allowed" }
      : { active: false, reason: "subject_not_present" };
  }

  if (policy.scope === "all_except") {
    const exclusionReason = excludedRecipientReason(policy.excluded_entity_ids, recipientEntityIds);

    if (exclusionReason !== null) {
      return {
        active: false,
        reason: exclusionReason,
      };
    }

    return { active: true, reason: "all_except" };
  }

  return { active: false, reason: "unauthorized_omit" };
}

function evaluateApplicableDirective(
  directive: CreatorDirective,
  options: CreatorDirectiveApplicableOptions,
): CreatorDirectiveApplicable {
  const recipientEntityIds = effectiveRecipientEntityIds(options);
  const disclosure = evaluateApplicableDisclosureRenderMode(directive, options);
  const activation = evaluateActivationMode(directive, options, disclosure);

  return {
    directive,
    recipient_entity_ids: recipientEntityIds,
    activation,
    disclosure,
    render_mode: disclosure.render_mode,
    reason: disclosure.reason,
  };
}

export type CreatorDirectiveRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export type CreatorDirectiveFamilySupersedeInput = {
  survivorId: CreatorDirectiveId;
  expectedSurvivorVersion: number;
  losers: Array<{
    id: CreatorDirectiveId;
    expectedVersion: number;
  }>;
};

export type CreatorDirectiveFamilySupersedeResult = Array<{
  id: CreatorDirectiveId;
  record_version: number;
}>;

export type CreatorDirectiveFamilyRevokeInput = {
  losers: Array<{
    id: CreatorDirectiveId;
    expectedVersion: number;
  }>;
  reason: string;
};

export type CreatorDirectiveFamilyRevokeResult = Array<{
  id: CreatorDirectiveId;
  record_version: number;
}>;

class CreatorDirectiveFamilySupersedeAbort extends Error {}
class CreatorDirectiveFamilyRevokeAbort extends Error {}

export class CreatorDirectiveRepository {
  private readonly clock: Clock;

  constructor(private readonly options: CreatorDirectiveRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  queue(input: CreatorDirectiveQueueInput): CreatorDirective {
    const parsed = creatorDirectiveQueueInputSchema.safeParse(input);

    if (!parsed.success) {
      throw new StorageError("Invalid creator directive input", {
        cause: parsed.error,
        code: "CREATOR_DIRECTIVE_INVALID",
      });
    }

    const now = parsed.data.createdAt ?? this.clock.now();
    const semanticSlot = parsed.data.semanticSlot ?? null;
    const record = creatorDirectiveSchema.parse({
      id: parsed.data.id ?? createCreatorDirectiveId(),
      record_version: 1,
      status: "active",
      kind: parsed.data.kind,
      created_by_entity_id: parsed.data.createdByEntityId,
      source_session_id: parsed.data.sourceSessionId,
      authorization_stream_entry_ids: uniqueIds(parsed.data.authorizationStreamEntryIds),
      content_source_stream_entry_ids: uniqueIds(parsed.data.contentSourceStreamEntryIds),
      subject_kind: parsed.data.subjectKind,
      subject_entity_id: parsed.data.subjectEntityId ?? null,
      semantic_slot: semanticSlot,
      canonical_fact:
        semanticSlot === null ? (parsed.data.canonicalFact ?? null) : parsed.data.semanticValue,
      operational_directive: parsed.data.operationalDirective ?? null,
      disclosure_policy: normalizeDisclosurePolicy(parsed.data.disclosurePolicy),
      activation_policy: normalizeActivationPolicy(parsed.data.activationPolicy),
      priority: parsed.data.priority,
      superseded_by: null,
      revoked_reason: null,
      created_at: now,
      updated_at: now,
    });
    const policy = record.disclosure_policy;
    const activationPolicy = record.activation_policy;

    const persist = this.db.transaction(() => {
      this.db
        .prepare(
          `
            INSERT INTO creator_directives (
              id, record_version, status, kind, created_by_entity_id, source_session_id,
              authorization_stream_entry_ids, content_source_stream_entry_ids,
              subject_kind, subject_entity_id, semantic_slot, canonical_fact,
              operational_directive, content_scope, allowed_entity_ids, excluded_entity_ids,
              activation_scope, activation_entity_ids, activation_excluded_entity_ids,
              subject_may_know, mention_policy, denied_audience_behavior, boundary_prompt,
              topic_tags, priority, superseded_by, revoked_reason, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          record.id,
          record.record_version,
          record.status,
          record.kind,
          record.created_by_entity_id,
          record.source_session_id,
          serializeJsonValue(record.authorization_stream_entry_ids),
          serializeJsonValue(record.content_source_stream_entry_ids),
          record.subject_kind,
          record.subject_entity_id,
          record.semantic_slot,
          record.canonical_fact,
          record.operational_directive,
          policy.content_scope,
          serializeJsonValue(policy.allowed_entity_ids),
          serializeJsonValue(policy.excluded_entity_ids),
          activationPolicy.scope,
          serializeJsonValue(activationPolicy.allowed_entity_ids),
          serializeJsonValue(activationPolicy.excluded_entity_ids),
          toStoredBoolean(policy.subject_may_know),
          policy.mention_policy,
          policy.denied_audience_behavior,
          policy.boundary_prompt,
          serializeJsonValue(policy.topic_tags),
          record.priority,
          record.superseded_by,
          record.revoked_reason,
          record.created_at,
          record.updated_at,
        );

      this.supersedeConflictingActiveDirectives(record);
    });

    persist();

    return record;
  }

  get(id: CreatorDirectiveId): CreatorDirective | null {
    const parsedId = parseCreatorDirectiveId(id);
    const row = this.db.prepare("SELECT * FROM creator_directives WHERE id = ?").get(parsedId) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapCreatorDirectiveRow(row);
  }

  list(filter: CreatorDirectiveListFilter = {}): CreatorDirective[] {
    const parsed = creatorDirectiveListFilterSchema.parse(filter);
    const filters: string[] = [];
    const values: unknown[] = [];

    if (parsed.status !== undefined) {
      filters.push("status = ?");
      values.push(parsed.status);
    }

    if (parsed.kind !== undefined) {
      filters.push("kind = ?");
      values.push(parsed.kind);
    }

    if (parsed.createdByEntityId !== undefined) {
      filters.push("created_by_entity_id = ?");
      values.push(parsed.createdByEntityId);
    }

    if (parsed.sourceSessionId !== undefined) {
      filters.push("source_session_id = ?");
      values.push(parsed.sourceSessionId);
    }

    if (parsed.subjectKind !== undefined) {
      filters.push("subject_kind = ?");
      values.push(parsed.subjectKind);
    }

    if (parsed.subjectEntityId !== undefined) {
      if (parsed.subjectEntityId === null) {
        filters.push("subject_entity_id IS NULL");
      } else {
        filters.push("subject_entity_id = ?");
        values.push(parsed.subjectEntityId);
      }
    }

    if (parsed.semanticSlot !== undefined) {
      if (parsed.semanticSlot === null) {
        filters.push("semantic_slot IS NULL");
      } else {
        filters.push("semantic_slot = ?");
        values.push(parsed.semanticSlot);
      }
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM creator_directives
          ${whereClause}
          ORDER BY priority DESC, created_at ASC
        `,
      )
      .all(...values) as Record<string, unknown>[];
    const records = rows.map((row) => mapCreatorDirectiveRow(row));

    if (parsed.topicTag === undefined) {
      return records;
    }

    return records.filter((record) =>
      record.disclosure_policy.topic_tags.includes(parsed.topicTag!),
    );
  }

  listApplicable(options: CreatorDirectiveApplicableOptions): CreatorDirectiveApplicable[] {
    const parsed = creatorDirectiveApplicableOptionsSchema.parse(options);

    return this.list({ status: "active" }).map((directive) =>
      evaluateApplicableDirective(directive, parsed),
    );
  }

  private supersedeActiveDirective(
    current: CreatorDirective,
    replacementId: CreatorDirectiveId,
  ): CreatorDirective | null {
    if (current.status !== "active") {
      return null;
    }

    const updatedAt = this.clock.now();
    const result = this.db
      .prepare(
        `
          UPDATE creator_directives
          SET status = 'superseded',
              superseded_by = ?,
              updated_at = ?,
              record_version = record_version + 1
          WHERE id = ?
            AND status = 'active'
            AND record_version = ?
        `,
      )
      .run(replacementId, updatedAt, current.id, current.record_version);

    if (result.changes === 0) {
      return null;
    }

    return creatorDirectiveSchema.parse({
      ...current,
      record_version: current.record_version + 1,
      status: "superseded",
      superseded_by: replacementId,
      updated_at: updatedAt,
    });
  }

  private supersedeConflictingActiveDirectives(record: CreatorDirective): void {
    if (record.semantic_slot === null) {
      return;
    }

    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM creator_directives
          WHERE status = 'active'
            AND id != ?
            AND kind = ?
            AND subject_kind = ?
            AND (
              (subject_entity_id IS NULL AND ? IS NULL)
              OR subject_entity_id = ?
            )
            AND semantic_slot = ?
          ORDER BY priority DESC, created_at ASC
        `,
      )
      .all(
        record.id,
        record.kind,
        record.subject_kind,
        record.subject_entity_id,
        record.subject_entity_id,
        record.semantic_slot,
      ) as Record<string, unknown>[];

    for (const row of rows) {
      this.supersedeActiveDirective(mapCreatorDirectiveRow(row), record.id);
    }
  }

  supersede(id: CreatorDirectiveId, replacementId: CreatorDirectiveId): CreatorDirective | null {
    const parsedId = parseCreatorDirectiveId(id);
    const parsedReplacementId = parseCreatorDirectiveId(replacementId);
    const current = this.get(parsedId);

    if (current === null) {
      return null;
    }

    return this.supersedeActiveDirective(current, parsedReplacementId);
  }

  supersedeFamilyAtomic(
    input: CreatorDirectiveFamilySupersedeInput,
  ): CreatorDirectiveFamilySupersedeResult | null {
    const survivorId = parseCreatorDirectiveId(input.survivorId);
    const expectedSurvivorVersion = z
      .number()
      .int()
      .positive()
      .parse(input.expectedSurvivorVersion);
    const losers = z
      .array(
        z.object({
          id: creatorDirectiveIdSchema,
          expectedVersion: z.number().int().positive(),
        }),
      )
      .min(1)
      .parse(input.losers);
    const loserIds = new Set<CreatorDirectiveId>();

    for (const loser of losers) {
      if (loser.id === survivorId || loserIds.has(loser.id)) {
        return null;
      }

      loserIds.add(loser.id);
    }

    const run = this.db.transaction((): CreatorDirectiveFamilySupersedeResult | null => {
      const survivor = this.get(survivorId);

      if (
        survivor === null ||
        survivor.status !== "active" ||
        survivor.record_version !== expectedSurvivorVersion
      ) {
        return null;
      }

      const currentLosers: CreatorDirective[] = [];

      for (const loser of losers) {
        const current = this.get(loser.id);

        if (
          current === null ||
          current.status !== "active" ||
          current.record_version !== loser.expectedVersion
        ) {
          return null;
        }

        currentLosers.push(current);
      }

      const updatedAt = this.clock.now();
      const updated: CreatorDirectiveFamilySupersedeResult = [];

      for (const current of currentLosers) {
        const result = this.db
          .prepare(
            `
              UPDATE creator_directives
              SET status = 'superseded',
                  superseded_by = ?,
                  updated_at = ?,
                  record_version = record_version + 1
              WHERE id = ?
                AND status = 'active'
                AND record_version = ?
            `,
          )
          .run(survivorId, updatedAt, current.id, current.record_version);

        if (result.changes === 0) {
          throw new CreatorDirectiveFamilySupersedeAbort();
        }

        updated.push({
          id: current.id,
          record_version: current.record_version + 1,
        });
      }

      return updated;
    });

    try {
      return run();
    } catch (error) {
      if (error instanceof CreatorDirectiveFamilySupersedeAbort) {
        return null;
      }

      throw error;
    }
  }

  reverseSupersede(
    id: CreatorDirectiveId,
    expectedSupersededById: CreatorDirectiveId,
    expectedRecordVersion: number,
  ): CreatorDirective | null {
    const parsedId = parseCreatorDirectiveId(id);
    const parsedSupersededById = parseCreatorDirectiveId(expectedSupersededById);
    const parsedRecordVersion = z.number().int().positive().parse(expectedRecordVersion);
    const current = this.get(parsedId);

    if (
      current === null ||
      current.status !== "superseded" ||
      current.superseded_by !== parsedSupersededById ||
      current.record_version !== parsedRecordVersion
    ) {
      return null;
    }

    const updatedAt = this.clock.now();
    const result = this.db
      .prepare(
        `
          UPDATE creator_directives
          SET status = 'active',
              superseded_by = NULL,
              updated_at = ?,
              record_version = record_version + 1
          WHERE id = ?
            AND status = 'superseded'
            AND superseded_by = ?
            AND record_version = ?
        `,
      )
      .run(updatedAt, parsedId, parsedSupersededById, parsedRecordVersion);

    if (result.changes === 0) {
      return null;
    }

    return creatorDirectiveSchema.parse({
      ...current,
      record_version: current.record_version + 1,
      status: "active",
      superseded_by: null,
      updated_at: updatedAt,
    });
  }

  revokeFamilyAtomic(
    input: CreatorDirectiveFamilyRevokeInput,
  ): CreatorDirectiveFamilyRevokeResult | null {
    const parsedReason = z.string().trim().min(1).parse(input.reason);
    const losers = z
      .array(
        z.object({
          id: creatorDirectiveIdSchema,
          expectedVersion: z.number().int().positive(),
        }),
      )
      .min(1)
      .parse(input.losers);
    const loserIds = new Set<CreatorDirectiveId>();

    for (const loser of losers) {
      if (loserIds.has(loser.id)) {
        return null;
      }

      loserIds.add(loser.id);
    }

    const run = this.db.transaction((): CreatorDirectiveFamilyRevokeResult | null => {
      const currentLosers: CreatorDirective[] = [];

      for (const loser of losers) {
        const current = this.get(loser.id);

        if (
          current === null ||
          current.status !== "active" ||
          current.record_version !== loser.expectedVersion
        ) {
          return null;
        }

        currentLosers.push(current);
      }

      const updatedAt = this.clock.now();
      const updated: CreatorDirectiveFamilyRevokeResult = [];

      for (const current of currentLosers) {
        const result = this.db
          .prepare(
            `
              UPDATE creator_directives
              SET status = 'revoked',
                  revoked_reason = ?,
                  updated_at = ?,
                  record_version = record_version + 1
              WHERE id = ?
                AND status = 'active'
                AND record_version = ?
            `,
          )
          .run(parsedReason, updatedAt, current.id, current.record_version);

        if (result.changes === 0) {
          throw new CreatorDirectiveFamilyRevokeAbort();
        }

        updated.push({
          id: current.id,
          record_version: current.record_version + 1,
        });
      }

      return updated;
    });

    try {
      return run();
    } catch (error) {
      if (error instanceof CreatorDirectiveFamilyRevokeAbort) {
        return null;
      }

      throw error;
    }
  }

  revoke(id: CreatorDirectiveId, reason: string): CreatorDirective | null {
    const parsedId = parseCreatorDirectiveId(id);
    const parsedReason = z.string().trim().min(1).parse(reason);
    const current = this.get(parsedId);

    if (current === null || current.status !== "active") {
      return null;
    }

    const updatedAt = this.clock.now();
    const result = this.db
      .prepare(
        `
          UPDATE creator_directives
          SET status = 'revoked',
              revoked_reason = ?,
              updated_at = ?,
              record_version = record_version + 1
          WHERE id = ?
            AND status = 'active'
            AND record_version = ?
        `,
      )
      .run(parsedReason, updatedAt, parsedId, current.record_version);

    if (result.changes === 0) {
      return null;
    }

    return creatorDirectiveSchema.parse({
      ...current,
      record_version: current.record_version + 1,
      status: "revoked",
      revoked_reason: parsedReason,
      updated_at: updatedAt,
    });
  }

  reverseRevoke(id: CreatorDirectiveId, expectedRecordVersion: number): CreatorDirective | null {
    const parsedId = parseCreatorDirectiveId(id);
    const parsedRecordVersion = z.number().int().positive().parse(expectedRecordVersion);
    const current = this.get(parsedId);

    if (
      current === null ||
      current.status !== "revoked" ||
      current.record_version !== parsedRecordVersion
    ) {
      return null;
    }

    const updatedAt = this.clock.now();
    const result = this.db
      .prepare(
        `
          UPDATE creator_directives
          SET status = 'active',
              revoked_reason = NULL,
              updated_at = ?,
              record_version = record_version + 1
          WHERE id = ?
            AND status = 'revoked'
            AND record_version = ?
        `,
      )
      .run(updatedAt, parsedId, parsedRecordVersion);

    if (result.changes === 0) {
      return null;
    }

    return creatorDirectiveSchema.parse({
      ...current,
      record_version: current.record_version + 1,
      status: "active",
      revoked_reason: null,
      updated_at: updatedAt,
    });
  }
}

export {
  disclosurePolicyBlocksPrivateOperation as creatorDirectiveDisclosureBlocksPrivateOperation,
  evaluateRenderMode as evaluateCreatorDirectiveRenderMode,
  hasEntity as creatorDirectiveHasEntity,
};
