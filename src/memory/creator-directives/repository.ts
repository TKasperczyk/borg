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
  creatorDirectiveApplicableOptionsSchema,
  creatorDirectiveListFilterSchema,
  creatorDirectiveQueueInputSchema,
  creatorDirectiveSchema,
  disclosurePolicySchema,
  type CreatorDirective,
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
    canonical_fact:
      row.canonical_fact === null || row.canonical_fact === undefined
        ? null
        : String(row.canonical_fact),
    operational_directive: row.operational_directive,
    disclosure_policy: disclosurePolicy,
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

type CreatorDirectiveRenderEvaluation = {
  render_mode: CreatorDirectiveRenderMode;
  reason: CreatorDirectiveRenderReason;
};

function boundaryOrOmit(
  directive: CreatorDirective,
  boundaryReason: CreatorDirectiveRenderReason,
  omitReason: CreatorDirectiveRenderReason = "unauthorized_omit",
): CreatorDirectiveRenderEvaluation {
  if (directive.disclosure_policy.denied_audience_behavior !== "render_boundary_when_relevant") {
    return { render_mode: "omit", reason: omitReason };
  }

  return { render_mode: "boundary", reason: boundaryReason };
}

function evaluateRenderEvaluation(
  directive: CreatorDirective,
  options: CreatorDirectiveApplicableOptions,
  audienceId: EntityId | null = options.currentAudienceEntityId,
): CreatorDirectiveRenderEvaluation {
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
    const isCreatorOperator =
      options.sessionRole === "operator" &&
      (options.currentSenderBorgRole === "creator" ||
        options.currentAudienceEntityId === directive.created_by_entity_id);

    return isCreatorOperator
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
  return evaluateRenderEvaluation(directive, options, audienceId).render_mode;
}

function effectiveRecipientEntityIds(
  options: CreatorDirectiveApplicableOptions,
): readonly (EntityId | null)[] {
  const participantEntityIds = options.participantEntityIds ?? [];

  if (participantEntityIds.length === 0) {
    return [options.currentAudienceEntityId];
  }

  return uniqueIds(participantEntityIds);
}

function evaluateApplicableRenderMode(
  directive: CreatorDirective,
  options: CreatorDirectiveApplicableOptions,
): CreatorDirectiveRenderEvaluation {
  const recipientEntityIds = effectiveRecipientEntityIds(options);

  if (recipientEntityIds.length <= 1) {
    return evaluateRenderEvaluation(directive, options, recipientEntityIds[0] ?? null);
  }

  const recipientEvaluations = recipientEntityIds.map((recipientEntityId) => ({
    isExcluded: hasEntity(directive.disclosure_policy.excluded_entity_ids, recipientEntityId),
    evaluation: evaluateRenderEvaluation(directive, options, recipientEntityId),
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

export type CreatorDirectiveRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

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
      canonical_fact: parsed.data.canonicalFact ?? null,
      operational_directive: parsed.data.operationalDirective,
      disclosure_policy: normalizeDisclosurePolicy(parsed.data.disclosurePolicy),
      priority: parsed.data.priority,
      superseded_by: null,
      revoked_reason: null,
      created_at: now,
      updated_at: now,
    });
    const policy = record.disclosure_policy;

    this.db
      .prepare(
        `
          INSERT INTO creator_directives (
            id, record_version, status, kind, created_by_entity_id, source_session_id,
            authorization_stream_entry_ids, content_source_stream_entry_ids,
            subject_kind, subject_entity_id, canonical_fact, operational_directive,
            content_scope, allowed_entity_ids, excluded_entity_ids, subject_may_know,
            mention_policy, denied_audience_behavior, boundary_prompt, topic_tags,
            priority, superseded_by, revoked_reason, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        record.canonical_fact,
        record.operational_directive,
        policy.content_scope,
        serializeJsonValue(policy.allowed_entity_ids),
        serializeJsonValue(policy.excluded_entity_ids),
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

    return this.list({ status: "active" }).map((directive) => {
      const evaluation = evaluateApplicableRenderMode(directive, parsed);

      return {
        directive,
        render_mode: evaluation.render_mode,
        reason: evaluation.reason,
      };
    });
  }

  supersede(id: CreatorDirectiveId, replacementId: CreatorDirectiveId): CreatorDirective | null {
    const parsedId = parseCreatorDirectiveId(id);
    const parsedReplacementId = parseCreatorDirectiveId(replacementId);
    const current = this.get(parsedId);

    if (current === null || current.status !== "active") {
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
      .run(parsedReplacementId, updatedAt, parsedId, current.record_version);

    if (result.changes === 0) {
      return null;
    }

    return creatorDirectiveSchema.parse({
      ...current,
      record_version: current.record_version + 1,
      status: "superseded",
      superseded_by: parsedReplacementId,
      updated_at: updatedAt,
    });
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
}

export { evaluateRenderMode as evaluateCreatorDirectiveRenderMode };
