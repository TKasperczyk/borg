import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import {
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
} from "../../util/ids.js";
import { creatorDirectiveMigrations } from "./migrations.js";
import { CreatorDirectiveRepository } from "./repository.js";
import { creatorDirectiveQueueInputSchema } from "./types.js";
import type { CreatorDirective, CreatorDirectiveQueueInput, DisclosurePolicy } from "./types.js";

function createRepository(clock = new ManualClock(1_000)) {
  const db = openDatabase(":memory:", {
    migrations: creatorDirectiveMigrations,
  });
  const repository = new CreatorDirectiveRepository({
    db,
    clock,
  });

  return { db, repository, clock };
}

function disclosurePolicy(overrides: Partial<DisclosurePolicy> = {}): DisclosurePolicy {
  return {
    content_scope: "public",
    allowed_entity_ids: [],
    excluded_entity_ids: [],
    subject_may_know: null,
    mention_policy: "answer_if_asked",
    denied_audience_behavior: "omit",
    boundary_prompt: null,
    topic_tags: [],
    ...overrides,
  };
}

function queueInput(
  overrides: Partial<CreatorDirectiveQueueInput> = {},
): CreatorDirectiveQueueInput {
  return {
    kind: "subject_fact",
    createdByEntityId: createEntityId(),
    sourceSessionId: createSessionId(),
    authorizationStreamEntryIds: [createStreamEntryId()],
    contentSourceStreamEntryIds: [createStreamEntryId()],
    subjectKind: "unknown",
    operationalDirective: "Use this creator-authorized directive when applicable.",
    disclosurePolicy: disclosurePolicy(),
    priority: 5,
    ...overrides,
  };
}

function modeById(records: readonly { directive: CreatorDirective; render_mode: string }[]) {
  return Object.fromEntries(records.map((record) => [record.directive.id, record.render_mode]));
}

describe("CreatorDirectiveRepository", () => {
  it("queues, gets, and lists creator directives by scalar and topic filters", () => {
    const { db, repository } = createRepository();
    const creator = createEntityId();
    const subject = createEntityId();
    const sessionId = createSessionId();
    const authorizationEntryId = createStreamEntryId();
    const contentEntryId = createStreamEntryId();

    try {
      const directive = repository.queue(
        queueInput({
          kind: "self_identity",
          createdByEntityId: creator,
          sourceSessionId: sessionId,
          authorizationStreamEntryIds: [authorizationEntryId],
          contentSourceStreamEntryIds: [contentEntryId],
          subjectKind: "entity",
          subjectEntityId: subject,
          canonicalFact: "Maya may know this fact.",
          operationalDirective: "Treat this as a durable creator directive.",
          disclosurePolicy: disclosurePolicy({
            content_scope: "allow_list",
            allowed_entity_ids: [subject],
            subject_may_know: true,
            topic_tags: ["Atlas", "atlas"],
          }),
          priority: 9,
        }),
      );

      expect(directive).toMatchObject({
        record_version: 1,
        status: "active",
        kind: "self_identity",
        created_by_entity_id: creator,
        source_session_id: sessionId,
        authorization_stream_entry_ids: [authorizationEntryId],
        content_source_stream_entry_ids: [contentEntryId],
        subject_kind: "entity",
        subject_entity_id: subject,
        canonical_fact: "Maya may know this fact.",
        operational_directive: "Treat this as a durable creator directive.",
        disclosure_policy: expect.objectContaining({
          content_scope: "allow_list",
          allowed_entity_ids: [subject],
          subject_may_know: true,
          topic_tags: ["atlas"],
        }),
        priority: 9,
        created_at: 1_000,
        updated_at: 1_000,
      });
      expect(repository.get(directive.id)).toEqual(directive);
      expect(repository.list({ status: "active" }).map((record) => record.id)).toEqual([
        directive.id,
      ]);
      expect(repository.list({ kind: "self_identity" }).map((record) => record.id)).toEqual([
        directive.id,
      ]);
      expect(repository.list({ createdByEntityId: creator }).map((record) => record.id)).toEqual([
        directive.id,
      ]);
      expect(repository.list({ sourceSessionId: sessionId }).map((record) => record.id)).toEqual([
        directive.id,
      ]);
      expect(
        repository
          .list({ subjectKind: "entity", subjectEntityId: subject })
          .map((record) => record.id),
      ).toEqual([directive.id]);
      expect(repository.list({ topicTag: "ATLAS" }).map((record) => record.id)).toEqual([
        directive.id,
      ]);
    } finally {
      db.close();
    }
  });

  it("rejects entity subject queue input without a subject entity id", () => {
    const { db, repository } = createRepository();

    try {
      const missingSubjectEntity = queueInput({
        subjectKind: "entity",
      });
      const missingSchemaResult = creatorDirectiveQueueInputSchema.safeParse(missingSubjectEntity);
      expect(missingSchemaResult.success).toBe(false);
      if (missingSchemaResult.success) {
        throw new Error("expected missing subject entity id to fail schema validation");
      }
      expect(missingSchemaResult.error.issues).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            path: ["subjectEntityId"],
            message: "entity subject requires subjectEntityId",
          }),
        ]),
      );

      let missingError: unknown = null;
      try {
        repository.queue(missingSubjectEntity);
      } catch (error) {
        missingError = error;
      }
      expect(missingError).toBeInstanceOf(StorageError);
      expect(missingError).toMatchObject({ code: "CREATOR_DIRECTIVE_INVALID" });

      const nullSubjectEntity = queueInput({
        subjectKind: "entity",
        subjectEntityId: null,
      });
      const nullSchemaResult = creatorDirectiveQueueInputSchema.safeParse(nullSubjectEntity);
      expect(nullSchemaResult.success).toBe(false);

      let nullError: unknown = null;
      try {
        repository.queue(nullSubjectEntity);
      } catch (error) {
        nullError = error;
      }
      expect(nullError).toBeInstanceOf(StorageError);
      expect(nullError).toMatchObject({ code: "CREATOR_DIRECTIVE_INVALID" });
    } finally {
      db.close();
    }
  });

  it("rejects empty authorization stream entry anchors", () => {
    const { db, repository } = createRepository();

    try {
      const input = queueInput({
        authorizationStreamEntryIds: [],
      });
      const schemaResult = creatorDirectiveQueueInputSchema.safeParse(input);
      expect(schemaResult.success).toBe(false);
      if (schemaResult.success) {
        throw new Error(
          "expected empty authorization stream entry anchors to fail schema validation",
        );
      }
      expect(schemaResult.error.issues).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            path: ["authorizationStreamEntryIds"],
          }),
        ]),
      );

      let error: unknown = null;
      try {
        repository.queue(input);
      } catch (caught) {
        error = caught;
      }
      expect(error).toBeInstanceOf(StorageError);
      expect(error).toMatchObject({ code: "CREATOR_DIRECTIVE_INVALID" });
    } finally {
      db.close();
    }
  });

  it("rejects empty content source stream entry anchors", () => {
    const { db, repository } = createRepository();

    try {
      const input = queueInput({
        contentSourceStreamEntryIds: [],
      });
      const schemaResult = creatorDirectiveQueueInputSchema.safeParse(input);
      expect(schemaResult.success).toBe(false);
      if (schemaResult.success) {
        throw new Error(
          "expected empty content source stream entry anchors to fail schema validation",
        );
      }
      expect(schemaResult.error.issues).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            path: ["contentSourceStreamEntryIds"],
          }),
        ]),
      );

      let error: unknown = null;
      try {
        repository.queue(input);
      } catch (caught) {
        error = caught;
      }
      expect(error).toBeInstanceOf(StorageError);
      expect(error).toMatchObject({ code: "CREATOR_DIRECTIVE_INVALID" });
    } finally {
      db.close();
    }
  });

  it("preserves empty topic tags across queue and get", () => {
    const { db, repository } = createRepository();

    try {
      const directive = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            topic_tags: [],
          }),
        }),
      );

      expect(directive.disclosure_policy.topic_tags).toEqual([]);
      expect(repository.get(directive.id)?.disclosure_policy.topic_tags).toEqual([]);
    } finally {
      db.close();
    }
  });

  it("evaluates single-audience applicability with exclusion first", () => {
    const { db, repository } = createRepository();
    const audience = createEntityId();
    const other = createEntityId();
    const subject = createEntityId();
    const queue = (
      priority: number,
      policy: Partial<DisclosurePolicy>,
      subjectEntityId: EntityId | null = null,
    ) =>
      repository.queue(
        queueInput({
          priority,
          subjectKind: subjectEntityId === null ? "unknown" : "entity",
          subjectEntityId,
          disclosurePolicy: disclosurePolicy(policy),
        }),
      );

    try {
      const excludedPublic = queue(10, {
        content_scope: "public",
        excluded_entity_ids: [audience],
        denied_audience_behavior: "render_boundary_when_relevant",
        topic_tags: ["atlas"],
      });
      const publicDirective = queue(9, {
        content_scope: "public",
      });
      const allowed = queue(8, {
        content_scope: "allow_list",
        allowed_entity_ids: [audience],
      });
      const subjectOnly = queue(
        7,
        {
          content_scope: "subject_only",
        },
        audience,
      );
      const subjectOnlyOtherAudience = queue(
        6,
        {
          content_scope: "subject_only",
        },
        subject,
      );
      const operatorOnly = queue(6, {
        content_scope: "operator_only",
      });
      const allExcept = queue(5, {
        content_scope: "all_except",
        excluded_entity_ids: [other],
      });
      const allExceptExcludedWithBoundary = queue(4, {
        content_scope: "all_except",
        excluded_entity_ids: [audience],
        denied_audience_behavior: "render_boundary_when_relevant",
        topic_tags: ["atlas"],
      });
      const allExceptExcludedWithOmit = queue(4, {
        content_scope: "all_except",
        excluded_entity_ids: [audience],
        denied_audience_behavior: "render_boundary_when_relevant",
        topic_tags: ["private"],
      });
      const deniedWithBoundary = queue(4, {
        content_scope: "allow_list",
        allowed_entity_ids: [other],
        denied_audience_behavior: "render_boundary_when_relevant",
        topic_tags: ["atlas"],
      });
      const deniedWithOmit = queue(3, {
        content_scope: "allow_list",
        allowed_entity_ids: [other],
        denied_audience_behavior: "render_boundary_when_relevant",
        topic_tags: ["private"],
      });

      const participantModes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: audience,
          topicTags: ["atlas"],
          sessionRole: "participant",
        }),
      );
      expect(participantModes).toMatchObject({
        [excludedPublic.id]: "boundary",
        [publicDirective.id]: "content",
        [allowed.id]: "content",
        [subjectOnly.id]: "content",
        [subjectOnlyOtherAudience.id]: "omit",
        [operatorOnly.id]: "omit",
        [allExcept.id]: "content",
        [allExceptExcludedWithBoundary.id]: "boundary",
        [allExceptExcludedWithOmit.id]: "omit",
        [deniedWithBoundary.id]: "boundary",
        [deniedWithOmit.id]: "omit",
      });

      const operatorModes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: audience,
          topicTags: ["atlas"],
          sessionRole: "operator",
        }),
      );
      expect(operatorModes[operatorOnly.id]).toBe("content");
    } finally {
      db.close();
    }
  });

  it("supersedes and revokes active directives only", () => {
    const { db, repository, clock } = createRepository();

    try {
      const original = repository.queue(queueInput({ priority: 3 }));
      const replacement = repository.queue(queueInput({ priority: 4 }));
      const revocable = repository.queue(queueInput({ priority: 5 }));

      clock.set(2_000);
      const superseded = repository.supersede(original.id, replacement.id);
      expect(superseded).toMatchObject({
        id: original.id,
        status: "superseded",
        superseded_by: replacement.id,
        record_version: 2,
        updated_at: 2_000,
      });
      expect(repository.supersede(original.id, replacement.id)).toBeNull();
      expect(repository.revoke(original.id, "already superseded")).toBeNull();

      clock.set(3_000);
      const revoked = repository.revoke(revocable.id, "creator withdrew it");
      expect(revoked).toMatchObject({
        id: revocable.id,
        status: "revoked",
        revoked_reason: "creator withdrew it",
        record_version: 2,
        updated_at: 3_000,
      });
      expect(repository.revoke(revocable.id, "already revoked")).toBeNull();
      expect(repository.supersede(revocable.id, replacement.id)).toBeNull();
      expect(repository.list({ status: "active" }).map((record) => record.id)).toEqual([
        replacement.id,
      ]);
    } finally {
      db.close();
    }
  });
});
