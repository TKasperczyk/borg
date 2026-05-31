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
import { CreatorDirectiveRepository, evaluateCreatorDirectiveRenderMode } from "./repository.js";
import { creatorDirectiveQueueInputSchema, creatorDirectiveSchema } from "./types.js";
import type {
  ActivationPolicy,
  CreatorDirective,
  CreatorDirectiveQueueInput,
  DisclosurePolicy,
} from "./types.js";

const BOUNDARY_PROMPT =
  "A creator-defined confidentiality boundary applies. Decline to discuss confidential details.";

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

function activationPolicy(overrides: Partial<ActivationPolicy> = {}): ActivationPolicy {
  return {
    scope: "same_as_disclosure",
    allowed_entity_ids: [],
    excluded_entity_ids: [],
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

function applicableById<T extends { directive: CreatorDirective }>(
  records: readonly T[],
): Record<string, T> {
  return Object.fromEntries(records.map((record) => [record.directive.id, record]));
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
          semanticSlot: "public_name",
          semanticValue: "Maya",
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
        semantic_slot: "public_name",
        canonical_fact: "Maya",
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
      expect(repository.list({ semanticSlot: "public_name" }).map((record) => record.id)).toEqual([
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

      const missingSlottedGroundedValue = creatorDirectiveSchema.safeParse({
        ...directive,
        canonical_fact: null,
      });
      expect(missingSlottedGroundedValue.success).toBe(false);
      if (missingSlottedGroundedValue.success) {
        throw new Error("expected slotted stored record without canonical_fact to fail validation");
      }
      expect(missingSlottedGroundedValue.error.issues).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            path: ["canonical_fact"],
            message: "slotted creator directive requires canonical_fact",
          }),
        ]),
      );
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

  it("queues and round-trips fact-only subject facts without operational directives", () => {
    const { db, repository } = createRepository();
    const subject = createEntityId();
    const audience = createEntityId();

    try {
      const input = queueInput({
        kind: "subject_fact",
        subjectKind: "entity",
        subjectEntityId: subject,
        canonicalFact: "The launch review is scheduled for Monday.",
        operationalDirective: null,
        disclosurePolicy: disclosurePolicy({
          content_scope: "allow_list",
          allowed_entity_ids: [audience],
          subject_may_know: true,
        }),
        activationPolicy: activationPolicy({
          scope: "allow_list",
          allowed_entity_ids: [audience],
        }),
      });
      const schemaResult = creatorDirectiveQueueInputSchema.safeParse(input);

      expect(schemaResult.success).toBe(true);

      const directive = repository.queue(input);

      expect(directive).toMatchObject({
        kind: "subject_fact",
        canonical_fact: "The launch review is scheduled for Monday.",
        operational_directive: null,
      });
      expect(repository.get(directive.id)).toEqual(directive);

      const applicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: audience,
          participantEntityIds: [audience],
          sessionRole: "participant",
        }),
      );

      expect(applicable[directive.id]).toMatchObject({
        activation: {
          active: true,
          reason: "explicit_allow",
        },
        render_mode: "content",
      });
    } finally {
      db.close();
    }
  });

  it.each(["response_policy", "routing_instruction"] as const)(
    "requires operational directives for %s records",
    (kind) => {
      const { db, repository } = createRepository();
      const input = queueInput({
        kind,
        canonicalFact: null,
        operationalDirective: null,
      });

      try {
        const schemaResult = creatorDirectiveQueueInputSchema.safeParse(input);
        expect(schemaResult.success).toBe(false);
        if (schemaResult.success) {
          throw new Error("expected behavioral directive without operationalDirective to fail");
        }
        expect(schemaResult.error.issues).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              path: ["operationalDirective"],
              message: "behavioral creator directive requires operationalDirective",
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

        const valid = repository.queue({
          ...input,
          operationalDirective: "Use this behavioral directive when active.",
        });
        const storedSchemaResult = creatorDirectiveSchema.safeParse({
          ...valid,
          operational_directive: null,
        });
        expect(storedSchemaResult.success).toBe(false);
        if (storedSchemaResult.success) {
          throw new Error("expected stored behavioral directive without operational_directive to fail");
        }
        expect(storedSchemaResult.error.issues).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              path: ["operational_directive"],
              message: "behavioral creator directive requires operational_directive",
            }),
          ]),
        );
      } finally {
        db.close();
      }
    },
  );

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

  it("rejects malformed queue input and disclosure policy shapes", () => {
    const { db, repository } = createRepository();
    const entity = createEntityId();
    const cases = [
      {
        input: queueInput({
          semanticSlot: "public_name",
        }),
        path: ["semanticValue"],
        message: "slotted creator directive requires semanticValue",
      },
      {
        input: queueInput({
          semanticValue: "Vesper",
          canonicalFact: "Borg's self-chosen name is Claude.",
        }),
        path: ["semanticValue"],
        message: "semanticValue requires semanticSlot",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "public",
            allowed_entity_ids: [entity],
          }),
        }),
        path: ["disclosurePolicy", "allowed_entity_ids"],
        message: "public scope requires empty allowed_entity_ids",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "public",
            excluded_entity_ids: [entity],
          }),
        }),
        path: ["disclosurePolicy", "excluded_entity_ids"],
        message: "public scope requires empty excluded_entity_ids",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "allow_list",
          }),
        }),
        path: ["disclosurePolicy", "allowed_entity_ids"],
        message: "allow_list requires at least one allowed entity",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "allow_list",
            allowed_entity_ids: [entity],
            excluded_entity_ids: [entity],
          }),
        }),
        path: ["disclosurePolicy", "excluded_entity_ids"],
        message: "allow_list allowed and excluded entity ids must not overlap",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
          }),
        }),
        path: ["disclosurePolicy", "excluded_entity_ids"],
        message: "all_except requires at least one excluded entity",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            allowed_entity_ids: [entity],
            excluded_entity_ids: [createEntityId()],
          }),
        }),
        path: ["disclosurePolicy", "allowed_entity_ids"],
        message: "all_except requires empty allowed_entity_ids",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "operator_only",
            allowed_entity_ids: [entity],
          }),
        }),
        path: ["disclosurePolicy", "allowed_entity_ids"],
        message: "operator_only requires empty allowed_entity_ids",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "subject_only",
          }),
        }),
        path: ["subjectEntityId"],
        message: "subject_only requires subjectEntityId",
      },
      {
        input: queueInput({
          subjectKind: "entity",
          subjectEntityId: entity,
          disclosurePolicy: disclosurePolicy({
            content_scope: "subject_only",
            allowed_entity_ids: [entity],
          }),
        }),
        path: ["disclosurePolicy", "allowed_entity_ids"],
        message: "subject_only requires empty allowed_entity_ids",
      },
      {
        input: queueInput({
          subjectKind: "entity",
          subjectEntityId: entity,
          disclosurePolicy: disclosurePolicy({
            content_scope: "subject_only",
            excluded_entity_ids: [entity],
          }),
        }),
        path: ["disclosurePolicy", "excluded_entity_ids"],
        message: "subject_only requires empty excluded_entity_ids",
      },
      {
        input: queueInput({
          subjectKind: "entity",
          subjectEntityId: entity,
          disclosurePolicy: disclosurePolicy({
            content_scope: "subject_only",
            subject_may_know: false,
          }),
        }),
        path: ["disclosurePolicy", "subject_may_know"],
        message: "subject_only requires subject_may_know to be true or null",
      },
      {
        input: queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            excluded_entity_ids: [entity],
            denied_audience_behavior: "render_boundary_when_relevant",
          }),
        }),
        path: ["disclosurePolicy", "boundary_prompt"],
        message: "render_boundary_when_relevant requires boundary_prompt",
      },
      {
        input: queueInput({
          subjectKind: "entity",
          subjectEntityId: entity,
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            excluded_entity_ids: [createEntityId()],
            subject_may_know: false,
          }),
        }),
        path: ["disclosurePolicy", "subject_may_know"],
        message: "subject_may_know=false requires subject exclusion or operator_only scope",
      },
    ];

    try {
      for (const testCase of cases) {
        const schemaResult = creatorDirectiveQueueInputSchema.safeParse(testCase.input);
        expect(schemaResult.success).toBe(false);
        if (schemaResult.success) {
          throw new Error("expected malformed disclosure policy to fail schema validation");
        }
        expect(schemaResult.error.issues).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              path: testCase.path,
              message: testCase.message,
            }),
          ]),
        );

        let error: unknown = null;
        try {
          repository.queue(testCase.input);
        } catch (caught) {
          error = caught;
        }
        expect(error).toBeInstanceOf(StorageError);
        expect(error).toMatchObject({ code: "CREATOR_DIRECTIVE_INVALID" });
      }
    } finally {
      db.close();
    }
  });

  it("keeps operator_only exclusions valid for deny-audience boundaries", () => {
    const { db, repository } = createRepository();
    const excluded = createEntityId();

    try {
      const directive = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "operator_only",
            excluded_entity_ids: [excluded],
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
          }),
        }),
      );
      const applicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: excluded,
          sessionRole: "participant",
        }),
      );

      expect(directive.disclosure_policy.excluded_entity_ids).toEqual([excluded]);
      expect(applicable[directive.id]?.render_mode).toBe("boundary");
      expect(applicable[directive.id]?.reason).toBe("explicit_exclude_boundary");
    } finally {
      db.close();
    }
  });

  it("activates allow-listed participant sessions without disclosing operator-only content", () => {
    const { db, repository } = createRepository();
    const creator = createEntityId();
    const alice = createEntityId();
    const bob = createEntityId();

    try {
      const directive = repository.queue(
        queueInput({
          kind: "response_policy",
          createdByEntityId: creator,
          operationalDirective: "Use this creator-authorized response policy when active.",
          disclosurePolicy: disclosurePolicy({
            content_scope: "operator_only",
          }),
          activationPolicy: activationPolicy({
            scope: "allow_list",
            allowed_entity_ids: [alice],
          }),
        }),
      );

      const aliceApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: alice,
          participantEntityIds: [alice],
          sessionRole: "participant",
        }),
      );
      expect(aliceApplicable[directive.id]).toMatchObject({
        activation: {
          active: true,
          reason: "explicit_allow",
        },
        disclosure: {
          render_mode: "omit",
          reason: "operator_only_omitted",
        },
        render_mode: "omit",
        reason: "operator_only_omitted",
      });

      const bobApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: bob,
          participantEntityIds: [bob],
          sessionRole: "participant",
        }),
      );
      expect(bobApplicable[directive.id]).toMatchObject({
        activation: {
          active: false,
          reason: "unauthorized_omit",
        },
        disclosure: {
          render_mode: "omit",
          reason: "operator_only_omitted",
        },
        render_mode: "omit",
      });

      const operatorApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: alice,
          currentSenderBorgRole: "creator",
          sessionRole: "operator",
        }),
      );
      expect(operatorApplicable[directive.id]).toMatchObject({
        activation: {
          active: true,
          reason: "explicit_allow",
        },
        disclosure: {
          render_mode: "content",
          reason: "operator_only",
        },
        render_mode: "content",
        reason: "operator_only",
      });
    } finally {
      db.close();
    }
  });

  it("keeps all_except activation inactive when an excluded recipient is present", () => {
    const { db, repository } = createRepository();
    const group = createEntityId();
    const alice = createEntityId();
    const bob = createEntityId();

    try {
      const directive = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            excluded_entity_ids: [bob],
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
          }),
          activationPolicy: activationPolicy({
            scope: "all_except",
            excluded_entity_ids: [bob],
          }),
        }),
      );

      const bobApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: bob,
          sessionRole: "participant",
        }),
      );
      expect(bobApplicable[directive.id]).toMatchObject({
        activation: {
          active: false,
          reason: "explicit_exclude",
        },
        disclosure: {
          render_mode: "boundary",
          reason: "explicit_exclude_boundary",
        },
        render_mode: "boundary",
      });

      const groupApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: group,
          participantEntityIds: [alice, bob],
          sessionRole: "participant",
        }),
      );
      expect(groupApplicable[directive.id]).toMatchObject({
        activation: {
          active: false,
          reason: "group_contains_excluded_entity",
        },
        disclosure: {
          render_mode: "boundary",
          reason: "group_contains_excluded_entity",
        },
        render_mode: "boundary",
        reason: "group_contains_excluded_entity",
      });

      const aliceApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: alice,
          sessionRole: "participant",
        }),
      );
      expect(aliceApplicable[directive.id]).toMatchObject({
        activation: {
          active: true,
          reason: "all_except",
        },
        disclosure: {
          render_mode: "content",
          reason: "public",
        },
        render_mode: "content",
      });
    } finally {
      db.close();
    }
  });

  it("applies activation allow_list exclusions before allowed recipient intersection", () => {
    const { db, repository } = createRepository();
    const group = createEntityId();
    const alice = createEntityId();
    const bob = createEntityId();

    try {
      const directive = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "public",
          }),
          activationPolicy: activationPolicy({
            scope: "allow_list",
            allowed_entity_ids: [alice],
            excluded_entity_ids: [bob],
          }),
        }),
      );

      const bobApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: bob,
          sessionRole: "participant",
        }),
      );
      expect(bobApplicable[directive.id]).toMatchObject({
        activation: {
          active: false,
          reason: "explicit_exclude",
        },
        disclosure: {
          render_mode: "content",
          reason: "public",
        },
      });

      const groupApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: group,
          participantEntityIds: [alice, bob],
          sessionRole: "participant",
        }),
      );
      expect(groupApplicable[directive.id]).toMatchObject({
        activation: {
          active: false,
          reason: "group_contains_excluded_entity",
        },
        disclosure: {
          render_mode: "content",
          reason: "public",
        },
      });

      const aliceApplicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: alice,
          sessionRole: "participant",
        }),
      );
      expect(aliceApplicable[directive.id]).toMatchObject({
        activation: {
          active: true,
          reason: "explicit_allow",
        },
        disclosure: {
          render_mode: "content",
          reason: "public",
        },
      });
    } finally {
      db.close();
    }
  });

  it("defaults activation to same_as_disclosure and preserves legacy render applicability", () => {
    const { db, repository } = createRepository();
    const creator = createEntityId();
    const audience = createEntityId();
    const other = createEntityId();

    try {
      const publicDirective = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "public",
          }),
          priority: 10,
        }),
      );
      const deniedDirective = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "allow_list",
            allowed_entity_ids: [other],
          }),
          priority: 9,
        }),
      );
      const boundaryDirective = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            excluded_entity_ids: [audience],
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
          }),
          priority: 8,
        }),
      );
      const operatorOnly = repository.queue(
        queueInput({
          createdByEntityId: creator,
          disclosurePolicy: disclosurePolicy({
            content_scope: "operator_only",
          }),
          priority: 7,
        }),
      );

      expect(publicDirective.activation_policy).toEqual({
        scope: "same_as_disclosure",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
      });

      const participantApplicable = repository.listApplicable({
        currentAudienceEntityId: audience,
        sessionRole: "participant",
      });
      for (const item of participantApplicable) {
        expect(item.disclosure).toEqual({
          render_mode: item.render_mode,
          reason: item.reason,
        });
        expect(item.activation.active).toBe(item.render_mode !== "omit");
      }

      const participantById = applicableById(participantApplicable);
      expect(participantById[publicDirective.id]?.activation.active).toBe(true);
      expect(participantById[deniedDirective.id]?.activation.active).toBe(false);
      expect(participantById[boundaryDirective.id]?.activation.active).toBe(true);
      expect(participantById[operatorOnly.id]?.activation.active).toBe(false);

      const operatorById = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: creator,
          currentSenderBorgRole: "creator",
          sessionRole: "operator",
        }),
      );
      expect(operatorById[operatorOnly.id]).toMatchObject({
        activation: {
          active: true,
          reason: "same_as_disclosure",
        },
        disclosure: {
          render_mode: "content",
          reason: "operator_only",
        },
        render_mode: "content",
      });
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

  it("applies subject_may_know=false before broad render scopes", () => {
    const { db, repository } = createRepository();
    const subject = createEntityId();
    const other = createEntityId();

    try {
      const directive = repository.queue(
        queueInput({
          subjectKind: "entity",
          subjectEntityId: subject,
          disclosurePolicy: disclosurePolicy({
            content_scope: "operator_only",
            subject_may_know: false,
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
          }),
        }),
      );
      const renderMode = evaluateCreatorDirectiveRenderMode(
        {
          ...directive,
          disclosure_policy: {
            ...directive.disclosure_policy,
            content_scope: "all_except",
            excluded_entity_ids: [other],
          },
        },
        {
          currentAudienceEntityId: subject,
          sessionRole: "participant",
        },
      );

      expect(renderMode).toBe("boundary");
      expect(renderMode).not.toBe("content");
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
      const excludedAllExcept = queue(10, {
        content_scope: "all_except",
        excluded_entity_ids: [audience],
        denied_audience_behavior: "render_boundary_when_relevant",
        boundary_prompt: BOUNDARY_PROMPT,
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
        boundary_prompt: BOUNDARY_PROMPT,
        topic_tags: ["atlas"],
      });
      const allExceptExcludedWithoutTopicOverlap = queue(4, {
        content_scope: "all_except",
        excluded_entity_ids: [audience],
        denied_audience_behavior: "render_boundary_when_relevant",
        boundary_prompt: BOUNDARY_PROMPT,
        topic_tags: ["private"],
      });
      const deniedWithBoundary = queue(4, {
        content_scope: "allow_list",
        allowed_entity_ids: [other],
        denied_audience_behavior: "render_boundary_when_relevant",
        boundary_prompt: BOUNDARY_PROMPT,
        topic_tags: ["atlas"],
      });
      const deniedExcludedWithBoundary = queue(4, {
        content_scope: "allow_list",
        allowed_entity_ids: [other],
        excluded_entity_ids: [audience],
        denied_audience_behavior: "render_boundary_when_relevant",
        boundary_prompt: BOUNDARY_PROMPT,
        topic_tags: ["atlas"],
      });
      const deniedWithOmit = queue(3, {
        content_scope: "allow_list",
        allowed_entity_ids: [other],
        denied_audience_behavior: "omit",
        topic_tags: ["private"],
      });

      const participantModes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: audience,
          sessionRole: "participant",
        }),
      );
      expect(participantModes).toMatchObject({
        [excludedAllExcept.id]: "boundary",
        [publicDirective.id]: "content",
        [allowed.id]: "content",
        [subjectOnly.id]: "content",
        [subjectOnlyOtherAudience.id]: "omit",
        [operatorOnly.id]: "omit",
        [allExcept.id]: "content",
        [allExceptExcludedWithBoundary.id]: "boundary",
        [allExceptExcludedWithoutTopicOverlap.id]: "boundary",
        [deniedWithBoundary.id]: "omit",
        [deniedExcludedWithBoundary.id]: "boundary",
        [deniedWithOmit.id]: "omit",
      });

      const operatorModes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: audience,
          currentSenderBorgRole: "creator",
          sessionRole: "operator",
        }),
      );
      expect(operatorModes[operatorOnly.id]).toBe("content");

      const nonCreatorOperatorModes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: audience,
          currentSenderBorgRole: null,
          sessionRole: "operator",
        }),
      );
      expect(nonCreatorOperatorModes[operatorOnly.id]).toBe("omit");

      const emptyParticipantModes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: audience,
          participantEntityIds: [],
          sessionRole: "participant",
        }),
      );
      expect(emptyParticipantModes).toMatchObject(participantModes);
    } finally {
      db.close();
    }
  });

  it("aggregates group recipient-set applicability before rendering content", () => {
    const { db, repository } = createRepository();
    const group = createEntityId();
    const alice = createEntityId();
    const bob = createEntityId();

    try {
      const excludedWithBoundary = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            excluded_entity_ids: [bob],
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
            topic_tags: ["atlas"],
          }),
          priority: 10,
        }),
      );
      const excludedWithoutTopicOverlap = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            excluded_entity_ids: [bob],
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
            topic_tags: ["private"],
          }),
          priority: 9,
        }),
      );
      const allowListAliceOnly = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "allow_list",
            allowed_entity_ids: [alice],
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
            topic_tags: ["atlas"],
          }),
          priority: 8,
        }),
      );
      const publicDirective = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "public",
          }),
          priority: 7,
        }),
      );

      const groupModes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: group,
          participantEntityIds: [alice, bob],
          sessionRole: "participant",
        }),
      );
      expect(groupModes).toMatchObject({
        [excludedWithBoundary.id]: "boundary",
        [excludedWithoutTopicOverlap.id]: "boundary",
        [allowListAliceOnly.id]: "omit",
        [publicDirective.id]: "content",
      });

      const singleRecipientModes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: alice,
          sessionRole: "participant",
        }),
      );
      expect(singleRecipientModes[allowListAliceOnly.id]).toBe("content");
    } finally {
      db.close();
    }
  });

  it("evaluates exactly one concrete group participant instead of the group entity", () => {
    const { db, repository } = createRepository();
    const group = createEntityId();
    const bob = createEntityId();

    try {
      const excludedBob = repository.queue(
        queueInput({
          disclosurePolicy: disclosurePolicy({
            content_scope: "all_except",
            excluded_entity_ids: [bob],
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
          }),
        }),
      );

      const modes = modeById(
        repository.listApplicable({
          currentAudienceEntityId: group,
          participantEntityIds: [bob],
          sessionRole: "participant",
        }),
      );

      expect(modes[excludedBob.id]).toBe("boundary");
      expect(modes[excludedBob.id]).not.toBe("content");
    } finally {
      db.close();
    }
  });

  it("propagates subject_may_know boundaries through group recipient aggregation", () => {
    const { db, repository } = createRepository();
    const group = createEntityId();
    const alice = createEntityId();
    const bob = createEntityId();

    try {
      const directive = repository.queue(
        queueInput({
          subjectKind: "entity",
          subjectEntityId: bob,
          disclosurePolicy: disclosurePolicy({
            content_scope: "operator_only",
            subject_may_know: false,
            denied_audience_behavior: "render_boundary_when_relevant",
            boundary_prompt: BOUNDARY_PROMPT,
          }),
        }),
      );

      const applicable = applicableById(
        repository.listApplicable({
          currentAudienceEntityId: group,
          participantEntityIds: [alice, bob],
          sessionRole: "participant",
        }),
      );

      expect(applicable[directive.id]?.render_mode).toBe("boundary");
      expect(applicable[directive.id]?.reason).toBe("subject_may_not_know");
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

  it("supersedes prior active directives sharing a non-null semantic slot", () => {
    const { db, repository } = createRepository();
    const creator = createEntityId();
    const audience = createEntityId();

    try {
      const original = repository.queue(
        queueInput({
          kind: "self_identity",
          createdByEntityId: creator,
          subjectKind: "borg_self",
          semanticSlot: "public_name",
          semanticValue: "Claude",
          canonicalFact: "Borg's self-chosen name is Claude.",
          operationalDirective: "Answer allowed audiences with Claude when asked.",
          priority: 100,
        }),
      );
      const nullSlot = repository.queue(
        queueInput({
          kind: "subject_fact",
          createdByEntityId: creator,
          subjectKind: "entity",
          subjectEntityId: audience,
          canonicalFact: "Alice has blue hair.",
          operationalDirective: "Answer allowed audiences with Alice's blue-hair fact.",
          priority: 50,
        }),
      );
      const replacement = repository.queue(
        queueInput({
          kind: "self_identity",
          createdByEntityId: creator,
          subjectKind: "borg_self",
          semanticSlot: "public_name",
          semanticValue: "Vesper",
          canonicalFact: "Borg's self-chosen name is Vesper.",
          operationalDirective: "Answer allowed audiences with Vesper when asked.",
          priority: 1,
        }),
      );

      expect(repository.get(original.id)).toMatchObject({
        id: original.id,
        status: "superseded",
        superseded_by: replacement.id,
      });
      expect(repository.get(nullSlot.id)).toMatchObject({
        id: nullSlot.id,
        status: "active",
        superseded_by: null,
      });
      expect(repository.list({ status: "active" }).map((record) => record.id)).toEqual([
        nullSlot.id,
        replacement.id,
      ]);

      const applicableNames = repository
        .listApplicable({
          currentAudienceEntityId: audience,
          sessionRole: "participant",
        })
        .flatMap((item) =>
          item.render_mode === "content" &&
          item.directive.kind === "self_identity" &&
          item.directive.canonical_fact !== null
            ? [item.directive.canonical_fact]
            : [],
        );

      expect(applicableNames).toEqual(["Vesper"]);
    } finally {
      db.close();
    }
  });
});
