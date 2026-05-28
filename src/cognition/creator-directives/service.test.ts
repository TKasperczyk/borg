import { describe, expect, it, vi } from "vitest";

import type { LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { EntityRecord } from "../../memory/commitments/index.js";
import {
  CreatorDirectiveRepository,
  creatorDirectiveMigrations,
} from "../../memory/creator-directives/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
} from "../../util/ids.js";
import { CREATOR_DIRECTIVE_TOOL_NAME } from "./extractor.js";
import { CreatorDirectiveTurnService } from "./service.js";

function creatorDirectiveResponse(candidate: Record<string, unknown>): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_creator_directive",
        name: CREATOR_DIRECTIVE_TOOL_NAME,
        input: {
          decision: "creator_directive",
          reason: "The creator gave explicit durable disclosure guidance.",
          candidates: [candidate],
        },
      },
    ],
  };
}

function candidate(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    kind: "self_identity",
    subject_kind: "borg_self",
    subject_entity_id: null,
    subject_label: "Borg",
    semantic_slot: "public_name",
    semantic_value: "Kestrel",
    canonical_fact: "Borg's self-chosen name is Kestrel.",
    operational_directive: "Answer allowed audiences with Borg's self-chosen name when asked.",
    disclosure_policy: {
      content_scope: "public",
      allowed_entity_ids: [],
      allowed_entity_labels: [],
      excluded_entity_ids: [],
      excluded_entity_labels: [],
      subject_may_know: true,
      mention_policy: "answer_if_asked",
      denied_audience_behavior: "omit",
      boundary_prompt: null,
      topic_tags: ["Kestrel"],
    },
    priority: 8,
    confidence: 0.9,
    reason: "Explicit durable public self-identity disclosure.",
    ...overrides,
  };
}

function entityRecord(
  id: EntityId,
  canonicalName: string,
  overrides: Partial<EntityRecord> = {},
): EntityRecord {
  return {
    id,
    canonical_name: canonicalName,
    aliases: [],
    kind: "person",
    borg_role: canonicalName === "Tom" ? "creator" : null,
    name_provenance: "user_declared",
    created_at: 1_000,
    ...overrides,
  };
}

function createHarness() {
  const db = openDatabase(":memory:", {
    migrations: creatorDirectiveMigrations,
  });
  const clock = new FixedClock(2_000);
  const repository = new CreatorDirectiveRepository({ db, clock });
  const creatorId = createEntityId();
  const aliceId = createEntityId();
  const bobId = createEntityId();
  const entities = new Map<EntityId, EntityRecord>([
    [creatorId, entityRecord(creatorId, "Tom")],
    [aliceId, entityRecord(aliceId, "Alice")],
    [bobId, entityRecord(bobId, "Bob")],
  ]);
  const findAllByName = vi.fn((name: string) =>
    [...entities.values()]
      .filter((entity) => entity.canonical_name === name)
      .map((entity) => entity.id),
  );
  const get = vi.fn((id: EntityId) => entities.get(id) ?? null);
  const resolve = vi.fn(
    (
      name: string,
      options: { kind?: EntityRecord["kind"]; provenance?: EntityRecord["name_provenance"] } = {},
    ) => {
      const matches = findAllByName(name);

      if (matches.length > 0) {
        return matches[0]!;
      }

      const entityId = createEntityId();
      entities.set(
        entityId,
        entityRecord(entityId, name, {
          kind: options.kind ?? "person",
          name_provenance: options.provenance ?? "unknown",
        }),
      );

      return entityId;
    },
  );
  const tracer = {
    enabled: true,
    includePayloads: false,
    emit: vi.fn(),
  };
  const service = new CreatorDirectiveTurnService({
    model: "haiku",
    creatorDirectiveRepository: repository,
    entityRepository: { findAllByName, get, resolve },
    tracer,
  });

  return {
    db,
    repository,
    service,
    creatorId,
    aliceId,
    bobId,
    entities,
    findAllByName,
    get,
    resolve,
    tracer,
  };
}

function baseInput(creatorId: EntityId, overrides = {}) {
  return {
    llmClient: new FakeLLMClient({
      responses: [creatorDirectiveResponse(candidate())],
    }),
    turnId: "turn-creator-directive",
    isUserTurn: true,
    userMessage: "Kestrel is my name, anyone can know.",
    audienceEntityId: creatorId,
    currentSenderEntityId: creatorId,
    currentSenderBorgRole: "creator" as const,
    currentSenderDisplayName: "Tom",
    sourceSessionId: DEFAULT_SESSION_ID,
    persistedUserEntryId: createStreamEntryId(),
    recentHistory: [],
    sessionId: createSessionId(),
    sessionAudienceRole: "operator" as const,
    ...overrides,
  };
}

describe("CreatorDirectiveTurnService", () => {
  it.each([
    {
      label: "non-user turn",
      overrides: { isUserTurn: false },
    },
    {
      label: "non-operator session",
      overrides: { sessionAudienceRole: "participant" as const },
    },
    {
      label: "non-creator speaker",
      overrides: { currentSenderBorgRole: null },
    },
    {
      label: "missing sender entity",
      overrides: { currentSenderEntityId: null },
    },
  ])("skips extraction for $label", async ({ overrides }) => {
    const harness = createHarness();
    const llmClient = new FakeLLMClient({
      responses: [creatorDirectiveResponse(candidate())],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
          ...overrides,
        }),
      );

      expect(result).toEqual([]);
      expect(llmClient.requests).toHaveLength(0);
      expect(harness.tracer.emit).not.toHaveBeenCalled();
    } finally {
      harness.db.close();
    }
  });

  it("persists extracted directives and emits trace events with session_id", async () => {
    const harness = createHarness();
    const userEntryId = createStreamEntryId();
    const sessionId = createSessionId();
    const llmClient = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse(
          candidate({
            disclosure_policy: {
              content_scope: "allow_list",
              allowed_entity_ids: [],
              allowed_entity_labels: ["Alice"],
              excluded_entity_ids: [],
              excluded_entity_labels: [],
              subject_may_know: true,
              mention_policy: "answer_if_asked",
              denied_audience_behavior: "omit",
              boundary_prompt: null,
              topic_tags: ["Alice"],
            },
          }),
        ),
      ],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
          persistedUserEntryId: userEntryId,
          sessionId,
        }),
      );

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        created_by_entity_id: harness.creatorId,
        source_session_id: DEFAULT_SESSION_ID,
        authorization_stream_entry_ids: [userEntryId],
        content_source_stream_entry_ids: [userEntryId],
        disclosure_policy: expect.objectContaining({
          content_scope: "allow_list",
          allowed_entity_ids: [harness.aliceId],
        }),
      });
      expect(harness.repository.list()).toHaveLength(1);
      expect(harness.tracer.emit).toHaveBeenCalledWith(
        "creator_directive_candidate_extracted",
        expect.objectContaining({
          turnId: "turn-creator-directive",
          session_id: sessionId,
          validationStatus: "candidate",
        }),
      );
      expect(harness.tracer.emit).toHaveBeenCalledWith(
        "creator_directive_persisted",
        expect.objectContaining({
          turnId: "turn-creator-directive",
          session_id: sessionId,
          validationStatus: "accepted",
        }),
      );
    } finally {
      harness.db.close();
    }
  });

  it("rejects slotted candidates without a semantic value before persistence", async () => {
    const harness = createHarness();
    const sessionId = createSessionId();
    const llmClient = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse(
          candidate({
            semantic_value: null,
            canonical_fact: "Borg's self-chosen name is Claude.",
            disclosure_policy: {
              content_scope: "public",
              allowed_entity_ids: [],
              allowed_entity_labels: [],
              excluded_entity_ids: [],
              excluded_entity_labels: [],
              subject_may_know: true,
              mention_policy: "answer_if_asked",
              denied_audience_behavior: "omit",
              boundary_prompt: null,
              topic_tags: ["Claude"],
            },
          }),
        ),
      ],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
          sessionId,
          userMessage: "I'd like you to choose a name for yourself.",
        }),
      );

      expect(result).toEqual([]);
      expect(harness.repository.list()).toEqual([]);
      expect(harness.tracer.emit).toHaveBeenCalledWith(
        "creator_directive_candidate_rejected",
        expect.objectContaining({
          turnId: "turn-creator-directive",
          session_id: sessionId,
          validationStatus: "rejected",
          reason: "invalid_payload",
        }),
      );
    } finally {
      harness.db.close();
    }
  });

  it("rejects ungrounded semantic values before persistence", async () => {
    const harness = createHarness();
    const sessionId = createSessionId();
    const llmClient = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse(
          candidate({
            semantic_value: "Claude",
            canonical_fact: "Borg's self-chosen name is Claude.",
            disclosure_policy: {
              content_scope: "public",
              allowed_entity_ids: [],
              allowed_entity_labels: [],
              excluded_entity_ids: [],
              excluded_entity_labels: [],
              subject_may_know: true,
              mention_policy: "answer_if_asked",
              denied_audience_behavior: "omit",
              boundary_prompt: null,
              topic_tags: ["Claude"],
            },
          }),
        ),
      ],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
          sessionId,
          userMessage: "I'd like you to choose a name for yourself.",
        }),
      );

      expect(result).toEqual([]);
      expect(harness.repository.list()).toEqual([]);
      expect(harness.tracer.emit).toHaveBeenCalledWith(
        "creator_directive_candidate_rejected",
        expect.objectContaining({
          turnId: "turn-creator-directive",
          session_id: sessionId,
          validationStatus: "rejected",
          reason: "ungrounded_slot_value",
        }),
      );
    } finally {
      harness.db.close();
    }
  });

  it("queues grounded self-identity values from the current message", async () => {
    const harness = createHarness();
    const llmClient = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse(
          candidate({
            semantic_value: "Vesper",
            canonical_fact: "Borg's self-chosen name is Claude.",
            disclosure_policy: {
              content_scope: "public",
              allowed_entity_ids: [],
              allowed_entity_labels: [],
              excluded_entity_ids: [],
              excluded_entity_labels: [],
              subject_may_know: true,
              mention_policy: "answer_if_asked",
              denied_audience_behavior: "omit",
              boundary_prompt: null,
              topic_tags: ["Vesper"],
            },
          }),
        ),
      ],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
          userMessage: "Alright Vesper, that name works.",
        }),
      );

      expect(result).toHaveLength(1);
      expect(result[0]).toMatchObject({
        semantic_slot: "public_name",
        canonical_fact: "Vesper",
      });
      expect(result[0]?.canonical_fact).not.toContain("Claude");
      expect(harness.repository.list()).toHaveLength(1);
    } finally {
      harness.db.close();
    }
  });

  it("normalizes salvageable public allow policies before persistence", async () => {
    const harness = createHarness();
    const userEntryId = createStreamEntryId();
    const sessionId = createSessionId();
    const llmClient = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse(
          candidate({
            disclosure_policy: {
              content_scope: "public",
              allowed_entity_ids: [harness.aliceId],
              allowed_entity_labels: [],
              excluded_entity_ids: [],
              excluded_entity_labels: [],
              subject_may_know: true,
              mention_policy: "answer_if_asked",
              denied_audience_behavior: "omit",
              boundary_prompt: null,
              topic_tags: ["Alice"],
            },
          }),
        ),
      ],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
          persistedUserEntryId: userEntryId,
          sessionId,
        }),
      );

      expect(result).toHaveLength(1);
      expect(result[0]?.disclosure_policy).toMatchObject({
        content_scope: "allow_list",
        allowed_entity_ids: [harness.aliceId],
        excluded_entity_ids: [],
      });
      expect(harness.tracer.emit).toHaveBeenCalledWith(
        "creator_directive_policy_normalized",
        expect.objectContaining({
          turnId: "turn-creator-directive",
          session_id: sessionId,
          candidate_index: 0,
          validationStatus: "normalized",
          original_scope: "public",
          normalized_scope: "allow_list",
          reason: "public_with_allowed_entities",
        }),
      );

      const aliceApplicable = harness.repository.listApplicable({
        currentAudienceEntityId: harness.aliceId,
        sessionRole: "participant",
      });
      const bobApplicable = harness.repository.listApplicable({
        currentAudienceEntityId: harness.bobId,
        sessionRole: "participant",
      });

      expect(aliceApplicable[0]).toMatchObject({
        directive: expect.objectContaining({ id: result[0]?.id }),
        render_mode: "content",
      });
      expect(bobApplicable[0]).toMatchObject({
        directive: expect.objectContaining({ id: result[0]?.id }),
        render_mode: "omit",
      });
    } finally {
      harness.db.close();
    }
  });

  it("creates unknown entity-subject labels in creator-directive context", async () => {
    const harness = createHarness();
    const sessionId = createSessionId();
    const llmClient = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse(
          candidate({
            kind: "subject_fact",
            subject_kind: "entity",
            subject_entity_id: null,
            subject_label: "Mallory",
            semantic_slot: null,
            semantic_value: null,
            canonical_fact: "Mallory knows the launch alias.",
            operational_directive: "Answer allowed audiences with Mallory's launch alias.",
          }),
        ),
      ],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
          sessionId,
        }),
      );

      const mallory = [...harness.entities.values()].find(
        (entity) => entity.canonical_name === "Mallory",
      );

      expect(result).toHaveLength(1);
      expect(harness.repository.list()).toHaveLength(1);
      expect(mallory).toMatchObject({
        canonical_name: "Mallory",
        name_provenance: "creator_directive",
      });
      expect(result[0]?.subject_entity_id).toBe(mallory?.id);
      expect(harness.resolve).toHaveBeenCalledWith("Mallory", {
        kind: "person",
        provenance: "creator_directive",
      });
      expect(harness.tracer.emit).toHaveBeenCalledWith(
        "creator_directive_persisted",
        expect.objectContaining({
          turnId: "turn-creator-directive",
          session_id: sessionId,
          validationStatus: "accepted",
        }),
      );
    } finally {
      harness.db.close();
    }
  });

  it("resolves existing non-person labels before creating creator-directive entities", async () => {
    const harness = createHarness();
    const planningTeamId = createEntityId();
    harness.entities.set(
      planningTeamId,
      entityRecord(planningTeamId, "planning-team", {
        kind: "group",
      }),
    );
    const llmClient = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse(
          candidate({
            disclosure_policy: {
              content_scope: "all_except",
              allowed_entity_ids: [],
              allowed_entity_labels: [],
              excluded_entity_ids: [],
              excluded_entity_labels: ["planning-team"],
              subject_may_know: true,
              mention_policy: "answer_if_asked",
              denied_audience_behavior: "omit",
              boundary_prompt: null,
              topic_tags: ["planning-team"],
            },
          }),
        ),
      ],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
        }),
      );

      expect(result).toHaveLength(1);
      expect(result[0]?.disclosure_policy.excluded_entity_ids).toEqual([planningTeamId]);
      expect(
        [...harness.entities.values()].filter(
          (entity) => entity.canonical_name === "planning-team",
        ),
      ).toHaveLength(1);
      expect(harness.resolve).not.toHaveBeenCalled();
    } finally {
      harness.db.close();
    }
  });

  it("rejects ambiguous entity-subject label matches", async () => {
    const harness = createHarness();
    const sessionId = createSessionId();
    const firstMallory = createEntityId();
    const secondMallory = createEntityId();
    harness.entities.set(firstMallory, entityRecord(firstMallory, "Mallory"));
    harness.entities.set(secondMallory, entityRecord(secondMallory, "Mallory"));
    const llmClient = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse(
          candidate({
            kind: "subject_fact",
            subject_kind: "entity",
            subject_entity_id: null,
            subject_label: "Mallory",
            semantic_slot: null,
            semantic_value: null,
            canonical_fact: "Mallory knows the launch alias.",
            operational_directive: "Answer allowed audiences with Mallory's launch alias.",
          }),
        ),
      ],
    });

    try {
      const result = await harness.service.extractAndPersist(
        baseInput(harness.creatorId, {
          llmClient,
          sessionId,
        }),
      );

      expect(result).toEqual([]);
      expect(harness.repository.list()).toEqual([]);
      expect(harness.resolve).not.toHaveBeenCalledWith("Mallory", expect.anything());
      expect(harness.tracer.emit).toHaveBeenCalledWith(
        "creator_directive_candidate_rejected",
        expect.objectContaining({
          turnId: "turn-creator-directive",
          session_id: sessionId,
          validationStatus: "rejected",
          reason: "ambiguous_subject_entity",
        }),
      );
    } finally {
      harness.db.close();
    }
  });
});
