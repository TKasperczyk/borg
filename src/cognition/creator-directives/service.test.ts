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

function entityRecord(id: EntityId, canonicalName: string): EntityRecord {
  return {
    id,
    canonical_name: canonicalName,
    aliases: [],
    kind: "person",
    borg_role: canonicalName === "Tom" ? "creator" : null,
    name_provenance: "user_declared",
    created_at: 1_000,
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
  const entities = new Map<EntityId, EntityRecord>([
    [creatorId, entityRecord(creatorId, "Tom")],
    [aliceId, entityRecord(aliceId, "Alice")],
  ]);
  const findByName = vi.fn((name: string) => {
    for (const entity of entities.values()) {
      if (entity.canonical_name === name) {
        return entity.id;
      }
    }

    return null;
  });
  const get = vi.fn((id: EntityId) => entities.get(id) ?? null);
  const tracer = {
    enabled: true,
    includePayloads: false,
    emit: vi.fn(),
  };
  const service = new CreatorDirectiveTurnService({
    model: "haiku",
    creatorDirectiveRepository: repository,
    entityRepository: { findByName, get },
    tracer,
  });

  return {
    db,
    repository,
    service,
    creatorId,
    aliceId,
    findByName,
    get,
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

  it("rejects entity-subject candidates with unresolvable labels", async () => {
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
      expect(harness.tracer.emit).toHaveBeenCalledWith(
        "creator_directive_candidate_rejected",
        expect.objectContaining({
          turnId: "turn-creator-directive",
          session_id: sessionId,
          validationStatus: "rejected",
          reason: "unknown_subject_entity",
        }),
      );
    } finally {
      harness.db.close();
    }
  });
});
