import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { decisionArtifactMigrations } from "../../memory/decision-artifacts/migrations.js";
import { DecisionArtifactRepository } from "../../memory/decision-artifacts/repository.js";
import {
  composeMigrations,
  openDatabase,
  type SqliteDatabase,
} from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import type { JsonValue } from "../../util/json-value.js";
import {
  createEntityId,
  createStreamEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import {
  compileDecisionArtifact,
  DECISION_ARTIFACT_TOOL_NAME,
  type EmitDecisionArtifactPatch,
} from "./compiler.js";

function emitDecisionArtifactPatchResponse(patch: EmitDecisionArtifactPatch) {
  return {
    text: "",
    input_tokens: 12,
    output_tokens: 8,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_decision_patch",
        name: DECISION_ARTIFACT_TOOL_NAME,
        input: patch,
      },
    ],
  };
}

function throwingResponse(): never {
  throw new Error("llm down");
}

function createTraceRecorder(): TurnTracer & {
  events: Array<{ event: TurnTraceEventName; data: TurnTraceData }>;
} {
  const events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];

  return {
    enabled: true,
    includePayloads: true,
    events,
    emit: vi.fn((event: TurnTraceEventName, data: TurnTraceData) => {
      events.push({ event, data });
    }),
  };
}

describe("compileDecisionArtifact", () => {
  let db: SqliteDatabase;
  let repository: DecisionArtifactRepository;
  let clock: FixedClock;
  let audience: EntityId;
  let self: EntityId;
  let alice: EntityId;
  let currentStreamEntryId: StreamEntryId;

  beforeEach(() => {
    db = openDatabase(":memory:", {
      migrations: composeMigrations(decisionArtifactMigrations),
    });
    clock = new FixedClock(2_000);
    repository = new DecisionArtifactRepository({
      db,
      clock,
    });
    audience = createEntityId();
    self = createEntityId();
    alice = createEntityId();
    currentStreamEntryId = createStreamEntryId();
  });

  afterEach(() => {
    db.close();
  });

  function baseInput(llmClient: FakeLLMClient) {
    return {
      llmClient,
      model: "claude-haiku-test",
      repository,
      audienceEntityId: audience,
      selfEntityId: self,
      speakerEntityId: alice,
      participants: [{ entityId: alice, displayName: "Alice" }],
      currentUserMessage: "Madrid 3, SS 3, Seville 4, Granada 3 is locked.",
      currentUserStreamEntryId: currentStreamEntryId,
      promptVisibleLedger: "Commitments: route order confirmed.",
      allowedSourceStreamEntryIds: [currentStreamEntryId],
      clock,
      turnId: "turn_decision_artifact_test",
    };
  }

  it("adds a locked decision emitted by the LLM", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
              owner_entity_id: audience,
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact(baseInput(llmClient));

    const artifact = repository.get(audience);
    expect(artifact?.entries).toHaveLength(1);
    expect(artifact?.entries[0]).toMatchObject({
      kind: "locked",
      text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
      owner_entity_id: audience,
      provenance_stream_entry_ids: [currentStreamEntryId],
    });
    expect(llmClient.requests[0]).toMatchObject({
      model: "claude-haiku-test",
      max_tokens: 1536,
      budget: "decision-artifact-compiler",
      tool_choice: { type: "tool", name: DECISION_ARTIFACT_TOOL_NAME },
    });
  });

  it("rejects an invalid owner entity id with a traced reason", async () => {
    const trace = createTraceRecorder();
    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "add",
              kind: "locked",
              text: "Madrid 3 / SS 3 / Seville 4 / Granada 3",
              owner_entity_id: createEntityId(),
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      tracer: trace,
    });

    expect(repository.get(audience)).toBeNull();
    expect(trace.events).toContainEqual(
      expect.objectContaining({
        event: "decision_artifact_compile_completed",
        data: expect.objectContaining({
          rejectedCount: 1,
          rejectionReasons: ["invalid_owner_entity_id"] satisfies JsonValue,
          applied: false,
        }),
      }),
    );
  });

  it("supersedes an existing entry with a replacement entry", async () => {
    const firstSource = createStreamEntryId();
    const initial = repository.upsert(
      audience,
      [
        {
          type: "add",
          kind: "locked",
          text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 2",
          provenance_stream_entry_ids: [firstSource],
        },
      ],
      {
        lastCompiledStreamEntryId: firstSource,
      },
    );
    const oldEntryId = initial?.entries[0]?.id;

    expect(oldEntryId).toBeDefined();

    const llmClient = new FakeLLMClient({
      responses: [
        emitDecisionArtifactPatchResponse({
          operations: [
            {
              type: "supersede",
              id: oldEntryId!,
              replacement: {
                kind: "locked",
                text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
                owner_entity_id: audience,
                source_stream_entry_ids: [currentStreamEntryId],
              },
              source_stream_entry_ids: [currentStreamEntryId],
            },
          ],
        }),
      ],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      allowedSourceStreamEntryIds: [firstSource, currentStreamEntryId],
    });

    const artifact = repository.get(audience);
    const oldEntry = artifact?.entries.find((entry) => entry.id === oldEntryId);
    const replacement = artifact?.entries.find((entry) => entry.id !== oldEntryId);

    expect(artifact?.entries).toHaveLength(2);
    expect(oldEntry?.superseded_by_id).toBe(replacement?.id);
    expect(replacement).toMatchObject({
      kind: "locked",
      text: "Locked route order: Madrid 3 / SS 3 / Seville 4 / Granada 3",
      provenance_stream_entry_ids: [currentStreamEntryId],
    });
  });

  it("skips gracefully when the LLM call fails", async () => {
    const onDegraded = vi.fn();
    const llmClient = new FakeLLMClient({
      responses: [throwingResponse],
    });

    await compileDecisionArtifact({
      ...baseInput(llmClient),
      onDegraded,
    });

    expect(repository.get(audience)).toBeNull();
    expect(onDegraded).toHaveBeenCalledWith("llm_failed", expect.any(Error));
  });

  it("does not bump the parent record version for a no-op compile", async () => {
    const initial = repository.upsert(audience, [
      {
        type: "add",
        kind: "live",
        text: "Question: Granada pacing",
        provenance_stream_entry_ids: [currentStreamEntryId],
      },
    ]);
    const llmClient = new FakeLLMClient({
      responses: [emitDecisionArtifactPatchResponse({ operations: [] })],
    });

    await compileDecisionArtifact(baseInput(llmClient));

    expect(repository.get(audience)?.record_version).toBe(initial?.record_version);
  });
});
