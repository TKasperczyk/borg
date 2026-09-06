import { afterEach, describe, expect, it, vi } from "vitest";

import { CorrectivePreferenceExtractorDegradedError } from "../../cognition/commitments/corrective-preference-extractor.js";
import type { StreamEntry } from "../../stream/index.js";
import { CallbackTracer, type CallbackTraceEntry } from "../../tracing/tracer.js";
import type { CommitmentId, SessionId } from "../../util/ids.js";
import {
  Borg,
  FakeLLMClient,
  ManualClock,
  ScriptedEmbeddingClient,
  borgInternals,
  createSessionId,
  createStreamEntryId,
  createTestConfig,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

function correctivePreferenceResponse(
  supersedesCommitmentId: CommitmentId | null = null,
  directiveSourceStreamEntryId: StreamEntry["id"] | null = null,
) {
  return {
    text: "",
    input_tokens: 6,
    output_tokens: 3,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_sidecar_corrective",
        name: "EmitCorrectivePreference",
        input: {
          classification: "corrective_preference",
          type: "boundary",
          kind: "boundary",
          enforcement_class: "critical",
          critical_domain: "audience_scope",
          directive: "Never disclose my private details outside this audience.",
          directive_source_stream_entry_id: directiveSourceStreamEntryId,
          directive_family: "private audience scope",
          closure_pressure_relevance: "neutral",
          priority: 90,
          reason: "The sender set a durable privacy boundary.",
          confidence: 0.97,
          supersedes_commitment_id: supersedesCommitmentId,
          retires_commitment_id: null,
          applies_to_audience_entity_id: null,
          relationship_claims: [],
          slot_negations: [],
        },
      },
    ],
  };
}

function noneResponse(
  slotNegations: Array<{
    subject_entity_id: string;
    slot_key: string;
    rejected_value: string | null;
    source_stream_entry_ids: string[];
    confidence: number;
  }> = [],
) {
  return {
    text: "",
    input_tokens: 5,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_sidecar_none",
        name: "EmitCorrectivePreference",
        input: {
          classification: "none",
          type: null,
          kind: null,
          enforcement_class: null,
          critical_domain: null,
          directive: null,
          directive_source_stream_entry_id: null,
          directive_family: null,
          closure_pressure_relevance: null,
          priority: null,
          reason: "This is ordinary conversation, not a durable correction.",
          confidence: 0.98,
          supersedes_commitment_id: null,
          retires_commitment_id: null,
          applies_to_audience_entity_id: null,
          relationship_claims: [],
          slot_negations: slotNegations,
        },
      },
    ],
  };
}

function retirementResponse(commitmentId: CommitmentId) {
  return {
    text: "",
    input_tokens: 6,
    output_tokens: 3,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_sidecar_retire",
        name: "EmitCorrectivePreference",
        input: {
          classification: "retire_commitment",
          type: null,
          kind: null,
          enforcement_class: null,
          critical_domain: null,
          directive: null,
          directive_source_stream_entry_id: null,
          directive_family: null,
          closure_pressure_relevance: null,
          priority: null,
          reason: "The sender explicitly stood down the supplied rule.",
          confidence: 0.96,
          supersedes_commitment_id: null,
          retires_commitment_id: commitmentId,
          applies_to_audience_entity_id: null,
          relationship_claims: [],
          slot_negations: [],
        },
      },
    ],
  };
}

type IngestionCoordinatorInternals = {
  deps: {
    entryIndex: {
      getCorrectivePreferenceIngestionReceipt(entryId: string): {
        status: "processed" | "retryable" | "dead_letter";
        failure_count: number;
        last_error: string | null;
      } | null;
    };
    commitmentRepository: {
      supersede(...args: unknown[]): unknown;
      revoke(...args: unknown[]): unknown;
    };
    identityService: {
      addCommitment(...args: unknown[]): unknown;
    };
    relationalSlotRepository: {
      applyAssertion(input: {
        subject_entity_id: ReturnType<Borg["entities"]["resolve"]>;
        slot_key: string;
        asserted_value: string;
        source_stream_entry_ids: ReturnType<typeof createStreamEntryId>[];
        confirmation: "direct";
      }): unknown;
      list(input: { subjectEntityId: ReturnType<Borg["entities"]["resolve"]> }): Array<{
        slot_key: string;
        state: string;
        contradicted_by_stream_entry_ids: string[];
      }>;
    };
    streamIngestionCoordinator?: {
      options: {
        extractor: {
          extractFromStream(): Promise<{ inserted: number; updated: number; skipped: number }>;
        };
        entryProcessor?: {
          processUserEntry(sessionId: SessionId, entry: StreamEntry): Promise<void>;
        };
      };
    };
  };
};

function installNoopEpisodicExtraction(
  borg: Borg,
  result: { inserted: number; updated: number; skipped: number } = {
    inserted: 0,
    updated: 0,
    skipped: 0,
  },
): void {
  const coordinator =
    borgInternals<IngestionCoordinatorInternals>(borg).deps.streamIngestionCoordinator;

  if (coordinator === undefined) {
    throw new Error("expected live ingestion coordinator");
  }

  coordinator.options.extractor = {
    extractFromStream: async () => result,
  };
}

async function appendTurn(input: {
  borg: Borg;
  sessionId: SessionId;
  user: string;
  senderEntityId?: ReturnType<Borg["entities"]["resolve"]>;
}): Promise<StreamEntry[]> {
  return input.borg.stream.appendMany(
    [
      {
        kind: "user_msg",
        content: input.user,
        ...(input.senderEntityId === undefined ? {} : { sender_entity_id: input.senderEntityId }),
      },
      { kind: "agent_msg", content: "Acknowledged." },
    ],
    { session: input.sessionId },
  );
}

describe("sidecar corrective-preference ingestion wiring", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  async function openHarness(input: { enabled: boolean; budget?: number | null }): Promise<{
    borg: Borg;
    llm: FakeLLMClient;
    traces: CallbackTraceEntry[];
  }> {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-sidecar-commitments-"));
    tempDirs.push(dataDir);
    const llm = new FakeLLMClient();
    const traces: CallbackTraceEntry[] = [];
    const tracer = new CallbackTracer({
      includePayloads: true,
      timestamp: () => 1_000,
      sink: (entry) => traces.push(entry),
    });
    const borg = await Borg.open({
      config: createTestConfig({
        dataDir,
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "fake-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "qwen-cognition",
            background: "qwen-background",
            extraction: "qwen-extraction",
            recallExpansion: "qwen-fast",
            correctivePreference: "qwen-corrective-preference",
          },
        },
      }),
      clock: new ManualClock(1_000),
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: llm,
      liveExtraction: true,
      liveCommitmentExtraction: input.enabled,
      liveCommitmentExtractionBudget: input.budget ?? null,
      tracer,
    });

    installNoopEpisodicExtraction(borg);
    return { borg, llm, traces };
  }

  it("classifies ordinary turns, persists sender-scoped boundaries, and retires the active rule", async () => {
    const { borg, llm, traces } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice", {
      kind: "person",
      provenance: "transport_sender",
    });

    try {
      llm.pushResponse(noneResponse());
      await appendTurn({
        borg,
        sessionId,
        user: "The weather is pleasant today.",
        senderEntityId,
      });
      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
        processedEntries: 2,
      });

      expect(borg.commitments.list({ activeOnly: true, audienceEntityId: senderEntityId })).toEqual(
        [],
      );
      expect(
        llm.requests.filter((request) => request.budget === "corrective-preference-extractor"),
      ).toHaveLength(1);
      expect(traces).toContainEqual(
        expect.objectContaining({
          event: "corrective_preference.ingestion.completed",
          outcome: "none",
          tokens_used: 7,
          audience_entity_id: senderEntityId,
        }),
      );

      const correctiveEntries = await appendTurn({
        borg,
        sessionId,
        user: "From now on, never disclose my private details outside this audience.",
        senderEntityId,
      });
      llm.pushResponse(correctivePreferenceResponse(null, correctiveEntries[0]?.id ?? null));
      await borg.episodic.ingest({ session: sessionId });

      const [commitment] = borg.commitments.list({
        activeOnly: true,
        audienceEntityId: senderEntityId,
      });

      expect(commitment).toMatchObject({
        type: "boundary",
        kind: "boundary",
        enforcement_class: "critical",
        critical_domain: "audience_scope",
        directive: "Never disclose my private details outside this audience.",
        directive_family: "private_audience_scope",
        priority: 90,
        restricted_audience: senderEntityId,
        committed_by_entity_id: senderEntityId,
        source_stream_entry_ids: [correctiveEntries[0]?.id],
        provenance: {
          kind: "online",
          process: "corrective-preference-extractor",
        },
      });

      if (commitment === undefined) {
        throw new Error("expected extracted commitment");
      }

      const correctiveRequest = llm.requests.filter(
        (request) => request.budget === "corrective-preference-extractor",
      )[1];
      const correctivePayload = JSON.parse(correctiveRequest?.messages[0]?.content ?? "{}") as {
        audience_entity_id?: string;
        speaker_entity_id?: string;
        speaker_display_name?: string;
        cross_audience_targets?: unknown[];
        recent_history?: unknown[];
      };

      expect(correctiveRequest?.model).toBe("qwen-corrective-preference");
      expect(correctivePayload).toMatchObject({
        audience_entity_id: senderEntityId,
        speaker_entity_id: senderEntityId,
        speaker_display_name: "Alice",
        cross_audience_targets: [],
      });
      expect(correctivePayload.recent_history).toHaveLength(2);

      llm.pushResponse(retirementResponse(commitment.id));
      await appendTurn({
        borg,
        sessionId,
        user: "You can stand down that medical-details rule now.",
        senderEntityId,
      });
      await borg.episodic.ingest({ session: sessionId });

      expect(borg.commitments.list({ activeOnly: true, audienceEntityId: senderEntityId })).toEqual(
        [],
      );
      expect(borg.commitments.get(commitment.id)).toMatchObject({
        revoked_reason: "The sender explicitly stood down the supplied rule.",
        revoke_provenance: {
          kind: "online",
          process: "corrective-preference-extractor",
        },
      });

      const retirementRequest = llm.requests.filter(
        (request) => request.budget === "corrective-preference-extractor",
      )[2];
      const retirementPayload = JSON.parse(retirementRequest?.messages[0]?.content ?? "{}") as {
        active_commitments?: Array<{ id: string }>;
      };
      expect(retirementPayload.active_commitments).toContainEqual(
        expect.objectContaining({ id: commitment.id }),
      );
    } finally {
      await borg.close();
    }
  });

  it("spends no classifier tokens when a user entry has no sender identity", async () => {
    const { borg, llm, traces } = await openHarness({ enabled: true });
    const sessionId = createSessionId();

    try {
      await appendTurn({ borg, sessionId, user: "No sender was stamped." });
      await borg.episodic.ingest({ session: sessionId });

      expect(llm.requests).toEqual([]);
      expect(traces).toContainEqual(
        expect.objectContaining({
          event: "corrective_preference.ingestion.completed",
          outcome: "skipped",
          reason: "missing_sender_entity_id",
          tokens_used: 0,
        }),
      );
    } finally {
      await borg.close();
    }
  });

  it("holds the shared watermark when the classifier degrades, then retries the same entry", async () => {
    const { borg, traces } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;
    const processor = internals.streamIngestionCoordinator?.options.entryProcessor;

    if (processor === undefined) {
      throw new Error("expected corrective-preference entry processor");
    }

    const processUserEntry = vi
      .spyOn(processor, "processUserEntry")
      .mockRejectedValueOnce(
        new CorrectivePreferenceExtractorDegradedError("llm_failed", new Error("gateway down")),
      )
      .mockResolvedValueOnce();

    try {
      const entries = await appendTurn({
        borg,
        sessionId,
        user: "Never disclose my diagnosis.",
        senderEntityId,
      });
      const userEntryId = entries[0]?.id;

      if (userEntryId === undefined) {
        throw new Error("expected user stream entry");
      }

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: false,
        processedEntries: 2,
        error: expect.any(CorrectivePreferenceExtractorDegradedError),
      });
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(userEntryId)).toEqual(
        expect.objectContaining({ status: "retryable", failure_count: 1 }),
      );

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
        processedEntries: 3,
      });
      expect(processUserEntry).toHaveBeenCalledTimes(2);
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(userEntryId)).toEqual(
        expect.objectContaining({ status: "processed", failure_count: 1 }),
      );
      expect(traces).toContainEqual(
        expect.objectContaining({
          event: "corrective_preference.ingestion.retry_scheduled",
          source_stream_entry_id: userEntryId,
          failure_count: 1,
        }),
      );
    } finally {
      await borg.close();
    }
  });

  it("dead-letters a persistent poison entry at the ceiling and continues the stream", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const { borg, traces } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;
    const processor = internals.streamIngestionCoordinator?.options.entryProcessor;

    if (processor === undefined) {
      throw new Error("expected corrective-preference entry processor");
    }

    const attempts = new Map<string, number>();
    vi.spyOn(processor, "processUserEntry").mockImplementation(async (_sessionId, entry) => {
      attempts.set(entry.id, (attempts.get(entry.id) ?? 0) + 1);

      if (entry.content === "poison") {
        throw new Error("persistent classifier poison");
      }
    });

    try {
      const poisonEntries = await appendTurn({
        borg,
        sessionId,
        user: "poison",
        senderEntityId,
      });
      const healthyEntries = await appendTurn({
        borg,
        sessionId,
        user: "healthy",
        senderEntityId,
      });
      const poisonId = poisonEntries[0]?.id;
      const healthyId = healthyEntries[0]?.id;

      if (poisonId === undefined || healthyId === undefined) {
        throw new Error("expected user stream entries");
      }

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: false,
      });
      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: false,
      });
      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
        processedEntries: 6,
      });

      expect(attempts.get(poisonId)).toBe(3);
      expect(attempts.get(healthyId)).toBe(1);
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(poisonId)).toEqual(
        expect.objectContaining({
          status: "dead_letter",
          failure_count: 3,
          last_error: expect.stringContaining("persistent classifier poison"),
        }),
      );
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(healthyId)).toEqual(
        expect.objectContaining({ status: "processed" }),
      );
      expect(traces).toContainEqual(
        expect.objectContaining({
          event: "corrective_preference.ingestion.dead_lettered",
          source_stream_entry_id: poisonId,
          reason: "retry_ceiling_reached",
          error: expect.stringContaining("persistent classifier poison"),
        }),
      );
      expect(consoleError).toHaveBeenCalledWith(
        "Corrective-preference ingestion entry dead-lettered",
        expect.objectContaining({ source_stream_entry_id: poisonId }),
      );

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: false,
        processedEntries: 0,
      });
    } finally {
      await borg.close();
    }
  });

  it("skips durable successes when a later batch entry forces replay", async () => {
    const { borg } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;
    const processor = internals.streamIngestionCoordinator?.options.entryProcessor;

    if (processor === undefined) {
      throw new Error("expected corrective-preference entry processor");
    }

    const attempts = new Map<string, number>();
    let failedSecond = false;
    vi.spyOn(processor, "processUserEntry").mockImplementation(async (_sessionId, entry) => {
      attempts.set(entry.id, (attempts.get(entry.id) ?? 0) + 1);

      if (entry.content === "second" && !failedSecond) {
        failedSecond = true;
        throw new Error("fail later entry once");
      }
    });

    try {
      const first = await appendTurn({ borg, sessionId, user: "first", senderEntityId });
      const second = await appendTurn({ borg, sessionId, user: "second", senderEntityId });
      const firstId = first[0]?.id;
      const secondId = second[0]?.id;

      if (firstId === undefined || secondId === undefined) {
        throw new Error("expected user stream entries");
      }

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: false,
      });
      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
      });

      expect(attempts.get(firstId)).toBe(1);
      expect(attempts.get(secondId)).toBe(2);
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(firstId)).toEqual(
        expect.objectContaining({ status: "processed" }),
      );
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(secondId)).toEqual(
        expect.objectContaining({ status: "processed", failure_count: 1 }),
      );
    } finally {
      await borg.close();
    }
  });

  it("immediately dead-letters a post-spend budget overflow and advances the watermark", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const { borg, llm, traces } = await openHarness({ enabled: true, budget: 5 });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;

    try {
      llm.pushResponse(correctivePreferenceResponse());
      const entries = await appendTurn({
        borg,
        sessionId,
        user: "Never disclose my diagnosis.",
        senderEntityId,
      });
      const userEntryId = entries[0]?.id;

      if (userEntryId === undefined) {
        throw new Error("expected user stream entry");
      }

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
        processedEntries: 2,
      });
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(userEntryId)).toEqual(
        expect.objectContaining({ status: "dead_letter", failure_count: 1 }),
      );
      expect(borg.commitments.list({ activeOnly: true })).toEqual([]);
      expect(traces).toContainEqual(
        expect.objectContaining({
          event: "corrective_preference.ingestion.dead_lettered",
          source_stream_entry_id: userEntryId,
          reason: "budget_exceeded",
        }),
      );
      expect(consoleError).toHaveBeenCalled();

      await borg.episodic.ingest({ session: sessionId });
      expect(
        llm.requests.filter((request) => request.budget === "corrective-preference-extractor"),
      ).toHaveLength(1);
    } finally {
      await borg.close();
    }
  });

  it("classifies salience-skipped user turns intentionally", async () => {
    const { borg, llm, traces } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    installNoopEpisodicExtraction(borg, { inserted: 0, updated: 0, skipped: 2 });

    try {
      llm.pushResponse(noneResponse());
      await appendTurn({
        borg,
        sessionId,
        user: "Ordinary but still safety-classified.",
        senderEntityId,
      });
      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
        extractionResult: { inserted: 0, updated: 0, skipped: 2 },
      });

      expect(
        llm.requests.filter((request) => request.budget === "corrective-preference-extractor"),
      ).toHaveLength(1);
      expect(traces).toContainEqual(
        expect.objectContaining({
          event: "corrective_preference.ingestion.completed",
          outcome: "none",
        }),
      );
    } finally {
      await borg.close();
    }
  });

  it("applies extractor slot negations through the cognition service", async () => {
    const { borg, llm } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const relationalSlots =
      borgInternals<IngestionCoordinatorInternals>(borg).deps.relationalSlotRepository;
    relationalSlots.applyAssertion({
      subject_entity_id: senderEntityId,
      slot_key: "relationship_to_borg",
      asserted_value: "manager",
      source_stream_entry_ids: [createStreamEntryId()],
      confirmation: "direct",
    });

    try {
      const entries = await appendTurn({
        borg,
        sessionId,
        user: "I am not your manager.",
        senderEntityId,
      });
      const userEntryId = entries[0]?.id;

      if (userEntryId === undefined) {
        throw new Error("expected user stream entry");
      }

      llm.pushResponse(
        noneResponse([
          {
            subject_entity_id: senderEntityId,
            slot_key: "relationship_to_borg",
            rejected_value: "manager",
            source_stream_entry_ids: [userEntryId],
            confidence: 0.99,
          },
        ]),
      );
      await borg.episodic.ingest({ session: sessionId });

      expect(relationalSlots.list({ subjectEntityId: senderEntityId })).toContainEqual(
        expect.objectContaining({
          slot_key: "relationship_to_borg",
          state: "quarantined",
          contradicted_by_stream_entry_ids: [userEntryId],
        }),
      );
    } finally {
      await borg.close();
    }
  });

  it("documents that a batched classifier sees the post-episodic relational-slot snapshot", async () => {
    const { borg, llm } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;
    const relationalSlots = internals.relationalSlotRepository;
    const coordinator = internals.streamIngestionCoordinator;

    if (coordinator === undefined) {
      throw new Error("expected live ingestion coordinator");
    }

    try {
      const earlier = await appendTurn({
        borg,
        sessionId,
        user: "That later color assertion is wrong.",
        senderEntityId,
      });
      const later = await appendTurn({ borg, sessionId, user: "later turn", senderEntityId });
      const earlierUserEntryId = earlier[0]?.id;
      const laterUserEntryId = later[0]?.id;

      if (earlierUserEntryId === undefined || laterUserEntryId === undefined) {
        throw new Error("expected batched user stream entries");
      }

      coordinator.options.extractor = {
        extractFromStream: async () => {
          relationalSlots.applyAssertion({
            subject_entity_id: senderEntityId,
            slot_key: "favorite_color",
            asserted_value: "green-from-later-turn",
            source_stream_entry_ids: [laterUserEntryId],
            confirmation: "direct",
          });
          return { inserted: 0, updated: 0, skipped: 0 };
        },
      };
      llm.pushResponse(
        noneResponse([
          {
            subject_entity_id: senderEntityId,
            slot_key: "favorite_color",
            rejected_value: "green-from-later-turn",
            source_stream_entry_ids: [earlierUserEntryId],
            confidence: 0.99,
          },
        ]),
      );
      llm.pushResponse(noneResponse());

      await borg.episodic.ingest({ session: sessionId });

      const firstClassifierRequest = llm.requests.find(
        (request) => request.budget === "corrective-preference-extractor",
      );
      const payload = JSON.parse(firstClassifierRequest?.messages[0]?.content ?? "{}") as {
        relational_slots?: Array<{ slot_key: string; value: string }>;
      };
      expect(payload.relational_slots).toContainEqual(
        expect.objectContaining({
          slot_key: "favorite_color",
          value: "green-from-later-turn",
        }),
      );
      expect(relationalSlots.list({ subjectEntityId: senderEntityId })).toContainEqual(
        expect.objectContaining({
          slot_key: "favorite_color",
          state: "quarantined",
          contradicted_by_stream_entry_ids: [earlierUserEntryId],
        }),
      );
    } finally {
      await borg.close();
    }
  });

  it("serializes concurrent family writes into one active sender-scoped commitment", async () => {
    const { borg, llm } = await openHarness({ enabled: true });
    const firstSession = createSessionId();
    const secondSession = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;

    try {
      llm.pushResponse(correctivePreferenceResponse());
      llm.pushResponse(correctivePreferenceResponse());
      const first = await appendTurn({
        borg,
        sessionId: firstSession,
        user: "Never disclose my private details.",
        senderEntityId,
      });
      const second = await appendTurn({
        borg,
        sessionId: secondSession,
        user: "Never disclose my private details.",
        senderEntityId,
      });

      await Promise.all([
        borg.episodic.ingest({ session: firstSession }),
        borg.episodic.ingest({ session: secondSession }),
      ]);

      const active = borg.commitments.list({
        activeOnly: true,
        audienceEntityId: senderEntityId,
      });
      expect(active).toHaveLength(1);
      expect(active[0]?.source_stream_entry_ids).toEqual(
        expect.arrayContaining([first[0]?.id, second[0]?.id]),
      );
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(first[0]!.id)).toEqual(
        expect.objectContaining({ status: "processed" }),
      );
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(second[0]!.id)).toEqual(
        expect.objectContaining({ status: "processed" }),
      );
    } finally {
      await borg.close();
    }
  });

  it("propagates commitment persistence failure into a receipt-backed retry", async () => {
    const { borg, llm } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;
    const addCommitment = vi
      .spyOn(internals.identityService, "addCommitment")
      .mockImplementationOnce(() => {
        throw new Error("simulated commitment insert failure");
      });

    try {
      llm.pushResponse(correctivePreferenceResponse());
      const entries = await appendTurn({
        borg,
        sessionId,
        user: "Never disclose my private details.",
        senderEntityId,
      });
      const userEntryId = entries[0]?.id;

      if (userEntryId === undefined) {
        throw new Error("expected user stream entry");
      }

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: false,
        error: expect.objectContaining({ message: "simulated commitment insert failure" }),
      });
      expect(borg.commitments.list({ activeOnly: true })).toEqual([]);
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(userEntryId)).toEqual(
        expect.objectContaining({ status: "retryable", failure_count: 1 }),
      );

      addCommitment.mockRestore();
      llm.pushResponse(correctivePreferenceResponse());
      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
      });
      expect(
        borg.commitments.list({ activeOnly: true, audienceEntityId: senderEntityId }),
      ).toHaveLength(1);
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(userEntryId)).toEqual(
        expect.objectContaining({ status: "processed", failure_count: 1 }),
      );
    } finally {
      await borg.close();
    }
  });

  it("rolls back a failed supersession and retries without a duplicate active commitment", async () => {
    const { borg, llm } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;
    const original = borg.commitments.add({
      type: "boundary",
      kind: "boundary",
      enforcementClass: "critical",
      criticalDomain: "audience_scope",
      directiveFamily: "private_audience_scope",
      directive: "Keep my private details in this audience.",
      priority: 85,
      audience: "Alice",
      provenance: { kind: "manual" },
    });
    const supersede = vi
      .spyOn(internals.commitmentRepository, "supersede")
      .mockImplementationOnce(() => {
        throw new Error("simulated supersession write failure");
      });

    try {
      llm.pushResponse(correctivePreferenceResponse(original.id));
      const entries = await appendTurn({
        borg,
        sessionId,
        user: "Replace that privacy rule with: never disclose private details.",
        senderEntityId,
      });
      const userEntryId = entries[0]?.id;

      if (userEntryId === undefined) {
        throw new Error("expected user stream entry");
      }

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: false,
        error: expect.objectContaining({ message: "simulated supersession write failure" }),
      });
      expect(borg.commitments.list({ activeOnly: true, audienceEntityId: senderEntityId })).toEqual(
        [expect.objectContaining({ id: original.id })],
      );
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(userEntryId)).toEqual(
        expect.objectContaining({ status: "retryable", failure_count: 1 }),
      );

      supersede.mockRestore();
      llm.pushResponse(correctivePreferenceResponse(original.id));
      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
      });

      const active = borg.commitments.list({
        activeOnly: true,
        audienceEntityId: senderEntityId,
      });
      expect(active).toHaveLength(1);
      expect(active[0]).toMatchObject({
        directive: "Never disclose my private details outside this audience.",
      });
      expect(active[0]?.id).not.toBe(original.id);
      expect(borg.commitments.get(original.id)).toMatchObject({
        superseded_by: active[0]?.id,
      });
    } finally {
      await borg.close();
    }
  });

  it("propagates retirement persistence failure and retries the active commitment", async () => {
    const { borg, llm } = await openHarness({ enabled: true });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const internals = borgInternals<IngestionCoordinatorInternals>(borg).deps;
    const commitment = borg.commitments.add({
      type: "boundary",
      kind: "boundary",
      enforcementClass: "critical",
      criticalDomain: "audience_scope",
      directiveFamily: "medical_privacy",
      directive: "Keep medical details private.",
      priority: 90,
      audience: "Alice",
      provenance: { kind: "manual" },
    });
    const revoke = vi.spyOn(internals.commitmentRepository, "revoke").mockImplementationOnce(() => {
      throw new Error("simulated retirement write failure");
    });

    try {
      llm.pushResponse(retirementResponse(commitment.id));
      const entries = await appendTurn({
        borg,
        sessionId,
        user: "Stand down that medical privacy rule.",
        senderEntityId,
      });
      const userEntryId = entries[0]?.id;

      if (userEntryId === undefined) {
        throw new Error("expected user stream entry");
      }

      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: false,
        error: expect.objectContaining({ message: "simulated retirement write failure" }),
      });
      expect(borg.commitments.get(commitment.id)?.revoked_at).toBeNull();
      expect(internals.entryIndex.getCorrectivePreferenceIngestionReceipt(userEntryId)).toEqual(
        expect.objectContaining({ status: "retryable", failure_count: 1 }),
      );

      revoke.mockRestore();
      llm.pushResponse(retirementResponse(commitment.id));
      await expect(borg.episodic.ingest({ session: sessionId })).resolves.toMatchObject({
        ran: true,
      });
      expect(borg.commitments.get(commitment.id)?.revoked_at).not.toBeNull();
    } finally {
      await borg.close();
    }
  });

  it("does not install the sidecar processor when live commitment extraction is disabled", async () => {
    const { borg, llm } = await openHarness({ enabled: false });
    const sessionId = createSessionId();
    const senderEntityId = borg.entities.resolve("Alice");
    const coordinator =
      borgInternals<IngestionCoordinatorInternals>(borg).deps.streamIngestionCoordinator;

    try {
      expect(coordinator?.options.entryProcessor).toBeUndefined();
      await appendTurn({ borg, sessionId, user: "Never expose secrets.", senderEntityId });
      await borg.episodic.ingest({ session: sessionId });
      expect(llm.requests).toEqual([]);
    } finally {
      await borg.close();
    }
  });
});
