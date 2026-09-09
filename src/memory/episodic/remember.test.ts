import { afterEach, describe, expect, it, vi } from "vitest";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { LLMCompleteResult } from "../../llm/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  type OfflineTestHarness,
} from "../../offline/test-support.js";
import { ConsolidatorProcess } from "../../offline/consolidator/index.js";
import { buildEpisodeEmbeddingText } from "./protected-lines.js";
import { RetrievalPipeline } from "../../retrieval/pipeline.js";
import { StreamWriter, hydrateStreamEntriesById } from "../../stream/index.js";
import { DEFAULT_SESSION_ID, createSessionId, createStreamEntryId } from "../../util/ids.js";
import { memoryDisclosureLabelFromEpisodeAccess } from "../common/disclosure-label.js";
import { TenantFactRememberer, tenantFactAuthorizationFromEntry } from "./remember.js";
import type { RememberTenantFactInput } from "./types.js";

const FACT = "Marcin będzie na urlopie od 14 do 18 września 2026.";
const CONSENT = "Mam urlop od 14 do 18 września 2026. Cały zespół może o tym wiedzieć.";

// Reuses the existing indexed-stream/LanceDB harness; the model interprets consent,
// while these tests exercise grounding, write durability, and downstream labels.
describe("rememberForTenant", () => {
  let harness: OfflineTestHarness;
  afterEach(async () => {
    await harness?.cleanup();
  });

  async function setup(authorized = true) {
    const response: LLMCompleteResult = {
      text: "",
      input_tokens: 10,
      output_tokens: 10,
      stop_reason: "tool_use",
      tool_calls: [
        {
          id: "consent",
          name: "ExtractAuthorizedTenantFact",
          input: {
            authorized,
            fact: authorized ? FACT : "",
            title: authorized ? "Urlop Marcina" : "",
            tags: authorized ? ["Marcin", "urlop"] : [],
            confidence: authorized ? 1 : 0,
          },
        },
      ],
    };
    const llmClient = new FakeLLMClient({ responses: [response] });
    harness = await createOfflineTestHarness({ llmClient });
    const speaker = harness.entityRepository.add({ canonicalName: "Marcin", kind: "person" });
    const other = harness.entityRepository.add({ canonicalName: "Tomasz", kind: "person" });
    const entryIndex = harness.createContext().entryIndex!;
    const source = await harness.streamWriter.append({
      kind: "user_msg",
      content: CONSENT,
      sender_entity_id: speaker.id,
      audience: speaker.id,
    });
    const privateEpisode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture({
        title: "Prywatna rozmowa",
        narrative: `${CONSENT} Prywatne powody wyjazdu.`,
        source_stream_ids: [source.id],
        audience_entity_id: speaker.id,
        origin_audience_entity_ids: [speaker.id],
        shared: false,
      }),
    );
    const options = {
      dataDir: harness.tempDir,
      entryIndex,
      episodicRepository: harness.episodicRepository,
      entityRepository: harness.entityRepository,
      embeddingClient: harness.embeddingClient,
      clock: harness.clock,
      model: "configured-extraction-slot",
      timeZone: "Europe/Warsaw",
      llmFactory: () => llmClient,
      createStreamWriter: (sessionId: typeof DEFAULT_SESSION_ID) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          entryIndex,
          sessionId,
          clock: harness.clock,
        }),
    };
    const input: RememberTenantFactInput = {
      requestId: "vacation-consent",
      sessionId: source.session_id,
      speakerEntityId: speaker.id,
      content: FACT,
      sourceEpisodeIds: [privateEpisode.id],
      sourceMessageIds: [source.id],
      authorizationMessageIds: [source.id],
    };
    return {
      speaker,
      other,
      source,
      privateEpisode,
      input,
      entryIndex,
      options,
      llmClient,
      service: new TenantFactRememberer(options),
    };
  }

  it("creates a separate authorized fact, leaves the private original intact, and recalls both with their own labels", async () => {
    const { service, input, source, privateEpisode, other, entryIndex, llmClient } = await setup();
    const before = await harness.episodicRepository.get(privateEpisode.id);
    const result = await service.remember(input);
    const publicEpisode = await harness.episodicRepository.get(result.episodeId);
    expect(result.episodeId).not.toBe(privateEpisode.id);
    expect(publicEpisode).toMatchObject({
      shared: true,
      origin_audience_entity_ids: [],
      source_stream_ids: [result.authorizationEntryId],
      lineage: { derived_from: [privateEpisode.id] },
    });
    expect(publicEpisode?.narrative).toContain(FACT);
    expect(publicEpisode?.narrative).toContain("Marcin's explicit authorization");
    expect(publicEpisode?.narrative).not.toContain("Prywatne powody");
    expect(await harness.episodicRepository.get(privateEpisode.id)).toEqual(before);
    expect(memoryDisclosureLabelFromEpisodeAccess(publicEpisode!)).toMatchObject({
      disclosureClass: "public",
    });
    expect(memoryDisclosureLabelFromEpisodeAccess(before!)).toMatchObject({
      disclosureClass: "relationship_private",
    });
    const receipts = await hydrateStreamEntriesById({
      dataDir: harness.tempDir,
      sessionId: source.session_id,
      streamEntryIds: [result.authorizationEntryId],
      entryIndex,
    });
    expect(
      tenantFactAuthorizationFromEntry(receipts.get(result.authorizationEntryId)!),
    ).toMatchObject({
      scope: "tenant",
      speaker_entity_id: input.speakerEntityId,
      source_episode_ids: [privateEpisode.id],
      source_message_ids: [source.id],
      authorization_message_ids: [source.id],
      fact: FACT,
    });
    expect(llmClient.requests[0]).toMatchObject({ model: "configured-extraction-slot" });
    const prompt = JSON.stringify(llmClient.requests[0]);
    expect(prompt).toContain(CONSENT);
    expect(prompt).toContain(new Date(source.timestamp).toISOString());
    expect(prompt).toContain("Europe/Warsaw");
    expect(prompt).toContain("relationship_private");
    expect(prompt).toContain("Only the designated authorization messages can grant permission");

    // Force only query vectors, so the assertion isolates cognition recall and disclosure.
    const pipeline = new RetrievalPipeline({
      episodicRepository: harness.episodicRepository,
      embeddingClient: {
        profile: harness.embeddingClient.profile,
        embed: async () => publicEpisode!.embedding,
        embedBatch: async (texts) => texts.map(() => publicEpisode!.embedding),
      },
      dataDir: harness.tempDir,
      entryIndex,
      clock: harness.clock,
    });
    for (const [kind, query] of [
      ["group", "Czy Marcin będzie w następnym tygodniu w pracy?"],
      ["private", "Czy Marcin będzie w przyszłym tygodniu w pracy, czy idzie na urlop?"],
    ] as const) {
      const audience =
        kind === "private"
          ? other
          : harness.entityRepository.add({ canonicalName: "AI Ninjas", kind: "group" });
      const hits = await pipeline.recallEpisodeHitsForCognition(query, {
        limit: 8,
        recordRetrieval: false,
        recallContext: {
          reader: "self",
          currentSessionId: createSessionId(),
          currentAudienceEntityId: audience.id,
          currentParticipantEntityIds: [other.id],
        },
        entityTerms: ["Urlop Marcina", "Prywatna rozmowa"],
      });
      const recalled = hits.find((hit) => hit.episode.id === result.episodeId);
      expect(recalled?.disclosureLabel).toMatchObject({ disclosureClass: "public" });
      expect(recalled?.citationChain.map((entry) => entry.id)).toEqual([
        result.authorizationEntryId,
      ]);
      expect(
        hits.find((hit) => hit.episode.id === privateEpisode.id)?.disclosureLabel,
      ).toMatchObject({ disclosureClass: "relationship_private" });
    }
    expect(llmClient.requests).toHaveLength(1);
  });

  it("keeps the authorized public fact recallable after consolidation with its private source", async () => {
    const { service, input, privateEpisode, entryIndex, other } = await setup();
    const privateBefore = await harness.episodicRepository.get(privateEpisode.id);
    const embed = vi
      .spyOn(harness.embeddingClient, "embed")
      .mockResolvedValue(privateEpisode.embedding);
    const remembered = await service.remember(input);
    const publicEpisode = (await harness.episodicRepository.get(remembered.episodeId))!;
    expect(embed).toHaveBeenCalledWith(buildEpisodeEmbeddingText(publicEpisode));
    const context = harness.createContext();
    context.llm.background = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "merge",
              name: "EmitConsolidation",
              input: {
                title: "Urlop Marcina",
                narrative: `${FACT} Prywatne powody wyjazdu.`,
              },
            },
          ],
        },
      ],
    });
    const consolidator = new ConsolidatorProcess({
      episodicRepository: harness.episodicRepository,
      registry: harness.registry,
    });
    const consolidated = await consolidator.run(context, { dryRun: false });
    expect(consolidated.errors).toEqual([]);
    const families = harness.episodicRepository.listConsolidationFamilies();
    expect(families).toHaveLength(1);
    const mergedId = families[0]!.current_version_episode_id;
    const merged = (await harness.episodicRepository.get(mergedId))!;
    expect(merged.source_stream_ids).toContain(remembered.authorizationEntryId);
    expect(memoryDisclosureLabelFromEpisodeAccess(merged).disclosureClass).toBe(
      "relationship_private",
    );
    expect(await harness.episodicRepository.get(remembered.episodeId)).toEqual(publicEpisode);
    expect(
      await harness.episodicRepository.get(privateEpisode.id, { includeArchived: true }),
    ).toEqual(privateBefore);
    // Public retention does not cause this already-covered pair to be consolidated again.
    expect((await consolidator.run(context, { dryRun: false })).changes).toEqual([]);

    const pipeline = new RetrievalPipeline({
      episodicRepository: harness.episodicRepository,
      embeddingClient: harness.embeddingClient,
      dataDir: harness.tempDir,
      entryIndex,
      clock: harness.clock,
    });
    for (const query of [
      "Czy Marcin będzie w następnym tygodniu w pracy?",
      "Czy Marcin będzie w przyszłym tygodniu w pracy, czy idzie na urlop?",
    ]) {
      const hits = await pipeline.recallEpisodeHitsForCognition(query, {
        limit: 8,
        recordRetrieval: false,
        recallContext: {
          reader: "self",
          currentAudienceEntityId: other.id,
          currentSessionId: createSessionId(),
          currentParticipantEntityIds: [other.id],
        },
      });
      const factHit = hits.find((hit) => hit.episode.id === remembered.episodeId)!;
      expect(factHit.disclosureLabel?.disclosureClass).toBe("public");
      expect(factHit.episode.narrative).not.toContain("Prywatne powody");
      expect(tenantFactAuthorizationFromEntry(factHit.citationChain[0]!)).toMatchObject({
        fact: FACT,
      });
      expect(
        hits.find((hit) => hit.episode.id === mergedId)?.disclosureLabel?.disclosureClass,
      ).toBe("relationship_private");
      expect(
        hits
          .find((hit) => hit.episode.id === mergedId)
          ?.citationChain.map(tenantFactAuthorizationFromEntry)
          .filter((authorization) => authorization !== null),
      ).toEqual([expect.objectContaining({ fact: FACT, episode_id: remembered.episodeId })]);
    }
  });

  it("rejects model-denied consent without writing an authorization or public episode", async () => {
    const { service, input, entryIndex } = await setup(false);
    const count = entryIndex.nextEntryIndex(DEFAULT_SESSION_ID);
    await expect(service.remember(input)).rejects.toMatchObject({
      code: "MEMORY_REMEMBER_NOT_AUTHORIZED",
    });
    expect(entryIndex.nextEntryIndex(DEFAULT_SESSION_ID)).toBe(count);
    expect((await harness.episodicRepository.list()).items).toHaveLength(1);
  });

  it.each([
    "wrong-speaker",
    "wrong-session",
    "missing-message",
    "inactive-message",
    "pending-message",
    "missing-episode",
  ])("rejects %s provenance before invoking the model", async (scenario) => {
    const { service, input, source, other, llmClient, entryIndex } = await setup();
    if (scenario === "wrong-speaker") input.speakerEntityId = other.id;
    if (scenario === "wrong-session") input.sessionId = createSessionId();
    if (scenario === "missing-message") input.sourceMessageIds = [createStreamEntryId(), source.id];
    if (scenario === "inactive-message")
      harness.db
        .prepare("UPDATE stream_entry_index SET active = 0 WHERE entry_id = ?")
        .run(source.id);
    if (scenario === "pending-message") entryIndex.setReceiptPending(source.id, true);
    if (scenario === "missing-episode") input.sourceEpisodeIds = [createEpisodeFixture().id];
    await expect(service.remember(input)).rejects.toMatchObject({
      code: "MEMORY_REMEMBER_SOURCE_INVALID",
    });
    expect(llmClient.requests).toHaveLength(0);
  });

  it("requires at least one authorization message, included in the source messages", async () => {
    const { service, input, llmClient } = await setup();
    await expect(service.remember({ ...input, authorizationMessageIds: [] })).rejects.toThrow();
    await expect(
      service.remember({ ...input, authorizationMessageIds: [createStreamEntryId()] }),
    ).rejects.toThrow();
    expect(llmClient.requests).toHaveLength(0);
  });

  it("coalesces concurrent retries, persists idempotency across service instances, and rejects changed payloads", async () => {
    const { service, input, options, llmClient } = await setup();
    const [first, second] = await Promise.all([service.remember(input), service.remember(input)]);
    expect(second).toEqual({ ...first, duplicate: true });
    const reopened = new TenantFactRememberer(options);
    expect(await reopened.remember(input)).toEqual(second);
    await expect(
      reopened.remember({ ...input, content: "A different fact" }),
    ).rejects.toMatchObject({ code: "MEMORY_REMEMBER_CONFLICT" });
    expect(llmClient.requests).toHaveLength(1);
    expect((await harness.episodicRepository.list()).items).toHaveLength(2);
  });

  it("resumes from the fsync'd receipt after a failed embedding, and a retry never resurrects a forgotten fact", async () => {
    const { service, input, options, llmClient } = await setup();
    const embedding = vi
      .spyOn(harness.embeddingClient, "embed")
      .mockRejectedValueOnce(new Error("embedding unavailable"));
    await expect(service.remember(input)).rejects.toThrow("embedding unavailable");
    const result = await new TenantFactRememberer(options).remember(input);
    expect(result.duplicate).toBe(true);
    expect(llmClient.requests).toHaveLength(1);
    expect(embedding).toHaveBeenCalledTimes(2);
    harness.episodicRepository.archiveEpisode(result.episodeId, {
      caller: "remember.test",
      reason: "Consent withdrawn",
      process: "correction",
    });
    await service.remember(input);
    expect(await harness.episodicRepository.get(result.episodeId)).toBeNull();
    expect(embedding).toHaveBeenCalledTimes(2);
  });
});
