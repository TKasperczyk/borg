import { describe, expect, it, vi } from "vitest";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
} from "../offline/test-support.js";
import { FixedClock } from "../util/clock.js";
import { parseEpisodeId, parseStreamEntryId } from "../util/ids.js";

import * as evidenceProjections from "./evidence-projections.js";
import { RetrievalPipeline } from "./pipeline.js";

const QUERY = "architecture";
const NOW_MS = 10_000_000_000;

// Pins the RetrievalProjection contract (see pipeline.ts): the episodes-only
// entry must never pay for the context lanes it provably cannot surface, and
// the full-context entries must keep running them. If a lane ever starts
// feeding projectEpisodes, this test is the tripwire that forces the skip to
// be re-justified.
describe("RetrievalProjection episodes-only", () => {
  async function createHarness() {
    return createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: new TestEmbeddingClient(new Map([[QUERY, [1, 0, 0, 0]]])),
    });
  }

  function laneSpies(pipeline: object) {
    const target = pipeline as Record<string, (...args: never[]) => unknown>;
    return {
      semantic: vi.spyOn(target, "collectSemanticRetrievals"),
      openQuestions: vi.spyOn(target, "collectOpenQuestions"),
      imagePerception: vi.spyOn(target, "collectImagePerceptionEvidenceWithDisclosureMode"),
      commitments: vi.spyOn(target, "collectCommitmentEvidence"),
    };
  }

  it("searchEpisodesForDisclosure skips every context lane", async () => {
    const harness = await createHarness();
    const spies = laneSpies(harness.retrievalPipeline);

    await harness.retrievalPipeline.searchEpisodesForDisclosure(QUERY, { limit: 3 });

    expect(spies.semantic).not.toHaveBeenCalled();
    expect(spies.openQuestions).not.toHaveBeenCalled();
    expect(spies.imagePerception).not.toHaveBeenCalled();
    expect(spies.commitments).not.toHaveBeenCalled();
  });

  it("searchWithContextForDisclosure still runs the full pipeline", async () => {
    const harness = await createHarness();
    const spies = laneSpies(harness.retrievalPipeline);

    await harness.retrievalPipeline.searchWithContextForDisclosure(QUERY, { limit: 3 });

    expect(spies.semantic).toHaveBeenCalled();
    expect(spies.openQuestions).toHaveBeenCalled();
    expect(spies.imagePerception).toHaveBeenCalled();
    expect(spies.commitments).toHaveBeenCalled();
  });

  it("makes zero lexical calls and disables exact-term reservation when the flag is off", async () => {
    const harness = await createHarness();

    try {
      for (const [index, fixture] of [
        { title: "Architecture baseline", participants: ["team"], vector: [1, 0, 0, 0] },
        { title: "Participant notes", participants: ["Hary"], vector: [0.8, 0.2, 0, 0] },
      ].entries()) {
        const source = await harness.streamWriter.append({
          kind: "user_msg",
          content: `fixture source ${index}`,
        });
        await harness.episodicRepository.createEpisode(
          createEpisodeFixture(
            {
              title: fixture.title,
              narrative: `${fixture.title} narrative.`,
              participants: fixture.participants,
              source_stream_ids: [source.id],
              created_at: NOW_MS - index * 1_000,
              updated_at: NOW_MS - index * 1_000,
            },
            fixture.vector,
          ),
        );
      }

      const flagOff = new RetrievalPipeline({
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        dataDir: harness.tempDir,
        clock: harness.clock,
        lexicalFusionEnabled: false,
      });
      const disclosureLexicalSpy = vi.spyOn(
        harness.episodicRepository,
        "searchByLexicalTermsForDisclosure",
      );
      const cognitionLexicalSpy = vi.spyOn(
        harness.episodicRepository,
        "recallByLexicalTermsForCognition",
      );
      const projectionSpy = vi.spyOn(evidenceProjections, "projectEpisodes");

      await flagOff.searchEpisodesForDisclosure(QUERY, {
        limit: 2,
        crossAudience: true,
        entityTerms: ["Hary"],
        recordRetrieval: false,
      });

      expect(disclosureLexicalSpy).not.toHaveBeenCalled();
      expect(cognitionLexicalSpy).not.toHaveBeenCalled();
      expect(projectionSpy).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        expect.objectContaining({ exactTermReservedSlots: 0 }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("pins flag-off serialized output to a literal fixed fixture", async () => {
    const harness = await createHarness();

    try {
      const fixtures = [
        createEpisodeFixture(
          {
            id: parseEpisodeId("ep_projectionfixa00"),
            title: "Architecture alpha",
            narrative: "Literal fixture alpha.",
            participants: ["team"],
            source_stream_ids: [parseStreamEntryId("strm_projectiona00000")],
            significance: 0.8,
            created_at: NOW_MS - 1_000,
            updated_at: NOW_MS - 1_000,
          },
          [1, 0, 0, 0],
        ),
        createEpisodeFixture(
          {
            id: parseEpisodeId("ep_projectionfixb00"),
            title: "Architecture beta",
            narrative: "Literal fixture beta.",
            participants: ["ops"],
            source_stream_ids: [parseStreamEntryId("strm_projectionb00000")],
            significance: 0.4,
            created_at: NOW_MS - 2_000,
            updated_at: NOW_MS - 2_000,
          },
          [0.8, 0.6, 0, 0],
        ),
      ];

      for (const episode of fixtures) {
        await harness.episodicRepository.createEpisode(episode);
      }

      const flagOff = new RetrievalPipeline({
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        dataDir: harness.tempDir,
        clock: harness.clock,
        lexicalFusionEnabled: false,
      });
      const results = await flagOff.searchEpisodesForDisclosure(QUERY, {
        limit: 2,
        crossAudience: true,
        recordRetrieval: false,
      });
      const serialized = JSON.stringify(
        results.map((item) => ({
          id: item.episode.id,
          title: item.episode.title,
          score: item.score,
          rawScore: item.rawScore,
          similarity: item.scoreBreakdown.similarity,
          decayedSalience: item.scoreBreakdown.decayedSalience,
          citationIds: item.citationChain.map((entry) => entry.id),
        })),
      );

      expect(serialized).toBe(
        '[{"id":"ep_projectionfixa00","title":"Architecture alpha","score":0.9399987163975426,"rawScore":0.9399987163975426,"similarity":1,"decayedSalience":0.7999957213251423,"citationIds":["strm_projectiona00000"]},{"id":"ep_projectionfixb00","title":"Architecture beta","score":0.6799987247456255,"rawScore":0.6799987247456255,"similarity":0.800000011920929,"decayedSalience":0.3999957213365841,"citationIds":["strm_projectionb00000"]}]',
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("runs the lexical known-term lane only when enabled", async () => {
    const harness = await createHarness();

    try {
      const source = await harness.streamWriter.append({
        kind: "user_msg",
        content: "lexical fixture source",
      });
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture(
          {
            title: "A conversation with Hary",
            narrative: "The title is the only lexical handle.",
            participants: ["team"],
            tags: ["planning"],
            source_stream_ids: [source.id],
            created_at: NOW_MS,
            updated_at: NOW_MS,
          },
          [0, 1, 0, 0],
        ),
      );
      const pipeline = new RetrievalPipeline({
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        dataDir: harness.tempDir,
        clock: harness.clock,
        lexicalFusionEnabled: true,
      });
      const lexicalSpy = vi.spyOn(harness.episodicRepository, "searchByLexicalTermsForDisclosure");

      const results = await pipeline.searchEpisodesForDisclosure(QUERY, {
        limit: 3,
        crossAudience: true,
        entityTerms: ["Hary"],
        recordRetrieval: false,
      });

      expect(lexicalSpy).toHaveBeenCalledWith(["Hary"], {
        crossAudience: true,
        audienceEntityId: undefined,
        limit: 8,
      });
      expect(results.some((item) => item.episode.title === "A conversation with Hary")).toBe(true);
    } finally {
      await harness.cleanup();
    }
  });

  it("keeps score equal to clamped rawScore on projected and direct episode paths", async () => {
    const harness = await createHarness();

    try {
      const source = await harness.streamWriter.append({
        kind: "user_msg",
        content: "boosted score fixture source",
      });
      const episode = createEpisodeFixture(
        {
          title: "Hary score ceiling",
          narrative: "A high-salience known-term episode.",
          participants: ["Hary"],
          source_stream_ids: [source.id],
          significance: 1,
          created_at: NOW_MS,
          updated_at: NOW_MS,
        },
        [0.8, 0.6, 0, 0],
      );
      await harness.episodicRepository.createEpisode(episode);
      const recordSpy = vi.spyOn(harness.episodicRepository, "recordRetrieval");
      const pipeline = new RetrievalPipeline({
        embeddingClient: harness.embeddingClient,
        episodicRepository: harness.episodicRepository,
        dataDir: harness.tempDir,
        clock: harness.clock,
      });
      const [result] = await pipeline.searchEpisodesForDisclosure(QUERY, {
        limit: 1,
        crossAudience: true,
        entityTerms: ["Hary"],
        attentionWeights: {
          semantic: 0.5,
          goal_relevance: 0,
          value_alignment: 0,
          mood: 0,
          time: 0,
          social: 0,
          entity: 0.5,
          heat: 0,
          suppression_penalty: 0,
        },
      });
      const direct = await pipeline.getEpisode(episode.id, { crossAudience: true });

      expect(result?.score).toBe(1);
      expect(result?.rawScore).toBeGreaterThan(1);
      expect(result?.scoreBreakdown.entityRelevance).toBe(1);
      expect(direct?.score).toBe(1);
      expect(direct?.rawScore).toBe(1);
      for (const item of [result, direct]) {
        expect(item?.score).toBe(Math.min(1, Math.max(0, item?.rawScore ?? Number.NaN)));
      }
      expect(recordSpy).toHaveBeenCalledWith(episode.id, NOW_MS, 1);
    } finally {
      await harness.cleanup();
    }
  });
});
