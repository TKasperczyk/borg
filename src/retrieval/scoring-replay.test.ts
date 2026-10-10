import { afterEach, describe, expect, it, vi } from "vitest";
import { BGE_SIMILARITY_MODEL, QWEN_SIMILARITY_MODEL } from "../config/similarity.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import type { EpisodeSearchCandidate } from "../memory/episodic/types.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  type OfflineTestHarness,
} from "../offline/test-support.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { FixedClock } from "../util/clock.js";
import { createSessionId, type EpisodeId, type StreamEntryId } from "../util/ids.js";
import { RetrievalPipeline } from "./pipeline.js";
import privateBaseline from "./fixtures/bge-recall-baseline-p.json";
import groupBaseline from "./fixtures/bge-recall-baseline-g.json";

type Recording = typeof privateBaseline;
type Lane = Recording["lanes"][number];

// The recordings contain lane scores and vector similarities, not the underlying
// salience/heat/social breakdown or original vectors. Reconstruct one equivalent
// set of repository signals per lane. These are synthetic decompositions, not
// claims about production stats. All scoring, intent boosts, pool fusion,
// deduplication, MMR and recency run through the unmocked RetrievalPipeline.
// Equal episode vectors isolate ranking; live-bank replay covers actual MMR.
function recordedCandidates(
  lane: Lane,
  recording: Recording,
  sourceId: StreamEntryId,
  worstRecency: boolean,
  audienceTerm: string,
): EpisodeSearchCandidate[] {
  return lane.candidates.map((row) => {
    const intentBoost = lane.kind === "known_term" ? 0.25 : lane.kind === "recent" ? 0.05 : 0;
    const auxiliary = row.score - 0.65 * row.vector_score - intentBoost;
    const entity = lane.kind === "known_term" && auxiliary >= 0.2 ? 1 : 0;
    const social =
      auxiliary - 0.2 * entity >= 0.03 && (lane.id !== audienceTerm || entity === 1) ? 0.2 : 0;
    const remainder = auxiliary - 0.2 * entity - 0.15 * social;
    const salience = Math.min(1, remainder / 0.35);
    const heat = (Math.max(0, remainder - 0.35) / 0.15) * 40;
    expect(salience).toBeGreaterThanOrEqual(0);
    expect(heat).toBeLessThanOrEqual(40);
    const occurredAt =
      worstRecency && row.episode_id === recording.target_episode_id
        ? recording.now_ms - 365 * 24 * 60 * 60_000
        : recording.now_ms;
    return {
      episode: createEpisodeFixture(
        {
          id: row.episode_id as EpisodeId,
          title: lane.id,
          tags: entity === 1 ? [lane.id] : [],
          participants: social > 0 ? [audienceTerm] : [],
          significance: salience,
          created_at: recording.now_ms,
          updated_at: recording.now_ms,
          start_time: occurredAt,
          end_time: occurredAt,
          source_stream_ids: [sourceId],
          shared: true,
        },
        [1, 0, 0, 0],
      ),
      similarity: row.vector_score,
      stats: {
        episode_id: row.episode_id as EpisodeId,
        retrieval_count: 0,
        use_count: 0,
        last_retrieved: recording.now_ms,
        win_rate: 0,
        tier: "T2",
        promoted_at: 0,
        promoted_from: null,
        gist: null,
        gist_generated_at: null,
        last_decayed_at: null,
        heat_multiplier: heat / 5,
        valence_mean: 0,
        archived: false,
      },
    };
  });
}

describe.each([privateBaseline, groupBaseline])("pipeline replay: $source_file", (recording) => {
  let harness: OfflineTestHarness;
  afterEach(async () => {
    vi.restoreAllMocks();
    await harness?.cleanup();
  });

  it.each([
    { model: BGE_SIMILARITY_MODEL, scale: 0.15, worstRecency: false },
    { model: BGE_SIMILARITY_MODEL, scale: 0.15, worstRecency: true },
    // Qwen is also the legacy negative control: its original arithmetic buries
    // this BGE recording. This verifies production profile selection, not a
    // helper's scale=1 identity branch.
    { model: QWEN_SIMILARITY_MODEL, scale: 1, worstRecency: false },
  ])(
    "uses $model through scoring and fusion (worst recency=$worstRecency)",
    async ({ model, scale, worstRecency }) => {
      const clock = new FixedClock(recording.now_ms);
      harness = await createOfflineTestHarness();
      const source = await harness.streamWriter.append({
        kind: "user_msg",
        content: "Replay fixture citation",
      });
      const knownLanes = recording.lanes.filter((lane) => lane.kind === "known_term");
      // P has an audience-only lookup; G already contains the handle in its
      // planner lanes. Keep those source/precedence paths in the real pipeline.
      const audienceTerm = (
        knownLanes.find((lane) => lane.source === "audience-aliases") ?? knownLanes[0]!
      ).id;
      const lanes = new Map(
        recording.lanes.map((lane) => [
          lane.id,
          recordedCandidates(lane, recording, source.id, worstRecency, audienceTerm),
        ]),
      );
      const repo = harness.episodicRepository;
      vi.spyOn(repo, "recallByVectorForCognition").mockImplementation(
        async (vector) =>
          lanes.get(vector[0] === 1 ? "recall_raw_text_0" : "recall_semantic_query_0")!,
      );
      vi.spyOn(repo, "recallByParticipantsOrTagsForCognition").mockImplementation(
        async (terms) => lanes.get(terms[0]!) ?? [],
      );
      const recent = lanes.get("recall_recent_0")!;
      vi.spyOn(repo, "listRecentForCognition").mockResolvedValue(recent.slice(0, 12));
      vi.spyOn(repo, "listHottestForCognition").mockResolvedValue(recent.slice(12));
      const llmClient = new FakeLLMClient({
        responses: [
          {
            text: "",
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "plan",
                name: "EmitRecallQueryPlan",
                input: {
                  resolved_query: recording.query,
                  semantic_variants: [{ strategy: "combined", query: "recall_semantic_query_0" }],
                  named_terms: knownLanes
                    .filter((lane) => lane.source === "llm-expansion")
                    .map((lane) => lane.id),
                  typed_queries: [],
                  temporal_cue: null,
                },
              },
            ],
          },
        ],
      });
      const tracer: TurnTracer = { enabled: true, includePayloads: false, emit: vi.fn() };
      const embeddingClient = {
        profile: { model, dimensions: 4 },
        embed: vi.fn(async (text: string) =>
          Float32Array.from([text === recording.query ? 1 : 2, 0, 0, 0]),
        ),
        embedBatch: vi.fn(async (texts: readonly string[]) =>
          texts.map(() => Float32Array.from([2, 0, 0, 0])),
        ),
      };
      const pipeline = new RetrievalPipeline({
        episodicRepository: repo,
        embeddingClient,
        llmClient,
        recallExpansionSemanticVariantCount: 1,
        dataDir: harness.tempDir,
        entryIndex: harness.createContext().entryIndex,
        clock,
        tracer,
      });
      const degraded = vi.fn();
      const hits = await pipeline.recallEpisodeHitsForCognition(recording.query, {
        limit: 24,
        mmrLambda: 0.7,
        recordRetrieval: false,
        recallContext: {
          reader: "self",
          currentSessionId: createSessionId(),
          currentAudienceEntityId: null,
          currentParticipantEntityIds: [],
        },
        audienceTerms: [audienceTerm],
        entityTerms: knownLanes
          .filter((lane) => lane.source === "perception-entities")
          .map((lane) => lane.id),
        attentionWeights: recording.attention_weights,
        recencyPrior: { weight: 0.15, halfLifeHours: 36 },
        traceTurnId: recording.turn_id,
        onDegraded: degraded,
      });
      expect(degraded).not.toHaveBeenCalled();
      expect(llmClient.requests).toHaveLength(1);
      const traces = vi
        .mocked(tracer.emit)
        .mock.calls.filter(([event]) => event === "retrieval.intent_candidates")
        .map(([, data]) => data);
      expect(traces.map((trace) => trace.intent_id)).toEqual(
        recording.lanes.map((lane) => lane.id),
      );
      const bestByEpisode = new Map<string, number>();
      for (const lane of recording.lanes) {
        const trace = traces.find((trace) => trace.intent_id === lane.id)!;
        expect(trace.intent_source).toBe(lane.source);
        const actual = trace.candidates as Array<{
          episode_id: string;
          score: number;
          vector_score: number;
        }>;
        expect(actual).toHaveLength(lane.candidates.length);
        for (const row of lane.candidates) {
          // Independent expected legacy/profile arithmetic, never patched into a
          // candidate score. A dropped scoringDefaults or intent multiplier fails.
          const expected =
            scale === 1
              ? row.score
              : 0.65 * row.vector_score + scale * (row.score - 0.65 * row.vector_score);
          const scored = actual.find((candidate) => candidate.episode_id === row.episode_id)!;
          expect(scored.score).toBeCloseTo(expected, 12);
          expect(scored.vector_score).toBe(row.vector_score);
          bestByEpisode.set(
            row.episode_id,
            Math.max(bestByEpisode.get(row.episode_id) ?? -Infinity, expected),
          );
        }
      }
      expect(hits).toHaveLength(24);
      for (const hit of hits) {
        const ageHours = (recording.now_ms - hit.episode.end_time) / 3_600_000;
        // This independently checks the pipeline's recency profile wiring after
        // evidence-pool fusion has chosen each episode's strongest lane.
        expect(hit.rawScore).toBeCloseTo(
          bestByEpisode.get(hit.episode.id)! + 0.15 * scale * Math.pow(0.5, ageHours / 36),
          12,
        );
      }
      const rank = hits.findIndex((hit) => hit.episode.id === recording.target_episode_id) + 1;
      expect(rank).toBeGreaterThan(0);
      if (scale === 1) expect(rank).toBeGreaterThan(8);
      else expect(rank).toBeLessThanOrEqual(8);
    },
  );
});
