import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createOfflineTestHarness,
  createEpisodeFixture,
  createSemanticNodeFixture,
  createRetrievalScoreFixture,
  type OfflineTestHarness,
} from "../../offline/test-support.js";
import {
  recordTestAnnotation,
  recordTestAnnotationWithMalformedHistory,
} from "../../offline/evidence-test-support.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { createEpisodicGetTool } from "./episodic-get.js";
import { createEpisodicSearchTool } from "./episodic-search.js";
import { createEpisodicRecentTool } from "./episodic-recent.js";
import { createSemanticWalkTool } from "./semantic-walk.js";
import { formatAutonomyTriggerContext } from "../../cognition/autonomy-trigger.js";
import { RECORD_ANNOTATION_GUIDANCE } from "../../memory/common/record-evidence.js";
import { MAX_EVIDENCE_PROMPT_CONTEXT_CHARS } from "../../memory/common/evidence-prompt.js";

const invocation = { sessionId: DEFAULT_SESSION_ID, origin: "autonomous" as const };
const correction = "Zażółć. 中文更正。 تصحيح. 🧭 e\u0301\nUnchanged second line.";
describe("audited record context in live tools and wakes", () => {
  let h: OfflineTestHarness | undefined;
  afterEach(async () => {
    await h?.cleanup();
    h = undefined;
  });
  it.each([
    { indexArchived: 0, statsArchived: 0, lifecycle: "active" },
    { indexArchived: 1, statsArchived: 0, lifecycle: "archived" },
    { indexArchived: 0, statsArchived: 1, lifecycle: "archived" },
    { indexArchived: 1, statsArchived: 1, lifecycle: "archived" },
  ] as const)(
    "get reports consistent lifecycle for index=$indexArchived, stats=$statsArchived",
    async ({ indexArchived, statsArchived, lifecycle }) => {
      h = await createOfflineTestHarness();
      const episode = await h.episodicRepository.createEpisode(
        createEpisodeFixture({ narrative: "Original retired text.\n原文。" }),
      );
      recordTestAnnotation(h.identityEventRepository, episode.id, correction);
      h.db
        .prepare("UPDATE episode_index SET archived = ? WHERE episode_id = ?")
        .run(indexArchived, episode.id);
      h.db
        .prepare("UPDATE episode_stats SET archived = ? WHERE episode_id = ?")
        .run(statsArchived, episode.id);
      const lifecycleRead = vi.spyOn(h.episodicRepository, "getLifecycleMany");
      const tool = createEpisodicGetTool({
        episodicRepository: h.episodicRepository,
        recordEvidenceReader: h,
      });
      const result = tool.outputSchema.parse(
        await tool.invoke({ episode_id: episode.id }, invocation),
      );
      expect(result.episode).toMatchObject({
        id: episode.id,
        narrative: episode.narrative,
        archived: lifecycle === "archived",
        lifecycle_status: lifecycle,
        annotations: [{ text: correction }],
      });
      if (lifecycle === "active") expect(result.episode).not.toHaveProperty("lifecycle");
      else expect(result.episode?.lifecycle).toBe(lifecycle);
      expect(lifecycleRead).toHaveBeenCalledExactlyOnceWith([episode.id]);
      expect(result.evidence_guidance).toBe(RECORD_ANNOTATION_GUIDANCE);
    },
  );
  it("get labels an inactive unarchived version consistently even without an annotation reader", async () => {
    h = await createOfflineTestHarness();
    const episode = await h.episodicRepository.createEpisode({
      ...createEpisodeFixture(),
      episode_kind: "consolidation_version",
    });
    const tool = createEpisodicGetTool({ episodicRepository: h.episodicRepository });
    const result = tool.outputSchema.parse(
      await tool.invoke({ episode_id: episode.id }, invocation),
    );
    expect(result.episode).toMatchObject({
      narrative: episode.narrative,
      archived: false,
      lifecycle: "inactive",
      lifecycle_status: "inactive",
    });
  });
  it("get still returns the unchanged record and valid correction with malformed annotation history", async () => {
    const emit = vi.fn();
    h = await createOfflineTestHarness({ tracer: { enabled: true, includePayloads: false, emit } });
    const episode = await h.episodicRepository.createEpisode(createEpisodeFixture());
    const correction = recordTestAnnotationWithMalformedHistory(h, episode.id);
    const tool = createEpisodicGetTool({
      episodicRepository: h.episodicRepository,
      recordEvidenceReader: h,
    });
    emit.mockClear();
    const result = tool.outputSchema.parse(
      await tool.invoke({ episode_id: episode.id }, invocation),
    );
    expect(result.episode).toMatchObject({
      id: episode.id,
      narrative: episode.narrative,
      archived: false,
      lifecycle_status: "active",
      annotations: [{ text: correction, authority: "Synthetic operator" }],
    });
    expect(result.episode).not.toHaveProperty("lifecycle");
    expect(result.episode!.annotations![0]).not.toHaveProperty("written_at");
    expect(emit).toHaveBeenCalledExactlyOnceWith("memory.annotations.skipped", {
      turnId: "memory-annotation-read",
      counts_by_reason: {
        invalid_timestamp: 1,
        invalid_json: 1,
        missing_text: 1,
        invalid_payload: 1,
      },
    });
  });
  it("search and recent attach only exact episode annotations and preserve their existing narrative excerpts", async () => {
    h = await createOfflineTestHarness();
    const episodes = await Promise.all(
      [0, 1].map((i) =>
        h!.episodicRepository.createEpisode(
          createEpisodeFixture({ narrative: `Synthetic ${i}.\n原文。` }),
        ),
      ),
    );
    recordTestAnnotation(h.identityEventRepository, episodes[0]!.id, correction);
    const searchEpisodes = async () =>
      episodes.map((episode) => ({
        episode,
        score: 0.7,
        rawScore: 0.7,
        scoreBreakdown: createRetrievalScoreFixture(),
        citationChain: [],
      }));
    const search = createEpisodicSearchTool({ searchEpisodes, recordEvidenceReader: h });
    const result = search.outputSchema.parse(
      await search.invoke({ query: "synthetic" }, invocation),
    );
    const before = await createEpisodicSearchTool({ searchEpisodes }).invoke(
      { query: "synthetic" },
      invocation,
    );
    expect(result.episodes[0]!.annotations?.[0]!.text).toBe(correction);
    expect(result.episodes[1]).not.toHaveProperty("annotations");
    expect(result.episodes.map((row) => row.narrative)).toEqual(
      before.episodes.map((row) => row.narrative),
    );
    const recent = createEpisodicRecentTool({
      listRecentEpisodes: (limit) => h!.episodicRepository.listRecentForCognition({ limit }),
      recordEvidenceReader: h,
    });
    const recentResult = recent.outputSchema.parse(await recent.invoke({}, invocation));
    expect(
      recentResult.episodes.find((row) => row.id === episodes[0]!.id)?.annotations?.[0]!.text,
    ).toBe(correction);

    const wake = {
      source_name: "open_question_dormant",
      source_type: "trigger" as const,
      event_id: "synthetic-wake",
      sort_ts: 1_000_000,
      payload: {
        open_question_id: "oq_synthetic",
        question: "Synthetic question?",
        last_touched: 1_000_000,
        related_episodes: result.episodes,
      },
    };
    const rendered = formatAutonomyTriggerContext(wake);
    expect(rendered).toContain(JSON.stringify(correction));
    expect(rendered.split(RECORD_ANNOTATION_GUIDANCE)).toHaveLength(2);
    expect(rendered).not.toContain("__borg_evidence_context");
    const bounded = formatAutonomyTriggerContext(
      {
        ...wake,
        payload: {
          ...wake.payload,
          related_episodes: Array.from({ length: 12 }, () => ({
            ...result.episodes[0],
            narrative: "n".repeat(4_000),
          })),
        },
      },
      4_000,
    );
    expect(bounded.length).toBeLessThanOrEqual(4_000);
    const payload = JSON.parse(bounded.split("payload:\n")[1]!.split("\nnote on ")[0]!);
    for (const row of payload.related_episodes ?? [])
      expect(row.annotations?.[0]?.text).toBe(correction);
  });
  it("walk attaches a node-only correction while preserving archived/status labels and unannotated source episodes", async () => {
    h = await createOfflineTestHarness();
    const source = await h.episodicRepository.createEpisode(createEpisodeFixture());
    const node = createSemanticNodeFixture({
      source_episode_ids: [source.id],
      archived: true,
      status: "superseded",
    });
    recordTestAnnotation(h.identityEventRepository, node.id, correction, {
      recordType: "semantic_node",
    });
    const tool = createSemanticWalkTool({
      recordEvidenceReader: h,
      walkGraph: async () => [{ node, edgePath: [] }],
    });
    const result = tool.outputSchema.parse(
      await tool.invoke({ node_id: node.id, relation: "supports" }, invocation),
    );
    expect(result.steps[0]?.node).toMatchObject({
      id: node.id,
      description: node.description,
      archived: true,
      status: "superseded",
      annotations: [{ text: correction }],
    });
    expect(result.steps[0]?.node).not.toHaveProperty("lifecycle");
    expect(
      h.identityEventRepository.listAnnotationsForTargets([
        { record_type: "episode", record_id: source.id },
      ])[0]?.annotations,
    ).toEqual([]);
  });
  it("bounds added context across an entire 32-node tool response before the model sees it", async () => {
    h = await createOfflineTestHarness();
    const nodes = Array.from({ length: 32 }, () =>
      createSemanticNodeFixture({ description: "Original description. ".repeat(200) }),
    );
    for (const node of nodes)
      for (let i = 0; i < 5; i++)
        recordTestAnnotation(h.identityEventRepository, node.id, "更".repeat(1_200), {
          recordType: "semantic_node",
          ts: i,
        });
    const walkGraph = async () => nodes.map((node) => ({ node, edgePath: [] }));
    const tool = createSemanticWalkTool({ recordEvidenceReader: h, walkGraph });
    const after = tool.outputSchema.parse(
      await tool.invoke({ node_id: nodes[0]!.id, relation: "supports" }, invocation),
    );
    const ordinaryTool = createSemanticWalkTool({ walkGraph });
    const before = ordinaryTool.outputSchema.parse(
      await ordinaryTool.invoke({ node_id: nodes[0]!.id, relation: "supports" }, invocation),
    );
    expect(JSON.stringify(after).length - JSON.stringify(before).length).toBeLessThanOrEqual(
      MAX_EVIDENCE_PROMPT_CONTEXT_CHARS,
    );
    expect(after.steps.map((row) => row.node.description)).toEqual(
      before.steps.map((row) => row.node.description),
    );
    expect(
      after.steps.reduce(
        (sum, row) =>
          sum + (row.node.annotations?.length ?? 0) + (row.node.annotations_omitted ?? 0),
        0,
      ),
    ).toBe(160);
    expect(after.steps.at(-1)?.node.annotations_omitted).toBe(5);
  });
});
