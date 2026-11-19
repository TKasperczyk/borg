import { describe, expect, it } from "vitest";
import { createEpisodeFixture, createSemanticNodeFixture } from "./test-support.js";
import { loadStageTwoBRenderers } from "./stage-two-b-renderer-test-support.js";
import { annotationPromptFields } from "../memory/common/episode-evidence.js";
import {
  MAX_EVIDENCE_PROMPT_CONTEXT_CHARS,
  MEMORY_EVIDENCE_GUIDANCE,
  type EpisodeEvidenceContext,
} from "../memory/common/evidence-prompt.js";
import { annotationTargetKey, type AnnotationTarget } from "../memory/identity/annotations.js";
import { estimatePromptTokens } from "../util/token-estimate.js";

const surfaces = [
  "self-narrator",
  "lived-experience-day-summarizer",
  "reflector",
  "ruminator",
  "associator",
  "overseer",
] as const;
type Surface = (typeof surfaces)[number];
const originals = Array.from({ length: 3 }, (_, index) =>
  createEpisodeFixture({
    title: `Synthetic ${index}`,
    narrative: `Original ${index}: 原文。\nالنص الأصلي.`,
    start_time: 1000 + index * 1000,
    end_time: 1500 + index * 1000,
    created_at: 1000,
    updated_at: 1000,
    tags: ["synthetic"],
    shared: true,
  }),
);
const node = createSemanticNodeFixture({
  label: "Synthetic node",
  description: "Original node description",
  source_episode_ids: originals.map((ep) => ep.id),
  archived: false,
});
type Episode = (typeof originals)[number] & { evidence_context?: EpisodeEvidenceContext };
const contexts = (episodes: Episode[], extra?: EpisodeEvidenceContext) => {
  const map = new Map<string, EpisodeEvidenceContext>();
  for (const episode of episodes) {
    const context = (episode as Episode & { evidence_context?: EpisodeEvidenceContext })
      .evidence_context;
    if (context)
      map.set(annotationTargetKey({ record_type: "episode", record_id: episode.id }), context);
  }
  if (extra)
    map.set(annotationTargetKey({ record_type: "semantic_node", record_id: node.id }), extra);
  return map;
};
function reader(episodes: Episode[], extra?: EpisodeEvidenceContext) {
  const byTarget = contexts(episodes, extra);
  return {
    episodicRepository: {
      getMany: async () => episodes,
      getLifecycleMany: (ids: string[]) =>
        new Map(
          ids.map((id) => [
            id,
            byTarget.get(annotationTargetKey({ record_type: "episode", record_id: id }))
              ?.lifecycle ?? "active",
          ]),
        ),
    },
    identityEventRepository: {
      listAnnotationsForTargets: (targets: AnnotationTarget[]) =>
        targets.map((target) => ({
          ...target,
          annotations: [
            ...(byTarget.get(annotationTargetKey(target))?.annotations ?? []),
            ...Array.from(
              { length: byTarget.get(annotationTargetKey(target))?.annotations_omitted ?? 0 },
              () => ({
                text: "Synthetic omitted history row",
                authority: "operator",
                written_at: undefined,
              }),
            ),
          ].map((a, index) => ({
            ...a,
            stream_entry_id: `str_synthetic_${index}`,
            written_at: a.written_at === undefined ? undefined : Date.parse(a.written_at),
          })),
        })),
    },
  };
}
async function renderSurface(
  surface: Surface,
  episodes: Episode[],
  dev = false,
  targetType: "episode" | "semantic_node" | "semantic_edge" = "semantic_node",
  nodeContext?: EpisodeEvidenceContext,
): Promise<string> {
  const path = `src/offline/${surface}/index.ts` as const;
  const r = await loadStageTwoBRenderers(path, dev);
  if (surface === "self-narrator")
    return r.buildObservationPrompt!(episodes, 2, 3, null, [], 1_000_000, 100, 1_000_000).prompt;
  if (surface === "lived-experience-day-summarizer")
    return r.buildDayPrompt!({
      candidate: { utcDay: "2026-01-02", dayStartMs: 86400000, dayEndMs: 172799999 },
      selfDecisionEvents: [],
      activityEvents: [],
      episodes,
      actionRecords: [],
      goalBlocks: [],
      countsSnapshot: { decisions: 0 },
      maxSelfDecisionEvents: 10,
      maxActivityEvents: 10,
      maxEpisodes: 50,
      maxActionRecords: 10,
    });
  if (surface === "reflector")
    return r.buildPrompt!({ key: "tag:synthetic", episodes, omittedEpisodeCount: 0 }, []);
  if (surface === "associator")
    return r.renderSamplePrompt!({ sampleId: "synthetic-sample", seed: 42, episodes }, new Map(), {
      complete: true,
      total_open_questions: 0,
      presented_count: 0,
      omitted_count: 0,
      rows: [],
    });
  if (surface === "ruminator")
    return r.buildResolutionPrompt!(
      {
        id: "oq_synthetic",
        question: "Synthetic question?",
        source: "reflection",
        unresolved_rumination_ticks: 0,
        related_episode_ids: [],
        related_semantic_node_ids: [],
      },
      r.buildResolutionEvidence!(episodes.map((episode) => ({ episode, score: 0.9 }))),
      [],
      0,
      [],
      5,
    );
  const content =
    targetType === "episode"
      ? (episodes[0] ?? originals[0]!)
      : targetType === "semantic_node"
        ? node
        : {
            id: "se_synthetic",
            from_node_id: node.id,
            to_node_id: node.id,
            relation: "related",
            confidence: 0.5,
            evidence_episode_ids: episodes.map((e) => e.id),
            created_at: 1000,
            last_verified_at: 1000,
            valid_from: 1000,
            valid_to: null,
          };
  return r.buildPrompt!(
    { type: targetType, id: content.id, content },
    reader(episodes, nodeContext),
    {
      target_type: targetType,
      target_id: content.id,
      source_episode_ids: episodes.map((e) => e.id),
      source_stream_ids: [],
      source_episodes: episodes.map((e) => ({
        id: e.id,
        audience_entity_id: null,
        origin_audience_entity_ids: [],
        shared: true,
      })),
      audience_entities: [],
      entries:
        episodes.length === 0
          ? []
          : [
              {
                source_episode_ids: episodes.map((e) => e.id),
                entry: {
                  id: "str_synthetic",
                  session_id: "sess_synthetic",
                  timestamp: 1000,
                  kind: "user_msg",
                  content: "原文。 Raw source\nLiteral second line.",
                },
              },
            ],
      missing_episode_ids: [],
      missing_stream_ids: [],
    },
    "Synthetic self state",
    3,
  );
}

describe("stage 2b prompt envelopes", () => {
  it.each(surfaces)(
    "%s ordinary empty and populated prompts are byte-identical to cc7877c4",
    async (surface) => {
      for (const episodes of [[], originals]) {
        const actual = await renderSurface(surface, episodes);
        expect(actual).toBe(await renderSurface(surface, episodes, true));
        expect(actual).not.toContain("annotations");
        expect(actual).not.toContain("__borg_evidence");
      }
    },
  );
  it.each(["episode", "semantic_edge"] as const)(
    "overseer ordinary %s target is byte-identical to cc7877c4",
    async (kind) => {
      expect(await renderSurface("overseer", originals, false, kind)).toBe(
        await renderSurface("overseer", originals, true, kind),
      );
    },
  );
  it.each(surfaces)(
    "%s attaches exact multilingual corrections without changing record text",
    async (surface) => {
      const text = "更正。 العربية. Français 🧭\nSecond line.";
      const annotations = [{ text, written_at: "2026-01-02T00:00:00.000Z", authority: "operator" }];
      const episodes = [
        { ...originals[0]!, evidence_context: { annotations } },
        ...originals.slice(1),
      ];
      const prompt = await renderSurface(surface, episodes);
      const ordinary = await renderSurface(surface, originals);
      expect(prompt).toContain(JSON.stringify(annotations));
      expect(prompt.split(MEMORY_EVIDENCE_GUIDANCE)).toHaveLength(2);
      expect(prompt).not.toContain("__borg_evidence");
      expect(prompt.length - ordinary.length).toBeLessThanOrEqual(
        MAX_EVIDENCE_PROMPT_CONTEXT_CHARS,
      );
      if (surface !== "overseer") {
        const rows = prompt
          .split("\n")
          .filter((line) => line.startsWith("{"))
          .map((line) => JSON.parse(line));
        for (const episode of originals) {
          const row = rows.find((row) => row.id === episode.id);
          expect(row.narrative).toBe(episode.narrative);
          expect(row.annotations).toEqual(
            episode.id === originals[0]!.id ? annotations : undefined,
          );
        }
      } else {
        expect(prompt).toContain(`episode_id=${originals[0]!.id}`);
        const target = prompt
          .split("\n")
          .filter((line) => line.startsWith("{"))
          .map((line) => JSON.parse(line))
          .find((row) => row.type === "semantic_node");
        expect(target.content.description).toBe(node.description);
        expect(target.content.annotations).toBeUndefined();
      }
    },
  );
  it.each(surfaces)("%s caps aggregate context and preserves omission counts", async (surface) => {
    const ordinaryEpisodes = Array.from({ length: 12 }, (_, i) =>
      createEpisodeFixture({ title: `Synthetic capped ${i}`, narrative: "Own text ".repeat(1000) }),
    );
    const evidence_context = annotationPromptFields(
      Array.from({ length: 5 }, (_, i) => ({
        text: `${i}:` + "更".repeat(1500),
        authority: "A".repeat(400),
        written_at: 1000 + i,
        stream_entry_id: `str_bound_${i}`,
      })),
    );
    const decorated = ordinaryEpisodes.map((e) => ({ ...e, evidence_context }));
    const prompt = await renderSurface(surface, decorated);
    const ordinary = await renderSurface(surface, ordinaryEpisodes);
    expect(prompt.length - ordinary.length).toBeLessThanOrEqual(MAX_EVIDENCE_PROMPT_CONTEXT_CHARS);
    expect(prompt).toContain('"annotations_omitted":');
    // The overseer uses exact-id labels on legacy source rows; all other surfaces use JSON rows.
    const rows = prompt
      .split("\n")
      .flatMap((line) => {
        if (line.startsWith("{")) return [JSON.parse(line)];
        if (line.startsWith("EPISODE ") && line.includes(" {"))
          return [JSON.parse(line.slice(line.indexOf(" {") + 1))];
        return [];
      })
      .filter((row) => row.annotations !== undefined || row.annotations_omitted !== undefined);
    expect(rows).toHaveLength(12);
    for (const row of rows) {
      if (surface !== "overseer")
        expect(row.narrative).toBe(
          ordinaryEpisodes.find((episode) => episode.id === row.id)!.narrative,
        );
      expect((row.annotations?.length ?? 0) + (row.annotations_omitted ?? 0)).toBe(5);
      for (const a of row.annotations ?? []) {
        expect(a.text.length).toBeLessThanOrEqual(1200);
        expect(a.authority.length).toBeLessThanOrEqual(200);
      }
    }
  });
  it("overseer labels an archived source and attaches a node-only annotation to the target", async () => {
    const nodeContext = { annotations: [{ text: "Node correction only", authority: "operator" }] };
    const prompt = await renderSurface(
      "overseer",
      [{ ...originals[0]!, evidence_context: { lifecycle: "archived" as const } }],
      false,
      "semantic_node",
      nodeContext,
    );
    const target = prompt
      .split("\n")
      .filter((line) => line.startsWith("{"))
      .map((line) => JSON.parse(line))
      .find((row) => row.type === "semantic_node");
    expect(target.content.annotations).toEqual(nodeContext.annotations);
    expect(target.content.lifecycle).toBeUndefined();
    expect(prompt).toContain('shared=true {"lifecycle":"archived"}');
  });
  it("self-narrator keeps ordinary selection identical and charges envelopes to both token budgets", async () => {
    const r = await loadStageTwoBRenderers("src/offline/self-narrator/index.ts", false);
    const dev = await loadStageTwoBRenderers("src/offline/self-narrator/index.ts", true);
    for (const allowance of [1, 200, 400, 600, 1000]) {
      expect(r.buildObservationPrompt!(originals, 2, 3, null, [], 10000, 100, allowance)).toEqual(
        dev.buildObservationPrompt!(originals, 2, 3, null, [], 10000, 100, allowance),
      );
    }
    const episodes = originals.map((episode) => ({
      ...episode,
      evidence_context: { annotations: [{ text: "x".repeat(1200), authority: "operator" }] },
    }));
    const result = r.buildObservationPrompt!(episodes, 2, 3, null, [], 10000, 100, 700);
    expect(result.sourceEpisodes.length).toBeLessThan(episodes.length);
    const full = r.buildObservationPrompt!(episodes, 2, 3, null, [], 1100, 100, 10000);
    expect(estimatePromptTokens(full.prompt)).toBeLessThanOrEqual(1100);
  });
  it("reflector keeps ordinary selection identical and charges envelopes to its cluster estimate", async () => {
    const r = await loadStageTwoBRenderers("src/offline/reflector/index.ts", false);
    const dev = await loadStageTwoBRenderers("src/offline/reflector/index.ts", true);
    for (const allowance of [1, 200, 400, 600, 1000]) {
      expect(r.boundClusterEpisodes!(originals, 1, 100, allowance)).toEqual(
        dev.boundClusterEpisodes!(originals, 1, 100, allowance),
      );
    }
    const decorated = originals.map((episode) => ({
      ...episode,
      evidence_context: { annotations: [{ text: "x".repeat(1200), authority: "operator" }] },
    }));
    expect(r.boundClusterEpisodes!(decorated, 1, 100, 700).episodes.length).toBeLessThan(
      dev.boundClusterEpisodes!(originals, 1, 100, 700).episodes.length,
    );
  });
});
