import { describe, expect, it } from "vitest";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import { ManualClock } from "../util/clock.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "./test-support.js";
import {
  recordTestAnnotation,
  recordTestAnnotationWithMalformedHistory,
} from "./evidence-test-support.js";
import { SelfNarratorProcess } from "./self-narrator/index.js";
import { LivedExperienceDaySummarizerProcess } from "./lived-experience-day-summarizer/index.js";
import { ReflectorProcess } from "./reflector/index.js";
import { RuminatorProcess } from "./ruminator/index.js";
import { AssociatorProcess } from "./associator/index.js";
import { OverseerProcess } from "./overseer/index.js";
import { MEMORY_EVIDENCE_GUIDANCE } from "../memory/common/evidence-prompt.js";

const processes = {
  selfNarrator: SelfNarratorProcess,
  daySummary: LivedExperienceDaySummarizerProcess,
  reflector: ReflectorProcess,
  ruminator: RuminatorProcess,
  associator: AssociatorProcess,
  overseer: OverseerProcess,
};
const dayStart = Date.UTC(2026, 5, 15);
function response(name: string, input: unknown) {
  return {
    text: "",
    input_tokens: 20,
    output_tokens: 10,
    stop_reason: "tool_use" as const,
    tool_calls: [{ id: "toolu_synthetic", name, input }],
  };
}
function jsonRows(prompt: string) {
  return prompt
    .split("\n")
    .filter((line) => line.startsWith("{"))
    .map((line) => JSON.parse(line));
}

describe("stage 2b live callers use audited context", () => {
  it.each(Object.keys(processes) as (keyof typeof processes)[])(
    "%s keeps valid annotations through malformed history at the model boundary",
    async (name) => {
      const clock = new ManualClock(dayStart + 1000);
      const llm = new FakeLLMClient();
      const harness = await createOfflineTestHarness({
        clock,
        llmClient: llm,
        configOverrides: {
          offline: {
            selfNarrator: { minSupportEpisodes: 2 },
            reflector: { minSupport: 2, maxInsightsPerRun: 1 },
            associator: { episodesPerSample: 3, maxSamplesPerRun: 1 },
            ruminator: {
              revisitPeriodMinDays: 0.001,
              revisitPeriodMaxDays: 0.001,
              resolveConfidenceThreshold: 0.01,
            },
            overseer: { lookbackHours: 72, maxChecksPerRun: 3 },
            livedExperienceDaySummarizer: { windowDays: 2, maxDaysPerRun: 1 },
          },
        },
      });
      try {
        const episodes: ReturnType<typeof createEpisodeFixture>[] = [];
        for (let i = 0; i < 3; i++)
          episodes.push(
            await harness.episodicRepository.createEpisode(
              createEpisodeFixture(
                {
                  title: `Synthetic ${i}`,
                  narrative: `Original ${i}. 原文。\nالنص الأصلي.`,
                  tags: ["synthetic-cluster"],
                  start_time: dayStart + 1000 + i,
                  end_time: dayStart + 2000 + i,
                  created_at: dayStart + 1000 + i,
                  updated_at: dayStart + 1000 + i,
                },
                [1, 0, 0, 0],
              ),
            ),
          );
        const text = recordTestAnnotationWithMalformedHistory(harness, episodes[0]!.id);
        recordTestAnnotation(
          harness.identityEventRepository,
          episodes[0]!.id,
          "Older valid correction",
          { ts: 10 },
        );
        // Same id, wrong record type: never inherits an annotation from another binding.
        recordTestAnnotation(
          harness.identityEventRepository,
          episodes[0]!.id,
          "WRONG TYPED TARGET",
          { recordType: "semantic_node" },
        );
        if (name === "daySummary") {
          harness.entityRepository.add({ canonicalName: "Synthetic self", kind: "self" });
          const goal = harness.goalsRepository.add({
            description: "Synthetic goal",
            priority: 1,
            provenance: { kind: "manual" },
          });
          harness.goalsRepository.block(
            goal.id,
            {
              blocker: { kind: "until", until: dayStart + 5000 },
              attempt_status: "attempted_unavailable",
              reason: "Synthetic wait",
            },
            { kind: "manual" },
          );
        }
        if (name === "ruminator")
          harness.openQuestionsRepository.add({
            question: "Synthetic question?",
            provenance: { kind: "manual" },
            source: "reflection",
            urgency: 0.8,
            related_episode_ids: [],
            related_semantic_node_ids: [],
            created_at: dayStart - 86400000,
            last_touched: dayStart - 86400000,
          });
        clock.advance(36 * 3600000);
        const replies = {
          selfNarrator: response("EmitSelfNarratorObservations", {
            observations: [],
            period_narrative: "Synthetic period narrative",
            period_decision: "continue_current",
            period_decision_confidence: 0.8,
          }),
          daySummary: response("EmitLivedExperienceDaySummary", {
            utc_day: "2026-06-15",
            gist: "Synthetic day",
            salience: 0.5,
            cited_episode_ids: [],
            cited_source_stream_entry_ids: [],
          }),
          reflector: response("EmitReflectorInsights", {
            label: "Synthetic insight",
            description: "Synthetic description",
            confidence: 0.4,
            source_episode_ids: episodes.map((e) => e.id),
          }),
          ruminator: response("EmitRuminatorDecisions", {
            outcome: "resolved",
            resolution_note: "Synthetic resolution",
            growth_marker: null,
          }),
          associator: response("EmitAssociations", { findings: [] }),
          overseer: response("EmitOverseerFlags", { flags: [] }),
        };
        for (let i = 0; i < 3; i++) llm.pushResponse(replies[name]);
        const process = new processes[name](harness);
        const result = await process.plan(harness.createContext());
        expect(result.errors).toEqual([]);
        const prompts = llm.requests.map((request) => String(request.messages[0]!.content));
        expect(prompts.length).toBeGreaterThan(0);
        const annotated = prompts.filter((prompt) => prompt.includes(text));
        expect(annotated).toHaveLength(1);
        for (const prompt of prompts) {
          expect(prompt).not.toContain("WRONG TYPED TARGET");
          expect(prompt).not.toContain("__borg_evidence");
          const rows = jsonRows(prompt)
            .map((row) => (row.type === "episode" ? row.content : row))
            .filter((row) => episodes.some((e) => e.id === row.id));
          for (const row of rows) {
            expect(row.narrative).toBe(episodes.find((e) => e.id === row.id)!.narrative);
            if (row.id === episodes[0]!.id) {
              expect(row.annotations.map((a: { text: string }) => a.text)).toEqual([
                text,
                "Older valid correction",
              ]);
              expect(row.annotations[0]).not.toHaveProperty("written_at");
              expect(Buffer.from(row.annotations[0].text)).toEqual(Buffer.from(text));
            } else expect(row.annotations).toBeUndefined();
          }
        }
        expect(annotated[0]!.split(MEMORY_EVIDENCE_GUIDANCE)).toHaveLength(2);
        expect(
          (await harness.episodicRepository.getMany(episodes.map((e) => e.id)))
            .map((e) => e.narrative)
            .sort(),
        ).toEqual(episodes.map((e) => e.narrative).sort());
      } finally {
        await harness.cleanup();
      }
    },
  );
  it("overseer retains index-only retired targets and getMany sources; node correction stays on the node", async () => {
    const llm = new FakeLLMClient({
      responses: Array.from({ length: 3 }, () => response("EmitOverseerFlags", { flags: [] })),
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      configOverrides: { offline: { overseer: { maxChecksPerRun: 3 } } },
    });
    try {
      const episode = await harness.episodicRepository.createEpisode(
        createEpisodeFixture({ narrative: "Original retired text" }),
      );
      const node = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture({
          source_episode_ids: [episode.id],
          description: "Original node text",
        }),
      );
      harness.db
        .prepare("UPDATE episode_index SET archived = 1 WHERE episode_id = ?")
        .run(episode.id);
      const correction = "节点更正。 تصحيح العقدة.";
      recordTestAnnotation(harness.identityEventRepository, node.id, correction, {
        recordType: "semantic_node",
      });
      harness.db
        .prepare(
          "INSERT INTO identity_events (record_type, record_id, action, new_value_json, provenance_kind, ts) VALUES ('semantic_node', ?, 'synthetic_audit', '{', 'manual', 2)",
        )
        .run(node.id);
      const result = await new OverseerProcess(harness).plan(harness.createContext());
      expect(result.errors).toEqual([]);
      const prompts = llm.requests.map((request) => String(request.messages[0]!.content));
      const targets = prompts
        .flatMap(jsonRows)
        .filter((row) => row.type === "episode" || row.type === "semantic_node");
      expect(targets.find((row) => row.type === "episode").content).toMatchObject({
        id: episode.id,
        narrative: episode.narrative,
        lifecycle: "archived",
      });
      expect(targets.find((row) => row.type === "episode").content.annotations).toBeUndefined();
      expect(targets.find((row) => row.type === "semantic_node").content).toMatchObject({
        id: node.id,
        description: node.description,
        annotations: [{ text: correction }],
      });
      expect(targets.find((row) => row.type === "semantic_node").content.lifecycle).toBeUndefined();
      expect(prompts.find((p) => p.includes(correction))).toContain(
        'shared=true {"lifecycle":"archived"}',
      );
    } finally {
      await harness.cleanup();
    }
  });
});
