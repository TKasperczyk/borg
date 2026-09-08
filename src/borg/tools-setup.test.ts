import { describe, expect, it, vi } from "vitest";

import {
  createEpisodeFixture,
  createSemanticEdgeFixture,
  createSemanticNodeFixture,
} from "../offline/test-support.js";
import { ManualClock } from "../util/clock.js";
import { createEntityId, createEpisodeId, DEFAULT_SESSION_ID } from "../util/ids.js";
import { buildToolDispatcher } from "./tools-setup.js";

describe("buildToolDispatcher", () => {
  it("batches semantic disclosure lookup and pins goal-retirement registration", async () => {
    const audienceEntityId = createEntityId();
    const privateEpisode = createEpisodeFixture({
      audience_entity_id: audienceEntityId,
      origin_audience_entity_ids: [audienceEntityId],
      shared: false,
    });
    const danglingEpisodeId = createEpisodeId();
    const root = createSemanticNodeFixture();
    const firstNode = createSemanticNodeFixture({
      source_episode_ids: [privateEpisode.id],
    });
    const secondNode = createSemanticNodeFixture({
      source_episode_ids: [privateEpisode.id, danglingEpisodeId],
    });
    const steps = [
      {
        node: firstNode,
        edgePath: [
          createSemanticEdgeFixture({
            from_node_id: root.id,
            to_node_id: firstNode.id,
            evidence_episode_ids: [danglingEpisodeId],
          }),
        ],
      },
      {
        node: secondNode,
        edgePath: [
          createSemanticEdgeFixture({
            from_node_id: firstNode.id,
            to_node_id: secondNode.id,
            evidence_episode_ids: [privateEpisode.id],
          }),
        ],
      },
    ];
    const getMany = vi.fn(async (episodeIds: readonly string[]) =>
      episodeIds.includes(privateEpisode.id) ? [privateEpisode] : [],
    );
    const dispatcher = buildToolDispatcher({
      dataDir: "/tmp/borg-tools-setup-test",
      entryIndex: {} as never,
      sourceStreamAudienceDisclosureResolver: {
        resolveLabels: () => ({
          commitmentLabels: [],
          goalLabels: [],
          commitmentLabelsById: new Map(),
          goalLabelsById: new Map(),
        }),
      } as never,
      retrievalPipeline: {} as never,
      episodicRepository: { getMany } as never,
      semanticNodeRepository: { get: vi.fn(async () => root) } as never,
      semanticGraph: { walk: vi.fn(async () => steps) } as never,
      commitmentRepository: {} as never,
      entityRepository: {} as never,
      goalsRepository: { retire: vi.fn() } as never,
      openQuestionsRepository: {
        listRuminationsInRange: vi.fn(() => []),
        get: vi.fn(() => null),
      } as never,
      identityService: {} as never,
      skillRepository: {} as never,
      trainOfThoughtRepository: {} as never,
      scheduledWakesRepository: {} as never,
      promptSurfaceHistoryRepository: {} as never,
      createStreamWriter: vi.fn() as never,
      clock: new ManualClock(1_000),
    });
    const tool = dispatcher.getDefinition("tool.semantic.walk");

    expect(tool).not.toBeNull();
    const output = (await tool!.invoke(
      { node_id: root.id, relation: "supports" },
      { sessionId: DEFAULT_SESSION_ID, origin: "deliberator" },
    )) as { steps: Array<{ node: { disclosure_label: { disclosure_class: string } } }> };

    expect(getMany).toHaveBeenCalledTimes(1);
    expect(getMany).toHaveBeenCalledWith([privateEpisode.id, danglingEpisodeId]);
    expect(output.steps.map((step) => step.node.disclosure_label.disclosure_class)).toEqual([
      "relationship_private",
      "unknown",
    ]);

    for (const name of ["tool.goals.block", "tool.goals.unblock"]) {
      expect(dispatcher.getDefinition(name)).toMatchObject({
        name,
        allowedOrigins: ["autonomous", "deliberator"],
        writeScope: "write",
      });
    }

    const goalsRetire = dispatcher.getDefinition("tool.goals.retire");

    expect(goalsRetire).toMatchObject({
      name: "tool.goals.retire",
      menuSummary: "Retire one of my own goals as done/superseded, with my reason.",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "write",
    });

    expect(dispatcher.getDefinition("tool.ownRecords.list")).toMatchObject({
      name: "tool.ownRecords.list",
      allowedOrigins: ["autonomous", "deliberator"],
      writeScope: "read",
    });
  });
});
