import { describe, expect, it } from "vitest";

import { createSemanticEdgeFixture, createSemanticNodeFixture } from "../../offline/test-support.js";
import {
  createSemanticNodeId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type SemanticNodeId,
} from "../../util/ids.js";
import { createSemanticWalkTool, type SemanticWalkToolOptions } from "./semantic-walk.js";

const context = {
  sessionId: DEFAULT_SESSION_ID,
  origin: "autonomous" as const,
};

function createHarness(steps: Awaited<ReturnType<SemanticWalkToolOptions["walkGraph"]>>) {
  return createSemanticWalkTool({
    walkGraph: async () => steps,
  });
}

describe("tool.semantic.walk", () => {
  it("projects correction refs of every shape, including ones superseded_by cannot hold", async () => {
    const rootId = createSemanticNodeId();
    const correctingNodeId = createSemanticNodeId();
    const correctingEntryId = createStreamEntryId();
    const nodeCorrected = createSemanticNodeFixture({
      status: "superseded",
      superseded_by: correctingNodeId,
      corrected_by: correctingNodeId,
      superseded_at: 2_000_000,
    });
    // The live shape superseded_by cannot represent: a contradiction whose
    // correcting reference is a stream entry rather than another node.
    const entryCorrected = createSemanticNodeFixture({
      status: "contradicted",
      corrected_by: correctingEntryId,
      superseded_at: 3_000_000,
    });
    const uncorrected = createSemanticNodeFixture({ status: "active" });
    const tool = createHarness(
      [nodeCorrected, entryCorrected, uncorrected].map((node) => ({
        node,
        edgePath: [createSemanticEdgeFixture({ from_node_id: rootId, to_node_id: node.id })],
      })),
    );

    const result = await tool.invoke({ node_id: rootId, relation: "contradicts" }, context);
    const parsed = tool.outputSchema.parse(result);

    expect(parsed.steps.map((step) => step.node.corrected_by)).toEqual([
      correctingNodeId,
      correctingEntryId,
      null,
    ]);
    expect(parsed.steps.map((step) => step.node.superseded_by)).toEqual([
      correctingNodeId,
      null,
      null,
    ]);
  });

  it("omits the embedding from every walked node", async () => {
    const rootId: SemanticNodeId = createSemanticNodeId();
    const tool = createHarness([
      {
        node: createSemanticNodeFixture({ id: rootId }),
        edgePath: [],
      },
    ]);

    const result = await tool.invoke({ node_id: rootId, relation: "supports" }, context);

    expect(result.steps[0]?.node).not.toHaveProperty("embedding");
  });
});
