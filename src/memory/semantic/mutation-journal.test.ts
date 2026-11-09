import { afterEach, expect, it } from "vitest";
import {
  createOfflineTestHarness,
  createSemanticNodeFixture,
  createSemanticEdgeFixture,
} from "../../offline/test-support.js";
import { SemanticMutationJournal } from "./mutation-journal.js";

const cleanup: Array<() => Promise<void>> = [];
afterEach(async () => {
  while (cleanup.length > 0) await cleanup.pop()?.();
});

it("records complete first before-images at all four write boundaries and keeps creations as creations", async () => {
  const harness = await createOfflineTestHarness();
  cleanup.push(harness.cleanup);
  const nodes = harness.semanticNodeRepository;
  const edges = harness.semanticEdgeRepository;
  const original = await nodes.insert(
    createSemanticNodeFixture({ archived: true, status: "active", aliases: ["old alias"] }),
  );
  const target = await nodes.insert(createSemanticNodeFixture());
  const edge = edges.addEdge(
    createSemanticEdgeFixture({
      from_node_id: original.id,
      to_node_id: target.id,
      confidence: 0.4,
    }),
  );
  const journal = new SemanticMutationJournal();
  const options = { mutationRecorder: journal };
  await nodes.update(
    original.id,
    { confidence: 0.6, embedding: Float32Array.of(0, 1, 0, 0), archived: false },
    options,
  );
  await nodes.update(original.id, { aliases: ["new alias"] }, options);
  const created = await nodes.insert(createSemanticNodeFixture(), options);
  await nodes.update(created.id, { confidence: 0.8 }, options);
  edges.mergeOpenEdgeEvidence({ ...edge, confidence: 0.8 }, options);
  edges.mergeOpenEdgeEvidence({ ...edge, confidence: 0.9 }, options);
  const createdEdge = edges.addEdge(
    createSemanticEdgeFixture({ from_node_id: created.id, to_node_id: target.id }),
    options,
  );
  edges.mergeOpenEdgeEvidence({ ...createdEdge, confidence: 0.9 }, options);
  expect(journal.nodes.get(original.id)).toEqual(original);
  expect(journal.nodes.get(created.id)).toBeNull();
  expect(journal.nodes.has(target.id)).toBe(false);
  expect(journal.edges.get(edge.id)).toEqual(edge);
  expect(journal.edges.get(createdEdge.id)).toBeNull();
  // The journal owns its arrays, not references that later writers can mutate.
  original.embedding[0] = 99;
  original.aliases.push("mutated outside repository");
  expect(journal.nodes.get(original.id)?.embedding[0]).not.toBe(99);
  expect(journal.nodes.get(original.id)?.aliases).toEqual(["old alias"]);
});

it("records edge creation before a post-commit callback throws", async () => {
  const harness = await createOfflineTestHarness();
  cleanup.push(harness.cleanup);
  const from = await harness.semanticNodeRepository.insert(createSemanticNodeFixture());
  const to = await harness.semanticNodeRepository.insert(createSemanticNodeFixture());
  const journal = new SemanticMutationJournal();
  expect(() =>
    harness.semanticEdgeRepository.addEdge(
      createSemanticEdgeFixture({
        from_node_id: from.id,
        to_node_id: to.id,
        relation: "contradicts",
      }),
      {
        mutationRecorder: journal,
        enqueueReview: () => {
          throw new Error("after commit");
        },
      },
    ),
  ).toThrow("after commit");
  const [id, before] = [...journal.edges][0]!;
  expect(before).toBeNull();
  expect(harness.semanticEdgeRepository.getEdge(id!)).not.toBeNull();
});
