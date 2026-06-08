import { describe, expect, it } from "vitest";

import { createSemanticNodeFixture } from "../../../offline/test-support.js";
import type { RetrievedSemantic } from "../../../retrieval/index.js";
import {
  DEFAULT_SESSION_ID,
  createEntityId,
  createEpisodeId,
  createSemanticEdgeId,
} from "../../../util/ids.js";
import type { BuilderSectionContext } from "../builder-context.js";
import { createSectionBuckets } from "../section-buckets.js";
import { addSemanticGraphSection } from "./semantic-graph.js";

describe("evidence-ledger semantic graph section", () => {
  it("renders disclosure labels for private semantic nodes and edges", () => {
    const alice = createEntityId();
    const episodeId = createEpisodeId();
    const buckets = createSectionBuckets();
    const disclosureLabel = {
      disclosureClass: "relationship_private" as const,
      originAudienceEntityIds: [alice],
      privateToEntityIds: [alice],
      publicToEntityIds: [],
    };
    const root = {
      ...createSemanticNodeFixture({
        label: "Alice private semantic root",
        source_episode_ids: [episodeId],
      }),
      disclosureLabel,
    } satisfies RetrievedSemantic["matched_nodes"][number];
    const support = {
      ...createSemanticNodeFixture({
        label: "Alice private semantic support",
        source_episode_ids: [episodeId],
      }),
      disclosureLabel,
    } satisfies RetrievedSemantic["support_hits"][number]["node"];
    const edge = {
      id: createSemanticEdgeId(),
      from_node_id: root.id,
      to_node_id: support.id,
      relation: "supports" as const,
      confidence: 0.8,
      evidence_episode_ids: [episodeId],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
      valid_from: 1_000_000,
      valid_to: null,
      invalidated_at: null,
      invalidated_by_edge_id: null,
      invalidated_by_review_id: null,
      invalidated_by_process: null,
      invalidated_reason: null,
      disclosureLabel,
    } satisfies RetrievedSemantic["support_hits"][number]["edgePath"][number];

    addSemanticGraphSection({
      input: {
        retrievedSemantic: {
          as_of: null,
          supports: [],
          contradicts: [],
          categories: [],
          matched_node_ids: [root.id],
          matched_nodes: [root],
          support_hits: [
            {
              root_node_id: root.id,
              node: support,
              edgePath: [edge],
            },
          ],
          causal_hits: [],
          contradiction_hits: [],
          category_hits: [],
        },
      },
      resolver: {
        currentSessionId: DEFAULT_SESSION_ID,
        streamEntriesById: new Map(),
        streamOrderById: new Map(),
        episodeScopesById: new Map(),
        episodeSourceStreamIdsById: new Map(),
      },
      buckets,
    } as unknown as BuilderSectionContext);

    const entries = buckets.get("semantic_graph")?.entries ?? [];
    const rootEntry = entries.find((entry) => entry.id === `semantic_node:${root.id}`);
    const edgeEntry = entries.find((entry) => entry.id === `semantic_edge:${edge.id}`);

    expect(rootEntry?.state).toContain("disclosure_class=relationship_private");
    expect(rootEntry?.state).toContain(`private-to=${alice}`);
    expect(rootEntry?.state).toContain("supported by private source episodes");
    expect(rootEntry?.state_metadata).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [alice],
        private_to_entity_ids: [alice],
      },
      disclosure_note:
        "supported by private source episodes; I can use this internally; I do not reveal source details to the current audience unless authorized",
    });
    expect(edgeEntry?.state).toContain("disclosure_class=relationship_private");
    expect(edgeEntry?.state_metadata).toMatchObject({
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [alice],
        private_to_entity_ids: [alice],
      },
    });
  });
});
