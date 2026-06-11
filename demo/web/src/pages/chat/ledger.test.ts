import type { EvidenceLedger } from "../../api/types";
import { summarizeLedger } from "./ledger";

describe("ledger summary", () => {
  it("aggregates real ledger sections by source_type and counts disclosure metadata", () => {
    const ledger: EvidenceLedger = {
      sections: [
        {
          id: "episodes",
          label: "Episodes",
          entries: [
            {
              id: "episode:1",
              source_type: "episode",
              state_metadata: {
                disclosure_label: { disclosure_class: "relationship_private" },
              },
            },
            {
              id: "episode:2",
              source_type: "episode",
              state: "disclosure_class=relationship_private",
            },
          ],
        },
        {
          id: "semantic_graph",
          label: "Semantic Graph",
          entries: [
            { id: "semantic_node:1", source_type: "semantic_node" },
            { id: "semantic_edge:1", source_type: "semantic_edge" },
          ],
        },
      ],
      transcriptIncluded: true,
      transcriptCompacted: false,
      originalTranscriptTokenEstimate: 0,
      compactedTranscriptEntryCount: 0,
      rawPreservedUserTranscriptEntryCount: 0,
      estimatedTokens: 0,
    };

    expect(summarizeLedger(ledger)).toMatchObject({
      chips: [
        { key: "EPI", value: 2 },
        { key: "SEM", value: 2 },
      ],
      disclosureCount: 1,
      totalEntries: 4,
    });
  });

  it("handles a null ledger honestly", () => {
    expect(summarizeLedger(null)).toBeNull();
  });
});
