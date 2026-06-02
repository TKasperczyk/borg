import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SemanticNodeDetail } from "./SemanticNodeDetail";

describe("SemanticNodeDetail", () => {
  it("renders semantic node content and provenance fields", () => {
    render(
      <SemanticNodeDetail
        label="candidate 1"
        node={{
          id: "semn_detail0000000",
          kind: "proposition",
          label: "Detailed node",
          description: "Full semantic node description.",
          domain: "project",
          aliases: ["Detailed alias"],
          confidence: 0.83,
          status: "active",
          source_episode_ids: ["ep_source000000000"],
          source_count: 1,
          created_at: 1,
          updated_at: 2,
        }}
      />,
    );

    expect(screen.getByText("candidate 1")).toBeInTheDocument();
    expect(screen.getByText("Detailed node")).toBeInTheDocument();
    expect(screen.getByText("Full semantic node description.")).toBeInTheDocument();
    expect(screen.getByText("confidence 0.83")).toBeInTheDocument();
    expect(screen.getByText("project")).toBeInTheDocument();
    expect(screen.getByText("Detailed alias")).toBeInTheDocument();
    expect(screen.getByText("ep_source000000000")).toBeInTheDocument();
  });
});
