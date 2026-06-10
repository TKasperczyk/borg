import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { renderWithInspector } from "../test/inspector";
import { PolicyValue } from "./PolicyValue";

describe("PolicyValue", () => {
  it("renders disclosure and activation scope values with explicit boundary domains", () => {
    render(
      <>
        <PolicyValue domain="content_scope" value="allow_list" />
        <PolicyValue domain="activation_scope" value="same_as_disclosure" />
      </>,
    );

    expect(screen.getByText("allow_list")).toHaveClass("tag", "purple");
    expect(screen.getByText("same_as_disclosure")).toHaveClass("tag", "info");
  });

  it("does not reuse privacy colors for mention policy chips", () => {
    render(<PolicyValue domain="mention_policy" value="answer_if_asked" />);

    const chip = screen.getByText("answer_if_asked");
    expect(chip).toHaveClass("tag");
    expect(chip).not.toHaveClass("info");
    expect(chip).not.toHaveClass("purple");
    expect(chip).not.toHaveClass("warn");
  });

  it("renders privacy and participation policy values through their own domains", () => {
    render(
      <>
        <PolicyValue domain="privacy_level" value="payload_on" />
        <PolicyValue domain="participation_policy" value="muted" />
      </>,
    );

    expect(screen.getByText("payload_on")).toHaveClass("tag", "warn");
    expect(screen.getByText("muted")).toHaveClass("tag", "solid");
  });

  it("wraps entity lists with count and entity refs", () => {
    renderWithInspector(
      <PolicyValue domain="entity-list" mode="excluded" value={["ent_aaaaaaaaaaaaaaaa"]} />,
    );

    expect(screen.getByText("1 entity")).toHaveClass("tag", "warn");
    expect(
      screen.getByRole("button", { name: "jump to ent_aaaaaaaaaaaaaaaa" }),
    ).toBeInTheDocument();
  });

  it("renders malformed entity-list values as raw fallback instead of a fabricated count", () => {
    render(<PolicyValue domain="entity-list" mode="allowed" value={{ bad: true }} />);

    expect(screen.getByText("{1 fields}")).toHaveClass("policy-value", "dim");
    expect(screen.queryByText("0 entities")).not.toBeInTheDocument();
  });
});
