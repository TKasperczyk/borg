import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { Rail } from "./Rail";

describe("Rail", () => {
  it("renders route badges from data and preserves route selection", () => {
    const setRoute = vi.fn();

    render(
      <Rail
        route="cognition"
        setRoute={setRoute}
        badges={{ review: { count: 3, severity: 2, label: "open reviews" } }}
      />,
    );

    const review = screen.getByRole("button", { name: "review" });
    expect(within(review).getByLabelText("open reviews: 3")).toHaveClass("sev-2");

    fireEvent.click(review);

    expect(setRoute).toHaveBeenCalledWith("review");
  });
});
