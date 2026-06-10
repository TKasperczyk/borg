import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ErrorState } from "./ErrorState";

describe("ErrorState", () => {
  it("renders a red alert with an optional retry action", () => {
    const onRetry = vi.fn();

    render(<ErrorState onRetry={onRetry}>failed to load</ErrorState>);

    expect(screen.getByRole("alert")).toHaveClass("notice", "error", "bad");
    const retry = screen.getByRole("button", { name: "retry" });
    expect(retry).toHaveClass("ghost");
    expect(retry).not.toHaveClass("danger");
    fireEvent.click(retry);

    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
