import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { activateOnEnterOrSpace, isInteractiveDescendantEvent } from "./keyboard";

function RowHarness({ onNested, onRow }: { onNested: () => void; onRow: () => void }) {
  return (
    <div
      data-testid="row"
      tabIndex={0}
      onClick={(event) => {
        if (!isInteractiveDescendantEvent(event.currentTarget, event.target)) {
          onRow();
        }
      }}
      onKeyDown={(event) => activateOnEnterOrSpace(event, onRow)}
    >
      <span>row surface</span>
      <button type="button" onClick={onNested}>
        nested control
      </button>
    </div>
  );
}

describe("keyboard activation helpers", () => {
  it("ignores click and Enter events from nested interactive descendants", () => {
    const onNested = vi.fn();
    const onRow = vi.fn();

    render(<RowHarness onNested={onNested} onRow={onRow} />);

    fireEvent.click(screen.getByRole("button", { name: "nested control" }));
    expect(onNested).toHaveBeenCalledTimes(1);
    expect(onRow).not.toHaveBeenCalled();

    fireEvent.keyDown(screen.getByRole("button", { name: "nested control" }), { key: "Enter" });
    expect(onRow).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText("row surface"));
    expect(onRow).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(screen.getByTestId("row"), { key: "Enter" });
    expect(onRow).toHaveBeenCalledTimes(2);
  });
});
