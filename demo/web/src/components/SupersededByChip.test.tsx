import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithInspector } from "../test/inspector";
import { SupersededByChip } from "./SupersededByChip";

describe("SupersededByChip", () => {
  it("keeps the local jump action and exposes inspector access", async () => {
    const onOpen = vi.fn();

    renderWithInspector(
      <SupersededByChip
        id="run_superseded1111"
        label="replacement"
        onOpen={onOpen}
        inspectType="maintenance_run"
      />,
      { inspector: true },
    );

    fireEvent.click(screen.getByRole("button", { name: "jump to run_superseded1111" }));
    expect(onOpen).toHaveBeenCalledWith("run_superseded1111");

    fireEvent.click(screen.getByRole("button", { name: "inspect run_superseded1111" }));

    expect(
      await screen.findByRole("dialog", { name: "Maintenance run inspector" }),
    ).toBeInTheDocument();
    expect(
      await screen.findByText(
        "Maintenance run does not have a direct resolver for run_superseded1111.",
      ),
    ).toBeInTheDocument();
  });
});
