import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { renderWithInspector } from "../../test/inspector";
import { IdChip } from "./IdChip";

afterEach(() => {
  Reflect.deleteProperty(window.navigator, "clipboard");
  Reflect.deleteProperty(window, "isSecureContext");
  vi.restoreAllMocks();
});

describe("IdChip", () => {
  it("shows a short id and copies the full id", async () => {
    const writeText = vi.fn<Clipboard["writeText"]>().mockResolvedValue(undefined);
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    Object.defineProperty(window, "isSecureContext", {
      configurable: true,
      value: true,
    });

    renderWithInspector(<IdChip id="semn_abcdefghijklmnop" type="semantic_node" />);

    expect(screen.getByText("semn_abc…mnop")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "copy semn_abcdefghijklmnop" }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith("semn_abcdefghijklmnop"));
  });
});
