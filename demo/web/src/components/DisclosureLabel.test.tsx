import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DisclosureLabel, collapseDisclosureClass } from "./DisclosureLabel";

describe("DisclosureLabel", () => {
  it("collapses the six memory disclosure classes without demoting private classes to public", () => {
    expect(collapseDisclosureClass("public")).toBe("public");
    expect(collapseDisclosureClass("relationship_private")).toBe("private");
    expect(collapseDisclosureClass("self_private")).toBe("private");
    expect(collapseDisclosureClass("operator_private")).toBe("operator");
    expect(collapseDisclosureClass("sensitive")).toBe("unknown");
    expect(collapseDisclosureClass("unknown")).toBe("unknown");
  });

  it("fails closed to unknown for unrecognized values", () => {
    expect(collapseDisclosureClass("private")).toBe("unknown");
    expect(collapseDisclosureClass("operator")).toBe("unknown");
    expect(collapseDisclosureClass("anything_else")).toBe("unknown");
    expect(collapseDisclosureClass(null)).toBe("unknown");

    render(<DisclosureLabel value="anything_else" />);

    expect(screen.getByText("unknown")).toBeInTheDocument();
  });
});
