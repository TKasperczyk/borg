import { describe, expect, it } from "vitest";

import { displayTargetSummary, displayValue, isInternalId } from "./screen-utils";

describe("screen-utils display id detection", () => {
  it("shortens structurally valid Borg ids", () => {
    expect(isInternalId("ent_abcdefghijklmnop")).toBe(true);
    expect(displayValue("ent_abcdefghijklmnop")).toBe("ent_abcd…mnop");
    expect(displayTargetSummary({ entity_id: "ent_abcdefghijklmnop" })).toBe(
      "entity ent_abcd…mnop",
    );
  });

  it("keeps semantic enum values literal", () => {
    expect(isInternalId("scheduled_reflection")).toBe(false);
    expect(displayValue("scheduled_reflection")).toBe("scheduled_reflection");
    expect(displayTargetSummary({ process: "scheduled_reflection" })).toBe(
      "process scheduled_reflection",
    );
  });

  it("rejects prefixed strings that do not match the real id structure", () => {
    expect(isInternalId("ent_xjqlqsbnnyiw3Bx")).toBe(false);
    expect(displayValue("ent_xjqlqsbnnyiw3Bx")).toBe("ent_xjqlqsbnnyiw3Bx");
    expect(isInternalId("procevi_abcdef0123456789")).toBe(true);
    expect(displayValue("procevi_abcdef0123456789")).toBe("procevi_…6789");
  });
});
