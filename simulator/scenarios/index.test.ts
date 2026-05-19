import { describe, expect, it } from "vitest";

import { findSimulatorScenario } from "./index.js";

describe("simulator scenarios", () => {
  it("loads the coding incident scenario", () => {
    const scenario = findSimulatorScenario("coding-incident");

    expect(scenario?.key).toBe("coding-incident");
    expect(scenario?.personas.map((persona) => [persona.key, persona.displayName])).toEqual([
      ["sara-incident", "Sara"],
      ["mike-incident", "Mike"],
    ]);
  });
});
