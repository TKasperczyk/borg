import { describe, expect, it } from "vitest";

import { createMigrations } from "./storage-setup.js";

describe("createMigrations", () => {
  it("assigns a unique name to every composed migration", () => {
    const names = createMigrations().map((migration) => migration.name);

    expect(names.length).toBeGreaterThan(0);
    expect(new Set(names).size).toBe(names.length);
  });
});
