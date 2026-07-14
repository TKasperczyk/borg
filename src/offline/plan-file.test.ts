import { describe, expect, it } from "vitest";

import { maintenanceRunIdHelpers } from "../util/ids.js";
import { maintenancePlanSchema } from "./plan-file.js";

describe("maintenancePlanSchema", () => {
  it("upgrades a serialized v1 plan by assigning a validated run id", () => {
    const plan = maintenancePlanSchema.parse({
      kind: "borg_maintenance_plan",
      version: 1,
      created_at: 1_000,
      processes: [],
    });

    expect(plan).toMatchObject({
      kind: "borg_maintenance_plan",
      version: 2,
      created_at: 1_000,
      processes: [],
    });
    expect(maintenanceRunIdHelpers.is(plan.run_id)).toBe(true);
  });
});
