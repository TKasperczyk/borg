import { z } from "zod";

import { consolidatorPlanSchema } from "./consolidator/index.js";
import { curatorPlanSchema } from "./curator/index.js";
import { reflectorPlanSchema } from "./reflector/index.js";
import { overseerPlanSchema } from "./overseer/index.js";

export const offlineProcessPlanSchema = z.discriminatedUnion("process", [
  consolidatorPlanSchema,
  reflectorPlanSchema,
  curatorPlanSchema,
  overseerPlanSchema,
]);

export const maintenancePlanSchema = z.object({
  kind: z.literal("borg_maintenance_plan"),
  version: z.literal(1),
  created_at: z.number().finite(),
  processes: z.array(offlineProcessPlanSchema),
});

export type OfflineMaintenanceProcessPlan = z.infer<typeof offlineProcessPlanSchema>;
export type MaintenancePlan = z.infer<typeof maintenancePlanSchema>;
