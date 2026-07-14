import { z } from "zod";

import {
  createMaintenanceRunId,
  maintenanceRunIdHelpers,
  parseMaintenanceRunId,
} from "../util/ids.js";

import { associatorPlanSchema } from "./associator/index.js";
import { beliefReviserPlanSchema } from "./belief-reviser/index.js";
import { commitmentReconcilerPlanSchema } from "./commitment-reconciler/index.js";
import { consolidatorPlanSchema } from "./consolidator/index.js";
import { creatorDirectiveReconcilerPlanSchema } from "./creator-directive-reconciler/index.js";
import { curatorPlanSchema } from "./curator/index.js";
import { ruminatorPlanSchema } from "./ruminator/index.js";
import { reflectorPlanSchema } from "./reflector/index.js";
import { reviewResolverPlanSchema } from "./review-resolver/index.js";
import { semanticExtractorProcessPlanSchema } from "./semantic-extractor/index.js";
import { selfNarratorPlanSchema } from "./self-narrator/index.js";
import { overseerPlanSchema } from "./overseer/index.js";
import { proceduralSynthesizerPlanSchema } from "./procedural-synthesizer/index.js";

export const offlineProcessPlanSchema = z.discriminatedUnion("process", [
  consolidatorPlanSchema,
  reflectorPlanSchema,
  semanticExtractorProcessPlanSchema,
  curatorPlanSchema,
  overseerPlanSchema,
  associatorPlanSchema,
  reviewResolverPlanSchema,
  ruminatorPlanSchema,
  selfNarratorPlanSchema,
  proceduralSynthesizerPlanSchema,
  beliefReviserPlanSchema,
  creatorDirectiveReconcilerPlanSchema,
  commitmentReconcilerPlanSchema,
]);

const maintenanceRunIdSchema = z
  .string()
  .refine((value) => maintenanceRunIdHelpers.is(value), {
    message: "Invalid maintenance run id",
  })
  .transform((value) => parseMaintenanceRunId(value));

const maintenancePlanV1Schema = z.object({
  kind: z.literal("borg_maintenance_plan"),
  version: z.literal(1),
  run_id: maintenanceRunIdSchema.optional(),
  created_at: z.number().finite(),
  processes: z.array(offlineProcessPlanSchema),
});

const maintenancePlanV2Schema = z.object({
  kind: z.literal("borg_maintenance_plan"),
  version: z.literal(2),
  run_id: maintenanceRunIdSchema,
  created_at: z.number().finite(),
  processes: z.array(offlineProcessPlanSchema),
});

export const maintenancePlanSchema = z
  .discriminatedUnion("version", [maintenancePlanV1Schema, maintenancePlanV2Schema])
  .transform((plan) =>
    plan.version === 2
      ? plan
      : {
          ...plan,
          version: 2 as const,
          run_id: plan.run_id ?? createMaintenanceRunId(),
        },
  );

export type OfflineMaintenanceProcessPlan = z.infer<typeof offlineProcessPlanSchema>;
export type MaintenancePlan = z.infer<typeof maintenancePlanSchema>;
