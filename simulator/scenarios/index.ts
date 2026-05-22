import { actionLifecycleScenario } from "./action-lifecycle.js";
import { actionArchiveLifecycleScenario } from "./action-archive-lifecycle.js";
import { beliefRevisionDomainsScenario } from "./belief-revision-domains.js";
import { capabilityBoundaryScenario } from "./capability-boundary.js";
import { codingIncidentScenario } from "./coding-incident.js";
import { criticalBoundaryRegenerationScenario } from "./critical-boundary-regeneration.js";
import { familyAgingParentScenario } from "./family-aging-parent.js";
import { kinshipCorrectnessScenario } from "./kinship-correctness.js";
import { kinshipHeadcountScenario } from "./kinship-headcount.js";
import { observationSourcePrecedenceScenario } from "./observation-source-precedence.js";
import { sessionReentryContinuityScenario } from "./session-reentry-continuity.js";
import { sharedStateCompactionScenario } from "./shared-state-compaction.js";
import { tripPlanningScenario } from "./trip-planning.js";
import type { SimulatorScenarioDefinition } from "../types.js";

export const simulatorScenarios = [
  tripPlanningScenario,
  codingIncidentScenario,
  familyAgingParentScenario,
  kinshipCorrectnessScenario,
  kinshipHeadcountScenario,
  observationSourcePrecedenceScenario,
  sessionReentryContinuityScenario,
  sharedStateCompactionScenario,
  capabilityBoundaryScenario,
  actionLifecycleScenario,
  actionArchiveLifecycleScenario,
  beliefRevisionDomainsScenario,
  criticalBoundaryRegenerationScenario,
] as const;

export function findSimulatorScenario(key: string): SimulatorScenarioDefinition | undefined {
  return simulatorScenarios.find((scenario) => scenario.key === key);
}

export function scenarioPersonas(): SimulatorScenarioDefinition["personas"] {
  return simulatorScenarios.flatMap((scenario) => scenario.personas);
}

export {
  actionLifecycleScenario,
  actionArchiveLifecycleScenario,
  beliefRevisionDomainsScenario,
  capabilityBoundaryScenario,
  codingIncidentScenario,
  criticalBoundaryRegenerationScenario,
  familyAgingParentScenario,
  kinshipCorrectnessScenario,
  kinshipHeadcountScenario,
  observationSourcePrecedenceScenario,
  sessionReentryContinuityScenario,
  sharedStateCompactionScenario,
  tripPlanningScenario,
};
