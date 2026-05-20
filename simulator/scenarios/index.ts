import { actionLifecycleScenario } from "./action-lifecycle.js";
import { beliefRevisionDomainsScenario } from "./belief-revision-domains.js";
import { capabilityBoundaryScenario } from "./capability-boundary.js";
import { codingIncidentScenario } from "./coding-incident.js";
import { familyAgingParentScenario } from "./family-aging-parent.js";
import { kinshipCorrectnessScenario } from "./kinship-correctness.js";
import { tripPlanningScenario } from "./trip-planning.js";
import type { SimulatorScenarioDefinition } from "../types.js";

export const simulatorScenarios = [
  tripPlanningScenario,
  codingIncidentScenario,
  familyAgingParentScenario,
  kinshipCorrectnessScenario,
  capabilityBoundaryScenario,
  actionLifecycleScenario,
  beliefRevisionDomainsScenario,
] as const;

export function findSimulatorScenario(key: string): SimulatorScenarioDefinition | undefined {
  return simulatorScenarios.find((scenario) => scenario.key === key);
}

export function scenarioPersonas(): SimulatorScenarioDefinition["personas"] {
  return simulatorScenarios.flatMap((scenario) => scenario.personas);
}

export {
  actionLifecycleScenario,
  beliefRevisionDomainsScenario,
  capabilityBoundaryScenario,
  codingIncidentScenario,
  familyAgingParentScenario,
  kinshipCorrectnessScenario,
  tripPlanningScenario,
};
