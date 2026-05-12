import { tripPlanningScenario } from "./trip-planning.js";
import type { SimulatorScenarioDefinition } from "../types.js";

export const simulatorScenarios = [tripPlanningScenario] as const;

export function findSimulatorScenario(key: string): SimulatorScenarioDefinition | undefined {
  return simulatorScenarios.find((scenario) => scenario.key === key);
}

export function scenarioPersonas(): SimulatorScenarioDefinition["personas"] {
  return simulatorScenarios.flatMap((scenario) => scenario.personas);
}

export { tripPlanningScenario };
