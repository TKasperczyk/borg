import { autonomousWakeScenario } from "./autonomous-wake.js";
import { commitmentRespectScenario } from "./commitment-respect.js";
import { contradictionHandlingScenario } from "./contradiction-handling.js";
import { goalProgressTrackingScenario } from "./goal-progress-tracking.js";
import { identityGuardRefusalScenario } from "./identity-guard-refusal.js";
import { moodPersistenceScenario } from "./mood-persistence.js";
import { multiSessionContinuityScenario } from "./multi-session-continuity.js";
import { openQuestionCreationScenario } from "./open-question-creation.js";
import { recallScenario } from "./recall.js";
import { toolUseCorrectnessScenario } from "./tool-use-correctness.js";
import { failingMockFixtureScenario } from "./failing-mock-fixture.js";
import type { Scenario } from "../types.js";

export const SCENARIOS: readonly Scenario[] = [
  recallScenario,
  commitmentRespectScenario,
  contradictionHandlingScenario,
  goalProgressTrackingScenario,
  autonomousWakeScenario,
  identityGuardRefusalScenario,
  toolUseCorrectnessScenario,
  openQuestionCreationScenario,
  moodPersistenceScenario,
  multiSessionContinuityScenario,
] as const;

export function getScenario(name: string): Scenario | undefined {
  return [...SCENARIOS, failingMockFixtureScenario].find((scenario) => scenario.name === name);
}

export { failingMockFixtureScenario };
