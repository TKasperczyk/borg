import { describe, expect, it } from "vitest";

import {
  actionLifecycleArcKeys,
  actionLifecycleArcs,
  actionLifecycleScenario,
} from "./action-lifecycle.js";
import {
  actionArchiveLifecycleArcKeys,
  actionArchiveLifecycleArcs,
  actionArchiveLifecycleScenario,
} from "./action-archive-lifecycle.js";
import {
  beliefRevisionDomainArcKeys,
  beliefRevisionDomainArcs,
  beliefRevisionDomainsScenario,
} from "./belief-revision-domains.js";
import {
  capabilityBoundaryArcKeys,
  capabilityBoundaryArcs,
  capabilityBoundaryScenario,
} from "./capability-boundary.js";
import {
  criticalBoundaryRegenerationArcKeys,
  criticalBoundaryRegenerationArcs,
  criticalBoundaryRegenerationScenario,
} from "./critical-boundary-regeneration.js";
import {
  familyAgingParentArcKeys,
  familyAgingParentArcs,
  familyAgingParentScenario,
} from "./family-aging-parent.js";
import {
  kinshipCorrectnessArcKeys,
  kinshipCorrectnessArcs,
  kinshipCorrectnessScenario,
} from "./kinship-correctness.js";
import {
  kinshipHeadcountArcKeys,
  kinshipHeadcountArcs,
  kinshipHeadcountScenario,
} from "./kinship-headcount.js";
import {
  observationSourcePrecedenceArcKeys,
  observationSourcePrecedenceArcs,
  observationSourcePrecedenceScenario,
} from "./observation-source-precedence.js";
import {
  sharedStateCompactionArcKeys,
  sharedStateCompactionArcs,
  sharedStateCompactionScenario,
} from "./shared-state-compaction.js";
import { findSimulatorScenario, simulatorScenarios } from "./index.js";
import type { SimulatorScenarioDefinition } from "../types.js";

type PersonaArcBeat = {
  speakerKey: string;
  intent: string;
  triggers: readonly string[];
  satisfyingResponse: string;
  pushback?: string;
};

type RegressionScenarioArc = {
  key: string;
  title: string;
  beats: readonly PersonaArcBeat[];
  expectedSignals: readonly string[];
  requiredSignals: readonly string[];
};

type RegressionScenarioFixture = {
  scenario: SimulatorScenarioDefinition;
  exportedArcKeys: readonly string[];
  arcs: readonly RegressionScenarioArc[];
  expectedPersonas: readonly (readonly [string, string])[];
  expectedArcKeys: readonly string[];
};

const regressionScenarioFixtures: readonly RegressionScenarioFixture[] = [
  {
    scenario: capabilityBoundaryScenario,
    exportedArcKeys: capabilityBoundaryArcKeys,
    arcs: capabilityBoundaryArcs,
    expectedPersonas: [
      ["nina-capability", "Nina"],
      ["omar-capability", "Omar"],
      ["jade-capability", "Jade"],
    ],
    expectedArcKeys: [
      "external-doc-edit",
      "external-monitoring",
      "physical-attendance",
      "fictional-rpg",
      "scheduled-recurring",
    ],
  },
  {
    scenario: actionLifecycleScenario,
    exportedArcKeys: actionLifecycleArcKeys,
    arcs: actionLifecycleArcs,
    expectedPersonas: [
      ["sara-action", "Sara"],
      ["mike-action", "Mike"],
    ],
    expectedArcKeys: [
      "participant-completion",
      "action-dedup",
      "impossible-recurring-followup",
      "agenda-closure",
      "same-session-reactivation",
    ],
  },
  {
    scenario: beliefRevisionDomainsScenario,
    exportedArcKeys: beliefRevisionDomainArcKeys,
    arcs: beliefRevisionDomainArcs,
    expectedPersonas: [
      ["dana-belief", "Dana"],
      ["eli-belief", "Eli"],
    ],
    expectedArcKeys: [
      "project-decision-superseded",
      "incident-hypothesis-contradicted",
      "rpg-canon-correction",
      "family-boundary-revoked",
      "personal-preference-updated",
    ],
  },
  {
    scenario: familyAgingParentScenario,
    exportedArcKeys: familyAgingParentArcKeys,
    arcs: familyAgingParentArcs,
    expectedPersonas: [
      ["nora-family", "Nora"],
      ["julian-family", "Julian"],
      ["priya-family", "Priya"],
    ],
    expectedArcKeys: [
      "initial-concern-surfacing",
      "incident-accumulation",
      "mom-conversation-aftermath",
      "practical-care-planning",
      "plan-revision-and-boundary-revocation",
      "stable-cadence-and-capability-boundary",
    ],
  },
  {
    scenario: kinshipCorrectnessScenario,
    exportedArcKeys: kinshipCorrectnessArcKeys,
    arcs: kinshipCorrectnessArcs,
    expectedPersonas: [
      ["avery-kinship", "Avery"],
      ["leo-kinship", "Leo"],
      ["priya-kinship", "Priya"],
    ],
    expectedArcKeys: ["partner-not-sibling"],
  },
  {
    scenario: kinshipHeadcountScenario,
    exportedArcKeys: kinshipHeadcountArcKeys,
    arcs: kinshipHeadcountArcs,
    expectedPersonas: [
      ["lara-headcount", "Lara"],
      ["mateo-headcount", "Mateo"],
      ["esme-headcount", "Esme"],
    ],
    expectedArcKeys: ["ambiguous-family-headcount"],
  },
  {
    scenario: observationSourcePrecedenceScenario,
    exportedArcKeys: observationSourcePrecedenceArcKeys,
    arcs: observationSourcePrecedenceArcs,
    expectedPersonas: [["nora-observation", "Nora"]],
    expectedArcKeys: ["latest-user-observation"],
  },
  {
    scenario: sharedStateCompactionScenario,
    exportedArcKeys: sharedStateCompactionArcKeys,
    arcs: sharedStateCompactionArcs,
    expectedPersonas: [
      ["iris-compaction", "Iris"],
      ["jon-compaction", "Jon"],
    ],
    expectedArcKeys: ["central-plan-update-compaction"],
  },
  {
    scenario: actionArchiveLifecycleScenario,
    exportedArcKeys: actionArchiveLifecycleArcKeys,
    arcs: actionArchiveLifecycleArcs,
    expectedPersonas: [
      ["rhea-archive", "Rhea"],
      ["tomas-archive", "Tomas"],
    ],
    expectedArcKeys: ["inactive-action-archive-buckets"],
  },
  {
    scenario: criticalBoundaryRegenerationScenario,
    exportedArcKeys: criticalBoundaryRegenerationArcKeys,
    arcs: criticalBoundaryRegenerationArcs,
    expectedPersonas: [
      ["maya-regeneration", "Maya"],
      ["sol-regeneration", "Sol"],
    ],
    expectedArcKeys: ["dad-boundary-regeneration"],
  },
];

function expectNonEmptyText(value: string): void {
  expect(value.trim()).toBe(value);
  expect(value.length).toBeGreaterThan(0);
}

describe("simulator scenarios", () => {
  it("loads the coding incident scenario", () => {
    const scenario = findSimulatorScenario("coding-incident");

    expect(scenario?.key).toBe("coding-incident");
    expect(scenario?.personas.map((persona) => [persona.key, persona.displayName])).toEqual([
      ["sara-incident", "Sara"],
      ["mike-incident", "Mike"],
    ]);
  });

  it("registers all built-in scenarios", () => {
    expect(simulatorScenarios.map((scenario) => scenario.key)).toEqual([
      "trip-planning",
      "coding-incident",
      "family-aging-parent",
      "kinship-correctness",
      "kinship-headcount",
      "observation-source-precedence",
      "shared-state-compaction",
      "capability-boundary",
      "action-lifecycle",
      "action-archive-lifecycle",
      "belief-revision-domains",
      "critical-boundary-regeneration",
    ]);
  });

  it.each(regressionScenarioFixtures)(
    "loads and validates the $scenario.key regression scenario",
    ({ scenario, exportedArcKeys, arcs, expectedPersonas, expectedArcKeys }) => {
      expect(findSimulatorScenario(scenario.key)).toBe(scenario);
      expect(scenario.personas.map((persona) => [persona.key, persona.displayName])).toEqual(
        expectedPersonas,
      );

      expectNonEmptyText(scenario.key);
      expectNonEmptyText(scenario.description);
      expectNonEmptyText(scenario.channelName);
      expect(scenario.personas.length).toBeGreaterThanOrEqual(1);
      expect(scenario.personas.length).toBeLessThanOrEqual(4);

      const personaKeys = new Set(scenario.personas.map((persona) => persona.key));
      const personaNames = new Set(scenario.personas.map((persona) => persona.displayName));
      expect(personaKeys.size).toBe(scenario.personas.length);
      expect(personaNames.size).toBe(scenario.personas.length);

      for (const persona of scenario.personas) {
        expectNonEmptyText(persona.key);
        expectNonEmptyText(persona.displayName);
        expect(persona.systemPrompt.length).toBeGreaterThan(0);
        expect(persona.systemPrompt).toContain("Conversation motivation beats:");
        expect(persona.systemPrompt).toContain("Use these beats as motivations, not a script.");
        expect(persona.systemPrompt).not.toContain("Scripted message plan:");

        for (const seedFact of persona.seedFacts ?? []) {
          expectNonEmptyText(seedFact);
        }
      }

      expect(exportedArcKeys).toEqual(expectedArcKeys);
      expect(arcs.map((arc) => arc.key)).toEqual(expectedArcKeys);

      const promptByPersonaKey = new Map(
        scenario.personas.map((persona) => [persona.key, persona.systemPrompt] as const),
      );
      const beatPersonaKeys = new Set(
        arcs.flatMap((arc) => arc.beats.map((beat) => beat.speakerKey)),
      );
      expect([...beatPersonaKeys].sort()).toEqual([...personaKeys].sort());

      for (const arc of arcs) {
        expectNonEmptyText(arc.key);
        expectNonEmptyText(arc.title);
        expect(arc.beats.length).toBeGreaterThanOrEqual(1);
        expect(arc.expectedSignals.length).toBeGreaterThanOrEqual(1);
        expect(arc.requiredSignals.length).toBeGreaterThanOrEqual(1);

        for (const signal of arc.expectedSignals) {
          expectNonEmptyText(signal);
        }

        const arcText = [
          arc.key,
          arc.title,
          ...arc.expectedSignals,
          ...arc.beats.flatMap((beat) => [
            beat.intent,
            ...beat.triggers,
            beat.satisfyingResponse,
            beat.pushback ?? "",
          ]),
        ].join("\n");

        for (const signal of arc.requiredSignals) {
          expectNonEmptyText(signal);
          expect(arcText).toContain(signal);
        }

        for (const beat of arc.beats) {
          expect(personaKeys.has(beat.speakerKey)).toBe(true);
          expectNonEmptyText(beat.intent);
          expect(beat.triggers.length).toBeGreaterThanOrEqual(1);
          expectNonEmptyText(beat.satisfyingResponse);
          expect(promptByPersonaKey.get(beat.speakerKey) ?? "").toContain(beat.intent);

          for (const trigger of beat.triggers) {
            expectNonEmptyText(trigger);
          }

          if (beat.pushback !== undefined) {
            expectNonEmptyText(beat.pushback);
          }
        }
      }
    },
  );
});
