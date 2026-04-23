import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { commitmentMigrations, CommitmentRepository } from "../commitments/index.js";
import {
  AutobiographicalRepository,
  GrowthMarkersRepository,
  GoalsRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
  selfMigrations,
} from "../self/index.js";

import { identityMigrations } from "./migrations.js";
import { IdentityEventRepository } from "./repository.js";
import { IdentityService } from "./service.js";

function createHarness(clock: FixedClock | ManualClock) {
  const db = openDatabase(":memory:", {
    migrations: [...selfMigrations, ...commitmentMigrations, ...identityMigrations],
  });
  const identityEvents = new IdentityEventRepository({
    db,
    clock,
  });
  const valuesRepository = new ValuesRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const goalsRepository = new GoalsRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const traitsRepository = new TraitsRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const autobiographicalRepository = new AutobiographicalRepository({
    db,
    clock,
  });
  const growthMarkersRepository = new GrowthMarkersRepository({
    db,
    clock,
  });
  const openQuestionsRepository = new OpenQuestionsRepository({
    db,
    clock,
  });
  const commitmentRepository = new CommitmentRepository({
    db,
    clock,
    identityEventRepository: identityEvents,
  });
  const identity = new IdentityService({
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    growthMarkersRepository,
    openQuestionsRepository,
    commitmentRepository,
    identityEventRepository: identityEvents,
  });

  return {
    db,
    identityEvents,
    valuesRepository,
    goalsRepository,
    traitsRepository,
    autobiographicalRepository,
    growthMarkersRepository,
    openQuestionsRepository,
    commitmentRepository,
    identity,
  };
}

describe("identity service", () => {
  it("requires review for manual, system, and offline overwrites of established episode-backed values", () => {
    const harness = createHarness(new FixedClock(1_000));

    try {
      const value = harness.valuesRepository.add({
        label: "accuracy",
        description: "Prefer grounded claims.",
        priority: 8,
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_aaaaaaaaaaaaaaaa" as const],
        },
      });
      harness.valuesRepository.reinforce(value.id, {
        kind: "episodes",
        episode_ids: ["ep_bbbbbbbbbbbbbbbb" as const],
      });
      harness.valuesRepository.reinforce(value.id, {
        kind: "episodes",
        episode_ids: ["ep_cccccccccccccccc" as const],
      });

      expect(harness.valuesRepository.get(value.id)?.state).toBe("established");
      expect(
        harness.identity.updateValue(
          value.id,
          {
            description: "Prefer flexible claims.",
          },
          {
            kind: "manual",
          },
        ),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
      expect(
        harness.identity.updateValue(
          value.id,
          {
            description: "Prefer system-maintained claims.",
          },
          {
            kind: "system",
          },
        ),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
      expect(
        harness.identity.updateValue(
          value.id,
          {
            description: "Prefer offline-maintained claims.",
          },
          {
            kind: "offline",
            process: "reflector",
          },
        ),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
      expect(
        harness.identityEvents.list({
          recordType: "value",
          recordId: value.id,
        }),
      ).not.toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "update",
            overwrite_without_review: true,
          }),
        ]),
      );
    } finally {
      harness.db.close();
    }
  });

  it("continues blocking manual trait overwrites after episode-backed promotion", () => {
    const clock = new ManualClock(1_000);
    const harness = createHarness(clock);
    const episodeIds = [
      "ep_aaaaaaaaaaaaaaaa",
      "ep_bbbbbbbbbbbbbbbb",
      "ep_cccccccccccccccc",
      "ep_dddddddddddddddd",
      "ep_eeeeeeeeeeeeeeee",
    ] as const;

    try {
      harness.traitsRepository.reinforce({
        label: "engaged",
        delta: 0.05,
        provenance: {
          kind: "offline",
          process: "reflector",
        },
        timestamp: clock.now(),
      });

      for (const episodeId of episodeIds) {
        clock.advance(100);
        harness.traitsRepository.reinforce({
          label: "engaged",
          delta: 0.05,
          provenance: {
            kind: "episodes",
            episode_ids: [episodeId],
          },
          timestamp: clock.now(),
        });
      }

      const trait = harness.traitsRepository.list()[0];

      expect(trait?.state).toBe("established");
      expect(
        harness.identity.updateTrait(
          trait!.id,
          {
            strength: 0.1,
          },
          {
            kind: "manual",
          },
        ),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
    } finally {
      harness.db.close();
    }
  });

  it("guards active episode-backed goals from offline overwrites", () => {
    const harness = createHarness(new FixedClock(2_000));

    try {
      const goal = harness.goalsRepository.add({
        description: "Stabilize the release train",
        priority: 5,
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_goalgoalgoalgoal" as const],
        },
      });

      expect(
        harness.identity.updateGoal(
          goal.id,
          {
            progress_notes: "Offline heuristic progress.",
            last_progress_ts: 2_000,
          },
          {
            kind: "offline",
            process: "reflector",
          },
        ),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
    } finally {
      harness.db.close();
    }
  });

  it("guards episode-backed autobiographical periods, open questions, and growth markers", () => {
    const harness = createHarness(new FixedClock(3_000));

    try {
      const period = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q2",
        start_ts: 1_000,
        narrative: "Grounded in lived evidence.",
        key_episode_ids: ["ep_periodperiodperi" as const],
        themes: ["learning"],
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_periodperiodperi" as const],
        },
      });
      const question = harness.openQuestionsRepository.add({
        question: "What pattern am I missing?",
        urgency: 0.6,
        related_episode_ids: ["ep_openquestionaaaa" as const],
        source: "reflection",
      });
      const marker = harness.growthMarkersRepository.add({
        ts: 2_500,
        category: "understanding",
        what_changed: "Became more precise about review gates.",
        evidence_episode_ids: ["ep_growthmarkeraaaa" as const],
        confidence: 0.7,
        source_process: "self-narrator",
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_growthmarkeraaaa" as const],
        },
      });

      expect(
        harness.identity.updatePeriod(
          period.id,
          {
            narrative: "Offline rewrite.",
          },
          {
            kind: "offline",
            process: "self-narrator",
          },
        ),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
      expect(
        harness.identity.updateOpenQuestion(
          question.id,
          {
            urgency: 0.8,
          },
          {
            kind: "system",
          },
        ),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
      expect(
        harness.identity.updateGrowthMarker(
          marker.id,
          {
            after_description: "Offline rewritten marker.",
          },
          {
            kind: "offline",
            process: "ruminator",
          },
        ),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
    } finally {
      harness.db.close();
    }
  });
});
