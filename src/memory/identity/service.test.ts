import { afterEach, describe, expect, it, vi } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { createEpisodeId } from "../../util/ids.js";
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
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("requires review for manual, system, and offline overwrites of established episode-backed values", () => {
    const harness = createHarness(new FixedClock(1_000));

    try {
      const value = harness.valuesRepository.add({
        label: "accuracy",
        description: "Prefer grounded claims.",
        priority: 8,
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
        },
      });
      harness.valuesRepository.reinforce(value.id, {
        kind: "episodes",
        episode_ids: ["ep_bbbbbbbbbbbbbbbb" as never],
      });
      harness.valuesRepository.reinforce(value.id, {
        kind: "episodes",
        episode_ids: ["ep_cccccccccccccccc" as never],
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

  it("guards established records even when current provenance is non-episode", () => {
    const harness = createHarness(new FixedClock(1_500));

    try {
      const value = harness.valuesRepository.add({
        label: "continuity",
        description: "Keep identity changes grounded.",
        priority: 6,
        provenance: {
          kind: "system",
        },
      });
      harness.valuesRepository.update(
        value.id,
        {
          state: "established",
          established_at: 1_500,
          provenance: {
            kind: "system",
          },
        },
        {
          kind: "system",
        },
      );
      const goal = harness.goalsRepository.add({
        description: "Keep established goals protected",
        priority: 3,
        provenance: {
          kind: "manual",
        },
      });
      const period = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q2",
        start_ts: 1_000,
        narrative: "",
        key_episode_ids: [],
        themes: [],
        provenance: {
          kind: "system",
        },
      });

      expect(
        harness.identity.updateValue(
          value.id,
          {
            description: "Offline rewrite.",
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
        harness.identity.updateGoal(
          goal.id,
          {
            progress_notes: "Offline progress.",
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
            episode_ids: [episodeId as never],
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
          episode_ids: ["ep_goalgoalgoalgoal" as never],
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
        key_episode_ids: ["ep_periodperiodperi" as never],
        themes: ["learning"],
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_periodperiodperi" as never],
        },
      });
      const question = harness.openQuestionsRepository.add({
        question: "What pattern am I missing?",
        urgency: 0.6,
        related_episode_ids: ["ep_openquestionaaaa" as never],
        source: "reflection",
      });
      const marker = harness.growthMarkersRepository.add({
        ts: 2_500,
        category: "understanding",
        what_changed: "Became more precise about review gates.",
        evidence_episode_ids: ["ep_growthmarkeraaaa" as never],
        confidence: 0.7,
        source_process: "self-narrator",
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_growthmarkeraaaa" as never],
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

  it("records create events for guarded period, growth marker, and open question adds", () => {
    const harness = createHarness(new FixedClock(3_500));
    const periodEpisodeId = createEpisodeId();
    const markerEpisodeId = createEpisodeId();
    const questionEpisodeId = createEpisodeId();

    try {
      const period = harness.identity.addPeriod({
        label: "2026-Q4",
        start_ts: 3_000,
        narrative: "A guarded period began.",
        key_episode_ids: [periodEpisodeId],
        themes: ["guarding"],
        provenance: {
          kind: "episodes",
          episode_ids: [periodEpisodeId],
        },
      });
      const marker = harness.identity.addGrowthMarker({
        ts: 3_100,
        category: "understanding",
        what_changed: "Creation writes now carry identity events.",
        evidence_episode_ids: [markerEpisodeId],
        confidence: 0.8,
        source_process: "self-narrator",
        provenance: {
          kind: "episodes",
          episode_ids: [markerEpisodeId],
        },
      });
      const question = harness.identity.addOpenQuestion({
        question: "Which identity writes still need review?",
        urgency: 0.7,
        related_episode_ids: [questionEpisodeId],
        source: "reflection",
      });

      expect(
        harness.identityEvents.list({
          recordType: "autobiographical_period",
          recordId: period.id,
        })[0],
      ).toMatchObject({
        action: "create",
        old_value: null,
        new_value: expect.objectContaining({
          id: period.id,
        }),
      });
      expect(
        harness.identityEvents.list({
          recordType: "growth_marker",
          recordId: marker.id,
        })[0],
      ).toMatchObject({
        action: "create",
        old_value: null,
        new_value: expect.objectContaining({
          id: marker.id,
        }),
      });
      expect(
        harness.identityEvents.list({
          recordType: "open_question",
          recordId: question.id,
        })[0],
      ).toMatchObject({
        action: "create",
        old_value: null,
        new_value: expect.objectContaining({
          id: question.id,
        }),
        provenance: {
          kind: "episodes",
          episode_ids: [questionEpisodeId],
        },
      });
    } finally {
      harness.db.close();
    }
  });

  it("guards and audits the implicit close when opening a new period", () => {
    const harness = createHarness(new FixedClock(3_600));
    const firstEpisodeId = createEpisodeId();
    const secondEpisodeId = createEpisodeId();

    try {
      const first = harness.identity.addPeriod({
        label: "2027-Q1",
        start_ts: 3_000,
        narrative: "First open period.",
        key_episode_ids: [firstEpisodeId],
        themes: ["identity"],
        provenance: {
          kind: "episodes",
          episode_ids: [firstEpisodeId],
        },
      });

      expect(() =>
        harness.identity.addPeriod({
          label: "2027-Q2",
          start_ts: 4_000,
          narrative: "Manual rollover.",
          key_episode_ids: [],
          themes: ["manual"],
          provenance: {
            kind: "manual",
          },
        }),
      ).toThrow(/requires review/);
      expect(harness.autobiographicalRepository.getPeriod(first.id)?.end_ts).toBeNull();

      const second = harness.identity.addPeriod({
        label: "2027-Q2",
        start_ts: 4_000,
        narrative: "Episode-backed rollover.",
        key_episode_ids: [secondEpisodeId],
        themes: ["identity"],
        provenance: {
          kind: "episodes",
          episode_ids: [secondEpisodeId],
        },
      });

      expect(harness.autobiographicalRepository.getPeriod(first.id)?.end_ts).toBe(4_000);
      expect(
        harness.identityEvents.list({
          recordType: "autobiographical_period",
          recordId: first.id,
        }),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "close",
            new_value: expect.objectContaining({
              end_ts: 4_000,
            }),
          }),
        ]),
      );
      expect(
        harness.identityEvents.list({
          recordType: "autobiographical_period",
          recordId: second.id,
        })[0],
      ).toMatchObject({
        action: "create",
      });
    } finally {
      harness.db.close();
    }
  });

  it("guards period and open question state changes", () => {
    const harness = createHarness(new FixedClock(3_700));
    const periodEpisodeId = createEpisodeId();
    const questionEpisodeId = createEpisodeId();

    try {
      const period = harness.autobiographicalRepository.upsertPeriod({
        label: "2027-Q1",
        start_ts: 3_000,
        narrative: "Episode-backed period.",
        key_episode_ids: [periodEpisodeId],
        themes: ["identity"],
        provenance: {
          kind: "episodes",
          episode_ids: [periodEpisodeId],
        },
      });
      const question = harness.openQuestionsRepository.add({
        question: "What state transition should be guarded?",
        urgency: 0.5,
        related_episode_ids: [questionEpisodeId],
        provenance: {
          kind: "episodes",
          episode_ids: [questionEpisodeId],
        },
        source: "reflection",
      });

      expect(harness.identity.closePeriod(period.id, 3_600, { kind: "manual" })).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
      expect(
        harness.identity.resolveOpenQuestion(
          question.id,
          {
            resolution_episode_id: createEpisodeId(),
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
        harness.identity.abandonOpenQuestion(question.id, "Not useful now.", {
          kind: "manual",
        }),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
      expect(
        harness.identity.bumpOpenQuestionUrgency(question.id, 0.2, {
          kind: "offline",
          process: "ruminator",
        }),
      ).toEqual(
        expect.objectContaining({
          status: "requires_review",
        }),
      );
      expect(harness.autobiographicalRepository.getPeriod(period.id)?.end_ts).toBeNull();
      expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
        status: "open",
        urgency: 0.5,
      });
    } finally {
      harness.db.close();
    }
  });

  it("rolls back period, growth marker, and open question updates when event recording fails", () => {
    const harness = createHarness(new FixedClock(4_000));

    try {
      const period = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q3",
        start_ts: 1_000,
        narrative: "Original period narrative.",
        key_episode_ids: ["ep_periodrollbackaa" as never],
        themes: ["stability"],
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_periodrollbackaa" as never],
        },
      });
      const question = harness.openQuestionsRepository.add({
        question: "Which invariant matters?",
        urgency: 0.4,
        related_episode_ids: ["ep_questionrollback" as never],
        source: "reflection",
      });
      const marker = harness.growthMarkersRepository.add({
        ts: 3_500,
        category: "understanding",
        what_changed: "Learned the original pattern.",
        evidence_episode_ids: ["ep_markerrollbackaa" as never],
        confidence: 0.6,
        source_process: "self-narrator",
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_markerrollbackaa" as never],
        },
      });
      const eventError = new Error("identity event insert failed");
      const recordSpy = vi.spyOn(harness.identityEvents, "record").mockImplementation(() => {
        throw eventError;
      });
      const provenance = {
        kind: "system" as const,
      };

      expect(() =>
        harness.identity.updatePeriod(
          period.id,
          {
            narrative: "Updated period narrative.",
          },
          provenance,
          {
            throughReview: true,
          },
        ),
      ).toThrow(eventError);
      expect(() =>
        harness.identity.updateGrowthMarker(
          marker.id,
          {
            after_description: "Updated marker.",
          },
          provenance,
          {
            throughReview: true,
          },
        ),
      ).toThrow(eventError);
      expect(() =>
        harness.identity.updateOpenQuestion(
          question.id,
          {
            urgency: 0.9,
          },
          provenance,
          {
            throughReview: true,
          },
        ),
      ).toThrow(eventError);

      expect(recordSpy).toHaveBeenCalledTimes(3);
      expect(harness.autobiographicalRepository.getPeriod(period.id)?.narrative).toBe(
        "Original period narrative.",
      );
      expect(harness.growthMarkersRepository.get(marker.id)?.after_description).toBeNull();
      expect(harness.openQuestionsRepository.get(question.id)?.urgency).toBe(0.4);
    } finally {
      harness.db.close();
    }
  });

  it("rolls back guarded creates when identity event recording fails", () => {
    const harness = createHarness(new FixedClock(4_500));
    const eventError = new Error("identity event insert failed");
    const periodEpisodeId = createEpisodeId();
    const markerEpisodeId = createEpisodeId();
    const questionEpisodeId = createEpisodeId();

    try {
      const recordSpy = vi.spyOn(harness.identityEvents, "record").mockImplementation(() => {
        throw eventError;
      });

      expect(() =>
        harness.identity.addPeriod({
          label: "2027-Q2",
          start_ts: 4_000,
          narrative: "Created then rolled back.",
          key_episode_ids: [periodEpisodeId],
          themes: ["rollback"],
          provenance: {
            kind: "episodes",
            episode_ids: [periodEpisodeId],
          },
        }),
      ).toThrow(eventError);
      expect(() =>
        harness.identity.addGrowthMarker({
          ts: 4_100,
          category: "understanding",
          what_changed: "Creation rollback was tested.",
          evidence_episode_ids: [markerEpisodeId],
          confidence: 0.6,
          source_process: "self-narrator",
          provenance: {
            kind: "episodes",
            episode_ids: [markerEpisodeId],
          },
        }),
      ).toThrow(eventError);
      expect(() =>
        harness.identity.addOpenQuestion({
          question: "Will this failed create remain visible?",
          urgency: 0.4,
          related_episode_ids: [questionEpisodeId],
          source: "reflection",
        }),
      ).toThrow(eventError);

      expect(recordSpy).toHaveBeenCalledTimes(3);
      expect(harness.autobiographicalRepository.listPeriods()).toHaveLength(0);
      expect(harness.growthMarkersRepository.list()).toHaveLength(0);
      expect(harness.openQuestionsRepository.list()).toHaveLength(0);
    } finally {
      harness.db.close();
    }
  });

  it("rolls back guarded state changes when identity event recording fails", () => {
    const harness = createHarness(new FixedClock(5_000));
    const periodEpisodeId = createEpisodeId();
    const resolveQuestionEpisodeId = createEpisodeId();
    const abandonQuestionEpisodeId = createEpisodeId();
    const bumpQuestionEpisodeId = createEpisodeId();

    try {
      const period = harness.autobiographicalRepository.upsertPeriod({
        label: "2027-Q3",
        start_ts: 4_000,
        narrative: "Open period.",
        key_episode_ids: [periodEpisodeId],
        themes: ["rollback"],
        provenance: {
          kind: "episodes",
          episode_ids: [periodEpisodeId],
        },
      });
      const resolveQuestion = harness.openQuestionsRepository.add({
        question: "Will resolve roll back?",
        urgency: 0.4,
        related_episode_ids: [resolveQuestionEpisodeId],
        source: "reflection",
      });
      const abandonQuestion = harness.openQuestionsRepository.add({
        question: "Will abandon roll back?",
        urgency: 0.5,
        related_episode_ids: [abandonQuestionEpisodeId],
        source: "reflection",
      });
      const bumpQuestion = harness.openQuestionsRepository.add({
        question: "Will bump roll back?",
        urgency: 0.6,
        related_episode_ids: [bumpQuestionEpisodeId],
        source: "reflection",
      });
      const eventError = new Error("identity event insert failed");
      const recordSpy = vi.spyOn(harness.identityEvents, "record").mockImplementation(() => {
        throw eventError;
      });
      const provenance = {
        kind: "episodes" as const,
        episode_ids: [createEpisodeId()],
      };

      expect(() =>
        harness.identity.closePeriod(period.id, 4_900, provenance, {
          throughReview: true,
        }),
      ).toThrow(eventError);
      expect(() =>
        harness.identity.resolveOpenQuestion(
          resolveQuestion.id,
          {
            resolution_episode_id: createEpisodeId(),
          },
          provenance,
          {
            throughReview: true,
          },
        ),
      ).toThrow(eventError);
      expect(() =>
        harness.identity.abandonOpenQuestion(abandonQuestion.id, "Rollback reason.", provenance, {
          throughReview: true,
        }),
      ).toThrow(eventError);
      expect(() =>
        harness.identity.bumpOpenQuestionUrgency(bumpQuestion.id, 0.2, provenance, {
          throughReview: true,
        }),
      ).toThrow(eventError);

      expect(recordSpy).toHaveBeenCalledTimes(4);
      expect(harness.autobiographicalRepository.getPeriod(period.id)?.end_ts).toBeNull();
      expect(harness.openQuestionsRepository.get(resolveQuestion.id)?.status).toBe("open");
      expect(harness.openQuestionsRepository.get(abandonQuestion.id)?.status).toBe("open");
      expect(harness.openQuestionsRepository.get(bumpQuestion.id)?.urgency).toBe(0.6);
    } finally {
      harness.db.close();
    }
  });
});
