import { afterEach, describe, expect, it } from "vitest";

import { createEpisodeFixture, createOfflineTestHarness } from "../../offline/test-support.js";
import { createEpisodeId, createProceduralEvidenceId, createSkillId } from "../../util/ids.js";

import { deriveProceduralContextKey, proceduralContextSchema } from "./context.js";
import { SkillSelector } from "./selector.js";

describe("SkillRepository", () => {
  let harness: Awaited<ReturnType<typeof createOfflineTestHarness>> | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("adds, gets, deletes, and updates outcome statistics", async () => {
    harness = await createOfflineTestHarness();
    const episode = createEpisodeFixture();
    await harness.episodicRepository.insert(episode);
    const skill = await harness.skillRepository.add({
      applies_when: "Rust lifetimes throw borrow checker errors",
      approach: "Reduce borrow scope and clone only at boundaries.",
      sourceEpisodes: [episode.id],
    });

    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      applies_when: "Rust lifetimes throw borrow checker errors",
      alpha: 1,
      beta: 1,
    });

    const success = harness.skillRepository.recordOutcome(skill.id, true, episode.id);
    const failure = harness.skillRepository.recordOutcome(skill.id, false);

    expect(success.alpha).toBe(2);
    expect(success.successes).toBe(1);
    expect(failure.beta).toBe(2);
    expect(failure.failures).toBe(1);
    expect(harness.skillRepository.getStats(skill.id).mean).toBeCloseTo(0.5, 3);

    await expect(harness.skillRepository.delete(skill.id)).resolves.toBe(true);
    expect(harness.skillRepository.get(skill.id)).toBeNull();
  });

  it("selects stronger skills more often and breaks ties toward fewer attempts", async () => {
    harness = await createOfflineTestHarness();
    const episode = createEpisodeFixture();
    await harness.episodicRepository.insert(episode);
    const strong = await harness.skillRepository.add({
      id: createSkillId(),
      applies_when: "Rust lifetime debugging",
      approach: "Introduce intermediate bindings.",
      sourceEpisodes: [episode.id],
      priorAlpha: 8,
      priorBeta: 2,
    });
    const weak = await harness.skillRepository.add({
      id: createSkillId(),
      applies_when: "Rust lifetime debugging",
      approach: "Guess and rerun.",
      sourceEpisodes: [episode.id],
      priorAlpha: 2,
      priorBeta: 8,
    });

    let strongSelections = 0;
    const selector = new SkillSelector({
      repository: harness.skillRepository,
      rng: (() => {
        let seed = 123456789;
        return () => {
          seed = (1664525 * seed + 1013904223) % 0x1_0000_0000;
          return seed / 0x1_0000_0000;
        };
      })(),
    });

    for (let index = 0; index < 200; index += 1) {
      const selection = await selector.select("Rust lifetime debugging", {
        k: 5,
      });

      if (selection?.skill.id === strong.id) {
        strongSelections += 1;
      }
    }

    expect(strongSelections).toBeGreaterThan(130);

    const tieSelector = new SkillSelector({
      repository: harness.skillRepository,
      sampler: () => 0.5,
    });
    harness.skillRepository.recordOutcome(strong.id, true);
    const tieSelection = await tieSelector.select("Rust lifetime debugging", {
      k: 5,
    });

    expect(tieSelection?.skill.id).toBe(weak.id);
  });

  it("updates outcome counters atomically under parallel writers", async () => {
    harness = await createOfflineTestHarness();
    const episode = createEpisodeFixture();
    await harness.episodicRepository.insert(episode);
    const skill = await harness.skillRepository.add({
      applies_when: "Shared concurrent skill",
      approach: "Use atomic counters in SQL.",
      sourceEpisodes: [episode.id],
    });

    await Promise.all(
      Array.from({ length: 100 }, (_, index) =>
        Promise.resolve().then(() =>
          harness?.skillRepository.recordOutcome(skill.id, index % 2 === 0, episode.id),
        ),
      ),
    );

    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      alpha: 51,
      beta: 51,
      attempts: 100,
      successes: 50,
      failures: 50,
    });
  });

  it("records skill context outcomes by primary key and lists context usage", async () => {
    harness = await createOfflineTestHarness();
    const skillId = createSkillId();
    const otherSkillId = createSkillId();
    const contextKey = deriveProceduralContextKey({
      problem_kind: "code_debugging",
      domain_tags: ["TypeScript"],
      audience_scope: "self",
    });

    expect(
      harness.proceduralContextStatsRepository.getContextStats(skillId, contextKey),
    ).toBeNull();

    const success = harness.proceduralContextStatsRepository.recordContextOutcome({
      skillId,
      contextKey,
      success: true,
      ts: 1_000,
    });
    const failure = harness.proceduralContextStatsRepository.recordContextOutcome({
      skillId,
      contextKey,
      success: false,
      ts: 2_000,
    });
    harness.proceduralContextStatsRepository.recordContextOutcome({
      skillId: otherSkillId,
      contextKey,
      success: true,
      ts: 3_000,
    });

    expect(success).toMatchObject({
      alpha: 2,
      beta: 1,
      attempts: 1,
      successes: 1,
      failures: 0,
      last_used: 1_000,
      last_successful: 1_000,
    });
    expect(failure).toMatchObject({
      alpha: 2,
      beta: 2,
      attempts: 2,
      successes: 1,
      failures: 1,
      last_used: 2_000,
      last_successful: 1_000,
    });
    expect(harness.proceduralContextStatsRepository.listForSkill(skillId)).toEqual([failure]);
    expect(
      harness.proceduralContextStatsRepository
        .listGlobalUsage(contextKey)
        .map((stats) => stats.skill_id),
    ).toEqual([otherSkillId, skillId]);
  });

  it("derives deterministic context keys from equivalent tag sets", () => {
    const contexts = [
      proceduralContextSchema.parse({
        problem_kind: "code_debugging",
        domain_tags: ["typescript", "debugging"],
        audience_scope: "self",
        context_key: "ignored",
      }),
      proceduralContextSchema.parse({
        problem_kind: "code_debugging",
        domain_tags: ["debugging", "typescript"],
        audience_scope: "self",
        context_key: "ignored",
      }),
      proceduralContextSchema.parse({
        problem_kind: "code_debugging",
        domain_tags: ["TypeScript"],
        audience_scope: "self",
        context_key: "ignored",
      }),
      proceduralContextSchema.parse({
        problem_kind: "code_debugging",
        domain_tags: ["typescript"],
        audience_scope: "self",
        context_key: "ignored",
      }),
    ];

    expect(contexts.map((context) => context.context_key)).toEqual([
      "code_debugging:typescript:self",
      "code_debugging:typescript:self",
      "code_debugging:typescript:self",
      "code_debugging:typescript:self",
    ]);
  });

  it("records global-only outcomes when no procedural context is present", async () => {
    harness = await createOfflineTestHarness();
    const episode = createEpisodeFixture();
    await harness.episodicRepository.insert(episode);
    const skill = await harness.skillRepository.add({
      applies_when: "TypeScript generic inference fails",
      approach: "Reduce the generic to a smaller reproduction.",
      sourceEpisodes: [episode.id],
    });

    const updated = harness.skillRepository.recordOutcome(skill.id, true, episode.id, null);

    expect(updated).toMatchObject({
      attempts: 1,
      successes: 1,
      alpha: 2,
      beta: 1,
    });
    expect(harness.proceduralContextStatsRepository.listForSkill(skill.id)).toEqual([]);
  });

  it("records global and context outcomes together when context is present", async () => {
    harness = await createOfflineTestHarness();
    const episode = createEpisodeFixture();
    await harness.episodicRepository.insert(episode);
    const skill = await harness.skillRepository.add({
      applies_when: "TypeScript generic inference fails",
      approach: "Reduce the generic to a smaller reproduction.",
      sourceEpisodes: [episode.id],
    });
    const proceduralContext = proceduralContextSchema.parse({
      problem_kind: "code_debugging",
      domain_tags: ["TypeScript"],
      audience_scope: "self",
      context_key: "ignored",
    });

    const updated = harness.skillRepository.recordOutcome(
      skill.id,
      false,
      episode.id,
      proceduralContext,
    );

    expect(updated).toMatchObject({
      attempts: 1,
      failures: 1,
      alpha: 1,
      beta: 2,
    });
    expect(
      harness.proceduralContextStatsRepository.getContextStats(
        skill.id,
        proceduralContext.context_key,
      ),
    ).toMatchObject({
      attempts: 1,
      failures: 1,
      alpha: 1,
      beta: 2,
    });
  });

  it("does not leave context stats behind when a contextual global outcome fails", async () => {
    harness = await createOfflineTestHarness();
    const missingSkillId = createSkillId();
    const proceduralContext = proceduralContextSchema.parse({
      problem_kind: "planning",
      domain_tags: ["roadmap"],
      audience_scope: "known_other",
      context_key: "ignored",
    });

    expect(() =>
      harness?.skillRepository.recordOutcome(missingSkillId, true, undefined, proceduralContext),
    ).toThrow(/Unknown skill id/);
    expect(
      harness.proceduralContextStatsRepository.getContextStats(
        missingSkillId,
        proceduralContext.context_key,
      ),
    ).toBeNull();
  });

  it("rolls back global skill counters when contextual stats recording fails", async () => {
    harness = await createOfflineTestHarness();
    const episode = createEpisodeFixture();
    await harness.episodicRepository.insert(episode);
    const skill = await harness.skillRepository.add({
      applies_when: "TypeScript generic inference fails",
      approach: "Reduce the generic to a smaller reproduction.",
      sourceEpisodes: [episode.id],
    });
    const proceduralContext = proceduralContextSchema.parse({
      problem_kind: "code_debugging",
      domain_tags: ["TypeScript"],
      audience_scope: "self",
      context_key: "ignored",
    });

    harness.db.exec(`
      CREATE TRIGGER fail_skill_context_stats_insert
      BEFORE INSERT ON skill_context_stats
      BEGIN
        SELECT RAISE(ABORT, 'injected context stats failure');
      END;
    `);

    expect(() =>
      harness?.skillRepository.recordOutcome(skill.id, true, episode.id, proceduralContext),
    ).toThrow(/injected context stats failure/);
    expect(harness.skillRepository.get(skill.id)).toMatchObject({
      alpha: 1,
      beta: 1,
      attempts: 0,
      successes: 0,
      failures: 0,
    });
    expect(
      harness.proceduralContextStatsRepository.getContextStats(
        skill.id,
        proceduralContext.context_key,
      ),
    ).toBeNull();
  });

  it("stages procedural evidence idempotently for the same pending attempt", async () => {
    harness = await createOfflineTestHarness();
    const pendingAttempt = {
      problem_text: "Fix the flaky Atlas deploy.",
      approach_summary: "Compare the failing deploy log against the last clean release.",
      selected_skill_id: null,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"] as never,
      turn_counter: 7,
      audience_entity_id: null,
    };

    const first = harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: pendingAttempt,
      classification: "success",
      evidenceText: "User confirmed the deploy worked.",
    });
    const second = harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: pendingAttempt,
      classification: "success",
      evidenceText: "User confirmed the deploy worked again.",
    });

    expect(second.id).toBe(first.id);
    expect(harness.proceduralEvidenceRepository.list()).toEqual([
      expect.objectContaining({
        grounded: true,
        skill_actually_applied: true,
      }),
    ]);
  });

  it("stores whether the selected skill was actually applied", async () => {
    harness = await createOfflineTestHarness();
    const pendingAttempt = {
      problem_text: "Fix the flaky Atlas deploy.",
      approach_summary: "Compare the failing deploy log against the last clean release.",
      selected_skill_id: null,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"] as never,
      turn_counter: 7,
      audience_entity_id: null,
    };

    const evidence = harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: pendingAttempt,
      classification: "success",
      evidenceText: "User confirmed a different workaround helped.",
      skillActuallyApplied: false,
    });

    expect(evidence.skill_actually_applied).toBe(false);
    expect(harness.proceduralEvidenceRepository.get(evidence.id)?.skill_actually_applied).toBe(
      false,
    );
  });

  it("persists procedural context on procedural evidence", async () => {
    harness = await createOfflineTestHarness();
    const proceduralContext = proceduralContextSchema.parse({
      problem_kind: "code_debugging",
      domain_tags: ["TypeScript"],
      audience_scope: "self",
      context_key: "ignored",
    });
    const pendingAttempt = {
      problem_text: "Fix the flaky Atlas deploy.",
      approach_summary: "Compare the failing deploy log against the last clean release.",
      selected_skill_id: null,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"] as never,
      turn_counter: 7,
      audience_entity_id: null,
      procedural_context: proceduralContext,
    };

    const evidence = harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: pendingAttempt,
      classification: "success",
      evidenceText: "User confirmed the deploy worked.",
    });

    expect(evidence.procedural_context).toEqual(proceduralContext);
    expect(evidence.pending_attempt_snapshot.procedural_context).toEqual(proceduralContext);
    expect(harness.proceduralEvidenceRepository.get(evidence.id)).toMatchObject({
      procedural_context: proceduralContext,
      pending_attempt_snapshot: expect.objectContaining({
        procedural_context: proceduralContext,
      }),
    });
  });

  it("reads pre-SP1 procedural evidence without procedural context", async () => {
    harness = await createOfflineTestHarness();
    const evidenceId = createProceduralEvidenceId();
    const pendingAttempt = {
      problem_text: "Fix the flaky Atlas deploy.",
      approach_summary: "Compare the failing deploy log against the last clean release.",
      selected_skill_id: null,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"],
      turn_counter: 7,
      audience_entity_id: null,
    };

    harness.db
      .prepare(
        `
          INSERT INTO procedural_evidence (
            id, pending_attempt_snapshot, classification, evidence_text,
            resolved_episode_ids, audience_entity_id, consumed_at, created_at,
            grounded, skill_actually_applied
          ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, 1, 1)
        `,
      )
      .run(
        evidenceId,
        JSON.stringify(pendingAttempt),
        "success",
        "Legacy row without procedural context.",
        "[]",
        1_000,
      );

    const evidence = harness.proceduralEvidenceRepository.get(evidenceId);

    expect(evidence?.procedural_context ?? null).toBeNull();
    expect(evidence?.pending_attempt_snapshot.procedural_context ?? null).toBeNull();
  });

  it("upgrades a grounded unclear evidence row when a later success/failure arrives", async () => {
    // Sprint 55 regression test: with Sprint 53's multi-turn pending
    // attempts, an early "unclear" outcome must not block a later
    // grounded success/failure from being persisted for the synthesizer.
    harness = await createOfflineTestHarness();
    const pendingAttempt = {
      problem_text: "Atlas deploy keeps flaking on the rollback step.",
      approach_summary: "Compare against the last clean release state.",
      selected_skill_id: null,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"] as never,
      turn_counter: 5,
      audience_entity_id: null,
    };
    const firstEpisodeId = createEpisodeId();
    const secondEpisodeId = createEpisodeId();

    const initial = harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: pendingAttempt,
      classification: "unclear",
      evidenceText: "User replied without saying whether it worked.",
      resolvedEpisodeIds: [firstEpisodeId],
    });
    expect(initial.classification).toBe("unclear");

    const upgraded = harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: pendingAttempt,
      classification: "success",
      evidenceText: "User confirmed the rollback comparison fixed it.",
      resolvedEpisodeIds: [secondEpisodeId],
    });

    expect(upgraded.id).toBe(initial.id);
    expect(upgraded.classification).toBe("success");
    expect(upgraded.evidence_text).toBe("User confirmed the rollback comparison fixed it.");
    const rows = harness.proceduralEvidenceRepository.list();
    expect(rows).toHaveLength(1);
    expect(rows[0]?.classification).toBe("success");
    expect(rows[0]?.resolved_episode_ids).toEqual([firstEpisodeId, secondEpisodeId]);
  });

  it("does not downgrade an actionable evidence row to unclear", async () => {
    harness = await createOfflineTestHarness();
    const pendingAttempt = {
      problem_text: "Fix the flaky deploy.",
      approach_summary: "Compare deploy logs.",
      selected_skill_id: null,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"] as never,
      turn_counter: 9,
      audience_entity_id: null,
    };

    const success = harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: pendingAttempt,
      classification: "success",
      evidenceText: "User confirmed the fix.",
    });

    const dedup = harness.proceduralEvidenceRepository.insert({
      pendingAttemptSnapshot: pendingAttempt,
      classification: "unclear",
      evidenceText: "Later turn re-graded as unclear.",
    });

    expect(dedup.id).toBe(success.id);
    expect(dedup.classification).toBe("success");
    expect(dedup.evidence_text).toBe("User confirmed the fix.");
  });

  it("does not select a skill below the configured similarity threshold", async () => {
    harness = await createOfflineTestHarness();
    const episode = createEpisodeFixture();
    await harness.episodicRepository.insert(episode);
    await harness.skillRepository.add({
      applies_when: "Atlas deployment rollback",
      approach: "Compare deploy logs and rollback state.",
      sourceEpisodes: [episode.id],
    });

    const selector = new SkillSelector({
      repository: harness.skillRepository,
      minSimilarity: 0.5,
    });

    await expect(selector.select("planning roadmap review", { k: 5 })).resolves.toBeNull();
  });
});
