import { afterEach, describe, expect, it } from "vitest";

import { createEpisodeFixture, createOfflineTestHarness } from "../../offline/test-support.js";
import { createSkillId } from "../../util/ids.js";

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
});
