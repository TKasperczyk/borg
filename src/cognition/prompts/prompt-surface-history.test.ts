import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import {
  PROMPT_SURFACE_BLOCKS,
  PROMPT_SURFACES,
  type PromptSurfaceBlock,
  type PromptSurfacePlacement,
} from "./prompt-surface-registry.js";
import { promptSurfaceHistoryMigrations } from "./prompt-surface-history-migrations.js";
import {
  PromptSurfaceHistoryRepository,
  buildPromptSurfaceProjection,
  hashPromptSurfaceProjection,
} from "./prompt-surface-history.js";

function testBlock(
  id: string,
  surfaces: readonly PromptSurfacePlacement[],
  overrides: Partial<Pick<PromptSurfaceBlock, "owner" | "purpose" | "renderCondition">> = {},
): PromptSurfaceBlock {
  return {
    id,
    owner: overrides.owner ?? "test.owner",
    purpose: overrides.purpose ?? "test purpose",
    renderCondition: overrides.renderCondition ?? "always",
    source: { file: "test.ts" },
    approxLines: null,
    approxChars: null,
    surfaces,
    render: () => null,
  };
}

function openRepo(blocks: readonly PromptSurfaceBlock[], clockMs = 1_000) {
  const clock = new ManualClock(clockMs);
  const db = openDatabase(":memory:", { migrations: promptSurfaceHistoryMigrations });
  const repo = new PromptSurfaceHistoryRepository({ db, clock, blocks });

  return { db, repo, clock };
}

describe("PromptSurfaceHistoryRepository", () => {
  it("keeps the canonical registry block ids and placement keys unique", () => {
    const seenBlockIds = new Set<string>();
    const duplicateBlockIds = new Set<string>();
    const seenPlacementKeys = new Set<string>();
    const duplicatePlacementKeys = new Set<string>();

    for (const block of PROMPT_SURFACE_BLOCKS) {
      if (seenBlockIds.has(block.id)) {
        duplicateBlockIds.add(block.id);
      }
      seenBlockIds.add(block.id);

      for (const placement of block.surfaces) {
        const key = `${block.id}:${placement.surface}:${placement.order}`;

        if (seenPlacementKeys.has(key)) {
          duplicatePlacementKeys.add(key);
        }
        seenPlacementKeys.add(key);
      }
    }

    expect([...duplicateBlockIds]).toEqual([]);
    expect([...duplicatePlacementKeys]).toEqual([]);
  });

  it("records the first observation as a baseline, not an everything-new diff", () => {
    const { repo } = openRepo([
      testBlock("alpha", [{ surface: PROMPT_SURFACES.baseDirect, order: 10 }]),
    ]);

    const observation = repo.observeCurrent();

    expect(observation.inserted).toBe(true);
    expect(observation.change).toEqual({
      observed_at: 1_000,
      from_hash: null,
      to_hash: observation.snapshot.hash,
      added_block_ids: [],
      removed_block_ids: [],
      added_surface_placements: [],
      removed_surface_placements: [],
    });
    expect(repo.countSnapshots()).toBe(1);
    expect(repo.countChanges()).toBe(1);
  });

  it("does not duplicate rows when the same hash is observed again", () => {
    const blocks = [testBlock("alpha", [{ surface: PROMPT_SURFACES.baseDirect, order: 10 }])];
    const { db, repo, clock } = openRepo(blocks);
    const first = repo.observeCurrent();

    clock.set(5_000);
    const restarted = new PromptSurfaceHistoryRepository({ db, clock, blocks });
    const second = restarted.observeCurrent();

    expect(second).toEqual({
      snapshot: first.snapshot,
      change: null,
      inserted: false,
    });
    expect(restarted.countSnapshots()).toBe(1);
    expect(restarted.countChanges()).toBe(1);
  });

  it("uses insertion order, not timestamps, when diffing against the latest prior snapshot", () => {
    const firstBlocks = [testBlock("first", [{ surface: PROMPT_SURFACES.baseDirect, order: 10 }])];
    const firstHash = hashPromptSurfaceProjection(buildPromptSurfaceProjection(firstBlocks));
    let secondBlocks: PromptSurfaceBlock[] | null = null;

    for (let index = 0; index < 200; index += 1) {
      const candidate = [
        testBlock(`second_${index}`, [{ surface: PROMPT_SURFACES.baseDirect, order: 10 }]),
      ];
      const candidateHash = hashPromptSurfaceProjection(buildPromptSurfaceProjection(candidate));

      if (candidateHash < firstHash) {
        secondBlocks = candidate;
        break;
      }
    }

    if (secondBlocks === null) {
      throw new Error("Could not find a lower lexical second hash fixture");
    }

    const thirdBlocks = [testBlock("third", [{ surface: PROMPT_SURFACES.baseDirect, order: 10 }])];
    const { db, repo, clock } = openRepo(firstBlocks, 1_000);

    const first = repo.observeCurrent();
    clock.set(1_000);
    const secondRepo = new PromptSurfaceHistoryRepository({ db, clock, blocks: secondBlocks });
    const second = secondRepo.observeCurrent();
    clock.set(1_000);
    const thirdRepo = new PromptSurfaceHistoryRepository({ db, clock, blocks: thirdBlocks });
    const third = thirdRepo.observeCurrent();

    expect(first.snapshot.observed_at).toBe(1_000);
    expect(second.snapshot.observed_at).toBe(1_000);
    expect(third.snapshot.observed_at).toBe(1_000);
    expect(second.snapshot.hash < first.snapshot.hash).toBe(true);
    expect(third.change?.from_hash).toBe(second.snapshot.hash);
  });

  it("diffs block adds/removes, order changes, and surface placement changes", () => {
    const baselineBlocks = [
      testBlock("alpha", [
        { surface: PROMPT_SURFACES.baseDirect, order: 10 },
        { surface: PROMPT_SURFACES.cacheableDynamic, order: 10 },
      ]),
      testBlock("beta", [{ surface: PROMPT_SURFACES.baseDirect, order: 20 }]),
      testBlock("gamma", [{ surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 10 }]),
    ];
    const nextBlocks = [
      testBlock("alpha", [
        { surface: PROMPT_SURFACES.baseDirect, order: 15 },
        { surface: PROMPT_SURFACES.cacheableDynamic, order: 10 },
      ]),
      testBlock("gamma", [{ surface: PROMPT_SURFACES.finalizerStaticSystem, order: 10 }]),
      testBlock("delta", [{ surface: PROMPT_SURFACES.s2PlannerSystem, order: 40 }]),
    ];
    const { db, repo, clock } = openRepo(baselineBlocks);

    const baseline = repo.observeCurrent();
    clock.set(2_000);
    const nextRepo = new PromptSurfaceHistoryRepository({ db, clock, blocks: nextBlocks });
    const next = nextRepo.observeCurrent();

    expect(next.inserted).toBe(true);
    expect(next.change?.from_hash).toBe(baseline.snapshot.hash);
    expect(next.change?.to_hash).toBe(next.snapshot.hash);
    expect(next.change?.added_block_ids).toEqual(["delta"]);
    expect(next.change?.removed_block_ids).toEqual(["beta"]);
    expect(next.change?.added_surface_placements).toEqual([
      { block_id: "alpha", surface: PROMPT_SURFACES.baseDirect, order: 15 },
      { block_id: "gamma", surface: PROMPT_SURFACES.finalizerStaticSystem, order: 10 },
      { block_id: "delta", surface: PROMPT_SURFACES.s2PlannerSystem, order: 40 },
    ]);
    expect(next.change?.removed_surface_placements).toEqual([
      { block_id: "alpha", surface: PROMPT_SURFACES.baseDirect, order: 10 },
      { block_id: "beta", surface: PROMPT_SURFACES.baseDirect, order: 20 },
      { block_id: "gamma", surface: PROMPT_SURFACES.finalizerDynamicSystem, order: 10 },
    ]);
    expect(nextRepo.countSnapshots()).toBe(2);
    expect(nextRepo.countChanges()).toBe(2);
  });

  it("excludes authored prose from the structural hash", () => {
    const structuralBlock = testBlock("alpha", [
      { surface: PROMPT_SURFACES.baseDirect, order: 10 },
    ]);
    const editoriallyChangedBlock = testBlock(
      "alpha",
      [{ surface: PROMPT_SURFACES.baseDirect, order: 10 }],
      {
        owner: "different.owner",
        purpose: "Different editorial purpose text.",
        renderCondition: "Different condition prose.",
      },
    );

    expect(hashPromptSurfaceProjection(buildPromptSurfaceProjection([structuralBlock]))).toBe(
      hashPromptSurfaceProjection(buildPromptSurfaceProjection([editoriallyChangedBlock])),
    );
  });

  it("returns an empty list for an unknown since-version cursor", () => {
    const { repo } = openRepo([
      testBlock("alpha", [{ surface: PROMPT_SURFACES.baseDirect, order: 10 }]),
    ]);

    repo.observeCurrent();

    expect(repo.listChanges({ sinceVersion: "0".repeat(64) })).toEqual([]);
  });

  it("can report the current structural surface even when it was not observed at boot", () => {
    const blocks = [testBlock("alpha", [{ surface: PROMPT_SURFACES.baseDirect, order: 10 }])];
    const { repo } = openRepo(blocks);

    expect(repo.current()).toEqual({
      hash: hashPromptSurfaceProjection(buildPromptSurfaceProjection(blocks)),
      observed_at: null,
      block_ids: ["alpha"],
      surface_placements: [{ block_id: "alpha", surface: PROMPT_SURFACES.baseDirect, order: 10 }],
    });
  });
});
