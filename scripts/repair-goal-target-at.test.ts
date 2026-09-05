import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg, FakeEmbeddingClient, type BorgOpenOptions } from "../src/index.js";
import { FakeLLMClient } from "../src/llm/test-support/fake-client.js";
import { createGoalId } from "../src/util/ids.js";
import {
  GOAL_TARGET_AT_REPAIR_REASON,
  main,
  parseGoalTargetAtRepairCliArgs,
  planGoalTargetAtRepair,
} from "./repair-goal-target-at.js";

const EMBEDDING_DIMS = 4;
const GUESSED_TARGET_AT = new Date("2026-08-18T10:46:40.000Z").getTime();

function createOutputBuffer(): { output: { write(chunk: string): true }; read(): string } {
  let value = "";

  return {
    output: {
      write(chunk: string) {
        value += chunk;
        return true;
      },
    },
    read: () => value,
  };
}

async function openTestBorg(options: BorgOpenOptions): Promise<Borg> {
  return Borg.open({
    ...options,
    embeddingDimensions: EMBEDDING_DIMS,
    embeddingClient: new FakeEmbeddingClient(EMBEDDING_DIMS),
    llmClient: new FakeLLMClient(),
    liveExtraction: false,
  });
}

describe("goal target_at repair", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      const tempDir = tempDirs.pop();

      if (tempDir !== undefined) {
        rmSync(tempDir, { recursive: true, force: true });
      }
    }
  });

  it("requires validated named flags and defaults to dry-run", () => {
    const first = createGoalId();
    const second = createGoalId();

    expect(
      parseGoalTargetAtRepairCliArgs([
        "--data-dir",
        "/tmp/example-bank",
        "--goal",
        `${first},${second},${first}`,
      ]),
    ).toMatchObject({
      help: false,
      apply: false,
      goalIds: [first, second],
    });
    expect(() => parseGoalTargetAtRepairCliArgs(["--goal", first])).toThrow(
      "--data-dir is required",
    );
    expect(() => parseGoalTargetAtRepairCliArgs(["--data-dir", "/tmp/example-bank"])).toThrow(
      "--goal is required",
    );
    expect(() =>
      parseGoalTargetAtRepairCliArgs([
        "--data-dir",
        "/tmp/example-bank",
        "--goal",
        "not-a-goal-id",
      ]),
    ).toThrow("Invalid goal id");
  });

  // Opens a real Borg instance and runs migrations twice; the default 15s test
  // timeout is not enough once the full suite is running these in parallel.
  it("refuses missing goals, clears selected deadlines through identity, and is idempotent", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-goal-target-at-repair-"));
    tempDirs.push(tempDir);
    const seed = await openTestBorg({ dataDir: tempDir });
    const first = seed.self.goals.add({
      description:
        "Clear the first guessed harness deadline while preserving every other field on this goal.",
      priority: 9,
      createdAt: 1_000,
      targetAt: GUESSED_TARGET_AT,
      provenance: { kind: "manual" },
    });
    const second = seed.self.goals.add({
      description: "Clear the second guessed deadline from the same harness batch.",
      priority: 7,
      createdAt: 2_000,
      targetAt: GUESSED_TARGET_AT,
      provenance: { kind: "manual" },
    });
    const alreadyNull = seed.self.goals.add({
      description: "This goal has no deadline and must remain an audited no-op.",
      priority: 5,
      createdAt: 3_000,
      provenance: { kind: "manual" },
    });
    const missing = createGoalId();
    await seed.close();

    const requestedIds = [first.id, second.id, alreadyNull.id];
    const plan = planGoalTargetAtRepair({ dataDir: tempDir, goalIds: requestedIds });

    expect(plan.candidates.map((candidate) => candidate.id)).toEqual([first.id, second.id]);
    expect(plan.candidates.map((candidate) => candidate.patch)).toEqual([
      { target_at: null },
      { target_at: null },
    ]);
    expect(plan.skipped.map((skip) => skip.id)).toEqual([alreadyNull.id]);
    expect(plan.refusals).toEqual([]);

    const databasePath = join(tempDir, "borg.db");
    const beforeDryRun = readFileSync(databasePath);
    const dryRunStdout = createOutputBuffer();
    const dryRunStderr = createOutputBuffer();
    const dryRunOpenBorg = vi.fn(openTestBorg);

    await expect(
      main(["--data-dir", tempDir, "--goal", [...requestedIds, missing].join(",")], {
        openBorg: dryRunOpenBorg,
        stdout: dryRunStdout.output,
        stderr: dryRunStderr.output,
      }),
    ).resolves.toBe(1);
    expect(dryRunOpenBorg).not.toHaveBeenCalled();
    expect(readFileSync(databasePath)).toEqual(beforeDryRun);
    expect(dryRunStdout.read()).toContain("mode=dry-run");
    expect(dryRunStdout.read()).toContain("action | id");
    expect(dryRunStdout.read()).toContain("status");
    expect(dryRunStdout.read()).toContain("priority");
    expect(dryRunStdout.read()).toContain("created_at");
    expect(dryRunStdout.read()).toContain("current target_at");
    expect(dryRunStdout.read()).toContain(String(GUESSED_TARGET_AT));
    expect(dryRunStdout.read()).toContain(first.description.slice(0, 79));
    expect(dryRunStdout.read()).toContain(`skipped id=${alreadyNull.id}`);
    expect(dryRunStdout.read()).toContain(`refused id=${missing}`);
    expect(dryRunStdout.read()).toContain("selected=2 skipped=1 refused=1");

    const refusedApplyOpenBorg = vi.fn(openTestBorg);

    await expect(
      main(["--data-dir", tempDir, "--goal", [...requestedIds, missing].join(","), "--apply"], {
        openBorg: refusedApplyOpenBorg,
        stdout: createOutputBuffer().output,
        stderr: createOutputBuffer().output,
      }),
    ).resolves.toBe(1);
    expect(refusedApplyOpenBorg).not.toHaveBeenCalled();

    const applyStdout = createOutputBuffer();

    await expect(
      main(["--data-dir", tempDir, "--goal", requestedIds.join(","), "--apply"], {
        openBorg: openTestBorg,
        stdout: applyStdout.output,
        stderr: createOutputBuffer().output,
      }),
    ).resolves.toBe(0);
    expect(applyStdout.read()).toContain("mode=apply");
    expect(applyStdout.read()).toContain("selected=2 skipped=1 refused=0");
    expect(applyStdout.read()).toContain("applied_total=2");
    expect(applyStdout.read()).toContain("failures=0");

    const inspection = await openTestBorg({ dataDir: tempDir });

    expect(inspection.self.goals.get(first.id)?.target_at).toBeNull();
    expect(inspection.self.goals.get(second.id)?.target_at).toBeNull();
    expect(inspection.self.goals.get(alreadyNull.id)?.target_at).toBeNull();

    const identityEvents = inspection.identity.listEvents({ limit: 100 });

    for (const goal of [first, second]) {
      const matchingEvents = identityEvents.filter(
        (event) => event.record_id === goal.id && event.reason === GOAL_TARGET_AT_REPAIR_REASON,
      );

      expect(matchingEvents).toHaveLength(1);
      expect(matchingEvents[0]).toMatchObject({
        record_type: "goal",
        action: "update",
        provenance: { kind: "manual" },
        overwrite_without_review: false,
        new_value: { id: goal.id, target_at: null },
      });
    }

    expect(
      identityEvents.filter(
        (event) =>
          event.record_id === alreadyNull.id && event.reason === GOAL_TARGET_AT_REPAIR_REASON,
      ),
    ).toEqual([]);
    await inspection.close();

    const secondPlan = planGoalTargetAtRepair({ dataDir: tempDir, goalIds: requestedIds });
    expect(secondPlan.candidates).toEqual([]);
    expect(secondPlan.skipped.map((skip) => skip.id)).toEqual(requestedIds);

    const secondApplyOpenBorg = vi.fn(openTestBorg);
    const secondApplyStdout = createOutputBuffer();

    await expect(
      main(["--data-dir", tempDir, "--goal", requestedIds.join(","), "--apply"], {
        openBorg: secondApplyOpenBorg,
        stdout: secondApplyStdout.output,
        stderr: createOutputBuffer().output,
      }),
    ).resolves.toBe(0);
    expect(secondApplyOpenBorg).not.toHaveBeenCalled();
    expect(secondApplyStdout.read()).toContain("selected=0 skipped=3 refused=0");
    expect(secondApplyStdout.read()).toContain("applied_total=0");
  }, 30_000);
});
