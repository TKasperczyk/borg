import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg, FakeEmbeddingClient, type BorgOpenOptions } from "../src/index.js";
import { FakeLLMClient } from "../src/llm/test-support/fake-client.js";
import {
  GOAL_SPEAKER_OWNER_REPAIR_REASON,
  main,
  parseGoalSpeakerOwnerRepairCliArgs,
  planGoalSpeakerOwnerRepair,
} from "./repair-goal-speaker-owner.js";

const EMBEDDING_DIMS = 4;

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

describe("goal speaker-owner repair", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop()!, { recursive: true, force: true });
    }
  });

  it("defaults to dry-run and accepts only named flags", () => {
    expect(parseGoalSpeakerOwnerRepairCliArgs(["--data-dir", "/tmp/example-bank"])).toMatchObject({
      help: false,
      apply: false,
    });
    expect(
      parseGoalSpeakerOwnerRepairCliArgs(["--data-dir", "/tmp/example-bank", "--apply"]),
    ).toMatchObject({ help: false, apply: true });
    expect(() => parseGoalSpeakerOwnerRepairCliArgs([])).toThrow("--data-dir is required");
    expect(() =>
      parseGoalSpeakerOwnerRepairCliArgs(["--data-dir", "/tmp/example-bank", "--goal", "x"]),
    ).toThrow("Unknown argument");
  });

  it("clears only matching creation owners without later owner assignments", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-goal-speaker-owner-repair-"));
    tempDirs.push(tempDir);
    const databasePath = join(tempDir, "borg.db");
    const seed = await openTestBorg({ dataDir: tempDir });
    const self = seed.entities.ensureSelf("Borg");
    const legacySpeaker = seed.entities.resolve("Legacy speaker", {
      kind: "person",
      provenance: "transport_sender",
    });
    const intentionalOwner = seed.entities.resolve("Intentional owner", {
      kind: "person",
      provenance: "user_declared",
    });
    const unauditedOwner = seed.entities.resolve("Unaudited owner", {
      kind: "person",
      provenance: "user_declared",
    });
    const promotionWithLegacyOwner = seed.self.goals.add({
      description: "Carry the extracted responsibility to completion.",
      priority: 7,
      ownerEntityId: legacySpeaker,
      provenance: { kind: "online", process: "goal-promotion-extractor" },
    });
    const promotionWithSelfOwner = seed.self.goals.add({
      description: "Preserve an explicitly self-owned promotion record.",
      priority: 6,
      ownerEntityId: self.id,
      provenance: { kind: "online", process: "goal-promotion-extractor" },
    });
    const promotionWithNullOwner = seed.self.goals.add({
      description: "Leave the already repaired promotion record unchanged.",
      priority: 5,
      ownerEntityId: null,
      provenance: { kind: "online", process: "goal-promotion-extractor" },
    });
    const manualWithOtherOwner = seed.self.goals.add({
      description: "Do not alter ownership from another creation path.",
      priority: 4,
      ownerEntityId: legacySpeaker,
      provenance: { kind: "manual" },
    });
    const promotionWithLaterOwnerChange = seed.self.goals.add({
      description: "Preserve a later intentional owner assignment.",
      priority: 4,
      ownerEntityId: legacySpeaker,
      provenance: { kind: "online", process: "goal-promotion-extractor" },
    });
    const laterOwnerUpdate = seed.identity.updateGoal(
      promotionWithLaterOwnerChange.id,
      { owner_entity_id: intentionalOwner },
      { kind: "manual" },
      { throughReview: true, reason: "Intentional owner assignment after extraction" },
    );
    expect(laterOwnerUpdate.status).toBe("applied");
    const promotionWithCurrentOwnerMismatch = seed.self.goals.add({
      description: "Preserve a current owner that no identity event explains.",
      priority: 3,
      ownerEntityId: legacySpeaker,
      provenance: { kind: "online", process: "goal-promotion-extractor" },
    });
    await seed.close();

    const raw = new DatabaseSync(databasePath);
    try {
      raw
        .prepare(
          `
            UPDATE goals
            SET owner_entity_id = ?, record_version = record_version + 1
            WHERE id = ?
          `,
        )
        .run(unauditedOwner, promotionWithCurrentOwnerMismatch.id);
    } finally {
      raw.close();
    }

    const plan = planGoalSpeakerOwnerRepair({ dataDir: tempDir });
    expect(plan.selfEntityId).toBe(self.id);
    expect(plan.candidates.map((candidate) => candidate.id)).toEqual([promotionWithLegacyOwner.id]);
    expect(plan.skipped).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: promotionWithLaterOwnerChange.id,
          reason: "a later identity event changed owner_entity_id",
        }),
        expect.objectContaining({
          id: promotionWithCurrentOwnerMismatch.id,
          reason: "current owner_entity_id differs from creation owner_entity_id",
        }),
      ]),
    );
    expect(plan.counts).toEqual({
      total: 6,
      selected: 1,
      creationEventMissing: 0,
      otherCreationPath: 1,
      creationOwnerMissing: 0,
      laterOwnerChange: 1,
      currentOwnerMismatch: 1,
      promotionOwnerNull: 1,
      promotionOwnerSelf: 1,
    });

    const beforeDryRun = readFileSync(databasePath);
    const dryRunOpenBorg = vi.fn(openTestBorg);
    const dryRunStdout = createOutputBuffer();
    await expect(
      main(["--data-dir", tempDir], {
        openBorg: dryRunOpenBorg,
        stdout: dryRunStdout.output,
        stderr: createOutputBuffer().output,
      }),
    ).resolves.toBe(0);
    expect(dryRunOpenBorg).not.toHaveBeenCalled();
    expect(readFileSync(databasePath)).toEqual(beforeDryRun);
    expect(dryRunStdout.read()).toContain("mode=dry-run");
    expect(dryRunStdout.read()).toContain(
      `skipped id=${promotionWithLaterOwnerChange.id} reason="a later identity event changed owner_entity_id"`,
    );
    expect(dryRunStdout.read()).toContain(
      `skipped id=${promotionWithCurrentOwnerMismatch.id} reason="current owner_entity_id differs from creation owner_entity_id"`,
    );
    expect(dryRunStdout.read()).toContain(
      "total=6 selected=1 creation_event_missing=0 other_creation_path=1 creation_owner_missing=0 later_owner_change=1 current_owner_mismatch=1 promotion_owner_null=1 promotion_owner_self=1",
    );

    const applyStdout = createOutputBuffer();
    await expect(
      main(["--data-dir", tempDir, "--apply"], {
        openBorg: openTestBorg,
        stdout: applyStdout.output,
        stderr: createOutputBuffer().output,
      }),
    ).resolves.toBe(0);
    expect(applyStdout.read()).toContain("applied_total=1");
    expect(applyStdout.read()).toContain("failures=0");

    const inspection = await openTestBorg({ dataDir: tempDir });
    expect(inspection.self.goals.get(promotionWithLegacyOwner.id)?.owner_entity_id).toBeNull();
    expect(
      inspection.self.goals.get(promotionWithLegacyOwner.id)?.counterparty_entity_id,
    ).toBeNull();
    expect(inspection.self.goals.get(promotionWithSelfOwner.id)?.owner_entity_id).toBe(self.id);
    expect(inspection.self.goals.get(promotionWithNullOwner.id)?.owner_entity_id).toBeNull();
    expect(inspection.self.goals.get(manualWithOtherOwner.id)?.owner_entity_id).toBe(legacySpeaker);
    expect(inspection.self.goals.get(promotionWithLaterOwnerChange.id)?.owner_entity_id).toBe(
      intentionalOwner,
    );
    expect(
      inspection.self.goals.get(promotionWithLaterOwnerChange.id)?.counterparty_entity_id,
    ).toBeNull();
    expect(inspection.self.goals.get(promotionWithCurrentOwnerMismatch.id)?.owner_entity_id).toBe(
      unauditedOwner,
    );
    expect(
      inspection.self.goals.get(promotionWithCurrentOwnerMismatch.id)?.counterparty_entity_id,
    ).toBeNull();
    expect(
      inspection.identity
        .listEvents({ limit: 100 })
        .filter(
          (event) =>
            event.record_id === promotionWithLegacyOwner.id &&
            event.reason === GOAL_SPEAKER_OWNER_REPAIR_REASON,
        ),
    ).toEqual([
      expect.objectContaining({
        record_type: "goal",
        action: "update",
        provenance: { kind: "manual" },
        new_value: expect.objectContaining({ owner_entity_id: null }),
      }),
    ]);
    await inspection.close();

    expect(planGoalSpeakerOwnerRepair({ dataDir: tempDir }).counts.selected).toBe(0);
  });
});
