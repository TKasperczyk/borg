import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg, FakeEmbeddingClient, type BorgOpenOptions } from "../src/index.js";
import { FakeLLMClient } from "../src/llm/test-support/fake-client.js";
import type { EntityId } from "../src/util/ids.js";
import { createEntityId } from "../src/util/ids.js";
import {
  main,
  parseAudienceScopingCliArgs,
  planAudienceScopingMigration,
  type AudienceScopingMigrationCandidate,
} from "./migrate-audience-scoping.js";

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

function migrationReason(candidate: AudienceScopingMigrationCandidate, toEntityId: EntityId) {
  return `BotArena continuous-room audience migration: ${candidate.sourceAudienceEntityId} -> ${toEntityId}`;
}

describe("audience-scoping migration", () => {
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

  it("requires the named flags, defaults to dry-run, and rejects destination overlap", () => {
    const from = createEntityId();
    const secondFrom = createEntityId();
    const to = createEntityId();

    expect(
      parseAudienceScopingCliArgs([
        "--data-dir",
        "/tmp/example-bank",
        "--from",
        `${from},${secondFrom}`,
        "--to",
        to,
      ]),
    ).toMatchObject({
      help: false,
      apply: false,
      fromEntityIds: [from, secondFrom],
      toEntityId: to,
    });
    expect(() => parseAudienceScopingCliArgs(["--from", from, "--to", to])).toThrow(
      "--data-dir is required",
    );
    expect(() =>
      parseAudienceScopingCliArgs([
        "--data-dir",
        "/tmp/example-bank",
        "--from",
        from,
        "--to",
        from,
      ]),
    ).toThrow(`--from must not contain destination audience entity ${from}`);
  });

  it("plans and applies only active legacy rows, audits them, and is idempotent", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-audience-scoping-"));
    tempDirs.push(tempDir);
    const seed = await openTestBorg({ dataDir: tempDir });
    const legacyOne = seed.entities.resolve("BotArena legacy thread one", { kind: "group" });
    const legacyTwo = seed.entities.resolve("BotArena legacy thread two", { kind: "group" });
    const destination = seed.entities.resolve("BotArena durable room", { kind: "group" });
    const nonGroup = seed.entities.resolve("Not a room", { kind: "person" });
    const firstCommitment = seed.identity.addCommitment({
      type: "rule",
      kind: "audience_rule",
      directiveFamily: "audience_migration_first",
      directive: "Keep the first legacy room informed about ongoing operational changes.",
      priority: 9,
      restrictedAudience: legacyOne,
      provenance: { kind: "manual" },
    });
    const secondCommitment = seed.identity.addCommitment({
      type: "promise",
      directiveFamily: "audience_migration_second",
      directive: "Continue the work agreed in the second legacy room after thread rotation.",
      priority: 7,
      restrictedAudience: legacyTwo,
      provenance: { kind: "manual" },
    });
    const firstGoal = seed.self.goals.add({
      description: "Finish the active work carried over from the first BotArena thread.",
      priority: 8,
      audienceEntityId: legacyOne,
      provenance: { kind: "manual" },
    });
    const secondGoal = seed.self.goals.add({
      description: "Preserve continuity for the active work from the second BotArena thread.",
      priority: 6,
      audienceEntityId: legacyTwo,
      provenance: { kind: "manual" },
    });
    const alreadyAtDestination = seed.identity.addCommitment({
      type: "rule",
      kind: "audience_rule",
      directiveFamily: "audience_migration_destination",
      directive: "This row already belongs to the durable BotArena room.",
      priority: 5,
      restrictedAudience: destination,
      provenance: { kind: "manual" },
    });
    const revokedBeforeMigration = seed.identity.addCommitment({
      type: "promise",
      directiveFamily: "audience_migration_revoked",
      directive: "This retired row must retain its legacy audience.",
      priority: 4,
      restrictedAudience: legacyOne,
      provenance: { kind: "manual" },
    });
    const revokedSnapshot = seed.commitments.revoke(
      revokedBeforeMigration.id,
      "Retired before audience migration",
      { kind: "manual" },
    );
    const doneBeforeMigration = seed.self.goals.add({
      description: "This completed goal must retain its legacy audience.",
      priority: 3,
      audienceEntityId: legacyTwo,
      provenance: { kind: "manual" },
    });
    const doneResult = seed.identity.updateGoal(
      doneBeforeMigration.id,
      { status: "done" },
      { kind: "manual" },
      { throughReview: true, reason: "Completed before audience migration" },
    );

    expect(revokedSnapshot).not.toBeNull();
    expect(doneResult.status).toBe("applied");

    const doneSnapshot = doneResult.status === "applied" ? doneResult.record : doneResult.current;
    await seed.close();

    const input = {
      dataDir: tempDir,
      fromEntityIds: [legacyOne, legacyTwo],
      toEntityId: destination,
    };
    const plan = planAudienceScopingMigration(input);
    const selectedIds = new Set(plan.candidates.map((candidate) => candidate.id));

    expect(plan.candidates).toHaveLength(4);
    expect(selectedIds).toEqual(
      new Set([firstCommitment.id, secondCommitment.id, firstGoal.id, secondGoal.id]),
    );
    expect(selectedIds.has(alreadyAtDestination.id)).toBe(false);
    expect(selectedIds.has(revokedBeforeMigration.id)).toBe(false);
    expect(selectedIds.has(doneBeforeMigration.id)).toBe(false);
    expect(plan.candidates.map((candidate) => candidate.sourceAudienceEntityId)).toEqual([
      legacyOne,
      legacyOne,
      legacyTwo,
      legacyTwo,
    ]);

    expect(() => planAudienceScopingMigration({ ...input, toEntityId: createEntityId() })).toThrow(
      "Destination audience entity does not exist",
    );
    expect(() => planAudienceScopingMigration({ ...input, toEntityId: nonGroup })).toThrow(
      `Destination audience entity ${nonGroup} must have kind "group"`,
    );

    const databasePath = join(tempDir, "borg.db");
    const beforeDryRun = readFileSync(databasePath);
    const dryRunStdout = createOutputBuffer();
    const dryRunStderr = createOutputBuffer();
    const dryRunOpenBorg = vi.fn(openTestBorg);

    await expect(
      main(["--data-dir", tempDir, "--from", `${legacyOne},${legacyTwo}`, "--to", destination], {
        openBorg: dryRunOpenBorg,
        stdout: dryRunStdout.output,
        stderr: dryRunStderr.output,
      }),
    ).resolves.toBe(0);
    expect(dryRunOpenBorg).not.toHaveBeenCalled();
    expect(readFileSync(databasePath)).toEqual(beforeDryRun);
    expect(dryRunStdout.read()).toContain("mode=dry-run");
    expect(dryRunStdout.read()).toContain(`source_audience=${legacyOne}`);
    expect(dryRunStdout.read()).toContain(`source_audience=${legacyTwo}`);
    expect(dryRunStdout.read()).toContain("kind       | id");
    expect(dryRunStdout.read()).toContain("total commitments=2 goals=2 rows=4");

    const applyStdout = createOutputBuffer();
    const applyStderr = createOutputBuffer();

    await expect(
      main(
        [
          "--data-dir",
          tempDir,
          "--from",
          `${legacyOne},${legacyTwo}`,
          "--to",
          destination,
          "--apply",
        ],
        {
          openBorg: openTestBorg,
          stdout: applyStdout.output,
          stderr: applyStderr.output,
        },
      ),
    ).resolves.toBe(0);
    expect(applyStdout.read()).toContain("mode=apply");
    expect(applyStdout.read()).toContain("applied_total commitments=2 goals=2 rows=4");
    expect(applyStdout.read()).toContain("failures=0");

    const inspection = await openTestBorg({ dataDir: tempDir });

    expect(inspection.commitments.get(firstCommitment.id)?.restricted_audience).toBe(destination);
    expect(inspection.commitments.get(secondCommitment.id)?.restricted_audience).toBe(destination);
    expect(inspection.self.goals.get(firstGoal.id)?.audience_entity_id).toBe(destination);
    expect(inspection.self.goals.get(secondGoal.id)?.audience_entity_id).toBe(destination);
    expect(inspection.commitments.get(alreadyAtDestination.id)).toEqual(alreadyAtDestination);
    expect(inspection.commitments.get(revokedBeforeMigration.id)).toEqual(revokedSnapshot);
    expect(inspection.self.goals.get(doneBeforeMigration.id)).toEqual(doneSnapshot);

    const identityEvents = inspection.identity.listEvents({ limit: 100 });

    for (const candidate of plan.candidates) {
      const matchingEvents = identityEvents.filter(
        (event) =>
          event.record_id === candidate.id &&
          event.reason === migrationReason(candidate, destination),
      );

      expect(matchingEvents).toHaveLength(1);
      expect(matchingEvents[0]).toMatchObject({
        record_type: candidate.kind,
        action: "update",
        provenance: { kind: "manual" },
        overwrite_without_review: false,
      });
      expect(matchingEvents[0]?.new_value).toMatchObject(
        candidate.kind === "commitment"
          ? { restricted_audience: destination }
          : { audience_entity_id: destination },
      );
    }

    await inspection.close();

    const secondPlan = planAudienceScopingMigration(input);
    expect(secondPlan.candidates).toEqual([]);

    const secondApplyOpenBorg = vi.fn(openTestBorg);
    const secondApplyStdout = createOutputBuffer();

    await expect(
      main(
        [
          "--data-dir",
          tempDir,
          "--from",
          `${legacyOne},${legacyTwo}`,
          "--to",
          destination,
          "--apply",
        ],
        {
          openBorg: secondApplyOpenBorg,
          stdout: secondApplyStdout.output,
          stderr: createOutputBuffer().output,
        },
      ),
    ).resolves.toBe(0);
    expect(secondApplyOpenBorg).not.toHaveBeenCalled();
    expect(secondApplyStdout.read()).toContain("total commitments=0 goals=0 rows=0");
    expect(secondApplyStdout.read()).toContain("applied_total commitments=0 goals=0 rows=0");
    // Seeds a real data dir and runs the migration twice; ~9s alone, but past the
    // 15s default when the full suite is running it under parallel load. Same
    // allowance the sibling scripts/repair-*.test.ts migrations already take.
  }, 30_000);
});
