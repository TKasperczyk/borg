import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { ProvenanceError } from "../../util/errors.js";
import { commitmentMigrations } from "./migrations.js";
import { CommitmentRepository, EntityRepository } from "./repository.js";

describe("commitment repository", () => {
  const manualProvenance = { kind: "manual" } as const;

  it("filters by audience and supports revoke/supersede", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const clock = new FixedClock(1_000);
    const entities = new EntityRepository({
      db,
      clock,
    });
    const commitments = new CommitmentRepository({
      db,
      clock,
    });
    const audience = entities.resolve("Sam");
    const about = entities.resolve("Atlas");
    const first = commitments.add({
      type: "boundary",
      directive: "Do not discuss Atlas outages with Sam",
      priority: 10,
      restrictedAudience: audience,
      aboutEntity: about,
      provenance: manualProvenance,
    });
    const second = commitments.add({
      type: "promise",
      directive: "Follow up tomorrow",
      priority: 5,
      provenance: manualProvenance,
    });
    const replacement = commitments.add({
      type: "promise",
      directive: "Follow up next week",
      priority: 6,
      provenance: manualProvenance,
    });

    expect(
      commitments.getApplicable({
        audience,
        aboutEntity: about,
        nowMs: 1_000,
      }),
    ).toEqual(expect.arrayContaining([first, second, replacement]));
    expect(
      commitments.getApplicable({
        audience: entities.resolve("Elsewhere"),
        aboutEntity: about,
        nowMs: 1_000,
      }),
    ).toEqual(expect.arrayContaining([second, replacement]));

    expect(commitments.revoke(first.id)?.revoked_at).toBe(1_000);
    expect(commitments.supersede(second.id, replacement.id)?.superseded_by).toBe(replacement.id);
    expect(
      commitments.list({
        activeOnly: true,
      }),
    ).toEqual([replacement]);

    db.close();
  });

  it("rejects provenance-less commitment creation", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const commitments = new CommitmentRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      expect(() =>
        commitments.add({
          type: "rule",
          directive: "Keep sources attached",
          priority: 1,
          provenance: undefined as never,
        }),
      ).toThrow(ProvenanceError);
    } finally {
      db.close();
    }
  });
});
