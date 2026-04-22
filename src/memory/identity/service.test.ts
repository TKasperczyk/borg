import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { commitmentMigrations, CommitmentRepository } from "../commitments/index.js";
import { selfMigrations, TraitsRepository, ValuesRepository } from "../self/index.js";

import { IdentityEventRepository } from "./repository.js";
import { IdentityService } from "./service.js";
import { identityMigrations } from "./migrations.js";

describe("identity service", () => {
  it("blocks manual overwrites for established episode-backed values and audits offline bypasses", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations, ...commitmentMigrations, ...identityMigrations],
    });
    const clock = new FixedClock(1_000);
    const identityEvents = new IdentityEventRepository({
      db,
      clock,
    });
    const valuesRepository = new ValuesRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });
    const traitsRepository = new TraitsRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });
    const commitmentRepository = new CommitmentRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });
    const identity = new IdentityService({
      valuesRepository,
      traitsRepository,
      commitmentRepository,
      identityEventRepository: identityEvents,
    });

    try {
      const value = valuesRepository.add({
        label: "accuracy",
        description: "Prefer grounded claims.",
        priority: 8,
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_aaaaaaaaaaaaaaaa" as const],
        },
      });
      valuesRepository.reinforce(value.id, {
        kind: "episodes",
        episode_ids: ["ep_bbbbbbbbbbbbbbbb" as const],
      });
      valuesRepository.reinforce(value.id, {
        kind: "episodes",
        episode_ids: ["ep_cccccccccccccccc" as const],
      });

      expect(valuesRepository.get(value.id)?.state).toBe("established");
      expect(
        identity.updateValue(
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

      const bypass = identity.updateValue(
        value.id,
        {
          description: "Prefer careful claims under revision.",
        },
        {
          kind: "offline",
          process: "reflector",
        },
      );

      expect(bypass).toEqual(
        expect.objectContaining({
          status: "applied",
          overwriteWithoutReview: true,
        }),
      );
      expect(
        identityEvents.list({
          recordType: "value",
          recordId: value.id,
        }),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "promote",
          }),
          expect.objectContaining({
            action: "update",
            overwrite_without_review: true,
          }),
        ]),
      );
    } finally {
      db.close();
    }
  });

  it("blocks manual overwrites for traits promoted from offline seed to episode-backed evidence", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations, ...commitmentMigrations, ...identityMigrations],
    });
    const clock = new ManualClock(1_000);
    const identityEvents = new IdentityEventRepository({
      db,
      clock,
    });
    const valuesRepository = new ValuesRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });
    const traitsRepository = new TraitsRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });
    const commitmentRepository = new CommitmentRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });
    const identity = new IdentityService({
      valuesRepository,
      traitsRepository,
      commitmentRepository,
      identityEventRepository: identityEvents,
    });
    const episodeIds = [
      "ep_aaaaaaaaaaaaaaaa",
      "ep_bbbbbbbbbbbbbbbb",
      "ep_cccccccccccccccc",
      "ep_dddddddddddddddd",
      "ep_eeeeeeeeeeeeeeee",
    ] as const;

    try {
      traitsRepository.reinforce({
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
        traitsRepository.reinforce({
          label: "engaged",
          delta: 0.05,
          provenance: {
            kind: "episodes",
            episode_ids: [episodeId],
          },
          timestamp: clock.now(),
        });
      }

      const trait = traitsRepository.list()[0];

      expect(trait).toEqual(
        expect.objectContaining({
          state: "established",
          provenance: {
            kind: "episodes",
            episode_ids: [
              "ep_eeeeeeeeeeeeeeee",
              "ep_dddddddddddddddd",
              "ep_cccccccccccccccc",
            ],
          },
        }),
      );
      expect(
        identity.updateTrait(
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
      db.close();
    }
  });
});
