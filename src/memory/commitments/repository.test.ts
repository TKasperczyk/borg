import { describe, expect, it } from "vitest";

import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { ProvenanceError } from "../../util/errors.js";
import { createStreamEntryId } from "../../util/ids.js";
import { identityMigrations, IdentityEventRepository } from "../identity/index.js";
import { commitmentMigrations } from "./migrations.js";
import { CommitmentRepository, EntityRepository } from "./repository.js";

describe("commitment repository", () => {
  const manualProvenance = { kind: "manual" } as const;

  it("filters by audience and supports revoke/supersede", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const clock = new FixedClock(1_000);
    const identityEvents = new IdentityEventRepository({
      db,
      clock,
    });
    const entities = new EntityRepository({
      db,
      clock,
    });
    const commitments = new CommitmentRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });
    const audience = entities.resolve("Sam");
    const about = entities.resolve("Atlas");
    const first = commitments.add({
      type: "boundary",
      directiveFamily: "atlas_outage_confidentiality",
      directive: "Do not discuss Atlas outages with Sam",
      priority: 10,
      restrictedAudience: audience,
      aboutEntity: about,
      provenance: manualProvenance,
    });
    const second = commitments.add({
      type: "promise",
      directiveFamily: "follow_up_tomorrow",
      directive: "Follow up tomorrow",
      priority: 5,
      provenance: manualProvenance,
    });
    const replacement = commitments.add({
      type: "promise",
      directiveFamily: "follow_up_next_week",
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

    expect(commitments.revoke(first.id, "user revoked it", manualProvenance)?.revoked_at).toBe(
      1_000,
    );
    expect(commitments.supersede(second.id, replacement.id)?.superseded_by).toBe(replacement.id);
    expect(
      commitments.list({
        activeOnly: true,
      }),
    ).toEqual([replacement]);

    db.close();
  });

  it("treats a null audience as public-only for active commitment lists", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
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

    try {
      const sam = entities.resolve("Sam");
      const publicCommitment = commitments.add({
        type: "promise",
        directiveFamily: "public_work_followup",
        directive: "Follow up on public work",
        priority: 5,
        provenance: manualProvenance,
      });
      const restricted = commitments.add({
        type: "boundary",
        directiveFamily: "sam_only_details_boundary",
        directive: "Do not discuss Sam-only details elsewhere",
        priority: 10,
        restrictedAudience: sam,
        provenance: manualProvenance,
      });

      expect(
        commitments.list({
          activeOnly: true,
          audience: null,
        }),
      ).toEqual([publicCommitment]);
      expect(
        commitments.getApplicable({
          audience: null,
          nowMs: 1_000,
        }),
      ).toEqual([publicCommitment]);
      expect(
        commitments.getApplicable({
          audience: sam,
          nowMs: 1_000,
        }),
      ).toEqual([restricted, publicCommitment]);
    } finally {
      db.close();
    }
  });

  it("does not apply restricted-audience commitments to other audiences", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
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

    try {
      const alice = entities.resolve("Alice");
      const bob = entities.resolve("Bob");
      const aliceOnly = commitments.add({
        type: "rule",
        directiveFamily: "alice_response_constraints",
        directive: "Use Alice's preferred response constraints.",
        priority: 8,
        restrictedAudience: alice,
        provenance: manualProvenance,
      });
      const publicCommitment = commitments.add({
        type: "preference",
        directiveFamily: "grounded_responses",
        directive: "Keep responses grounded.",
        priority: 4,
        provenance: manualProvenance,
      });

      expect(
        commitments.getApplicable({
          audience: alice,
          nowMs: 1_000,
        }),
      ).toEqual([aliceOnly, publicCommitment]);
      expect(
        commitments.getApplicable({
          audience: bob,
          nowMs: 1_000,
        }),
      ).toEqual([publicCommitment]);
    } finally {
      db.close();
    }
  });

  it("keeps restricted-audience commitments isolated bidirectionally", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
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

    try {
      const alice = entities.resolve("Alice");
      const bob = entities.resolve("Bob");
      const aliceOnly = commitments.add({
        type: "rule",
        directiveFamily: "preferred_response_constraints",
        directive: "Use Alice's preferred response constraints.",
        priority: 8,
        restrictedAudience: alice,
        provenance: manualProvenance,
      });
      const bobOnly = commitments.add({
        type: "rule",
        directiveFamily: "preferred_response_constraints",
        directive: "Use Bob's preferred response constraints.",
        priority: 7,
        restrictedAudience: bob,
        provenance: manualProvenance,
      });

      expect(
        commitments.getApplicable({
          audience: alice,
          nowMs: 1_000,
        }),
      ).toEqual([aliceOnly]);
      expect(
        commitments.getApplicable({
          audience: bob,
          nowMs: 1_000,
        }),
      ).toEqual([bobOnly]);
    } finally {
      db.close();
    }
  });

  it("stores optional source stream entry ids for online commitments", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const clock = new FixedClock(1_000);
    const commitments = new CommitmentRepository({
      db,
      clock,
    });

    try {
      const streamEntryId = createStreamEntryId();
      const commitment = commitments.add({
        type: "preference",
        directiveFamily: "response_pattern_corrections",
        directive: "Preserve response-pattern corrections.",
        priority: 7,
        provenance: {
          kind: "online",
          process: "corrective-preference-extractor",
        },
        sourceStreamEntryIds: [streamEntryId],
      });

      expect(commitment.source_stream_entry_ids).toEqual([streamEntryId]);
      expect(commitments.get(commitment.id)?.source_stream_entry_ids).toEqual([streamEntryId]);
    } finally {
      db.close();
    }
  });

  it("finds corrective preferences by evidence stream entry id", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const clock = new FixedClock(1_000);
    const commitments = new CommitmentRepository({
      db,
      clock,
    });

    try {
      const correctiveEntryId = createStreamEntryId();
      const unrelatedEntryId = createStreamEntryId();
      commitments.add({
        type: "preference",
        directiveFamily: "corrected_behavior",
        directive: "Do not perform corrected behavior.",
        priority: 7,
        provenance: {
          kind: "online",
          process: "corrective-preference-extractor",
        },
        sourceStreamEntryIds: [correctiveEntryId],
      });
      commitments.add({
        type: "preference",
        directiveFamily: "ordinary_online_preference",
        directive: "Ordinary online preference.",
        priority: 5,
        provenance: {
          kind: "online",
          process: "goal-promotion-extractor",
        },
        sourceStreamEntryIds: [unrelatedEntryId],
      });

      expect(commitments.findByEvidenceStreamEntryId(correctiveEntryId)).toBe(true);
      expect(commitments.findByEvidenceStreamEntryId(unrelatedEntryId)).toBe(false);
    } finally {
      db.close();
    }
  });

  it("applies commitments made to an entity only for that entity by default", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
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

    try {
      const alice = entities.resolve("Alice");
      const bob = entities.resolve("Bob");
      const madeToAlice = commitments.add({
        type: "promise",
        directiveFamily: "alice_deployment_summary",
        directive: "Send Alice the deployment summary",
        priority: 5,
        madeToEntity: alice,
        provenance: manualProvenance,
      });
      const global = commitments.add({
        type: "rule",
        directiveFamily: "attach_sources",
        directive: "Keep sources attached",
        priority: 4,
        provenance: manualProvenance,
      });

      expect(
        commitments.getApplicable({
          audience: alice,
          nowMs: 1_000,
        }),
      ).toEqual([madeToAlice, global]);
      expect(
        commitments.getApplicable({
          audience: bob,
          nowMs: 1_000,
        }),
      ).toEqual([global]);
      expect(
        commitments.getApplicable({
          audience: null,
          nowMs: 1_000,
        }),
      ).toEqual([global]);
    } finally {
      db.close();
    }
  });

  it("can look up an entity by name without creating one", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const entities = new EntityRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      expect(entities.findByName("Unknown")).toBeNull();
      expect(
        (db.prepare("SELECT COUNT(*) AS count FROM entities").get() as { count: number }).count,
      ).toBe(0);
    } finally {
      db.close();
    }
  });

  it("materializes expiration and records an identity event", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const clock = new FixedClock(1_000);
    const identityEvents = new IdentityEventRepository({
      db,
      clock,
    });
    const commitments = new CommitmentRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });

    try {
      const expiring = commitments.add({
        type: "promise",
        directiveFamily: "reply_before_noon",
        directive: "Reply before noon",
        priority: 4,
        provenance: manualProvenance,
        createdAt: 100,
        expiresAt: 900,
      });

      expect(
        commitments.getApplicable({
          nowMs: 1_000,
        }),
      ).toEqual([]);
      expect(commitments.get(expiring.id)?.expired_at).toBe(900);
      expect(
        identityEvents.list({
          recordType: "commitment",
          recordId: expiring.id,
        }),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "expire",
            ts: 900,
          }),
        ]),
      );
    } finally {
      db.close();
    }
  });

  it("dedupes active commitments by directive family and audience scope", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const clock = new ManualClock(1_000);
    const commitments = new CommitmentRepository({
      db,
      clock,
    });

    try {
      const firstEntryId = createStreamEntryId();
      const secondEntryId = createStreamEntryId();
      const first = commitments.add({
        type: "preference",
        directiveFamily: "no_terminal_valediction",
        directive: "Do not add terminal valedictions.",
        priority: 7,
        provenance: manualProvenance,
        sourceStreamEntryIds: [firstEntryId],
      });

      clock.set(2_000);

      const second = commitments.add({
        type: "preference",
        directiveFamily: "No Terminal Valediction",
        directive: "Do not close with ritual farewell lines.",
        priority: 9,
        provenance: manualProvenance,
        sourceStreamEntryIds: [secondEntryId],
      });
      const active = commitments.list({ activeOnly: true });

      expect(second.id).toBe(first.id);
      expect(active).toHaveLength(1);
      expect(active[0]).toMatchObject({
        id: first.id,
        directive_family: "no_terminal_valediction",
        priority: 9,
        last_reinforced_at: 2_000,
        source_stream_entry_ids: [firstEntryId, secondEntryId],
      });
    } finally {
      db.close();
    }
  });

  it("does not merge commitments across directive families", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const commitments = new CommitmentRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      commitments.add({
        type: "preference",
        directiveFamily: "no_terminal_valediction",
        directive: "Do not add terminal valedictions.",
        priority: 7,
        provenance: manualProvenance,
      });
      commitments.add({
        type: "preference",
        directiveFamily: "respond_substantively",
        directive: "Respond substantively before closing.",
        priority: 7,
        provenance: manualProvenance,
      });

      expect(commitments.list({ activeOnly: true })).toHaveLength(2);
    } finally {
      db.close();
    }
  });

  it("does not merge commitments across audience scopes", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
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

    try {
      const tom = entities.resolve("Tom");
      const alice = entities.resolve("Alice");
      commitments.add({
        type: "preference",
        directiveFamily: "no_terminal_valediction",
        directive: "Do not add terminal valedictions for Tom.",
        priority: 7,
        restrictedAudience: tom,
        provenance: manualProvenance,
      });
      commitments.add({
        type: "preference",
        directiveFamily: "no_terminal_valediction",
        directive: "Do not add terminal valedictions for Alice.",
        priority: 7,
        restrictedAudience: alice,
        provenance: manualProvenance,
      });

      expect(commitments.list({ activeOnly: true })).toHaveLength(2);
    } finally {
      db.close();
    }
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
          directiveFamily: "attach_sources",
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
