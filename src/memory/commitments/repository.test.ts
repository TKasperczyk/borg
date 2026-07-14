import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { ProvenanceError } from "../../util/errors.js";
import {
  createCommitmentId,
  createEntityId,
  createSharedStateEntryId,
  createStreamEntryId,
} from "../../util/ids.js";
import { identityMigrations, IdentityEventRepository } from "../identity/index.js";
import { commitmentMigrations } from "./migrations.js";
import { CommitmentRepository, EntityRepository } from "./repository.js";

describe("commitment repository", () => {
  const manualProvenance = { kind: "manual" } as const;

  it("keeps borg_role creator as a single repository-level role", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const entities = new EntityRepository({
      db,
      clock: new FixedClock(1_000),
    });
    const tom = entities.add({
      id: createEntityId(),
      canonicalName: "Tom",
    });
    const ada = entities.add({
      id: createEntityId(),
      canonicalName: "Ada",
    });

    try {
      expect(entities.setBorgRole(tom.id, "creator")?.borg_role).toBe("creator");
      expect(entities.getCreator()?.id).toBe(tom.id);

      expect(entities.setBorgRole(ada.id, "creator")?.borg_role).toBe("creator");
      expect(entities.getCreator()?.id).toBe(ada.id);
      expect(entities.get(tom.id)?.borg_role).toBeNull();

      expect(entities.setBorgRole(ada.id, null)?.borg_role).toBeNull();
      expect(entities.getCreator()).toBeNull();
    } finally {
      db.close();
    }
  });

  it("resolves the self entity by kind, not by name", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const entities = new EntityRepository({
      db,
      clock: new FixedClock(1_000),
    });
    entities.add({
      id: createEntityId(),
      canonicalName: "Alex",
    });

    try {
      expect(entities.getSelf()).toBeNull();

      const self = entities.add({
        id: createEntityId(),
        canonicalName: "self-renamed",
        kind: "self",
      });

      expect(entities.getSelf()?.id).toBe(self.id);
    } finally {
      db.close();
    }
  });

  it("keeps the seeded self audience label aligned with getSelf", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const entities = new EntityRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      const selfId = entities.resolve("self", {
        kind: "self",
        provenance: "assistant_seeded",
      });

      expect(entities.getSelf()?.id).toBe(selfId);
      expect(
        entities.resolve("self", {
          kind: "self",
          provenance: "assistant_seeded",
        }),
      ).toBe(selfId);
    } finally {
      db.close();
    }
  });

  it("ensures one stable self entity while preserving prior names as aliases", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const entities = new EntityRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      const first = entities.ensureSelf("team-agent");
      expect(first.canonical_name).toBe("team-agent");
      expect(first.aliases).toEqual(["self"]);
      expect(entities.resolve("self", { kind: "self" })).toBe(first.id);

      entities.addAlias(first.id, "memory-borg");
      const renamed = entities.ensureSelf("Team Memory", {
        provenance: "user_declared",
      });
      const repeated = entities.ensureSelf("Team Memory");

      expect(renamed.id).toBe(first.id);
      expect(repeated.id).toBe(first.id);
      expect(repeated.aliases).toEqual(["self", "memory-borg", "team-agent"]);
      expect(repeated.name_provenance).toBe("user_declared");

      const configuredRename = entities.ensureSelf("configured-agent");
      expect(configuredRename.id).toBe(first.id);
      expect(configuredRename.canonical_name).toBe("configured-agent");
      expect(configuredRename.aliases).toEqual([
        "self",
        "memory-borg",
        "team-agent",
        "Team Memory",
      ]);
      expect(configuredRename.name_provenance).toBe("config_default_user");
      expect(entities.list({ kind: "self" })).toHaveLength(1);
    } finally {
      db.close();
    }
  });

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

  it("lists active commitments globally without audience or made-to filtering", () => {
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
    const madeTo = entities.resolve("Riley");
    const restricted = commitments.add({
      type: "boundary",
      directiveFamily: "sam_boundary",
      directive: "Keep Sam-specific boundary active.",
      priority: 10,
      restrictedAudience: audience,
      provenance: manualProvenance,
    });
    const madeToCommitment = commitments.add({
      type: "promise",
      directiveFamily: "riley_followup",
      directive: "Follow up with Riley.",
      priority: 5,
      madeToEntity: madeTo,
      provenance: manualProvenance,
    });

    try {
      expect(
        commitments.list({
          activeOnly: true,
          nowMs: 1_000,
        }),
      ).toEqual(expect.arrayContaining([restricted, madeToCommitment]));
      expect(
        commitments.getApplicable({
          audience: null,
          nowMs: 1_000,
        }),
      ).toEqual([]);
    } finally {
      db.close();
    }
  });

  it("counts active commitments without materializing expired records", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const clock = new FixedClock(20);
    const identityEvents = new IdentityEventRepository({
      db,
      clock,
    });
    const commitments = new CommitmentRepository({
      db,
      clock,
      identityEventRepository: identityEvents,
    });
    const commitmentRows = () =>
      db.prepare("SELECT id, expired_at FROM commitments ORDER BY id").all();
    const identityEventRows = () =>
      db.prepare("SELECT record_type, record_id, action FROM identity_events ORDER BY id").all();

    try {
      commitments.add({
        type: "promise",
        directiveFamily: "expired metrics fixture",
        directive: "Follow up before the old deadline.",
        priority: 5,
        provenance: manualProvenance,
        createdAt: 1,
        expiresAt: 10,
      });
      const beforeCommitments = commitmentRows();
      const beforeEvents = identityEventRows();

      expect(commitments.countActive()).toBe(0);
      expect(commitmentRows()).toEqual(beforeCommitments);
      expect(identityEventRows()).toEqual(beforeEvents);
    } finally {
      db.close();
    }
  });

  it("stores and counts active commitments by kind", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const commitments = new CommitmentRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      const boundary = commitments.add({
        type: "boundary",
        kind: "boundary",
        directiveFamily: "do_not_discuss_atlas",
        directive: "Do not discuss Atlas.",
        priority: 9,
        provenance: manualProvenance,
      });
      commitments.add({
        type: "preference",
        kind: "participant_preference",
        directiveFamily: "morning_meetings",
        directive: "Prefer morning meetings.",
        priority: 4,
        provenance: manualProvenance,
      });
      commitments.add({
        type: "rule",
        kind: "process_norm",
        directiveFamily: "expired_process",
        directive: "Use the expired process.",
        priority: 3,
        provenance: manualProvenance,
        createdAt: 100,
        expiresAt: 500,
      });

      expect(commitments.get(boundary.id)?.kind).toBe("boundary");
      expect(commitments.get(boundary.id)).toMatchObject({
        enforcement_class: "critical",
        critical_domain: "audience_scope",
      });
      expect(commitments.countActiveByKind(1_000)).toEqual({
        assistant_commitment: 0,
        audience_rule: 0,
        participant_preference: 1,
        boundary: 1,
        process_norm: 0,
      });
      expect(commitments.countActiveByEnforcementClass(1_000)).toEqual({
        critical: 1,
        advisory: 1,
      });
    } finally {
      db.close();
    }
  });

  it("counts revoked, expired, and canonicalized commitments", () => {
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
      const active = commitments.add({
        type: "promise",
        directiveFamily: "active fixture",
        directive: "Keep the active fixture.",
        priority: 5,
        provenance: manualProvenance,
      });
      const revoked = commitments.add({
        type: "promise",
        directiveFamily: "revoked fixture",
        directive: "Retire the revoked fixture.",
        priority: 5,
        provenance: manualProvenance,
      });
      const expired = commitments.add({
        type: "promise",
        directiveFamily: "expired fixture",
        directive: "Expire the old fixture.",
        priority: 5,
        provenance: manualProvenance,
        createdAt: 500,
        expiresAt: 900,
      });
      const canonicalized = commitments.add({
        type: "promise",
        directiveFamily: "canonicalized fixture",
        directive: "Canonicalize the fixture.",
        priority: 5,
        provenance: manualProvenance,
      });

      commitments.revoke(revoked.id, "test revocation", manualProvenance);
      commitments.revoke(canonicalized.id, "canonicalized", manualProvenance, undefined, {
        canonicalizedByArtifactEntryId: createSharedStateEntryId(),
      });

      expect(commitments.countActive()).toBe(1);
      expect(commitments.countRevoked()).toBe(2);
      expect(commitments.countExpired()).toBe(1);
      expect(commitments.countCanonicalized()).toBe(1);
      expect(commitments.get(active.id)?.revoked_at).toBeNull();
      expect(commitments.get(expired.id)?.expires_at).toBe(900);
    } finally {
      db.close();
    }
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

  it("stores audience-label provenance and upgrades after user declaration", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const entities = new EntityRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      const tom = entities.resolve("Tom", {
        provenance: "transport_audience_label",
      });

      expect(entities.get(tom)?.name_provenance).toBe("transport_audience_label");

      const resolvedAgain = entities.resolve("Tom", {
        provenance: "user_declared",
      });

      expect(resolvedAgain).toBe(tom);
      expect(entities.get(tom)?.name_provenance).toBe("user_declared");

      entities.resolve("Tom", {
        provenance: "config_default_user",
      });
      expect(entities.get(tom)?.name_provenance).toBe("user_declared");
    } finally {
      db.close();
    }
  });

  it("migrates legacy commitments to assistant commitment kind", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-commitments-"));
    const dbPath = join(tempDir, "borg.db");
    const commitmentId = createCommitmentId();

    try {
      const legacyDb = openDatabase(dbPath, {
        migrations: commitmentMigrations.filter((migration) => migration.id <= 9),
      });

      legacyDb
        .prepare(
          `
            INSERT INTO commitments (
              id, type, directive, priority, source_episode_ids, created_at,
              directive_family, last_reinforced_at, provenance_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(commitmentId, "rule", "Keep the legacy rule.", 5, "[]", 1, "legacy_rule", 1, "manual");
      legacyDb.close();

      const db = openDatabase(dbPath, {
        migrations: commitmentMigrations,
      });
      const commitments = new CommitmentRepository({
        db,
        clock: new FixedClock(1_000),
      });

      try {
        expect(commitments.get(commitmentId)?.kind).toBe("assistant_commitment");
      } finally {
        db.close();
      }
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("falls back to created_at when a legacy row has a null last_reinforced_at", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-commitments-"));
    const dbPath = join(tempDir, "borg.db");
    const commitmentId = createCommitmentId();
    const createdAt = 1_700_000_000_000;

    try {
      const legacyDb = openDatabase(dbPath, {
        migrations: commitmentMigrations.filter((migration) => migration.id <= 9),
      });

      legacyDb
        .prepare(
          `
            INSERT INTO commitments (
              id, type, directive, priority, source_episode_ids, created_at,
              directive_family, last_reinforced_at, provenance_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          commitmentId,
          "rule",
          "Keep the legacy rule.",
          5,
          "[]",
          createdAt,
          "legacy_rule",
          null,
          "manual",
        );
      legacyDb.close();

      const db = openDatabase(dbPath, {
        migrations: commitmentMigrations,
      });
      const commitments = new CommitmentRepository({
        db,
        clock: new FixedClock(1_000),
      });

      try {
        const record = commitments.get(commitmentId);
        // Without the fallback, Number(null) -> 0 would surface as a 1970-epoch recency label.
        expect(record?.last_reinforced_at).toBe(createdAt);
      } finally {
        db.close();
      }
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });

  it("defaults new entities to person and lists by kind", () => {
    const db = openDatabase(":memory:", {
      migrations: commitmentMigrations,
    });
    const entities = new EntityRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      const alice = entities.resolve("Alice");
      const project = entities.resolve("Atlas", {
        kind: "abstract",
      });
      const group = entities.add({
        canonicalName: "planning-room",
        kind: "group",
      });

      expect(entities.get(alice)?.kind).toBe("person");
      expect(entities.get(project)?.kind).toBe("abstract");
      expect(entities.get(group.id)?.kind).toBe("group");
      expect(entities.list({ kind: "person" }).map((entity) => entity.id)).toEqual([alice]);
      expect(entities.list({ kind: "group" }).map((entity) => entity.id)).toEqual([group.id]);
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
        kind: "participant_preference",
        directiveFamily: "no_terminal_valediction",
        directive: "Do not add terminal valedictions.",
        priority: 7,
        provenance: manualProvenance,
        sourceStreamEntryIds: [firstEntryId],
      });

      clock.set(2_000);

      const second = commitments.add({
        type: "preference",
        kind: "participant_preference",
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

  it("does not merge commitments with the same directive family and scope but different kind or type", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const commitments = new CommitmentRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      const assistantCommitment = commitments.add({
        type: "promise",
        kind: "assistant_commitment",
        directiveFamily: "release_window",
        directive: "Keep the release window locked.",
        priority: 5,
        provenance: manualProvenance,
      });
      const boundary = commitments.add({
        type: "boundary",
        kind: "boundary",
        directiveFamily: "release_window",
        directive: "Do not disclose the release window externally.",
        priority: 9,
        provenance: manualProvenance,
      });
      const active = commitments.list({ activeOnly: true });

      expect(boundary.id).not.toBe(assistantCommitment.id);
      expect(active.map((commitment) => commitment.id)).toEqual([
        boundary.id,
        assistantCommitment.id,
      ]);
      expect(active.map((commitment) => commitment.kind)).toEqual([
        "boundary",
        "assistant_commitment",
      ]);
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

  it("can bypass directive-family merge for explicit replacement inserts", () => {
    const db = openDatabase(":memory:", {
      migrations: composeMigrations(commitmentMigrations, identityMigrations),
    });
    const clock = new ManualClock(1_000);
    const commitments = new CommitmentRepository({
      db,
      clock,
    });

    try {
      const first = commitments.add({
        type: "preference",
        directiveFamily: "no_terminal_valediction",
        directive: "Do not add terminal valedictions.",
        priority: 7,
        provenance: manualProvenance,
      });

      clock.set(2_000);

      const replacement = commitments.add({
        type: "preference",
        directiveFamily: "no_terminal_valediction",
        directive: "Do not close with ritual farewell lines.",
        priority: 9,
        provenance: manualProvenance,
        skipDirectiveFamilyMerge: true,
      });

      expect(replacement.id).not.toBe(first.id);
      expect(
        commitments
          .list({ activeOnly: true })
          .filter((commitment) => commitment.directive_family === "no_terminal_valediction"),
      ).toHaveLength(2);
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
