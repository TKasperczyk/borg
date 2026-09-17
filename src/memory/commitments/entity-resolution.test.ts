import { describe, expect, it } from "vitest";
import { openDatabase } from "../../storage/sqlite/index.js";
import { createEntityId } from "../../util/ids.js";
import { commitmentMigrations } from "./migrations.js";
import { EntityRepository } from "./repository.js";

describe("entity identity resolution", () => {
  it("resolves audience canonical names, aliases, and IDs across every kind", () => {
    const db = openDatabase(":memory:", { migrations: commitmentMigrations });
    const entities = new EntityRepository({ db });
    try {
      for (const kind of ["person", "group", "abstract", "self"] as const) {
        const entity = entities.add({
          canonicalName: `${kind} placeholder`,
          aliases: [`transport:${kind}`],
          kind,
          provenance: "user_declared",
        });
        expect(entities.resolveAudienceLabel(entity.canonical_name)).toBe(entity.id);
        expect(entities.resolveAudienceLabel(`transport:${kind}`)).toBe(entity.id);
        expect(entities.resolveAudienceLabel(entity.id)).toBe(entity.id);
        expect(entities.get(entity.id)?.name_provenance).toBe("user_declared");
      }
      expect(entities.list()).toHaveLength(4);
      expect(() => entities.resolveAudienceLabel(createEntityId())).toThrow("Unknown audience");
    } finally {
      db.close();
    }
  });

  it("records explicit session handles so later episode stamping shares the binding", () => {
    const db = openDatabase(":memory:", { migrations: commitmentMigrations });
    const entities = new EntityRepository({ db });
    try {
      const participant = entities.add({
        canonicalName: "Participant",
        provenance: "user_declared",
      });
      expect(entities.resolveAudienceLabel("transport:channel", participant.id)).toBe(
        participant.id,
      );
      expect(entities.resolveAudienceLabel("transport:channel")).toBe(participant.id);
      const other = entities.add({ canonicalName: "Other participant" });
      expect(() => entities.resolveAudienceLabel("transport:channel", other.id)).toThrow(
        "conflicts",
      );
      expect(entities.list()).toHaveLength(2);
    } finally {
      db.close();
    }
  });

  it("keeps the self ID through renames and rejects another self", () => {
    const db = openDatabase(":memory:", { migrations: commitmentMigrations });
    const entities = new EntityRepository({ db });
    try {
      const self = entities.add({ canonicalName: "Initial identity", kind: "self" });
      entities.rename(self.id, "Chosen identity");
      expect(entities.resolve("Initial identity", { kind: "self" })).toBe(self.id);
      expect(entities.getSelf()?.canonical_name).toBe("Chosen identity");
      expect(() => entities.add({ canonicalName: "Another identity", kind: "self" })).toThrow(
        "already has",
      );
      expect(entities.list({ kind: "self" })).toHaveLength(1);
    } finally {
      db.close();
    }
  });

  it("preserves self kind when an external identity is renamed without a kind override", () => {
    const db = openDatabase(":memory:", { migrations: commitmentMigrations });
    const entities = new EntityRepository({ db });
    try {
      const identity = {
        source: "fixture",
        externalId: "identity",
        canonicalName: "Initial identity",
      };
      const id = entities.resolveExternal({ ...identity, kind: "self" });
      expect(entities.resolveExternal({ ...identity, canonicalName: "Chosen identity" })).toBe(id);
      expect(entities.getSelf()).toMatchObject({
        id,
        kind: "self",
        canonical_name: "Chosen identity",
      });
    } finally {
      db.close();
    }
  });
});
