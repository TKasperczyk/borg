import { describe, expect, it, vi } from "vitest";
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
      const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
      expect(entities.resolveAudienceLabel("transport:channel", other.id)).toBe(other.id);
      expect(warning).toHaveBeenCalledWith("entity.audience_binding_mismatch", {
        label: "transport:channel",
        boundEntityId: other.id,
        labelEntityId: participant.id,
      });
      warning.mockRestore();
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
      expect(entities.resolveAudienceLabel("Initial identity")).toBe(self.id);
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
  it("keeps redirect chains readable and excludes tombstones from every name lookup", () => {
    const db = openDatabase(":memory:", { migrations: commitmentMigrations });
    const entities = new EntityRepository({ db });
    try {
      const sourceId = entities.resolveExternal({
        source: "fixture",
        externalId: "old",
        canonicalName: "Transport placeholder",
      });
      const intermediate = entities.add({
        canonicalName: "Middle placeholder",
        aliases: ["middle:alias"],
      });
      const survivor = entities.add({ canonicalName: "Final placeholder", kind: "self" });
      entities.merge(sourceId, intermediate.id);
      entities.merge(intermediate.id, survivor.id);
      expect(entities.get(sourceId)?.id).toBe(survivor.id);
      expect(entities.get(intermediate.id)?.id).toBe(survivor.id);
      expect(entities.merge(sourceId, survivor.id).id).toBe(survivor.id);
      expect(() => entities.merge(sourceId, intermediate.id)).toThrow("live handle");
      expect(entities.findByExternalId("fixture", "old")).toBe(survivor.id);
      expect(entities.resolveAudienceLabel("Transport placeholder", sourceId)).toBe(survivor.id);
      expect(entities.findByNames([sourceId, "Transport placeholder", "middle:alias"])).toEqual(
        new Map([
          [sourceId, survivor.id],
          ["Transport placeholder", survivor.id],
          ["middle:alias", survivor.id],
        ]),
      );
      expect(entities.list()).toHaveLength(1);
      expect(db.prepare("SELECT count(*) AS n FROM entities").get()).toEqual({ n: 3 });
      entities.rename(sourceId, "Renamed placeholder");
      expect(entities.getSelf()?.canonical_name).toBe("Renamed placeholder");
      expect(entities.resolveAudienceLabel("Final placeholder")).toBe(survivor.id);
      expect(
        entities.resolveReferences({
          sender_entity_id: sourceId,
          reply_target_entity_id: intermediate.id,
          entity_ids: [sourceId, survivor.id],
        }),
      ).toEqual({
        sender_entity_id: survivor.id,
        reply_target_entity_id: survivor.id,
        entity_ids: [survivor.id, survivor.id],
      });
      expect(
        db
          .prepare("SELECT canonical_name FROM resolved_entities WHERE lookup_id = ?")
          .get(sourceId),
      ).toEqual({ canonical_name: "Renamed placeholder" });
      db.prepare("UPDATE entities SET merged_into = ? WHERE id = ?").run(sourceId, survivor.id);
      expect(() => entities.get(sourceId)).toThrow("cycle");
    } finally {
      db.close();
    }
  });

  it("retains a renamed person's old transport label", () => {
    const db = openDatabase(":memory:", { migrations: commitmentMigrations });
    try {
      const entities = new EntityRepository({ db });
      const id = entities.resolveAudienceLabel("transport:placeholder");
      entities.rename(id, "Declared placeholder");
      expect(entities.resolveAudienceLabel("transport:placeholder")).toBe(id);
      expect(entities.list()).toHaveLength(1);
    } finally {
      db.close();
    }
  });
});
