import { describe, expect, it } from "vitest";
import { createEntityId, createSemanticNodeId } from "../../util/ids.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { commitmentMigrations } from "../commitments/migrations.js";
import { EntityRepository } from "../commitments/repository.js";
import { projectMachineReferences } from "./reference-projection.js";

describe("machine reference projection", () => {
  it("projects keys, participant codecs and serialized machine envelopes but preserves authored content", () => {
    const db = openDatabase(":memory:", { migrations: commitmentMigrations });
    try {
      const entities = new EntityRepository({ db });
      const source = entities.add({ canonicalName: "Source placeholder" });
      const survivor = entities.add({ canonicalName: "Survivor placeholder" });
      entities.merge(source.id, survivor.id);
      const input = {
        kind: "user_msg",
        content: source.id,
        description: source.id,
        sender_entity_id: source.id,
        metadata: {
          participants: [`entity_id:${source.id}`],
          attribution: { [source.id]: { entity_id: source.id } },
          scope_map: { [JSON.stringify({ made_to_entity: source.id })]: 1 },
          structural_scope_keys: [JSON.stringify({ restricted_audience: source.id })],
          state_json: JSON.stringify({ audience_entity_id: source.id, description: source.id }),
        },
      };
      const projected = entities.resolveReferences(input);
      expect(projected.content).toBe(source.id);
      expect(projected.description).toBe(source.id);
      expect(projected.sender_entity_id).toBe(survivor.id);
      expect(projected.metadata.participants).toEqual([`entity_id:${survivor.id}`]);
      expect(projected.metadata.attribution).toEqual({ [survivor.id]: { entity_id: survivor.id } });
      expect(projected.metadata.scope_map).toEqual({
        [JSON.stringify({ made_to_entity: survivor.id })]: 1,
      });
      expect(projected.metadata.structural_scope_keys).toEqual([
        JSON.stringify({ restricted_audience: survivor.id }),
      ]);
      expect(JSON.parse(projected.metadata.state_json)).toEqual({
        audience_entity_id: survivor.id,
        description: source.id,
      });
      const pasted = { ...input, content: JSON.stringify(input.metadata) };
      expect(entities.resolveReferences(pasted).content).toBe(pasted.content);
      const structured = { ...input, content: input.metadata };
      expect(entities.resolveReferences(structured).content).toEqual(input.metadata);
      expect(entities.resolveReferences(projected)).toEqual(projected);
      // Offline repair uses exactly the same field contracts and codecs, in either direction.
      expect(
        projectMachineReferences(input, (id) => (id === source.id ? survivor.id : id)),
      ).toEqual(projected);
      expect(
        projectMachineReferences(projected, (id) => (id === survivor.id ? source.id : id)),
      ).toEqual(input);
    } finally {
      db.close();
    }
  });

  it("preserves tuple cardinality and only coalesces expiry maps with their documented policy", () => {
    const source = createSemanticNodeId(),
      survivor = createSemanticNodeId();
    const resolve = (id: string) => (id === source ? survivor : id);
    expect(projectMachineReferences({ node_ids: [source, survivor] }, resolve)).toEqual({
      node_ids: [survivor, survivor],
    });
    expect(
      projectMachineReferences(
        { suppressedHandles: { [`semantic_node:${source}`]: 2, [`semantic_node:${survivor}`]: 5 } },
        resolve,
      ),
    ).toEqual({ suppressedHandles: { [`semantic_node:${survivor}`]: 5 } });
    expect(() =>
      projectMachineReferences(
        { [source]: createEntityId(), [survivor]: createEntityId() },
        resolve,
      ),
    ).toThrow("Conflicting");
  });
});
