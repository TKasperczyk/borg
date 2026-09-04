import { describe, expect, it } from "vitest";

import type { EntityId } from "../util/ids.js";

import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelFromMetadata,
  memoryDisclosureLabelMetadata,
  memoryDisclosureLabelSchema,
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "./recall-context.js";

describe("memory disclosure labels", () => {
  const alice = "ent_aaaaaaaaaaaaaaaa" as EntityId;
  const bob = "ent_bbbbbbbbbbbbbbbb" as EntityId;

  it("requires explicit public labels instead of treating empty private origins as public", () => {
    expect(relationshipPrivateMemoryDisclosureLabel([])).toMatchObject({
      disclosureClass: "unknown",
      privateToEntityIds: [],
      publicToEntityIds: [],
    });
    expect(relationshipPrivateMemoryDisclosureLabel([null, undefined])).toMatchObject({
      disclosureClass: "unknown",
      privateToEntityIds: [],
      publicToEntityIds: [],
    });
    expect(publicMemoryDisclosureLabel()).toMatchObject({
      disclosureClass: "public",
    });
  });

  it("fails closed for empty unknown-origin episode access and conflicting shared origins", () => {
    expect(
      memoryDisclosureLabelFromEpisodeAccess({
        audience_entity_id: null,
        origin_audience_entity_ids: [],
        shared: false,
      }),
    ).toMatchObject({
      disclosureClass: "unknown",
    });
    expect(
      memoryDisclosureLabelFromEpisodeAccess({
        audience_entity_id: null,
        origin_audience_entity_ids: [],
        shared: true,
      }),
    ).toMatchObject({
      disclosureClass: "public",
    });
    expect(
      memoryDisclosureLabelFromEpisodeAccess({
        audience_entity_id: alice,
        origin_audience_entity_ids: [alice],
        shared: true,
      }),
    ).toMatchObject({
      disclosureClass: "relationship_private",
      privateToEntityIds: [alice],
    });
  });

  it("preserves disclosure origin order while sorting authorization ids", () => {
    const expected = {
      disclosureClass: "relationship_private",
      // Origins keep first-occurrence chronology; authorization sets use lexical ID order.
      originAudienceEntityIds: [bob, alice],
      privateToEntityIds: [alice, bob],
      publicToEntityIds: [],
    } as const;

    expect(relationshipPrivateMemoryDisclosureLabel([bob, alice, bob])).toEqual(expected);
    expect(
      combineMemoryDisclosureLabels([
        relationshipPrivateMemoryDisclosureLabel([bob]),
        relationshipPrivateMemoryDisclosureLabel([alice]),
      ]),
    ).toEqual(expected);
    expect(
      memoryDisclosureLabelFromEpisodeAccess({
        origin_audience_entity_ids: [bob, alice],
        shared: false,
      }),
    ).toEqual(expected);
    expect(unknownMemoryDisclosureLabel([bob, alice, bob])).toEqual({
      ...expected,
      disclosureClass: "unknown",
    });
    expect(selfPrivateMemoryDisclosureLabel([bob, alice, bob])).toEqual({
      ...expected,
      disclosureClass: "self_private",
    });
  });

  it("canonicalizes label ordering at schema and metadata boundaries", () => {
    const label: MemoryDisclosureLabel = {
      disclosureClass: "operator_private",
      originAudienceEntityIds: [bob, alice, bob],
      privateToEntityIds: [bob, alice, bob],
      publicToEntityIds: [bob, alice, bob],
    };
    const expected = {
      disclosureClass: "operator_private",
      // Origins keep first-occurrence chronology; authorization sets use lexical ID order.
      originAudienceEntityIds: [bob, alice],
      privateToEntityIds: [alice, bob],
      publicToEntityIds: [alice, bob],
    } as const;

    expect(memoryDisclosureLabelSchema.parse(label)).toEqual(expected);
    expect(memoryDisclosureLabelFromMetadata(memoryDisclosureLabelMetadata(label))).toEqual(
      expected,
    );
    expect(
      memoryDisclosureLabelFromMetadata({
        disclosure_class: label.disclosureClass,
        origin_audience_entity_ids: label.originAudienceEntityIds,
        private_to_entity_ids: label.privateToEntityIds,
        public_to_entity_ids: label.publicToEntityIds,
      }),
    ).toEqual(expected);
  });
});
