import { describe, expect, it } from "vitest";

import type { EntityId } from "../util/ids.js";

import {
  memoryDisclosureLabelFromEpisodeAccess,
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
} from "./recall-context.js";

describe("memory disclosure labels", () => {
  const alice = "ent_aaaaaaaaaaaaaaaa" as EntityId;

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
});
