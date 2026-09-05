import { describe, expect, it } from "vitest";

import type { EntityId } from "../../util/ids.js";

import {
  isMemoryDisclosureLabelVisibleToAnyAudience,
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "./disclosure-label.js";

const ALICE = "ent_alice00000000001" as EntityId;
const BOB = "ent_bob0000000000001" as EntityId;

describe("isMemoryDisclosureLabelVisibleToAnyAudience", () => {
  it("shows public records to anyone, including an empty audience", () => {
    expect(isMemoryDisclosureLabelVisibleToAnyAudience(publicMemoryDisclosureLabel(), [])).toBe(
      true,
    );
    expect(isMemoryDisclosureLabelVisibleToAnyAudience(publicMemoryDisclosureLabel(), [BOB])).toBe(
      true,
    );
  });

  it("never shows records without provenance", () => {
    expect(
      isMemoryDisclosureLabelVisibleToAnyAudience(unknownMemoryDisclosureLabel([ALICE]), [ALICE]),
    ).toBe(false);
  });

  it("shows private records only when one of the audiences is among the entities they are private to", () => {
    const relationship = relationshipPrivateMemoryDisclosureLabel([ALICE]);
    const self = selfPrivateMemoryDisclosureLabel([ALICE, BOB]);

    expect(isMemoryDisclosureLabelVisibleToAnyAudience(relationship, [ALICE])).toBe(true);
    expect(isMemoryDisclosureLabelVisibleToAnyAudience(relationship, [BOB])).toBe(false);
    expect(isMemoryDisclosureLabelVisibleToAnyAudience(relationship, [BOB, ALICE])).toBe(true);
    expect(isMemoryDisclosureLabelVisibleToAnyAudience(relationship, [])).toBe(false);
    expect(isMemoryDisclosureLabelVisibleToAnyAudience(self, [BOB])).toBe(true);
  });
});
