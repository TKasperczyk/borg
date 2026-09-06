import { describe, expect, it } from "vitest";

import { parseIdentityEventDisclosureSources, type IdentityEvent } from "../identity/index.js";
import { createEntityId, createGoalId } from "../../util/ids.js";
import {
  goalMemoryDisclosureLabel,
  identityEventMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "./disclosure-serializers.js";
import {
  memoryDisclosureLabelMetadata,
  relationshipPrivateMemoryDisclosureLabel,
} from "./disclosure-label.js";

describe("goal disclosure serialization", () => {
  it.each([false, true])(
    "preserves historical goal block snapshots and fails closed when their source label is missing (labeled=%s)",
    async (labeled) => {
      const owner = createEntityId();
      const sourceOwner = createEntityId();
      const event: IdentityEvent = {
        id: 1,
        record_type: "goal",
        record_id: createGoalId(),
        action: "block",
        old_value: null,
        new_value: {
          owner_entity_id: owner,
          block_history: [
            {
              blocker: { kind: "until", until: 5_000 },
              blocked_at: 1_000,
              reason: "Rationale from another participant's memory",
              ...(labeled
                ? {
                    disclosure_label: memoryDisclosureLabelMetadata(
                      relationshipPrivateMemoryDisclosureLabel([sourceOwner]),
                    ),
                  }
                : {}),
            },
          ],
        },
        reason: null,
        provenance: { kind: "online", process: "tool.goals.block" },
        review_item_id: null,
        overwrite_without_review: false,
        ts: 1_000,
      };
      const original = structuredClone(event);
      const label = await identityEventMemoryDisclosureLabel(event);
      expect(label.disclosureClass).toBe(labeled ? "relationship_private" : "unknown");
      expect(label.privateToEntityIds).toContain(owner);
      if (labeled) expect(label.privateToEntityIds).toContain(sourceOwner);
      expect(event).toEqual(original);
    },
  );

  it("does not authorize a private goal's counterparty as audience or origin", () => {
    const privateAudience = createEntityId();
    const thirdPartyCounterparty = createEntityId();
    const goal = {
      audience_entity_id: privateAudience,
      owner_entity_id: null,
      counterparty_entity_id: thirdPartyCounterparty,
    };

    const payload = memoryDisclosurePayloadFields(goalMemoryDisclosureLabel(goal));

    expect(payload.disclosure_label).toMatchObject({
      origin_audience_entity_ids: [privateAudience],
      private_to_entity_ids: [privateAudience],
    });
    expect(payload.disclosure_label.origin_audience_entity_ids).not.toContain(
      thirdPartyCounterparty,
    );
    expect(payload.disclosure_label.private_to_entity_ids).not.toContain(thirdPartyCounterparty);

    const identitySources = parseIdentityEventDisclosureSources({
      id: 1,
      record_type: "goal",
      record_id: createGoalId(),
      action: "create",
      old_value: null,
      new_value: goal,
      reason: null,
      provenance: { kind: "online", process: "goal-promotion-extractor" },
      review_item_id: null,
      overwrite_without_review: false,
      ts: 1_000,
    } satisfies IdentityEvent);

    expect(identitySources.audienceEntityIds).toEqual([privateAudience]);
    expect(identitySources.audienceEntityIds).not.toContain(thirdPartyCounterparty);
  });
});
