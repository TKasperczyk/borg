import { describe, expect, it } from "vitest";

import { parseIdentityEventDisclosureSources, type IdentityEvent } from "../identity/index.js";
import { createEntityId, createGoalId } from "../../util/ids.js";
import {
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
} from "./disclosure-serializers.js";

describe("goal disclosure serialization", () => {
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
