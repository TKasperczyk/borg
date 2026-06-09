import { describe, expect, it } from "vitest";

import { createCreatorDirectiveId, createEntityId } from "../../../util/ids.js";
import {
  CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_SUBKINDS,
  creatorDirectiveReconciliationReviewRefsSchema,
  creatorDirectiveReconciliationSubkindSchema,
} from "./creator-directive-reconciliation.js";

describe("creator directive reconciliation review handler", () => {
  it("keeps legacy subkinds readable without exposing them as emitted subkinds", () => {
    const firstId = createCreatorDirectiveId();
    const secondId = createCreatorDirectiveId();
    const creatorId = createEntityId();
    const familyKey = {
      kind: "response_policy",
      subject_kind: "system",
      subject_entity_id: null,
    } as const;
    const scopeEquivalence = {
      created_by_entity_id: creatorId,
      disclosure_policy: {
        content_scope: "public",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
        subject_may_know: null,
        mention_policy: "answer_if_asked",
        denied_audience_behavior: "omit",
        boundary_prompt: null,
        topic_tags: [],
      },
      activation_policy: {
        scope: "same_as_disclosure",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
      },
    } as const;
    const baseRefs = {
      target_type: "creator_directive_reconciliation",
      directive_ids: [firstId, secondId],
      family_key: familyKey,
      members: [
        {
          id: firstId,
          family_key: familyKey,
          scope_equivalence: scopeEquivalence,
        },
        {
          id: secondId,
          family_key: familyKey,
          scope_equivalence: scopeEquivalence,
        },
      ],
      judgment: {
        member_ids: [firstId, secondId],
        verdict: "conflicting",
        resolution: "escalate",
        survivor_id: null,
        loser_ids: [],
        confidence: "medium",
        rationale: "Manual review fixture.",
      },
    } as const;

    expect(CREATOR_DIRECTIVE_RECONCILIATION_REVIEW_SUBKINDS).toEqual([
      "conflict",
      "disclosure_widening",
    ]);
    expect(creatorDirectiveReconciliationSubkindSchema.safeParse("conflict").success).toBe(true);
    expect(
      creatorDirectiveReconciliationSubkindSchema.safeParse("same_content_different_scope")
        .success,
    ).toBe(false);
    expect(
      creatorDirectiveReconciliationReviewRefsSchema.parse({
        ...baseRefs,
        subkind: "same_content_different_scope",
      }).subkind,
    ).toBe("same_content_different_scope");
    expect(
      creatorDirectiveReconciliationReviewRefsSchema.parse({
        ...baseRefs,
        subkind: "low_confidence_redundancy",
      }).subkind,
    ).toBe("low_confidence_redundancy");
  });
});
