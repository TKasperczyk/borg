import { describe, expect, it } from "vitest";

import type { SocialProfile } from "../../memory/social/index.js";
import type { EntityId } from "../../util/ids.js";
import type { PerceptionResult } from "../types.js";
import { deriveProceduralContext } from "./context-derivation.js";

function makePerception(
  mode: PerceptionResult["mode"],
  entities: readonly string[] = [],
): Pick<PerceptionResult, "mode" | "entities"> {
  return {
    mode,
    entities: [...entities],
  };
}

function makeAudienceProfile(entityId: EntityId): SocialProfile {
  return {
    entity_id: entityId,
    trust: 0.7,
    attachment: 0.4,
    communication_style: null,
    shared_history_summary: null,
    last_interaction_at: 1_000,
    interaction_count: 3,
    commitment_count: 0,
    sentiment_history: [],
    notes: null,
    created_at: 100,
    updated_at: 1_000,
  };
}

describe("deriveProceduralContext", () => {
  it("derives problem kind and tags from problem-solving text and entities", () => {
    const context = deriveProceduralContext({
      userMessage: "Fix this TypeScript compiler error in the deploy path.",
      perception: makePerception("problem_solving", ["TypeScript", "Deploy"]),
      isSelfAudience: true,
      audienceEntityId: null,
    });

    expect(context).toMatchObject({
      problem_kind: "code_debugging",
      domain_tags: ["deploy", "typescript"],
      audience_scope: "self",
      context_key: "code_debugging:deploy,typescript:self",
    });
  });

  it("derives audience scope from self, known, and unknown audiences", () => {
    const knownEntityId = "ent_aaaaaaaaaaaaaaaa" as EntityId;
    const firstContactEntityId = "ent_bbbbbbbbbbbbbbbb" as EntityId;

    expect(
      deriveProceduralContext({
        userMessage: "Plan the Atlas roadmap.",
        perception: makePerception("problem_solving", ["Atlas"]),
        isSelfAudience: true,
        audienceEntityId: null,
      })?.audience_scope,
    ).toBe("self");
    expect(
      deriveProceduralContext({
        userMessage: "Plan the Atlas roadmap.",
        perception: makePerception("problem_solving", ["Atlas"]),
        isSelfAudience: false,
        audienceEntityId: knownEntityId,
        audienceProfile: makeAudienceProfile(knownEntityId),
      })?.audience_scope,
    ).toBe("known_other");
    expect(
      deriveProceduralContext({
        userMessage: "Plan the Atlas roadmap.",
        perception: makePerception("problem_solving", ["Atlas"]),
        isSelfAudience: false,
        audienceEntityId: firstContactEntityId,
        audienceProfile: null,
      })?.audience_scope,
    ).toBe("unknown");
  });

  it("returns null for empty unknown context", () => {
    expect(
      deriveProceduralContext({
        userMessage: "",
        perception: makePerception("idle"),
        isSelfAudience: false,
        audienceEntityId: null,
      }),
    ).toBeNull();
  });

  it("returns null for no-audience problem-solving with no useful context", () => {
    expect(
      deriveProceduralContext({
        userMessage: "",
        perception: makePerception("problem_solving"),
        isSelfAudience: false,
        audienceEntityId: null,
        audienceProfile: null,
      }),
    ).toBeNull();
  });

  it("classifies problem-solving with only a code entity as code design", () => {
    expect(
      deriveProceduralContext({
        userMessage: "",
        perception: makePerception("problem_solving", ["Code"]),
        isSelfAudience: true,
        audienceEntityId: null,
      }),
    ).toMatchObject({
      problem_kind: "code_design",
      domain_tags: [],
      audience_scope: "self",
      context_key: "code_design::self",
    });
  });

  it("caps domain tags after canonicalization, dedupe, and generic filtering", () => {
    const context = deriveProceduralContext({
      userMessage: "",
      perception: makePerception("problem_solving", [
        "TypeScript",
        "typescript",
        "Code",
        "code",
        "lifetime",
        "borrow",
      ]),
      isSelfAudience: true,
      audienceEntityId: null,
    });

    expect(context?.domain_tags).toHaveLength(3);
    expect(new Set(context?.domain_tags)).toEqual(
      new Set(["typescript", "lifetime", "borrow"]),
    );
  });
});
