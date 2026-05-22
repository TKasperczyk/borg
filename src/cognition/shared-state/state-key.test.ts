import { describe, expect, it } from "vitest";

import {
  sharedStateKeyTokens,
  similarStateKeyClusterCount,
  stateKeysAreNearDuplicate,
  tokenizeStateKey,
} from "./state-key.js";

describe("shared-state state key utilities", () => {
  it("tokenizes state keys across dotted, underscored, and dashed segments", () => {
    expect(tokenizeStateKey("Observation..NORA_video-call--")).toEqual([
      "observation",
      "nora",
      "video",
      "call",
    ]);
    expect(tokenizeStateKey("...")).toEqual([]);
  });

  it("tokenizes off-shape state key separators defensively", () => {
    expect(tokenizeStateKey("observation.nora/april_call")).toEqual(
      tokenizeStateKey("observation.nora.april_call"),
    );
    expect(tokenizeStateKey("observation nora april call")).toEqual([
      "observation",
      "nora",
      "april",
      "call",
    ]);
    expect(tokenizeStateKey("observation.x!!@#y")).toEqual(["observation", "x", "y"]);
    expect(tokenizeStateKey("")).toEqual([]);
  });

  it("does not treat exact matches as near-duplicates", () => {
    expect(
      stateKeysAreNearDuplicate(
        "observation.nora.video_call_repeated_question",
        "observation.nora.video_call_repeated_question",
      ),
    ).toBe(false);
  });

  it("does not flag different root buckets", () => {
    expect(
      stateKeysAreNearDuplicate(
        "observation.nora.video_call_repeated_question",
        "decision.nora.video_call_repeated_question",
      ),
    ).toBe(false);
  });

  it("does not flag same-root different objects", () => {
    expect(
      stateKeysAreNearDuplicate("observation.nora.video_call", "observation.nora.tea_tin"),
    ).toBe(false);
  });

  it("flags suffix churn for the same thread", () => {
    expect(
      stateKeysAreNearDuplicate(
        "observation.nora.video_call_repeated_question_reconfirm",
        "observation.nora.video_call_repeated_question",
      ),
    ).toBe(true);
    expect(
      stateKeysAreNearDuplicate(
        "incident.api.deploy_regression_v2",
        "incident.api.deploy_regression",
      ),
    ).toBe(true);
  });

  it("flags similar threads across the second segment when token overlap remains high", () => {
    expect(
      stateKeysAreNearDuplicate(
        "observation.family.april_video_call_repeated_question",
        "observation.nora.april_video_call_repeated_question",
      ),
    ).toBe(true);
  });

  it("flags near-duplicate threads across slash and dot separator variation", () => {
    expect(
      stateKeysAreNearDuplicate(
        "observation.nora/april_call_repeated_question",
        "observation.nora.april_call_repeated_question",
      ),
    ).toBe(true);
  });

  it("does not flag one- or two-token keys", () => {
    expect(stateKeysAreNearDuplicate("observation", "observation_reconfirm")).toBe(false);
    expect(stateKeysAreNearDuplicate("decision.architecture", "decision.architecture_v2")).toBe(
      false,
    );
  });

  it("returns shared unique tokens in left-key order", () => {
    expect(
      sharedStateKeyTokens(
        "observation.nora.video_call_repeated_question_reconfirm",
        "observation.nora.video_call_repeated_question",
      ),
    ).toEqual(["observation", "nora", "video", "call", "repeated", "question"]);
  });

  it("counts connected clusters of similar active keys", () => {
    expect(
      similarStateKeyClusterCount([
        "observation.nora.video_call_repeated_question",
        "observation.nora.video_call_repeated_question_reconfirm",
        "observation.family.april_video_call_repeated_question",
        "observation.nora.april_video_call_repeated_question",
        "decision.architecture.api",
      ]),
    ).toBe(1);
  });
});
