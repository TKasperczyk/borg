import { describe, expect, it } from "vitest";

import type { ValueRecord } from "../../../memory/self/index.js";
import { createValueId } from "../../../util/ids.js";
import type { SelfSnapshot } from "../types.js";
import { summarizeVoiceAnchors } from "./voice-anchors.js";

function valueRecord(input: { label: string; state: ValueRecord["state"] }): ValueRecord {
  return {
    id: createValueId(),
    record_version: 1,
    label: input.label,
    description: `${input.label} description`,
    priority: 1,
    created_at: 1_000,
    last_affirmed: null,
    state: input.state,
    established_at: input.state === "established" ? 1_000 : null,
    confidence: 0.8,
    last_tested_at: null,
    last_contradicted_at: null,
    support_count: input.state === "established" ? 1 : 0,
    contradiction_count: 0,
    evidence_episode_ids: [],
    provenance: {
      kind: "manual",
    },
  };
}

describe("summarizeVoiceAnchors", () => {
  it("labels established self-value anchors for model-facing prompt text", () => {
    const snapshot: SelfSnapshot = {
      values: [valueRecord({ label: "groundedness", state: "established" })],
      goals: [],
      traits: [],
    };
    const summary = summarizeVoiceAnchors(snapshot);

    expect(summary).toContain("groundedness");
    expect(summary).toContain("disclosure_class=self_private");
    expect(summary).toContain("voice_note");
  });

  it("returns null when no values are established", () => {
    const snapshot: SelfSnapshot = {
      values: [valueRecord({ label: "experimental flexibility", state: "candidate" })],
      goals: [],
      traits: [],
    };

    expect(summarizeVoiceAnchors(snapshot)).toBeNull();
  });
});
