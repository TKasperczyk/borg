import { describe, expect, it } from "vitest";

import {
  parseStoredProvenance,
  summarizeProvenanceForPrompt,
  toStoredProvenance,
} from "./provenance.js";

describe("provenance", () => {
  it("round-trips online provenance through stored columns", () => {
    const stored = toStoredProvenance({
      kind: "online",
      process: "reflector",
    });

    expect(stored).toEqual({
      provenance_kind: "online",
      provenance_episode_ids: "[]",
      provenance_process: "reflector",
    });
    expect(parseStoredProvenance(stored)).toEqual({
      kind: "online",
      process: "reflector",
    });
  });

  it("round-trips online reflector provenance with stream evidence", () => {
    const stored = toStoredProvenance({
      kind: "online_reflector",
      evidence_episode_ids: [],
      evidence_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
    });

    expect(stored).toEqual({
      provenance_kind: "online_reflector",
      provenance_episode_ids: "[]",
      provenance_stream_entry_ids: '["strm_aaaaaaaaaaaaaaaa"]',
      provenance_process: "reflector",
    });
    expect(parseStoredProvenance(stored)).toStrictEqual({
      kind: "online_reflector",
      evidence_episode_ids: [],
      evidence_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa"],
    });
    expect(summarizeProvenanceForPrompt(parseStoredProvenance(stored))).toBe("(online: reflector)");
  });

  it("keeps a named online reflector writer through stored columns", () => {
    const stored = toStoredProvenance({
      kind: "online_reflector",
      process: "tool.openQuestions.resolve",
      evidence_episode_ids: [],
      evidence_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
    });

    expect(stored.provenance_process).toBe("tool.openQuestions.resolve");
    const parsed = parseStoredProvenance(stored);
    expect(parsed).toStrictEqual({
      kind: "online_reflector",
      process: "tool.openQuestions.resolve",
      evidence_episode_ids: [],
      evidence_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa"],
    });
    expect(summarizeProvenanceForPrompt(parsed)).toBe("(online: tool.openQuestions.resolve)");
  });
});
