import { describe, expect, it } from "vitest";

import { resolveAutobiographicalRecallWindow } from "../../autobiographical-recall.js";
import type { AutobiographicalRecallResult } from "../../autobiographical-recall.js";
import { DEFAULT_SESSION_ID, createEntityId } from "../../../util/ids.js";
import type { BuilderSectionContext } from "../builder-context.js";
import { createSectionBuckets } from "../section-buckets.js";
import { addAutobiographicalRecallSection } from "./autobiographical-recall.js";

const NOW_MS = 1_788_302_709_101;
// The cue the perception judge minted on 2026-09-01 from an inbound message that merely
// quoted the instant: a full 24h band closing 30h before the read.
const CUE_END_MS = 1_788_193_260_000;
const CUE_START_MS = CUE_END_MS - 86_400_000;

function recallFixture(): AutobiographicalRecallResult {
  return {
    window: {
      startMs: CUE_START_MS,
      endMs: CUE_END_MS,
      label: "24h window ending 2026-08-31 16:21Z",
      source: "perception_temporal_cue",
    },
    evidence: [
      {
        id: "stream_reflection_1",
        kind: "stream_reflection",
        groupId: "reflection",
        groupLabel: "Reflections",
        occurredAt: CUE_END_MS - 60_000,
        relativeAge: "1d ago",
        score: 0.5,
        text: "A reflection inside the cue window.",
        disclosureLabel: {
          disclosureClass: "public",
          originAudienceEntityIds: [],
          privateToEntityIds: [],
          publicToEntityIds: [],
        },
        sourceStreamEntryIds: [],
        sourceEpisodeIds: [],
        metadata: {},
      },
    ],
  };
}

function buildSection(recall: AutobiographicalRecallResult) {
  const buckets = createSectionBuckets();
  addAutobiographicalRecallSection({
    input: {
      sessionId: DEFAULT_SESSION_ID,
      nowMs: NOW_MS,
      audienceEntityId: createEntityId(),
      autobiographicalRecall: recall,
    },
    nowMs: NOW_MS,
    buckets,
  } as unknown as BuilderSectionContext);
  return buckets;
}

describe("evidence-ledger autobiographical recall section", () => {
  it("names the window's provenance, so a cue-scoped past band is not read as the present", () => {
    const framing = buildSection(recallFixture()).get("autobiographical_recall")?.framing;

    expect(framing?.text).toContain(
      "My window here is resolved from a temporal reference in this turn's inbound text, including one the text only mentions rather than asks about, so a perception_temporal_cue window can close well before now, and evidence outside window_start_ms/window_end_ms is absent by that scope rather than missing from the store.",
    );
  });

  it("stamps the bounds the framing sends the reader to, alongside the cue's own wording", () => {
    const entry = buildSection(recallFixture()).get("autobiographical_recall")?.entries[0];

    // The label is the cue's phrase, not a rendering of the bounds; both have to be on the row
    // or the reader cannot tell a historical window from the current one.
    expect(entry?.state_metadata).toMatchObject({
      window_start_ms: CUE_START_MS,
      window_end_ms: CUE_END_MS,
      window_label: "24h window ending 2026-08-31 16:21Z",
      window_source: "perception_temporal_cue",
    });
    expect(NOW_MS - CUE_END_MS).toBeGreaterThan(24 * 60 * 60 * 1000);
  });

  it("keeps a cue window in the past rather than extending it to the read", () => {
    const window = resolveAutobiographicalRecallWindow(
      { sinceTs: CUE_START_MS, untilTs: CUE_END_MS, label: "24h window ending 2026-08-31 16:21Z" },
      NOW_MS,
    );

    expect(window.source).toBe("perception_temporal_cue");
    expect(window.endMs).toBe(CUE_END_MS);
    expect(window.endMs).toBeLessThan(NOW_MS);
  });
});
