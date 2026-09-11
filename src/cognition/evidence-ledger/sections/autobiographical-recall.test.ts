import { describe, expect, it } from "vitest";

import {
  resolveAutobiographicalRecallWindow,
  shouldRecallAutobiographicalEvidence,
} from "../../autobiographical-recall.js";
import type { AutobiographicalRecallResult } from "../../autobiographical-recall.js";
import { TEMPORAL_CUE_JUDGE_SYSTEM_PROMPT } from "../../perception/temporal-cue.js";
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
      "My window here is resolved from a temporal reference in this turn's inbound text, so a perception_temporal_cue window can close well before now, and evidence outside window_start_ms/window_end_ms is absent by that scope rather than missing from the store.",
    );
  });

  it("reads a merely-mentioned window as the extractor's miss, not as the section's intent", () => {
    // The framing used to say the window resolves from a reference "including one the text only
    // mentions rather than asks about" -- describing as intent the exact case the judge is told to
    // return has_cue=false on. That left a spurious window indistinguishable from a scope the
    // sender asked for: on a cue-gated turn the section is present either way, the bounds look the
    // same either way, and nothing on the page said one of the two readings is a defect.
    const framing = buildSection(recallFixture()).get("autobiographical_recall")?.framing;

    expect(framing?.text).toContain(
      "A window minted from a reference the inbound text only mentions rather than asks about is that extractor missing its own instruction, not this section's intent: it is told that a quoted, cited, restated or reported window is not a cue, because a cue replaces my recent default outright instead of narrowing it.",
    );
    expect(framing?.text).toContain(
      "So a perception_temporal_cue window ending well before now is as consistent with a mention that should have been ignored as with a scope the sender asked for, and in the first case what the bounds cost me is my own recent record for this turn.",
    );
  });

  it("keeps the framing's account of the judge true to the judge's own prompt", () => {
    // The framing above is a claim about a different module's prompt copy. Pin the two together:
    // if the judge stops being told that a merely-reported window is not a cue, or stops being
    // told the replacement asymmetry that makes the miss expensive, this section's framing becomes
    // a description of a prompt that no longer exists -- and the reader has no way to notice.
    expect(TEMPORAL_CUE_JUDGE_SYSTEM_PROMPT).toContain(
      "quoted, cited, restated or reported windows return has_cue=false",
    );
    expect(TEMPORAL_CUE_JUDGE_SYSTEM_PROMPT).toContain(
      "replaces the default recent window outright rather than adding to it",
    );
  });

  it("names the cue as the gate on the section, not only as the bound on its window", () => {
    // shouldRecallAutobiographicalEvidence admits a non-self, non-operator, non-reflective turn
    // on the cue alone, so on those turns the cue decides the section exists as well as how wide
    // it is. Naming only the bound leaves the presence readable as a property of the store.
    const framing = buildSection(recallFixture()).get("autobiographical_recall")?.framing;

    expect(framing?.text).toContain(
      "That same reference also decides whether this section runs at all: with none in the inbound text I recall autobiographically only when the audience is myself, when the session role is operator, or when perception read this turn as reflective, so on any other turn the section's presence is itself evidence that the inbound text named a time, and its absence there is that reference missing rather than the store being empty.",
    );
  });

  it("gates the section on the cue alone for a plain non-self, non-operator turn", () => {
    const base = {
      isSelfAudience: false,
      sessionAudienceRole: "participant",
      perceptionMode: "problem_solving",
    } as unknown as Parameters<typeof shouldRecallAutobiographicalEvidence>[0];

    expect(shouldRecallAutobiographicalEvidence({ ...base, temporalCue: null })).toBe(false);
    expect(
      shouldRecallAutobiographicalEvidence({
        ...base,
        temporalCue: { sinceTs: CUE_START_MS, untilTs: CUE_END_MS, label: "cue" },
      }),
    ).toBe(true);
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
