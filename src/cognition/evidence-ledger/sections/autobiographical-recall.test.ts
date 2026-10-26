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
// A historical band outside the seven-day recent default.
const CUE_END_MS = NOW_MS - 14 * 86_400_000;
const CUE_START_MS = CUE_END_MS - 86_400_000;

function recallFixture(): AutobiographicalRecallResult {
  return {
    window: resolveAutobiographicalRecallWindow(
      {
        sinceTs: CUE_START_MS,
        untilTs: CUE_END_MS,
        label: "historical band",
      },
      NOW_MS,
    ),
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
  it("describes additive time bands and the unchanged recent default", () => {
    const framing = buildSection(recallFixture()).get("autobiographical_recall")?.framing;
    expect(framing?.text).toContain("My recent seven-day default is always recalled");
    expect(framing?.text).toContain(
      "never replaces my recent record or fills a gap between the bands",
    );
    expect(framing?.text).toContain(
      "spurious cue may add irrelevant older evidence but cannot remove recent evidence",
    );
  });

  it("keeps the framing's account of the judge true to the judge's own prompt", () => {
    expect(TEMPORAL_CUE_JUDGE_SYSTEM_PROMPT).toContain(
      "quoted, cited, restated or reported windows return has_cue=false",
    );
    expect(TEMPORAL_CUE_JUDGE_SYSTEM_PROMPT).toContain("does not replace recent evidence");
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
      window_label: "historical band",
      window_source: "perception_temporal_cue",
    });
    expect(NOW_MS - CUE_END_MS).toBeGreaterThan(24 * 60 * 60 * 1000);
  });

  it("retains the recent default and adds a disjoint historical cue", () => {
    const window = resolveAutobiographicalRecallWindow(
      { sinceTs: CUE_START_MS, untilTs: CUE_END_MS, label: "24h window ending 2026-08-31 16:21Z" },
      NOW_MS,
    );

    expect(window.source).toBe("recent_default");
    expect(window.startMs).toBe(NOW_MS - 7 * 86_400_000);
    expect(window.endMs).toBe(NOW_MS);
    expect(window.additionalWindow).toMatchObject({
      source: "perception_temporal_cue",
      startMs: CUE_START_MS,
      endMs: CUE_END_MS,
    });
  });
});
