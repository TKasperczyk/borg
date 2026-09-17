import type { AutobiographicalRecallEvidenceItem } from "../../autobiographical-recall.js";
import { autobiographicalRecallCapStateMetadata } from "../autobiographical-recall-cap-metadata.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "../entry-metadata.js";
import {
  addEntry,
  AUTOBIOGRAPHICAL_RECALL_TRUST_RANK,
  setSectionFraming,
} from "../section-buckets.js";
import { SELF_DECISION_LABEL_SCOPE_FRAMING, type EvidenceLedgerSourceType } from "../types.js";

const AUTOBIOGRAPHICAL_RECALL_FRAMING =
  "Autobiographical recall entries are past evidence for me to re-examine during this turn. I treat recalled self_decision rows as historical decisions and rationales, not standing verdicts; I revise them when current evidence warrants. " +
  `${SELF_DECISION_LABEL_SCOPE_FRAMING} This section does carry reaches, separately, as outbound_attempt rows, so a label and an attempt from one turn appear here as two different rows. Those rows compete for recency-ordered slots shared with my thoughts and silence decisions rather than holding a reserved budget of their own, so an attempt missing from this section may simply have been outranked by later stream entries, and its absence here is not evidence that no reach was made. My recent seven-day default is always recalled when this section runs. A perception_temporal_cue adds an earlier time band within the same cap, after the recent default; it never replaces my recent record or fills a gap between the bands. Each row carries its own band's window_start_ms/window_end_ms. A quoted, cited, restated or reported window is not a cue under the extractor's instruction; a spurious cue may add irrelevant older evidence but cannot remove recent evidence. That same reference also decides whether this section runs at all: with none in the inbound text I recall autobiographically only when the audience is myself, when the session role is operator, or when perception read this turn as reflective, so on any other turn the section's presence is itself evidence that the inbound text named a time, and its absence there is that reference missing rather than the store being empty. Apart from the reduction figures, the framing counts here are the source kinds present in the assembly, one key per kind, so they sum to rows_assembled exactly and a kind this object does not name contributed no rows to it.`;

function sourceTypeForAutobiographicalItem(
  item: AutobiographicalRecallEvidenceItem,
): EvidenceLedgerSourceType {
  switch (item.kind) {
    case "episode":
      return "episode";
    case "action":
      return "action_record";
    case "goal":
      return "system_metadata";
    case "stream_reflection":
    case "silence_decision":
    case "outbound_attempt":
    case "observed_presence":
      return "system_metadata";
    case "activity":
    case "self_decision":
    case "observed_social_event":
    case "open_question":
    case "autobiographical_period":
      return "system_metadata";
  }
}

export function addAutobiographicalRecallSection(context: BuilderSectionContext): void {
  const recall = context.input.autobiographicalRecall;

  if (recall === null || recall === undefined || recall.evidence.length === 0) {
    return;
  }

  // This section assembles twelve kinds. Printing the population beside a single counted kind
  // fixed the missing denominator and left the complementary misread standing: with only
  // self_decision named, every other kind is counted nowhere, so a page of five goal rows under
  // `{"rows_assembled":48,"self_decision":10}` invites reading ten against seven printed rows as
  // if the two figures described the same set. One key per kind present makes the object a
  // decomposition of its own population -- the kinds sum to rows_assembled by construction, so
  // the page's own kinds are locatable in it and a kind absent from the object contributed zero
  // rather than going unmeasured. Built from the rows rather than from a fixed key list, so a
  // kind added to the union appears here without this call site being revisited.
  const kindCounts: Record<string, number> = {};

  for (const item of recall.evidence) {
    kindCounts[item.kind] = (kindCounts[item.kind] ?? 0) + 1;
  }

  setSectionFraming(context.buckets, "autobiographical_recall", {
    text: AUTOBIOGRAPHICAL_RECALL_FRAMING,
    counts: {
      rows_assembled: recall.evidence.length,
      ...kindCounts,
    },
  });

  for (const item of recall.evidence) {
    const window =
      item.occurredAt < recall.window.startMs
        ? (recall.window.additionalWindow ?? recall.window)
        : recall.window;
    const stateMetadata = appendMemoryDisclosureStateMetadata({
      stateMetadata: {
        group_id: item.groupId,
        group_label: item.groupLabel,
        source_kind: item.kind,
        // Raw epoch ms, not ISO -- see the convention note in ../recent-lived-experience.ts.
        occurred_at: item.occurredAt,
        relative_age: item.relativeAge,
        window_start_ms: window.startMs,
        window_end_ms: window.endMs,
        window_label: window.label,
        window_source: window.source,
        source_stream_ids: [...item.sourceStreamEntryIds],
        source_episode_ids: [...item.sourceEpisodeIds],
        ...item.metadata,
        ...autobiographicalRecallCapStateMetadata(item.capMetadata),
      },
      disclosureLabel: item.disclosureLabel,
      currentAudienceEntityId: context.input.audienceEntityId,
    });

    addEntry(context.buckets, "autobiographical_recall", {
      id: `autobiographical_recall:${item.id}`,
      source_type: sourceTypeForAutobiographicalItem(item),
      session_scope: "global",
      actor: "memory",
      trust_rank: AUTOBIOGRAPHICAL_RECALL_TRUST_RANK,
      text: item.text,
      value: `${item.groupId}/${item.kind}`,
      state: appendMemoryDisclosureState({
        state: `score=${item.score.toFixed(2)} group=${item.groupId}`,
        disclosureLabel: item.disclosureLabel,
      }),
      state_metadata: stateMetadata,
      taint: item.metadata.taint === "quarantined" ? "quarantined" : "none",
      via_retrieval: true,
    });
  }
}
