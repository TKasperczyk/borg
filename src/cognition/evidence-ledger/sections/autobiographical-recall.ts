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
import type { EvidenceLedgerSourceType } from "../types.js";

const AUTOBIOGRAPHICAL_RECALL_FRAMING =
  "Autobiographical recall entries are past evidence for me to re-examine during this turn. I treat recalled self_decision rows as historical decisions and rationales, not standing verdicts; I revise them when current evidence warrants.";

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

  setSectionFraming(context.buckets, "autobiographical_recall", {
    text: AUTOBIOGRAPHICAL_RECALL_FRAMING,
    counts: {
      self_decision: recall.evidence.filter((item) => item.kind === "self_decision").length,
    },
  });

  for (const item of recall.evidence) {
    const stateMetadata = appendMemoryDisclosureStateMetadata({
      stateMetadata: {
        group_id: item.groupId,
        group_label: item.groupLabel,
        source_kind: item.kind,
        // Raw epoch ms, not ISO -- see the convention note in ../recent-lived-experience.ts.
        occurred_at: item.occurredAt,
        relative_age: item.relativeAge,
        window_start_ms: recall.window.startMs,
        window_end_ms: recall.window.endMs,
        window_label: recall.window.label,
        window_source: recall.window.source,
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
