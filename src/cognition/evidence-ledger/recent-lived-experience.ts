import { recentLivedExperienceDisclosureLabel } from "../../memory/activity/index.js";
import type { RecentLivedExperienceRow } from "../../memory/activity/index.js";
import type { EntityId } from "../../util/ids.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "./entry-metadata.js";
import { CROSS_SESSION_ACTIVITY_TRUST_RANK } from "./section-buckets.js";
import type { EvidenceLedgerEntry } from "./types.js";

export function buildRecentLivedExperienceLedgerEntry(input: {
  row: RecentLivedExperienceRow;
  index: number;
  audienceEntityId: EntityId | null;
}): EvidenceLedgerEntry {
  const disclosureLabel = recentLivedExperienceDisclosureLabel(input.row);

  return {
    id: `recent_lived_experience:${input.index + 1}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: CROSS_SESSION_ACTIVITY_TRUST_RANK,
    text: input.row.text,
    value: input.row.kind,
    state: appendMemoryDisclosureState({ state: "active", disclosureLabel }),
    state_metadata: appendMemoryDisclosureStateMetadata({
      stateMetadata: {
        lived_experience_kind: input.row.kind,
        // Raw epoch ms, not an ISO string. One key name, two conventions across the ledger: this
        // section, the observed-event rows in audience-standing.ts and sections/autobiographical-
        // recall.ts all emit the number, while sections/episodes.ts and sections/semantic-graph.ts
        // emit `new Date(...).toISOString()` under the same `occurred_at`. On the epoch side the
        // only human-readable time beside it is the coarse `relative_age` ("yesterday", "22h ago"),
        // so any exact clock reading of these rows is arithmetic the model does by hand.
        // Measured 2026-08-18 against a live finalizer prompt (turn 5f9c3495, 2026-08-17): five
        // inbound arrival stamps appeared exactly once each as bare epochs, no clock form of them
        // -- correct or otherwise -- appeared anywhere in the prompt, and 22 other epochs in that
        // same prompt did appear next to their correct ISO forms. The model's own conversion of the
        // five was uniformly off by a constant 40:00 with seconds preserved: one conversion applied
        // to a list, not five misreadings, and nothing on the surface marked the result as derived.
        // Whether the epoch-side sections should render ISO as the others do changes the live
        // surface, so it is a design call rather than a cleanup.
        occurred_at: input.row.occurredAt,
        relative_age: input.row.relativeAge,
        ...input.row.metadata,
      },
      disclosureLabel,
      currentAudienceEntityId: input.audienceEntityId,
    }),
    ...(input.row.plannerDecision === undefined
      ? {}
      : {
          planner_metadata: {
            decision_outcome_ref: input.row.plannerDecision.outcomeReference,
            decision_summary: input.row.plannerDecision.summary,
            decision_rationale: input.row.plannerDecision.rationale,
          },
        }),
    taint: "none",
  };
}
