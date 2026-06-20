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
        occurred_at: input.row.occurredAt,
        relative_age: input.row.relativeAge,
        ...input.row.metadata,
      },
      disclosureLabel,
      currentAudienceEntityId: input.audienceEntityId,
    }),
    taint: "none",
  };
}
