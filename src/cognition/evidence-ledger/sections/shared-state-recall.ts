import { sharedStateMemoryDisclosureLabel } from "../../../memory/common/disclosure-serializers.js";
import { formatRelativeAge } from "../../../util/relative-time.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "../entry-metadata.js";
import { SHARED_STATE_RECALL_TRUST_RANK, addEntry } from "../section-buckets.js";

export function addSharedStateRecallSection(context: BuilderSectionContext): void {
  for (const entry of context.input.sharedStateRecall ?? []) {
    const disclosureLabel = sharedStateMemoryDisclosureLabel(entry);
    const stateParts = [
      `kind=${entry.kind}`,
      `state_key=${entry.state_key}`,
      entry.last_updated_turn_global === null
        ? null
        : `last_updated_turn_global=${entry.last_updated_turn_global}`,
    ].filter((part): part is string => part !== null);

    addEntry(context.buckets, "shared_state_recall", {
      id: `shared_state_recall:${entry.id}`,
      source_type: "shared_state",
      session_scope: "global",
      actor: "memory",
      trust_rank: SHARED_STATE_RECALL_TRUST_RANK,
      text: entry.text,
      value: entry.kind,
      state: appendMemoryDisclosureState({
        state: stateParts.join(" "),
        disclosureLabel,
      }),
      state_metadata: appendMemoryDisclosureStateMetadata({
        stateMetadata: {
          shared_state_entry_id: entry.id,
          audience_entity_id: entry.audience_entity_id,
          owner_entity_id: entry.owner_entity_id,
          state_key: entry.state_key,
          kind: entry.kind,
          provenance_stream_entry_ids: [...entry.provenance_stream_entry_ids],
          last_updated_stream_entry_ids: [...entry.last_updated_stream_entry_ids],
          last_updated_at: new Date(entry.last_updated_at).toISOString(),
          ...(context.nowMs === undefined
            ? {}
            : { relative_age: formatRelativeAge(entry.last_updated_at, context.nowMs) }),
          last_updated_turn_global: entry.last_updated_turn_global,
          canonicalizes: entry.canonicalizes,
        },
        disclosureLabel,
        currentAudienceEntityId: context.input.audienceEntityId,
      }),
      citations: [...entry.provenance_stream_entry_ids],
      taint: "none",
      via_retrieval: true,
    });
  }
}
