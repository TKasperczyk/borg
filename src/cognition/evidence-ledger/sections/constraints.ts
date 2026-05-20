import type { BuilderSectionContext } from "../builder-context.js";
import {
  COMMITMENT_TRUST_RANK,
  DISCOURSE_TRUST_RANK,
  addEntry,
  cappedTrustRank,
} from "../section-buckets.js";
import {
  commitmentScope,
  persistenceClassFromProvenance,
} from "../scope-resolver.js";

export function addCommitmentsAndConstraintsSection(context: BuilderSectionContext): void {
  for (const commitment of context.input.applicableCommitments) {
    addEntry(
      context.buckets,
      "commitments_and_constraints",
      cappedTrustRank({
        id: `commitment:${commitment.id}`,
        source_type: "commitment",
        session_scope: commitmentScope(commitment, context.resolver),
        actor: "memory",
        trust_rank: COMMITMENT_TRUST_RANK,
        text: commitment.directive,
        value: commitment.directive_family,
        state:
          commitment.revoked_at !== null
            ? "revoked"
            : commitment.expired_at !== null
              ? "expired"
              : "active",
        state_metadata: {
          commitment_kind: commitment.kind,
          commitment_type: commitment.type,
        },
        taint: "none",
        ...persistenceClassFromProvenance(
          { streamEntryIds: commitment.source_stream_entry_ids ?? [] },
          context.resolver,
        ),
      }),
    );
  }

  const stopState = context.input.workingMemory.discourse_state?.stop_until_substantive_content;

  if (stopState !== undefined && stopState !== null) {
    addEntry(context.buckets, "commitments_and_constraints", {
      id: "discourse_constraint:stop_until_substantive_content",
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: DISCOURSE_TRUST_RANK,
      text: stopState.reason,
      value: stopState.provenance,
      state: "active",
      taint: "none",
    });
  }
}
