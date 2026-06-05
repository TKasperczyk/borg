import { isQuarantinedUserEntryMarker } from "../../../stream/index.js";
import { stringifyPromptContent } from "../../../util/token-estimate.js";
import { commitmentReconciliationReviewDisclosureLabel } from "../../../memory/semantic/index.js";
import { correctionMemoryDisclosureLabel } from "../../disclosure-labels.js";
import {
  renderMemoryDisclosureLabelForModel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../../../retrieval/index.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "../entry-metadata.js";
import { QUARANTINE_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import {
  persistenceClassFromProvenance,
  reviewQueueScope,
  reviewQueueStreamIds,
  scopeFromStreamIds,
} from "../scope-resolver.js";

function commitmentReviewText(
  review: NonNullable<BuilderSectionContext["input"]["pendingCommitmentReviews"]>[number],
): string {
  const lines = [review.reason];

  for (const member of review.members) {
    const disclosure = renderMemoryDisclosureLabelForModel(
      member.disclosure_label ?? unknownMemoryDisclosureLabel(),
    );
    const directive = member.directive ?? member.directive_family;

    lines.push(`- commitment ${member.id}: ${directive} (${disclosure})`);
  }

  return lines.join("\n");
}

export function addContradictionsAndQuarantinesSection(context: BuilderSectionContext): void {
  if (context.input.frameAnomaly?.status === "ok") {
    addEntry(context.buckets, "contradictions_quarantines", {
      id: `frame_anomaly:${context.input.frameAnomaly.kind}`,
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: QUARANTINE_TRUST_RANK,
      text: context.input.frameAnomaly.rationale,
      value: context.input.frameAnomaly.kind,
      state: "quarantined",
      taint: "quarantined",
    });
  }

  for (const entry of context.streamEntries) {
    if (!isQuarantinedUserEntryMarker(entry)) {
      continue;
    }

    addEntry(context.buckets, "contradictions_quarantines", {
      id: `stream_quarantine:${entry.id}`,
      source_type: "system_metadata",
      session_scope: "current_session",
      actor: "system",
      trust_rank: QUARANTINE_TRUST_RANK,
      text: stringifyPromptContent(entry.content),
      stream_index: context.resolver.streamOrderById.get(entry.id),
      state: "quarantined",
      taint: "quarantined",
    });
  }

  for (const correction of context.input.pendingCorrections) {
    const disclosureLabel =
      (correction as { disclosureLabel?: MemoryDisclosureLabel }).disclosureLabel ??
      correctionMemoryDisclosureLabel(correction.refs);
    addEntry(
      context.buckets,
      "contradictions_quarantines",
      cappedTrustRank({
        id: `review_queue:${correction.id}`,
        source_type: "system_metadata",
        session_scope: reviewQueueScope(correction, context.resolver),
        actor: "system",
        trust_rank: QUARANTINE_TRUST_RANK,
        text: correction.reason,
        value: correction.kind,
        state: appendMemoryDisclosureState({
          state: correction.resolved_at === null ? "open" : "resolved",
          disclosureLabel,
        }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: undefined,
          disclosureLabel,
        }),
        taint: "contested",
        ...persistenceClassFromProvenance(
          { streamEntryIds: reviewQueueStreamIds(correction) },
          context.resolver,
        ),
      }),
    );
  }

  for (const review of context.input.pendingCommitmentReviews ?? []) {
    const disclosureLabel = commitmentReconciliationReviewDisclosureLabel(review.refs);

    addEntry(
      context.buckets,
      "contradictions_quarantines",
      cappedTrustRank({
        id: `review_queue:${review.review_id}`,
        source_type: "system_metadata",
        session_scope: scopeFromStreamIds(review.source_stream_entry_ids, context.resolver),
        actor: "system",
        trust_rank: QUARANTINE_TRUST_RANK,
        text: commitmentReviewText(review),
        value: `${review.refs.target_type}:${review.subkind}`,
        state: appendMemoryDisclosureState({
          state: "open",
          disclosureLabel,
        }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: {
            review_kind: review.refs.target_type,
            review_subkind: review.subkind,
            commitment_ids: review.commitment_ids,
          },
          disclosureLabel,
        }),
        taint: "contested",
        ...persistenceClassFromProvenance(
          { streamEntryIds: review.source_stream_entry_ids },
          context.resolver,
        ),
      }),
    );
  }

  const contradictionCount = context.input.retrievedSemantic?.contradiction_hits.length ?? 0;

  if (contradictionCount > 0) {
    addEntry(context.buckets, "contradictions_quarantines", {
      id: "semantic_contradictions:retrieved",
      source_type: "system_metadata",
      session_scope: "global",
      actor: "memory",
      trust_rank: QUARANTINE_TRUST_RANK,
      text: `Retrieved semantic contradiction hits: ${contradictionCount}`,
      state: "present",
      taint: "contested",
    });
  }
}
