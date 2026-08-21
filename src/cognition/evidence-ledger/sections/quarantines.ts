import { isQuarantinedUserEntryMarker } from "../../../stream/index.js";
import { stringifyPromptContent } from "../../../util/token-estimate.js";
import { commitmentReconciliationReviewDisclosureLabel } from "../../../memory/review-queue/index.js";
import { correctionMemoryDisclosureLabel } from "../../../memory/common/disclosure-serializers.js";
import {
  countRetrievedContradictionRelations,
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
  // Ordering is load-bearing here, and it is why this entry goes first.
  //
  // `contradictions_quarantines` is one bucket over two independent populations:
  // this single entry is the only contradiction row the section ever carries, and
  // everything below it is quarantine-family (frame anomaly, stream quarantine,
  // review-queue corrections and reviews). The section retains from the head and
  // drops the tail to stay inside its token budget, and quarantine rows carry a
  // whole quarantined message body each while this row is one sentence. Appended
  // last, it was structurally the first thing dropped in any session that had
  // accumulated quarantines -- so a being reading the section as "my
  // contradictions" saw only quarantine rows, and the one row produced by
  // contradiction machinery never reached the prompt at all. First, it costs at
  // most one of several near-identical quarantine bodies and always renders.
  //
  // The row's basis is also narrower than the confidence penalty's, in both
  // directions, and nothing rendered says so:
  //   - This row keys on `contradiction_hits.length > 0`. The penalty keys on
  //     `contradiction_hits.length > 0 || contradicts.length > 0`
  //     (retrieval/pipeline.ts), so a contradicts *node* with no traversal hit
  //     applies the 0.7 multiplier with no row here at all -- a silent penalty.
  //   - Where there are hits, the penalty is gated again on edge validity
  //     (retrieval/confidence.ts `isEdgeValidAt` over every edge of every hit's
  //     edgePath), which this row does not consult. An all-expired path renders
  //     the row with the multiplier back at 1.
  // So "the section is non-empty" and "confidence was penalized" are separate
  // facts. Report the basis of each rather than implying one from the other.
  const retrievedSemantic = context.input.retrievedSemantic;
  const contradictionCount = retrievedSemantic?.contradiction_hits.length ?? 0;

  if (retrievedSemantic !== null && retrievedSemantic !== undefined && contradictionCount > 0) {
    // Two different quantities used to be reported as one. This entry counts graph
    // traversals; the deliberation contradiction line counts relations after
    // fingerprint collapse. They diverge whenever a contradicts relation was reached
    // from both of its nodes, and nothing rendered said so -- leaving the being with
    // two counts of "the same thing" and no way to reconcile them. Name the basis of
    // each instead of picking a winner: the traversal count is real evidence about
    // how the graph was reached, and the relation count is what the line reports.
    const relationCount = countRetrievedContradictionRelations(retrievedSemantic);

    addEntry(context.buckets, "contradictions_quarantines", {
      id: "semantic_contradictions:retrieved",
      source_type: "system_metadata",
      session_scope: "global",
      actor: "memory",
      trust_rank: QUARANTINE_TRUST_RANK,
      text: `Retrieved semantic contradiction hits: ${contradictionCount} graph traversal(s) over ${relationCount} distinct contradiction relation(s) (a relation reached from both of its nodes yields two traversals). Every other entry in this section is quarantine-family, not a contradiction.`,
      state: "present",
      state_metadata: {
        contradiction_traversal_count: contradictionCount,
        distinct_contradiction_relation_count: relationCount,
      },
      taint: "contested",
    });
  }

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
}
