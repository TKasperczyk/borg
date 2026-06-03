import type { ActualFrameAnomalyClassification } from "../../cognition/frame-anomaly/index.js";
import type { FrameAnomalyDisposition } from "../../cognition/lifecycle/turn-phase/perception-phase.js";
import type { EntityId, SessionId, StreamEntryId } from "../../util/ids.js";
import { OBSERVED_EVENT_UNKNOWN_SPEAKER_SENTINEL } from "./constants.js";
import { deriveObservedEventDimensions } from "./derive.js";
import type { ObservedEventRecordInput } from "./repository.js";

export type BuildObservedEventEmissionInput = {
  occurredAt: number;
  sessionId: SessionId;
  disposition: FrameAnomalyDisposition;
  actionableFrameAnomaly: ActualFrameAnomalyClassification | null;
  speakerEntityId: EntityId | null;
  audienceEntityId: EntityId | null;
  sourceUserEntryIds: readonly StreamEntryId[];
};

export function buildObservedEventEmission(
  input: BuildObservedEventEmissionInput,
): ObservedEventRecordInput | null {
  if (input.disposition !== "quarantine" || input.actionableFrameAnomaly === null) {
    return null;
  }

  const actionable = input.actionableFrameAnomaly;
  const dimensions = deriveObservedEventDimensions({
    disposition: input.disposition,
    classificationKind: actionable.kind,
  });
  const speakerKey = input.speakerEntityId ?? OBSERVED_EVENT_UNKNOWN_SPEAKER_SENTINEL;
  // Fire-once key anchors on the EARLIEST pending source entry (sourceUserEntryIds[0]), NOT the
  // full set. The backlog prefix is built earliest-first from the (stuck) watermark and always
  // includes that first entry; a crash-replay may EXPAND the prefix ([A] -> [A, B]) but never
  // changes its first element, so the anchor is invariant across replays and a replay is a no-op.
  // A genuine re-push lands in a later turn (watermark advanced) whose earliest entry differs ->
  // a new key -> recurrence_count correctly bumps. (Keying on the full set would change on
  // expansion and double-count; min-by-id would be unstable because stream ids are random.)
  const anchorSourceEntryId = input.sourceUserEntryIds[0];

  return {
    occurredAt: input.occurredAt,
    sessionId: input.sessionId,
    stance: dimensions.stance,
    taint: dimensions.taint,
    beliefEffect: dimensions.beliefEffect,
    classificationKind: dimensions.classificationKind,
    disclosureClass: "social_observed",
    interactionText: actionable.rationale,
    recurrenceKey: `${input.sessionId}:${speakerKey}:${actionable.kind}`,
    fireDedupKey:
      anchorSourceEntryId === undefined
        ? undefined
        : `${input.sessionId}|${actionable.kind}|${anchorSourceEntryId}`,
    speakerEntityId: input.speakerEntityId,
    audienceEntityId: input.audienceEntityId,
    sourceEntityId: null,
    sourceStreamEntryIds: input.sourceUserEntryIds,
  };
}
