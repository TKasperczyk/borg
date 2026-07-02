import type { FrameAnomalyKind } from "../../cognition/frame-anomaly/index.js";
import type { FrameAnomalyDisposition } from "../../cognition/lifecycle/turn-phase/perception-phase.js";
import type {
  ObservedEventBeliefEffect,
  ObservedEventStance,
  ObservedEventTaint,
} from "./types.js";

export type ObservedEventDerivationInput = {
  disposition: FrameAnomalyDisposition;
  classificationKind: FrameAnomalyKind;
};

export type ObservedEventDerivedDimensions = {
  stance: ObservedEventStance;
  taint: ObservedEventTaint;
  beliefEffect: ObservedEventBeliefEffect;
  classificationKind: FrameAnomalyKind;
};

const OBSERVED_EVENT_DIMENSIONS_BY_DISPOSITION = {
  quarantine: {
    stance: "rejected_frame",
    taint: "quarantined",
    beliefEffect: "unchanged",
  },
  trusted_operator_control: {
    stance: "accepted_frame",
    taint: "none",
    beliefEffect: "updated",
  },
  trusted_peer_channel: {
    stance: "noted_frame",
    taint: "none",
    beliefEffect: "unchanged",
  },
  none: {
    stance: "noted_frame",
    taint: "none",
    beliefEffect: "unchanged",
  },
} as const satisfies Record<
  FrameAnomalyDisposition,
  {
    stance: ObservedEventStance;
    taint: ObservedEventTaint;
    beliefEffect: ObservedEventBeliefEffect;
  }
>;

export function deriveObservedEventDimensions(
  input: ObservedEventDerivationInput,
): ObservedEventDerivedDimensions {
  return {
    ...OBSERVED_EVENT_DIMENSIONS_BY_DISPOSITION[input.disposition],
    classificationKind: input.classificationKind,
  };
}
