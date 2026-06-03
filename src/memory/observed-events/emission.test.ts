import { describe, expect, it } from "vitest";

import type { ActualFrameAnomalyClassification } from "../../cognition/frame-anomaly/index.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../../util/ids.js";
import {
  OBSERVED_EVENT_UNKNOWN_SPEAKER_SENTINEL,
  buildObservedEventEmission,
} from "./index.js";

function classification(
  overrides: Partial<ActualFrameAnomalyClassification> = {},
): ActualFrameAnomalyClassification {
  return {
    status: "ok",
    kind: "frame_assignment_claim",
    confidence: 0.94,
    rationale: "rozmówca forsował odrzuconą ramę społeczną z tokenem-Powierzchnia",
    ...overrides,
  };
}

describe("buildObservedEventEmission", () => {
  it("returns null for non-emitting dispositions", () => {
    const sessionId = createSessionId();
    const speakerEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const sourceUserEntryIds = [createStreamEntryId()];

    expect(
      buildObservedEventEmission({
        occurredAt: 1_000,
        sessionId,
        disposition: "none",
        actionableFrameAnomaly: classification(),
        speakerEntityId,
        audienceEntityId,
        sourceUserEntryIds,
      }),
    ).toBeNull();
    expect(
      buildObservedEventEmission({
        occurredAt: 1_000,
        sessionId,
        disposition: "trusted_operator_control",
        actionableFrameAnomaly: null,
        speakerEntityId,
        audienceEntityId,
        sourceUserEntryIds,
      }),
    ).toBeNull();
  });

  it("maps quarantine to a social observed-event record input with structural keys", () => {
    const sessionId = createSessionId();
    const speakerEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const anchorEntryId = createStreamEntryId();
    const laterEntryId = createStreamEntryId();
    const actionable = classification();

    const emission = buildObservedEventEmission({
      occurredAt: 1_000,
      sessionId,
      disposition: "quarantine",
      actionableFrameAnomaly: actionable,
      speakerEntityId,
      audienceEntityId,
      sourceUserEntryIds: [anchorEntryId, laterEntryId],
    });

    expect(emission).not.toBeNull();
    expect(emission).toMatchObject({
      occurredAt: 1_000,
      sessionId,
      stance: "rejected_frame",
      taint: "quarantined",
      beliefEffect: "unchanged",
      classificationKind: "frame_assignment_claim",
      disclosureClass: "social_observed",
      interactionText: actionable.rationale,
      recurrenceKey: `${sessionId}:${speakerEntityId}:frame_assignment_claim`,
      speakerEntityId,
      audienceEntityId,
      sourceEntityId: null,
      // full set retained for provenance...
      sourceStreamEntryIds: [anchorEntryId, laterEntryId],
    });
    // ...but the fire-once key anchors on the EARLIEST entry only.
    expect(emission?.fireDedupKey).toBe(`${sessionId}|frame_assignment_claim|${anchorEntryId}`);
    // structural: keys are ids + kind enum, never the rationale text.
    expect(emission?.recurrenceKey).toContain(sessionId);
    expect(emission?.recurrenceKey).toContain(speakerEntityId);
    expect(emission?.recurrenceKey).toContain("frame_assignment_claim");
    expect(emission?.fireDedupKey).toContain(sessionId);
    expect(emission?.fireDedupKey).toContain(anchorEntryId);
    expect(emission?.fireDedupKey).toContain("frame_assignment_claim");
    expect(emission?.recurrenceKey).not.toContain("Powierzchnia");
    expect(emission?.fireDedupKey).not.toContain("Powierzchnia");
  });

  it("keeps the fire-once key stable when a crash-replay expands the backlog prefix", () => {
    // Regression guard for the replay double-count class: the backlog prefix may grow [A] -> [A, B]
    // on a retry, but the earliest entry (the anchor) does not change, so the fire-once key is
    // identical and the replay is a no-op. A genuinely new push in a later turn has a different
    // earliest entry -> different key -> recurrence_count correctly bumps.
    const sessionId = createSessionId();
    const speakerEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const anchorEntryId = createStreamEntryId();
    const lateArrivalEntryId = createStreamEntryId();
    const differentPushEntryId = createStreamEntryId();
    const base = {
      occurredAt: 1_000,
      sessionId,
      disposition: "quarantine" as const,
      actionableFrameAnomaly: classification(),
      speakerEntityId,
      audienceEntityId,
    };

    const original = buildObservedEventEmission({ ...base, sourceUserEntryIds: [anchorEntryId] });
    const replayExpanded = buildObservedEventEmission({
      ...base,
      sourceUserEntryIds: [anchorEntryId, lateArrivalEntryId],
    });
    const laterPush = buildObservedEventEmission({
      ...base,
      sourceUserEntryIds: [differentPushEntryId, anchorEntryId],
    });

    // expansion on replay -> identical fire-once key (no double-count); the appended entry is absent
    expect(replayExpanded?.fireDedupKey).toBe(original?.fireDedupKey);
    expect(replayExpanded?.fireDedupKey).not.toContain(lateArrivalEntryId);
    // a later genuine push (different earliest entry) -> different key
    expect(laterPush?.fireDedupKey).not.toBe(original?.fireDedupKey);
  });

  it("uses the unknown-speaker sentinel only in the recurrence key", () => {
    const sessionId = createSessionId();
    const sourceUserEntryIds = [createStreamEntryId()];

    const emission = buildObservedEventEmission({
      occurredAt: 1_000,
      sessionId,
      disposition: "quarantine",
      actionableFrameAnomaly: classification({
        kind: "system_prompt_claim",
      }),
      speakerEntityId: null,
      audienceEntityId: null,
      sourceUserEntryIds,
    });

    expect(emission?.recurrenceKey).toBe(
      `${sessionId}:${OBSERVED_EVENT_UNKNOWN_SPEAKER_SENTINEL}:system_prompt_claim`,
    );
    expect(emission?.speakerEntityId).toBeNull();
  });
});
