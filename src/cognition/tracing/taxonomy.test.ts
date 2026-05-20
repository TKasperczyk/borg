import { describe, expect, it } from "vitest";

import { canonicalTraceEventName, phaseForTraceEventName } from "./taxonomy.js";

describe("trace taxonomy", () => {
  it("routes normalized trace events to their shared reporting phase", () => {
    expect(phaseForTraceEventName("extraction.actions.completed")).toBe("extraction");
    expect(phaseForTraceEventName("extraction.commitments.transitioned")).toBe("extraction");
    expect(phaseForTraceEventName("review_resolver.completed")).toBe("review");
    expect(phaseForTraceEventName("turn.rejected")).toBe("session");
    expect(phaseForTraceEventName("frame_anomaly.completed")).toBe("perception");
    expect(phaseForTraceEventName("frame_anomaly.degraded_fail_open")).toBe("perception");
    expect(phaseForTraceEventName("semantic_revision.completed")).toBe("retrieval");
    expect(phaseForTraceEventName("shared_state.compile.completed")).toBe("retrieval");
    expect(phaseForTraceEventName("shared_state.reconcile.completed")).toBe("retrieval");
    expect(phaseForTraceEventName("decision_artifact_compile.completed")).toBe("retrieval");
    expect(phaseForTraceEventName("decision_artifact_reconcile.completed")).toBe("retrieval");
    expect(canonicalTraceEventName("decision_artifact_compile.completed")).toBe(
      "shared_state.compile.completed",
    );
    expect(canonicalTraceEventName("decision_artifact_reconcile.completed")).toBe(
      "shared_state.reconcile.completed",
    );
    expect(canonicalTraceEventName("shared_state.compile.completed")).toBe(
      "shared_state.compile.completed",
    );
  });
});
