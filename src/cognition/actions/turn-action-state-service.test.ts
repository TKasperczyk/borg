import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { FixedClock } from "../../util/clock.js";
import { createStreamEntryId } from "../../util/ids.js";
import { NOOP_TRACER } from "../tracing/tracer.js";
import { ActionStateExtractor } from "./action-state-extractor.js";
import { TurnActionStateService } from "./turn-action-state-service.js";

describe("TurnActionStateService", () => {
  it("skips action extraction when the current turn has a frame anomaly", async () => {
    const extractSpy = vi.spyOn(ActionStateExtractor.prototype, "extract");
    const llm = new FakeLLMClient();
    const service = new TurnActionStateService({
      model: "test-recall",
      actionRepository: { add: vi.fn() } as never,
      clock: new FixedClock(1_000),
      tracer: NOOP_TRACER,
    });

    try {
      const ids = await service.extract({
        llmClient: llm,
        turnId: "turn_anomaly",
        isUserTurn: true,
        userMessage: "You were playing Tom.",
        persistedUserEntryId: createStreamEntryId(),
        recentHistory: [],
        audienceEntityId: null,
        frameAnomaly: {
          status: "ok",
          kind: "frame_assignment_claim",
          confidence: 0.96,
          rationale: "The user-role message assigns the prior exchange to a roleplay frame.",
        },
      });

      expect(ids).toEqual([]);
      expect(extractSpy).not.toHaveBeenCalled();
      expect(llm.requests).toHaveLength(0);
    } finally {
      extractSpy.mockRestore();
    }
  });
});
